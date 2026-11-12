"""CPU-only checks of the Transformers 5.22.2 Triton MXFP4 layout contract."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent_support import gpt_oss_model
from agent_support.gpt_oss_model import check_mxfp4_dependencies, verify_packed_mxfp4


@dataclass(frozen=True)
class Device:
    type: str
    index: int | None = None

    def __str__(self):
        return f"{self.type}:{self.index}" if self.index is not None else self.type


class StridedLayout:
    pass


class WrongLayout:
    pass


class Backing:
    def __init__(self, count, dtype="uint8", device=Device("cuda", 0), is_meta=False):
        self.count, self.dtype, self.device, self.is_meta = count, dtype, device, is_meta

    def numel(self):
        return self.count

    def element_size(self):
        return 1 if self.dtype == "uint8" else 2


class Tensor:
    def __init__(self, shape, dtype, count, *, layout=None, backing_dtype="uint8",
                 device=Device("cuda", 0), is_meta=False):
        self.shape, self.dtype = shape, dtype
        self.storage = SimpleNamespace(data=Backing(count, backing_dtype, device, is_meta),
                                       layout=layout if layout is not None else StridedLayout())


class Expert:
    def __init__(self):
        self.num_experts = 2
        self.hidden_size = 32
        self.intermediate_size = 32
        self._is_hf_initialized = True
        self._parameters = {}
        self.gate_up_proj = Tensor((2, 32, 64), "FP4", 2048)
        self.down_proj = Tensor((2, 32, 32), "FP4", 1024)
        self.gate_up_proj_precision_config = SimpleNamespace(
            weight_scale=Tensor((2, 1, 64), "uint8", 128))
        self.down_proj_precision_config = SimpleNamespace(
            weight_scale=Tensor((2, 1, 32), "uint8", 64))


class Model:
    def __init__(self, count=2):
        self.config = SimpleNamespace(num_hidden_layers=count,
                                      quantization_config={"quant_method": "mxfp4"})
        self.modules = {f"model.layers.{i}.mlp.experts": Expert() for i in range(count)}
        self.parameters = {"model.embed_tokens.weight": Backing(32, dtype="bf16")}

    def named_modules(self):
        return iter(self.modules.items())

    def named_parameters(self):
        return iter(self.parameters.items())


def setup():
    model = Model()
    torch_api = SimpleNamespace(Tensor=Backing, uint8="uint8", device=Device)
    hub = SimpleNamespace(tensor=SimpleNamespace(Tensor=Tensor, FP4="FP4"),
                          tensor_details=SimpleNamespace(layout=SimpleNamespace(
                              StridedLayout=StridedLayout,
                              make_default_matmul_mxfp4_w_layout=lambda **_: (StridedLayout, {}))))
    return model, torch_api, hub


def check(model, torch_api, hub):
    return verify_packed_mxfp4(model, torch_api, hub, Expert)


def test_valid_custom_packed_tensors_and_metadata():
    model, torch_api, hub = setup()
    result = check(model, torch_api, hub)
    assert result["quantization_method"] == "mxfp4"
    assert result["verified_expert_modules"] == 2
    assert result["verified_packed_projections"] == 4
    assert result["weight_layout_type"] == "StridedLayout"
    assert result["weight_storage_dtype"] == "uint8"
    assert result["backing_storage_device"] == "cuda:0"
    assert result["packed_weight_storage_bytes"] == 6144
    assert result["scale_storage_bytes"] == 384


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"], "gate_up_proj", None),
     "gate_up_proj: missing Triton packed projection"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].down_proj_precision_config,
                       "weight_scale", None), "down_proj: missing MXFP4 scale tensor"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"], "gate_up_proj",
                       Backing(2048, dtype="bf16")), "gate_up_proj: missing Triton packed projection"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].gate_up_proj.storage.data,
                       "dtype", "bf16"), "packed backing storage must be torch.uint8"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].gate_up_proj.storage,
                       "layout", WrongLayout()), "wrong MXFP4 weight storage layout"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].gate_up_proj_precision_config.weight_scale.storage.data,
                       "dtype", "bf16"), "MXFP4 scale storage must be torch.uint8"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].gate_up_proj_precision_config.weight_scale.storage,
                       "layout", WrongLayout()), "wrong MXFP4 scale storage layout"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].down_proj.storage.data,
                       "device", Device("cpu")), "down_proj: packed backing storage is uninitialized or not on cuda:0"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].down_proj_precision_config.weight_scale.storage.data,
                       "device", Device("cpu")), "down_proj: scale storage is uninitialized, incomplete, or not on cuda:0"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].gate_up_proj.storage.data,
                       "is_meta", True), "gate_up_proj: packed backing storage is uninitialized"),
    (lambda m: setattr(m.modules["model.layers.0.mlp.experts"].down_proj_precision_config.weight_scale.storage.data,
                       "count", 0), "down_proj: scale storage is uninitialized"),
    (lambda m: m.modules.pop("model.layers.1.mlp.experts"), "expert-module mismatch"),
    (lambda m: m.config.quantization_config.update(dequantize=True), "MXFP4 disabled"),
    (lambda m: setattr(m.parameters["model.embed_tokens.weight"], "device", Device("cpu")),
     "model.embed_tokens.weight: ordinary model parameter"),
])
def test_rejects_invalid_or_partial_quantization(mutation, diagnostic):
    model, torch_api, hub = setup()
    mutation(model)
    with pytest.raises(RuntimeError, match=diagnostic):
        check(model, torch_api, hub)


def test_rejects_registered_parameter_even_when_uint8():
    model, torch_api, hub = setup()
    model.modules["model.layers.0.mlp.experts"]._parameters["gate_up_proj"] = Backing(2048)
    with pytest.raises(RuntimeError, match="remains an nn.Parameter"):
        check(model, torch_api, hub)


def test_rejects_bad_shape_and_storage_count():
    model, torch_api, hub = setup()
    model.modules["model.layers.0.mlp.experts"].gate_up_proj.shape = (2, 32, 32)
    with pytest.raises(RuntimeError, match="logical tensor shape"):
        check(model, torch_api, hub)
    model, torch_api, hub = setup()
    model.modules["model.layers.0.mlp.experts"].gate_up_proj.storage.data.count = 3000
    with pytest.raises(RuntimeError, match="wrong byte count"):
        check(model, torch_api, hub)


def test_early_kernels_and_backend_compatibility():
    versions = {"transformers": "5.22.2", "kernels": "0.17.2",
                "triton": "3.4.0", "accelerate": "1.15.0"}
    assert check_mxfp4_dependencies("5.22.2", versions.__getitem__)["kernels"] == "0.17.2"
    for bad in ("0.17.1", "0.18.0"):
        versions["kernels"] = bad
        with pytest.raises(RuntimeError, match="kernels==0.17.2"):
            check_mxfp4_dependencies("5.22.2", versions.__getitem__)
    versions["kernels"] = "0.17.2"
    versions["triton"] = "3.3.0"
    with pytest.raises(RuntimeError, match="Triton >=3.4.0"):
        check_mxfp4_dependencies("5.22.2", versions.__getitem__)
    versions["triton"] = "3.4.0"
    with pytest.raises(RuntimeError, match="Transformers 5.22.2 differs"):
        check_mxfp4_dependencies("5.21.0", versions.__getitem__)


def test_constructor_rejects_bad_stack_before_importing_model(monkeypatch):
    def reject(_):
        raise RuntimeError("kernels==0.17.2 required before loading")
    monkeypatch.setattr(gpt_oss_model, "check_mxfp4_dependencies", reject)
    with pytest.raises(RuntimeError, match="before loading"):
        gpt_oss_model.GptOss20BModel({"backend_version": "5.22.2"})
