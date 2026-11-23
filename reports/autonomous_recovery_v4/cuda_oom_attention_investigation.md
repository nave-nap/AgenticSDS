# GPT-OSS CUDA OOM: attention and archived telemetry audit

## Scope and evidence

This is an experimental follow-up based on cache-release commit `890eb0a`.
The eight archived diagnostic runs under `reports/v4_cuda_telemetry/` were read
without changing them. They contain 254 GPT-OSS generations (247 successful,
seven CUDA OOM) and eight Qwen telemetry files. Three seed-89 controls and two
seed-89 cache-release repetitions fail at GPT-OSS request 29. One seed-89
cache-release run ends after request 27 on a different trajectory; it does not
demonstrate that request 29 was made safe. Seed 104 fails at request 41 in both
arms. These are repeated trajectories, not 254 independent task samples.

The analysis utility is `python -m agent_support.gpu_memory_analysis PATH ...`.
It joins start/end/cache-release records by file and request ID, reports
per-request input length, output ceiling, peak allocated, start/end reserved,
global free, OOM status, and Qwen placement. It marks OOM peaks as *lower
bounds*: the allocation that failed is absent from PyTorch's observed peak.
Its descriptive slope and context bins use successful generations only.

## Verified implementation and memory mechanism

`GptOss20BModel` loads pinned `openai/gpt-oss-20b` revision
`6cee5e81ee83917806bbde320786a8fb61efebee` in a separate worker using
Transformers 5.22.2, `device_map={"": 0}`, `dtype="auto"`, and the model's
MXFP4 checkpoint. The loader verifies all 24 expert layers have packed MXFP4
Triton weights on CUDA 0. It uses `torch.inference_mode()` and `model.eval()`.
The current call to `generate()` samples with temperature 1, top-p 1, top-k 0,
at most 1,024 new tokens, and the Harmony final return token as EOS. It asks
for plain sequences, not attention/hidden-state matrices or scores. Inputs
and the output tensor are local to `generate()`; no cross-request cache or
output reference is stored by the worker. The final diagnostic snapshot occurs
before the method frame unwinds, so end allocated may briefly include its local
input/output tensors. The next request's starting allocation is the useful
retention check.

The pinned [model configuration](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json)
has 24 alternating sliding/full attention layers, 64 query heads, eight KV
heads, 64-dimensional heads, sliding window 128, and `use_cache=true`.
Transformers 5.22.2 creates a per-generation dynamic KV cache by default and
automatically uses `logits_to_keep=1` during generation, so a full
sequence-by-vocabulary logits tensor is not the likely explanation. The
KV cache is also unlikely to account for the spike: with the pinned 12 full
layers, eight KV heads, 64 head dimensions, and two bytes per key/value
element, 7,555 tokens require about 0.17 GiB for full-layer KV; the 12
128-token sliding layers add only a few MiB. Even a larger actual cache cannot
readily explain the measured 19–21 GiB rise without another allocation path.
[pinned model implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py)
declares `_supports_sdpa=False` and falls back to eager attention. Its eager
path repeats KV heads, materializes a `[batch, 64, query_length, key_length]`
attention tensor, adds a mask, concatenates sinks, normalizes and softmaxes,
and then multiplies by values. Even the sliding-layer eager path uses a
dense attention score tensor with a sliding mask. At 7,555 tokens, *one* BF16
64-head square matrix is about 6.80 GiB; multiple live intermediates can
plausibly add 17–21 GiB to the 12.84 GiB model baseline. This is a source-code
mechanism, not a GPU kernel-level attribution.

Qwen runs in the parent process. Its NF4 model uses `device_map="auto"`; all
eight Qwen `model_loaded` records show `hf_device_map={"": "0"}` on an NVIDIA
L40S. Qwen reserves about 6.71 GiB. Thus both processes use CUDA 0 and
compete for the same device's global free memory, although their PyTorch
allocators are separate. The runner generates a Qwen proposal and then invokes
GPT-OSS sequentially; these are not simultaneous active generations. Qwen's
resident weights and reserved pool still reduce GPT-OSS headroom.

## Archived memory-versus-context results

| GPT-OSS input tokens | Successful requests | Median per-request peak above start allocated |
| ---: | ---: | ---: |
| 2,000–2,999 | 56 | 2.07 GiB |
| 3,000–3,999 | 53 | 3.88 GiB |
| 4,000–4,999 | 97 | 8.09 GiB |
| 5,000–5,999 | 36 | 10.97 GiB |
| 6,000–6,999 | 5 | 17.43 GiB |

The largest successful repeated request is 6,828 tokens, with a 30.27 GiB
peak allocated. A descriptive fit of peak excess against input tokens squared
across successful records gives roughly 0.372 GiB per million tokens squared.
Repeated identical trajectories and varying generation length limit any
predictive interpretation. No successful request in these files exceeds
6,828 input tokens, so the 7,269/7,555 OOM points must not be treated as
completed memory-demand measurements.

| Seed / arm | Failing request | Input | Start allocated | Start reserved | Start global free | Observed peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 89 / control (three runs) | 29 | 7,555 | 12.84 GiB | 35.67 GiB | 1.01 GiB | 33.79 GiB lower bound |
| 89 / cache release (two failing runs) | 29 | 7,555 | 12.84 GiB | 12.87 GiB | 23.81 GiB | 33.74 GiB lower bound |
| 104 / control | 41 | 7,269 | 12.84 GiB | 29.65 GiB | 7.02 GiB | 32.27 GiB lower bound |
| 104 / cache release | 41 | 7,269 | 12.84 GiB | 12.87 GiB | 23.80 GiB | 32.19 GiB lower bound |

Cache release demonstrably lowers the between-request *reserved* pool, but
the same large requests still OOM. The repeated 12.84 GiB starting allocation
argues against growing live cross-request tensors as the primary cause. The
sharp within-request rise and the pinned eager-attention implementation make
dense prefill intermediates the leading explanation. The exact failing CUDA
allocation, kernel, and phase remain unobserved.

## Ranked interventions

1. **Opt-in GPT-OSS FlexAttention** (implemented below). The pinned model
   explicitly supports it, and the pinned Transformers FlexAttention adapter
   [handles GPT-OSS attention sinks](https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/integrations/flex_attention.py).
   It avoids the eager square attention tensors while keeping the same model,
   prompt, output ceiling, sampling arguments, and KV-cache strategy. Kernel
   arithmetic can change sampled tokens, so this is a separate treatment, not
   a numerically identical replay of the frozen baseline. Locust runtime and
   L40S compilation still require a GPU smoke before a batch.
2. **Offloaded dynamic KV cache.** Transformers supports it and it preserves
   attention semantics, but the estimated KV cache at this length is much
   smaller than the observed 19–21 GiB temporary rise; offload transfers also
   affect timing. Consider only if FlexAttention leaves a cache-driven OOM.
3. **Model placement changes.** Moving Qwen or model layers off GPU 0 would
   create more headroom but changes placement, latency, and the matched
   execution environment. It is a larger intervention.
4. **Prompt/output truncation, different quantization, or allocator tuning.**
   These change the policy's information, decoding opportunity, model
   representation, or allocator behavior without directly addressing the
   verified square-attention path. They are not part of this candidate.

Do **not** set `attn_implementation="sdpa"`: this pinned GPT-OSS class
explicitly rejects it because its sink-aware attention is not standard SDPA.
The model lists some FlashAttention implementations, but their kernel and
hardware compatibility on L40S and this MXFP4 stack have not been verified;
none is enabled here.

## Opt-in candidate and matched GPU validation

Set `AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=1` only in a *new* diagnostic run.
Default/unset and `0` pass the original loader arguments unchanged. The switch
checks PyTorch FlexAttention availability and CUDA capability before loading,
requests `attn_implementation="flex_attention"`, and verifies the selected
backend afterward. It does not change model weights, expert quantization,
generation arguments, prompts, budgets, Qwen, support choice, or OOM handling.
Keep `AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1` in both comparison arms and
hold `AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE` equal (prefer `0`) so attention
is the only intervention. Use new output roots, the same pinned revisions and
seed/task order, and no old run directories. First run a one-task Locust smoke
that reaches at least 7,555 input tokens, checking actual backend selection,
packed MXFP4 verification, per-request peaks, OOM status, latency, and valid
Harmony output. Then pair fresh seed-89/104 runs under the same task and
resource conditions. Because sampling can diverge, compare memory at matched
input lengths or replay identical saved controller messages in a diagnostic
inference harness before attributing an episode-level success difference to
the attention backend. No GPU run has been launched here.

Read-only inspection command (the full JSON can be redirected outside the
archive):

```bash
python -m agent_support.gpu_memory_analysis ~/AgenticSDS/reports/v4_cuda_telemetry > /tmp/v4_cuda_analysis.json
python - <<'PY'
import json
report = json.load(open('/tmp/v4_cuda_analysis.json'))
print(json.dumps(report['summary'], indent=2))
PY
```

## Historical runtime resolution and compatibility gates

The evaluator's `requirements.txt` pins Transformers 4.57.6. The GPT-OSS
worker is launched through `AGENT_SUPPORT_GPTOSS_PYTHON` in a separate
subprocess and `check_mxfp4_dependencies()` requires Transformers 5.22.2,
`kernels==0.17.2`, and a compatible Triton. The user inspected all eight
historical Locust `run.json` files and confirmed that the *worker* actually
reported Transformers 5.22.2, PyTorch 2.8.0+cu128, CUDA 12.8, L40S on CUDA 0,
the pinned GPT-OSS revision, packed MXFP4, and the frozen decoding settings.
The present worker environment has the same reported versions and can import
FlexAttention. Those historical `run.json` files were not transferred to this
Mac; the direct historical metadata confirmation is user-supplied, distinct
from the local source-code check.

The pinned GPT-OSS class advertises `_supports_flex_attn=True` and
`_supports_sdpa=False`. Its attention forward passes both a sliding-window
setting and the sink weights to the selected backend. The pinned
[FlexAttention adapter](https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/integrations/flex_attention.py)
renormalizes attention output for sinks; the pinned
[mask interface](https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/masking_utils.py)
creates compressed block masks for full and sliding attention. The
[generation implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/generation/utils.py)
continues to prepare the same dynamic cache unless `cache_implementation`
is explicitly changed; this candidate does not change it. MXFP4 expert
weights are verified after model load regardless of attention backend. These
checks establish a supported code path, not successful compilation or
numerical equivalence on the actual L40S; the GPU smoke must establish those.

## Inference-only validation harness

`agent_support.gpt_oss_attention_probe` is separate from the frozen runner.
It can extract a saved `controller_messages` event from an episode trace or
create a deterministic synthetic fixture with the *pinned worker tokenizer*
without loading model weights. A run loads the GPT-OSS worker, then Qwen in
the evaluator process, verifies Qwen's whole `hf_device_map` is CUDA 0, and
performs the same fixed Qwen warmup in both conditions. It checks the worker's
actual attention implementation and SHA-256 fingerprint of the full GPT-OSS
input token sequence before making one GPT-OSS generation. It rejects runtime,
model revision, MXFP4, GPU, or decoding settings that differ from the
historical run metadata, and refuses cache release. Each fresh output
directory contains `probe_result.json`, the GPT-OSS CUDA JSONL, Qwen CUDA
JSONL, and the worker log. The result records both Python runtimes, active
backend, exact input fingerprint, generation outcome, model inference and IPC
latency, and start/end/peak GPU memory. An OOM is recorded as an outcome;
the harness does not retry it.

Only CUDA telemetry JSONL was transferred to this Mac; those files contain
token counts but **not** the controller messages. Exact request replay is
possible if the Locust episode JSON retains `support_trace.events[*].controller_messages`
and `controller_cost.input_tokens`, as the runner writes. Use `list-episode`
to locate the 7,555- or 7,269-token event and `export-episode` to copy its
messages into a new fixture. If those episodes are unavailable, the synthetic
fixture provides an equivalent *length stress test*, not an exact historical
replay. A fresh episode trace can be captured in a new diagnostic run without
editing or overwriting the frozen results. The OOM request's exact RNG state
also depends on prior generations; reseeding once permits a matched
eager-versus-Flex input comparison but is not bitwise historical replay.

After the branch is committed and deployed into a new Locust worktree, use
the two existing pinned interpreters and a **new** output root:

```bash
cd "$DEPLOY_REPO"
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME='<same pinned GPT-OSS cache root as the archived runs>'
export HF_HOME='<same pinned Qwen cache root as the archived runs>'
export PROBE_ROOT=/shared_data0/evanpan/agent_support/gptoss_attention_probe_v1
test ! -e "$PROBE_ROOT"
mkdir -p "$PROBE_ROOT/fixtures" "$PROBE_ROOT/logs"
"$VENV_PATH/bin/python" -c 'import transformers, torch; print(transformers.__version__, torch.__version__)'
"$AGENT_SUPPORT_GPTOSS_PYTHON" -c 'import transformers, torch; from torch.nn.attention.flex_attention import flex_attention; print(transformers.__version__, torch.__version__, torch.version.cuda)'
HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME" "$AGENT_SUPPORT_GPTOSS_PYTHON" \
  -m agent_support.gpt_oss_attention_probe synthetic \
  --target-tokens 512 --output "$PROBE_ROOT/fixtures/short.json"
HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME" "$AGENT_SUPPORT_GPTOSS_PYTHON" \
  -m agent_support.gpt_oss_attention_probe synthetic \
  --target-tokens 7555 --output "$PROBE_ROOT/fixtures/long.json"
```

If a saved episode is present, create a *separate* exact fixture instead:

```bash
"$VENV_PATH/bin/python" -m agent_support.gpt_oss_attention_probe list-episode \
  --episode '<saved seed-89 episode JSON>'
"$VENV_PATH/bin/python" -m agent_support.gpt_oss_attention_probe export-episode \
  --episode '<saved seed-89 episode JSON>' --decision '<decision number>' \
  --event-index '<event index>' --output "$PROBE_ROOT/fixtures/seed89-exact.json"
```

The new `scripts/gpt_oss_attention_probe.sbatch` accepts `PROBE_FIXTURE`,
`PROBE_BACKEND=eager|flex_attention`, and a unique `PROBE_OUTPUT_DIR`. Set
`AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0` for both. Run the short fixture in
each backend first; inspect `input_probe.attention_implementation`, identical
`input_ids_sha256`, MXFP4 metadata, and generation status. Then run the same
long fixture in each backend and compare the GPU JSONL peak allocated,
reserved/free before and after, OOM result, and latency. Use one job at a time
on an otherwise comparable L40S. Backend activation, lower memory, and
better episode outcomes are separate claims; this probe can establish only
the first two and long-context generation reliability.

Submit each job separately after inspecting the preceding result (commands
shown for manual execution; none were submitted during implementation):

```bash
export GPU_PARTITION='<verified L40S partition>'
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export PROBE_FIXTURE="$PROBE_ROOT/fixtures/short.json"
export PROBE_BACKEND=eager
export PROBE_OUTPUT_DIR="$PROBE_ROOT/short_eager"
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$PROBE_ROOT/logs/short-eager-%j.out" scripts/gpt_oss_attention_probe.sbatch
# Repeat with short/flex_attention, then long/eager and long/flex_attention,
# changing PROBE_FIXTURE, PROBE_BACKEND, PROBE_OUTPUT_DIR, and log filename.
```

For each completed job, inspect `probe_result.json`; compare
`input_probe.input_ids_sha256` and `max_new_tokens` across the pair before
interpreting memory differences. The raw GPT-OSS telemetry under each
`diagnostics/` directory provides the before/end/peak allocated and reserved
bytes and global free bytes. The two output directories must be distinct.
