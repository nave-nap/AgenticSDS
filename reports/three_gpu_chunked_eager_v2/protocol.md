# Three-GPU chunked eager feasibility, v2

## Failed v1 evidence

The user supplied the worker log and CUDA JSONL from Slurm job 169107, commit `60e5eb0a9c4d715511d3d12785ed83d6877e669e`. These files remain under `reports/three_gpu_mxfp4_eager_v1/debug_logs/` and are not part of this experiment's output. The worker log contains model-loading progress only; it has **no Python traceback**. The JSONL identifies a 9.42 GiB CUDA allocation failure during the 8,887-token generation, but cannot identify the precise Python line. The v1 worker caught and returned the exception without printing its traceback.

The size matches one BF16 eager attention score matrix: `1 batch × 64 heads × 8,887 queries × 8,887 keys × 2 bytes = 9.415 GiB`. In [pinned GPT-OSS eager attention](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L206-L232), the matrix is materialized by `torch.matmul`, then additional full-sized matrices are created by masking, sink concatenation, max subtraction, and softmax. Given 25.93 GiB allocated at failure and a roughly 6.42 GiB loaded baseline, two such matrices were likely live when the third was requested. The exact failing line is unresolved without a traceback; the candidates are max subtraction and softmax. This is an inference from shapes and allocator numbers, not a traced stack frame.

Both GPUs showed a small memory rise during the 70-token warmup, consistent with execution crossing the 12/12 split. During the failed long request, GPU 1 stayed near its loaded baseline, so that request did not reach its layers. Placement checks alone cannot prove forward execution. The new run records each layer's actual `q_proj` input device and completed output device during prefill.

GPT-OSS builds full and sliding causal masks on the first device before iterating layers. Each BF16 8,887-by-8,887 mask is about 0.147 GiB, far smaller than the 9.42 GiB allocation. The 9.43 GiB of inactive split blocks shows fragmentation, but an allocator setting cannot remove the multiple full-sized attention matrices required by this eager implementation. No allocator setting is changed in v2. SDPA is [unsupported for GPT-OSS attention sinks](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/gpt_oss); the pinned model's compatible FlashAttention list does not establish a verified L40S option. FlexAttention is excluded because of the prior instability.

## Runtime change

V2 registers `gptoss_chunked_eager_v1` through Transformers' [supported attention and matching mask registries](https://huggingface.co/docs/transformers/v5.20.2/attention_interface). It calls the pinned eager function on 256-query slices, each attending to the full key range with the same causal or sliding mask and learned attention sink. The largest score slice at 8,887 keys is about 0.271 GiB. This preserves weights, packed MXFP4 experts, input messages, decoding, frozen tasks, policy code, and limits. Query-block GEMMs may produce small floating-point differences, so v2 remains a separate runtime experiment.

The worker records full Python tracebacks for two-GPU errors, attention chunk start/end/error snapshots, per-layer prefill entry and completion on their executing GPUs, and per-GPU peak allocated/reserved and sampled minimum free memory. The feasibility gate requires one completed 8,887-token generation, all 24 layers executing on the expected devices, chunked attention on layers 0 and 23, allocation growth on both GPUs, and at least 10 GiB sampled free on each worker GPU. The bounded job uses one 32-token warmup and one 8,887-token request with the frozen 1,024-token output ceiling; it never advances TextCraft.

## Offline checks

```bash
cd /Users/evanpan/AgenticSDS
python3 -m agent_support.three_gpu_feasibility preflight
python3 -m pytest -q tests/test_gpt_oss_chunked_eager.py tests/test_three_gpu_runtime.py
bash -n scripts/three_gpu_chunked_eager.sbatch scripts/three_gpu_mxfp4.sbatch
git diff --check
```

## Next GPU validation command, pending authorization

Publish the new implementation commit to `three-gpu-chunked-eager-v2`, then deploy that commit in a fresh clean Locust worktree. Set the same pinned environment paths and L40S partition used in v1. The output root is new: `/shared_data0/evanpan/agent_support/three_gpu_chunked_eager_v2`. Do not reuse the failed v1 root or modify job 169107's files.

```bash
cd /home/evanpan/AgenticSDS-three-gpu-chunked-eager-v2
export THREE_GPU_ROOT=/shared_data0/evanpan/agent_support/three_gpu_chunked_eager_v2
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 169107 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test ! -e "$THREE_GPU_ROOT"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility preflight
mkdir -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=feasibility
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/feasibility-%j.out" scripts/three_gpu_chunked_eager.sbatch
```

Set `THREE_GPU_MESSAGES_FILE` to a JSON file containing the exact archived 8,887-token messages before the `sbatch` command if available. Otherwise the driver constructs an exact-length synthetic request and records that source. Inspect `feasibility_8887/feasibility_result.json`, the worker log, and JSONL. Advance to a fresh matched smoke and pilot only if `status` is `infrastructure_valid`. A failed result stops the gate; report its traceback and layer/attention events before considering another runtime change.
