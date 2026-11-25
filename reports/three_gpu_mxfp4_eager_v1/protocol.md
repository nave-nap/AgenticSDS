# Three-GPU packed MXFP4 eager runtime, v1

## Scope and decision rule

This is a fresh runtime experiment. The existing v3 pilot and batch directories, all four policy definitions, frozen manifests, all policy/config JSON, prompts, memory corpus and index, model revisions, decoding, action limits, and evaluation code paths remain unchanged. The only opt-in change is physical model placement in the GPT-OSS worker and its placement/memory validation. Qwen is isolated on allocated GPU 0. The worker sees allocated GPUs 1 and 2 as CUDA ordinals 0 and 1 and loads GPT-OSS 24 decoder layers 12/12, with the embedding on its first device and final norm/head on its second. Attention is explicitly eager. The new output root is `/shared_data0/evanpan/agent_support/three_gpu_mxfp4_eager_v1`.

The approach uses Transformers 5.22.2's explicit module `device_map`, not `"auto"`. Its [pinned GPT-OSS model](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py) marks whole decoder layers as unsplittable. Its [pinned MXFP4 integration](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/integrations/mxfp4.py) uses each layer's target device when constructing packed Triton tensors. The [pinned quantizer](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/quantizers/quantizer_mxfp4.py) accepts a device map for a prequantized checkpoint. These are source-level support signals, not a GPU success claim. At load, the runtime checks every ordinary parameter, every packed expert backing tensor, and every scale on the assigned device. It rejects missing, CPU, meta, one-GPU, or dequantized placement. The bounded feasibility test is required to establish that dispatch and generation work on the two L40S cards.

The feasibility job makes one 32-token warmup request and one request with exactly 8,887 input tokens and at most 1,024 output tokens. It does not advance TextCraft. If `THREE_GPU_MESSAGES_FILE` points to a JSON list of the exact historical messages (or an object with `messages`), the job checks that it tokenizes to 8,887 and replays it. The local archive has no exact 8,887-token message; without that file, the job constructs a synthetic request of the same token length and labels it `synthetic_exact_length`. In that case, a later exact-message replay is needed before making a claim about the particular historical request.

The worker writes JSONL load and generation records under `feasibility_8887/diagnostics/`. They contain the actual module map, verified packed weight and scale bytes by worker GPU, per-GPU allocated/reserved peaks, sampled minimum global free memory, sample counts, and error type/detail including OOM. The feasibility result includes Qwen GPU-0 placement, load-time peaks, and sampled minimum free memory. `infrastructure_valid` requires allocation growth and free-memory samples on both GPT-OSS GPUs. A failed load or generation leaves a `failed` result and prevents smoke/batch submission via the gate.

## CPU-only checks

From the committed Mac checkout, run:

```bash
cd /Users/evanpan/AgenticSDS
python3 -m agent_support.three_gpu_feasibility preflight
python3 -m pytest -q tests/test_three_gpu_runtime.py tests/test_gpu_memory_diagnostics.py
bash -n scripts/three_gpu_mxfp4.sbatch
git diff --check
```

The CPU preflight checks the frozen 30-task manifest and both v4 arm configs and validates the M2-200 corpus/index digests. It reports the unchanged model revisions and config digests. It does not verify CUDA placement.

## Exact Locust commands, to run after GPU-job authorization

First publish the implementation commit from the Mac. On Locust, deploy it to a **new clean worktree**. Use the existing verified Qwen and GPT-OSS environments and cache; the paths below are those recorded in prior Locust diagnostics. The partition query retrieves the prior L40S partition; if Slurm accounting has expired, supply that same partition explicitly.

```bash
cd /Users/evanpan/AgenticSDS
git push origin HEAD:refs/heads/three-gpu-mxfp4-eager-v1
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST"
export BASE_REPO=/home/evanpan/AgenticSDS
export DEPLOY_REPO=/home/evanpan/AgenticSDS-three-gpu-mxfp4-eager-v1
export THREE_GPU_ROOT=/shared_data0/evanpan/agent_support/three_gpu_mxfp4_eager_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 163640 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test ! -e "$DEPLOY_REPO"
test ! -e "$THREE_GPU_ROOT"
git -C "$BASE_REPO" fetch origin three-gpu-mxfp4-eager-v1
git -C "$BASE_REPO" worktree add --detach "$DEPLOY_REPO" origin/three-gpu-mxfp4-eager-v1
cd "$DEPLOY_REPO"
test -z "$(git status --porcelain --untracked-files=all)"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility preflight
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval preflight --arm expanded_v3_v1
mkdir -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=feasibility
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/feasibility-%j.out" scripts/three_gpu_mxfp4.sbatch
```

Set `THREE_GPU_MESSAGES_FILE=/absolute/path/to/exact_messages.json` before `sbatch` if the exact historical request is available. Do not submit a second feasibility job to the same output directory. Inspect the Slurm log, `feasibility_8887/feasibility_result.json`, worker log, and GPT-OSS JSONL diagnostics. Verify `status=infrastructure_valid`, eager attention, Qwen's sole GPU-0 map, 12 packed expert modules and nonzero packed weight/scale bytes on each worker GPU, both GPUs' allocation growth and sampled minimum free memory, and no OOM. If any check fails, stop and report the sharding limitation before considering another engine.

Only after that result is valid, initialize a new evaluation plan and run matched two-task smokes. Each job checks the feasibility gate against its current commit and frozen protocol. Inspect both smoke traces before starting a pilot batch.

```bash
cd "$DEPLOY_REPO"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility gate \
  --result "$THREE_GPU_ROOT/feasibility_8887/feasibility_result.json"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval init \
  --root "$THREE_GPU_ROOT/evaluation"
export THREE_GPU_PHASE=smoke RECOVERY_ARM=expanded_v3_v1
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/smoke-expanded-%j.out" scripts/three_gpu_mxfp4.sbatch
export THREE_GPU_PHASE=smoke RECOVERY_ARM=autonomous_recovery_v4
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/smoke-recovery-%j.out" scripts/three_gpu_mxfp4.sbatch
```

After inspecting both matched smokes, run one matched pilot batch per arm. This reuses the frozen first three development tasks within a fresh root. It is a runtime feasibility pilot, not held-out evidence.

```bash
cd "$DEPLOY_REPO"
export THREE_GPU_PHASE=batch RECOVERY_ARM=expanded_v3_v1 RECOVERY_BATCH=0
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/pilot-expanded-%j.out" scripts/three_gpu_mxfp4.sbatch
export THREE_GPU_PHASE=batch RECOVERY_ARM=autonomous_recovery_v4 RECOVERY_BATCH=0
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/pilot-recovery-%j.out" scripts/three_gpu_mxfp4.sbatch
```

No GPU job has been submitted by this implementation. No existing allocation is modified.
