# Eager replay: L40S memory identity correction

The operator reports that the bounded FlexAttention replay reproduced the
fatal illegal-memory-access error on request 10 with
`CUDA_LAUNCH_BLOCKING=1`. The synchronized traceback runs through
`create_sliding_window_causal_mask → flex_attention_mask → create_block_mask`
to a compiled Triton kernel launch. This materially narrows the likely
failing path to FlexAttention block-mask construction. The replay result files
were not transferred to this workspace, so exact memory measurements and the
full synchronized traceback remain to be independently checked.

The first default-backend replay loaded **eager** attention but stopped at the
diagnostic worker identity check before any generation. The archived and new
L40S devices reported different `gpu_total_memory_bytes`. That value is
capacity telemetry, not a model or kernel identity. The diagnostic validator
now records both values in `replay_result.json` without requiring equality.
It still requires the same GPT-OSS model/revision, Transformers/PyTorch/CUDA/
Triton/kernels versions, packed quantization, decoding, prompt template,
reasoning setting, GPU name and compute capability. For the default run it
requires the active backend to be exactly `eager`; for the Flex run it requires
exactly `flex_attention`. No policy, evaluator, or model implementation changed.

The original Flex replay and preflight-failed eager outputs must remain
untouched. The next step is **one eager-only replay** in a fresh worker and
fresh output directory. Do not submit another Flex job for this correction.

## Mac: publish the corrected diagnostic commit

```bash
cd /private/tmp/AgenticSDS-seed51-fatal-diagnostic
test -z "$(git status --porcelain)"
export DIAG_COMMIT="$(git rev-parse HEAD)"
git push origin gptoss-seed51-fatal-diagnostic
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin gptoss-seed51-fatal-diagnostic && test ! -e /home/evanpan/AgenticSDS-seed51-eager-replay-fix && git worktree add --detach /home/evanpan/AgenticSDS-seed51-eager-replay-fix $DIAG_COMMIT"
ssh "$LOCUST_HOST"
```

Use the operator's working SSH connection if the hostname or authentication
setup differs. The separate Locust worktree preserves the prior diagnostic
source checkout.

## Locust: CPU-only preflight

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-seed51-eager-replay-fix
export DIAG_DIR=/shared_data0/evanpan/agent_support/gptoss_seed51_cuda_diagnostic_v1
export DIAG_OUTPUT_DIR="$DIAG_DIR/eager_blocking_memory_tolerant_v1"
export REPLAY_BACKEND=default
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export GPU_PARTITION="$(sacct -j 164919 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test -d "$DEPLOY_REPO"
test -r "$DIAG_DIR/source_audit/replay_plan.json"
test ! -e "$DIAG_OUTPUT_DIR"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
cd "$DEPLOY_REPO"
test -z "$(git status --porcelain)"
bash -n reports/gptoss_seed51_cuda/run_replay.sbatch
"$VENV_PATH/bin/python" -m reports.gptoss_seed51_cuda.replay \
  --plan "$DIAG_DIR/source_audit/replay_plan.json" \
  --backend default --output-dir "$DIAG_OUTPUT_DIR" --check-only
```

If Slurm accounting no longer identifies the L40S partition, inspect
`sinfo -o '%P %G %m %l'` and set `GPU_PARTITION` to the verified L40S
partition. The preflight loads no GPU model.

## Locust: one eager-only GPU job after authorization

```bash
cd "$DEPLOY_REPO"
sbatch --wait --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/eager-memory-tolerant-%j.out" \
  reports/gptoss_seed51_cuda/run_replay.sbatch
```

The job starts a fresh GPT-OSS worker, replays at most ten archived requests,
and stops at the first error. It exits nonzero on a replay failure. Inspect:

```bash
"$VENV_PATH/bin/python" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["DIAG_OUTPUT_DIR"])
value = json.loads((root / "replay_result.json").read_text())
print("status:", value["status"])
print("active backend:", value.get("worker_metadata", {}).get("attention_implementation"))
print("GPU capacity bytes:", value.get("gpu_total_memory_bytes"))
print("completed requests:", value["completed_generations"])
print("first failure:", value.get("first_failure"))
for row in value["generations"]:
    print(row["request_id"], row.get("status"), row.get("input_tokens"),
          row.get("output_tokens"), row.get("memory_before"), row.get("memory_after"))
PY
find "$DIAG_OUTPUT_DIR" -type f | sort
```

Compare this eager result with the **existing** Flex result by request count,
fatal error, loaded/peak allocated and reserved memory, global free memory,
and Qwen colocation. A ten-request pass establishes feasibility for this
bounded case only; it is not a long-run reliability claim. If eager OOMs or
fails, keep the four-policy evaluation paused and use the protocol's bounded
runtime decision gate. Do not rerun a fatal request within the same worker.
