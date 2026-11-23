# Seed 89: first GPU operation diagnostic

## Archived input audit

`audit_archive.py` compared the two episode JSON files, the first 22 GPT-OSS
telemetry start/end pairs, and `archived_replay_plan.json`. It found all 22
complete controller message lists in historical order. Requests 1–8 belong to
seed 151; requests 9–22 belong to seed 89. The full per-request mapping, exact
input-token counts, output-token counts where available, and SHA-256 hashes of
the message lists are in `archive_audit.json`.

| Seed 89 decision | GPT-OSS request IDs | Input tokens |
| --- | --- | --- |
| 1 | 9–10 | 2445, 2799 |
| 2 | 11–13 | 2375, 2632, 2862 |
| 3 | 14–16 | 2676, 2983, 3205 |
| 4 | 17 | 3016 |
| 5 | 18–20 | 3071, 3352, 3568 |
| 6 | 21–22 | 3383, 3724 |

Request 21 generated 157 tokens and requested `inspect_state_v1` for inventory.
The support returned `empty`. Request 22 is its follow-up controller call. It
has a 1,024-token output ceiling and no archived output. Its message-list SHA-256
is `6800e9955958bf890e0409c2265a4f711b86e6373ca3f2f6d142e0e53ee75492`.

The historical archive contains *message hashes and token counts*, but no
historical token-ID hashes or attention-mask tensors. Matching serialized
messages alone therefore does not establish token identity. The CPU-only
`tokenize_plan.py` must run with the pinned GPT-OSS tokenizer and date-pinned
Harmony template on Locust before any GPU replay. It verifies all 22 historical
counts and writes runtime token-ID hashes. The replay then compares each
worker-side token-ID hash to that manifest before generating.

The code uses the same model load, chat template, `model.generate` kwargs,
sampling settings, and 1,024-token ceiling for these requests. There is no
archived request-specific `use_cache` or attention-backend override. The
tokenized attention mask and effective model/generation `use_cache` values are
not in the old logs. The replay's `probe_input` records mask length, nonzero
count, mask hash, and available cache flags per request. Runtime cache state
and internal kernels may still differ with request length and prior executions.

The reported successful 7,555-token synthetic probe is evidence against a
simple prompt-length threshold, but its result and token IDs are outside the
transferred seed 89 archive. It is not changed by this diagnostic.

## Bounded experiment

`replay.py --mode sequence` loads one persistent GPT-OSS worker, then co-resident
Qwen on the same L40S, matching the archived loader order. It replays the 13
archived Qwen inputs and the 22 archived GPT-OSS controller inputs through seed
89 decision 6. It reseeds GPT-OSS at the original episode boundary. Generated
responses do not alter later inputs; later messages come from the archive. It
creates no TextCraft environment and invokes no recovery support. It stops on
the first worker error or incomplete generation and never exceeds 22 GPT-OSS
generations. Qwen's generated outputs and GPU allocator history may differ from
the historical run.

For this diagnostic only, the Slurm script enables `CUDA_LAUNCH_BLOCKING=1`,
worker tracebacks, CUDA memory telemetry, and `AGENT_SUPPORT_GPTOSS_PHASE_TRACE=1`.
The worker records request ID and the first failing phase: input transfer
(including a diagnostic synchronization), pre-generation synchronization,
`model.generate`, or post-generation synchronization. Phase events contain no
CUDA memory queries. After a fatal CUDA error, generation-end telemetry skips
its memory snapshot. A failure within `model.generate` may still require a
kernel-level debugger to distinguish FlexAttention, MXFP4, sampling, or cache
operations. The full worker traceback is the next evidence to inspect.

After the sequential result is understood, `--mode isolated22` performs only
archived request 22 with a fresh worker and Qwen co-resident. It is a separate
experiment; its worker request ID is 1 while its archived request ID remains 22.
One archived Qwen decision-6 input provides limited co-resident warmup. This
mode cannot replace the sequential replay.

The base repair commit `1609f85` already classifies structured illegal-memory
access as `controller_cuda_fatal`, marks the worker unusable, preserves the
first detail and optional traceback, and stops the v4 episode before another
fallback environment action. The added offline tests cover request/phase
tracebacks, phase telemetry, the 22-call stop, tokenizer preflight, and the
fresh-worker mode.

## Locust deployment and Slurm commands

These commands prepare and **describe** the GPU jobs. Do not submit either
`sbatch` command until the user authorizes GPU jobs. Use a fresh diagnostic
directory and a fresh deploy worktree; the frozen evaluation artifacts remain
untouched.

On the Mac, from this isolated worktree after committing the diagnostic changes:

```bash
cd /private/tmp/agenticsds_flex_fatal
git push origin autonomous-recovery-v4-flex-fatal-cuda
export LOCUST_HOST=locust-login.seas.upenn.edu
export DIAG_DIR=/shared_data0/evanpan/agent_support/flex89_first_gpu_op_v1
ssh "$LOCUST_HOST" "test ! -e '$DIAG_DIR' && mkdir -p '$DIAG_DIR'"
scp reports/flex89_bounded_diagnostic/{archived_replay_plan.json,audit_archive.py,tokenize_plan.py,replay.py,run_replay.sbatch} \
  "$LOCUST_HOST:$DIAG_DIR/"
```

On Locust, prepare the exact code and CPU-only tokenizer manifest:

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-flex89-first-op
export DIAG_DIR=/shared_data0/evanpan/agent_support/flex89_first_gpu_op_v1
export ARCHIVE_DIR=/shared_data0/evanpan/agent_support/autonomous_recovery_v4/flex_attention_seed89_v1
export RUN_DIR="$ARCHIVE_DIR/runs/autonomous_recovery_v4/batch_03/textcraft_20261123T025300Z_37826937"
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export TOKEN_MANIFEST="$DIAG_DIR/token_manifest.json"
export GPU_PARTITION="$(sacct -j 163640 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test -d "$RUN_DIR"
test -d "$HF_HOME"
test ! -e "$DEPLOY_REPO"
cd /home/evanpan/AgenticSDS
git fetch origin autonomous-recovery-v4-flex-fatal-cuda
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-recovery-v4-flex-fatal-cuda
export PYTHONPATH="$DEPLOY_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$VENV_PATH/bin/python" "$DIAG_DIR/audit_archive.py" \
  --archive "$ARCHIVE_DIR" --plan "$DIAG_DIR/archived_replay_plan.json" \
  --output "$DIAG_DIR/archive_audit_remote.json"
"$AGENT_SUPPORT_GPTOSS_PYTHON" "$DIAG_DIR/tokenize_plan.py" \
  --plan "$DIAG_DIR/archived_replay_plan.json" --config "$RUN_DIR/config.json" \
  --output "$TOKEN_MANIFEST"
"$VENV_PATH/bin/python" "$DIAG_DIR/replay.py" \
  --plan "$DIAG_DIR/archived_replay_plan.json" --token-manifest "$TOKEN_MANIFEST" \
  --backend flex_attention --output-dir "$DIAG_DIR/sequence" --check-only
```

After authorization, submit only the sequential job first:

```bash
export REPLAY_BACKEND=flex_attention REPLAY_MODE=sequence
export DIAG_OUTPUT_DIR="$DIAG_DIR/sequence"
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/sequence-%j.out" "$DIAG_DIR/run_replay.sbatch"
```

Inspect `sequence/replay_result.json`, `sequence/workers/gpt_oss_controller_worker.log`,
and `sequence/diagnostics/gpt_oss_cuda_memory_*.jsonl`. Only after interpreting
that result and obtaining separate authorization for the isolated test, use:

```bash
export REPLAY_BACKEND=flex_attention REPLAY_MODE=isolated22
export DIAG_OUTPUT_DIR="$DIAG_DIR/isolated22"
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/isolated22-%j.out" "$DIAG_DIR/run_replay.sbatch"
```

No GPU run has been launched here. The first failing GPU operation remains
unknown until the blocking replay and worker traceback are captured.
