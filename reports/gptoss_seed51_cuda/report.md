# Bounded GPT-OSS CUDA diagnostic for Slurm 164919

## Confirmed from the transferred source archive

The archive `slurm-164919-diagnostics.tar.gz` has SHA-256
`4539897be1fe345a283454d8b88087942a66a743ac7b88d5cefea60a206429a4`.
Its run is `state_aware_v1`, TextCraft seed 51, commit
`701c25d4a76903743148210b3f3e780f246d7131`, on an NVIDIA L40S with
active `flex_attention`. The episode ended `controller_cuda_fatal` after one
environment action and is marked infrastructure-invalid. The worker logged a
complete `torch.AcceleratorError` traceback for GPT-OSS request 10 during
`model_generate`. The visible frame is GPT-OSS `post_attention_layernorm`, at
`hidden_states.to(torch.float32)`.

The episode trace contains ten exact GPT-OSS controller message lists: three
in decision 1 and seven in decision 2. There are no GPT-OSS
`reason_over_artifacts_v1` support generations interleaved. Two memory
retrievals account for the episode's two support model calls; they use the
embedding model, not this GPT-OSS worker. Qwen's two input message lists are
also saved. The CPU audit verified the frozen configuration, archived worker
identity, run/episode/log hashes, request order, counts, and fatal marker.
The source `run.json`, episode JSON, and worker log SHA-256 values are,
respectively, `213a0f8c98f57662a1c4dc53a9e06214426e699e056c773179d65f811994825b`,
`19f2c623fe496a7a421533544bed86c4732f021d0fe72f703633eda5be10065c`,
and `da08f515674ae1da9c4463848ea5665c71b4ed81d72b8f22a15c47aad5154051`.

| Request | Input tokens | Output tokens | Output ceiling |
| ---: | ---: | ---: | ---: |
| 1 | 2,278 | 151 | 1,024 |
| 2 | 2,818 | 306 | 1,024 |
| 3 | 2,656 | 173 | 1,024 |
| 4 | 2,418 | 92 | 1,024 |
| 5 | 2,727 | 75 | 1,024 |
| 6 | 2,947 | 240 | 1,024 |
| 7 | 4,926 | 98 | 1,024 |
| 8 | 4,565 | 141 | 1,024 |
| 9 | 5,861 | 443 | 1,024 |
| **10** | **5,451** | **unknown** | **1,024** |

The source archive has no GPU-memory diagnostic JSONL. The ninth successful
generation reported a GPT-OSS worker peak of 14,514,766,336 bytes allocated
(13.52 GiB) and 14,873,001,984 bytes reserved (13.85 GiB). The L40S reported
47,676,129,280 total bytes (44.40 GiB). These are worker metrics, not total
co-resident GPU usage; request 10's memory at failure is unknown. The worker
uses PyTorch 2.8.0+cu128, Transformers 5.22.2, Triton 3.4.0,
`kernels==0.17.2`, packed MXFP4, and GPT-OSS revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. Sampling remains temperature
1, top-p 1, top-k 0, and a 1,024-token ceiling.

The evaluator returns normally after recording a terminal worker failure, so
Slurm's `COMPLETED`/exit 0 means the process wrote its result. It does **not**
make the stage valid: the stage gate rejects infrastructure-invalid runs.

## What remains unknown

The saved traceback was taken with asynchronous CUDA execution. PyTorch
explains that an error can surface at a later API call, so the layernorm
`.to()` frame does not establish layernorm as the faulty kernel
([PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)).
FlexAttention, packed MXFP4/Triton expert operations, and another GPU
operation remain hypotheses. The error type is illegal memory access, not a
reported CUDA out-of-memory failure. Previous failures at different request
numbers support intermittency but do not identify a kernel.

The historical output-token count for request 10 is unrecoverable from these
files: `generate()` raised before returning decoded IDs. The replay records
`output_tokens: null` on any fatal request, the exact prompt count and output
ceiling, and the last successful output count. A token-progress callback could
bound the number of completed tokens in a later instrumented experiment, but
would change generation timing; it is intentionally excluded from this first
bounded replay.

The *messages* are exactly replayable. Historical tokenizer token-ID hashes,
the worker's intermediate RNG state, compiled-kernel cache, KV-cache contents,
and CUDA allocator state were not archived. Replaying the ten archived inputs
with the same pinned worker and Qwen prompts is a controlled reproduction
attempt, not a bit-for-bit reconstruction of the original GPU execution.

## CPU-only preparation completed

`audit.py` produced a ten-request replay plan from the transferred archive
without loading a model or environment. `replay.py --check-only` validated the
plan and reported request 10's 5,451 input tokens. The diagnostic-only unit
tests passed, and the complete offline suite passed **263 tests**. The Slurm
script passed `bash -n` and the policy/configuration diff is empty.
The source archive and all four-policy experiment roots remain untouched.

## Separate Locust deployment and CPU preflight

These are commands for the operator. They have **not** been run on Locust from
this workspace; the documented SSH hostname did not resolve here. Set
`LOCUST_HOST` to your working SSH hostname if it differs.

On the Mac, after this diagnostic branch is committed:

```bash
cd /private/tmp/AgenticSDS-seed51-fatal-diagnostic
test -z "$(git status --porcelain)"
export DIAG_COMMIT="$(git rev-parse HEAD)"
git push origin gptoss-seed51-fatal-diagnostic
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin gptoss-seed51-fatal-diagnostic && test ! -e /home/evanpan/AgenticSDS-seed51-fatal-diagnostic && git worktree add --detach /home/evanpan/AgenticSDS-seed51-fatal-diagnostic $DIAG_COMMIT"
ssh "$LOCUST_HOST"
```

On the Locust login node, perform only CPU preflight:

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-seed51-fatal-diagnostic
export RUN_DIR=/shared_data0/evanpan/agent_support/recovery_policy30_terminal_review_v1/smoke/state_aware_v1/textcraft_20261123T211827Z_ab9f308f
export DIAG_DIR=/shared_data0/evanpan/agent_support/gptoss_seed51_cuda_diagnostic_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export GPU_PARTITION="$(sacct -j 164919 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test -d "$RUN_DIR"
test ! -e "$DIAG_DIR"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
mkdir -p "$DIAG_DIR"
cd "$DEPLOY_REPO"
test -z "$(git status --porcelain)"
bash -n reports/gptoss_seed51_cuda/run_replay.sbatch
"$VENV_PATH/bin/python" -m reports.gptoss_seed51_cuda.audit \
  --run-dir "$RUN_DIR" --output-dir "$DIAG_DIR/source_audit"
"$VENV_PATH/bin/python" -m reports.gptoss_seed51_cuda.replay \
  --plan "$DIAG_DIR/source_audit/replay_plan.json" \
  --backend flex_attention --output-dir "$DIAG_DIR/flex_blocking" --check-only
```

Inspect `source_audit/audit.json` for ten reconstructed requests and the
source-file hashes above. If the partition accounting record expired, verify
the L40S partition with `sinfo -o '%P %G %m %l'`. CPU preflight does not load
either model or submit a GPU job.

## Proposed single-variable GPU experiments — authorization required

**Experiment A: blocking replay.** After authorization, run once in a fresh
worker with the original FlexAttention backend and pinned settings. The
diagnostic additionally enables `CUDA_LAUNCH_BLOCKING=1`, traceback capture,
and per-request memory telemetry. It stops on the first error and never
retries inside that process. The diagnostic exits nonzero on failure, unlike
the benchmark evaluator.

```bash
cd "$DEPLOY_REPO"
export REPLAY_BACKEND=flex_attention DIAG_OUTPUT_DIR="$DIAG_DIR/flex_blocking"
sbatch --wait --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/flex-blocking-%j.out" \
  reports/gptoss_seed51_cuda/run_replay.sbatch
```

Inspect `flex_blocking/replay_result.json`, its worker log, and
`diagnostics/gpt_oss_cuda_memory_*.jsonl`. The JSONL start record captures
allocated/reserved memory before each generation; the end record captures
input count and peak memory when CUDA remains usable. A fatal end deliberately
skips further CUDA measurements. The output-token count remains unknown if
`generate()` aborts before returning.

**Experiment B: attention backend only.** If A fails or finishes, and after a
separate authorization, run the same ten prompts in another fresh worker with
only `AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=0` (the pinned model's default
backend) changed. Both jobs use blocking and telemetry. The replay verifies
and records the *actual* default backend; it makes no assumption that the
default is eager. Compare request completion, traceback, loaded and peak
memory, and Qwen colocation. If the default backend OOMs, record that as an
infeasible workaround, not a task failure.

```bash
export REPLAY_BACKEND=default DIAG_OUTPUT_DIR="$DIAG_DIR/default_blocking"
sbatch --wait --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/default-blocking-%j.out" \
  reports/gptoss_seed51_cuda/run_replay.sbatch
```

Do not treat a single clean ten-request replay as proof of stability. The
smallest plausible workaround is disabling FlexAttention **if** the default
backend fits the L40S alongside Qwen and survives repeated bounded runs from
fresh workers. Its expected reliability and memory impact are unmeasured
today. If blocking identifies a kernel in FlexAttention, that strengthens the
case for this workaround; if it identifies MXFP4/Triton, changing attention
alone may not help. If the traceback remains ambiguous, add a separate,
small layer/attention/expert boundary trace before using kernel-level tools.

**Recommendation:** keep the four-policy evaluation paused. First obtain the
blocking traceback and memory telemetry, then compare the default backend in
one paired bounded replay. Only after the candidate backend has demonstrated
repeatability and adequate L40S memory headroom should a *new* evaluation
protocol explicitly select and verify it for all four arms. Never fold these
diagnostic runs into the existing experiment root or score them as tasks.
