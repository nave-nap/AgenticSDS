# Seed 89 replay: worker traceback follow-up

## What the reproduced failure establishes

The user-reported sequential replay completed requests 1–21 and failed during
`model.generate()` on request 22. Input transfer and pre-generation CUDA
synchronization succeeded; post-generation synchronization was not reached.
The earlier request completed its post-generation synchronization. The first
failing GPU operation remains unknown because the saved `worker_traceback` is
null and the worker log lacks a traceback. The generated-token count before
failure is unknown.

## Why the traceback disappeared

The worker captured `traceback.format_exc()` in its exception handler when
`AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1`, but only put it in the JSON protocol
response. The parent recognized `CUDA error: an illegal memory access...` as
fatal, yet its matcher omitted the replay's `CUDA driver error: an illegal memory
access...` wording. That response therefore took the generic `RuntimeError`
branch, which discarded the `traceback`, `request_id`, and `failed_phase` fields.
The replay's `getattr(exc, "worker_traceback", None)` consequently returned null.
This is the code-path explanation consistent with the reported result; the raw
worker protocol response was not transferred for direct inspection.

The patch now captures the original traceback at the start of the worker's
exception handler, before creating the structured response. With the opt-in
flag, it writes the same traceback both to that response and to the worker's
stderr log. The parent recognizes the `CUDA driver error:` form as a typed
`ControllerWorkerFatalCUDA` and retains the traceback, request ID, and failed
phase. It also retains those fields on generic worker errors. These changes
perform no CUDA operation after failure.

## Pinned generation path

The repository calls the pinned GPT-OSS model with the archived `input_ids` and
`attention_mask`, sampling enabled, temperature 1.0, top-p 1.0, top-k 0, and a
1,024-token ceiling. No per-request attention or cache override is supplied.
The replay checks the archived message digest, pinned tokenizer token-ID digest,
worker model metadata, FlexAttention backend, and L40S identity before each
generation. The optional `probe_input` also reports attention-mask shape/hash
and available cache settings. These checks do not reconstruct an unarchived
historical KV-cache state.

In [Transformers 5.22.2 generation code](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/generation/utils.py),
`generate()` prepares a cache and enters `_sample()`. `_sample()` performs a
prefill forward, then repeats model forwards using the cache, processes logits,
and samples with softmax plus `torch.multinomial`. The default cache path is a
`DynamicCache` when caching is enabled and no other cache implementation is
selected; the actual effective setting is recorded by the replay probe.

In the pinned [GPT-OSS model](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py),
the model constructs full and sliding causal masks if `generate()` has not
already supplied them. Each attention layer projects Q/K/V, applies rotary
embeddings, updates the KV cache, and dispatches through the selected attention
interface. The [FlexAttention mask path](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/masking_utils.py)
constructs a `BlockMask` with `_compile=True`, using query and key lengths.
The [FlexAttention integration](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/integrations/flex_attention.py)
uses compiled PyTorch FlexAttention, an attention score modifier, and sink
renormalization. The pinned [MXFP4 expert path](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/integrations/mxfp4.py)
routes tokens into Triton `matmul_ogs` gate/up and down projections. Length
changes prefill shapes, block masks, expert routing sizes, and potential kernel
specializations. PyTorch 2.8's [FlexAttention implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py)
uses a default sparse block size of 128, while its [Inductor kernel](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/flex_attention.py)
tiles work over query blocks. This is a set of possible length-sensitive sites,
not a diagnosis of which one failed.

The current `model_generate` phase marker encloses all of cache preparation,
mask construction, prefill attention, MXFP4 expert kernels, sampled token
selection, and decode cache updates. It cannot distinguish those operations.
The smallest next step is to rerun the bounded sequential replay with the
repaired traceback. If the traceback still ends at a broad `model.generate`
boundary, add diagnostic-only, CPU-side entry/exit hooks on the model forward,
decoder layers, attention modules, and expert modules, recording layer index
and query length. With blocking launches, the last entered section and first
missing exit would narrow prefill versus decode and attention versus MoE.
Only if that remains ambiguous should kernel-level CUDA debugging be added.

## Isolated request-22 mode

`--mode isolated22` selects seed 89 decision 6 event index 1, the archived
request ID 22. It passes that exact message list and 1,024-token ceiling,
checks the 3,724-token count and pinned token-ID hash, uses the same worker
configuration, verifies FlexAttention and NVIDIA L40S metadata, seeds the
episode, and keeps Qwen on the same GPU. It runs one archived Qwen decision-6
input as limited warmup. The fresh worker's internal request ID is 1, while the
reported archived request ID stays 22. This test distinguishes a request-specific
failure from state accumulated across the sequential replay; it does not
replicate the latter's allocator or compilation history.

## Exact next Locust commands

No GPU job is submitted by this patch. First push the new local commit from
`/private/tmp/agenticsds_flex_fatal` and prepare a fresh Locust worktree and
diagnostic directory. Compare the fetched commit with the Mac hash printed by
`git rev-parse HEAD`.

```bash
cd /private/tmp/agenticsds_flex_fatal
git rev-parse HEAD
git push origin autonomous-recovery-v4-flex-fatal-cuda
```

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-flex89-traceback
export DIAG_DIR=/shared_data0/evanpan/agent_support/flex89_traceback_v1
export ARCHIVE_DIR=/shared_data0/evanpan/agent_support/autonomous_recovery_v4/flex_attention_seed89_v1
export RUN_DIR="$ARCHIVE_DIR/runs/autonomous_recovery_v4/batch_03/textcraft_20261123T025300Z_37826937"
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export GPU_PARTITION="$(sacct -j 163640 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test -d "$RUN_DIR"
test ! -e "$DEPLOY_REPO"
test ! -e "$DIAG_DIR"
cd /home/evanpan/AgenticSDS
git fetch origin autonomous-recovery-v4-flex-fatal-cuda
export TRACEBACK_COMMIT="$(git rev-parse origin/autonomous-recovery-v4-flex-fatal-cuda)"
git worktree add --detach "$DEPLOY_REPO" "$TRACEBACK_COMMIT"
test "$(git -C "$DEPLOY_REPO" rev-parse HEAD)" = "$TRACEBACK_COMMIT"
mkdir -p "$DIAG_DIR"
cp "$DEPLOY_REPO"/reports/flex89_bounded_diagnostic/{archived_replay_plan.json,tokenize_plan.py,replay.py,run_replay.sbatch} "$DIAG_DIR/"
export PYTHONPATH="$DEPLOY_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKEN_MANIFEST="$DIAG_DIR/token_manifest.json"
"$AGENT_SUPPORT_GPTOSS_PYTHON" "$DIAG_DIR/tokenize_plan.py" \
  --plan "$DIAG_DIR/archived_replay_plan.json" --config "$RUN_DIR/config.json" \
  --output "$TOKEN_MANIFEST"
"$VENV_PATH/bin/python" "$DIAG_DIR/replay.py" \
  --plan "$DIAG_DIR/archived_replay_plan.json" --token-manifest "$TOKEN_MANIFEST" \
  --backend flex_attention --mode sequence --output-dir "$DIAG_DIR/sequence" --check-only
```

After explicit authorization, submit the sequential traceback replay:

```bash
export REPLAY_BACKEND=flex_attention REPLAY_MODE=sequence
export DIAG_OUTPUT_DIR="$DIAG_DIR/sequence"
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/sequence-%j.out" "$DIAG_DIR/run_replay.sbatch"
```

Inspect `sequence/replay_result.json`, `sequence/workers/gpt_oss_controller_worker.log`,
and `sequence/diagnostics/gpt_oss_cuda_memory_*.jsonl` before deciding on any
further run. After separate authorization, the isolated test command is:

```bash
export REPLAY_BACKEND=flex_attention REPLAY_MODE=isolated22
export DIAG_OUTPUT_DIR="$DIAG_DIR/isolated22"
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$DIAG_DIR/isolated22-%j.out" "$DIAG_DIR/run_replay.sbatch"
```
