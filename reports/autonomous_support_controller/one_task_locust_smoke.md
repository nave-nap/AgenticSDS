# One-task TextCraft autonomous-controller smoke on Locust

This is a deployment procedure, not a benchmark result. Do not submit a job
until the checkout, paths, packages, caches, and GPU allocation have passed
the checks below. No historical M2 corpus/index or result directory is
modified. The smoke uses the first task in the frozen 20-task development
manifest; `agent_support.autonomous_evaluation.STAGE_SIZES["smoke"] == 1`.

## Reviewed source boundary

The intended commit contains the opt-in `agent_support/autonomous_*.py`
framework/evaluation files, GPT-OSS worker/backend, prompted controller,
memory adapter, support operations, sequence analysis, existing learned-gate
files, their tests/configs/docs, `agent_support/common.py`, `runner.py`, and
`scripts/autonomous_support_eval.sbatch`. The untracked
`reports/complete_research_summary/` materials are historical reporting work
and are excluded from this deployment commit. `data/` is ignored and must not
be added. Before committing, inspect `git diff --cached --name-only` and
`git diff --cached --check`.

The repository's `origin` is `https://github.com/nave-nap/AgenticSDS.git`.
Push a new reviewed branch, then create a **separate clean worktree** from
that branch on Locust. The old Locust checkout and all M2 artifacts remain in
place. Use the existing Qwen environment via `VENV_PATH` and the existing
pinned GPT-OSS environment via `AGENT_SUPPORT_GPTOSS_PYTHON`; do not install a
third environment just for this smoke.

## Frozen memory paths and model identities

Set `AGENT_SUPPORT_M2_200_ARTIFACT_DIR` to an absolute directory. The new
evaluation CLI resolves `corpus_200.json` and `index_200.json` for both the
autonomous and always-on M2-200 arms from that directory, while retaining the
configured SHA-256 digests. The Slurm script defaults the variable to the
user-supplied Locust directory:

```
/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
```

For reproducibility, export the **same** value in the login shell for
preflight and job submission. The path is embedded in the saved config
snapshot and affects its digest. `AGENT_SUPPORT_DATA` controls output and
default model cache locations, not the frozen memory path. The frozen
development manifest and recipe-order snapshot are checked by preflight.

The autonomous config pins Qwen/Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` and GPT-OSS-20B revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. The run snapshots
`config.json` (both IDs/revisions), `run.json` (controller worker metadata and
revisions), `manifest.json`, recipe order, episode JSON, summary, and the
worker stderr log. The matching operational control is the
`qwen_standalone` arm, which shares all task-agent settings and the first
frozen task. Historical GPT-OSS standalone is not a matched control.

## GPU fit and safest first measurement

An L40S has 48 GB of GPU memory according to
[NVIDIA](https://www.nvidia.com/en-sg/data-center/l40s/). OpenAI gives a
rough 16 GB requirement for packed GPT-OSS-20B MXFP4 in its
[Transformers guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers),
while the preserved standalone GPT-OSS run recorded about 20–21 GB reserved
GPU memory. Qwen2.5-7B is loaded in NF4, and `MPNetEmbedder` explicitly loads
on CPU. These facts make co-residency on an otherwise free 48 GB L40S
plausible, **not verified**: separate PyTorch processes, generation KV cache,
Triton kernels, allocator fragmentation, and other node users matter.
[Transformers' MXFP4 documentation](https://huggingface.co/docs/transformers/main/quantization/mxfp4)
lists CUDA compute capability 7.5+; the pinned worker checks its actual
dependencies and packed MXFP4 representation and rejects dequantization.
OpenAI's Transformers guide describes a narrower Hopper-or-later support
range, so documentation alone does not settle the L40S/Ada case for the
pinned stack.
Only successful simultaneous initialization and generation on the assigned
L40S establishes compatibility.

The first autonomous smoke loads Qwen, starts the one persistent GPT-OSS
worker, then loads MPNet on CPU before the first task step. The Slurm script
samples `nvidia-smi` GPU total/used memory every two seconds during smoke
and records a separate CSV. The worker metadata records its own peak CUDA
allocation/reservation. Review both the CSV and worker metadata; sampling
can miss a brief peak, so an out-of-memory failure or packed-weight check is
still decisive. The job defaults to `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` for smoke; missing cached checkpoints or kernels
cause a visible failure rather than a download.

## Mac: prepare and transfer the intended commit

Run from `/Users/evanpan/AgenticSDS`. The selective staging command excludes
the unrelated historical report directory. Review the staged list before
committing. These commands are **for the user to run**, not actions performed
by this preparation step:

```sh
git add -- agent_support/common.py agent_support/runner.py \
  agent_support/autonomous_*.py agent_support/controller_backend.py \
  agent_support/gpt_oss_*.py agent_support/intervention_replay.py \
  agent_support/learned_*.py agent_support/llm_support_controller.py \
  agent_support/support_sequence_analysis.py \
  configs/autonomous_evaluation configs/autonomous_learning \
  configs/textcraft_autonomous_support_framework.json \
  configs/textcraft_prompted_gptoss_capabilities.json \
  configs/textcraft_prompted_gptoss_controller.json \
  docs/autonomous_capabilities.md docs/autonomous_support_framework.md \
  docs/learned_support_gate.md docs/prompted_support_controller.md \
  reports/autonomous_support_controller scripts/autonomous_support_eval.sbatch \
  tests/test_autonomous_capabilities.py tests/test_autonomous_evaluation.py \
  tests/test_autonomous_support.py tests/test_learned_support.py \
  tests/test_llm_support_controller.py
git diff --cached --name-only
git diff --cached --check
python3 -m pytest -q tests
git switch -c autonomous-smoke-v1
git commit -m "Prepare autonomous TextCraft one-task smoke"
git push -u origin autonomous-smoke-v1
```

Unrelated untracked local reports may keep this Mac checkout dirty after
commit. The **separate Locust worktree** must be clean for
`require_clean_commit()`.

## Locust: discovery and preflight

Start in the **existing** Locust AgenticSDS clone. Its exact path and the
partition name are not known in this checkout. Confirm the remote, then
create a separate worktree without touching earlier runs:

```sh
EXISTING_REPO="$(git rev-parse --show-toplevel)"
git remote -v
git fetch origin autonomous-smoke-v1
DEPLOY_REPO="$(dirname "$EXISTING_REPO")/AgenticSDS-autonomous-smoke-v1"
test ! -e "$DEPLOY_REPO"
test -w "$(dirname "$EXISTING_REPO")"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-smoke-v1
cd "$DEPLOY_REPO"
git status --short
git rev-parse HEAD
```

Locate the **existing** Qwen and GPT-OSS Python executables and model caches
from earlier Locust jobs. Do not assume the local Mac `.venv` or a guessed
`gptoss_env` path is present there. Set the following after discovery:

```sh
env | sort | grep -E '^(AGENT_SUPPORT|HF_HOME|VENV_PATH|CONDA_PREFIX)=' || true
ls -ld "$EXISTING_REPO/.venv" "$EXISTING_REPO/.venv/bin/python" 2>/dev/null || true
find /shared_data0/evanpan/agent_support -maxdepth 6 -path '*/bin/python*' -print
find /shared_data0/evanpan/agent_support -maxdepth 5 -type d -name hub -print
```

Then export the paths you have verified:

```sh
export VENV_PATH="<existing-Qwen-environment-directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<existing-pinned-GPT-OSS-python>"
export AGENT_SUPPORT_DATA="<writable-persistent-agent-support-root>"
export HF_HOME="<existing-Qwen-model-cache-root>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<existing-GPT-OSS-model-cache-root>"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -w "$AGENT_SUPPORT_DATA"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
command -v sbatch
sinfo -o '%P %G %m %l'
"$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation preflight --arm autonomous
"$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation preflight --arm qwen_standalone
"$VENV_PATH/bin/python" -c 'import importlib.metadata as m; print({n:m.version(n) for n in ("torch","transformers","accelerate","bitsandbytes","sentence-transformers","textcraft")})'
"$AGENT_SUPPORT_GPTOSS_PYTHON" -c 'from agent_support.gpt_oss_model import check_mxfp4_dependencies; print(check_mxfp4_dependencies("5.22.2"))'
```

The angle-bracket environment/cache values are discovery placeholders, not
verified cluster paths. Check model snapshots in the chosen cache roots and
whether the selected GPU partition actually supplies an L40S; if it contains
mixed GPUs, inspect `scontrol show nodes` for a suitable constraint before
submitting. Keep the pinned package environments separate.

## Submit only after review

Set `GPU_PARTITION` to a verified L40S-capable partition from `sinfo`. Slurm
opens its output file before the script starts, so create the new log
directory first. The script requests one GPU, six CPUs, 64 GB system RAM,
and two hours by default. It does not guarantee the GPU model if the partition
contains mixed types.

```sh
export GPU_PARTITION="<verified-L40S-partition>"
mkdir -p "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs"
EVAL_ARM=autonomous EVAL_STAGE=smoke \
  sbatch --partition="$GPU_PARTITION" \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/autonomous-smoke-%j.out" \
  scripts/autonomous_support_eval.sbatch
```

The matching Qwen-only operational smoke uses the same first task and agent
settings, without the GPT-OSS controller:

```sh
EVAL_ARM=qwen_standalone EVAL_STAGE=smoke \
  sbatch --partition="$GPU_PARTITION" \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/qwen-standalone-smoke-%j.out" \
  scripts/autonomous_support_eval.sbatch
```

No command in this document was submitted by the preparation work.

## Expected outputs and trace inspection

The job prints its exact `run_dir`. For the autonomous smoke it is under
`$AGENT_SUPPORT_DATA/autonomous_support_evaluation/runs/smoke/autonomous/textcraft_<UTC>_<ID>/`.
The standalone smoke is under the sibling `smoke/qwen_standalone/`. The
autonomous worker log is `workers/gpt_oss_controller_worker.log` inside its
run directory. GPU telemetry is a separate file at
`$AGENT_SUPPORT_DATA/autonomous_support_evaluation/telemetry/autonomous-smoke-<SlurmJobID>.csv`.

After locating the exact run path in the Slurm log, set `RUN_DIR` to that
absolute path and inspect it without altering the frozen inputs:

```sh
export RUN_DIR="<printed-autonomous-run-directory>"
"$VENV_PATH/bin/python" -c 'import sys; from agent_support.analysis import load_run; m,c,manifest,episodes=load_run(sys.argv[1]); e=next(iter(episodes.values())); print("run",m["run_id"],"tasks",len(episodes),"status",e["status"]); print("task_model",c["model_id"],c["model_revision"]); print("controller",m.get("controller_model_metadata",{}).get("model_id"),m.get("controller_model_metadata",{}).get("model_revision")); [(print("decision",r["decision"],"proposal",r["proposed_action"],"final",r["action"],"stop",r["support_trace"]["termination_reason"],"events",[(x.get("controller_decision",{}).get("kind"),x.get("request",{}).get("operation"),x.get("result",{}).get("status")) for x in r["support_trace"]["events"]])) for r in e["trace"]]' "$RUN_DIR"
PYTHONHASHSEED=0 "$VENV_PATH/bin/python" -m agent_support.intervention_replay --run "$RUN_DIR"
```

Inspect `run.json`, `config.json`, `summary.json`, the episode JSON under
`episodes/`, the worker log, and the telemetry CSV. A complete support trace
has a state and controller decision for each executed controller turn, each
support request and structured result in order, controller/support costs,
the termination reason, final action, and environment outcome. Choosing no
support is a valid behavior. Confirm `run.json` is `complete`; a failed or
interrupted run cannot be treated as a smoke pass.
