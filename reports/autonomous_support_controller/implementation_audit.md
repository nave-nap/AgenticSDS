# AgenticSDS autonomous support: implementation audit

Audited 2026-11-18 from the local checkout at `/Users/evanpan/AgenticSDS`.
This is an implementation and artifact audit, not a new experiment. No source,
configuration, historical-result, or model file was changed for this audit.
The Locust filesystem and scheduler are not mounted here.

## A. Executive assessment

An opt-in sequential support path exists: Qwen proposes an action, a
replaceable controller may request several registered supports, and the runner
validates one final action before stepping TextCraft or ALFWorld. The prompted
GPT-OSS-20B controller has a strict JSON decision contract and can receive
earlier support results, but every demonstrated choice so far came from a fake
or scripted model. Four capabilities are configured for the TextCraft research
arm; frozen M2-200 corpus/index structure and digests pass local CPU preflight.
The history and state inspectors are visible-data summaries, not semantic
proofs of task progress or prerequisites. A CPU ridge gate and trainer exist,
but no real paired intervention dataset, fitted gate artifact, or learned-policy
evaluation exists. No new autonomous-controller real-model smoke or full
benchmark run was found in the checkout or accessible local run stores.
Historical GPT-OSS, M2-11, and delegation results are real but cannot establish
the new controller's effect. Locust deployment is unverified; the checkout is
dirty and the cluster's model environments, artifact mount paths, GPU memory,
and output permissions need to be checked before the first smoke.

## B. Implementation status table

| Component | Status | Evidence | Remaining work |
| --- | --- | --- | --- |
| Sequential interfaces and controller replacement | Implemented, CPU tested | `agent_support/autonomous_support.py`: `SupportContext`, `SupportRequest`, `SupportResult`, `SupportController`, `SupportExecutor`, `make_support_controller` | Real-model behavior untested |
| Multi-support loop, result feedback, explicit execute | Implemented, CPU tested | `run_support_loop` updates `current.results` after each operation; `SupportDecision("execute")`; repeated and mixed-operation tests in `tests/test_autonomous_support.py` | Inspect actual GPT-OSS sequences |
| Limits and safe fallback | Implemented, CPU tested | `SupportBudget` caps operations per decision/task, additional model calls/tokens, and depth; `run_support_loop` handles malformed requests, failures, and fallback; `autonomous_runner.run_autonomous_episode` reparses final action | Measure real prompt/token pressure and fallback rate |
| Per-decision trace and cost accounting | Implemented with limits | `SupportTrace.events`, controller/support costs, before/after budgets, final environment outcome in `agent_support/autonomous_runner.py`; `agent_support/support_sequence_analysis.py` | Worker-error costs can be estimated; embedding calls have no tokenizer-token count; validate real trace completeness |
| Prompted autonomous controller | Implemented, unverified with real model | `agent_support/llm_support_controller.py`: visible context plus proposed action/capability catalog/budgets; strict `execute` or one `request` JSON; fresh decision after each result | One real GPT-OSS smoke; measure malformed output and overhead |
| Model backends | Code present, Locust unverified | Qwen NF4/greedy `agent_support/model.py`; separate persistent GPT-OSS MXFP4 worker `agent_support/controller_backend.py`, `gpt_oss_controller_worker.py`, `gpt_oss_model.py`; one worker shared by controller and reasoning support | Verify both pinned environments/checkpoints coexist on assigned GPU |
| Episodic memory | Adapter implemented; frozen files CPU validated | `FrozenTrajectoryRetrieval.search` reuses `TrajectoryMemory` and MPNet; `RetrieveEpisodicMemory` accepts controller-written query/scope/top-k, returns similarities and provenance; local 200-trajectory artifacts pass preflight | Verify actual MPNet embedding and retrieval inside Locust smoke; configure Locust paths |
| History and failure inspection | Functional but heuristic | `InspectHistory` returns recent/failure/repeated action-feedback pairs using visible feedback text | It does not infer failure causes or prove progress |
| Additional reasoning / plan revision | Interface implemented, fake-model tested | `ReasonOverArtifacts` accepts controller-written objective and `note`, `plan`, or `candidate_action`; uses shared GPT-OSS backend | Real model output and action quality untested |
| State/prerequisite inspection | Visible-state summary only | `InspectState` reads task prompt and observed history, never `episode.env`; returns displayed recipes, prior inventory feedback, craft feedback | Cannot verify current hidden inventory or actual prerequisites; avoid claiming it can |
| Evaluation pipeline | Implemented, CPU preflight tested | Six arms in `agent_support/autonomous_evaluation.py` and `configs/autonomous_evaluation/`; staged smoke/pilot/full gates; comparison and sequence scripts; `scripts/autonomous_support_eval.sbatch` | No submitted new-arm jobs or reviewed gates; resource/paths unverified on Locust |
| Learning baseline | Code implemented, research evaluation incomplete | `agent_support/learned_support_gate.py` cost-sensitive 13-feature ridge gate; `learned_support_training.py`; `learned_support_compare.py`; synthetic tests | No real branch collector, intervention labels, fitted artifact, or evaluation; learned arm absent from staged six-arm CLI |
| Counterfactual replay | Environment-prefix helper implemented; policy counterfactuals absent | `agent_support/intervention_replay.py` replays exact logged prefix and branches first actions; historical TextCraft replay checks documented in `learned_support_methodology.md` | Need audited same-state support-choice branch rollouts, controlled seeds, and terminal outcomes |

The production controller is **not** a hardcoded memory→reasoning→verification
pipeline. Its prompt requests one valid operation or execute at each step.
Tests with predetermined operation sequences demonstrate the framework, not
autonomous model behavior. Four operations are enabled in
`configs/autonomous_evaluation/autonomous.json`: memory retrieval, history
inspection, reasoning, and visible state inspection. `InspectCandidate` exists
in the operation registry but is not enabled by that evaluation arm.

## C. Testing and experiments inventory

| Category | Verified execution | Model / environment | Tasks and result | Artifact or limitation |
| --- | --- | --- | --- | --- |
| 1. CPU unit tests | `python3 -m pytest -q tests` passed **98/98** during this audit | Fake/scripted task and controller models; mostly fake episodes and operations | Tests of no-support, one/multiple/repeated supports, budgets, fallback, strict parsing, logging, worker JSON protocol, learned synthetic fit | `tests/test_autonomous_support.py`, `test_llm_support_controller.py`, `test_autonomous_capabilities.py`, `test_autonomous_evaluation.py`, `test_learned_support.py`; no real LLM inference |
| 2. CPU integration/preflight | All six `agent_support.autonomous_evaluation preflight --arm ...` commands passed during this audit | Real local frozen M2-200 JSON, manifest, structural index validation; fake embedder and fake models in tests | 20-task manifest digest `4ec57f…fa4900e`; 200 corpus records; no task-solving run | Preflight checks structure/digests/exclusions, not MPNet inference or GPU loading. `test_run_wires_frozen_artifacts_into_controller_without_gpu` uses fake episode/task/controller/embedding models |
| 2. Historical TextCraft replay audit | Previously recorded CPU audit in `learned_support_methodology.md`: standalone 298/298, reconsideration 280/280, M2-11 297/297 action outcomes matched with `PYTHONHASHSEED=0` | Real TextCraft 0.0.3 environment, recorded actions; **no model regeneration** | 875 environment transitions; supports reproducible environment prefixes on this installation | Does not establish GPT-OSS/Qwen sampling reproducibility or support causality |
| 3. New real-model smoke | **None verified** | Intended Qwen2.5-7B proposer + GPT-OSS-20B controller, TextCraft | 0 new tasks | No new `autonomous_support_v1` completed run found in repository, accessible `agent_support_data`, `/private/tmp/gptoss_m2_artifacts`, or local `agent_research` inventory |
| 4. New full benchmark | **None verified** | Intended same two models, frozen 20-task TextCraft development set | 0 new tasks | Configs, Slurm script, mock traces, and report templates are preparation, not execution |
| Historical completed runs, separate | Rechecked with `analysis.load_run`: `slurm_152866` standalone, `slurm_153283` matched reconsideration, `slurm_153974` M2-11 | Actual GPT-OSS task agent, TextCraft | Each 20 tasks, each 14 successes | `/private/tmp/gptoss_m2_artifacts/outputs/{gptoss_standalone,gptoss_matched_reconsideration,gptoss_m2_reconstructed11}/textcraft_slurm_<ID>`; different task agent/memory from new cohort |
| Earlier delegation runs, separate | `reports/complete_research_summary/consolidated_results.json` inventories 22 historical runs, including the above three | Qwen/GPT-OSS, TextCraft and ALFWorld, assorted conditions | Historical report-level results; not new autonomous arm | See `reports/complete_research_summary/source_inventory.json`; do not use as matched new-controller controls |

The user reports a completed M2-200 GPT-OSS evaluation on Locust. Its run
directory, episode traces, and summary were not available in this local audit,
so that **evaluation** remains externally reported rather than locally
verified. Its **frozen corpus/index** are locally present at
`~/agent_support_data/m2_200/`, and the new configs' pinned digests passed
preflight. Files named `example_traces_mock.json` and
`capability_traces_mock.json` are synthetic examples. No real learned-policy
artifact or intervention dataset was found.

## D. Critical blockers and other risks

### Blocking before the first new real-model smoke

1. **Clean, deployed code is required.** `agent_support/autonomous_evaluation.py:require_clean_commit` rejects the current checkout: many new source/config/report files are untracked and `common.py`/`runner.py` are modified. This was confirmed by invoking that function. A reviewed commit deployed to Locust is required for its staged runner. Do not mistake the current `git rev-parse HEAD` value for the implementation's provenance.
2. **Locust artifact resolution is unverified.** `configs/autonomous_evaluation/autonomous.json` and `qwen_m2_200.json` point to `~/agent_support_data/m2_200/…`. The user-supplied Locust files are instead `/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts/{corpus_200,index_200}.json`. `AGENT_SUPPORT_DATA` controls output/cache in the Slurm script; it does **not** rewrite config memory paths. Arrange an explicitly reviewed config/path mapping on Locust, preserve the pinned digests, and rerun preflight there. `/shared_data0` is absent on this Mac.
3. **Pinned model runtimes and checkpoint caches are unverified on Locust.** The job needs a Qwen environment (`requirements.txt` pins Transformers 4.57.6, bitsandbytes 0.50.2, TextCraft 0.0.3) and a separate GPT-OSS worker environment (`gpt_oss_model.check_mxfp4_dependencies` requires Transformers 5.22.2, kernels 0.17.2, Triton ≥3.4). `AGENT_SUPPORT_GPTOSS_PYTHON` and `AGENT_SUPPORT_GPTOSS_HF_HOME` must be set. Local `.venv` has TextCraft but lacks Transformers, torch, bitsandbytes, sentence-transformers, accelerate, kernels, and Triton; local GPT-OSS executable/cache and Qwen cache were not found. This says nothing conclusive about Locust.
4. **Single-GPU co-residency is unmeasured.** `scripts/autonomous_support_eval.sbatch` requests one GPU and 64 GB system RAM. `runner.run` loads Qwen, then starts a persistent GPT-OSS worker, while MPNet runs on CPU. Historical standalone GPT-OSS recorded roughly 20–21 GB reserved GPU memory; Qwen adds memory. Confirm the assigned GPU's capacity and successful dual-model startup during smoke. Do not assume a one-GPU request suffices on every Locust partition.

### Important before the full 20-task evaluation

- Verify Slurm partition/time limits, checkpoint caches, `AGENT_SUPPORT_DATA` write access, and a pre-existing Slurm log directory on Locust. The script uses `umask 077` and creates run/cache directories, but Slurm opens `--output` before the script runs.
- Review actual controller JSON validity, parser errors, fallback frequency, empty retrievals, budget terminations, and calls/tokens/wall time in one-task and five-task traces. The prompted controller can add a model call at every environment decision plus further support calls; caps prevent unbounded loops but may still be expensive.
- Support decisions/results and their order are logged, but `llm_support_controller.py` and `autonomous_operations.py` keep only the first 4,096 characters of each model's parsed text in event metadata; the worker's full Harmony completion is not persisted in the support trace. Assess whether this is sufficient for failure forensics before the full run.
- Match contemporaneous **Qwen** standalone, reconsideration, M2-200, random, and fixed-reasoning controls. The historical GPT-OSS/M2-11 14/20 results have a different task agent and memory scale; `autonomous_evaluation_compare.py` correctly treats them as descriptive references.
- Check resumed-job expectations: `runner.run` atomically writes each completed episode and summary and marks the run failed on exception, but creates a new UUID directory on every invocation and has no in-episode or automatic run resume. A preempted smoke/pilot must be treated as a distinct attempt.
- Preserve provenance of any cluster-specific path mapping and verify the exact model revisions, package versions, manifest digest, recipe-order snapshot, and memory digests in each run. The six-arm full-run gate requires a reviewed smoke and pilot for every arm on matching code/runtime.

### Later, for the learned research direction

- `learned_support_training.train_gate` consumes asserted paired records; it does not collect them. `intervention_replay.branch_first_actions` checks first actions after an identical environment prefix but does not run alternative support decisions and downstream terminal rollouts. An audited collector and branch traces are needed before any intervention-effect labels can be trusted.
- The ridge gate learns only whether to continue consulting the prompted controller. It does not learn operation choice or argument construction. `configs/autonomous_learning/learned_gate.template.json` has a zero artifact digest and no fitted file. `agent_support.autonomous_evaluation.ARMS` excludes `learned_gate_v1`, so the existing six-arm staged CLI cannot run it directly.
- `InspectState` does not query hidden inventory or prove prerequisites. `InspectHistory` uses simple failure-text matching; accepted actions, action replacements, retrieval similarity, and support activity are not measures of goal progress.
- The current comparison is exploratory on a previously used 20-task development set. A held-out result and/or controlled same-state interventions are needed for generalization or causal claims.

## E. Next five actions

1. **Review and commit the current implementation/protocol as a clean snapshot.** Files: all new `agent_support/autonomous_*`, controller/backend/operation modules, six configs, tests, Slurm script, and this audit. Verify that only intended files are staged, `python3 -m pytest -q tests` passes, and `require_clean_commit()` succeeds. GPU: no. Depends on nothing.
2. **Audit Locust prerequisites and resolve artifact paths without modifying frozen artifacts.** Files: `scripts/autonomous_support_eval.sbatch`, `configs/autonomous_evaluation/{autonomous,qwen_m2_200,random_support}.json`, pinned environments and model caches. Verify corpus/index readable with matching digests, Qwen and GPT-OSS executables/dependencies/revisions available, GPU memory and partition limits, writable output/log roots. GPU: no for metadata checks; a GPU is needed only to verify loading. Depends on 1 for deployment.
3. **Run one-task real-model smoke for `autonomous`.** Use `agent_support.autonomous_evaluation run --arm autonomous --stage smoke` through the reviewed job script once step 2 passes. Verify both models load, one valid TextCraft action is stepped, per-decision support decisions/results are logged, costs and fallback reasons are coherent, and the run is `complete`. A zero-support choice is valid. GPU: yes. Depends on 1–2.
4. **Inspect smoke trace and run five-task pilot.** Files: episode JSON, `run.json`, `summary.json`, worker log, `support_sequence_analysis.py`; then `--stage pilot --smoke-run <actual path>`. Verify support composition/no-support distribution, parser and generation failures, retrieval provenance, budget stops, task outcomes, and measured wall/GPU costs. GPU: yes for pilot. Depends on 3.
5. **Complete matched development controls and reviewed full-run gates.** Run the contemporaneous five controls through smoke/pilot, inspect all six arm traces, write their gates, then run the frozen 20-task cohort and analyze with `autonomous_evaluation_compare.py`. Verify identical manifest, task agent, revisions, runtime, seed, action limits, memory digests where applicable, and report success beside realized cost. GPU: yes. Depends on 4 and each control's smoke/pilot. Only after this should paired intervention-data collection for learning become a priority.

## F. Recommended first real experiment

Start with the frozen first TextCraft development task, Qwen proposer, and
`autonomous` prompted GPT-OSS controller using
`configs/autonomous_evaluation/autonomous.json`. Keep the four registered
capabilities and budgets as configured; do not tune on the smoke outcome. Use
the matching `qwen_standalone` one-task smoke as an operational control, not a
statistical comparison. Record run/worker metadata, the raw proposal,
controller JSON outputs, support operations and results, final parsed action,
environment feedback, calls/tokens by role, retrieval time, total time, and
termination reason. Pass criteria are a complete one-task run, validated
final action or documented safe parse fallback, no unaccounted model calls,
and inspectable support trace. A multi-support sequence is **not** required
for this first task; the model must be free to choose none. A meaningful
performance comparison begins only with matched multi-task controls and the
full frozen 20-task cohort. Earlier GPT-OSS/M2-11 runs are reference context.

## G. Exact commands

Commands below are read-only validation or discovery. No GPU job is submitted
by this audit. Run Mac commands from the local checkout:

```sh
cd /Users/evanpan/AgenticSDS
git status --short
python3 -m pytest -q tests
for arm in qwen_standalone qwen_reconsideration qwen_m2_200 random_support autonomous fixed_reasoning_2; do
  python3 -m agent_support.autonomous_evaluation preflight --arm "$arm"
done
python3 -c 'from agent_support.autonomous_evaluation import require_clean_commit; print(require_clean_commit())'
```

The final command currently fails with `GPU evaluation requires a clean
checkout`; it should pass only after review/commit. To inspect one preserved
historical run without treating it as a new controller experiment:

```sh
PYTHONHASHSEED=0 .venv/bin/python -m agent_support.intervention_replay \
  --run /private/tmp/gptoss_m2_artifacts/outputs/gptoss_standalone/textcraft_slurm_152866
```

On Locust, first enter the actual clone and discover resources rather than
assuming a home-directory checkout or GPU partition. The artifact paths below
are user supplied and were **not** accessible from this Mac:

```sh
pwd
git rev-parse --show-toplevel
git status --short
command -v sbatch
command -v sinfo
sinfo -o '%P %G %m %l'
test -r /shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts/corpus_200.json
test -r /shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts/index_200.json
test -x .venv/bin/python
printf 'GPT-OSS Python: %s\n' "${AGENT_SUPPORT_GPTOSS_PYTHON:-UNSET}"
printf 'Output root: %s\n' "${AGENT_SUPPORT_DATA:-UNSET}"
.venv/bin/python -m agent_support.autonomous_evaluation preflight --arm autonomous
```

The final preflight uses the current config's `~/agent_support_data/m2_200/`
paths and may fail until those paths are deliberately mapped to the shared
Locust artifacts. Inspect `scripts/autonomous_support_eval.sbatch` and the
protocol's submission command only after the above checks pass. The exact
Locust checkout, environment executables, cache root, output root, partition,
and log path are unknown here, so no invented `sbatch` command is supplied.

## H. Final status

- Framework implementation: **complete** for the opt-in CPU-tested path
- Autonomous controller: **complete** as code, real behavior unverified
- Support capabilities: **partial** (four functional interfaces; memory inference and reasoning unverified with real models; prerequisite inspection is limited)
- Real-model integration: **unverified**
- Locust deployment: **unverified**, with known local clean-checkout and cluster-path prerequisites
- Autonomous-controller benchmark results: **not completed** in accessible evidence
- Learned controller: **implemented baseline, untrained and unevaluated**
- **Recommended immediate next action:** review/commit the current code and protocol, then audit Locust paths, model environments, and GPU capacity before a one-task smoke.
