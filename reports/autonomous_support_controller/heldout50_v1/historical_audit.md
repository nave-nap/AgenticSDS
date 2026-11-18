# TextCraft historical baseline audit before the 50-task evaluation

## Source and integrity

I read the 21 run directories in the user-supplied, read-only archive at
`/Users/evanpan/Downloads/historical_baseline_audit/`. All **21 runs have
complete task-level results and trajectories**, totaling **396 episode JSON
files**. For every run, `run.json` says `complete`, the config and selected
manifest hashes match their snapshots, task IDs and initial-observation
digests match the saved manifest, and `summary.json` success totals reconcile
with episodes. Eighteen runs include checkpoints; every recorded episode hash
matches. The Qwen held-out M2, autonomous v3 smoke, and three-task replanning
pilot have no checkpoint file, so their episode content cannot be checked
against an independent checkpoint. This audit did not change archive files.

[`historical_run_inventory.csv`](historical_run_inventory.csv) records all 21
source paths, model revisions, configs, task counts, hashes, support settings,
compute totals, and classification. [`historical_task_results.csv`](historical_task_results.csv)
records all 396 per-task outcomes, action counts, tokens, and observation
digests. The archive contains:

| Group | Runs | Coverage | Relevant finding |
| --- | ---: | --- | --- |
| Qwen reconsideration-policy experiment | 4 | 20 development tasks each | Standalone 4/20, always reconsider 4/20, random reconsider 5/20, failure reconsider 3/20; commit `2cc1a41b`, same Qwen revision as v3 |
| Early modular policy pilots | 4 | First 3 development tasks each | Qwen standalone/random/always/failure reconsideration; pilot only, different commits |
| Corrected Qwen/GPT-OSS delegation set | 4 | 20 development tasks each | Qwen standalone 4/20, random delegation 9/20, failure delegation 8/20, GPT-OSS standalone 14/20; commit `4ec819ce` |
| Delegation budget sweep | 4 | 20 development tasks each | Random caps 1/10: 6/20 and 10/20; failure caps 1/10: 6/20 and 9/20 |
| Additional Qwen standalone | 1 | 20 development tasks | 4/20 at commit `09b1d86a` |
| Qwen dynamic-replanning pilot | 1 | First 3 development tasks | 1/3; 120 action/decision caps, so not a matched control |
| **Qwen legacy M2 trajectory memory** | **1** | **All 100 held-out tasks** | **14/100**, commit `aba4da79`; 30 action/decision caps and Qwen revision match v3; uses an older corpus/index, with hashes `d6e2e065…`/`6f5eabf7…` |
| **GPT-OSS M2-200** | **1** | **20 development tasks** | **15/20**, commit `5ec1518e`; GPT-OSS is the task agent, not Qwen; corpus/index hashes `96a06b5e…`/`0571341c…` |
| Autonomous v3 smoke | 1 | Development seed 51 | 1/1, commit `e62b110`; 14 actions, 23 controller calls; excluded from primary evaluation |

All 20-task development runs use manifest SHA-256 `4ec57f36…` and recipe-order
SHA-256 `5d3c08b0…`. The held-out Qwen M2 run uses the frozen 100-task
manifest SHA-256 `5c8dbc6c…`, the same recipe-order digest, TextCraft 0.0.3,
Qwen revision `a09a3545…`, NF4 quantization, 100 output tokens, history window
8, generation seed 42, and 30 action/decision limits. Its runtime reports
Torch `2.11.0+cu128` and Transformers `4.57.6`, matching the v3 smoke's Qwen
environment. Its `agent_support/policy.py`, `model.py`, and `benchmarks.py`
match the current files at source commit `aba4da79`; the runner has changed to
add autonomous support. These checks support a **secondary paired outcome
comparison** on identical held-out tasks. They do not make its older M2 corpus
equivalent to v3's frozen M2-200 memory or prove identical sampled
trajectories. The old Qwen corpus/index files and their provenance are absent
from the archive, so their content/exclusions cannot be independently audited.

The GPT-OSS M2-200 run is a distinct experiment: GPT-OSS-20B is both the
primary action model and the reconsideration model; its 20 development tasks
do not cover the proposed held-out50 set. Its 15/20 result is not a Qwen
M2-200 baseline. The earlier GPT-OSS M2-11 run in another local archive is
also GPT-OSS-primary and development-only.

## Qwen reconsideration and missing evidence

The archive **does contain a complete 20-task development Qwen reconsideration
run** (`textcraft_slurm_142830`, 4/20) and short pilots. It **does not contain
a held-out Qwen reconsideration run**. Existing notes report 12/100 held-out
reconsideration and 6/20 development in a different historical series, but
the corresponding `run.json`, `config.json`, `manifest.json`, `summary.json`,
`episodes/*.json`, and checkpoints were not in these 21 directories. Do not
substitute the 4/20 run for those claims or fill held-out cells from aggregate
numbers. A complete held-out Qwen standalone run and held-out random/failure
delegation or fixed-reasoning runs are likewise absent. The archive contains
no Qwen-primary M2-200 run. Locust should be searched for these exact run
directories before scheduling baseline GPU work.

## Frozen task selection and coverage matrix

The frozen [`heldout50_v1.json`](../../../manifests/textcraft/heldout50_v1.json)
selects the **first 50 eligible tasks in the existing seed-42 ordered
held-out 100-task manifest**, excluding seed 51. This rule was chosen before
opening the newly supplied archive and the user approved it. The parent
manifest digest is `5c8dbc6caa2db221cc6ed4ce3208d9c53157af515563895a927957c3ac8349a0`;
the 50-task manifest digest is
`614de2034ad3fa14063a9fb5d6013a6754212584020b3e994a7cdb65f5b84c88`.
Seed 51 is
absent. The 50 tasks have unique IDs and observation digests. The local frozen
M2-200 corpus excludes the held-out 100-task parent manifest and has zero ID
or observation-fingerprint overlap with those 100 tasks. Overlap with other
unavailable training sources remains uncertain. A separate per-run evaluation
lock records the Locust runtime and historical sources.

[`baseline_coverage_matrix.csv`](baseline_coverage_matrix.csv) lists every
selected task and its verified coverage. The Qwen legacy M2 run has **50/50
complete task-level results** on this proposal (7/50 successes). No other run
in the supplied archive covers any of these 50 tasks. This 7/50 count was
computed only after the outcome-independent first-50 rule; it did not affect
selection. The earlier 20-task development results and seed-51 smoke are
separate, not missing-cell substitutes.

## Additional Locust path leads

The 21-run download is not proof that Locust has no other completed runs.
The historical held-out workflow writes full TextCraft results under
`/home/evanpan/agent_support_data/results/textcraft_*/`, with its lock at
`/home/evanpan/agent_support_data/heldout_evaluation/lock.json`.
Its documented three-condition plan explicitly included a Qwen standalone
held-out run. The downloaded archive included the M2 result at
`/home/evanpan/agent_support_data/replacement_evaluation/m2_runs/textcraft_20261110T052311Z_16f5bf04`
but did **not** include the corresponding standalone run. Search those
`results/` and `heldout_evaluation/` locations on Locust before scheduling
Qwen. The archive also identifies `/home/evanpan/agent_support_data/standalone_models/`
and `/home/evanpan/agent_support_data/hybrid_delegation/`, but the copied runs
there cover development tasks. No exact held-out standalone run path has been
verified locally. A candidate needs identical IDs and observations, Qwen
revision, NF4/greedy/100-token settings, eight-step history, 30/30 limits,
seed 42, TextCraft 0.0.3, recipe-order hash, complete episode snapshots,
and a reviewed code commit/prompt/parser comparison. The new evaluator checks
the stored protocol and requires explicit review of a different old commit.

| Baseline for proposed 50 | Verified coverage | Reuse class | Next action |
| --- | ---: | --- | --- |
| Qwen standalone | 0/50 | Not available for paired primary comparison | Locate held-out run; otherwise run missing tasks |
| Qwen reconsideration | 0/50 | Not available for paired comparison | Locate reported held-out100 run and audit config/episodes |
| Qwen legacy M2 trajectory memory | 50/50 | **Partially comparable**: same tasks/task agent/runtime, different memory corpus and fixed policy | Reuse as labeled secondary historical M2 comparison; obtain corpus/index provenance if possible |
| Qwen M2-200 | 0/50 | Not available | Run only if this exact control is needed; do not relabel legacy M2 |
| Random/failure GPT-OSS delegation | 0/50 | Development-only contextual results | Do not rerun by default; shared-subset run only if justified |
| Fixed GPT-OSS reasoning | 0/50 | Not located | Consider later compute-matched control on a preregistered subset |
| GPT-OSS M2-200 | 0/50 | **Not comparable as Qwen control**: different task agent and tasks | Keep as development context |
| Autonomous v3 | 0/50 | Seed-51 smoke only | Main 50-task run after protocol review |

**Minimum new GPU work, conditional on Locust discovery:** autonomous v3 on
the 50 proposed tasks, plus Qwen standalone only for held-out tasks lacking a
verified compatible prior outcome. Under the supplied archive alone, that is
50 autonomous tasks and 50 Qwen standalone tasks. The legacy Qwen M2 results
need no rerun for a clearly labeled secondary comparison. The reported
held-out reconsideration run may eliminate another prospective control run if
its complete files pass the same task/protocol audit. Do not schedule new
random, failure, or fixed-reasoning conditions merely to fill a table.

The v3 smoke episode took 313.5 seconds after model initialization, while the
whole run spanned roughly 14 minutes; one task is insufficient for a walltime
distribution. Begin any later batched run with at most five tasks, inspect
actual Slurm elapsed and the longest episode, then choose whether ten-task
batches fit the partition limit. This is a planning recommendation, not a GPU
submission. The research report should state paired sample sizes, task-level
success disagreements, action/cost differences, support sequences and parser
failures, and uncertainty; action revisions alone do not establish benefit.
