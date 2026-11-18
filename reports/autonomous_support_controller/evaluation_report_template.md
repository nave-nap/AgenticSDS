# TextCraft autonomous support evaluation report

**Protocol version / code commit:** [fill]
**Run date, cluster/GPU, Python and package versions:** [fill]
**Frozen development manifest digest:** `4ec57f36416ffe7f97b004b67d107e94750fa20b889d95f580690f4dfca4900e`
**M2-200 corpus/index digests:** [fill from run metadata]
**Run directories, config digests, seeds, budget settings, worker metadata:** [table]
**Smoke and pilot gate files reviewed:** [paths and findings]

## Prespecified hypotheses

1. Autonomous support will improve paired task success over the matched Qwen
   standalone and simple random selector at comparable realized compute.
2. Autonomous support will use fewer support operations than always-on M2
   while preserving or improving task success.
3. A fixed two-call reasoning control will reveal whether extra GPT-OSS
   inference alone explains an observed gain. Its calls, tokens, and time
   must be shown beside success; the schedule is only approximately matched.

These hypotheses are exploratory on a repeatedly inspected 20-task
development set. Record any deviations before viewing full-run outcomes.

## Run integrity

| Arm | Run ID | Complete tasks | Commit | Model revision | Config digest | Memory digest | Seed | Gate | Integrity issues |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- |
| Qwen standalone | | | | | | n/a | | | |
| Qwen reconsideration | | | | | | n/a | | | |
| Qwen M2-200 | | | | | | | | | |
| Random support | | | | | | | | | |
| Autonomous | | | | | | | | | |
| Fixed reasoning 2 | | | | | | n/a | | | |

Verify identical ordered task IDs and observations, Qwen task-agent settings,
30-action/decision limits, recipe-order snapshot, code and runtime versions.
List failed or preempted jobs separately; do not silently replace them.

## Task success and compute

| Arm | Success / 20 (Wilson 95% interval) | Env actions / failures | Total model calls | GPT-OSS calls | Controller calls / tokens | Support operations | Input / output tokens | Inference s | Controller / support wall s | Parser / generation failures | Budget stops |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- | ---: |
| Qwen standalone | | | | | | | | | | | |
| Qwen reconsideration | | | | | | | | | | | |
| Qwen M2-200 | | | | | | | | | | | |
| Random support | | | | | | | | | | | |
| Autonomous | | | | | | | | | | | |
| Fixed reasoning 2 | | | | | | | | | | | |

Report paired helped/hurt task IDs, success-rate differences, exact
discordance p-values, Holm-adjusted values across autonomous contrasts, and
paired bootstrap intervals. Show per-task model calls and tokens with each
outcome. Compare success against **realized** calls, tokens, and time; equal
limits do not imply equal compute.

## Controller procedure analysis

- No-support decisions / all decisions: [fill]
- Ordered sequence counts and lengths: [fill]
- Unique operation combinations and repeated operations: [fill]
- Capability counts, empty/error results, and retrieval provenance: [fill]
- Decisions composing two or more capabilities: [fill]
- Proposed-action replacements and final-action validity: [fill]
- Sequence frequencies by successful versus failed task: [fill]
- Task support-count and success association: [fill, descriptive only]
- Controller and support cost distribution by task: [fill]
- Budget exhaustion, malformed controller output, support errors: [fill]

Use `support_sequence_analysis.py` output and inspect raw per-decision traces
for surprising sequences. Include representative **successful and failed**
episodes, not only favorable examples. State whether retrieved training
memories were relevant to the current visible state; similarity alone is not
a relevance label.

## Historical context, separated from the matched cohort

Preserved GPT-OSS standalone, matched reconsideration, and M2-11 runs each
solved 14/20 on this manifest at seed 42, with 303, 562, and 598 model calls.
Record their exact run paths and provenance if included. They use a different
task agent and memory size from the primary cohort and are not pooled into
its paired comparisons.

## Interpretation and limitations

Answer whether any autonomous gain persists relative to the random and fixed
reasoning controls **after inspecting actual computation**. Identify task
swaps and costs. Do not attribute a task outcome to one support call from
observational trace correlation. The current runner lacks verified
counterfactual replay of the same state under alternate support procedures;
therefore per-call causal effects cannot be estimated. Note prior exposure
to this development set, single-seed sampling, GPU/preemption differences,
memory quality limits, and any protocol deviations. Define the next frozen
held-out test only after this development analysis is finalized.
