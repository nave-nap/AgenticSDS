# M2 action-conditioned memory protocol

## Frozen development baseline

The reviewed no-support pilots are recorded in [`baseline_pilot.json`](baseline_pilot.json).
ALFWorld run `f95c88b5` solved 2/20 development tasks; TextCraft run `39ab9386`
solved 4/20. Both used commit `0368e003da216b958cf8cc4350e26f3f7ac57067`,
Qwen2.5-7B-Instruct with NF4 quantization, 30 environment actions and 30 model
decisions. Their summaries and hashes are frozen in that record. The exact
ALFWorld development manifest is now in `manifests/alfworld/development.json`;
the TextCraft development manifest was already committed. Keep the full raw
pilot run directories on Drive. Do not use either pilot's trajectories as memory.

## Training-only corpus

Use [`notebooks/m2_launcher.ipynb`](../notebooks/m2_launcher.ipynb) on a Colab
GPU. It creates 40 ALFWorld training games from the benchmark's `train` split
and 40 TextCraft tasks from seeds beginning at 1000. The training manifest
generator rejects task IDs and task-content hashes found in the development
or held-out manifests. The corpus builder checks that boundary again and
accepts only complete baseline runs whose manifest partition is `training`.
The M2 runner also requires proof that its evaluation manifest was excluded
when the corpus was built.

The corpus stores one record per distinct environment transition: task, observation,
action, feedback, and whether the action failed or the episode succeeded.
It does not store parse-only decisions. An index embeds a compact task,
observation, and proposed-action description with
`sentence-transformers/all-MiniLM-L6-v2`, mean pooling, and cosine similarity.
The index and corpus are saved separately under Drive's
`agent_support/memory/<benchmark>/` and are linked by a content hash.

## M2 behavior

The first model call uses the unchanged baseline prompt and proposes one
action. M2 attempts retrieval at every valid proposal. It considers only
actions accepted by the environment within ultimately successful training
episodes. TextCraft examples must have the same command type and target item;
ALFWorld examples must have the same command verb and share an object type
(ignoring instance numbers). MiniLM ranks matching examples, and up to three
are shown to the model for one extra action-choice call. If there is no match,
the original proposal executes without a second model call. If the second
response does not parse, the original proposal executes and the support parse
error is logged. The environment action budget and original action parser
stay the same. Model decisions count the original action choice; `model_calls`
also counts any extra support call. `run.json` records the memory-selection
policy version.

This filter was added after the first three-task smoke runs showed that most
retrieved TextCraft examples were failed actions and frequently displaced
workable proposals. The corpus and index remain unchanged; this policy uses
the outcome labels already stored in the training-only corpus. The 40-task
training corpus has limited positive coverage, so support may activate rarely.

Each supported trace includes the original proposal, memory IDs, similarity
scores, displayed content, retrieval time, second response, extra tokens and
inference time, and whether the action changed. The full run snapshots the
configuration and memory artifact hashes. Compare supported and baseline
runs only on the same development task manifest, model settings, benchmark
versions, and action budget. If a shared baseline behavior bug is fixed,
rerun the baseline before comparison.

The paired comparison command checks the exact full development manifest,
baseline agent settings, Python and benchmark versions, and source benchmark
configuration hashes. It records both Git commits because the M2 runner was
added after the frozen baseline. It reports helped and hurt tasks, action and
token changes, extra model calls, and support hit rate:

```bash
python3 -m agent_support.analysis BASELINE_RUN_DIR M2_RUN_DIR \
  --support-comparison --output comparison.json
```

The baseline-to-baseline command without `--support-comparison` remains strict
about identical configurations and Git commits.

## Execution order

1. Publish this code and open `m2_launcher.ipynb` in a fresh Colab GPU runtime.
2. Run its setup and training-manifest cells. Confirm each training manifest
   has 40 tasks and the development and held-out exclusions are present.
3. Enable `RUN_TRAINING`, collect both baseline training runs, and confirm
   `run.json` says `complete` with 40 tasks each.
4. Enable `BUILD_MEMORY`, build and inspect both corpus and index files.
5. Enable `RUN_M2_SMOKE`, run three development tasks per benchmark, and
   inspect the full traces before a 20-task supported pilot.
6. Enable `RUN_M2_PILOT` with training, memory building, and smoke flags off.
   Run all 20 development tasks for both benchmarks on the frozen manifest.
7. Enable `RUN_M2_COMPARE` to write paired comparison JSON files to Drive.
   Inspect trajectories before drawing conclusions from the aggregate scores.

The local repository cannot perform the training and M2 runs without a CUDA GPU and the
benchmark installations. The notebook leaves all run flags off by default.
