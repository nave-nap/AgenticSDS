# Agent support research

This repository starts with a reproducible **no-support baseline** for ALFWorld and TextCraft and adds support conditions for paired evaluation. The exploratory notebooks remain in `Support Testing/` as source material. The Python package is the experiment implementation; the Colab notebooks install dependencies and invoke it.

## Baseline decisions

- Model: `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4, deterministic generation.
- ALFWorld: numbered admissible-command policy from the original notebook's first complete trajectory. The later ReAct and cooling-demo experiments are separate policy variants and are not mixed into this baseline. The original `HISTORY_WINDOW=8` declaration was not applied in the loop; this implementation applies it.
- TextCraft: the original single-command policy and prompt. It includes the later parser correction for underscores and rejects `using` in `get` commands. The later prompt expansion and persistent failure summary are excluded because they alter agent behavior.
- Parse errors consume a model decision but not an environment action. Both counts are recorded. The baseline ends at `max_decisions` or `max_actions`, whichever comes first.
- ALFWorld retains the original repeated two-action loop stop after six actions. Environment step errors end an episode and are recorded in its trace.
- On Python 3.13+, the ALFWorld adapter applies a scoped TextWorld 1.7.0 grammar compatibility fix: game variables are passed explicitly to expression evaluation. This addresses a `NameError: name 'r' is not defined` seen during `env.reset()`; it does not change the game files or agent policy.
- The same finalized parsing and harness must be used for any later support condition. If the harness changes, rerun the baseline.

## Reproducibility

Manifests store exact ALFWorld game paths relative to the benchmark split and a SHA-256 hash of each game file. Episodes verify both the file hash and `extra.gamefile` reported at reset. TextCraft 0.0.3 uses unordered collections during task generation, so the same seed can produce a different task across runtimes. TextCraft manifests store the full initial observation and goal. Each episode creates a fresh environment, replays that exact task and goal from the manifest, clears inventory, and verifies that every displayed recipe exists in the installed recipe catalog. `agent_support/textcraft_recipe_order.json` freezes the recipe-file load order and verifies recipe contents by hash. The Colab launcher sets `PYTHONHASHSEED=0` for process-level consistency. These checks freeze the agent-visible task and recipe mechanics across conditions.

The TextCraft development and held-out manifests are committed under `manifests/textcraft/`. They were generated and verified in separate local processes with TextCraft 0.0.3 and `PYTHONHASHSEED=0`. Colab will reuse them and reject a task mismatch. ALFWorld manifests must be generated in Colab after downloading benchmark data:

```bash
python -m agent_support.manifests alfworld --config /content/alfworld_config.yaml --split eval_in_distribution --dev-count 20 --heldout-count 100 --output-dir /content/drive/MyDrive/agent_support/manifests/alfworld
```

Manifest generation fails if the requested number of distinct tasks is unavailable. Keep manifests and results on Drive. Never use held-out trajectories to build a memory corpus.

## Run

Use `notebooks/colab_launcher.ipynb` on a Colab GPU. It clones a configured GitHub repository, mounts Drive, installs dependencies, generates manifests if needed, and runs the baseline. The repository URL must point to this project after it is pushed to GitHub. The launcher leaves the pilot disabled until the smoke results have been reviewed.

The repository is published at `https://github.com/nave-nap/AgenticSDS.git`. The Colab launchers fetch a selected commit from that repository.

Open the launcher notebook in Colab, select a GPU runtime, set `GITHUB_REPO`, and run the cells through the smoke test. Share the two smoke run directories from Drive for trajectory review before setting `RUN_PILOT=True`.

Command line after setup:

```bash
python -m agent_support.runner --config configs/alfworld_baseline.json --manifest /content/drive/MyDrive/agent_support/manifests/alfworld/development.json --output-root /content/drive/MyDrive/agent_support/results --limit 3
PYTHONHASHSEED=0 python -m agent_support.runner --config configs/textcraft_baseline.json --manifest /content/drive/MyDrive/agent_support/manifests/textcraft/development.json --output-root /content/drive/MyDrive/agent_support/results --limit 3
```

Remove `--limit` for the full pilot. The runner writes a unique run directory, a config and manifest snapshot, `run.json`, one JSON episode file at a time, and `summary.json`. The `--limit` value is saved in run metadata so a smoke run cannot be mistaken for the complete manifest. An ALFWorld run also saves its source YAML configuration and hash. A failed task identity check aborts the run.

## Local validation and analysis

No GPU is needed for the local contract tests:

```bash
python3 -m pytest -q
```

Compare complete runs that use the same task IDs and compatible baseline settings:

```bash
python3 -m agent_support.analysis RUN_DIR_A RUN_DIR_B
```

The analysis command rejects incomplete runs, mismatched task sets, and incompatible settings. It reports success, environment actions, model decisions, tokens, failed actions, and per-task differences. It does not silently compare smoke runs to pilots.

For a full development baseline versus M2 run, use `python3 -m agent_support.analysis BASELINE_RUN_DIR M2_RUN_DIR --support-comparison --output comparison.json`. This accepts the support-only config keys and different Git commits while requiring the same full task manifest, model settings, action budget, and benchmark package versions. It reports paired helped/hurt tasks and support cost.

## Current execution status

The reviewed 20-task baseline pilots are frozen in [docs/baseline_pilot.json](docs/baseline_pilot.json): ALFWorld 2/20 and TextCraft 4/20. The original full trajectories remain on Drive. No baseline rerun is needed unless shared baseline behavior changes.

For M2, follow [the training and retrieval protocol](docs/m2_protocol.md) and [the M2 Colab launcher](notebooks/m2_launcher.ipynb). A separate training-task corpus must be collected on a GPU before M2 can run. The development and held-out trajectories cannot enter that corpus.

The next memory experiment uses [successful-trajectory retrieval and action-conditioned M2](docs/trajectory_memory_protocol.md). Its [separate Colab launcher](notebooks/trajectory_memory_launcher.ipynb) reuses the completed training runs, builds one MPNet-indexed corpus per benchmark, then evaluates a retrieval-only ExpeL adaptation and a proposed-action variant against the frozen baseline. These conditions are `expel_retrieval_only_v1` and `m2_trajectory_v1`; they do not overwrite `m2_always` or its results. Local contract tests can run without a GPU; the new smoke and pilot runs still require GPU compute.

The runner now uses a [replaceable per-turn support policy](docs/support_policy.md) to decide when existing mechanisms are invoked. The default policy preserves the current conditions; each run and decision records its policy and selections for later analysis.

For a Slurm cluster, use [the cluster setup and run guide](docs/locust_cluster.md). It stages the existing training and baseline archives from Downloads into a checked transfer directory, resolves the Colab-specific config paths on the cluster, builds memory in CPU jobs, and submits the smoke and pilot runs as GPU jobs.
