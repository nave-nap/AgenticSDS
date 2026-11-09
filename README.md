# Agent support research

This repository starts with a reproducible **no-support baseline** for ALFWorld and TextCraft. The exploratory notebooks remain in `Support Testing/` as source material. The Python package is the experiment implementation; the Colab notebook only installs dependencies and invokes it.

## Baseline decisions

- Model: `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4, deterministic generation.
- ALFWorld: numbered admissible-command policy from the original notebook's first complete trajectory. The later ReAct and cooling-demo experiments are separate policy variants and are not mixed into this baseline. The original `HISTORY_WINDOW=8` declaration was not applied in the loop; this implementation applies it.
- TextCraft: the original single-command policy and prompt. It includes the later parser correction for underscores and rejects `using` in `get` commands. The later prompt expansion and persistent failure summary are excluded because they alter agent behavior.
- Parse errors consume a model decision but not an environment action. Both counts are recorded. The baseline ends at `max_decisions` or `max_actions`, whichever comes first.
- ALFWorld retains the original repeated two-action loop stop after six actions. Environment step errors end an episode and are recorded in its trace.
- On Python 3.13+, the ALFWorld adapter applies a scoped TextWorld 1.7.0 grammar compatibility fix: game variables are passed explicitly to expression evaluation. This addresses a `NameError: name 'r' is not defined` seen during `env.reset()`; it does not change the game files or agent policy.
- The same finalized parsing and harness must be used for any later support condition. If the harness changes, rerun the baseline.

## Reproducibility

Manifests store exact ALFWorld game paths relative to the benchmark split and a SHA-256 hash of each game file. Episodes verify both the file hash and `extra.gamefile` reported at reset. TextCraft 0.0.3 uses unordered Python sets during task generation, so manifest generation and runs must start in a fresh process with `PYTHONHASHSEED=0`; the Colab launcher does this for its subprocesses. TextCraft manifests store a seed and the full initial observation. Generation resets each seed in **two fresh environments** and refuses a mismatch. Every episode resets a new environment and refuses an observation mismatch. A matching observation does not prove hidden state is identical; if the benchmark exposes serializable task state later, extend the manifest check before claiming full replay fidelity.

The TextCraft development and held-out manifests are committed under `manifests/textcraft/`. They were generated and verified in separate local processes with TextCraft 0.0.3 and `PYTHONHASHSEED=0`. Colab will reuse them and reject a task mismatch. ALFWorld manifests must be generated in Colab after downloading benchmark data:

```bash
python -m agent_support.manifests alfworld --config /content/alfworld_config.yaml --split eval_in_distribution --dev-count 20 --heldout-count 100 --output-dir /content/drive/MyDrive/agent_support/manifests/alfworld
```

Manifest generation fails if the requested number of distinct tasks is unavailable. Keep manifests and results on Drive. Never use held-out trajectories to build a memory corpus.

## Run

Use `notebooks/colab_launcher.ipynb` on a Colab GPU. It clones a configured GitHub repository, mounts Drive, installs dependencies, generates manifests if needed, and runs the baseline. The repository URL must point to this project after it is pushed to GitHub. The launcher leaves the pilot disabled until the smoke results have been reviewed.

This directory is already a local Git repository on `main`. To publish it, create an empty GitHub repository without an initial README, then run from this directory:

```bash
git remote add origin https://github.com/YOUR_USERNAME/AgenticSDS.git
git push -u origin main
```

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

## Current execution status

Local validation can run here. GPU smoke tests and pilot scores require a Colab GPU, benchmark installations/data, a GitHub copy of this repository, and a mounted Google Drive. No benchmark result should be reported until those runs complete and their trajectories have been inspected.
