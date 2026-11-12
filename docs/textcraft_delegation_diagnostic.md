# TextCraft weak-trajectory delegation diagnostic

This development-only diagnostic asks a registered model adapter for the next action at 40 decision points from the complete Experiment 1 Qwen never-support run (`slurm_142829`). It is a targeted probe of whether an alternative model can respond differently when Qwen is struggling. It is **not** a representative 40-decision accuracy estimate: selection uses Qwen's immediate outcomes, and many points are correlated within tasks.

The [dataset manifest](../diagnostics/textcraft_delegation_v1/dataset.json) pins the source archive, source run/manifest/config/model revisions, prompt source hash, and output file hashes. The 40 points cover all 20 development tasks, at most three points per task, with eight each of:

- Immediately repeated failed commands (for example, another unsuccessful `craft ... red dye` after that same command failed).
- Crafts that receive missing-item feedback (for example, trying to craft a gray bed before the required bed and dye are available).
- `get` commands that receive unavailable-item feedback (for example, `get 1 cyan dye`).
- Off-recipe crafting commands that the environment rejects (for example, `craft 6 planks using 1 spruce logs` when that exact recipe is not displayed).
- Productive controls: four successful gathers and four successful crafts on a visible goal dependency path.

The category is assigned **after** seeing Qwen's action and immediate feedback. It is an analysis label, not a feature supplied to the alternative model. The productive-path check expands generic recipe ingredients such as `planks` to displayed matching recipes. Category labels are still descriptive; they do not prove that a different action would improve final task success.

## Files and information boundary

- [`inputs.jsonl`](../diagnostics/textcraft_delegation_v1/inputs.jsonl) contains only pre-generation context: task ID and decision number, visible task goal, current observation, the last eight visible action/feedback pairs, and the exact `system`/`user` messages reconstructed with the unchanged `single_command_v1` prompt. The prediction command reads this file and `dataset.json`; it never opens the labels or replay prefixes.
- [`labels.jsonl`](../diagnostics/textcraft_delegation_v1/labels.jsonl) stores the category, original Qwen action, and **immediate** observed outcome. It is for later scoring only.
- [`replay.jsonl`](../diagnostics/textcraft_delegation_v1/replay.jsonl) stores the task manifest record and full **past** action prefix needed to reconstruct the environment state. It contains no future trajectory, reference solution, or hidden simulator state. It is for offline replay, never model generation.

The original Qwen prompt did not include an explicit separate “current observation” section. It placed the initial task and the last eight action/feedback pairs in the user message. The extra structured fields in `inputs.jsonl` document that context; the adapter receives only `messages`. The builder verifies each trace's observation continuity and parser result, the episode checkpoint hashes, the source run's exact summary, and that the prompt implementation matches the source framework commit.

## Rebuild the dataset

With the Experiment 1 ZIP available locally, run this from the `gptoss20b_standalone` checkout, using a **new** output directory:

```bash
python3 -m agent_support.delegation_diagnostic build \
  --archive "$HOME/Downloads/experiment1_for_codex.zip" \
  --output-dir /tmp/textcraft_delegation_rebuild
python3 - <<'PY'
from agent_support.common import read_json
original = read_json('diagnostics/textcraft_delegation_v1/dataset.json')
rebuilt = read_json('/tmp/textcraft_delegation_rebuild/dataset.json')
assert original == rebuilt
print('40 inputs reproduced with identical file hashes')
PY
```

No model files are downloaded by this command. The historical ZIP and all source runs remain untouched.

## Generate next-action predictions later

After the GPT-OSS environment and storage are ready, run the standalone adapter against a new owner-only output directory. `--limit 3` is a small inference smoke; omit it for all 40 points. These are commands to run later; no GPU inference was launched while preparing this diagnostic.

```bash
cd "$HOME/AgenticSDS-gptoss20b"
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export VENV_GPTOSS="$AGENT_SUPPORT_DATA/venvs/gptoss20b"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export PYTHONHASHSEED=0
"$VENV_GPTOSS/bin/python" -m agent_support.delegation_diagnostic predict \
  --config configs/standalone_textcraft_dev_gptoss20b_never.json \
  --inputs diagnostics/textcraft_delegation_v1/inputs.jsonl \
  --output-dir "$AGENT_SUPPORT_DATA/delegation_diagnostic/gptoss20b_3point_smoke" \
  --limit 3
```

After that output has been checked, use a **different** output directory for the full diagnostic:

```bash
"$VENV_GPTOSS/bin/python" -m agent_support.delegation_diagnostic predict \
  --config configs/standalone_textcraft_dev_gptoss20b_never.json \
  --inputs diagnostics/textcraft_delegation_v1/inputs.jsonl \
  --output-dir "$AGENT_SUPPORT_DATA/delegation_diagnostic/gptoss20b_full40"
```

The script uses `load_components(config)` and `make_model(config)`, so any registered adapter with the standard `generate` and `metadata` methods can run against these messages. The result directory is new and cannot be overwritten. It contains one JSON prediction per point and `run.json` with configuration/input and completed-prediction hashes, loaded model metadata, and Git provenance. GPT-OSS predictions include generation status, token counts, and peak GPU memory. Keep the output directory private because raw completions may include reasoning. Sampling is seeded by the adapter configuration, but restarting a process does not promise bitwise-identical predictions.

## Evaluate from the same state

The TextCraft version and recipe order were pinned in the source run. Exact one-step replay is feasible **if** the installed environment reproduces every past observation. The `replay` command resets the task twice, replays the entire past action prefix twice, checks each stored observation and the original Qwen action's immediate outcome, then applies the alternative action from the same verified state. It refuses to report a comparison if reproduction diverges. A parsed-invalid alternative has no environment step. Start the replay Python process with `PYTHONHASHSEED=0`; the TextCraft adapter requires that setting for recipe ordering. A local CPU check with TextCraft 0.0.3 reproduced all 40 selected Qwen prefixes and immediate outcomes exactly.

```bash
"$VENV_GPTOSS/bin/python" -m agent_support.delegation_diagnostic replay \
  --predictions "$AGENT_SUPPORT_DATA/delegation_diagnostic/gptoss20b_full40" \
  --inputs diagnostics/textcraft_delegation_v1/inputs.jsonl \
  --labels diagnostics/textcraft_delegation_v1/labels.jsonl \
  --replay-prefixes diagnostics/textcraft_delegation_v1/replay.jsonl \
  --output "$AGENT_SUPPORT_DATA/delegation_diagnostic/gptoss20b_immediate_replay.json"
```

The replay report counts valid alternatives, changed actions, immediate failures avoided, and new immediate failures by category. **Immediate feedback is not task success.** An alternative can avoid an error yet waste an action or make later completion harder. A stronger claim would require a controlled continuation from each branch with the same fixed continuation policy and interaction budget; that experiment is not implemented here. Do not use the held-out manifest to tune this diagnostic.
