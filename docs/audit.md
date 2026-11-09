# Notebook audit and baseline choices

## ALFWorld

The notebook installs ALFWorld, downloads benchmark data, loads `AlfredTWEnv` from a downloaded base config, and loads Qwen2.5-7B-Instruct with 4-bit NF4. Its first complete agent loop enumerates `admissible_commands`, asks for a one-based number, parses it strictly, executes the chosen command, and uses `next_info["won"][0]` for success. It writes a trace to `/content/alfworld_trace.json`. Later cells experiment with ReAct demonstrations and a distinct free-form action policy, including special `think:` turns. These are not one consistent baseline. The notebook's `train_eval="train"` and `num_train_games=1` also do not define a held-out evaluation set. It does not preserve exact game identity in the saved result. The declared history window of eight is not applied in the first loop.

We retain the numbered action policy and its task prompt. We use exact game-file manifests, verify the game identity at reset, and apply the eight-action history window. The ReAct branch stays in the exploratory notebook.

The first Colab smoke attempt on Python 3.13 revealed a TextWorld 1.7.0 grammar failure during `env.reset()`: `EvalSymbol.derive` mutates `locals()` and then calls `eval`, leaving game variable `r` unavailable. Python 3.13 changed function-local namespace behavior. The adapter now passes the same grammar module globals and game variables as an explicit evaluation namespace. The compatibility code is isolated in `agent_support/textworld_compat.py` and used only on Python 3.13+.

## TextCraft

The notebook uses `textcraft==0.0.3`, a fresh `TextCraft` environment per episode, Qwen2.5-7B-Instruct in NF4, a fixed 30-decision loop, a command parser, JSON trajectories, and Drive storage. It creates a seed/initial-observation manifest. A later cell rebuilds the manifest in fresh environments and checks identity, demonstrating that repeated seeds alone cannot be trusted. The final two cells patch the parser, expand the system prompt, and add a persistent failure summary. Those changes make the notebook's behavior dependent on cell order.

We retain the original command policy with the parser correction. The prompt expansion and failure summary are excluded from the no-support baseline. Inspection of TextCraft 0.0.3 source shows that `reset` forms recipe lists and the goal list from Python sets, then uses a seed. Python's process-level hash randomization can reorder those sets across runtimes despite the same seed. The launcher pins `PYTHONHASHSEED=0` before subprocesses start. The manifest generator verifies each task across two fresh environments and the runner verifies the initial observation before inference. Further task-state serialization is unresolved and should be investigated on a real Colab installation before claiming hidden-state replay fidelity.

## Common protocol

Use one checkpoint and quantization, deterministic generation, fixed prompts and parsers per benchmark, exact task manifests, a 30-environment-action cap, and separate decision counts. Save full trajectories, token use, wall time, package versions, config, and manifest snapshots. A parser or harness correction applies to all conditions and requires rerunning the baseline. Local tests use fake model and benchmark adapters; real inference belongs in Colab.
