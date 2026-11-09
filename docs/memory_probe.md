# Training-only sequence-memory probe

This diagnostic compares three next-action choices on saved TextCraft training
tasks: the unchanged baseline, the frozen M2 action-conditioned memory, and a
short sequence from another successful training task. It does **not** change
the M2 condition, use development or held-out trajectories, or measure full
episode success.

The sequence source is selected automatically before Qwen generates an action.
The selector requires (1) a different successful training task, (2) an
accepted source action mentioning the first word of the current goal, and
(3) at least one recipe line shared by the two tasks. It ranks by the number
of shared recipe lines and displays the last four accepted source actions
with their real environment feedback. If there is no source, the original
baseline action is retained. The current M2 corpus is also filtered to exclude
the task being probed, matching evaluation-time separation.

The six default training seeds include three where this small corpus has a
cross-task sequence (1009, 1003, 1027) and three where it does not (1018,
1007, 1023). This is a targeted diagnostic, not an unbiased estimate of
benchmark accuracy. The report records source task IDs, memory IDs, chosen
actions, one-step environment acceptance, model outputs, costs, and hashes.

## Run on Colab

Open `notebooks/m2_launcher.ipynb` in a GPU runtime. Set `REVISION` to the
commit containing this probe; leave all `RUN_*` flags `False`. Run the settings
cell and section 1 setup cell only. The saved training runs and M2 corpus/index
must already exist in Drive. Then run this **new** cell:

```python
from agent_support.common import read_json
from datetime import datetime, timezone
import os, subprocess, sys

corpus_path = root / 'memory' / 'textcraft' / 'corpus.json'
index_path = root / 'memory' / 'textcraft' / 'index.json'
corpus = read_json(corpus_path)
source_id = next(item['run_id'] for item in corpus['source_runs']
                 if item['benchmark'] == 'textcraft')
training_run = root / 'training_runs' / f'textcraft_{source_id}'
if not training_run.exists() or not index_path.exists():
    raise RuntimeError('Training run or M2 index missing from Drive')

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = root / 'diagnostics' / f'textcraft_sequence_probe_{stamp}_{commit[:7]}.json'
command = [sys.executable, '-m', 'agent_support.memory_probe',
           '--config', str(repo / 'configs' / 'textcraft_baseline.json'),
           '--training-run', str(training_run),
           '--corpus', str(corpus_path), '--index', str(index_path),
           '--output', str(output)]
subprocess.run(command, check=True, cwd=repo,
               env={**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUNBUFFERED': '1'})
print('Send this file for review:', output)
```

The code runs in a fresh Python process because TextCraft requires
`PYTHONHASHSEED=0` at interpreter startup. A notebook setting alone cannot
change the current kernel's hash randomization.

## Read the report

For each task, compare `baseline`, `m2`, and `sequence` action and
`check.accepted`. If M2 has no matching record, `m2.fallback_reason` is
`no_relevant_memory`. If no suitable sequence exists,
`sequence.fallback_reason` is `no_relevant_sequence`. An accepted first action
is only a prerequisite to solving the whole task; it does not imply a higher
full-task success rate. A promising result would justify a larger training-only
probe, then a separately versioned development pilot.
