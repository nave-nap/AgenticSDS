# Run the trajectory-memory experiment on a Slurm cluster

The cluster workflow uses the same Python runner as Colab. Evan's account is on Penn's Locust cluster at `locust-login.seas.upenn.edu`. SSH uses the PennKey username `evanpan`, password, and two-factor authentication. Off campus, connect to the [Penn VPN](https://hpc.engineering.upenn.edu/using-penn-vpn-to-access-resources/) first. The Slurm partition, account, Python module, and GPU model must be checked from the account's allocation. Slurm's [official `sbatch` documentation](https://slurm.schedmd.com/sbatch.html) describes the resource flags used here.

## 1. Transfer the frozen inputs

The user's existing downloads contain the full baseline results and full training runs. `manifests/alfworld/heldout.json` must also be downloaded from `MyDrive/agent_support/manifests/alfworld/heldout.json`. From this repository on the local computer:

```bash
python3 -m agent_support.cluster_data \
  --training-archive "$HOME/Downloads/training_runs-20261109T035837Z-1-001.zip" \
  --baseline-archive "$HOME/Downloads/results-20261109T021721Z-1-001.zip" \
  --alfworld-heldout "$HOME/Downloads/heldout.json" \
  --output data/locust_transfer
```

The command checks the frozen baseline run IDs and manifest hashes, the original ALFWorld held-out manifest hash, and training versus evaluation task separation. It stages only the two full training runs and two frozen baseline runs, plus their manifests and ALFWorld YAML. The prepared `data/locust_transfer/` is ignored by Git and must be transferred separately from the repository. It has already been built in this local workspace.

After connecting to campus VPN if needed, copy the prepared directory. SSH/SCP will prompt for the PennKey authentication and two-factor step:

```bash
LOCUST_LOGIN='evanpan@locust-login.seas.upenn.edu'
scp -r data/locust_transfer "$LOCUST_LOGIN:~/agent_support_data"
```

No Google Drive mount is needed on the cluster. If `scp` is unavailable, use the cluster's approved file-transfer method to place this directory at the same destination. Do not put the downloaded results or Hugging Face cache in Git.

## 2. Install the code and benchmark data

Log in with `ssh evanpan@locust-login.seas.upenn.edu`. On the cluster login node, clone the published repository and select the cluster-workflow commit (shown in the setup handoff). Select a Python/CUDA module supported by Locust; if Python 3.13.16 and the frozen baseline package versions are unavailable, plan a new 20-task cluster baseline for each benchmark before comparing success rates.

```bash
git clone https://github.com/nave-nap/AgenticSDS.git
cd AgenticSDS
git checkout CLUSTER_WORKFLOW_COMMIT
PYTHON_BIN=python3.13 bash scripts/locust_install.sh
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
```

The script creates `.venv/` and installs the repository's pinned requirements. The Qwen runner uses PyTorch CUDA; verify the cluster's PyTorch build and GPU driver in a GPU allocation. If needed, install a compatible CUDA PyTorch build according to the cluster's instructions. Install the ALFWorld benchmark data into the persistent `ALFWORLD_DATA` location if absent:

```bash
source .venv/bin/activate
alfworld-download
```

The original ALFWorld YAML uses `$ALFWORLD_DATA` and the runner verifies each game against its frozen hash. If compute nodes lack outbound network access, populate the shared Hugging Face cache with `Qwen/Qwen2.5-7B-Instruct` and `sentence-transformers/all-mpnet-base-v2` before submitting jobs.

## 3. Discover the Slurm partition and preflight

On the cluster login node:

```bash
sinfo -o '%P %G %l'
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export PYTHONHASHSEED=0
.venv/bin/python -m agent_support.cluster_workflow preflight --benchmark alfworld
.venv/bin/python -m agent_support.cluster_workflow preflight --benchmark textcraft
```

`preflight` checks transferred runs and manifests, ALFWorld data, and version compatibility with the frozen Colab baseline. Its `frozen_baseline_mismatches` list is a decision point. If nonempty, run a new full **cluster baseline** for that benchmark and use it as A in the comparison. The already transferred training runs remain valid memory sources because the corpus builder checks task and content identity.

## 4. Build memory on CPU

Set `CPU_PARTITION` to a CPU-capable partition from `sinfo` that your account may use. [Public group notes from 2024](https://github.com/Dobriban-Group/wiki/wiki/Compute-resources) mention `whartonstat`, but confirm current access before using it. The job script inherits `AGENT_SUPPORT_DATA`, `ALFWORLD_DATA`, and `HF_HOME` from the submission shell. Run the two jobs and inspect their `slurm-<jobid>.out` logs. The memory directories are `agent_support_data/trajectory_memory/{alfworld,textcraft}`.

```bash
CPU_PARTITION='YOUR_CPU_PARTITION'
sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=16G --time=02:00:00 \
  scripts/locust_job.sh build-memory --benchmark alfworld
sbatch --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=16G --time=02:00:00 \
  scripts/locust_job.sh build-memory --benchmark textcraft
```

The builder uses the completed baseline training runs; it does not recollect trajectories. It writes corpus and MPNet index files with hashes and refuses to overwrite differing artifacts.

## 5. Smoke tests, then full pilots on GPU

Set `GPU_PARTITION` to an available GPU partition in your allocation. Confirm that a node has enough GPU memory for Qwen2.5-7B-Instruct in NF4; the job checks `torch.cuda.is_available()` before loading the model. Submit one smoke job, inspect `run.json`, `summary.json`, and episode traces under `agent_support_data/results/`, then repeat for both benchmarks and both support conditions. If this Locust Slurm installation does not accept `--gpus=1`, use its documented `--gres=gpu:1` syntax.

```bash
GPU_PARTITION='YOUR_GPU_PARTITION'
sbatch --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  scripts/locust_job.sh smoke --benchmark alfworld --condition expel_retrieval_only_v1
```

After reviewing the smoke results, replace `smoke` with `pilot` and allow a longer time limit for each full 20-task run:

```bash
sbatch --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
  scripts/locust_job.sh pilot --benchmark alfworld --condition expel_retrieval_only_v1
```

Run the same commands for `m2_trajectory_v1` and for `textcraft`. If the preflight reported baseline incompatibility, also submit `pilot --condition baseline` for each affected benchmark on the same environment. Run IDs are unique, so prior results are preserved. Do not compare a three-task smoke run to a full pilot.

## 6. Compare paired runs

If the frozen baseline was compatible, run:

```bash
PYTHONHASHSEED=0 .venv/bin/python -m agent_support.cluster_workflow compare --benchmark alfworld
PYTHONHASHSEED=0 .venv/bin/python -m agent_support.cluster_workflow compare --benchmark textcraft
```

If you ran a new cluster baseline, add `--baseline-run "$AGENT_SUPPORT_DATA/results/EXACT_CLUSTER_BASELINE_DIR"` to that benchmark's command. This writes A–B, A–C, and B–C comparisons under `agent_support_data/comparisons/`. The comparison rejects incomplete or mismatched runs and requires B and C to use the same memory hashes and retrieval settings.

`squeue -u "$USER"` shows queued/running jobs, and `tail -f slurm-JOBID.out` follows a job log. When a job fails, its `run.json` records the underlying exception; share that and the end of its Slurm log for debugging.
