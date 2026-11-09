# Run the trajectory-memory experiment on a Slurm cluster

The cluster workflow uses the same Python runner as Colab. Evan's account is on Penn's Locust cluster at `locust-login.seas.upenn.edu`. SSH uses the PennKey username `evanpan`, password, and two-factor authentication. Off campus, connect to the [Penn VPN](https://hpc.engineering.upenn.edu/using-penn-vpn-to-access-resources/) first. The Slurm partition, account, Python module, and GPU model must be checked from the account's allocation. Slurm's [official `sbatch` documentation](https://slurm.schedmd.com/sbatch.html) describes the resource flags used here.

The retrieval-only and M2 configs pin the [per-turn support policy](support_policy.md) `fixed_by_condition_v1`. It records which support calls were selected at every decision, so later policy versions can be compared without changing the retrieval mechanisms.

The repository is already public on GitHub. The experiment data bundle is ignored by Git. The setup below makes its directory and the cloned working directory owner-only, so ordinary cluster users should not be able to open the data or job logs through filesystem permissions. Cluster administrators may still access files, and Slurm may show other users basic job metadata such as account name, job name, and state. Check the actual directory permissions after transfer with `ls -ld "$HOME/agent_support_data" "$HOME/AgenticSDS"`.

Keep the original archives and a copy of completed results on your own computer. The cluster still needs working copies while jobs run. Locust's `/home` has a **50 GB per-user quota** according to the administrator's notice; `/shared_data0/` and compute-node local scratch are outside that quota, but the notice also says `/shared_data0/` is filling up. Before transferring or downloading models, log in and check usage:

```bash
quota -s 2>/dev/null || true
du -sh "$HOME"
df -h /home /shared_data0
```

The transfer bundle is about 4 MB. A local Python environment, ALFWorld data, and especially the Qwen model cache require much more space. If your home directory has sufficient headroom under 50 GB, the simple paths below are fine. If not, use a writable group directory under `/shared_data0/` for `AGENT_SUPPORT_DATA`, `ALFWORLD_DATA`, and `HF_HOME` after checking its free space and permissions. Do not assume `/shared_data0/$USER` exists or is writable. Compute-node scratch is temporary: copy any results you want to keep back to persistent storage before the job ends. The install script disables pip's download cache to avoid an unnecessary second copy of packages in `/home`.

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

After logging in, restrict access to the transferred directory with `chmod 700 "$HOME/agent_support_data"`. No Google Drive mount is needed on the cluster. If `scp` is unavailable, use the cluster's approved file-transfer method to place this directory at the same destination. Do not put the downloaded results or Hugging Face cache in Git.

## 2. Install the code and benchmark data

Log in with `ssh evanpan@locust-login.seas.upenn.edu`. On the cluster login node, clone the published repository and select the cluster-workflow commit (shown in the setup handoff). Select a Python/CUDA module supported by Locust; if Python 3.13.16 and the frozen baseline package versions are unavailable, plan a new 20-task cluster baseline for each benchmark before comparing success rates.

```bash
umask 077
git clone https://github.com/nave-nap/AgenticSDS.git
cd AgenticSDS
git checkout CLUSTER_WORKFLOW_COMMIT
chmod 700 "$HOME/AgenticSDS" "$HOME/agent_support_data"
PYTHON_BIN=python3.13 bash scripts/locust_install.sh
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export ALFWORLD_DATA="$AGENT_SUPPORT_DATA/alfworld"
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
export ALFWORLD_DATA="$AGENT_SUPPORT_DATA/alfworld"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export PYTHONHASHSEED=0
.venv/bin/python -m agent_support.cluster_workflow preflight --benchmark alfworld
.venv/bin/python -m agent_support.cluster_workflow preflight --benchmark textcraft
```

`preflight` checks transferred runs and manifests, ALFWorld data, and version compatibility with the frozen Colab baseline. Its `frozen_baseline_mismatches` list is a decision point. If nonempty, run a new full **cluster baseline** for that benchmark and use it as A in the comparison. The already transferred training runs remain valid memory sources because the corpus builder checks task and content identity.

The `home_resident_storage` field shows which configured data and cache locations are under your home directory. It does not measure your personal quota; use `quota` and `du` for that. If the install or model download approaches the quota, move the working directory and cache to a writable shared location before submitting more jobs.

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

These are batch jobs: Slurm releases the GPU when the program exits. Avoid leaving an interactive `salloc` GPU session idle. Check `squeue -u "$USER"` and cancel a job you no longer need with `scancel JOB_ID`.

## 6. Compare paired runs

If the frozen baseline was compatible, run:

```bash
PYTHONHASHSEED=0 .venv/bin/python -m agent_support.cluster_workflow compare --benchmark alfworld
PYTHONHASHSEED=0 .venv/bin/python -m agent_support.cluster_workflow compare --benchmark textcraft
```

If you ran a new cluster baseline, add `--baseline-run "$AGENT_SUPPORT_DATA/results/EXACT_CLUSTER_BASELINE_DIR"` to that benchmark's command. This writes A–B, A–C, and B–C comparisons under `agent_support_data/comparisons/`. The comparison rejects incomplete or mismatched runs and requires B and C to use the same memory hashes and retrieval settings.

`squeue -u "$USER"` shows queued/running jobs, and `tail -f slurm-JOBID.out` follows a job log. When a job fails, its `run.json` records the underlying exception; share that and the end of its Slurm log for debugging.

After a run, download its `results/` and `comparisons/` directories to your computer for a durable local copy. For the default home-directory layout, run these commands on your Mac from the repository:

```bash
mkdir -p data/locust_downloads
scp -r evanpan@locust-login.seas.upenn.edu:agent_support_data/results data/locust_downloads/
scp -r evanpan@locust-login.seas.upenn.edu:agent_support_data/comparisons data/locust_downloads/
```

If `AGENT_SUPPORT_DATA` is on `/shared_data0/`, substitute its actual absolute path in the remote side of those commands. Keep the cluster copies until you have verified the download; then remove only copies you no longer need. The frozen input bundle on your computer and in Drive remains the original source.
