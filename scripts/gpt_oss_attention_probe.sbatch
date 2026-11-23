#!/usr/bin/env bash
#SBATCH --job-name=gptoss-attention-probe
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077

repo_root="${SLURM_SUBMIT_DIR:?Submit from the isolated attention-probe worktree}"
: "${PROBE_FIXTURE:?Set the immutable fixture JSON path}"
: "${PROBE_OUTPUT_DIR:?Set a new output directory}"
: "${PROBE_BACKEND:?Set eager or flex_attention}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set the pinned GPT-OSS Python}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set the pinned GPT-OSS cache}"
: "${HF_HOME:?Set the pinned Qwen cache}"
venv_path="${VENV_PATH:-/home/evanpan/AgenticSDS/.venv}"
test -x "$venv_path/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$PROBE_FIXTURE"
test ! -e "$PROBE_OUTPUT_DIR"
test "${AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE:-0}" != 1
case "$PROBE_BACKEND" in eager|flex_attention) ;; *) exit 2 ;; esac

export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
cd "$repo_root"
"$venv_path/bin/python" -m agent_support.gpt_oss_attention_probe run \
  --fixture "$PROBE_FIXTURE" --backend "$PROBE_BACKEND" \
  --output-dir "$PROBE_OUTPUT_DIR"
