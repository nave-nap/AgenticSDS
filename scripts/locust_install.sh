#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_path="${VENV_PATH:-$repo_root/.venv}"

"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" -m pip install --upgrade pip
"$venv_path/bin/python" -m pip install -r "$repo_root/requirements.txt"
"$venv_path/bin/python" - <<'PY'
import platform
import torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("GPU availability must be checked inside an allocated GPU job.")
PY
