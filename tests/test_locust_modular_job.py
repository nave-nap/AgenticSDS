"""The Locust wrapper keeps a requested smoke run small and resumable."""

import os
import subprocess
from pathlib import Path


def test_locust_wrapper_passes_limit_and_resumes_existing_run(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake_python = venv / "bin" / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then printf 'textcraft\\n'; "
        "else printf '%s\\n' \"$@\" > \"$AUDIT_ARGS_PATH\"; fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    arguments = tmp_path / "arguments.txt"
    env = {**os.environ, "AGENT_SUPPORT_DATA": str(tmp_path / "data"),
           "AGENT_SUPPORT_CONFIG": str(repo / "configs/textcraft_modular_development.json"),
           "AGENT_SUPPORT_LIMIT": "3", "SLURM_JOB_ID": "77",
           "SLURM_SUBMIT_DIR": str(repo), "VENV_PATH": str(venv),
           "AUDIT_ARGS_PATH": str(arguments)}
    script = repo / "scripts/locust_modular_job.sh"
    subprocess.run(["bash", str(script)], env=env, check=True)
    first = arguments.read_text().splitlines()
    assert first[first.index("--limit") + 1] == "3"
    assert first[first.index("--run-key") + 1] == "slurm_77"
    assert "--resume" not in first

    (tmp_path / "data/modular_results/textcraft_slurm_77").mkdir(parents=True)
    subprocess.run(["bash", str(script)], env=env, check=True)
    assert arguments.read_text().splitlines()[-1] == "--resume"

    isolated = tmp_path / "data/experiment1_textcraft_development"
    env["AGENT_SUPPORT_OUTPUT_ROOT"] = str(isolated)
    subprocess.run(["bash", str(script)], env=env, check=True)
    isolated_args = arguments.read_text().splitlines()
    assert isolated_args[isolated_args.index("--output-root") + 1] == str(isolated)
    assert "--resume" not in isolated_args
