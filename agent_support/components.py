"""Load explicitly configured extension modules before registry validation."""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path


def load_components(config: dict) -> dict[str, dict]:
    names = config.get("component_modules", []) if config.get("schema_version") == 2 else []
    if not isinstance(names, list) or len(names) != len(set(names)) or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name)
            for name in names):
        raise ValueError("component_modules must be a list of distinct Python module names")
    loaded = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
        if path is None or not path.is_file():
            raise ValueError(f"Component module has no readable source file: {name}")
        loaded[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return loaded
