"""Canonical single-prompt token IDs for the isolated GPT-OSS probe."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def single_input_token_ids(encoded) -> list[int]:
    """Extract one complete input_ids row from BatchEncoding or a bare ID list."""
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Tokenized input must contain one nonempty ID sequence")
    if isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("Tokenized input must contain exactly one sequence")
        values = values[0]
    if not values or any(type(token) is not int or token < 0 for token in values):
        raise ValueError("Tokenized input must be a sequence of nonnegative integer IDs")
    return list(values)


def input_ids_sha256(ids: list[int]) -> str:
    """Hash a compact JSON array of IDs, with no container metadata or batch axis."""
    ids = single_input_token_ids(ids)
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
