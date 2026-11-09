"""Compatibility for TextWorld PDDL grammar evaluation on Python 3.13+.

TextWorld 1.7.0 adds game variables with ``locals().update`` before ``eval``.
Python 3.13 gives each ``locals()`` call an independent snapshot in functions,
so expressions such as ``r.name`` fail with NameError. Keep the same module
globals and game variables, but pass the evaluation namespace explicitly.
"""

from __future__ import annotations

import sys
from types import ModuleType


def patch_eval_symbol(textgen: ModuleType) -> bool:
    """Patch one TextWorld textgen module; return whether a patch was needed."""
    original = textgen.EvalSymbol.derive
    if getattr(original, "_agent_support_explicit_scope", False):
        return False

    def derive(symbol, context=None):
        context = context or symbol.context
        namespace = vars(textgen).copy()
        namespace.update({"self": symbol, "context": context})
        namespace.update(context["variables"])
        value = eval(symbol.expression, namespace, namespace)
        return [textgen.TerminalSymbol(value)]

    derive._agent_support_explicit_scope = True
    textgen.EvalSymbol.derive = derive
    return True


def install_if_needed() -> bool:
    """Apply the compatibility fix before a one-game ALFWorld environment resets."""
    if sys.version_info < (3, 13):
        return False
    import textworld.envs.pddl.textgen as textgen

    return patch_eval_symbol(textgen)
