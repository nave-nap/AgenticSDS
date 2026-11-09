"""Regression for TextWorld grammar variables on Python 3.13+."""

from types import ModuleType

from agent_support.textworld_compat import patch_eval_symbol


def test_eval_symbol_receives_game_variables_explicitly():
    textgen = ModuleType("fake_textgen")

    class TerminalSymbol:
        def __init__(self, value):
            self.value = value

    class EvalSymbol:
        def __init__(self, expression, context):
            self.expression = expression
            self.context = context

        def derive(self, context=None):
            context = context or self.context
            locals().update(context["variables"])
            return [TerminalSymbol(eval(self.expression))]

    textgen.TerminalSymbol = TerminalSymbol
    textgen.EvalSymbol = EvalSymbol
    textgen.format_room = lambda room: f"Room: {room}"

    assert patch_eval_symbol(textgen)
    assert not patch_eval_symbol(textgen)
    symbol = EvalSymbol("format_room(r)", {"variables": {"r": "kitchen"}})
    assert symbol.derive()[0].value == "Room: kitchen"
    assert symbol.derive({"variables": {"r": "bedroom"}})[0].value == "Room: bedroom"
