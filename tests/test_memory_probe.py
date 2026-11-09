from agent_support.common import canonical_digest
from agent_support.memory_probe import recipes, select_sequence, sequence_messages, without_task


def episode(task_id, goal, recipe_lines, actions, success):
    initial = "Crafting commands:\n" + "\n".join(recipe_lines) + f"\n\nGoal: craft {goal}."
    return {"task_id": task_id, "initial_observation": initial, "success": success,
            "trace": [{"action": action, "output_observation": feedback,
                       "outcome": {"failed_action": failed}}
                      for action, feedback, failed in actions]}


def test_sequence_selection_uses_other_successful_task_with_shared_recipes():
    recipe = "craft 4 oak planks using 1 oak logs"
    query = episode("textcraft:seed:1", "oak pressure plate", [recipe], [], False)
    same_task = episode("textcraft:seed:1", "oak pressure plate", [recipe],
                        [("get 1 oak logs", "Got 1 oak logs", False)], True)
    failed_task = episode("textcraft:seed:2", "oak door", [recipe],
                          [("get 1 oak logs", "Got 1 oak logs", False)], False)
    unrelated = episode("textcraft:seed:4", "barrel", [recipe],
                        [("get 1 spruce logs", "Got 1 spruce logs", False)], True)
    source = episode("textcraft:seed:3", "oak button", [recipe], [
        ("get 1 oak logs", "Got 1 oak logs", False),
        ("craft 4 oak planks using 1 oak logs", "missing items", True),
        ("craft 4 oak planks using 1 oak logs", "Crafted planks", False),
        ("craft 1 oak button using 1 oak planks", "Crafted button", False),
    ], True)
    selected, overlap, steps = select_sequence(query, {
        item["task_id"]: item for item in (same_task, failed_task, unrelated, source)})
    assert selected["task_id"] == "textcraft:seed:3"
    assert overlap == 1
    assert [step["action"] for step in steps] == [
        "get 1 oak logs", "craft 4 oak planks using 1 oak logs",
        "craft 1 oak button using 1 oak planks"]
    messages = sequence_messages([{"role": "system", "content": "fixed"},
                                  {"role": "user", "content": "current task"}], selected, steps)
    assert messages[0]["content"] == "fixed"
    assert "current task" in messages[1]["content"]
    assert "Got 1 oak logs" in messages[1]["content"]
    assert "missing items" not in messages[1]["content"]
    assert recipes(query["initial_observation"]) == {recipe}


def test_leave_one_task_out_keeps_index_order_and_updates_hash():
    records = [{"task_id": "train:1", "memory_id": "a"},
               {"task_id": "train:2", "memory_id": "b"},
               {"task_id": "train:1", "memory_id": "c"}]
    corpus = {"records": records, "source_runs": [{"run_id": "run"}]}
    index = {"memory_ids": ["a", "b", "c"], "vectors": [[1], [2], [3]],
             "corpus_sha256": canonical_digest(corpus)}
    filtered, filtered_index = without_task(corpus, index, "train:1")
    assert filtered["records"] == [records[1]]
    assert filtered_index["memory_ids"] == ["b"]
    assert filtered_index["vectors"] == [[2]]
    assert filtered_index["corpus_sha256"] == canonical_digest(filtered)
    assert corpus["records"] == records


def test_sequence_selection_abstains_without_matching_successful_path():
    recipe = "craft 1 book using 3 paper, 1 leather"
    query = episode("textcraft:seed:5", "book", [recipe], [], False)
    unrelated = episode("textcraft:seed:6", "barrel", [recipe],
                        [("get 3 sugar cane", "Got sugar cane", False)], True)
    assert select_sequence(query, {query["task_id"]: query,
                                   unrelated["task_id"]: unrelated}) is None
