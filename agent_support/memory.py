"""Training-only trajectory corpus and a small semantic retrieval index."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .analysis import load_run
from .common import canonical_digest, read_json, validate_manifest, write_json_atomic


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_POLICY = "successful_episode_valid_action_same_target_v1"


def _identities(manifest: dict) -> tuple[set[str], set[str]]:
    benchmark = manifest["benchmark"]
    fingerprint = "game_sha256" if benchmark == "alfworld" else "observation_sha256"
    return ({task["task_id"] for task in manifest["tasks"]},
            {task[fingerprint] for task in manifest["tasks"]})


def build_corpus(run_dirs: list[str | Path], excluded_manifests: list[str | Path]) -> dict:
    if not run_dirs or not excluded_manifests:
        raise ValueError("Training runs and development/held-out exclusion manifests are required")
    exclusions: dict[str, tuple[set[str], set[str]]] = {}
    exclusion_hashes = {}
    exclusion_partitions: dict[str, set[str]] = {}
    for path in excluded_manifests:
        manifest = read_json(path)
        validate_manifest(manifest)
        if manifest.get("partition") not in {"development", "heldout"}:
            raise ValueError(f"Exclusion manifest must be development or heldout: {path}")
        benchmark = manifest["benchmark"]
        exclusion_partitions.setdefault(benchmark, set()).add(manifest["partition"])
        ids, hashes = _identities(manifest)
        old_ids, old_hashes = exclusions.setdefault(benchmark, (set(), set()))
        old_ids.update(ids)
        old_hashes.update(hashes)
        exclusion_hashes[str(path)] = canonical_digest(manifest)
    if any(partitions != {"development", "heldout"} for partitions in exclusion_partitions.values()):
        raise ValueError("Both development and held-out exclusions are required for each benchmark")
    records = []
    source_runs = []
    seen = set()
    seen_transitions = set()
    for path in run_dirs:
        metadata, config, manifest, episodes = load_run(path)
        benchmark = manifest["benchmark"]
        if benchmark not in exclusions or manifest.get("partition") != "training":
            raise ValueError(f"Run lacks a training manifest and exclusion set: {path}")
        if config.get("condition") != "baseline" or metadata.get("condition") != "baseline":
            raise ValueError(f"Memory source must be a baseline training run: {path}")
        if metadata.get("limit") is not None or metadata.get("selected_task_count") != metadata.get("total_manifest_task_count"):
            raise ValueError(f"Memory source must complete its full training manifest: {path}")
        ids, hashes = _identities(manifest)
        excluded_ids, excluded_hashes = exclusions[benchmark]
        if ids & excluded_ids or hashes & excluded_hashes:
            raise ValueError(f"Training tasks overlap development or held-out tasks: {path}")
        if benchmark == "alfworld" and any(task["split"] != "train" for task in manifest["tasks"]):
            raise ValueError("ALFWorld memory must come from the train split")
        if seen & {(benchmark, task_id) for task_id in ids}:
            raise ValueError("A training task appears in multiple source runs")
        seen.update((benchmark, task_id) for task_id in ids)
        for task in manifest["tasks"]:
            episode = episodes[task["task_id"]]
            for step in episode["trace"]:
                if "outcome" not in step:
                    continue
                action = step["action"]
                outcome = step["outcome"]
                transition = (benchmark, task["task_id"], step["input_observation"],
                              action, step["output_observation"])
                if transition in seen_transitions:
                    continue
                seen_transitions.add(transition)
                record = {
                    "memory_id": canonical_digest([metadata["run_id"], task["task_id"], step["decision"]])[:20],
                    "benchmark": benchmark,
                    "source_run_id": metadata["run_id"],
                    "task_id": task["task_id"],
                    "task_prompt": episode["task_prompt"],
                    "observation": step["input_observation"],
                    "action": action,
                    "feedback": step["output_observation"],
                    "failed_action": bool(outcome.get("failed_action")),
                    "episode_success": bool(episode["success"]),
                    "action_success": bool(outcome.get("success")),
                }
                records.append(record)
        source_runs.append({"run_id": metadata["run_id"], "benchmark": benchmark,
                            "manifest_sha256": metadata["selected_manifest_sha256"],
                            "git_commit": metadata.get("git_commit"), "task_count": len(manifest["tasks"])})
    if not records:
        raise ValueError("Training runs contain no environment actions")
    return {"schema_version": 1, "source_runs": source_runs,
            "excluded_manifest_sha256": exclusion_hashes, "records": records}


def retrieval_text(record: dict) -> str:
    task = record["task_prompt"]
    if "Goal: craft " in task:
        task = "craft " + task.rsplit("Goal: craft ", 1)[1]
    return (f"Proposed action: {record['action']}\nTask: {task[:200]}\n"
            f"Observation: {record['observation'][:400]}")


def related_action(benchmark: str, proposed: str, remembered: str) -> bool:
    """Require an action target match before using semantic similarity."""
    if benchmark == "textcraft":
        pattern = r"^(get|craft)\s+\d+\s+(.+?)(?:\s+using\s+|$)"
        left = re.match(pattern, proposed.lower())
        right = re.match(pattern, remembered.lower())
        return bool(left and right and left.group(1) == right.group(1)
                    and left.group(2).strip() == right.group(2).strip())
    if benchmark == "alfworld":
        stop = {"go", "to", "from", "with", "in", "on", "at", "under", "up", "down",
                "take", "move", "open", "close", "use", "examine", "heat", "cool", "clean", "slice"}

        def parts(action: str) -> tuple[str, set[str]]:
            words = action.lower().split()
            if not words:
                return "", set()
            return words[0], {word for word in words[1:] if word not in stop and not word.isdigit()}

        left_verb, left_objects = parts(proposed)
        right_verb, right_objects = parts(remembered)
        return bool(left_objects and right_objects and left_verb == right_verb
                    and left_objects & right_objects)
    raise ValueError(f"Unknown benchmark: {benchmark}")


class SentenceEmbedder:
    """Mean-pooled MiniLM embeddings using dependencies already in the GPU runner."""

    def __init__(self, model_id: str = EMBEDDING_MODEL, revision: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).cpu().eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 32):
            batch = self.tokenizer(texts[start:start + 32], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt")
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.tolist())
        return vectors


def build_index(corpus: dict, embedder: SentenceEmbedder) -> dict:
    records = corpus["records"]
    return {"schema_version": 1, "embedding_model": embedder.model_id,
            "embedding_revision": getattr(embedder, "revision", None),
            "corpus_sha256": canonical_digest(corpus),
            "memory_ids": [record["memory_id"] for record in records],
            "vectors": embedder.encode([retrieval_text(record) for record in records])}


class ActionMemory:
    def __init__(self, corpus: dict, index: dict, embedder: SentenceEmbedder, top_k: int = 3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if (index["corpus_sha256"] != canonical_digest(corpus)
                or index["embedding_model"] != embedder.model_id
                or index.get("embedding_revision") != getattr(embedder, "revision", None)):
            raise ValueError("Memory index does not match corpus or embedding model")
        records = corpus["records"]
        if index["memory_ids"] != [record["memory_id"] for record in records]:
            raise ValueError("Memory index order does not match corpus")
        vectors = index["vectors"]
        if (len(vectors) != len(records) or not vectors or not isinstance(vectors[0], list)
                or not vectors[0] or any(len(vector) != len(vectors[0]) for vector in vectors)):
            raise ValueError("Memory vectors are malformed")
        self.records = records
        self.vectors = vectors
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, benchmark: str, task_prompt: str, observation: str, action: str) -> tuple[list[dict], float]:
        started = time.perf_counter()
        candidates = [i for i, record in enumerate(self.records)
                      if record["benchmark"] == benchmark and record["episode_success"]
                      and not record["failed_action"]
                      and related_action(benchmark, action, record["action"])]
        if not candidates:
            return [], round(time.perf_counter() - started, 3)
        query = retrieval_text({"task_prompt": task_prompt, "observation": observation, "action": action})
        vector = self.embedder.encode([query])[0]
        if len(vector) != len(self.vectors[0]):
            raise ValueError("Memory query vector dimension differs from index")
        similarities = {i: sum(a * b for a, b in zip(self.vectors[i], vector)) for i in candidates}
        ranked = sorted(candidates, key=lambda i: (-float(similarities[i]), self.records[i]["memory_id"]))
        best = []
        used_tasks = set()
        for i in ranked:
            if self.records[i]["task_id"] in used_tasks:
                continue
            best.append(i)
            used_tasks.add(self.records[i]["task_id"])
            if len(best) == self.top_k:
                break
        hits = [{**self.records[i], "similarity": round(float(similarities[i]), 6)} for i in best]
        return hits, round(time.perf_counter() - started, 3)


def support_messages(base_messages: list[dict], proposed_action: str, hits: list[dict], benchmark: str) -> list[dict]:
    lines = [f"You proposed: {proposed_action}",
             "The following actions were accepted by the environment in training episodes that ultimately succeeded, "
             "and they share the proposed action's target or object. "
             "Their inventories and states may differ from yours. Keep the proposed action unless an example clearly supports a better choice."]
    for i, hit in enumerate(hits, 1):
        task = hit["task_prompt"]
        if "Goal: craft " in task:
            task = "craft " + task.rsplit("Goal: craft ", 1)[1]
        lines.extend([f"Experience {i}:", f"Task: {task[:300]}",
                      f"Observation: {hit['observation'][:400]}", f"Action: {hit['action']}",
                      f"Feedback: {hit['feedback'][:400]}",
                      f"Outcome: {'failed action' if hit['failed_action'] else 'valid action'}; "
                      f"episode {'succeeded' if hit['episode_success'] else 'did not succeed'}."])
    lines.append("Choose the best action now. " +
                 ("Reply only with one number from the available actions list." if benchmark == "alfworld"
                  else "Reply with exactly one Action: command using the task's allowed format."))
    messages = [dict(message) for message in base_messages]
    messages[-1]["content"] += "\n\n" + "\n".join(lines)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Completed baseline training run directory; repeatable")
    parser.add_argument("--exclude-manifest", action="append", required=True,
                        help="Development or held-out manifest; repeat for both partitions")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    corpus = build_corpus(args.run, args.exclude_manifest)
    output = Path(args.output_dir)
    corpus_path = output / "corpus.json"
    if corpus_path.exists() and read_json(corpus_path) != corpus:
        raise FileExistsError(f"Existing memory corpus differs: {corpus_path}")
    if not corpus_path.exists():
        write_json_atomic(corpus_path, corpus)
    index = build_index(corpus, SentenceEmbedder())
    index_path = output / "index.json"
    if index_path.exists() and read_json(index_path) != index:
        raise FileExistsError(f"Existing memory index differs: {index_path}")
    if not index_path.exists():
        write_json_atomic(index_path, index)
    print(json.dumps({"records": len(corpus["records"]), "source_runs": corpus["source_runs"],
                      "corpus_sha256": index["corpus_sha256"], "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
