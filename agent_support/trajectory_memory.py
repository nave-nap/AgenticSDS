"""Versioned successful-trajectory memory for task retrieval and M2.

This is an ExpeL retrieval-only adaptation: it reuses successful training
trajectories, but does not reproduce ExpeL's reflection retries, insight
extraction, manual ReAct examples, or GPT-based policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .analysis import load_run
from .common import canonical_digest, read_json, write_json_atomic
from .memory import build_corpus


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
POLICY = "successful_trajectory_task_or_proposal_query_v1"


def task_key(benchmark: str, task_prompt: str) -> str:
    if benchmark == "textcraft":
        if "Goal: craft " not in task_prompt:
            raise ValueError("TextCraft memory task is missing its goal")
        return "craft " + task_prompt.rsplit("Goal: craft ", 1)[1].splitlines()[0]
    if benchmark == "alfworld":
        return task_prompt.strip()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def build_trajectory_corpus(run_dirs: list[str | Path], excluded_manifests: list[str | Path]) -> dict:
    # Reuse the audited training-only task and content-hash separation checks.
    checked = build_corpus(run_dirs, excluded_manifests)
    records = []
    for directory in run_dirs:
        metadata, _, manifest, episodes = load_run(directory)
        for task in manifest["tasks"]:
            episode = episodes[task["task_id"]]
            if not episode["success"]:
                continue
            steps = [{"observation": step["input_observation"],
                      "action": step["action"], "feedback": step["output_observation"],
                      "failed_action": bool(step["outcome"].get("failed_action", False))}
                     for step in episode["trace"] if "outcome" in step]
            if not steps:
                continue
            records.append({"memory_id": canonical_digest([metadata["run_id"], task["task_id"],
                                                            "successful_trajectory_v1"])[:20],
                            "benchmark": manifest["benchmark"], "task_id": task["task_id"],
                            "task_fingerprint": task["game_sha256" if manifest["benchmark"] == "alfworld"
                                                     else "observation_sha256"],
                            "source_run_id": metadata["run_id"],
                            "task_prompt": task_key(manifest["benchmark"], episode["task_prompt"]),
                            "initial_observation": episode["initial_observation"],
                            "steps": steps})
    if not records:
        raise ValueError("Training runs contain no successful trajectories")
    return {"schema_version": 1, "kind": "successful_trajectory_retrieval_only",
            "source_runs": checked["source_runs"],
            "excluded_manifest_sha256": checked["excluded_manifest_sha256"],
            "records": records}


class MPNetEmbedder:
    """Use the SentenceTransformers pooling configured for ExpeL's MPNet model."""

    def __init__(self, model_id: str = EMBEDDING_MODEL, revision: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.model = SentenceTransformer(model_id, revision=revision, device="cpu")
        self.revision = revision or getattr(self.model[0].auto_model.config, "_commit_hash", None)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def build_trajectory_index(corpus: dict, embedder: MPNetEmbedder) -> dict:
    records = corpus["records"]
    step_items = [(record["memory_id"], record["task_prompt"], step)
                  for record in records for step in record["steps"] if not step["failed_action"]]
    if not step_items:
        raise ValueError("Successful trajectories contain no accepted actions")
    task_texts = ["Task: " + record["task_prompt"] for record in records]
    step_texts = [f"Task: {task}\nObservation: {step['observation'][:300]}\nAction: {step['action']}"
                  for _, task, step in step_items]
    return {"schema_version": 2, "embedding_model": embedder.model_id,
            "embedding_revision": embedder.revision,
            "corpus_sha256": canonical_digest(corpus),
            "memory_ids": [record["memory_id"] for record in records],
            "task_vectors": embedder.encode(task_texts),
            "step_memory_ids": [memory_id for memory_id, _, _ in step_items],
            "step_vectors": embedder.encode(step_texts)}


def render_trajectory(record: dict) -> str:
    lines = [f"Past successful task: {record['task_prompt']}"]
    # Keep the source trajectory intact for audit, but do not teach the agent
    # rejected commands from successful episodes (common in TextCraft).
    accepted = [step for step in record["steps"] if not step["failed_action"]]
    for number, step in enumerate(accepted, 1):
        lines.append(f"{number}. Action: {step['action']}")
        lines.append(f"   Feedback: {step['feedback'].strip()[:200]}")
    return "\n".join(lines)


def memory_messages(base_messages: list[dict], hits: list[dict], proposed_action: str | None = None) -> list[dict]:
    if not hits:
        return base_messages
    lines = ["Past successful training tasks (examples; adapt them to the current state and available actions):"]
    for hit in hits:
        lines.append(hit["content"])
    if proposed_action is not None:
        lines.append(f"Your proposed action was: {proposed_action}")
    lines.append("Choose one action for the current task using the required output format.")
    messages = [dict(message) for message in base_messages]
    content = messages[-1]["content"]
    guidance = "\n\n".join(lines)
    # Preserve the baseline's final answer cue. TextCraft expects the prompt to
    # end in Action:, and ALFWorld expects a numbered available action.
    if content.endswith("\nAction:"):
        messages[-1]["content"] = content[:-len("\nAction:")] + "\n\n" + guidance + "\n\nAction:"
    elif content.endswith("Choose the best next action."):
        cue = "Choose the best next action."
        messages[-1]["content"] = content[:-len(cue)] + guidance + "\n\n" + cue
    else:
        messages[-1]["content"] = content + "\n\n" + guidance
    return messages


class TrajectoryMemory:
    def __init__(self, corpus: dict, index: dict, embedder: MPNetEmbedder,
                 top_k: int, token_budget: int, token_counter):
        if top_k < 1 or token_budget < 1:
            raise ValueError("Retrieval count and token budget must be positive")
        if (index["corpus_sha256"] != canonical_digest(corpus)
                or index["embedding_model"] != embedder.model_id
                or index.get("embedding_revision") != embedder.revision):
            raise ValueError("Trajectory index does not match corpus or embedding model")
        if index["memory_ids"] != [record["memory_id"] for record in corpus["records"]]:
            raise ValueError("Trajectory index order does not match corpus")
        if (index.get("schema_version") != 2
                or len(index["task_vectors"]) != len(corpus["records"])
                or not index["task_vectors"]
                or len(index["step_vectors"]) != len(index["step_memory_ids"])
                or not index["step_vectors"]):
            raise ValueError("Trajectory vectors are missing")
        expected_step_ids = [record["memory_id"] for record in corpus["records"]
                             for step in record["steps"] if not step["failed_action"]]
        if index["step_memory_ids"] != expected_step_ids:
            raise ValueError("Trajectory step vector order does not match corpus")
        width = len(index["task_vectors"][0])
        if not width or any(len(vector) != width for vector in index["task_vectors"] + index["step_vectors"]):
            raise ValueError("Trajectory vectors have inconsistent dimensions")
        self.records = corpus["records"]
        self.task_vectors = index["task_vectors"]
        self.step_vectors = index["step_vectors"]
        self.step_memory_ids = index["step_memory_ids"]
        self.embedder = embedder
        self.top_k = top_k
        self.token_budget = token_budget
        self.token_counter = token_counter

    def retrieve(self, benchmark: str, task_prompt: str, observation: str | None = None,
                 proposed_action: str | None = None) -> tuple[list[dict], float]:
        started = time.perf_counter()
        task_query = "Task: " + task_key(benchmark, task_prompt)
        queries = [task_query]
        if proposed_action is not None:
            queries.append(f"{task_query}\nObservation: {(observation or '')[:300]}\nAction: {proposed_action}")
        query_vectors = self.embedder.encode(queries)
        if len(query_vectors) != len(queries) or any(len(vector) != len(self.task_vectors[0]) for vector in query_vectors):
            raise ValueError("Trajectory query vector dimension differs from index")
        candidates = [i for i, record in enumerate(self.records) if record["benchmark"] == benchmark]
        task_scores = [sum(a * b for a, b in zip(query_vectors[0], stored))
                       for stored in self.task_vectors]
        scores = task_scores[:]
        if proposed_action is not None:
            step_scores = {}
            for memory_id, stored in zip(self.step_memory_ids, self.step_vectors):
                similarity = sum(a * b for a, b in zip(query_vectors[1], stored))
                step_scores[memory_id] = max(similarity, step_scores.get(memory_id, float("-inf")))
            scores = [0.4 * task_scores[i] + 0.6 * step_scores.get(record["memory_id"], -1.0)
                      for i, record in enumerate(self.records)]
        ranked = sorted(candidates, key=lambda i: (-scores[i], self.records[i]["memory_id"]))
        hits = []
        used_tokens = 0
        for i in ranked:
            content = render_trajectory(self.records[i])
            tokens = self.token_counter(content)
            if used_tokens + tokens > self.token_budget:
                continue
            score = scores[i]
            hits.append({"memory_id": self.records[i]["memory_id"],
                         "task_id": self.records[i]["task_id"],
                         "similarity": round(float(score), 6),
                         "tokens": tokens, "content": content})
            used_tokens += tokens
            if len(hits) == self.top_k:
                break
        return hits, round(time.perf_counter() - started, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--exclude-manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    corpus = build_trajectory_corpus(args.run, args.exclude_manifest)
    output = Path(args.output_dir)
    corpus_path = output / "corpus.json"
    if corpus_path.exists() and read_json(corpus_path) != corpus:
        raise FileExistsError(f"Existing trajectory corpus differs: {corpus_path}")
    if not corpus_path.exists():
        write_json_atomic(corpus_path, corpus)
    index = build_trajectory_index(corpus, MPNetEmbedder())
    index_path = output / "index.json"
    if index_path.exists() and read_json(index_path) != index:
        raise FileExistsError(f"Existing trajectory index differs: {index_path}")
    if not index_path.exists():
        write_json_atomic(index_path, index)
    print(json.dumps({"successful_trajectories": len(corpus["records"]),
                      "source_runs": corpus["source_runs"],
                      "corpus_sha256": index["corpus_sha256"],
                      "output_dir": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
