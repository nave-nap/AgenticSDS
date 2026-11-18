"""Read-only, controller-queryable adapter over frozen M2 trajectory memory."""

from __future__ import annotations

import time

from .common import canonical_digest
from .trajectory_memory import TrajectoryMemory, render_trajectory


class FrozenTrajectoryRetrieval:
    """Reuse TrajectoryMemory's index validation and embeddings with a free query."""

    def __init__(self, corpus: dict, index: dict, embedder, top_k: int,
                 token_budget: int, token_counter):
        self.memory = TrajectoryMemory(corpus, index, embedder, top_k,
                                       token_budget, token_counter)
        self.max_top_k = top_k
        self.max_example_tokens = token_budget
        self.corpus_sha256 = canonical_digest(corpus)
        self.index_sha256 = canonical_digest(index)

    def search(self, benchmark: str, query: str, scope: str, top_k: int) -> tuple[list[dict], float]:
        if scope not in {"task", "steps", "both"} or not 1 <= top_k <= self.max_top_k:
            raise ValueError("Invalid retrieval scope or count")
        started = time.perf_counter()
        encoded = self.memory.embedder.encode([query])
        if len(encoded) != 1 or len(encoded[0]) != len(self.memory.task_vectors[0]):
            raise ValueError("Memory query vector dimension differs from frozen index")
        vector = encoded[0]
        task_scores = [sum(a * b for a, b in zip(vector, stored))
                       for stored in self.memory.task_vectors]
        step_scores = {}
        for memory_id, stored in zip(self.memory.step_memory_ids, self.memory.step_vectors):
            similarity = sum(a * b for a, b in zip(vector, stored))
            step_scores[memory_id] = max(similarity, step_scores.get(memory_id, float("-inf")))
        candidates = []
        for index, record in enumerate(self.memory.records):
            if record["benchmark"] != benchmark:
                continue
            task_score = task_scores[index]
            step_score = step_scores.get(record["memory_id"])
            score = (task_score if scope == "task" else
                     step_score if scope == "steps" else
                     0.4 * task_score + 0.6 * (step_score if step_score is not None else -1.0))
            if score is not None:
                candidates.append((index, score, task_score, step_score))
        candidates.sort(key=lambda item: (-item[1], self.memory.records[item[0]]["memory_id"]))
        hits = []
        used_tokens = 0
        for index, score, task_score, step_score in candidates:
            record = self.memory.records[index]
            content = render_trajectory(record)
            tokens = self.memory.token_counter(content)
            if tokens < 0 or used_tokens + tokens > self.memory.token_budget:
                continue
            hits.append({"memory_id": record["memory_id"],
                         "source_run_id": record["source_run_id"],
                         "source_task_id": record["task_id"],
                         "task_prompt": record["task_prompt"],
                         "similarity": round(float(score), 6),
                         "task_similarity": round(float(task_score), 6),
                         "step_similarity": (round(float(step_score), 6)
                                             if step_score is not None else None),
                         "content": content, "tokens": tokens})
            used_tokens += tokens
            if len(hits) == top_k:
                break
        return hits, time.perf_counter() - started
