# Mac CPU preflight — 2026-11-18 (America/New_York)

Worktree: `/Users/evanpan/AgenticSDS-autonomous-recovery-v4`, branch `autonomous-recovery-v4`, based on committed `gptoss-controls-v1` commit `24723b2`.

- Full CPU suite: **160 passed** (including v4 decision-schema, bounded-state, support-chaining, visible-prerequisite, episode/checkpoint, resume, and paired-analysis tests).
- Both Slurm scripts passed `bash -n`; all three new Python modules passed AST parsing.
- A stubbed GPT-OSS and Qwen drove the real TextCraft 0.0.3 adapter through a native `inventory` action with `PYTHONHASHSEED=0`. The trace recorded one Qwen call, one controller call, one environment action, and zero support operations.
- Frozen 30-task development manifest digest: `150c7bbd973936e0f3ef8fe1266060e88de0181f0068ce98bc7bce0989a2d562`.
- Frozen recipe-order digest: `5d3c08b0404cf0d10571f2ae31edd6375facbc51928ff719ed36106837374c8c`.
- Pinned local M2-200 corpus/index were available, with **zero task-ID or observation-hash overlap** with the 30 selected tasks.

| Arm | Config SHA-256 | Local runtime checks |
|---|---|---|
| Expanded v3 v1 | `5f769de449fb064e4a034044b2cc7a58b3507096e2acf7bcd33e8fe9733c4cf3` | TextCraft 0.0.3 true; Qwen packages, GPT-OSS executable/cache false |
| Autonomous Recovery v4 | `a8e57f46a11646aaaac5b50b3cc386c6770e77ff1ad557f7115de03ff4885d1e` | TextCraft 0.0.3 true; Qwen packages, GPT-OSS executable/cache false |

The false checks reflect the Mac's CPU-only environment. Locust preflight must show all checks true and disjoint memory before plan initialization. No actual Qwen/GPT-OSS inference, Slurm job, or 30-task evaluation has run from this worktree. No commit or push was made.
