# CPU-only evaluation integration check — 2026-11-23

The three `recovery_policy_eval preflight --policy ...` checks completed on the
Mac without model inference or GPU work. Each selected the intended registry
entry, validated the frozen 30-task manifest and v4 controller protocol, and
found the pinned M2-200 corpus/index with zero task overlap. All three used
the same v4 config digest
`a8e57f46a11646aaaac5b50b3cc386c6770e77ff1ad557f7115de03ff4885d1e`
and manifest digest
`150c7bbd973936e0f3ef8fe1266060e88de0181f0068ce98bc7bce0989a2d562`.

| Policy ID | Policy configuration SHA-256 | Selector/protocol/memory checks |
| --- | --- | --- |
| `original_v4` | `3cbd31d336be037890248d921527c043c829f6fafbd469b14447d8de0a35894d` | Passed |
| `state_aware_v1` | `b0f4a2a0bb4daf77b74a4464616107b5e3e69d8170c3602f16e38f8877bcb314` | Passed |
| `accuracy_first_v1` | `14a19273a3afb2b04b589610014d7981a9fb355d2071dbef964b1150c7ad4733` | Passed |

Mac runtime checks: TextCraft 0.0.3 true; Qwen GPU dependencies, GPT-OSS
Python path, and GPT-OSS cache false. These must all be true in the pinned
Locust environment before stage execution. CPU preflight does not test CUDA,
the controller worker, or benchmark success.

Offline suite: **238 passed**. Both new Slurm scripts passed `bash -n`;
`git diff --check` passed. No GPU job was launched.
