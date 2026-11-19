# CPU preflight — 2026-11-18 (America/New_York)

Repository checkout: `/Users/evanpan/AgenticSDS`. No GPU or Locust job was run.

- `python3 -m pytest -q tests`: **153 passed**.
- `bash -n` on both new Slurm scripts: passed.
- `git diff --check`: passed.
- `python3 -m compileall -q` on both new Python modules: passed.
- Both control configs resolve against the unchanged v3 GPT-OSS worker specification and Qwen baseline settings.
- Frozen 50-task manifest SHA-256: `614de2034ad3fa14063a9fb5d6013a6754212584020b3e994a7cdb65f5b84c88`.
- Recipe-order SHA-256: `5d3c08b0404cf0d10571f2ae31edd6375facbc51928ff719ed36106837374c8c`.
- `.venv` has TextCraft `0.0.3`.
- With `PYTHONHASHSEED=0`, a CPU stub drove the real development-task TextCraft adapter through `inventory` (1 executed, 0 failed) and an unavailable craft recipe (1 executed, 1 failed). Both traces recorded 1 model call and 0 external support operations.

| Arm | Config SHA-256 | Local runtime checks |
|---|---|---|
| GPT-OSS standalone v1 | `cc1130021f1ff1b79bdf81224a7aa47a0fb819d8a4f564df1bc8b6ea271a7f62` | TextCraft true; GPT-OSS executable/cache false |
| GPT-OSS review-only v1 | `20fc7ca885c09580676fdccfdb9f32e20f5118b7e21417d0a2b621079c85bb0c` | TextCraft true; GPT-OSS executable/cache and Qwen dependencies false |

The false checks reflect this Mac's CPU-only local environments. On Locust, `init` refuses to write a plan until all runtime checks are true. The development smoke jobs then test the pinned GPT-OSS backend and simultaneous model loading on an L40S. No held-out control results or four-arm comparison are available yet.
