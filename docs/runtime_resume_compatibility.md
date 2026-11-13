# Runtime metadata on checkpoint resume

The runner keeps the first accepted `model_metadata` unchanged and writes a
`runtime_attempts` entry for every model load that yields metadata. Each entry
contains the complete metadata, whether this is a resume, the time, and a
field-level compatibility result. A rejected resume records the current
metadata and previous/current values before raising. Completed episode files
and checkpoint hashes remain untouched.

The check normalizes both metadata objects through JSON before comparison.
Every field still matches exactly except these allocation observations:

- `gpu_total_memory_bytes` may differ by at most 1 GiB. A larger change is
  rejected and reported.
- Scalar `pid`, `process_id`, `worker_pid`, `gpu_uuid`, `gpu_device_uuid`, and
  `gpu_pci_bus_id` values may change when present on both attempts. Current
  standalone and hybrid model metadata do not contain those identifiers.

The same recursive comparison covers standalone metadata and metadata nested
under each hybrid worker. Model and tokenizer revisions, template hash and
prompt settings, decoding, quantization and packed-weight checks, backend and
dependency versions, worker interpreter/cache paths, GPU name and compute
capability, and all unlisted fields remain exact. Qwen metadata now records
compute capability and total GPU memory explicitly. The actual difference in
job 146453 cannot be reconstructed because its failed resume did not save the
second metadata object. Its old run must not be resumed across this commit.

## Targeted Locust validation before the matched suite

Push the new commit, make a clean checkout, and run the CPU suite there. Keep
the previously successful CUDA/Triton environment and shared model caches.
Use a **new** output root for each validation run. Do not modify old artifacts.

1. Run the weight-free pinned-tokenizer check from
   [the token-reservation instructions](gpt_oss_token_reservation_fix.md).
2. Run a fresh five-task GPT-OSS standalone development job on one L40S. Check
   `run.json` contains one accepted `runtime_attempts` entry and that all saved
   episode hashes match `checkpoint.json`.
3. Run a fresh five-task hybrid random development job on one L40S. Inspect
   task five's `generation_attempts`, `support_events`, and reservation fields.
   Confirm at least one delegated call occurs after several earlier decisions;
   if not, inspect further development tasks or the existing failure-triggered
   configuration. Do not alter the selection policies for this check.
4. After at least one episode is saved in a fresh validation job, request a
   Slurm requeue of that **validation job only**. Check the same run key resumes,
   `runtime_attempts` has both complete metadata objects and an accepted
   compatibility result, and saved episode bytes/checkpoint hashes are
   unchanged. A different L40S node is the strongest cross-allocation check.
5. Inspect both standalone and hybrid `runtime_attempts`. If an allocation
   difference exceeds the explicit allowance or a semantic field changes,
   stop and inspect the persisted field-level diagnostics before the full suite.

The existing [hybrid submission procedure](stronger_model_delegation.md) has
the environment variables and one-L40S Slurm commands. Set
`AGENT_SUPPORT_LIMIT=5` and use unique validation output roots for steps 2–3;
unset it for the four subsequent matched 20-task runs. Fresh matched runs
must all use the final corrected commit and separate output roots. GPT-OSS
sampling is reseeded by episode on restart; bitwise trajectory replay across
processes is not promised.
