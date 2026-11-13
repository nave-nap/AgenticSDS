# GPT-OSS delegation token reservation check

`GptOss20BModel.count_messages_tokens` now measures `input_ids` from the tensor
encoding used by generation. Transformers 5 returns a mapping from
`apply_chat_template()` by default, so `len()` on the old non-tensor result
counted mapping fields. The runner records the input estimate and selected
model's generation limit with each delegation reservation; a genuine mismatch
still aborts with the estimated and measured input/output token counts.

Before any GPU job, verify the cached pinned tokenizer on Locust with no model
weights loaded. Run this from a clean checkout of the fix branch:

```bash
cd "$HOME/AgenticSDS-hybrid-fix"
export AGENT_SUPPORT_GPTOSS_PYTHON="/shared_data0/$USER/agent_support/venvs/gptoss20b/bin/python"
export HF_HOME="/shared_data0/$USER/agent_support/hf_cache"
"$AGENT_SUPPORT_GPTOSS_PYTHON" -m scripts.check_gpt_oss_token_count
```

The check requires Transformers 5.22.2, the pinned model revision, the same
Harmony template hash as the prior successful standalone run, prompt date,
reasoning effort, generation prompt, and special tokens. It prints the old and
correct reservations for short and long histories. It uses `local_files_only`
and does not download weights.

To audit the four saved episodes from the failed random run, use its existing
run directory. This reads artifacts only:

```bash
cd "$HOME/AgenticSDS-hybrid-fix"
export AGENT_SUPPORT_QWEN_PYTHON="$HOME/AgenticSDS/.venv/bin/python"
export FAILED_RANDOM_RUN="/absolute/path/to/original/textcraft_slurm_JOBID"
"$AGENT_SUPPORT_QWEN_PYTHON" -m scripts.audit_hybrid_reservations "$FAILED_RANDOM_RUN"
```

If `support_token_budget` blocks are absent, the old undercount did not affect
selection through the aggregate token cap in those saved episodes. The failed
fifth episode has no immutable episode trace, so its unrecorded decisions
cannot be audited from the four completed episodes.

Start subsequent runs in new output roots, with the same pinned development
manifest, seeds, model revisions, worker environments, and four policy configs.
Use one L40S at a time. For a final paired report, rerun both standalone
references and both hybrid policies on the fix commit so the report's commit
integrity check passes. Keep the failed random run as a separate artifact.
The existing resume check requires exact configuration, provenance, and worker
model metadata; do not resume the old failed run across this code change.
