# Artifact contracts (pre-inference implementation contract)

All JSON objects include schema_version=1. Reject NaN/Infinity and unknown scientific configuration keys; numeric metrics use finite values or explicit null plus reason. UTF-8 sorted-key JSON for hashing. Each artifact manifest records file byte lengths and SHA-256, never self-hashes recursively.

## DATA_FREEZE.json

Fields: protocol_sha256, source_sha256, candidates_sha256, exclusions_sha256, tokenizer_files[{relative_name,sha256,bytes}], selection_policy, local_freeze_utc, published_before_freeze=false unless verified, model_inference_performed=false, candidate_count, eligible_count, selected_count=64, rejected[{word,reason}], rows_sha256.

## dataset.json

Fields: schema_version, task_id, prompt_template, rows[64]. Each row has row_id, pair_id, orientation (0/1), recipient_word, donor_word, recipient_token_id, donor_token_id, recipient_prefix_ids[int], donor_row_id, source_position, final_position, negative_position, prefix_length. Pair assertions: two orientations are exact inverses; same positions, all but source ID equal; distinct words and answer IDs. Word IDs never mistaken for position indices.

## RUN_BUDGET.json and run_metadata.json

Budget: max_attempted_forwards=2816, baseline_forward_cap=128, patch_forward_cap=2688, baseline_wall_seconds=300, patch_wall_seconds=2100, execution_wall_seconds=2400, termination_grace_seconds=15, max_rss_bytes=3221225472, max_output_bytes=1073741824, max_prefix_tokens=32, cpu_threads=2, paid_compute_limit=0, frozen_utc, protocol_sha256.

Metadata: model_repo, revision, model_files[{name,sha256,bytes}], config_sha256, source_sha256, data_freeze_sha256, runtime_versions, hardware_description, numerical_settings, seed, start/end_utc, stage, upstream_baseline_manifest_sha256, wall_seconds, max_rss_observed, enforcement_details. Gate and resource stop reasons are explicit.

## attempts.jsonl

Fields: attempt_index, stage, row_id, pair_id, orientation, block (null for baseline), condition, started_utc, completed_utc|null, status in planned/completed/failed/uncertain, elapsed_seconds|null, output_file|null, error_type|null. Append events with same attempt_index if needed, never overwrite earlier events; audit reconstructs state transitions and counts unique attempted IDs. No hidden retries.

## outcomes.jsonl

Fields: attempt_index, row_id, pair_id, orientation, condition, block|null, patched_positions[int]|"all", donor_row_id|null, logits_file, logits_sha256, logits_dtype=float32, logits_shape=[49152], baseline_logits_sha256, donor_logits_sha256, recipient_logit, donor_logit, recipient_probability, donor_probability, recipient_margin, donor_margin, raw_transfer_delta, argmax_token_id, argmax_logit, greedy_recipient, greedy_donor, pairwise_donor_positive, exact_pairwise_tie, off_target, genuine_switch, off_target_move, kl_baseline_to_patch, residual_delta_norm, recipient_residual_norm, residual_norm_ratio, expected_identity in recipient/donor/none, identity_max_abs_error|null, finite=true, elapsed_seconds. Scores are independently recomputable from vectors. Norms must be independently recomputable from baseline cached patch-site states and declared positions.

Baseline rows use explicit nullable patch fields. Capture-check rows reference untouched logits and contain block_state_files[7], state_shapes and hashes. Raw state arrays shape [prefix_length,576]. No output label or answer token is fed into the model prefix.

## gate.json

Fields: total_rows=64, total_pairs=32, greedy_correct_count, positive_margin_count, jointly_competent_pair_count, readonly_identity_pass_count, failed_row_ids, failed_pair_ids, numeric_failure_ids, thresholds, passed, complete, raw_artifact_manifest_sha256. A failed gate still has all baseline outcomes unless technical interruption prevented them; incomplete never counts as pass.

## summary.json and audit.json

Summary has complete/partial flag, expected/observed cell counts, all 42 cells including controls, pair-level outcomes for all 32 groups, all denominators, pointwise intervals, 12-cell simultaneous raw-Δ bands, control_error_maxima, normalized_eligible_pair_ids, raw negative/overshoot counts, missing cells, exclusions with reasons (only prespecified normalized secondary exclusions), bootstrap seed/replications, method caveats.

Audit has source/data/model hash results, coverage checks, numerical identity results, independently reconstructed metric errors, gate reconstruction, group averaging checks, sample-size accounting, budget/attempt accounting, missing/corrupt files, passed boolean. Missing/corrupt vectors invalidate independent audit, even if scalar summaries appear plausible.
