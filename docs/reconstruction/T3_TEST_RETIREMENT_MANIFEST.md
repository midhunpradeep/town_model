# T3 Test Retirement Manifest

**Starting accepted T2 SHA:** `c6aad87a45ac75835f02670f32fdeca5167f18f7`
**Accepted historical implementation containing the original assertions:** [`2de9c5f29b436f727720e87132d0b4be25a40768`](https://github.com/midhunpradeep/town_model/tree/remediation/phase-3-last-two-findings)
**T2 full-suite baseline:** 191 passed, 132 failed, 0 skipped.

## Scope and disposition

This manifest records the exact original pytest node IDs or parameterized node groups whose historical classifier/math/execution assertions were retired, split, or replaced during T3. It includes all 132 node IDs reported failing at accepted T2 HEAD, plus passing legacy assertions that depended on now-removed classifier/objective behavior and obsolete direct-class T2 rejection tests. The enumerated baseline node list is preserved below so the evidence is reproducible; original test code remains recoverable from the accepted historical revision.

The current replacement tests do **not** validate a replacement classifier. They cover only supported independent boundaries: image-to-point projections, dataset/ontology targets and masks, caller-supplied value reducers, cache/integrity, historical metadata/snapshot reading, atomic generic storage, and explicit unavailable-operation behavior.

Test collection reconciles as follows: baseline 323 nodes; current suite 180 nodes. **147 original node IDs remain unchanged**, **6 original IDs were retargeted/split under the same name**, and **27 new node IDs were added**. The 176 manifest entries account for every other original ID plus the six retargeted cases.

Independent ontology assertions were preserved:

- `tests/test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations` now ends after asserted/entailed/unknown metadata and canonical snapshot class-ID checks; its old region query and geometry publication portion were retired.
- `tests/test_phase3_d5a_ontology_entailments.py::test_geometric_violation_does_not_change_ontology_entailment` was split. The old geometry query was retired; the entailment assertion remains in `test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry`.

The manifest accounts for **176 original node IDs** across **129 test groups**. By group: 68 deleted, 3 split, and 58 replaced. By node: 96 deleted, 3 split, and 77 replaced. This includes the 132 T2 full-suite failures plus 44 previously passing or mixed-purpose assertions that no longer have an active implementation. See the JSON records below for each node, behavior, rationale, retained assertions, and replacement references.

## Machine-readable per-node record

<!-- T3_RETIREMENT_DATA_START -->
```json
{
  "starting_t2_revision": "c6aad87a45ac75835f02670f32fdeca5167f18f7",
  "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768",
  "baseline_full_suite": {
    "passed": 191,
    "failed": 132,
    "skipped": 0
  },
  "baseline_failed_node_ids": [
    "tests/test_classifier_teardown_boundary.py::test_classifier_free_points_match_compatibility_adapter_exactly",
    "tests/test_dashboard_app.py::test_dashboard_image_embeddings_inspects_a_saved_run_without_loading_checkpoints",
    "tests/test_dashboard_app.py::test_dashboard_image_embeddings_page_loads_without_training_or_checkpoints",
    "tests/test_dashboard_app.py::test_image_embeddings_displays_worker_subprocess_failure_and_captured_log",
    "tests/test_dashboard_app.py::test_image_embeddings_launch_button_uses_validated_detached_worker_service",
    "tests/test_dashboard_app.py::test_image_embeddings_preflight_rejects_invalid_draft_without_creating_run_artifacts",
    "tests/test_phase2_artifact_errors.py::test_cli_and_resume_report_corruption_as_actionable_errors",
    "tests/test_phase2_artifact_errors.py::test_dashboard_reader_reports_corrupt_checkpoint_as_failed_run",
    "tests/test_phase2_artifact_errors.py::test_geometry_laboratory_app_test_survives_corrupt_checkpoint",
    "tests/test_phase2_artifact_errors.py::test_invalid_and_truncated_checkpoints_raise_actionable_artifact_errors",
    "tests/test_phase2_artifact_errors.py::test_structurally_incompatible_checkpoint_is_rejected_without_losing_integrity_checks",
    "tests/test_phase2_cli.py::test_cli_mapping_command_creates_all_model_family_runs",
    "tests/test_phase2_cli.py::test_cli_runs_both_fixed_fixtures_and_evaluates_inspects_exports",
    "tests/test_phase2_dashboard_geometry.py::test_high_dimensional_view_uses_actual_sample_anchor_and_explicit_slice",
    "tests/test_phase2_dashboard_geometry.py::test_historical_snapshot_without_optional_activity_remains_readable",
    "tests/test_phase2_dashboard_geometry.py::test_invalid_or_missing_runs_are_reported_without_crashing",
    "tests/test_phase2_dashboard_geometry.py::test_saved_run_discovery_selection_region_metrics_and_comparison",
    "tests/test_phase2_dashboard_geometry.py::test_snapshot_activity_tracks_initial_intermediate_and_final_geometry",
    "tests/test_phase2_dashboard_geometry.py::test_streamlit_geometry_laboratory_loads_saved_runs_without_retraining",
    "tests/test_phase2_dashboard_geometry.py::test_town_plot_keeps_overlap_reinclusion_and_window_boundaries_faithful[2-axes0]",
    "tests/test_phase2_dashboard_geometry.py::test_town_plot_keeps_overlap_reinclusion_and_window_boundaries_faithful[3-axes1]",
    "tests/test_phase2_experiments.py::test_all_three_fixed_classifiers_train_reload_and_share_identical_fixture",
    "tests/test_phase2_experiments.py::test_missing_snapshot_or_unknown_split_fails_clearly",
    "tests/test_phase2_experiments.py::test_trainable_mapping_has_matched_initial_encoders_and_gradient_paths",
    "tests/test_phase2_geometry.py::test_box_closed_boundaries_margin_and_class_shapes",
    "tests/test_phase2_geometry.py::test_closed_window_boundary_is_excluded_even_when_margin_is_zero",
    "tests/test_phase2_geometry.py::test_contained_town_geometry_extremes_and_gradients",
    "tests/test_phase2_geometry.py::test_dimension_aware_initialization_is_seeded_and_diverse",
    "tests/test_phase2_geometry.py::test_empty_batches_chunked_equivalence_and_validation",
    "tests/test_phase2_geometry.py::test_extreme_float32_raw_parameters_remain_valid_and_seeded",
    "tests/test_phase2_geometry.py::test_geometry_initialization_is_label_agnostic_and_houses_differ",
    "tests/test_phase2_geometry.py::test_minimum_width_feasibility_and_float32_physical_invariants",
    "tests/test_phase2_geometry.py::test_multiple_windows_and_classes_follow_literal_boolean_set_operations",
    "tests/test_phase2_geometry.py::test_physical_box_helper_rejects_width_below_configured_minimum",
    "tests/test_phase2_geometry.py::test_target_volume_initialization_materially_improves_high_dimensional_window_activity[16]",
    "tests/test_phase2_geometry.py::test_target_volume_initialization_materially_improves_high_dimensional_window_activity[2]",
    "tests/test_phase2_geometry.py::test_target_volume_initialization_materially_improves_high_dimensional_window_activity[8]",
    "tests/test_phase2_geometry.py::test_window_boundary_exclusion_and_other_house_reinclusion",
    "tests/test_phase2_gradients.py::test_high_dimensional_geometry_hinge_has_finite_loss_and_gradients",
    "tests/test_phase2_gradients.py::test_positive_inside_window_activates_window_branch_and_negative_inside_house_activates_frame",
    "tests/test_phase2_gradients.py::test_positive_outside_and_negative_inside_box_have_point_and_geometry_gradients",
    "tests/test_phase2_min_width_preflight.py::test_box_and_town_validate_their_actual_window_parameterization",
    "tests/test_phase2_min_width_preflight.py::test_composed_config_rejects_infeasible_width_before_fixture_training_or_artifacts",
    "tests/test_phase2_min_width_preflight.py::test_default_small_and_composed_town_boundary_widths_validate",
    "tests/test_phase2_min_width_preflight.py::test_geometry_and_composed_config_agree_at_family_feasibility_limits",
    "tests/test_phase2_min_width_preflight.py::test_invalid_min_width_values_are_rejected_by_config_and_constructors",
    "tests/test_phase2_models_losses.py::test_baseline_zero_logit_threshold_is_inclusive_and_bce_is_unreduced",
    "tests/test_phase2_models_losses.py::test_classifiers_share_order_shapes_fixed_domainthing_and_dtype[BoxClassifier]",
    "tests/test_phase2_models_losses.py::test_classifiers_share_order_shapes_fixed_domainthing_and_dtype[NeuralBaseline]",
    "tests/test_phase2_models_losses.py::test_classifiers_share_order_shapes_fixed_domainthing_and_dtype[TownClassifier]",
    "tests/test_phase2_models_losses.py::test_encoder_gradients_flow_and_points_remain_bounded",
    "tests/test_phase2_models_losses.py::test_signed_distance_hinge_shared_for_box_and_town_and_validates_margin",
    "tests/test_phase2_resume.py::test_incompatible_resolved_config_is_rejected_on_checkpoint_reload",
    "tests/test_phase2_resume.py::test_modified_fixture_arrays_are_rejected_on_reload",
    "tests/test_phase2_resume.py::test_resume_restores_adam_and_appends_steps_without_mutating_resolved_config",
    "tests/test_phase2_resume.py::test_schema_one_checkpoint_without_window_policy_remains_loadable",
    "tests/test_phase2_resume.py::test_zero_margin_split_resume_matches_uninterrupted_run_and_keeps_identity",
    "tests/test_phase2_resume.py::test_zero_margin_training_reload_and_default_margin",
    "tests/test_phase3_cache.py::test_frozen_feature_cache_does_not_change_training_trajectory",
    "tests/test_phase3_cli.py::test_image_cli_runs_and_inspects_an_immutable_smoke_configuration",
    "tests/test_phase3_d5a_ontology_entailments.py::test_geometric_violation_does_not_change_ontology_entailment",
    "tests/test_phase3_d5b_relation_publication.py::test_cli_surfaces_actionable_post_publication_failure",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[compositional-baseline]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[compositional-box]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[compositional-town]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[global-baseline]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[global-box]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[global-town]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[hybrid-baseline]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[hybrid-box]",
    "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[hybrid-town]",
    "tests/test_phase3_models.py::test_common_neural_modules_are_byte_identical_and_classifiers_are_independent",
    "tests/test_phase3_models.py::test_component_classifier_reshaping_preserves_box_parameters_and_reshapes_town_activity",
    "tests/test_phase3_models.py::test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact",
    "tests/test_phase3_models.py::test_hand_computed_known_mask_positive_weight_and_per_image_component_reduction",
    "tests/test_phase3_models.py::test_hybrid_pizza_supervision_alone_reaches_deepsets_and_component_projection",
    "tests/test_phase3_models.py::test_image_objective_masks_unknowns_uses_positive_weights_and_omits_empty_components",
    "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[16]",
    "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[2]",
    "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[5]",
    "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[8]",
    "tests/test_phase3_models.py::test_old_snapshot_without_explicit_diagnostic_axes_is_rejected_clearly",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[compositional-baseline]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[compositional-box]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[compositional-town]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[global-baseline]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[global-box]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[global-town]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[hybrid-baseline]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[hybrid-box]",
    "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[hybrid-town]",
    "tests/test_phase3_models.py::test_snapshot_diagnostic_axes_roundtrip_preserve_class_geometry_and_instance_activity",
    "tests/test_phase3_models.py::test_zero_roi_models_produce_rho_zero_pizza_point",
    "tests/test_phase3_relations.py::test_exact_box_relations_cover_equal_nested_partial_touch_and_disjoint_cases",
    "tests/test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations",
    "tests/test_phase3_relations.py::test_town_membership_respects_overlapping_houses_closed_windows_and_reinclusion",
    "tests/test_phase3_relations.py::test_town_no_sampled_containment_counterexample_is_inconclusive_not_proven",
    "tests/test_phase3_relations.py::test_town_sampling_finds_verified_counterexample_and_is_reproducible",
    "tests/test_phase3_runs.py::test_dashboard_resume_uses_only_compatible_stopped_checkpoint_and_detached_cli",
    "tests/test_phase3_runs.py::test_dashboard_worker_launches_detached_cli_process_and_duplicate_launch_is_rejected",
    "tests/test_phase3_runs.py::test_detached_worker_graceful_stop_and_compatible_resume",
    "tests/test_phase3_runs.py::test_family_matrix_requires_a_strict_comparison_configuration",
    "tests/test_phase3_runs.py::test_forced_worker_death_reconciles_status_preserves_checkpoint_and_resumes",
    "tests/test_phase3_runs.py::test_inspection_uses_live_kernel_lock_not_a_reused_or_stale_manifest_pid",
    "tests/test_phase3_runs.py::test_invalid_configuration_fails_before_any_run_directory_is_created",
    "tests/test_phase3_runs.py::test_invalid_dashboard_launch_fails_preflight_without_creating_output[invalid_config0-GT ROIs only]",
    "tests/test_phase3_runs.py::test_invalid_dashboard_launch_fails_preflight_without_creating_output[invalid_config1-positive integer counts]",
    "tests/test_phase3_runs.py::test_invalid_dashboard_launch_fails_preflight_without_creating_output[invalid_config2-integer feature indices]",
    "tests/test_phase3_runs.py::test_invalid_model_and_vision_preflight_has_no_run_cache_or_cli_side_effects",
    "tests/test_phase3_runs.py::test_run_resolution_persists_original_split_ids_and_ignores_visual_preferences_for_identity",
    "tests/test_phase3_runs.py::test_run_service_lifecycle_stop_and_faithful_resume",
    "tests/test_phase3_runs.py::test_stop_api_rejects_nonrunning_state_and_writes_only_a_graceful_request",
    "tests/test_phase3_runs.py::test_worker_persists_nonzero_subprocess_exit_code_and_command_context",
    "tests/test_phase3_training.py::test_exploratory_training_persists_initial_metrics_in_checkpoint_and_run_result",
    "tests/test_phase3_training.py::test_fixed_visual_comparison_pairs_identical_initial_states_across_classifier_families",
    "tests/test_phase3_training.py::test_invalid_device_fails_before_checkpoint_directory_creation",
    "tests/test_phase3_training.py::test_legacy_checkpoint_objective_and_feature_contract_is_not_silently_resumed",
    "tests/test_phase3_training.py::test_mid_epoch_resume_matches_uninterrupted_training_trajectory",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[compositional-baseline]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[compositional-box]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[compositional-town]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[global-baseline]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[global-box]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[global-town]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[hybrid-baseline]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[hybrid-box]",
    "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[hybrid-town]",
    "tests/test_phase3_training.py::test_one_real_image_optimizer_update_saves_resumable_training_state",
    "tests/test_phase3_training.py::test_snapshot_capture_does_not_change_training_trajectory",
    "tests/test_phase3_training.py::test_trainable_fpn_real_image_run_updates_fpn_and_reports_gradient_activity",
    "tests/test_phase3_training.py::test_training_loop_uses_resolved_positive_weight_vectors_and_versions_checkpoint",
    "tests/test_phase3_weighting.py::test_weighting_semantic_version_rejects_legacy_checkpoint_identity"
  ],
  "accounted_original_node_count": 176,
  "entry_count": 129,
  "disposition_group_counts": {
    "replaced": 58,
    "deleted": 68,
    "split": 3
  },
  "entries": [
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_caller_supplied_objective_terms_still_aggregate_without_scoring"
      ],
      "historical_behavior": "Asserts historical score-to-penalty/objective behavior or tests generic mask/weight/aggregation behavior through the old objective module.",
      "retirement_reason": "Historical score conversion and objective formulas were removed; value-only masked reduction/aggregation is retained without approving an objective.",
      "disposition": "replaced",
      "independent_assertions_retained": "Caller-supplied value reduction, known-mask exclusion, gradients, per-image eligibility, and effective-batch aggregation remain tested.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_generic_weighted_reduction_and_effective_batch_aggregation_remain_independent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_construction_inference_geometry_and_losses_are_unavailable"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_retired_model_modules_and_package_exports_are_absent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_direct_historical_image_penalty_rejects_every_loss_configuration_without_side_effects[baseline-signed_bce]",
        "tests/test_classifier_backend_unavailable.py::test_direct_historical_image_penalty_rejects_every_loss_configuration_without_side_effects[box-signed_bce]",
        "tests/test_classifier_backend_unavailable.py::test_direct_historical_image_penalty_rejects_every_loss_configuration_without_side_effects[box-signed_distance]"
      ],
      "historical_behavior": "Asserts historical behavior `test direct historical image penalty rejects every loss configuration without side effects`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_t3_architecture.py::test_retired_modules_and_historical_executable_definitions_are_absent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_direct_image_objective_rejects_valid_supplied_scores_before_mutation_or_artifacts"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_t3_architecture.py::test_retired_modules_and_historical_executable_definitions_are_absent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_process_imports_and_dashboard_model_pages_do_not_reach_classifiers"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_image_cli_metadata_inspection_remains_read_only"
      ],
      "historical_behavior": "Asserts old Image Embeddings launch/preflight/worker/prediction/geometry controls or successful model execution.",
      "retirement_reason": "Model-dependent controls and recomputation were removed; dataset dashboard pages and safe persisted metadata remain.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset pages remain tested; the model page verifies explicit unavailability and metadata-only reads.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_image_execution_and_checkpoint_entrypoints_fail_before_model_construction"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_image_point_path_and_visual_only_feature_factory_remain_independent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_image_point_path_and_visual_only_feature_extractor_remain_independent"
      ],
      "historical_behavior": "Asserts historical behavior `test image point path and visual only feature extractor remain independent`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_image_point_path_and_visual_only_feature_factory_remain_independent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_image_preflight_validates_without_constructing_or_claiming_a_backend"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments0-synthetic classifier training]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments1-synthetic classifier training]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments10-synthetic classifier training resume]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments11-synthetic classifier geometry export]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments2-image classifier experiment preflight]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments3-image-to-ontology classifier training]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments4-automatic image classifier matrix execution]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments5-image-to-ontology classifier training resume]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments6-historical image classifier checkpoint evaluation]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments7-historical ontology-region relation diagnosis]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments8-control a historical image classifier worker]",
        "tests/test_classifier_backend_unavailable.py::test_model_dependent_cli_commands_fail_nonzero_before_artifacts[arguments9-synthetic classifier evaluation]"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_read_only_run_inspection_and_checksums_remain_available"
      ],
      "historical_behavior": "Asserts historical behavior `test read only run inspection and checksums remain available`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_image_and_synthetic_metadata_inspection_are_read_only",
        "tests/test_classifier_backend_unavailable.py::test_checksum_validated_snapshot_values_remain_readable_without_model"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_rejected_operations_do_not_create_runs_workers_checkpoints_or_exports"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_image_point_path_and_visual_only_feature_factory_remain_independent"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_rejected_resume_and_stop_leave_existing_lifecycle_and_artifacts_unchanged"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_synthetic_training_loading_evaluation_resume_and_geometry_are_unavailable"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_backend_unavailable.py::test_worker_snapshot_and_relation_execution_are_unavailable_before_side_effects"
      ],
      "historical_behavior": "Asserts hard box/town membership, signed-margin, region-relation, or old relation CLI behavior.",
      "retirement_reason": "Those geometric algorithms and duplicate NumPy/Torch membership implementations were removed, not retained behind guards.",
      "disposition": "replaced",
      "independent_assertions_retained": "Generic no-clobber JSON publication and ontology metadata assertions remain tested separately.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_teardown_boundary.py::test_classifier_free_points_match_compatibility_adapter_exactly"
      ],
      "historical_behavior": "Checks the image-to-point projection shapes/values while also constructing the old image classifier adapter or asserting old classifier outputs.",
      "retirement_reason": "The compatibility adapter and classifier outputs were removed; the independent point pathway remains.",
      "disposition": "replaced",
      "independent_assertions_retained": "Point shapes, bounds, sample/component IDs, validity masks, composition mode, and absence of classifier outputs remain tested.",
      "replacement_tests": [
        "tests/test_classifier_teardown_boundary.py::test_image_point_model_preserves_points_and_identity_without_classifier_outputs"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_classifier_teardown_boundary.py::test_dataset_and_point_imports_do_not_load_concrete_classifiers"
      ],
      "historical_behavior": "Checks that importing dataset/config/CLI/dashboard/snapshot/point layers does not eagerly import concrete classifier or geometry implementations.",
      "retirement_reason": "The old import-isolation assertion was expanded from deferred imports to the T3 requirement that historical implementation modules are absent; the original node was replaced with an absence-aware boundary test.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset, configuration, CLI, dashboard, snapshot, and image-point imports remain isolated from model execution.",
      "replacement_tests": [
        "tests/test_classifier_teardown_boundary.py::test_dataset_and_point_imports_do_not_load_or_expose_retired_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_dashboard_app.py::test_dashboard_image_embeddings_inspects_a_saved_run_without_loading_checkpoints"
      ],
      "historical_behavior": "Asserts old Image Embeddings launch/preflight/worker/prediction/geometry controls or successful model execution.",
      "retirement_reason": "Model-dependent controls and recomputation were removed; dataset dashboard pages and safe persisted metadata remain.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset pages remain tested; the model page verifies explicit unavailability and metadata-only reads.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations",
        "tests/test_dashboard_app.py::test_image_embeddings_displays_only_persisted_metadata_and_checksums"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_dashboard_app.py::test_dashboard_image_embeddings_page_loads_without_training_or_checkpoints"
      ],
      "historical_behavior": "Asserts old Image Embeddings launch/preflight/worker/prediction/geometry controls or successful model execution.",
      "retirement_reason": "Model-dependent controls and recomputation were removed; dataset dashboard pages and safe persisted metadata remain.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset pages remain tested; the model page verifies explicit unavailability and metadata-only reads.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations",
        "tests/test_dashboard_app.py::test_image_embeddings_displays_only_persisted_metadata_and_checksums"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_dashboard_app.py::test_image_embeddings_displays_worker_subprocess_failure_and_captured_log"
      ],
      "historical_behavior": "Asserts old Image Embeddings launch/preflight/worker/prediction/geometry controls or successful model execution.",
      "retirement_reason": "Model-dependent controls and recomputation were removed; dataset dashboard pages and safe persisted metadata remain.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset pages remain tested; the model page verifies explicit unavailability and metadata-only reads.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations",
        "tests/test_dashboard_app.py::test_image_embeddings_displays_only_persisted_metadata_and_checksums"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_dashboard_app.py::test_image_embeddings_launch_button_uses_validated_detached_worker_service"
      ],
      "historical_behavior": "Asserts old Image Embeddings launch/preflight/worker/prediction/geometry controls or successful model execution.",
      "retirement_reason": "Model-dependent controls and recomputation were removed; dataset dashboard pages and safe persisted metadata remain.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset pages remain tested; the model page verifies explicit unavailability and metadata-only reads.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations",
        "tests/test_dashboard_app.py::test_image_embeddings_displays_only_persisted_metadata_and_checksums"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_dashboard_app.py::test_image_embeddings_preflight_rejects_invalid_draft_without_creating_run_artifacts"
      ],
      "historical_behavior": "Asserts old Image Embeddings launch/preflight/worker/prediction/geometry controls or successful model execution.",
      "retirement_reason": "Model-dependent controls and recomputation were removed; dataset dashboard pages and safe persisted metadata remain.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dataset pages remain tested; the model page verifies explicit unavailability and metadata-only reads.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations",
        "tests/test_dashboard_app.py::test_image_embeddings_displays_only_persisted_metadata_and_checksums"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_artifact_errors.py::test_cli_and_resume_report_corruption_as_actionable_errors"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_artifact_errors.py::test_dashboard_reader_reports_corrupt_checkpoint_as_failed_run"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_artifact_errors.py::test_geometry_laboratory_app_test_survives_corrupt_checkpoint"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_artifact_errors.py::test_invalid_and_truncated_checkpoints_raise_actionable_artifact_errors"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_artifact_errors.py::test_structurally_incompatible_checkpoint_is_rejected_without_losing_integrity_checks"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_cli.py::test_cli_mapping_command_creates_all_model_family_runs"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "replaced",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_cli.py::test_cli_runs_both_fixed_fixtures_and_evaluates_inspects_exports"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "replaced",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_high_dimensional_view_uses_actual_sample_anchor_and_explicit_slice"
      ],
      "historical_behavior": "Asserts historical behavior `test high dimensional view uses actual sample anchor and explicit slice`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_historical_snapshot_without_optional_activity_remains_readable"
      ],
      "historical_behavior": "Asserts historical behavior `test historical snapshot without optional activity remains readable`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_invalid_or_missing_runs_are_reported_without_crashing"
      ],
      "historical_behavior": "Asserts historical behavior `test invalid or missing runs are reported without crashing`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_saved_run_discovery_selection_region_metrics_and_comparison"
      ],
      "historical_behavior": "Asserts historical behavior `test saved run discovery selection region metrics and comparison`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_snapshot_activity_tracks_initial_intermediate_and_final_geometry"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_streamlit_geometry_laboratory_loads_saved_runs_without_retraining"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_dashboard_geometry.py::test_town_plot_keeps_overlap_reinclusion_and_window_boundaries_faithful[2-axes0]",
        "tests/test_phase2_dashboard_geometry.py::test_town_plot_keeps_overlap_reinclusion_and_window_boundaries_faithful[3-axes1]"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_experiments.py::test_all_three_fixed_classifiers_train_reload_and_share_identical_fixture"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_experiments.py::test_fixed_nonconvex_fixture_has_closed_window_exclusion_and_multiple_regions"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "deleted",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_experiments.py::test_fixture_splits_and_masks_are_deterministic_and_shared"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "deleted",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_experiments.py::test_missing_snapshot_or_unknown_split_fails_clearly"
      ],
      "historical_behavior": "Asserts historical behavior `test missing snapshot or unknown split fails clearly`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_experiments.py::test_trainable_mapping_has_matched_initial_encoders_and_gradient_paths"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_box_closed_boundaries_margin_and_class_shapes"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_closed_window_boundary_is_excluded_even_when_margin_is_zero"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_contained_town_geometry_extremes_and_gradients"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_dimension_aware_initialization_is_seeded_and_diverse"
      ],
      "historical_behavior": "Asserts historical behavior `test dimension aware initialization is seeded and diverse`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_empty_batches_chunked_equivalence_and_validation"
      ],
      "historical_behavior": "Asserts historical behavior `test empty batches chunked equivalence and validation`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_extreme_float32_raw_parameters_remain_valid_and_seeded"
      ],
      "historical_behavior": "Asserts historical behavior `test extreme float32 raw parameters remain valid and seeded`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_geometry_initialization_is_label_agnostic_and_houses_differ"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_minimum_width_feasibility_and_float32_physical_invariants"
      ],
      "historical_behavior": "Asserts historical behavior `test minimum width feasibility and float32 physical invariants`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_multiple_windows_and_classes_follow_literal_boolean_set_operations"
      ],
      "historical_behavior": "Asserts historical behavior `test multiple windows and classes follow literal boolean set operations`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_physical_box_helper_rejects_width_below_configured_minimum"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_target_volume_initialization_materially_improves_high_dimensional_window_activity[16]",
        "tests/test_phase2_geometry.py::test_target_volume_initialization_materially_improves_high_dimensional_window_activity[2]",
        "tests/test_phase2_geometry.py::test_target_volume_initialization_materially_improves_high_dimensional_window_activity[8]"
      ],
      "historical_behavior": "Asserts historical behavior `test target volume initialization materially improves high dimensional window activity`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_geometry.py::test_window_boundary_exclusion_and_other_house_reinclusion"
      ],
      "historical_behavior": "Asserts historical behavior `test window boundary exclusion and other house reinclusion`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_gradients.py::test_high_dimensional_geometry_hinge_has_finite_loss_and_gradients"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_gradients.py::test_positive_inside_window_activates_window_branch_and_negative_inside_house_activates_frame"
      ],
      "historical_behavior": "Asserts historical behavior `test positive inside window activates window branch and negative inside house activates frame`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_gradients.py::test_positive_outside_and_negative_inside_box_have_point_and_geometry_gradients"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_min_width_preflight.py::test_box_and_town_validate_their_actual_window_parameterization"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_min_width_preflight.py::test_composed_config_rejects_infeasible_width_before_fixture_training_or_artifacts"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_min_width_preflight.py::test_default_small_and_composed_town_boundary_widths_validate"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_min_width_preflight.py::test_geometry_and_composed_config_agree_at_family_feasibility_limits"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_min_width_preflight.py::test_invalid_min_width_values_are_rejected_by_config_and_constructors"
      ],
      "historical_behavior": "Asserts historical behavior `test invalid min width values are rejected by config and constructors`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_models_losses.py::test_all_unknown_instances_are_absent_from_objective_and_have_zero_gradient"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [
        "tests/test_phase2_models_losses.py::test_all_unknown_instances_are_absent_from_reduction_and_have_zero_gradient"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_models_losses.py::test_baseline_zero_logit_threshold_is_inclusive_and_bce_is_unreduced"
      ],
      "historical_behavior": "Asserts historical behavior `test baseline zero logit threshold is inclusive and bce is unreduced`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_models_losses.py::test_classifiers_share_order_shapes_fixed_domainthing_and_dtype[BoxClassifier]",
        "tests/test_phase2_models_losses.py::test_classifiers_share_order_shapes_fixed_domainthing_and_dtype[NeuralBaseline]",
        "tests/test_phase2_models_losses.py::test_classifiers_share_order_shapes_fixed_domainthing_and_dtype[TownClassifier]"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_models_losses.py::test_encoder_gradients_flow_and_points_remain_bounded"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_models_losses.py::test_signed_distance_hinge_shared_for_box_and_town_and_validates_margin"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_resume.py::test_incompatible_resolved_config_is_rejected_on_checkpoint_reload"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_resume.py::test_modified_fixture_arrays_are_rejected_on_reload"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "deleted",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_resume.py::test_resume_restores_adam_and_appends_steps_without_mutating_resolved_config"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_resume.py::test_schema_one_checkpoint_without_window_policy_remains_loadable"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_resume.py::test_zero_margin_split_resume_matches_uninterrupted_run_and_keeps_identity"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase2_resume.py::test_zero_margin_training_reload_and_default_margin"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_architecture.py::test_image_dashboard_delegates_training_to_detached_execution_service"
      ],
      "historical_behavior": "Asserts the old dashboard experiment controls delegated to a detached worker service.",
      "retirement_reason": "The model-dependent controls and worker launch UI were removed; the page is metadata-only.",
      "disposition": "replaced",
      "independent_assertions_retained": "Dashboard startup and read-only persisted run/snapshot metadata remain tested.",
      "replacement_tests": [
        "tests/test_phase3_architecture.py::test_image_dashboard_is_metadata_only_and_has_no_model_execution_or_geometry"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_cache.py::test_frozen_feature_cache_does_not_change_training_trajectory"
      ],
      "historical_behavior": "Compares cached and uncached historical classifier training trajectories.",
      "retirement_reason": "No active classifier training trajectory exists after T3; cache identity, integrity, locking, and visual feature retrieval remain independent.",
      "disposition": "replaced",
      "independent_assertions_retained": "Strict feature-cache identity and integrity tests remain active.",
      "replacement_tests": [
        "tests/test_phase3_cache.py::test_frozen_cache_has_strict_identity_and_only_deterministic_train_validation_panel"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_cli.py::test_image_cli_runs_and_inspects_an_immutable_smoke_configuration"
      ],
      "historical_behavior": "Expects `town experiment image run` to train a historical model, publish a run/checkpoint/snapshot, and then inspect that model-dependent run.",
      "retirement_reason": "The image training and snapshot-producing operations were removed; metadata-only inspection remains supported but the old command sequence cannot execute.",
      "disposition": "replaced",
      "independent_assertions_retained": "Persisted run inspection and checksummed snapshot reading are tested from caller-created historical metadata fixtures; dataset CLI behavior remains independently tested.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication",
        "tests/test_dashboard_app.py::test_image_embeddings_displays_only_persisted_metadata_and_checksums"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_config.py::test_default_and_selectable_image_objective_identities_and_scientific_hashes"
      ],
      "historical_behavior": "Asserts historical behavior `test default and selectable image objective identities and scientific hashes`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_phase3_config.py::test_historical_objective_metadata_and_run_identity_compatibility_are_stable"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_d5a_ontology_entailments.py::test_geometric_violation_does_not_change_ontology_entailment"
      ],
      "historical_behavior": "Combines an OWL entailment with a geometric region-violation query.",
      "retirement_reason": "The geometric query executed removed classifier-region semantics; the ontology entailment is independent.",
      "disposition": "split",
      "independent_assertions_retained": "The MushroomTopping → VegetarianTopping entailment is retained without any region query.",
      "replacement_tests": [
        "tests/test_phase3_d5a_ontology_entailments.py::test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_d5b_relation_publication.py::test_cli_surfaces_actionable_post_publication_failure"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication",
        "tests/test_phase3_d5b_relation_publication.py::test_single_writer_publishes_complete_json_and_cleans_temporary_file"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[compositional-baseline]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[compositional-box]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[compositional-town]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[global-baseline]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[global-box]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[global-town]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[hybrid-baseline]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[hybrid-box]",
        "tests/test_phase3_models.py::test_all_nine_image_model_configurations_forward_and_backpropagate[hybrid-town]"
      ],
      "historical_behavior": "Asserts historical behavior `test all nine image model configurations forward and backpropagate`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_all_unit_image_weights_reproduce_unweighted_known_label_mean"
      ],
      "historical_behavior": "Asserts historical score-to-penalty/objective behavior or tests generic mask/weight/aggregation behavior through the old objective module.",
      "retirement_reason": "Historical score conversion and objective formulas were removed; value-only masked reduction/aggregation is retained without approving an objective.",
      "disposition": "replaced",
      "independent_assertions_retained": "Caller-supplied value reduction, known-mask exclusion, gradients, per-image eligibility, and effective-batch aggregation remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_weighted_instance_reduction_masks_unknown_values_and_gradients",
        "tests/test_phase3_models.py::test_weighted_effective_batch_accumulation_matches_full_batch_gradients",
        "tests/test_phase3_models.py::test_unit_weights_reproduce_unweighted_known_value_mean",
        "tests/test_phase3_models.py::test_undefined_effective_batch_is_reported_without_fabricated_zero_objective"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_common_neural_modules_are_byte_identical_and_classifiers_are_independent"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_component_classifier_reshaping_preserves_box_parameters_and_reshapes_town_activity"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact"
      ],
      "historical_behavior": "Asserts historical behavior `test default family objective penalties and signed distance hinge are numerically exact`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_hand_computed_known_mask_positive_weight_and_per_image_component_reduction"
      ],
      "historical_behavior": "Asserts historical score-to-penalty/objective behavior or tests generic mask/weight/aggregation behavior through the old objective module.",
      "retirement_reason": "Historical score conversion and objective formulas were removed; value-only masked reduction/aggregation is retained without approving an objective.",
      "disposition": "split",
      "independent_assertions_retained": "Caller-supplied value reduction, known-mask exclusion, gradients, per-image eligibility, and effective-batch aggregation remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_known_mask_values_reduce_per_instance_then_per_image_without_score_conversion"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_hybrid_pizza_supervision_alone_reaches_deepsets_and_component_projection"
      ],
      "historical_behavior": "Asserts historical behavior `test hybrid pizza supervision alone reaches deepsets and component projection`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_image_objective_masks_unknowns_uses_positive_weights_and_omits_empty_components"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[16]",
        "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[2]",
        "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[5]",
        "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract[8]"
      ],
      "historical_behavior": "Checks the image-to-point projection shapes/values while also constructing the old image classifier adapter or asserting old classifier outputs.",
      "retirement_reason": "The compatibility adapter and classifier outputs were removed; the independent point pathway remains.",
      "disposition": "replaced",
      "independent_assertions_retained": "Point shapes, bounds, sample/component IDs, validity masks, composition mode, and absence of classifier outputs remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_objective_undefined_for_entire_effective_batch_is_actionable"
      ],
      "historical_behavior": "Asserts historical score-to-penalty/objective behavior or tests generic mask/weight/aggregation behavior through the old objective module.",
      "retirement_reason": "Historical score conversion and objective formulas were removed; value-only masked reduction/aggregation is retained without approving an objective.",
      "disposition": "replaced",
      "independent_assertions_retained": "Caller-supplied value reduction, known-mask exclusion, gradients, per-image eligibility, and effective-batch aggregation remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_weighted_instance_reduction_masks_unknown_values_and_gradients",
        "tests/test_phase3_models.py::test_weighted_effective_batch_accumulation_matches_full_batch_gradients",
        "tests/test_phase3_models.py::test_unit_weights_reproduce_unweighted_known_value_mean",
        "tests/test_phase3_models.py::test_undefined_effective_batch_is_reported_without_fabricated_zero_objective"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_old_snapshot_without_explicit_diagnostic_axes_is_rejected_clearly"
      ],
      "historical_behavior": "Exercises old model-dependent snapshot generation, stored geometry/margins, or the legacy snapshot schema error.",
      "retirement_reason": "Snapshot generation reconstructed historical outputs/geometry and was removed; only checksum-validated read-only snapshot loading remains.",
      "disposition": "replaced",
      "independent_assertions_retained": "Safe schema/checksum/offset validation and stored-value reading remain tested with hand-authored metadata fixtures.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_snapshot_reader_rejects_unsupported_schema_without_model_reconstruction"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[compositional-baseline]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[compositional-box]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[compositional-town]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[global-baseline]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[global-box]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[global-town]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[hybrid-baseline]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[hybrid-box]",
        "tests/test_phase3_models.py::test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier[hybrid-town]"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_snapshot_diagnostic_axes_roundtrip_preserve_class_geometry_and_instance_activity"
      ],
      "historical_behavior": "Exercises old model-dependent snapshot generation, stored geometry/margins, or the legacy snapshot schema error.",
      "retirement_reason": "Snapshot generation reconstructed historical outputs/geometry and was removed; only checksum-validated read-only snapshot loading remains.",
      "disposition": "replaced",
      "independent_assertions_retained": "Safe schema/checksum/offset validation and stored-value reading remain tested with hand-authored metadata fixtures.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_weighted_effective_batch_accumulation_matches_full_batch_gradients"
      ],
      "historical_behavior": "Asserts historical score-to-penalty/objective behavior or tests generic mask/weight/aggregation behavior through the old objective module.",
      "retirement_reason": "Historical score conversion and objective formulas were removed; value-only masked reduction/aggregation is retained without approving an objective.",
      "disposition": "replaced",
      "independent_assertions_retained": "Caller-supplied value reduction, known-mask exclusion, gradients, per-image eligibility, and effective-batch aggregation remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_weighted_instance_reduction_masks_unknown_values_and_gradients",
        "tests/test_phase3_models.py::test_weighted_effective_batch_accumulation_matches_full_batch_gradients",
        "tests/test_phase3_models.py::test_unit_weights_reproduce_unweighted_known_value_mean",
        "tests/test_phase3_models.py::test_undefined_effective_batch_is_reported_without_fabricated_zero_objective"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_weighted_instance_reduction_uses_sum_of_known_class_weights_and_masks_unknown_gradients"
      ],
      "historical_behavior": "Asserts historical score-to-penalty/objective behavior or tests generic mask/weight/aggregation behavior through the old objective module.",
      "retirement_reason": "Historical score conversion and objective formulas were removed; value-only masked reduction/aggregation is retained without approving an objective.",
      "disposition": "replaced",
      "independent_assertions_retained": "Caller-supplied value reduction, known-mask exclusion, gradients, per-image eligibility, and effective-batch aggregation remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_weighted_instance_reduction_masks_unknown_values_and_gradients",
        "tests/test_phase3_models.py::test_weighted_effective_batch_accumulation_matches_full_batch_gradients",
        "tests/test_phase3_models.py::test_unit_weights_reproduce_unweighted_known_value_mean",
        "tests/test_phase3_models.py::test_undefined_effective_batch_is_reported_without_fabricated_zero_objective"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_models.py::test_zero_roi_models_produce_rho_zero_pizza_point"
      ],
      "historical_behavior": "Checks the image-to-point projection shapes/values while also constructing the old image classifier adapter or asserting old classifier outputs.",
      "retirement_reason": "The compatibility adapter and classifier outputs were removed; the independent point pathway remains.",
      "disposition": "replaced",
      "independent_assertions_retained": "Point shapes, bounds, sample/component IDs, validity masks, composition mode, and absence of classifier outputs remain tested.",
      "replacement_tests": [
        "tests/test_phase3_models.py::test_zero_roi_image_points_produce_the_empty_deepsets_point"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_relations.py::test_exact_box_relations_cover_equal_nested_partial_touch_and_disjoint_cases"
      ],
      "historical_behavior": "Asserts hard box/town membership, signed-margin, region-relation, or old relation CLI behavior.",
      "retirement_reason": "Those geometric algorithms and duplicate NumPy/Torch membership implementations were removed, not retained behind guards.",
      "disposition": "deleted",
      "independent_assertions_retained": "Generic no-clobber JSON publication and ontology metadata assertions remain tested separately.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations"
      ],
      "historical_behavior": "Combines authoritative asserted/entailed/unknown ontology metadata checks with a later old-geometry relation query.",
      "retirement_reason": "The geometry relation computation and geometry-backed publication path were removed; ontology metadata itself is independent and retained.",
      "disposition": "split",
      "independent_assertions_retained": "Ontology metadata assertions and canonical snapshot-ID validation remain executable.",
      "replacement_tests": [
        "tests/test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_relations.py::test_town_membership_respects_overlapping_houses_closed_windows_and_reinclusion"
      ],
      "historical_behavior": "Asserts hard box/town membership, signed-margin, region-relation, or old relation CLI behavior.",
      "retirement_reason": "Those geometric algorithms and duplicate NumPy/Torch membership implementations were removed, not retained behind guards.",
      "disposition": "deleted",
      "independent_assertions_retained": "Generic no-clobber JSON publication and ontology metadata assertions remain tested separately.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_relations.py::test_town_no_sampled_containment_counterexample_is_inconclusive_not_proven"
      ],
      "historical_behavior": "Asserts hard box/town membership, signed-margin, region-relation, or old relation CLI behavior.",
      "retirement_reason": "Those geometric algorithms and duplicate NumPy/Torch membership implementations were removed, not retained behind guards.",
      "disposition": "deleted",
      "independent_assertions_retained": "Generic no-clobber JSON publication and ontology metadata assertions remain tested separately.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_relations.py::test_town_sampling_finds_verified_counterexample_and_is_reproducible"
      ],
      "historical_behavior": "Asserts hard box/town membership, signed-margin, region-relation, or old relation CLI behavior.",
      "retirement_reason": "Those geometric algorithms and duplicate NumPy/Torch membership implementations were removed, not retained behind guards.",
      "disposition": "deleted",
      "independent_assertions_retained": "Generic no-clobber JSON publication and ontology metadata assertions remain tested separately.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_dashboard_resume_uses_only_compatible_stopped_checkpoint_and_detached_cli"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_dashboard_worker_launches_detached_cli_process_and_duplicate_launch_is_rejected"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_detached_worker_graceful_stop_and_compatible_resume"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_family_matrix_requires_a_strict_comparison_configuration"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_forced_worker_death_reconciles_status_preserves_checkpoint_and_resumes"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_inspection_uses_live_kernel_lock_not_a_reused_or_stale_manifest_pid"
      ],
      "historical_behavior": "Expected metadata inspection to mutate a stale running manifest based on process/lock state.",
      "retirement_reason": "T2 deliberately made inspection read-only; lifecycle reconciliation is not part of inspection.",
      "disposition": "replaced",
      "independent_assertions_retained": "Read-only manifest/snapshot integrity is tested; generic lock ownership tests remain.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_image_and_synthetic_metadata_inspection_are_read_only"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_invalid_configuration_fails_before_any_run_directory_is_created"
      ],
      "historical_behavior": "Asserts historical behavior `test invalid configuration fails before any run directory is created`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "replaced",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_invalid_dashboard_launch_fails_preflight_without_creating_output[invalid_config0-GT ROIs only]",
        "tests/test_phase3_runs.py::test_invalid_dashboard_launch_fails_preflight_without_creating_output[invalid_config1-positive integer counts]",
        "tests/test_phase3_runs.py::test_invalid_dashboard_launch_fails_preflight_without_creating_output[invalid_config2-integer feature indices]"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_invalid_model_and_vision_preflight_has_no_run_cache_or_cli_side_effects"
      ],
      "historical_behavior": "Compares cached and uncached historical classifier training trajectories.",
      "retirement_reason": "No active classifier training trajectory exists after T3; cache identity, integrity, locking, and visual feature retrieval remain independent.",
      "disposition": "replaced",
      "independent_assertions_retained": "Strict feature-cache identity and integrity tests remain active.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_run_resolution_persists_original_split_ids_and_ignores_visual_preferences_for_identity"
      ],
      "historical_behavior": "Asserts successful model preflight/run resolution, run identity, and original split selection.",
      "retirement_reason": "Model-dependent preflight cannot return an executable run; pure dataset subset ordering and config identity remain independent.",
      "disposition": "replaced",
      "independent_assertions_retained": "Original train/validation split selection and stable config identity remain tested without model construction.",
      "replacement_tests": [
        "tests/test_phase3_runs.py::test_run_config_identity_and_original_subset_order_remain_metadata_only",
        "tests/test_phase3_config.py::test_historical_objective_metadata_and_run_identity_compatibility_are_stable"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_run_service_lifecycle_stop_and_faithful_resume"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "replaced",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_stop_api_rejects_nonrunning_state_and_writes_only_a_graceful_request"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_runs.py::test_worker_persists_nonzero_subprocess_exit_code_and_command_context"
      ],
      "historical_behavior": "Asserts successful model-dependent CLI, worker, launch, stop, preflight, or matrix execution.",
      "retirement_reason": "Supported operations now fail with the centralized unavailable-backend error before side effects.",
      "disposition": "replaced",
      "independent_assertions_retained": "New CLI/API rejection tests verify the explicit error and artifact non-publication.",
      "replacement_tests": [
        "tests/test_classifier_backend_unavailable.py::test_supported_model_operations_fail_explicitly_without_side_effects",
        "tests/test_classifier_backend_unavailable.py::test_classifier_dependent_cli_routes_reject_before_reads_or_publication"
      ],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_experimental_inverse_frequency_is_explicit_and_capped"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "deleted",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_exploratory_training_persists_initial_metrics_in_checkpoint_and_run_result"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_fixed_visual_comparison_pairs_identical_initial_states_across_classifier_families"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_invalid_device_fails_before_checkpoint_directory_creation"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_legacy_checkpoint_objective_and_feature_contract_is_not_silently_resumed"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_mid_epoch_resume_matches_uninterrupted_training_trajectory"
      ],
      "historical_behavior": "Loads, reconstructs, evaluates, resumes, or plots a historical classifier checkpoint.",
      "retirement_reason": "Model-specific checkpoint deserialization/reconstruction is no longer an active operation; no checkpoint fallback is permitted.",
      "disposition": "deleted",
      "independent_assertions_retained": "Checksummed read-only JSON/NPZ snapshot and metadata readers remain covered by separate tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[compositional-baseline]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[compositional-box]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[compositional-town]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[global-baseline]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[global-box]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[global-town]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[hybrid-baseline]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[hybrid-box]",
        "tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate[hybrid-town]"
      ],
      "historical_behavior": "Asserts historical behavior `test nine real image training smokes update save reload and evaluate`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_one_real_image_optimizer_update_saves_resumable_training_state"
      ],
      "historical_behavior": "Asserts historical behavior `test one real image optimizer update saves resumable training state`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_positive_weights_ignore_negative_and_unknown_support_counts"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_snapshot_capture_does_not_change_training_trajectory"
      ],
      "historical_behavior": "Asserts historical behavior `test snapshot capture does not change training trajectory`.",
      "retirement_reason": "The behavior depends on a removed classifier implementation or its model-dependent execution path.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent dataset, vision, point, reducer, identity, metadata, and storage assertions are retained in the referenced infrastructure tests.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_trainable_fpn_real_image_run_updates_fpn_and_reports_gradient_activity"
      ],
      "historical_behavior": "Asserts historical classifier family construction, parameterization, membership, margin, output, loss, or model-specific gradients.",
      "retirement_reason": "T3 removes the historical mathematical implementation itself; recreating these assertions would select old model semantics.",
      "disposition": "deleted",
      "independent_assertions_retained": "Independent point projection/composition, target/mask contracts, generic reducers, and vision infrastructure remain covered elsewhere.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_training_loop_uses_resolved_positive_weight_vectors_and_versions_checkpoint"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_training.py::test_weight_normalization_is_stable_against_the_selected_class_subset"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_weighting.py::test_cap_zero_support_none_and_explicit_experimental_alternative"
      ],
      "historical_behavior": "Generates or trains the historical synthetic classifier fixtures/experiments.",
      "retirement_reason": "The synthetic fixture/training module is not the authoritative Pizzaïolo pipeline and was removed with model-dependent execution.",
      "disposition": "deleted",
      "independent_assertions_retained": "Real dataset/ontology generation, validation, and inspection tests remain active.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_weighting.py::test_effective_number_formula_is_hand_computable_for_two_betas"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_weighting.py::test_inverse_sqrt_uses_known_positive_count_ratio_and_common_unit_scale"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    },
    {
      "original_node_ids": [
        "tests/test_phase3_weighting.py::test_weighting_semantic_version_rejects_legacy_checkpoint_identity"
      ],
      "historical_behavior": "Asserts a historical positive-class weighting policy/formula or its checkpoint semantics.",
      "retirement_reason": "The former weighting-resolution policy is archived configuration metadata, not an approved executable objective; no replacement weighting formula was selected.",
      "disposition": "deleted",
      "independent_assertions_retained": "Caller-supplied positive weights remain accepted by generic reducer tests; their derivation is not asserted.",
      "replacement_tests": [],
      "accepted_historical_revision": "2de9c5f29b436f727720e87132d0b4be25a40768"
    }
  ],
  "disposition_node_counts": {
    "replaced": 77,
    "deleted": 96,
    "split": 3
  },
  "test_collection_change_summary": {
    "starting_t2_collected_nodes": 323,
    "current_collected_nodes_at_manifest_checkpoint": 180,
    "unchanged_original_node_ids_retained": 147,
    "same_node_id_retargeted_or_split": 6,
    "new_node_ids_added": 27,
    "original_node_ids_accounted_in_entries": 176
  }
}
```
<!-- T3_RETIREMENT_DATA_END -->
