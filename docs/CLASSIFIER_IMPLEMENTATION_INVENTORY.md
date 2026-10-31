# Classifier implementation inventory

**Purpose:** factual inventory of the classifier and geometry code present in the accepted repository, to make a later author-led model-consolidation/reconstruction inspectable. This records what the code does; it does not endorse the choices, define a replacement model, or recommend a redesign.

**Baseline:** accepted implementation `a098a0a3c66a30200bd85906212cb07ef8bd19eb` (Phase 3); its parent history includes the independently accepted Phase 2 implementation. No implementation was changed to produce this document. See [acceptance index](ACCEPTANCE_INDEX.md), [current project status](CURRENT_PROJECT_STATUS.md), and the unchanged [expanded specification](../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md).

## How to read the model descriptions

The following are kept separate throughout:

1. **Mathematical definition:** the set, score or objective written as a formula.
2. **Code parameterization:** the tensors and transforms used to represent it.
3. **Differentiable training surrogate:** the value whose gradients are used during optimization.
4. **Hard inference rule:** the Boolean prediction rule.
5. **Empirically observed behavior:** tests or reported run outcomes, not a mathematical guarantee.

A signed margin is not a hard membership decision. A falling training objective is not necessarily a rising hard-membership F1. The baseline's logits and geometric margins also have different meanings and scales.

## Implementation locations

| Concern | Implementation |
|---|---|
| Point classifier interface and outputs | `src/town_model/models/classifiers.py`: `PointClassifier`, `ClassifierOutput` |
| Linear baseline / classifier factory | `classifiers.py`: `NeuralBaseline`, `make_classifier` |
| Box/town geometry and parameter transforms | `src/town_model/models/geometry.py`: `BoxGeometry`, `TownGeometry`, `evaluate_town_corners`, `_bounded_corners` |
| Phase 2 synthetic objective and reducer | `classifiers.py`: `unreduced_penalties`; `src/town_model/models/losses.py`: `reduce_supervision` |
| Phase 3 image projections/composition | `src/town_model/models/image_embeddings.py`: `ComponentPointProjection`, `GlobalPointProjection`, `DeepSetsComposition`, `HybridPointFusion`, `ImageOntologyModel` |
| Image objectives/aggregation | `src/town_model/training/image_objectives.py`: `_penalties`, `_instance_terms`, `image_objective_terms`, `aggregate_effective_batch` |
| Image positive-weight construction | `src/town_model/training/image_trainer.py`: `_positive_weights_from_counts`, `_positive_weights` |
| Optimizers/checkpoints | `experiments/synthetic.py` (Phase 2); `training/image_trainer.py` (Phase 3) |
| Relation inspection | `src/town_model/inspection/relation_diagnostics.py`; box/town regions only, not neural baseline hyperplanes |

## Shared classifier contract

`PointClassifier` receives finite `torch.float32` points shaped `[N,d]` and returns a `ClassifierOutput` with Boolean trainable-class predictions `[N,C]`, plus logits for the baseline or signed margins for geometric families. Trainable class order is explicit. `DomainThing` is a fixed-positive output for each observed entity and is not a learned class row or training-loss column. Only the declared trainable vocabulary is represented; unsupported classes are not silently added as negative heads.

Phase 3 uses the same classifier instance for component points and final pizza points. Component points also feed DeepSets in compositional/hybrid modes. It does not have separate pizza/component classifier heads.

## 1. Neural baseline

### Definition and code parameterization

For point `x ∈ R^d` and class `c`, `NeuralBaseline.linear` computes the affine score

```text
z_c(x) = w_c · x + b_c
```

through one `nn.Linear(d,C)` layer. The score is a logit, not a geometric distance or an OWL region. The current head has no hidden layer.

### Training surrogate and hard prediction

`unreduced_penalties()` in Phase 2 and `_penalties()` in Phase 3 use unreduced binary cross-entropy with logits:

```text
L_c = softplus(z_c) - y_c z_c
```

The hard output is inclusive at zero: `z_c >= 0` predicts positive (equivalent to sigmoid probability at least 0.5). `DomainThing` is added separately as fixed true.

### Evidence

- Numerical boundary/loss test: `tests/test_phase2_models_losses.py::test_baseline_zero_logit_threshold_is_inclusive_and_bce_is_unreduced`.
- Shared outputs/order: `test_classifiers_share_order_shapes_fixed_domainthing_and_dtype`.
- Phase 3 objective formula/zero-logit numerical check: `tests/test_phase3_models.py::test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact`.

## 2. Pure box geometry

### Mathematical definition

One closed axis-aligned box is represented per trainable class within `[-1,1]^d`:

```text
B_c = {x : l_ci ≤ x_i ≤ r_ci for every i}
```

The signed coordinate-clearance margin used by both geometric families is

```text
s_B(x) = min_i {x_i - l_i, r_i - x_i}.
```

It is not Euclidean distance. For an interior point it is positive; for an outside point it is negative; on any limiting face it is zero. Exact membership uses all inclusive coordinate comparisons, independent of the sign convention at zero.

### Code parameterization

`BoxGeometry` owns two float32 trainable tensors, each `[C,d]`:

- `raw_low` — unconstrained low-corner logits;
- `raw_width` — unconstrained high-corner/remaining-width logits.

With raw values clamped to `[-12,12]`, floor `f`, and sigmoid `σ`, `_bounded_corners()` computes:

```text
low = -1 + (2 - f) σ(raw_low)
remaining = max(0, 1 - low - f)
high = 1 - remaining (1 - σ(raw_width))
```

This produces ordered corners in the domain and a minimum physical axis width. `_set_bounded_corners()` initializes the raw values from desired physical corners. The raw clamp keeps transforms finite but makes them locally saturated outside the clamp interval.

Let `R = 16 * torch.finfo(float32).eps`. The configured width is augmented by `R`; direct validation requires `0 < min_width < 2 - 2R`. This numerical reserve belongs to the transform, not to membership or the loss. `physical_boxes()` returns the actual transformed corners (or fixed registered buffers for the test/diagnostic constructor `from_physical`). Optimizer steps do not repair or clip physical corners afterward.

### Training and hard inference

Training differentiates through `signed_margin`; hard prediction uses exact closed-box comparisons in `BoxGeometry.evaluate()`. It does not threshold the margin. A boundary can therefore have zero margin and still be a positive member.

### Evidence

- Formula, shape, closed-boundary and zero-margin behavior: `tests/test_phase2_geometry.py::test_box_closed_boundaries_margin_and_class_shapes`.
- Extreme raw values and transform validity: `test_extreme_float32_raw_parameters_remain_valid_and_seeded`, `test_minimum_width_feasibility_and_float32_physical_invariants`.
- Direct geometric point/parameter gradients for positive-outside and negative-inside examples: `tests/test_phase2_gradients.py::test_positive_outside_and_negative_inside_box_have_point_and_geometry_gradients`.
- Chunked evaluation and gradient parity, including town geometry: `test_empty_batches_chunked_equivalence_and_validation`.

## 3. Town geometry: houses, frames and windows

### Mathematical definition

For each class, the represented town is a union of houses. A house is a closed frame with a union of closed windows subtracted:

```text
H_h = F_h \ (⋃_w W_hw)
T   = ⋃_h H_h
```

The signed margins implemented by `evaluate_town_corners()` are:

```text
s_F(x)   = min_i {x_i - l_Fi, r_Fi - x_i}
s_W(x)   = min_i {x_i - l_Wi, r_Wi - x_i}
s_H(x)   = min(s_F(x), -s_W1(x), ..., -s_WW(x))
s_T(x)   = max_h s_Hh(x)
```

Exact hard membership instead evaluates:

```text
in_frame = all(l_F ≤ x ≤ r_F)
in_window = any_w all(l_W ≤ x ≤ r_W)
in_house = in_frame AND NOT in_window
in_town = any_h in_house
```

Consequences: frame boundaries are included; a closed-window boundary is excluded from its house; a second house can reinclude a point excluded by the first. A window boundary can have zero signed margin while exact membership is false.

### Code parameterization and supported window mode

`TownGeometry` has fixed positive `houses` and `windows` counts for every class. Trainable parameters are:

- `raw_frame_low`, `raw_frame_width`: `[C,H,d]`;
- `raw_window_width`, `raw_window_position`: `[C,H,W,d]`.

The frame uses the same ordered-corner transform as boxes. For configured minimum width `m`, the code uses clearance `m+R`, window floor `m+R`, and frame floor `3*(m+R)+R`; feasibility validation requires `0 < m < (2-5R)/3`. For frame width `q`, interior span `q - 2*clearance`, and sigmoid width/position fractions `a,p`, the contained window is computed as:

```text
window_width = window_floor + (interior - window_floor) a
slack        = interior - window_width
window_low   = frame_low + clearance + slack p
window_high  = frame_high - clearance - slack (1-p)
```

Thus windows remain strictly inside the frame with positive axis width and clearance by construction, subject to the float32 reserves. Raw values are clamped to `[-12,12]` before sigmoid.

**Only `window_mode="contained"` is trainable/supported.** `TownGeometry` rejects other modes. `evaluate_town_corners()` can evaluate literal frame-minus-window set operations when supplied windows outside frames; that utility does not mean an independent-window parameterization is available for learning.

### Initialization

Initialization is seeded and label-agnostic:

- Box centers start as small seeded normal offsets around zero (`std` scale `0.025`); widths begin in approximately `[1.82,1.94]` per axis before fitting to the domain.
- Town frame centers use seeded small variation and deterministic distinct house offsets; frame widths begin in approximately `[1.82,1.92]` per axis before domain fitting.
- Town windows are placed inside the frames using seeded positions. The default `window_target_volume=0.25` sets a per-axis fraction `q(d)=0.25**(1/d)` of available interior span above the minimum window floor:

```text
window_width = floor + q(d) * (interior - floor)
```

The actual window/frame volume fraction can differ slightly because of the floor and finite geometry. Initialization does not inspect labels or positive examples. Initial physical centers/widths, mode, seed and the volume setting are persisted in model/run metadata.

### Evidence

- Exact frame/window boundary, exclusion/reinclusion and literal multiple-window/class behavior: `tests/test_phase2_geometry.py::test_window_boundary_exclusion_and_other_house_reinclusion`, `test_closed_window_boundary_is_excluded_even_when_margin_is_zero`, `test_multiple_windows_and_classes_follow_literal_boolean_set_operations`.
- Transform, containment, finite precision, initialization and min-width constraints: `test_contained_town_geometry_extremes_and_gradients`, `test_dimension_aware_initialization_is_seeded_and_diverse`, `test_minimum_width_feasibility_and_float32_physical_invariants`.
- Dimension-aware occupancy/gradient probe at d=2/8/16: `test_target_volume_initialization_materially_improves_high_dimensional_window_activity`.
- Town branch gradients for positive-inside-window and negative-inside-house examples: `tests/test_phase2_gradients.py::test_positive_inside_window_activates_window_branch_and_negative_inside_house_activates_frame`.
- Empty batches, chunked membership/margins and gradients: `test_empty_batches_chunked_equivalence_and_validation`.

## 4. Image-to-point pathway used with the same classifiers

This is image feature/projection code, not detector code. `SharedVisualFeatureExtractor` uses GT ROIs, ROIAlign and a whole-original-image ROI through a MobileNetV3-Large/FPN feature adapter. No detector proposals or detector class scores are produced.

`ComponentPointProjection` maps a 260-value component vector (256 pooled FPN values plus four normalized original-box coordinates) through `260→256→LayerNorm/ReLU→128/ReLU→dropout(0.1)→d→tanh`. `GlobalPointProjection` maps `[256,7,7]` via a `1×1` convolution to 32 channels, flattens 1,568 values, applies `1568→256` with ReLU and `256→d` with tanh.

`DeepSetsComposition` computes `ρ(Σ_k φ(x_k))`, with a shared two-linear-layer `φ: d→128→128`, ReLU activations, a two-linear-layer `ρ:128→128→d`, and final tanh. Invalid padded components are masked before sum. Empty component collections yield `ρ(0)`. `HybridPointFusion` concatenates global/composed points and applies `2d→128→d` with ReLU then tanh. The selected pizza mode chooses global, composed, or fused points; only the final selected pizza point receives pizza supervision. Component points also receive component loss and, for compositional/hybrid modes, pizza gradients through aggregation.

Evidence includes `tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract`, `test_deepsets_is_permutation_invariant_multiplicity_sensitive_and_empty_safe`, `test_zero_roi_models_produce_rho_zero_pizza_point`, `test_hybrid_pizza_supervision_alone_reaches_deepsets_and_component_projection`, and `test_nine_real_image_training_smokes_update_save_reload_and_evaluate` in `tests/test_phase3_training.py`.

## 5. Differentiable objectives, masks and weighting

### Phase 2 synthetic objectives

`unreduced_penalties()` returns a `[N,C]` matrix:

- baseline: standard unreduced BCE-with-logits;
- box/town: signed-distance hinge, `relu(m - (2y-1)s)`, with default `m=0.1` and finite `m≥0`.

`reduce_supervision()` in `models/losses.py` combines the penalty with `known_mask`, an optional trainable-class mask and optional positive weights. For instance `n`:

```text
L_n = Σ_c mask_nc * w_nc * penalty_nc / Σ_c mask_nc * w_nc
```

The positive weight applies only when `y=1`; known negatives have weight 1. Unknowns and masked/nontrainable classes contribute neither numerator nor denominator. Instances with zero denominator are excluded from the final mean. An all-unknown batch returns a connected zero with zero supervised gradient. Phase 2 experiments configure unit weights (`class_weighting="none"`).

### Phase 3 image objectives

`_penalties()` in `training/image_objectives.py` uses standard BCE-with-logits for the baseline. For box/town it supports two configured surrogates on the same signed margin `s`:

```text
signed_distance: relu(m - (2y-1)s)
signed_bce:      softplus(-(2y-1)s / τ)
```

The documented defaults are `m=0.1` and `τ=0.1`; the margin may be zero. The baseline continues to use ordinary BCE regardless of geometric-loss configuration. `reference_box_bce` and `reference_town_distance` are not implemented.

`_instance_terms()` uses the same known-weight-sum normalization. `image_objective_terms()` averages supervised component occurrences within each eligible image; `aggregate_effective_batch()` then averages over eligible images across the full accumulation window. Pizza terms are averaged over eligible images. Configured component/pizza coefficients are applied after each task-level reduction. A task with no support is omitted and reported; if all enabled tasks have no support in the effective batch, `UndefinedImageObjective` is raised.

### Phase 3 positive weights

`_positive_weights_from_counts()` derives weights separately for pizza and component supervision, from known-positive counts in the original training subset only:

```text
inverse_sqrt:      r_c = 1/sqrt(n_c+)
effective_number:  r_c = (1-β)/(1-β**n_c+)
```

Raw values are normalized over all positive-supported canonical in-scope classes for that supervision level so the most-supported class has weight one, capped, then projected to selected model classes. Known-negative weight remains one; unknown support is not used. Zero-support exploratory classes get finite weight one; comparison coverage fails for required in-scope zero-positive classes. The additional `experimental_inverse_frequency` is named separately. Default policy is `none`.

The all-canonical normalization is a documented implementation convention, not a value claimed to be uniquely specified or scientifically optimal. Exact formula/weight tests are in `tests/test_phase3_weighting.py`; reducer and accumulation checks are in `tests/test_phase3_models.py::test_weighted_instance_reduction_uses_sum_of_known_class_weights_and_masks_unknown_gradients`, `test_weighted_effective_batch_accumulation_matches_full_batch_gradients`, `test_all_unit_image_weights_reproduce_unweighted_known_label_mean`, and `test_hand_computed_known_mask_positive_weight_and_per_image_component_reduction`.

## 6. Gradient propagation, optimizer and checkpoint state

- Geometric training scores remain differentiable through exact `min`/`max` margin operations and the raw-to-physical coordinate transforms. Boolean membership is not the training score and does not provide this gradient.
- Gradients can flow from signed margins to geometry parameters and point coordinates. In Phase 2 mapping experiments, they continue through the synthetic `tanh` point encoder. In Phase 3, the classifier geometry gradients continue through the selected component/pizza points and projection/composition networks; visual gradients depend on the configured frozen/trainable visual policy.
- `torch.amin`/`torch.maximum` branch selection, hinge satisfaction, ReLU, sigmoid saturation from the raw clamp and dropout create sparse or zero gradients in some branches/steps. The code logs gradient norms/zero fractions/activity; it does not dynamically add/remove houses/windows or auto-reset inactive geometry.
- Phase 2 synthetic runs optimize with Adam over the classifier and optional synthetic point encoder. The saved checkpoint includes model/encoder state, initial states, Adam optimizer state, resolved config identity, fixture identity, class order and step. Resume reloads Adam state and appends steps; the synthetic harness uses full-training-split batches and has no learning-rate scheduler.
- Phase 3 image training uses configured AdamW parameter groups for pretrained backbone, FPN, projections/composition and classifier geometry. It saves model/visual states, optimizer and scheduler state, RNG state, epoch/batch cursor and sampler order, best-monitor/history state, resolved config/data/class identity, environment and parent checkpoint hashes. Resume requires matching identity and a safe accumulation boundary.

Relevant evidence: Phase 2 gradient tests cited above and `tests/test_phase2_experiments.py::test_trainable_mapping_has_matched_initial_encoders_and_gradient_paths`; Phase 3 `tests/test_phase3_training.py::test_mid_epoch_resume_matches_uninterrupted_training_trajectory`, `test_trainable_fpn_real_image_run_updates_fpn_and_reports_gradient_activity`, and `test_snapshot_capture_does_not_change_training_trajectory`.

## 7. Existing tests, learning outcomes and diagnostics

### Synthetic Phase 2 runs (reported, exploratory)

The Phase 2 implementation report records seed 42, `d=2`, 320 generated points (224 train / 96 validation), 60 Adam steps, and four percent unknown-label probability. Validation macro-F1 values were:

| Fixture | Neural baseline / BCE | Box / hinge | Town / hinge |
|---|---:|---:|---:|
| Fixed simple | 0.9134 | 0.6611 | 0.8203 |
| Fixed nonconvex window | 0.3974 | 0.7535 | 0.8912 |
| Trainable mapping window | 0.5636 | 0.3336 | 0.6688 |

These are small, single-seed synthetic diagnostics. They demonstrate different outcomes across fixtures, not a general family ordering. BCE and hinge scalar losses are not comparable. The report retains parameter counts, masks, checkpoints, gradients and historical limitations.

The tests include exact boundary/margin cases, extreme float32 parameter values and feasibility, high-dimensional gradients, unknown-mask/weight reductions, fixture determinism, matched mapping initialization, checkpoint reload, Adam resume parity, corrupt artifacts, visualization and saved-snapshot behavior.

### Image-path reports (reported, exploratory)

The accepted Phase 3 review reports nine real-RGB optimizer/checkpoint/reload smoke cells for `{baseline, box, town} × {global, compositional, hybrid}`. The Phase 3 remediation report records a corrected, single-seed, 128/48 image-subset GT-ROI demonstration (selected ontology IDs 2 and 29, hybrid `d=8`, frozen ImageNet MobileNet/FPN). Best validation pizza/component GT-ROI macro-F1 was baseline `1.0000/0.9709`, box `0.7368/0.0000`, town `1.0000/0.9709`; the poor box component result is retained. These are pre-detector exploratory observations, not final scientific comparisons. No test model outcomes were consulted.

### Visualization and diagnostics

- Phase 2 Geometry Laboratory shows actual saved model state, hard membership, signed margins, boundaries/corners, points/known masks, baseline logit contours, slices, history, snapshots and gradient/activity diagnostics. High-dimensional regions are evaluated on exact slices; projected markers are identified as projections.
- Phase 3 inspection snapshots persist bounded JSON/NPZ traces with declared diagnostic axes, class geometry, points, scores/masks, reference IDs and optional visual features. Image dashboard views use GT overlays and saved traces; saved snapshot inspection is read-only and does not load optimizer checkpoints.
- Relation diagnostics support exact interval checks for box pairs and seeded sampled town queries verified with exact town membership. No finite sample without a witness is treated as a proof. Baseline hyperplanes are not handled as box/town regions by the relation API.

## 8. Differences from the expanded scientific specification

These are factual implementation gaps/choices, not recommendations:

| Specification item | Current code/reports |
|---|---|
| Scientific stages: detector pretraining → frozen-detector GT-ROI embeddings → predicted-ROI joint fine-tuning | Current image path is pre-detector and GT-ROI only. No detector, detector checkpoint, predicted-ROI training, joint detector loss or end-to-end component metric exists. |
| Five-way classifier/objective core | Baseline BCE and both signed geometric objectives are implemented; the full controlled core has not been run as the specified canonical scientific comparison. Phase 3 configs apply one geometric-loss choice per run. |
| Independent and contained town-window parameterizations with matched physical initialization | Only strictly-contained trainable windows exist. Non-contained corners can be evaluated by the pure corner utility, but no independent trainable mode/paired study exists. |
| Reference box BCE and reference town distance | Neither objective is implemented. Current geometric training supports `signed_bce` and `signed_distance`; Phase 2 uses `signed_distance`. |
| Comparison quality, repeated seeds, detector measures and final held-out evaluation | Reported runs are small/single-seed exploratory demonstrations or smoke tests. No final test partition model outcome is reported. |
| Relation reasoning and final RQ2 method | Exact box and sampled town diagnostics exist; full OWL/ALC reasoning, supervisor-approved pair metrics and scientific RQ2 analysis do not. |
| Baseline capacity | The implemented baseline is the specified shared linear head, not a stronger MLP; classifier capacities differ and are not matched. |
| Adaptive region capacity/gates/smooth operators | Houses/windows are fixed-capacity; no dynamic gates, smooth min/max or automatic inactivity repair exists. |

See [`docs/CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md) for the full remaining engineering checklist and [`docs/PHASE_3_SCOPE.md`](PHASE_3_SCOPE.md) for the accepted sequencing boundary. The expanded specification remains unchanged and authoritative.

## 9. Questions left for the author

The following are recorded without selecting an answer:

- Which existing model definitions, parameterizations and loss variants are to remain in the author-directed consolidation/reconstruction?
- How should the specification's independent-window mode and paired-geometry comparison be handled in the eventual experimental baseline?
- What source-faithful mathematical correction and approval process applies to `reference_town_distance`?
- Which common positive-weight normalization and cap settings are selected for the intended scientific comparison?
- What supervisor-approved relation protocol operationalizes RQ2, including pair scope, unknown ontology relations, aggregate metrics and sufficient evidence for a relation claim?
- Which provisional defaults and model-capacity confounds must be frozen or varied in the final study?

This inventory does not resolve these questions or introduce new model definitions. It is an inspection aid for the author, not a reconstruction plan.
