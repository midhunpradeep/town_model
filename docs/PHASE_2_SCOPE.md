# Phase 2 scope — geometry, comparable classifiers, and interactive verification

**Status:** implemented as a bounded milestone on `implementation/phase-2-geometry-classifiers`, based on the accepted Phase 1 baseline `bf2e3080a07734c50b9c679aa812d1e1e933db91`.

## Authority and sequencing amendment

The unchanged scientific authority remains [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md), particularly §§1.2–1.3, 4.5, 5, 6, 9, 10 and Appendix B. This file records the approved Phase 2 sequencing amendment; it does **not** rewrite or supersede the expanded specification.

The implementation sequence was changed so the neural baseline, pure-box classifier, and town classifier are built together. Both geometric families use `signed_distance` as their first training objective. They are compared on common synthetic fixtures before any image encoder, detector, or pizza-composition architecture is introduced. A small Streamlit Geometry Laboratory is included now to inspect real saved model states, optimization, predictions, and gradients.

## Included in this milestone

- A shared typed flat-point classifier contract for `[N,d]` float32 points, stable trainable class ordering, fixed-positive outputs, predictions, and optional diagnostics.
- One shared `Linear(d,C_train)` multilabel neural head, standard unreduced `BCEWithLogitsLoss`, and inclusive zero-logit prediction threshold (`logit >= 0`).
- A pure-box classifier with one trainable bounded box per trainable class.
- A town classifier with a shared per-class region vocabulary, default two houses and one strictly contained window per house. Membership is evaluated as Boolean set operations.
- Exact min/max signed coordinate-clearance margins and the same unreduced signed-distance hinge for box and town models.
- A shared known/unknown supervision reducer with per-instance normalization, known-negative weight one, an optional positive-weight vector, fixed/unsupported class exclusion, and zero-supervision handling.
- Two CPU-feasible synthetic experiment modes:
  - `fixed`: fixed points and classifier-only learning, with simple linearly representable labels and a nonconvex frame-minus-window / two-lobe fixture.
  - `mapping`: raw synthetic features pass through independent copies of the same `tanh` point encoder initialization; all three model families jointly train their own encoder and classifier. Labels are never encoder inputs.
- File-based run groups with resolved config, fixture/split identity, environment, initial state, optimizer state, history, validation metrics, final checkpoints, geometry/point snapshots, reload/evaluation, and compatible Adam resume.
- A Geometry Laboratory in the existing Streamlit application, while preserving Phase 1 dataset overview, image explorer, and generation pages. It can inspect actual saved model geometry, train/validation points and target masks, exact membership/margins, baseline logit boundary, history, gradients/activity, snapshots, and side-by-side model runs.
- CLI commands for both experiments, checkpoint evaluation, run inspection, actual geometry export, and resume.

## Mathematical and numerical contract implemented

The bounded domain is `[-1,1]^d`. A box includes both frame boundaries. A house removes the **closed** union of its windows from its frame; a window-boundary point is therefore excluded from that house. A town is the union of its houses, so a separate house can reinclude an excluded point.

For each box:

` s_B(x) = min_i { x_i - l_i, r_i - x_i } `

For each house and town:

` s_H(x) = min(s_F(x), -s_W1(x), ..., -s_Wk(x)) `

` s_T(x) = max_h s_Hh(x) `

These are signed coordinate-clearance surrogates, not Euclidean distances. Exact membership is a separate Boolean computation; a zero margin does not determine whether a boundary point is a member.

For geometric models the unreduced penalty is:

`max(0, m - (2*y - 1)*s(x))`, with default `m=0.1` and `m >= 0`.

The neural head uses standard unreduced BCE-with-logits. Both penalty matrices go through the same reducer. Unknown labels do not become negatives; instances with no known trainable labels are absent from the mean. `DomainThing` is a fixed-positive presentation output and is not a trainable head or loss column. Unsupported classes are not added as negative columns.

### Parameterization decision

Raw corner logits are clamped to `[-12,12]` before sigmoid. Box low corners are mapped into the domain and high corners are mapped above their lows while reserving a positive minimum width. Town frames reserve enough width for one positive-width window and positive clearances on both sides; contained-window coordinates are parameterized within that frame interior. A 16-float32-epsilon numerical reserve is added to the configured `1e-5` minimum to protect the actual float32 corner differences. This is a parameterization safety reserve, not a membership epsilon. No physical-corner repair is performed after optimizer updates.

Initial boxes/frames are label-agnostic and broad (roughly 1.82–1.94 units per axis before domain-fit adjustment); distinct houses receive seeded offsets. Windows start as substantial centered exclusions (about 72% of available interior width per axis). Seeds, physical centers/widths, train/validation occupancy, activity, and final corners are persisted. Initialization never examines positive examples.

## Explicitly deferred requirements and non-goals

These are deferred rather than silently redefined or claimed complete:

- Faster R-CNN, CNN/FPN, ROIAlign, image normalization/coordinate integration, predicted detections, matching, and every image-to-point pathway.
- Full Pizzaïolo dataset/image training, component/pizza supervision, the full pizza-composition architecture, global/compositional/hybrid modes, and image generalization experiments.
- Detector, CNN, detector/classifier joint training, all three original detector training phases, production-scale dataset runs, and the full end-to-end detector acceptance smoke test.
- `signed_bce`, `reference_box_bce`, and `reference_town_distance`; only baseline BCE and geometric `signed_distance` are implemented. No historical reference town loss was substituted.
- Independent-window parameterization. The implemented trainable mode is genuinely contained; other mode values are rejected. Corner-level evaluation nevertheless follows literal frame-minus-window set operations for geometry verification.
- Full experiment dashboard features such as editing all experiment settings, local subprocess launch/cancel, multi-job scheduling, detector/image traces, and comprehensive run management.
- Advanced ontology relation diagnostics, proof of inclusion/disjointness for towns, PCA/UMAP, OWL/ALC reasoning, ontology-axiom regularization, and final RQ2 protocols.
- Adaptive house/window capacity, gates, smooth min/max, geometry regularizers, dynamic ontology class creation, or full weighting-policy implementations.
- Any deferred continuous-dimension, force-field, inertia, toroidal, binary-tree, or other embedding experiment.

This milestone demonstrates geometry/classifier optimization on controlled synthetic points only. It does not establish generalization to Pizzaïolo images, prove ontology-relation preservation, or settle the thesis research questions.
