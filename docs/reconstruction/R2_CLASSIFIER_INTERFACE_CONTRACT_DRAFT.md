# R2 Classifier-Facing Interface Contract — Draft for Author Review

> **DRAFT — NOT APPROVED SPECIFICATION.** Prepared for R2 / issue [#14](https://github.com/midhunpradeep/town_model/issues/14), child of planning map [#2](https://github.com/midhunpradeep/town_model/issues/2). This document records evidence and proposals; it is not an authoritative scientific source, does not amend A1/F11/F12, and grants no implementation or deployment authorization. All proposals remain pending explicit author review. Issue [#3](https://github.com/midhunpradeep/town_model/issues/3) remains OPEN for explicit author confirmation; this draft does not answer or close it.
>
> **Baseline inspected:** `7ebc2b62c175288b0673524fbfae9f9d6f15a4ff` (`main`, fetched `origin/main`, and live remote `main` matched at investigation start).

## 1. Status vocabulary and authority

Every substantive statement below uses one of these statuses:

- **Already approved** — directly fixed by a cited author-approved source. Repository documentation acceptance establishes faithful recording, not proof, executable implementation, or demonstrated learning.
- **Existing engineering fact** — verified in retained code/tests or the accepted teardown at the stated revision. It is not automatically a future requirement.
- **Proposed** — a candidate interface/default requiring explicit author approval before it becomes a specification.
- **Unresolved** — the available sources/code do not select an answer; implementation must wait for the stated boundary.
- **Deferred** — outside this initial interface/evaluator scope.

The original [A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) is authority for F0–F10; the separately accepted [F11 addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) and [F12 addendum](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) are their own authorities. Together the unchanged sources report 70 + 12 + 12 = **94 author-reported approvals**. F12.1–F12.8 remain a grouped geometry block without per-ID formula mapping. Their independent repository-document acceptances do not establish mathematical proof or implementation. The [readiness frontier](A1_IMPLEMENTATION_READINESS.md) is derived planning only.

Accepted teardown authority is the [T5 final review](../../reviews/classifier_teardown_t5_final_acceptance.md), accepting implementation `19222424f5ad8347deddc6dfb2d7187bd85609c0` with report commit `2b9df576aac2fd95f093bbb3b046813620a822a5`. T0/T1's [independent review](../../reviews/classifier_teardown_t0_t1_independent_review.md), the [T2 report](T2_DISCONNECTION_REPORT.md), [T3 report](T3_REMOVAL_REPORT.md), and [T4 report](T4_HISTORICAL_PRESENTATION_REPORT.md) are teardown evidence, not classifier-reconstruction authority. The old [classifier inventory](../CLASSIFIER_IMPLEMENTATION_INVENTORY.md) is descriptive historical evidence only.

## 2. Scientific semantics that this draft does not reopen

| Status | Fixed boundary | Source |
|---|---|---|
| **Already approved** | Class predicates have evidence-supported known/unknown supervision; open-world absence is not automatically false. Global class predicates share one latent domain, and cross-type negatives are not automatic. | [F0.1/F1.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L22), [F2.3/F2.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L29); [F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31). |
| **Already approved** | A mathematical classifier consumes flat latent points `[N,d]` plus its own classifier state, not image/dataset/ontology source metadata as point inputs. A point represents an observed entity in shared `Δ`; component occurrences remain distinct. | [F2.1–F2.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L25), [F4.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54). |
| **Already approved** | Both physical domains are distinct comparison conditions: `Δ=[-1,1]^d` and `Δ=ℝ^d`. A bounded range must not be described as unbounded. | [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33). |
| **Already approved** | Hard geometric prediction is exact Boolean set membership. Boxes are closed; a house is its closed frame minus ambient strict window interiors; a town is a union of houses. F3.6-R supersedes F3.6. Both contained and independent windows use those same set semantics. | [F3.2/F3.4–F3.7-R](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L39), especially [F3.6-R](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L43). |
| **Already approved** | `W=0` is defined; `H=1,W=0` is exactly one box, with no fake window state. Positive trainable primitive widths do not prohibit useful degenerate **fixed evaluator fixtures**. | [F3.9–F3.11](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L46). |
| **Already approved** | Affine trainable-class logits are `z_c(x)=w_c·x+b_c`, with hard prediction `z_c≥0`. `DomainThing` is fixed, parameter-free, and always positive relative to the selected learned domain. The approved output boundary separates ordered Boolean `contains:[N,C_train]` from family-specific numerical measurements. | [F4.2–F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L55). |
| **Already approved** | Box signed coordinate clearance is `min_j(x_j-l_j,r_j-x_j)`, not Euclidean distance. Town clearance uses the approved exact min/max composition. Hard membership and numerical clearance are separate; retain raw physical units. | [F5.1–F5.6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L61). |
| **Already approved** | Geometry owns sets/measurements; objectives own per-label penalties; reducers own masks/weighting/aggregation; optimization owns updates; evaluation independently verifies predictions. The primary five conditions and F8 known-mask/aggregation rules are not changed here. | [F5.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L62), [F6.1–F6.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L68), [F8.1–F8.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L93). |
| **Already approved** | F12 specifies trainable geometry/domain transforms and optimizer/stage lifecycle policies. Those approvals do not implement them and do not block pure evaluators that take explicit physical corners and have no optimizer state. | [F12 geometry block](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L36), [F12 readiness boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L194); [readiness P0-A/B/C](A1_IMPLEMENTATION_READINESS.md#L37). |
| **Deferred** | No calibrated probabilities, Euclidean signed distances, new regularizers/losses, ontology-relation training penalties, detector/ROI policy, Phase 4, or scientific training/evaluation protocol is specified here. | [F3.3/F5/F6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L40), [F12 comparison boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L215), T5 review sections J–Q. |

## 3. Retained engineering interface inventory

### 3.1 Dataset vocabulary, targets, masks, and identities

**Existing engineering fact.** `PreparedDataset` exposes `canonical_class_ids`, aligned `class_names`/IRIs, separate fixed-positive and unsupported IDs, and a vocabulary-manifest hash ([`dataset/image_data.py:22–37`](../../src/town_model/dataset/image_data.py#L22), [`image_data.py:80–123`](../../src/town_model/dataset/image_data.py#L80)). Loading validates contiguous manifest IDs, preserves the manifest's trainable order, and excludes fixed-positive entries from trainable IDs ([`image_data.py:91–103`](../../src/town_model/dataset/image_data.py#L91)). `targets_in_trainable_order()` projects full target and known-mask vectors through that authoritative ID tuple ([`image_data.py:126–141`](../../src/town_model/dataset/image_data.py#L126)); it does not derive vocabulary from the current record or batch. `build_class_vocabulary()` assigns IDs in deterministic ontology-IRI order and records trainable/fixed/unsupported status ([`dataset/ontology.py:450–491`](../../src/town_model/dataset/ontology.py#L450)). The retained example has 60 trainable classes, one fixed-positive `DomainThing`, and five unsupported classes; tests verify disjointness and target alignment ([`test_phase3_dataset.py:20–27,46–54`](../../tests/test_phase3_dataset.py#L20)).

`ImageBatch` carries targets and known masks but no class-ID tuple; its pizza and component target widths must agree. Component occurrence IDs and detector categories are separate identity/input metadata ([`contracts.py:23–39`](../../src/town_model/contracts.py#L23)). **Existing engineering fact:** the dataset owns canonical order; **unresolved:** the exact future binding of that order to classifier parameter/output rows and the presentation of fixed-class output (see decision D1).

### 3.2 Image/vision contracts

**Existing engineering fact.** `ImageBatch.validate()` requires images `[B,3,H,W]`, `B≥1`, finite image values, one identity/ragged entry per image, pizza targets `[B,C]`, boolean matching pizza masks, finite targets and binary known targets. Each component box list is float32 `[K_b,4]`, finite and positive-area; IDs/categories lengths must match it. Component targets/masks are `[K_b,C]`, masks are bool, widths match pizza targets, targets are finite, and known targets are binary ([`contracts.py:41–85`](../../src/town_model/contracts.py#L41)). Image dtype is not explicitly validated there.

`VisionFeatureBatch` contains `component_vectors`, a validity mask, global spatial features, padded component IDs, and optional diagnostics ([`contracts.py:131–137`](../../src/town_model/contracts.py#L131)). `SharedVisualFeatureExtractor` produces component vectors `[B,K,260]`, global features `[B,256,7,7]`, and a bool prefix-valid mask; `K` is the maximum component count in the image batch and may be zero. Its component vector concatenates ROI visual features with normalized pixel-derived ROI coordinates; detector category/confidence are not copied into the point tensor. The extractor rejects nonfinite generated features ([`vision/features.py:35–42,69–115`](../../src/town_model/vision/features.py#L35)). The retained data contract keeps target labels, detector category IDs and ROI confidence outside the semantic point tensor. Pixel-derived ROI coordinates are an existing visual feature, consistent with F2.5.

### 3.3 Point-only projection and identity transport

**Existing engineering fact.** `ImagePointModel` accepts `VisionFeatureBatch` plus caller-supplied sample IDs; it accepts positive integer `d` and one of `global`, `compositional`, or `hybrid` modes ([`image_points.py:99–139`](../../src/town_model/models/image_points.py#L99)). Its output `ImagePointResult` carries sample IDs, padded component IDs, validity mask, component points `[B,K,d]`, global/composed/final pizza points `[B,d]`, selected mode, and optional diagnostics—no classifier predictions, logits or loss ([`contracts.py:149–161`](../../src/town_model/contracts.py#L149)).

The current architecture is:

- component projection `260→256→128→d`, terminal `Tanh` ([`image_points.py:15–34`](../../src/town_model/models/image_points.py#L15));
- global projection `256×7×7→32×7×7→256→d`, terminal `tanh` ([`image_points.py:37–52`](../../src/town_model/models/image_points.py#L37));
- DeepSets `φ`, bool-mask gating and sum over `K`, then `ρ` with terminal `Tanh` ([`image_points.py:55–81`](../../src/town_model/models/image_points.py#L55));
- hybrid fusion of global/composed points, terminal `tanh` ([`image_points.py:84–96`](../../src/town_model/models/image_points.py#L84)).

Tests establish shapes/ranges, identity and mask preservation, no `contains`/`logits`, permutation-invariant masked sum composition, and the `K=0` result ([`test_phase3_models.py:34–76`](../../tests/test_phase3_models.py#L34)). `ImagePointModel.forward()` checks component/global feature shapes, `len(sample_ids)==B`, and mask shape; `DeepSetsComposition` checks bool mask dtype. The point path has no general explicit finite/dtype check and does not independently validate the shape/uniqueness/validity correspondence of `component_ids`; tensor layers may raise framework runtime errors for incompatible tensor dtypes. Those are gaps in retained validation, not approval to trust malformed adapters.

The validity mask gates composition, but `component_points` still has padded slots. A future flattening adapter must select only valid component occurrences; it must not send padding as semantic entities.

### 3.4 Retained reducers and unavailable classifier behavior

**Existing engineering fact.** `models.losses.reduce_supervision()` accepts caller-supplied finite `[N,C]` values, targets, bool known mask, optional positive weights and optional trainable-class mask. It excludes unknown/fixed-masked entries, weights known positives only, normalizes each supervised instance over its known class weight sum, averages eligible instances, and reports counts. It computes no logits, membership, penalty formula or selected objective ([`models/losses.py:27–88`](../../src/town_model/models/losses.py#L27)). If no row is supervised it returns a connected numerical zero plus false eligibility flags; that return must not be mistaken for a defined enabled training task.

`training.supervision.reduce_weighted_instances()` similarly reduces caller-supplied `[N,C]` values against known targets and optional positive weights, returning per-instance values and eligibility ([`training/supervision.py:43–69`](../../src/town_model/training/supervision.py#L43)). `aggregate_effective_batch()` accepts per-image values/eligibility for component and pizza tasks, averages eligible images separately, does not renormalize remaining task coefficients, reports undefined tasks, and raises `UndefinedImageObjective` if no enabled contribution exists ([`supervision.py:72–113`](../../src/town_model/training/supervision.py#L72)). It expects the caller to compute entity-level and component-within-image reductions first; it does not calculate objective penalties, the per-image component average, positive-weight policies, or comprehensive support diagnostics. `reduce_weighted_instances()` checks matching shapes, bool masks, known-target binary values, weight shape and finite-positive weights, but does not explicitly validate `values` rank or finiteness. `aggregate_effective_batch()` rejects an empty microbatch list and raises when no enabled task contributes, but relies on caller-aligned per-image tensors and does not validate coefficient nonnegativity or finite values. The aggregation tests demonstrate caller-value behavior and gradient parity ([`test_phase3_models.py:79–177`](../../tests/test_phase3_models.py#L79)).

**Existing engineering fact.** `ClassifierBackendUnavailableError` is the central fail-closed exception ([`classifier_backend.py:8–24`](../../src/town_model/classifier_backend.py#L8)). T5 confirms classifier-dependent training, evaluation, resume, interpretation, worker and dashboard actions reject without publishing or mutating artifacts. Current regression coverage includes API and CLI paths ([`test_classifier_backend_unavailable.py:103–199`](../../tests/test_classifier_backend_unavailable.py#L103)). T3/T5 record these historical active-source modules as removed: `models/classifiers.py`, `models/geometry.py`, `models/image_embeddings.py`, `models/classifier_transport.py`, `models/image_model_validation.py`, `training/image_objectives.py`, `experiments/synthetic.py`, and `dashboard/geometry.py`. Retired public symbols are explicitly absent in [`test_classifier_backend_unavailable.py:82–100`](../../tests/test_classifier_backend_unavailable.py#L82); the [T3 removal report](T3_REMOVAL_REPORT.md) and [T5 review](../../reviews/classifier_teardown_t5_final_acceptance.md) provide the retirement evidence. This historical list is not a future design spec. No active classifier family, geometry implementation, classifier-specific objective, or image classifier result DTO remains.

## 4. Proposed minimal typed classifier interface

Everything in this section is **Proposed**, not approved. It describes a contract shape, not implementation code.

### 4.1 Core point input and result variants

**Proposed.** A pure classifier call takes exactly a flat point tensor plus classifier-owned parameters/state and a model-level domain configuration. It does not accept `ImageBatch`, `VisionFeatureBatch`, targets, masks, ontology objects, recipe labels, detector IDs/confidences, ROI records, or a point-to-image structure.

```text
PointTensor: float tensor [N,d], with d fixed by classifier state; N policy is unresolved (D2-A)

AffineResult:
    trainable_class_ids: ordered tuple[int, ...], length C_train
    contains: bool tensor [N,C_train]
    affine_logits: float tensor [N,C_train]
    fixed_domainthing_contains: bool tensor [N]       # presentation unresolved

GeometryResult:
    family: Literal["box", "town"]
    trainable_class_ids: ordered tuple[int, ...], length C_train
    contains: bool tensor [N,C_train]
    signed_coordinate_clearance: float32 tensor [N,C_train]
    fixed_domainthing_contains: bool tensor [N]       # presentation unresolved
```

Use a discriminated result union (`AffineResult | GeometryResult`), not a generic `scores` field. Affine state has weights `[C_train,d]` and biases `[C_train]`; `affine_logits = points @ weights.T + bias`. For affine results, `contains == (affine_logits >= 0)` per F4.2. Geometry results carry a `box`/`town` discriminator. For geometric results, `contains` is exact Boolean set membership; `signed_coordinate_clearance` is the separately typed F5 measurement. Do not derive geometric membership by thresholding clearance, and do not call clearance Euclidean distance. No probabilities or loss values belong in either result.

**Proposed D1 invariant — author approval required; not an additional already-approved F0–F12 decision.** The exact ordered trainable class-ID tuple that assigns parameter rows must be persistently bound to the classifier's training identity, together with the relevant canonical class-vocabulary/manifest identity (for example, the canonical vocabulary manifest hash). Save that binding with the bound model/run/checkpoint. Training, inference, and checkpoint restoration must validate the saved binding against the expected canonical class vocabulary by exact tuple and identity equality; identical class counts are insufficient. A reordered or mismatched binding must fail explicitly before checkpoint/model/optimizer state is mutated, never silently relabel or permute trained rows. A pure mathematical kernel may remain class-count-only if a clearly identified outer binding layer enforces and persists this invariant. Whether the binding is owned by classifier state or an immutable bound wrapper remains an unresolved D1 author choice; either owner must serialize it with the model/run/checkpoint. Output metadata alone is not sufficient to establish the order under which parameter rows were trained.

**Proposed, pending D1.** Represent fixed `DomainThing` separately from trainable columns as one bool per point, always true for valid points in the selected domain. The field is shown separately because F4.4 says `C_train` and F4.3 says `DomainThing` has no trainable parameter. Whether the fixed predicate is returned as a separate vector, a separate typed fixed-class record, or materialized by an outer presentation adapter is not settled here. Do not append it to `contains:[N,C_train]` without author confirmation.

### 4.2 Shape, dtype, ordering, and failure rules

**Proposed, pending explicit D2-A–E author approval.** For a classifier bound to latent dimension `d` and trainable class count `C_train`:

- points have exact rank/shape `[N,d]`; parameters and corners have their exact declared shapes; no scalar, `[d]`, `[B,K,d]`, accidental transpose, implicit expansion, or broadcasting;
- `d≥1` is the **Proposed** D2-C recommendation, not an approved rule. `N=0` and `C_train=0` remain explicit D2-A/B choices; see §5;
- membership is `torch.bool [N,C_train]`; affine logits have shape `[N,C_train]` and an explicitly agreed floating dtype; geometry clearance is computed/returned as `torch.float32 [N,C_train]` under approved F7.3;
- each result's ordered IDs have length `C_train`, are unique, and match the persisted D1 binding exactly; the class family selects exactly one numeric measurement field;
- reject malformed rank/shape, incompatible dimensions, duplicate/mismatched IDs, nonfinite values, invalid geometry, unsupported dtype/device, or values outside the declared numerical envelope. No implicit cast, clipping, repair, score fallback, axis reordering, or broadcast.

**Proposed D2-E validation policy.** Require exact rank and shape, finite values, and points/parameters on the same explicitly supported device; reject device mismatch instead of silently transferring tensors. Enforce the declared finite operating envelope for float32 geometry calculations. F7.3 fixes geometry transforms/evaluation/loss to float32 and requires a tested finite envelope; it does **not** scientifically fix affine parameter/logit dtype or an affine envelope. Strict float32 affine inputs, parameters, and logits are a **Proposed** initial policy only; an explicitly selected floating dtype is an alternative. Author approval is requested for affine dtype/envelope, the supported-device set, and the exact validation policy. Silent casting can change threshold/clearance behavior, broadcasting can hide class/dimension errors, and clipping or numerical repair can change the queried point or geometry.

**Proposed.** Domain mode is part of classifier/model state and run/checkpoint identity. Bounded points/corners are expected within `[-1,1]^d`, with closed-face equality valid; unbounded mode must not clip to a finite substitute. D2-D distinguishes a pure mathematical membership query outside `Δ` from an image/classifier adapter receiving an invalid point for its configured bounded domain. Strict adapter rejection is recommended, not approved; see §5 and §11.

## 5. Empty, domain-invalid, and numerical input boundary

The mathematical and engineering layers are distinct. The following observations were verified at the fixed starting SHA; reducer/point-path observations are not new scientific approvals. D2-A–E are separate unresolved author choices. Every recommendation below is **Proposed**, not an approved requirement.

| Case / decision | Existing behavior / approved mathematics | Alternatives, consequences, and proposed default |
|---|---|---|
| **D2-A — flat point classifier `N=0`** | **No classifier exists**, so no current classifier behavior. `ImageBatch.validate()` explicitly rejects `B=0` (`contracts.py:41–46`). A pure evaluator is not an `ImageBatch`; `[0,d]` can have a shaped empty result axis. | Author choice: accept `[0,d]` and return correctly typed `[0,C_train]` outputs, or reject zero rows. Acceptance simplifies empty mathematical evaluations and adapters with no selected entities, but requires tests for every result variant/reducer boundary; rejection simplifies nonempty batching but forces adapters to filter or fail before calls. **Proposed recommendation:** pure evaluators accept `N=0`. Author approval requested. |
| **D2-B — `C_train=0`** | **No active classifier contract.** Generic reducers were probed with `[N,0]`; they return empty class-axis results and no eligible instances. This establishes neither classifier nor training semantics. | Author choice: permit zero trainable columns in a pure evaluator, with fixed `DomainThing` still separate, or require at least one trainable class. Permitting it gives a well-shaped empty class axis but requires an empty persisted D1 binding; requiring one rejects empty vocabularies early. **Proposed recommendation:** permit `C_train=0` for pure mathematical evaluation only. Irrespective of tensor shape or this choice, absence of eligible training supervision remains undefined under F8.4; it must not become a fabricated zero objective. Author approval requested. |
| **D2-C — latent dimension `d=0`** | F3.10 permits useful degenerate fixed-corner fixtures but does not approve zero latent dimensions. Retained `ImagePointModel` rejects `d<1`; that is an engineering fact for that image model, not an author decision for future pure evaluators. | Author choice: reject `d=0` as an invalid classifier dimension, or define an explicit zero-dimensional mathematical contract. For `d=0`, affine weights would have shape `[C_train,0]` and logits would be bias-only; geometric `min` over coordinates is undefined unless a new convention is supplied. Image projection heads currently require `d≥1`; broader supported dimensions must be stated by each future evaluator. **Proposed recommendation:** reject `d=0` and require `d≥1`. Author approval requested. |
| `B=0` image batch versus `N=0` flat points | `ImageBatch` and `collate_image_records()` reject empty image batches. A direct `ImagePointModel` probe at the starting SHA accepted `B=0` tensors, but no test establishes it as a supported public contract. | Preserve the existing `ImageBatch B=0` restriction as a separate engineering fact; it neither settles nor implies D2-A. Do not change the image batch rule in this draft. |
| Zero component occurrences `K=0` | **Existing engineering fact:** for `B>0`, `ImagePointModel` returns component points `[B,0,d]`; DeepSets sums the empty set to zero before applying `rho`, and compositional pizza points remain `[B,d]`. Covered by [`test_phase3_models.py:52–76`](../../tests/test_phase3_models.py#L52). This is not geometry window count `W`. | Preserve current `K=0` support. A future adapter selects zero component entities; an independently selected pizza point remains. No padding or synthetic component is added. |
| Geometry windows `W=0` | **Already approved:** F3.11 defines zero windows and `H=1,W=0` as one-box equivalence; F5.3 sets `s_H=s_F`. The old evaluator was removed, so there is no current execution behavior. | Preserve approved `W=0` mathematical semantics. Proposed pure evaluator tests compare membership and clearance with the box case; no dummy window is introduced. |
| Single house/no-window | **Already approved:** exact set equivalence, not merely a close numerical match. | Proposed property tests compare `contains` exactly and clearance with the approved F5.3 composition. |
| Degenerate fixed corners | F3.10 permits fixed degenerate evaluator fixtures where useful; trainable primitive widths remain positive. | **Proposed:** a pure fixed-corner evaluator may admit equal endpoints (`l≤r`) when declared and must reject reversed endpoints. This does not imply `d=0` is supported. |
| **D2-D — bounded-domain external query versus adapter violation** | F2.7 distinguishes bounded `Δ=[-1,1]^d` from `ℝ^d`; no active classifier chooses external-query behavior. | For a pure mathematical query, either reject points outside `Δ` as outside its input contract or define membership against a set contained in `Δ`, returning false outside that set. Separately, an image/classifier adapter configured for bounded `Δ` may strictly reject an encoder point outside its configured domain, or pass it through as a non-member while reporting the domain violation. Treating these as the same policy can hide a bad point head. **Proposed recommendation:** strict adapter rejection before scoring; pure evaluator policy must be explicit and may return false only under its declared set-query contract. No clipping. Author approval requested. |
| **D2-E — rank, finite values, dtype, device, numerical envelope** | F7.3 **already fixes geometry transforms/evaluation/loss to float32** and requires an explicit tested finite operating envelope. F4 does not set affine parameter/logit dtype. Retained point/reducer checks do not establish a future classifier's validation contract. | Alternatives include exact-rank/shape and finite/envelope checks with explicit dtype/device rules, or relying on framework errors and implicit conversion/broadcast. **Proposed recommendation:** require exact shapes, reject nonfinite/out-of-envelope inputs and unsupported dtype/device, require points and parameters on the same declared supported device, and never silently cast, broadcast, clip, or repair. Use float32 for geometry as approved; proposing float32 affine inputs/parameters/logits is not scientifically fixed and requires author approval. Silent casts can change threshold/clearance behavior; broadcasting can conceal axis mistakes; clipping/repair changes the input or geometry. Author approval requested for the exact policy, affine dtype/envelope, and supported-device set. |

## 6. Bounded/unbounded points versus retained `tanh` projections

**Already approved.** F2.7 requires both `[-1,1]^d` and `ℝ^d` as scientifically distinct domains. F12's bounded width-first and unbounded center/half-width transforms specify trainable **geometry** parameterization, not the image point encoder ([F12.2.1–2.2](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L40)). F7.6 requires the two geometry-domain transforms to preserve conceptual physical `(c,h)` and permit physically matched starts; F3.7 likewise requires matched physical initial coordinates for contained/independent window comparisons ([F7.6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L87), [F3.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L44)).

**Existing engineering fact.** All point-producing paths in retained `ImagePointModel` currently end in `tanh` (section 3.3). They therefore emit bounded points; they do not implement the approved unbounded point-domain run. This does not mean the approved unbounded domain is invalid, and it does not make current projections domain-independent.

**Proposed, requires separate author approval and its own implementation/review contract.** Add a domain-selected terminal activation policy while preserving the point-only call boundary and all current classifier-free guarantees:

1. Keep the existing bounded path exactly as-is, including terminal `tanh` at component, global, composition, and hybrid outputs.
2. For a genuine `ℝ^d` image-point run, replace only those final output `tanh` activations with identity/linear outputs in all point-producing heads that can contribute to the selected mode. Do not add post-hoc scaling, clipping, or a finite “unbounded” range. Do not change inputs, occurrence IDs, masks, composition mode, or add classifier semantics to `ImagePointResult`.
3. Put the explicit domain/activation choice in configuration and model/run/checkpoint identity. Tensor parameter shapes may remain load-compatible, but bounded checkpoint weights interpreted through identity activations do **not** preserve predictions or model meaning. Do not auto-convert/relabel a bounded checkpoint as an unbounded checkpoint; require explicit provenance/initialization for each domain.
4. Preserve trainability of the projection heads; gradients remain differentiable through their terminal mapping. This is a material point-architecture behavior change even if replacing `Tanh` with `Identity` adds/removes no parameter tensors. Obtain author approval and a separate bounded implementation/test/review task before changing `image_points.py`.
5. For scientific comparisons, keep domain modes separate and declared. Match geometry's physical `(c,h)` starts as F7.6 requires; preserve F3.7 matched physical corners across window modes. A common seed and compatible shared parameter initialization for encoder heads is a **recommended proposed control**, not an existing approval that initial point outputs must match exactly. Record actual initial point coordinates, occupancy/range and geometry corners; report that the final activation is the domain treatment. Do not use output rescaling to force apparent matching.

**Unresolved D3.** The accepted sources do not prescribe a point-head unbounded activation or a point-encoder matched-initialization protocol. Alternatives include (a) separate bounded-tanh and unbounded-linear point heads as proposed; or (b) keep the current point architecture and leave unbounded image integration unavailable while pure supplied-point evaluators support `ℝ^d`. Alternative (b) preserves architecture/checkpoint behavior but does not satisfy unbounded image-point integration. D3 blocks unbounded image integration and its training, not pure physical-corner evaluators or a flat classifier over explicitly supplied points. No projection changes are made by this draft.

## 7. Point-only image integration adapter boundary

**Already approved.** Point inference is primarily from RGB/ROI visual evidence. Recipe/ontology targets, detector categories and detector confidences are not semantic point inputs; pixel-derived ROI coordinates are a distinct authorized visual feature ([F2.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L30)).

**Existing engineering fact.** `SharedVisualFeatureExtractor` carries ROI visual features and normalized ROI coordinates to the point model, but not targets/categories/confidences as point tensors. `ImagePointResult` preserves sample IDs, padded occurrence IDs, validity mask and all component/pizza point branches. Global, compositional and hybrid modes choose `final_pizza_points` differently; none emits semantic membership.

**Proposed.** A later, separately reviewed adapter—not the mathematical classifier—will:

1. Select `component_points[b,k]` only where `component_valid_mask[b,k]` is true and its occurrence ID is present; reject inconsistent mask/identity/shape alignment rather than guessing. Each selected occurrence retains `(sample_id, occurrence_id)` and contributes exactly one row.
2. Select `final_pizza_points[b]` per image when the requested task is pizza-level; retain `sample_id` and the selected `pizza_mode`. Do not accidentally classify `global_pizza_points` or `composed_pizza_points` in addition to final points unless a separately approved task explicitly requires it.
3. Concatenate selected points into flat `[N,d]` and carry an external row-to-entity map for output/target reconstruction. Inverse mapping uses this explicit map, never arithmetic assumptions about ragged offsets or incidental batch order.
4. Preserve gradients by using differentiable indexing/concatenation; do not detach, convert through NumPy, or recompute point values. Tests must prove a classifier/objective gradient flows through selected valid points and not through discarded padding.
5. Keep labels/masks on the objective adapter side, aligned by dataset-owned IDs. Targets, ontology recipe information, detector categories/confidences, and target-derived ordering must never enter point values or select which point a class sees.

**Unresolved.** The exact adapter record/class ownership and whether component and pizza entities are evaluated in one concatenated flat call or in separate calls are not fixed by F4.1. Both can preserve the mathematical flat interface if each row has an explicit identity map. Recommended default: one call per semantic entity/task batch or a combined flat call with typed entity references; choose based on integration needs without changing classifier semantics. This blocks image integration/training only; not pure flat classifiers or point-independent evaluators.

## 8. Objectives, reducers, masks, and diagnostics

**Already approved.** The five primary objectives remain affine+BCE-with-logits, box+signed logistic, box+signed hinge, town+signed logistic, town+signed hinge, unregularized. F8.1 requires known-weight-normalized per-entity penalty means, component aggregation within eligible images and then across eligible images, separate pizza aggregation, and omission of zero-supervision entities/images rather than fake zeros. F8.2 keeps unit weights primary; named positive-weight variants use known-positive TRAIN counts separately for component and pizza levels, while known negatives remain weight one and unknowns mask out. F8.4 omits/logs undefined enabled tasks without renormalizing other coefficients and fails when all enabled objectives are undefined. See [F6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L68) and [F8](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L93).

**Existing engineering fact.** The retained reducers already provide generic caller-value, mask, positive-only-weight and eligibility/aggregation seams; they do not interpret classifier outputs or choose losses. Some F8 structure is caller-owned: a future adapter/objective must calculate per-label penalties, known-trainable target/mask projection, within-image component averaging, optional train-count weight vectors and diagnostics before calling generic aggregation. Current weight helpers do not calculate F8.2 formulas. `aggregate_effective_batch()` reports eligible-image counts and undefined task names but does not itself log them; per-entity label counts can be sourced from `SupervisionReduction` or computed by an adapter. No retained function here alone demonstrates full F8 compliance.

**Proposed ownership sequence:**

1. Dataset-bound adapter projects known labels/masks into the fixed trainable class order; fixed/unsupported columns are excluded, and F11.1 forbids inferred cross-type negatives.
2. Family-specific objective receives the correctly typed measurement and known target per label, computes exactly the approved per-label penalty, and returns unreduced `[N,C_train]` penalties plus any independently computed diagnostics. The classifier does not compute losses.
3. Generic reducer masks unknowns, applies only an explicitly configured positive weighting policy, and reduces each eligible entity. No extra positive weighting or regularization is implicit.
4. Image aggregator averages supervised components within image, then eligible images; pizza entities are aggregated independently. Undefined-task reporting/failure follows F8.4; no task-coefficient renormalization.
5. Trainer/optimizer integration remains a later task; evaluation independently checks hard `contains` values and does not infer correctness from training penalties.

**Readiness assessment:** existing reducers are reusable and broadly respect F5 ownership. They need a separately scoped adapter/conformance audit to establish F8.1/F8.2/F8.4 end-to-end; they do not need to be rewritten merely because a classifier interface is designed. F8.5 accumulation behavior and F12 optimizer transactions are outside pure reducer reconstruction.

## 9. Module ownership and dependency direction

The following is **Proposed engineering organization**, not a scientific requirement:

```text
point/geometry/loss-free classifier math
    imports: tensor/runtime primitives + minimal classifier DTOs only
    accepts: [N,d] points + own state/domain
    returns: typed membership and family measurement

image-point adapter (later integration task)
    depends on: ImagePointResult, explicit entity IDs/masks, dataset class binding
    performs: valid-point selection, row identity map, output reconstruction
    does not become a classifier input dependency

family objective (later task)
    depends on: typed classifier result + target/mask inputs
    returns: unreduced per-label penalties; no optimizer updates

generic reducer (retained)
    depends on: values, targets, known masks, explicit weights/eligibility
    performs: mask/weight/aggregation only

training / evaluation orchestration (later tasks)
    depends on: accepted classifier + objective + adapter/reducer
    operational CLI/API/dashboard wiring remains disabled until separately accepted
```

Keep dataset/ontology classes out of classifier math modules; keep image/ROI and presentation imports out of classifier math. If a shared DTO is needed, place it in a low-level contract module with no reverse dependency. The current `models/__init__.py` exports only classifier-independent point/reducer surfaces ([`models/__init__.py`](../../src/town_model/models/__init__.py)); this draft does not add exports or reconnect imports.

## 10. Fail-closed and deployment boundary

**Already accepted operational fact.** Until later separately reviewed integration explicitly reconnects an approved backend, classifier-dependent training, evaluation, checkpoint reconstruction, resume, interpretation, snapshots, relation queries, workers, CLI/API and dashboard actions remain unavailable through `ClassifierBackendUnavailableError`. T5 verified explicit rejection and no artifact publication/mutation. The issue design or a pure module existing does not authorize a runtime route.

**Proposed verification rule for every later pure-module task:** tests may import and call that module directly with explicit mathematical inputs, but must also exercise the **complete accepted T5 unavailable-operation matrix** as the comprehensive source for disabled routes ([T5 §J — Unavailable-operation matrix and artifact non-publication](../../reviews/classifier_teardown_t5_final_acceptance.md#j-unavailable-operation-matrix-and-artifact-non-publication)). Do not substitute an abbreviated illustrative list. This includes `export-geometry` and all listed training, evaluation, resume, checkpoint reconstruction/evaluation, interpretation/snapshot, relation, worker, CLI/API, and dashboard surfaces. Unauthorized direct API calls must raise `ClassifierBackendUnavailableError`; CLI surfaces must retain their normal fail-closed error presentation. At the operation boundary, unauthorized `export-geometry` must raise `ClassifierBackendUnavailableError` (with the CLI wrapper retaining its normal unavailable exit). Geometry-export rejection must create no files/artifacts and must not mutate any pre-existing output or state; the same before/after no-side-effect check applies across the matrix. Do not turn a pure evaluator into a hidden dispatch fallback. Reconnection requires its own implementation scope, approval, review, and positive-path/no-side-effect test matrix.

## 11. Source-linked decision register for author review

The decisions below are the remaining interface/engineering boundaries; they do not reopen F0–F12 mathematical choices. Recommendations are not approvals.

### D1 — Persist and validate class-column binding; present fixed `DomainThing`

- **Status:** Dataset vocabulary/order is an **Existing engineering fact**. The parameter-row binding invariant and fixed-class result representation are **Unresolved**; the binding below is **Proposed**, not an additional already-approved F0–F12 decision.
- **Exact boundary:** The exact ordered trainable class-ID tuple used to assign parameter rows, plus the relevant canonical class-vocabulary/manifest identity (for example, a stable vocabulary manifest hash), must be bound to classifier training identity and saved with its bound model/run/checkpoint. Training, inference, and checkpoint restoration validate exact equality against the expected vocabulary. A mismatch or reorder fails explicitly before state mutation; equal class counts do not suffice, and trained rows are never silently relabeled or permuted. A class-count-only pure mathematical kernel is allowed only if a clearly identified outer layer enforces this persistent binding. Separately decide whether fixed `DomainThing` is a `[N]` output, a typed fixed-class record, or materialized by an outer presentation adapter; keep it outside trainable columns.
- **Evidence:** [F4.1–F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54); [`PreparedDataset` and trainable-order projection](../../src/town_model/dataset/image_data.py#L22); [manifest vocabulary construction](../../src/town_model/dataset/ontology.py#L450); [order/exclusion tests](../../tests/test_phase3_dataset.py#L20).
- **Alternatives:** (a) a pure kernel stores only `C_train`, with an immutable bound wrapper owning and serializing the exact tuple/manifest identity; (b) classifier state owns and serializes that binding. The precise ownership—classifier state versus immutable bound wrapper—remains an author choice, but either option must save and validate the binding. For `DomainThing`, retain the existing alternatives above.
- **Consequences:** An adapter that attaches its current IDs only to outputs cannot identify the training-time order; a same-width checkpoint could be silently relabeled. Exact validation makes class meaning reproducible and rejects same-width permutations/manifest changes without mutating state. Fixed/unsupported classes among trainable columns would violate the F4 boundary.
- **Proposed recommendation:** Keep the mathematical kernel class-count-only if desired, but make a persisted immutable binding at the bound classifier/run/checkpoint layer mandatory; validate exact ordered tuple and canonical vocabulary identity during training, inference, and restoration before using or mutating state. Keep `DomainThing` separate and unresolved until author approval.
- **Author approval requested:** approve or amend the binding invariant and its owner, and separately choose the fixed `DomainThing` representation. The test contract must cover correct restoration, same-width permutations, manifest mismatches, and rejection without state mutation.
- **Blocks:** dataset-facing classifier results, training/inference identity, and checkpoint mapping. Does **not** block class-free pure box/town evaluators. It does not require pure mathematical kernels to import ontology or dataset metadata.

### D2 — Cardinality, domain, and input-validity author choices

- **Status:** D2-A through D2-E are all **Unresolved**. Each recommendation is **Proposed** and requires explicit author approval; none is an existing classifier requirement. No active classifier settles these choices.
- **Evidence:** [F4.1/F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54), [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33), [F7.3–F7.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L84), [F8.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L98); `ImageBatch` nonempty validation at [`contracts.py:41–46`](../../src/town_model/contracts.py#L41); empty-K tests at [`test_phase3_models.py:70–76`](../../tests/test_phase3_models.py#L70); reducer shape behavior at [`models/losses.py:27–88`](../../src/town_model/models/losses.py#L27).

**D2-A — `N=0`.** Should a pure classifier accept `[0,d]` and return correctly shaped empty outputs, or reject zero rows? Acceptance permits empty mathematical evaluations and empty adapter selections but requires typed empty-output tests; rejection simplifies batching but requires adapters to filter or fail. **Proposed recommendation:** accept `N=0` for pure evaluators. Author approval explicitly requested. Keep `ImageBatch B=0` rejection as a separate existing engineering fact; it does not answer this question.

**D2-B — `C_train=0`.** Should a pure evaluator allow zero trainable columns while fixed `DomainThing` remains separate, or require at least one trainable class? Allowance creates `[N,0]` trainable outputs and an empty class binding; requiring one class rejects empty vocabularies at the interface. **Proposed recommendation:** allow `C_train=0` for pure evaluation only. Regardless of shape policy, lack of eligible training supervision remains undefined under F8.4; no empty tensor shape makes a training objective defined. Author approval explicitly requested.

**D2-C — `d=0`.** Should zero latent dimension be rejected or admitted under a fully specified zero-dimensional contract? For `d=0`, the geometric `min` over coordinates is undefined without a new convention; affine weights have shape `[C_train,0]` and logits become bias-only. Current image projections reject `d=0`, but that does not decide a future pure mathematical evaluator's policy. Supported dimensions beyond zero must be declared by each evaluator, not assumed. **Proposed recommendation:** reject `d=0` and require `d≥1`. Author approval explicitly requested.

**D2-D — bounded-domain external queries.** A pure mathematical membership query for a point outside `Δ` may reject it as outside the evaluator's input contract, or return false when the represented set is explicitly a subset of `Δ`. Separately, an image/classifier adapter receiving an out-of-`Δ` point despite being configured for bounded `Δ` may reject the invalid adapter value or pass it as a non-member with explicit violation reporting; these are not the same question. **Proposed recommendation:** strict adapter rejection before scoring; specify the pure-query behavior separately and never clip. Author approval explicitly requested.

**D2-E — input validity and dtype.** Specify exact rank/shape, finite-value, dtype, device, and numerical-envelope rules. **Already approved:** F7.3 requires float32 geometry transforms/evaluation/loss and a tested finite operating envelope. **Not fixed scientifically:** affine parameter/logit dtype and affine numerical envelope. **Proposed recommendation:** reject rank/shape mismatch and nonfinite/out-of-envelope values; require points and parameters on the same explicitly supported device; do not silently cast, broadcast, clip, or repair. Float32 affine behavior is a proposal, not an extension of F7.3. Explicitly approve or amend the affine dtype/envelope, supported-device set, and exact validation policy. Silent casting can change thresholds/clearances, broadcasting can conceal axis errors, and clipping/repair changes the queried values.

The retained `K=0` support and approved `W=0` semantics are separate and remain unchanged. These D2 choices concern pure classifier/evaluator and adapter input contracts; they do not authorize training or modify F8.

### D3 — Unbounded image points versus terminal `tanh`

- **Status:** F2.7's two physical domains are **Already approved**; current retained tanh projections are an **Existing engineering fact**; unbounded projection activation/checkpoint/comparison policy is **Unresolved** and the section 6 change is **Proposed**.
- **Exact boundary:** Whether and how the retained point projection is changed to emit genuine `ℝ^d` points for unbounded image runs while bounded runs retain `tanh`; specify checkpoint identity and matching-initialization comparisons.
- **Evidence:** [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33), [F7.6–F7.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L87), [F3.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L44), retained terminal `tanh` at [`image_points.py:15–96`](../../src/town_model/models/image_points.py#L15); F12 geometry transform does not define this encoder output head ([F12](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L40)).
- **Alternatives:** Domain-specific terminal `tanh` vs identity heads as proposed; or defer unbounded image integration and implement pure supplied-point `ℝ^d` evaluators first. A finite rescale/clip is not a valid unbounded alternative.
- **Consequences:** Identity outputs alter point values and semantics for same-shaped weights; parameter-tensor load compatibility is not semantic checkpoint compatibility. Deferral preserves current point model while withholding unbounded image integration.
- **Recommended default:** keep bounded checkpoint path unchanged; design a new identity-head unbounded path with domain in run/checkpoint identity, no implicit migration, common seed/parameter initialization, and actual coordinate/occupancy reporting. Require author approval and a separate point-architecture task/review before changing code.
- **Blocks:** unbounded image integration and training; not pure physical-corner evaluators or flat classifiers over caller-supplied points in either domain.

### D4 — Image entity flattening and reconstruction policy

- **Status:** Point and identity tensor shapes are **Existing engineering facts**; the proposed row adapter is **Proposed**; combined versus separate calls and identity container are **Unresolved**.
- **Exact boundary:** Define how valid component occurrences and selected final pizza points become flat `[N,d]`, how IDs/targets/outputs align, and whether entity kinds share a call.
- **Evidence:** [F2.4/F2.5/F4.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L28), [`ImagePointResult`](../../src/town_model/contracts.py#L149), valid mask and point creation [`image_points.py:133–168`](../../src/town_model/models/image_points.py#L133), zero/padded tests [`test_phase3_models.py:34–76`](../../tests/test_phase3_models.py#L34).
- **Alternatives:** One flat call with explicit typed entity references and inverse map; separate component and pizza calls with separate maps. In either case only valid occurrences are selected, identity is explicit, and gradients flow.
- **Consequences:** Incorrect flattening can score padding, drop/duplicate repeated occurrences, misalign labels, or detach gradients. Combined calls are not mathematically required by F4.1.
- **Recommended default:** choose the simplest adapter that carries an explicit `(entity_kind, sample_id, occurrence_id?)` row map; start with separate component/pizza calls if that simplifies target/aggregation alignment. This is adapter engineering, not classifier semantics.
- **Blocks:** image integration and training only. Does **not** block pure flat classifiers/evaluators.

### D5 — Reducer conformance boundary

- **Status:** F5.2/F8 are **Already approved**; retained caller-value reducers are an **Existing engineering fact**; exact future image adapter is **Proposed**.
- **Exact boundary:** Confirm no reducer rewrite is required before pure evaluators, and specify a separate later conformance task for target/known-mask projection, per-label objective values, entity/image hierarchy, configured weighting, undefined tasks and diagnostics.
- **Evidence:** [F5.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L62), [F8.1–F8.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L93), [`reduce_supervision`](../../src/town_model/models/losses.py#L27), [`reduce_weighted_instances`/`aggregate_effective_batch`](../../src/town_model/training/supervision.py#L43), aggregation tests [`test_phase3_models.py:143–177`](../../tests/test_phase3_models.py#L143).
- **Alternatives:** Reuse reducers behind a thin explicit adapter with additional conformance tests; or design a separate adapter module if current caller APIs cannot represent F8 without changing generic reducer meaning.
- **Consequences:** Placing penalties in the classifier violates F5.2; rewriting generic reducers prematurely risks changing retained behavior. Existing helpers do not compute primary losses or F8.2 train-count weight formulas.
- **Recommended default:** retain generic reducers; later add objective/adapter conformance tests and only change reducer internals if a falsified criterion requires it. No new objective or regularization.
- **Blocks:** full supervised image training/aggregation conformance, not pure evaluators or flat membership implementation.

**Existing separate author action, not a new question:** #3's source-linked set/membership proposal still awaits explicit author confirmation. The A1 citations above remain approved authority; this draft neither resolves #3 nor treats its pending status as permission to alter F3 semantics. The author should confirm/amend #3 separately before any task claims to settle that issue.

## 12. Minimum independent implementation/review sequence (planning only)

No implementation task is authorized by this draft. After the author explicitly approves the relevant R2 contract items (and any applicable issue #3 confirmation), readiness should be tracked per module, not as one all-or-nothing image-training milestone.

1. **Pure physical-corner box Boolean evaluator.** Inputs are supplied finite physical corners/points and explicit `Δ`; no raw-to-physical transform, class IDs, dataset, image path, trainable parameterization, or optimizer. Verify exact closed-face membership, corner order/dimension/domain failures, fixed degenerate fixtures where approved, the empty-point behavior selected by D2-A, and class-free behavior. Scientific basis: F3.2/F3.4/F7.4; readiness says this is not blocked by P0-A/B/C.
2. **Box signed-coordinate clearance kernel.** Add `min_j(x_j-l_j,r_j-x_j)` in float32 physical coordinates, demonstrably non-Euclidean, separate from Boolean membership. Hand-computable interior/boundary/exterior cases; dimension/finite/range tests. Scientific basis: F5.1/F5.6.
3. **Pure house/town membership and exact clearance composition.** Supplied corners only; F3.6-R boundaries; contained and independent windows share semantics; `W=0` and `H=1,W=0` exact equivalence; `s_H`/`s_T` exact min/max. No geometry transforms, trainable params, source ontology, or optimizer. Scientific basis: F3.2/F3.5/F3.6-R/F3.11/F5.3/F5.5.
4. **Pure affine flat-point evaluator.** Only after the approved class-axis and empty/dtype contract is recorded. Test exact formula, `z≥0` membership (including zero), class-column order, `N=0`/`C=0` policy, finite/dimension errors. No probabilities or objective.
5. **Pure geometry-classifier binding/output DTO.** Only after D1–D3 decisions needed by its scope; typed result variants, class-axis metadata, fixed `DomainThing` handling, and cross-domain compatibility checks. No image adapter.
6. **Image point/domain adapter.** Separate task after D1–D4 and, for unbounded image points, D3's point-architecture approval. Test valid-occurrence flattening, IDs/targets/output round-trip, padding exclusion, no label/category/confidence leakage, and gradient flow.
7. **Objective/reducer adapter.** Separate F8 conformance task after approved typed outputs. Compute only the five primary penalties already approved; test known/unknown masking, weighting and entity/image reductions, undefined supervision, and diagnostics. No optimizer lifecycle.
8. **Training/evaluation/route reconnection.** Deferred to independently scoped, approved integration tasks after objective, F12 lifecycle, evaluation protocol and operational safeguards are accepted. Do not reconnect T5-disabled surfaces as part of steps 1–7.

The first three proposed slices can be reviewed independently from P0-A/B/C because they accept supplied physical geometry and do not train it. `W=0` is mathematically approved, but its pure evaluator still needs its own engineering contract/tests. No F12 transform, optimizer transaction, detector, Phase 4, or learning claim is implied.

## 13. Verification checklist for this draft and later component contracts

| Component | Proposed verification evidence (not run as implementation tests here) |
|---|---|
| Typed output contract | Assert exact `[N,C_train]` shapes, bool membership, one family-specific numerical field, exact bound class-axis identity, separate fixed output; test zero axes only according to the author-approved policy and reject ambiguous variants. |
| Class-axis binding | Persist the exact ordered trainable-ID tuple and canonical vocabulary/manifest identity (for example, its canonical vocabulary manifest hash) with the bound model/run/checkpoint; validate during training, inference, and restore before state mutation. Test correct restoration, same-width permutations, manifest mismatch, explicit rejection, and no relabel/reorder. Prove fixed/unsupported IDs have no trainable columns. |
| D2 cardinality/domain/input policies | After author approval, test the selected `N=0`, `C_train=0`, and `d=0` policies; keep `ImageBatch B=0` rejection separate and `K=0` support/`W=0` semantics intact. Distinguish pure out-of-`Δ` queries from bounded-adapter violations. Test exact rank/shape, finite values, geometry float32/envelope, explicitly approved affine dtype/envelope, same-device policy, and rejection without implicit casts, broadcasting, clipping, or repair. Verify no-supervision stays undefined under F8.4 regardless of empty tensor shapes. |
| Pure box evaluator/clearance | Hand-computable corners, faces, outside, degenerate fixed geometry if enabled, dimensions/domain, finite float32 operating envelope, and `N=0` behavior per approved D2-A; demonstrate signed coordinate clearance is not Euclidean and is not membership. |
| House/town evaluator | Window interior vs boundary, frame boundary, overlapping-window exclusion, independent external windows, exact union, `W=0`, single-house equivalence, min/max clearance; no trainable-width inference from degenerate fixtures. |
| Affine evaluator | Exact weight/dot/bias values, threshold equality at zero, row mapping under the bound class identity, approved `N=0`/`C_train=0`/`d=0` policies, invalid ranks/dtypes/nonfinite inputs; no softmax/probabilities. |
| Point adapter | ID/mask alignment, all and only valid occurrences, repeated occurrence identity preservation, final pizza mode selection, row-map inverse, no detach, zero-padded gradients, no labels/categories/confidence as point inputs. |
| Domain-specific image heads | Bounded path remains numerically/checkpoint compatible; unbounded path has no terminal `tanh`/clip/scale; both retain trainable gradients; checkpoint/run identity distinguishes activation/domain; matched initialization evidence reports point distributions/geometry starts. |
| Reducer/objective adapter | Per-label penalties originate in objective; unknown/fixed/unsupported exclusion; positive-only configured weights; F8 hierarchy and accumulation gradient parity; undefined tasks and counts; no new regularizer/objective. |
| Deferred differentiable-clearance contract | Carry F5.5's framework automatic-differentiation tie-subgradient and tie/branch-frequency diagnostics into later differentiable-clearance acceptance. This is not a prerequisite for non-trainable pure evaluators. |
| Fail-closed boundary | Run the complete T5 §J unavailable-operation matrix, not an illustrative subset. Explicitly include `export-geometry`; direct unauthorized calls raise `ClassifierBackendUnavailableError`, and every rejected route creates no files/artifacts and leaves existing outputs/state unchanged. Preserve training, evaluation, resume, checkpoint reconstruction, interpretation/snapshot, relation, worker, CLI/API, and dashboard failures. |

## 14. Current unresolved author decisions and scope impact

**D1–D3 remain unapproved.** The recommendations below are proposals for author review, not requirements or new F0–F12 approvals. #3 remains separately pending explicit author confirmation.

1. **D1 — class binding and fixed output.** Decide whether classifier state or an immutable bound wrapper owns the exact ordered trainable-ID tuple and canonical vocabulary/manifest identity (including a stable manifest identity/hash). The proposed invariant persists this binding with the bound model/run/checkpoint and validates exact equality during training, inference, and restoration before state mutation; same-width permutations must fail. Separately decide the fixed `DomainThing` representation. This blocks dataset-facing classifier outputs and checkpoint/integration mapping, not class-free pure evaluators.
2. **D2 — complete author-choice frontier:**

   | Item | Unresolved alternatives | Proposed recommendation (not approved) / scope |
   |---|---|---|
   | **D2-A: `N=0`** | Accept `[0,d]` with correctly shaped empty outputs, or reject zero rows. | Accept in pure evaluators. Approval requested. Impacts mathematical evaluation, batching, adapters, and tests; keep `ImageBatch B=0` rejection separate. |
   | **D2-B: `C_train=0`** | Permit zero trainable columns with fixed `DomainThing` separate, or require at least one trainable class. | Permit for pure evaluation only. Approval requested. No eligible training supervision remains undefined under F8.4 regardless of valid shapes. |
   | **D2-C: `d=0`** | Reject as an invalid dimension, or define a zero-dimensional contract for affine shapes and geometric reductions. | Reject and require `d≥1`. Approval requested. The geometric coordinate `min` is undefined over no coordinates; current image-model rejection is not authority for pure evaluators. |
   | **D2-D: bounded external query** | A pure set query outside `Δ` may reject or return false under an explicit subset-membership contract. A bounded adapter may reject an out-of-domain point or pass it as non-member with a violation report. | Strict adapter rejection; specify pure-query behavior separately and never clip. Approval requested. |
   | **D2-E: validity/dtype/device/envelope** | Define exact rank/shape, finite checks, dtype, device, and numerical-envelope policy; distinguish geometry from affine. | Exact-shape/finite/envelope validation; points and parameters on one explicitly supported device; no silent cast, broadcast, clip, or repair. Float32 geometry and its tested envelope are already fixed by F7.3; float32 affine behavior and affine envelope are proposals requiring approval. |

   These policies block final flat-classifier input/empty-axis contracts as applicable, but do not make pure physical-corner evaluator planning depend on D4/D5 integration or optimizer policy. Preserve current `K=0` support and approved `W=0` semantics; no zero-supervision training objective is created by tensor shape.
3. **D3 — domain-specific image projection.** Approve or amend the proposed unbounded identity-head policy, or defer unbounded image integration. This blocks unbounded image adapter/training and point-architecture changes, not pure evaluators over supplied points or flat classifiers in an explicitly selected domain.
4. **#3 — existing explicit author confirmation.** Remains OPEN and unresolved here. Keep source-approved F3 choices intact; do not treat this draft, recommendations, or silence as resolution.

D4 (image entity adapter) and D5 (objective/reducer adapter conformance) remain later engineering concerns; they do not reopen classifier mathematics and are not prerequisites for the first pure evaluator slices. Optimizer/training policies likewise remain outside pure non-trainable evaluator planning. F11 audit A–J, final relation/statistical protocols, detector policy, and comparison-quality training remain outside R2 and must not be smuggled into this contract.

## 15. Repository and review boundary

This document was prepared on branch `docs/r2-classifier-interface-contract` from fixed starting revision `7ebc2b62c175288b0673524fbfae9f9d6f15a4ff`. It is a new draft under `docs/reconstruction/`, not an edit to current-status, acceptance, A1/F11/F12 sources, readiness authority, or historical review evidence. Issue #14 is the execution journal and is natively linked as a child of #2. The draft must receive independent documentation review before the author makes final contract decisions. No classifier, geometry, loss, optimizer, route reconnection, data generation, training, or scientific evaluation was implemented or begun.
