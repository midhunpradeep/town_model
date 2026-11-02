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
PointTensor: float tensor [N,d], with d fixed by classifier state and N ≥ 0

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

**Proposed.** Every output records enough immutable class-axis metadata to make column order explicit; the length of that order must equal `C_train` and all tensor columns. The mathematical kernel itself can remain class-ID agnostic and carry only `C_train` in its state; a thin immutable binding/adapter can pair the dataset-owned ordered IDs with its output. This keeps ontology-source metadata out of the mathematical point input while making output interpretation auditable. Whether IDs belong on the bound classifier object or only on the output wrapper is unresolved (D1).

**Proposed, pending D1.** Represent fixed `DomainThing` separately from trainable columns as one bool per point, always true for valid points in the selected domain. The field is shown separately because F4.4 says `C_train` and F4.3 says `DomainThing` has no trainable parameter. Whether the fixed predicate is returned as a separate vector, a separate typed fixed-class record, or materialized by an outer presentation adapter is not settled here. Do not append it to `contains:[N,C_train]` without author confirmation.

### 4.2 Shape, dtype, ordering, and failure rules

**Proposed.** For a classifier bound to latent dimension `d` and trainable class count `C_train`:

- points have exact rank/shape `[N,d]`; no scalar, `[d]`, `[B,K,d]`, accidental transpose, or broadcasting;
- `d≥1`; `N≥0`; output rows align exactly with input rows;
- membership is `torch.bool [N,C_train]`;
- affine logits are floating `[N,C_train]`; geometric clearance is `torch.float32 [N,C_train]` under approved F7.3;
- each result's ordered IDs have length `C_train`, are unique, and match the bound axis exactly;
- the class family selects exactly one numeric measurement field; ambiguous/missing fields fail rather than being guessed;
- wrong rank/dimension/class-axis length, duplicate/mismatched IDs, invalid domain points/corners, nonfinite values, unsupported dtype/device, or malformed geometry fail explicitly. No implicit cast, clip, repair, score fallback, axis reordering, or broadcast.

F7.3 already fixes geometry transform/loss/evaluation to float32 and a declared finite operating envelope. A matching float32 affine output/input policy is **Proposed**, not an F4 approval. Exact error class/text and the affine numeric dtype remain engineering details for author review (D2); this draft recommends typed validation errors (`TypeError` for dtype, `ValueError` for shape/domain, a documented numerical error for nonfinite/out-of-envelope values), not silent framework-dependent failures.

**Proposed.** A domain mode is part of classifier/model state and run/checkpoint identity, not a per-point metadata input. In bounded mode points/corners must satisfy `[-1,1]^d` and closed-face equality is valid. In unbounded mode do not clip to a finite substitute; still reject nonfinite or out-of-declared-float32-envelope inputs. Whether an out-of-domain point is a rejected adapter input or a valid negative membership query is an explicit D2 boundary; the recommended initial adapter behavior is reject, while the pure mathematical set evaluator can report false for a point outside `Δ` if that evaluator's contract expressly accepts it.

## 5. Empty and degenerate input boundary

The mathematical and engineering layers are distinct. The following observations were verified at the fixed starting SHA with the repo Python environment; reducer/point-path observations are not new scientific approvals.

| Case | Existing behavior / approved mathematics | Proposed R2 handling |
|---|---|---|
| Flat point classifier `N=0` | **No classifier exists**, so no current classifier behavior. `ImageBatch.validate()` explicitly rejects `B=0` (`contracts.py:41–46`). An independent pure evaluator is not an `ImageBatch`, and `[0,d]` has a well-defined empty result axis. | **Proposed:** accept `[0,d]` in pure evaluators and return correctly typed `[0,C_train]` outputs. Empty image batches remain rejected by the current image loader unless separately changed. |
| `C_train=0` | **No active classifier contract.** Both generic reducers were probed with `[N,0]`; they return empty class-axis results and no eligible instances. That does not establish classifier or training semantics. | **Proposed:** permit `[N,0]` in a pure evaluator if an empty trainable vocabulary is explicitly bound; fixed `DomainThing` remains separate. Any enabled training task with no trainable supervision is undefined and must fail/omit according to F8, not report a fabricated zero objective. Author confirmation requested in D2. |
| `B=0` image batch vs `N=0` flat points | `ImageBatch` and `collate_image_records()` reject empty image batches. A direct `ImagePointModel` probe at the starting SHA accepted `B=0` tensors and produced empty outputs, but there is no test establishing that as a supported public contract. | **Proposed:** do not infer classifier `N=0` policy from either image behavior. Keep image-batch `B=0` restriction; specify flat evaluator `N=0` separately. |
| Zero component occurrences `K=0` | **Existing engineering fact:** for `B>0`, `ImagePointModel` returns component points `[B,0,d]`; DeepSets sums the empty set to zero before applying `rho`, and compositional pizza points remain `[B,d]`. Covered by [`test_phase3_models.py:52–76`](../../tests/test_phase3_models.py#L52). This is not geometry window count `W`. | **Proposed:** flattening selects zero component entities; the independent pizza point, if selected by its task, remains. No padding or synthetic component is added. |
| Geometry windows `W=0` | **Already approved:** F3.11 defines zero windows and `H=1,W=0` as one-box equivalence; F5.3 sets `s_H=s_F`. The old evaluator was removed, so no current execution behavior exists. | **Proposed:** pure evaluator accepts an empty window collection and stores no dummy window parameters. Verify membership and clearance equivalence to the box case. |
| Single house/no-window | **Already approved:** exact set equivalence, not merely a close numerical match. | **Proposed:** property tests compare `contains` exactly and clearance with the approved F5.3 composition. |
| Degenerate fixed corners / `d=0` | F3.10 permits fixed degenerate evaluator fixtures where useful, but trainable primitive widths remain positive. The retained `ImagePointModel` rejects `d<1`. | **Proposed:** a pure fixed-corner evaluator may admit equal lower/upper endpoints (`l≤r`) as degenerate geometry when declared; it must reject reversed endpoints. Latent dimension `d=0` is not a valid classifier dimension. |
| Invalid rank, dtype, nonfinite, or incompatible domain | There is no classifier to establish current behavior. Point model validation is partial; `ImageBatch` has its own nonempty/finite rules. | **Proposed:** reject explicitly before scoring; do not transfer image-specific checks blindly or accept nonfinite values because a generic reducer accepts some tensor shape. Validate numerical operating range separately from mere finiteness. |

**Unresolved D2:** explicit author choice on whether `C_train=0` is a valid pure evaluator axis and whether out-of-domain points are rejected at the classifier adapter versus queried as definite non-members. Recommended: allow empty axes for pure evaluators, keep training undefined, and reject out-of-domain adapter inputs; this does not change set-membership semantics inside `Δ`.

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

**Already accepted operational fact.** Until later separately reviewed integration explicitly reconnects an approved backend, classifier-dependent training, evaluation, checkpoint reconstruction, resume, interpretation, snapshots, relation queries, workers, CLI/API and dashboard actions remain unavailable through `ClassifierBackendUnavailableError`. T5 verified both explicit rejection and no artifact publication/mutation. The issue design or a pure module existing does not authorize a runtime route.

**Proposed verification rule for every later pure-module task:** tests may import and call that module directly with explicit supplied mathematical inputs; they must also prove that CLI/API/dashboard/worker training, evaluation, resume, checkpoint, interpretation and relation operations still fail before file/artifact side effects. Do not turn a pure evaluator into a hidden dispatch fallback. Reconnection requires its own implementation scope, approval, review and no-side-effect/positive-path test matrix.

## 11. Source-linked decision register for author review

The decisions below are the remaining interface/engineering boundaries; they do not reopen F0–F12 mathematical choices. Recommendations are not approvals.

### D1 — Bind canonical class order and present fixed `DomainThing`

- **Status:** Existing engineering fact for dataset vocabulary; **Unresolved** future classifier-row/output binding and fixed-class result representation.
- **Exact boundary:** The dataset owns a stable ordered set of trainable IDs; F4.4 requires ordered `contains:[N,C_train]`; F4.3 requires fixed-positive `DomainThing`. Decide how ordered IDs bind to parameter rows/output metadata and whether the fixed predicate is a separate typed output or an outer presentation operation. Do not place fixed/unsupported classes among trainable columns or infer order from a batch.
- **Evidence:** [F4.1–F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54); [`PreparedDataset` and trainable-order projection](../../src/town_model/dataset/image_data.py#L22); [manifest vocabulary construction](../../src/town_model/dataset/ontology.py#L450); [order/exclusion tests](../../tests/test_phase3_dataset.py#L20).
- **Alternatives:** (a) classifier is class-count-only; an immutable dataset adapter binds canonical IDs and attaches them to results; (b) bound classifier stores the ordered IDs as output-axis metadata. For `DomainThing`, (i) a separate `[N]` fixed output, (ii) a separate typed fixed-class result, or (iii) outer adapter materialization.
- **Consequences:** Different mappings permute/rename output columns and can silently misalign supervision/checkpoints; folding fixed entries into `C_train` violates the trainable shape boundary and changes parameter meaning.
- **Recommended default:** dataset manifest remains vocabulary/order authority; classifier math stores only class count and class-parameter rows; a bound output adapter carries the exact ordered IDs; fixed `DomainThing` is a separate bool `[N]` equal true for valid points in `Δ`; unsupported classes have no output column. This is the narrowest F4.1/F4.4-consistent option, not approved.
- **Blocks:** flat classifier's dataset-facing typed result, image integration, and training/checkpoint class mapping. Does **not** block unlabeled pure box/town/affine mathematical evaluators with explicit dimensions/parameter matrices.

### D2 — Empty axes, validity, and domain-invalid point behavior

- **Status:** **Proposed** contract; **Unresolved** for `C_train=0`, out-of-domain classifier queries, and exact affine dtype. No active classifier settles these choices.
- **Exact boundary:** Permit or reject `N=0`; permit or reject `C_train=0`; define `d=0`; define finite/dtype/device/domain validation and whether an out-of-domain point is rejected by the adapter or returned as non-member by a pure set query.
- **Evidence:** [F4.1/F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54), [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33), [F7.3–F7.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L84); `ImageBatch` nonempty validation at [`contracts.py:41–46`](../../src/town_model/contracts.py#L41); empty-K tests at [`test_phase3_models.py:70–76`](../../tests/test_phase3_models.py#L70); reducer shape behavior at [`models/losses.py:27–88`](../../src/town_model/models/losses.py#L27).
- **Alternatives:** Reject all empty classifier batches/classes for simplicity; or permit mathematically shape-correct zero axes in pure evaluators and have a training adapter mark zero supervision undefined. For out-of-domain points, reject at adapter, or let a pure set evaluator return false outside `Δ` while keeping the adapter strict. For numeric inputs, rely on framework failures/casts, or validate exact shape, floating dtype/device, finite values and declared operating envelope before calculation.
- **Consequences:** Empty-axis policy affects empty-selection and unsupported-vocabulary behavior, but not the set mathematics. Permissive silent casts/broadcasts can hide axis/dimension errors. Training on no known labels must not become a fabricated zero objective under F8.4.
- **Recommended default:** `d≥1`; pure evaluators accept `N=0` and optionally `C_train=0` with correctly shaped outputs; no training objective is defined when no eligible labels exist; classifier adapter rejects out-of-domain/nonfinite/unsupported tensors; pure set membership may define outside-`Δ` as false only if explicitly specified. Keep `ImageBatch B=0` restriction independent. Geometry remains float32 per F7.3; propose float32 affine interface for initial parity, subject to author confirmation.
- **Blocks:** final flat classifier shape/input contract, especially empty and dataset-free adapters. Does **not** block fixed-corner geometry kernels whose inputs explicitly satisfy their own approved preconditions. Empty `ImageBatch` remains deferred.

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

1. **Pure physical-corner box Boolean evaluator.** Inputs are supplied finite physical corners/points and explicit `Δ`; no raw-to-physical transform, class IDs, dataset, image path, trainable parameterization, or optimizer. Verify exact closed-face membership, corner order/dimension/domain failures, fixed degenerate fixtures where approved, empty point axis, and class-free behavior. Scientific basis: F3.2/F3.4/F7.4; readiness says this is not blocked by P0-A/B/C.
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
| Typed output contract | Assert exact `[N,C_train]` shapes, bool membership, one family-specific numerical field, ordered-ID length/identity, separate fixed output; test zero axes per approved policy and reject ambiguous variants. |
| Class-axis binding | Compare output IDs to canonical manifest order; reject permutations/duplicates/mismatch; prove fixed/unsupported IDs have no trainable parameter columns; never inspect incidental batch labels to build the vocabulary. |
| Pure box evaluator/clearance | Hand-computable corners, faces, outside, degenerate fixed geometry if enabled, dimensions/domain, finite float32 operating envelope, empty `N`; demonstrate signed coordinate clearance is not Euclidean and is not membership. |
| House/town evaluator | Window interior vs boundary, frame boundary, overlapping-window exclusion, independent external windows, exact union, `W=0`, single-house equivalence, min/max clearance; no trainable-width inference from degenerate fixtures. |
| Affine evaluator | Exact weight/dot/bias values, threshold equality at zero, one-hot class-row mapping, `N=0`/`C=0` policy, invalid ranks/dtypes/nonfinite inputs; no softmax/probabilities. |
| Point adapter | ID/mask alignment, all and only valid occurrences, repeated occurrence identity preservation, final pizza mode selection, row-map inverse, no detach, zero-padded gradients, no labels/categories/confidence as point inputs. |
| Domain-specific image heads | Bounded path remains numerically/checkpoint compatible; unbounded path has no terminal `tanh`/clip/scale; both retain trainable gradients; checkpoint/run identity distinguishes activation/domain; matched initialization evidence reports point distributions/geometry starts. |
| Reducer/objective adapter | Per-label penalties originate in objective; unknown/fixed/unsupported exclusion; positive-only configured weights; F8 hierarchy and accumulation gradient parity; undefined tasks and counts; no new regularizer/objective. |
| Fail-closed boundary | Positive tests only call pure modules directly; existing CLI/API/dashboard/worker routes still raise `ClassifierBackendUnavailableError` before reads/writes, checkpoint loading, locks, caches or publication. |

## 14. Current unresolved author decisions and scope impact

Only three grouped author decisions plus the already-pending #3 confirmation are recommended to avoid repeated approval cycles:

1. **D1 — class binding/fixed output:** approve or amend the canonical trainable-ID-to-column mapping and a separate `DomainThing` representation. Blocks dataset-facing flat result, image integration, and training; not pure evaluators.
2. **D2 — cardinality/validation:** approve zero-class behavior, strict input validity, and adapter treatment of out-of-domain points. Blocks final flat classifier contract; not physical-corner evaluator slices with explicit preconditions. `N=0` is recommended for pure evaluation; empty `ImageBatch` remains separate.
3. **D3 — domain-specific image projection:** approve the narrow terminal-activation proposal or defer unbounded image integration. Blocks unbounded image adapter/training and any material point architecture work; not flat classifiers over supplied points or pure geometry evaluators.
4. **#3 — existing explicit author confirmation:** remains OPEN, not resolved here. Keep the source-approved F3 choices intact; do not treat this draft, agent recommendations, or silence as its resolution.

D4 is an adapter engineering decision that can be settled within a later approved integration task without reopening classifier mathematics. D5 is a later reducer-conformance task, not a blocker to pure classifiers. F11 audit A–J, final relation/statistical protocols, detector policy and comparison-quality training remain outside R2 and must not be smuggled into this contract.

## 15. Repository and review boundary

This document was prepared on branch `docs/r2-classifier-interface-contract` from fixed starting revision `7ebc2b62c175288b0673524fbfae9f9d6f15a4ff`. It is a new draft under `docs/reconstruction/`, not an edit to current-status, acceptance, A1/F11/F12 sources, readiness authority, or historical review evidence. Issue #14 is the execution journal and is natively linked as a child of #2. The draft must receive independent documentation review before the author makes final contract decisions. No classifier, geometry, loss, optimizer, route reconnection, data generation, training, or scientific evaluation was implemented or begun.
