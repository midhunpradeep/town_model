# Representing Ontologies Using Town Embeddings
## Integrated research and implementation specification — v1.1 (expanded traceable review draft)

**Date:** 28 October 2026  
**Decision coverage:** Q1–Q191, with a line-by-line decision ledger in Appendix A. Major revisions and supersessions are explicit; details not finally settled are flagged in Appendix C.  
**Audience:** Thesis author, supervising researcher, and an implementation agent (Luna).  
**Status:** Expanded review draft. Not all mathematical details are resolved; see Appendix C. **Not** authorization to implement without the author's explicit final approval.  
**Previous records:** `thesis_architecture_snapshot_v0_1.md`, `thesis_architecture_checkpoint_v0_2.md`. This document supersedes conflicts in those checkpoints; Appendix A tracks all 191 decisions, and Appendix C calls out remaining genuine under-specification.

> **Reading convention.** **MUST** denotes a contractual behavior or testable invariant. **DEFAULT** denotes a configurable, provisional starting point—not an empirically justified optimum. **DEFERRED** denotes an intentionally excluded first-version experiment. **IMPLEMENTATION FREEDOM** denotes a choice Luna may make without altering the science. Rationales are included to explain how each decision serves the thesis, not merely to demand particular code.

---

# 1. Thesis intent, research claims, and limits

## 1.1 Why build this system?

The thesis proposal, *Representing Ontologies using Town Embeddings* (19 June 2026), poses two research questions:

- **RQ1 — Predictive quality:** Do town embeddings cause a relevant loss of predictive quality relative to pure box embeddings? The working hypothesis is that towns preserve comparable multilabel prediction performance despite greater geometric complexity.
- **RQ2 — Ontology structure:** Can the greater expressivity of regions with holes improve the representation of ontology relations, particularly overlap, exclusion and disjointness, while preserving membership and subsumption?

These are **distinct empirical questions**. A geometric classifier can classify all observed points correctly yet fail to represent a global subclass relation `HamTopping ⊑ MeatTopping` as region containment. Conversely, a geometrically sensible representation may be harder to optimize for classification. The system must expose both predictive results and the actual learned regions so that the two claims can be assessed separately. The definitive relation-level evaluation protocol awaits the supervisor discussion, but data and geometric diagnostics necessary for it must be retained from the start.

The thesis proposal originally mentioned YOLO as the detector. **The subsequent architecture discussion explicitly replaced YOLO with a modular Faster R-CNN-style detector.** This is an intentional design amendment, not an omission. Cone geometry remains an optional future extension, outside the required implementation.

## 1.2 What is being learned?

Each *whole pizza image* and each *individual visible component occurrence* is mapped into a point in one learned latent domain. Ontology classes are represented by global class parameters: a neural linear decision head, a single axis-aligned box, or a union of perforated boxes (“town”). The model must learn from RGB images at inference; image-level and component-level ontology targets are constructed from the generator annotations during dataset preparation.

A shared class region must be evaluated against both pizza and component points. This creates a meaningful **common ontology space** rather than unrelated component and pizza classifiers. A global pizza point may be derived from the entire image, from a permutation-invariant combination of component points, or from a learned fusion of both. This allows a direct investigation of whether composition of observed components is useful for ontology prediction.

**Important limits on interpretation:**

- These are learnable **point/region classifiers** supervised by ontology-expanded instance labels; no ontology-axiom loss is imposed during initial training. Ontology relations may or may not emerge geometrically.
- Classes are input-independent, global parameters; images determine **points**, not per-image class boxes.
- The learned domain contains pizza/base/topping/flag entities represented in this dataset; it is **not** claimed to be the complete OWL domain or an implementation of full ALC semantics.
- Source pizza type, ingredient counts, country metadata, annotation labels and detector category predictions **must not become semantic input features** for the final ontology embedding. Their legitimate uses are supervision, quality checks, detector training, matching, and diagnostics.

## 1.3 Scientific comparison and confounds

The **canonical first comparison** consists of five configurations, with common data, split, CNN/embedding architecture, seed list and initialization policy:

| Family | Primary classification objective |
|---|---|
| Neural baseline | Multilabel BCE on shared linear class head |
| One box/class | Signed-margin BCE |
| One box/class | Signed-margin hinge distance |
| One town/class | Signed-margin BCE |
| One town/class | Signed-margin hinge distance |

**DEFAULT:** hybrid pizza representation, latent dimension `d=16`, class weighting `none`, common image resolution `224×224`, initial dataset `4,800` images. Select one town window mode explicitly for this five-way core and hold it fixed; then perform a **paired independent-versus-contained window comparison**. Both modes are mandatory to implement, but there is no requirement to run every possible Cartesian product of settings.

Available individual experimental factors: model family, allowed loss, global/compositional/hybrid mode, embedding dimension (2/8/16 initially), town window parameterization, number of houses/windows at run creation, class weighting, detector settings, and training phase. Each configuration is a separate run, not a separate copied codebase.

The baseline is **not capacity-matched** to a town with several houses/windows. It is a controlled *architectural* baseline with a shared point encoder and a simple head, not proof that all methods have equal parameter count or expressiveness. Report parameter counts and detection quality; a stronger MLP classification baseline is deferred.

## 1.4 Decision rationale at a glance

This table is a navigation aid for readers who want to understand the design's reasoning before examining the implementation contracts. The alternatives are not prohibited in principle; they were deliberately excluded from the *initial controlled study*.

| Agreed choice | Reason in terms of thesis intent | Tradeoff acknowledged | Decision lineage |
|---|---|---|---|
| One latent space and one class head for pizza and component points | Makes geometric class membership meaningful for both parts and wholes, and permits tests of shared ontology structure | May create optimization tension between very different instance types | Q1, Q9–Q14, Q94–Q95, Q101, Q150 |
| CNN/FPN with class-aware Faster R-CNN rather than YOLO | Provides region proposals **and shared ROI features** for transparent component-level embedding and staged training | Greater integration complexity and substantial detection cost | Q11, Q15–Q16, Q28, Q35–Q39, Q121–Q130 |
| Three composition modes and permutation-invariant DeepSets | Separates global visual recognition from aggregating individually grounded components; sums preserve multiplicity | A basic sum may struggle with scale and complex interactions | Q9, Q18–Q19, Q30, Q47–Q48, Q97–Q98 |
| Bounded points and ordered bounded geometry | Makes universal-domain/complement interpretation and plotting finite and testable | `tanh`/sigmoid saturation and limited expressive range | Q27, Q40, Q49–Q53, Q96, Q153, Q173 |
| Town with houses minus windows | Tests whether holes and unions represent negative/exclusion structure better than a single convex box | More parameters, sparse branch gradients, possible inactive windows | Q21, Q41, Q52–Q57, Q61, Q131–Q135 |
| Signed BCE versus signed hinge on the **same** margin | Isolates the learning penalty from the geometry and provides gradients for important misclassification cases | Neither is identical to the literature's loss; hinge gradients stop after margin | Q61–Q62, Q74–Q78, Q103, Q113–Q115 |
| Additional literature-inspired reference loss modes | Allows comparison with the motivations and limitations of existing work | Requires carefully documented adaptations and source-conformance checks | Q103, Q113–Q115, Q158, Q174 |
| Dataset layer owns ontology expansion and target masks | Keeps experimental models generic, eliminates label logic duplication and protects from leakage | Requires richer schema and strong validation before training | Revised Q43/Q72, Q67–Q73, Q85–Q88, Q161–Q165 |
| Three training phases; GT ROIs then predicted ROIs | Separates representation learnability under perfect localization from end-to-end robustness under detection errors | Phase transition introduces distribution shift; NMS/box selection is nondifferentiable | Q4, Q16, Q32, Q42, Q81–Q82, Q143–Q149 |
| Exact membership for predictions, differentiable scores for training | Preserves the intended set-theoretic meaning even at window boundaries | Continuous surrogates cannot encode every boundary convention | Q56, Q61, Q105, Q151–Q152 |
| No initial ontology-axiom loss, but relation diagnostics required | Makes emergent structure an empirical result instead of forcing it by construction | Correct prediction alone may not recover subclass/disjointness relations | Q5, Q7, Q26, Q184–Q185 |
| Fixed dataset, paired initializations, repeated seeds, held-out test | Makes observed differences more attributable to representation and loss | Does not equalize head capacity or eliminate all stochasticity | Q34, Q53, Q87, Q100, Q112, Q142, Q168, Q186, Q189 |

---

# 2. System organization and dependency boundaries

## 2.1 One self-contained research repository

The integrated project must have independently testable, composable responsibilities, conceptually:

```text
research-repository/
  dataset/        # generator adaptation, ontology semantics, targets, manifests, audit
  detection/      # Faster R-CNN integration, FPN, ROI features and coordinates
  embeddings/     # point projections, DeepSets, fusion
  geometry/       # boxes, houses, towns, membership, margins, diagnostics
  models/         # baseline / box / town assembly and common output contract
  losses/         # unreduced objectives, masking/weighting and aggregation
  training/       # phases, optimizer, accumulation, checkpoint/resume
  evaluation/     # detection, pizza/component metrics, relation diagnostics
  experiments/    # config schema, provenance, run store, comparison utilities
  dashboard/      # local UI, launcher, experiment and geometry explorers
  tests/          # unit, numerical, schema, integration and smoke tests
```

The directory names are suggestions, **not required public APIs**. Prefer modules that own one responsibility and depend on narrow typed contracts. Avoid circular imports and “god” classes coordinating all behavior. Shared mathematical primitives belong in `geometry`, not duplicated in model variants; dataset reasoning belongs in `dataset`, not the PyTorch loader. The dashboard should call the same validated configuration and run APIs as the CLI.

## 2.2 Reuse without creating a second source of truth

Copy/adapt as much tested, model-independent code from the existing `fourth_attempt/` preparation project as practical. The **new repository becomes the maintained implementation**; it does not import `fourth_attempt/` at runtime and does **not modify that original project**. Do not edit the sibling `baseline_model` project or the pinned upstream Pizzaïolo generator. Reuse tests where valid, update tests where the accepted ontology-label semantics differ.

**Rationale:** the generator and label-expansion work already exist, but separate data/model repositories would add version synchronization and repeated schema logic. A dedicated model-independent package within one repository gives independence of *responsibility* without operational fragmentation.

## 2.3 Configuration authority

Every module defines typed, validated parameters; a single composed **resolved experiment schema** owns the authoritative configuration. The same schema must power CLI overrides and editable dashboard drafts. Reject invalid combinations (e.g., reference town distance on a pure box, window settings on neural baseline, an unsupported class vocabulary). At run launch, resolve defaults and persist an immutable configuration, class-manifest hash and data-version ID. Changing settings creates a new run, never retroactively edits an executed run.

All **relevant hyperparameters MUST be visible and configurable in the dashboard**: architecture, detector/anchors/filtering, latent dimension, house/window counts and modes, initialization distribution, loss mode/temperature/hinge margin, class-weight formula and cap, coefficients, optimizer/groups, schedule, batch/accumulation, clipping, precision, checkpoints, snapshot intervals and visualization settings. Actual hyperparameter search/auto-tuning is deferred.

---

# 3. Dataset preparation: complete, independent semantic contract

## 3.1 Source materials and generated artifacts

Use the pinned Pizzaïolo source revision `SysReIC/pizzaiolo@6c0bcb221a957c184012beaf103bebcde4b1f793` (see References). The generator produces `224×224` synthetic RGB pizza images, original metadata, bounding-box JSON (multiple instances per component class), contours and a segmentation representation. Its recipe catalogue consists of 15 named types and one random/unspecified type.

**DEFAULT:** require explicit generation count, initially `4,800` (=300 per generator type), type-balanced before splitting; split `60%/20%/20%` by generator type, seed `42`. Because splitting rounds per type, record actual counts rather than assuming exact global ratios. **No prepared dataset yet exists in `fourth_attempt/`.** A separate historical 6,000-image dataset audited elsewhere must not be confused with results from the new target dataset.

Dataset generation is an **explicit operation**, never a side effect of starting training. Produced dataset versions are immutable and reusable across configurations. Save generator revision, preparation-code revision, ontology snapshot and SHA-256, generation config, split manifests, schema version and validation report.

## 3.2 Ontology classes versus individuals

The uploaded `pizzaiolo.xml` declares 66 named OWL classes and 5 country individuals: `America`, `England`, `France`, `Germany`, `Italy`. The audited source provides **32 image-level class targets**. Component labels originally provide 13 distinct ancestor/category targets but omit the concrete type for 16 topping classes and 2 base classes. **Correct this during integration**: e.g., a `HamTopping` instance must be positive for `HamTopping`, `MeatTopping`, `PizzaTopping`, `Food`, `DomainThing`. For `JalapenoPepperTopping`, preserve `PepperTopping`, `VegetableTopping`, `VegetarianTopping`, `SpicyTopping`, along with itself and `PizzaTopping`, `Food`, `DomainThing`. For `DeepPanBase`, include `DeepPanBase`, `PizzaBase`, `Food`, `DomainThing`.

`Country` is an OWL **class**; `America` and `Italy` are OWL **individuals**. Do not convert countries into new fake ontology class regions. Country flag component embeddings receive class membership `Country|DomainThing`, whereas their visual identities are separately detector-supervised categories. The pinned generator actually renders four flags: America, England, France, Italy. **Germany is not a supported rendered category** despite existing as an ontology individual. Do not modify ontology semantics to invent `RealGermanPizza` or an unrendered Germany flag.

Expected union: `32 image classes ∪ (13 component classes + 18 concrete component classes)` minus shared `DomainThing` and `Food` = **61 distinct supervised OWL classes**. `DomainThing` will be fixed, leaving an anticipated **60 trainable classes**; this count must be verified at manifest generation, not hard-coded. Five value/spiciness classes (`ValuePartition`, `Spiciness`, `Hot`, `Medium`, `Mild`) lack direct instance-point supervision here. Preserve the **complete ontology inventory** for later diagnostics; do not train unsupported heads on fabricated negatives.

## 3.3 Label meaning: complete synthetic rows versus incomplete observations

The dataset must provide **positive, known-negative, and unknown** states with a machine-readable mask. A suitable normalized representation has `labels` and `known_mask` over the canonical class vocabulary, or explicit positive/unknown sets that compile to those vectors. A missing positive class is *not universally equivalent to negative* unless completeness is established.

For **validated rows from this pinned synthetic generator**, its topping list/counts and base identify deliberately complete ingredient facts, and a blank country means **no origin flag was selected or rendered**; origin-dependent negative labels are consequently justified under this **dataset's closed-world contract**. This is not a claim about open-world OWL inference. If imported, corrupt or incomplete rows contain unknown metadata, mark appropriate targets unknown when supported or reject the row; do not silently turn uncertainty into a negative. Full three-valued evaluation of every possible external OWL fragment is *not* required.

Preserve the audited `infer_pizzaiolo_label` semantics, including all 32 intended image classes and overlap among recipes/derived labels. It already covers `GenerousPizza`, `ProteinPizza` and `VerySpicyPizza`. Do not rewrite it merely because an excerpt omitted rules. Derive applicable class ancestry, equivalences relevant to component labels, and instance-level positive closure in the dataset module. Classes outside a given instance's scope may be **known negative** when the complete entity type and annotations justify that conclusion (e.g., a ham component is not a pizza). The manifest declares which targets are meaningful, fixed, trainable or unsupported; an unsupported class is not a negative target.

## 3.4 Mandatory integrity gate

Before any real training, validate and produce a machine-readable report and dashboard-readable summary:

1. Ontology and vocabulary hashes; every output class is declared, with class/individual distinction and complete expected closure; no `HotTopping` helper accidentally promoted to an OWL class.
2. Topping counts are nonnegative integers; `hasTopping` equals the set of concrete types with positive counts; valid required base; image-level inferred labels respect tested recipe predicates, expected roots, named-pizza implication, vegetarian/nonvegetarian exclusivity, origin exclusivity and other supported invariants.
3. Country field and rendered flag annotations agree, including **no-origin ↔ no flag**; the four renderable identities are recognized; Germany rendering is not treated as supported; mismatches fail with sample IDs and file paths.
4. Boxes, labels, contours/segmentation references, image files, component occurrence IDs and detector categories are valid and mutually consistent. **Do not demand each component's segmentation mask to exist as a separate file:** upstream uses per-element masks internally to construct segmentation.
5. Image IDs and component IDs are stable and deterministic; duplicate same-type toppings remain distinct. Train/validation/test reference sets are disjoint, collectively cover the dataset and preserve generator-type stratification. No silent alternate splitting.
6. For each relevant class and each split, publish counts of known positives, known negatives and unknowns, separately for pizza-level and component-level supervision; report classes with no positive support.
7. An exploratory run may warn about absent/rare positives. A **comparison-quality** run must refuse to start if any *required trainable class* lacks known-positive training supervision under the intended training protocol. Classes with absent validation/test positives are reported and handled by defined evaluation rules, rather than secretly excluded from the manifest.

The initial dataset is assumed to provide exhaustive annotations for supported *visible* components; validate that contract for end-to-end component evaluation. Missing annotations must not be interpreted as an ordinary negative without supporting completeness evidence.

## 3.5 Export and loader interface

Use documented, framework-neutral artifacts: images (PNG or original equivalent), stable IDs, original-image pixel `xyxy` component boxes (or rigorously documented convertible representation), per-component detector category, per-instance ontology positives/unknowns or vectors/masks, pizza-level targets/masks, split files, ontology/class/category manifests, and provenance. Keep raw generator metadata in dataset audit files but **never** expose it as predictive input.

A single generic model loader yields:

```text
Sample:
  sample_id: stable str
  image: RGB image, original W×H
  pizza_targets: [C_train] {0,1}
  pizza_known_mask: [C_train] {0,1}
  components: list of {
    component_id, bbox_xyxy_original: [4], detector_category_id,
    ontology_targets: [C_train], ontology_known_mask: [C_train]
  }
```

A collator returns variable-length detector target lists where Torchvision expects them and padded/valid-masked component arrays where embedding/composition code benefits (`[B,K,...]`, `[B,K]`). The model loader converts **schema to tensors**, not raw generator facts to ontology reasoning. Data manifests, not model code, establish class ordering. Checkpoints refuse incompatible vocabulary/schema hashes unless an explicitly implemented migration is invoked.

---

# 4. Model: detection, point projections and composition

## 4.1 Detector and shared visual features

**MUST:** two-stage Faster R-CNN-style **class-aware** detector built from maintained Torchvision components, with MobileNetV3-Large + FPN as the default shared backbone. Initialize **ImageNet backbone weights only**, not a COCO-specific classification head; initialize new heads for **22 visual foreground categories + background**, derived from the generated detector-category manifest (16 toppings, 2 bases, 4 rendered flags). Pin Torch/Python/Torchvision compatibility and verify actual FPN maps are 256-channel where assumed.

Input native size defaults to `224×224`. Choose and log RPN anchors/aspect ratios using **actual training-split box statistics**; supply a validated small-object fallback if data has not been generated. Train later **three backbone stages** by default during Phases 1 and 3, mapping “stage” to explicit version-pinned MobileNet blocks; configurable full unfreezing. Higher resolution, ResNet or pretrained COCO detector comparisons are deferred.

Torchvision normally exposes training losses and inference detections through different modes. Implement a **version-tested integration adapter** that obtains transformed feature maps, detector losses and final selected predictions in one logical forward, **without recomputing the expensive CNN backbone**. Reuse RPN/ROI building blocks; do not copy or rewrite the detection algorithm merely to gain access to outputs. Centralize RGB scaling, ImageNet normalization, resizing/padding and coordinate conversion, so data is not normalized or resized twice.

**DEFAULT detector filtering**: confidence `0.25`, ordinary class-aware NMS IoU `0.5`, maximum `100` selected detections/image, optional conservative cross-category deduplication at IoU `0.9`. Every threshold is configurable. Such cross-category suppression may hide genuinely overlapping components; record its effects and allow disabling. Detector class labels/confidences are for training, selection and diagnostics **only**. Neither the numeric label nor the class-score vector may be appended to the ontology point projection or given to DeepSets. This preserves a visual-learning rather than label-leakage experiment.

## 4.2 Features and coordinates

**Component feature extraction**: ROIAlign with output size `7×7` on the shared 256-channel FPN feature maps; spatial mean gives 256 visual numbers. Append `(cx/W, cy/H, w/W, h/H)` computed from the **original-image-coordinate** predicted/GT box with actual original image dimensions, producing 260 numbers. Do not concatenate detector category/confidence.

**Global pizza feature extraction**: use a full-image ROI mapped correctly into processed-image feature coordinates, yielding `256×7×7` spatial features. The global branch must work even when the detector produces zero component detections. This preserves actual image-layout information before projection.

Original boxes, internal transformed/padded boxes, ROIAlign coordinates and exported predictions must have central, tested conversion functions. A box projected to another resolution must preserve intended region. Returned predicted boxes and diagnostic overlays use original-image coordinates; ROIAlign consumes the transformed coordinates expected by the pinned implementation. No hard-coded “divide by 224” where image dimensions can vary.

## 4.3 Point heads

All point outputs lie in a bounded latent domain `Δ = [-1,1]^d` via `tanh` at appropriate **point-producing** heads. **DEFAULT `d=16`**, with 2 and 8 supported and arbitrary validated dimensions possible. Separate projection networks for component and global branches, sharing the CNN features and output vocabulary/geometry:

```text
Component: 260 → Linear(256) → LayerNorm + ReLU
               → Linear(128) + ReLU + Dropout(0.1)
               → Linear(d) → tanh → component point [d]

Global: 256×7×7 → 1×1 Conv(32 channels) → flatten(1568)
               → Linear(256) + activation → Linear(d)
               → tanh → global pizza point [d]
```

Non-specified minor activation/normalization placements must be recorded as implementation decisions; they must not silently change across model families. **Match initial weights** of architecturally identical CNN/projection/composition modules across compared runs and train them independently afterward; geometry/head parameters are separately initialized. This isolates meaningful architectural differences without pretending the independently trained latent spaces retain common coordinates.

## 4.4 Pizza point construction

Three selectable, equally supported modes:

1. **`global`**: pizza point is the global image projection. Component classification remains an active auxiliary task, so this is *global pizza composition with shared-backbone component supervision*, not training with images alone.
2. **`compositional`**: every selected component occurrence produces one point; apply DeepSets. A shared `φ: d→128` two-layer MLP transforms each point, **sum** transformed vectors over valid components, then `ρ: 128→d` two-layer MLP maps to the composed pizza point (with bounded final output). Multiplicity and positional information are retained through separate occurrences and each point's visual+position features. Aggregation is permutation-invariant. No attention in first version.
3. **`hybrid`**: concatenate global and composed pizza points (`2d`) and apply a learned `2d→128→d` fusion MLP with final bounded output. Apply pizza-level classification loss **only to the final fused pizza point**, not the independent global/composed intermediates; component-level labels still supervise individual components. Expose all intermediates for diagnostics.

For zero detected components, the DeepSets sum is the zero vector and `ρ(0)` yields a valid composed embedding. Record an empty-component diagnostic flag; do not crash, fabricate a topping or introduce a special pseudo-detection. The **same exact component point** is both classified against ontology classes and passed to DeepSets—no independent “composition-only” component encoder. Component points receive classification gradients and, in compositional/hybrid mode, pizza-level gradients through aggregation.

## 4.5 Shared classification interface

Each family produces concept predictions for any flat collection of instance points `[N,d]` over the same manifest:

- **Neural baseline:** one `Linear(d,C_train)` multilabel logit head and standard BCEWithLogits; no category-specific pizza/component head.
- **Box:** one trainable bounded axis-aligned box per trainable ontology class.
- **Town:** one trainable fixed-capacity set of houses and windows per trainable class.

`DomainThing` is a **fixed positive output for all observed entity points** in every family and is omitted from the trainable class parameters, optimization losses and headline predictive metrics. Its universal interpretation is relative to the **restricted learned dataset entity domain**, not all OWL individuals. `owl:Thing` and `owl:Nothing` remain logical metadata, not ordinary learned outputs.

---

# 5. Geometry: exact meaning and trainable representations

## 5.1 Set semantics

An axis-aligned closed body box in `d` dimensions is

`B(l,r) = {p ∈ Δ : ∀i, l_i ≤ p_i ≤ r_i}`.

A house removes the union of **closed** window boxes from its closed body:

`H = B \ (⋃_{w=1..W} W_w)`.

A town is a union of houses:

`T = ⋃_{h=1..H} H_h`.

A point on a **body boundary is included** unless a window removes it. A point on a **window boundary is excluded** from that house. Another house can reinclude a point excluded by the first. Independent windows outside their body have only the effective subtraction `B∩W`. True membership must be computed as Boolean set operations, not inferred from an approximate margin at zero.

**DEFAULT town capacity:** `H=2` body boxes per class and `W=1` window per body. Fixed equal capacity per class within a run; configurable at creation. No dynamic house/window creation, deletion, masks or exact-inactivity gates. With strictly positive contained windows, exact empty holes cannot be learned; acknowledge and diagnose this representational limitation.

**Two required parameterizations:**

- `independent`: each window constrained to Δ but free to extend partly or wholly outside its parent body.
- `contained`: each window strictly contained in its own parent body, with finite-precision-safe width/clearance constraints.

These modes implement **identical set subtraction and loss interfaces**; only the mapping from trainable parameters to physical window corners differs. A paired comparison MUST start with the same **actual** physical house and window coordinates (within numerical precision), not just the same RNG seed, and can diverge afterward.

## 5.2 Numerical parameterization

The original proposed ordered-corner transform was `l = -1+2 sigmoid(a)`, `r = l+(1-l) sigmoid(b)` per coordinate. Q153 and Q173 strengthen this: **MUST** enforce a configurable positive minimum axis width, DEFAULT `ε=10^-5` domain units, in actual float32 and under extreme raw-parameter values. Contained windows require parent-body widths sufficient for a positive window plus **strict positive clearance on both sides**; a rule such as `body_width ≥ 3ε`, `window_width ≥ ε`, `left/right_clearance ≥ ε` is an acceptable strategy. The parameterization must produce valid physical geometry *by construction*, not rely exclusively on clipping coordinates after an optimizer step. Use differentiable/stable transforms where feasible; saturation/precision behavior must be documented and tested. Do not assume an arbitrarily extreme sigmoid input remains mathematically strict after rounding.

All geometry transformation, membership, distance and loss calculations use float32 even if optional AMP is enabled for the CNN. Small widths, near-boundary cases, overlapping bodies, nested windows and tied min/max branches need numerical tests. Do not silently auto-reset inactive geometry.

## 5.3 Signed margins versus exact membership

For a body box:

` s_B(p) = min_i { p_i - l_i, r_i - p_i } `.

For a house with windows:

` s_H(p) = min(s_B(p), -s_W1(p), ..., -s_WW(p)) `.

For a town:

` s_T(p) = max_h s_Hh(p) `.

These are **signed coordinate-clearance surrogates**, not the Euclidean distance to an arbitrary region. They are positive for interior membership and negative for nonmembership away from boundaries. At a boundary, `s=0` cannot distinguish an included body boundary from an excluded window boundary: the hard `contains` operation is authoritative for prediction; signed margins are used for optimization and soft-score diagnostics. Do **not** add undocumented epsilon offsets to conceal this distinction.

Exact `min` and `max` are the first-version aggregation methods. They permit gradients almost everywhere but concentrate them on limiting branches, possibly leaving houses or windows inactive. A smooth log-sum-exp alternative is deferred; log gradient/activity statistics instead of silently changing the mathematics.

## 5.4 Geometry APIs and inspectability

The geometry core receives **flat** point batches `[N,d]` and returns at least:

```text
contains: [N,C_train] bool       # authoritative hard set membership
signed_margin: [N,C_train] f32  # differentiable training surrogate
optional: body/window membership, margins/distances, winners,
          limiting faces, physical corners, activity/degeneracy statistics
```

A surrounding adapter flattens `[B,K,d]` components, honors validity masks, evaluates geometry, then restores output structure. Pure boxes and towns use the same public interface and shared underlying box primitives, but pure boxes do not allocate dummy town windows. Structured diagnostics must remain **optional** to avoid materializing large `[N,C,H,W,d]` arrays during every normal step. Support configurable chunking by points or classes with equivalence and gradient tests.

**Initialization:** label-agnostic, seeded, dimension-aware broad regions with controlled variation. Begin body boxes sufficiently large to interact with points in high-dimensional space; distinct houses start at small seed-controlled offsets rather than identical positions. Windows should initially cause **nontrivial exclusions**, using dimension-aware fractions of their bodies, rather than becoming virtually zero-volume at `d=16`. Window modes use matched initial physical shapes. Store initial centers, per-axis widths, domain volume fractions, effective overlaps, occupancy and seeds. Do not prefit class regions around known positive example embeddings in the primary comparison (data-driven initialization is deferred). This ensures geometry learning, rather than supervised placement during initialization, explains the result.

---

# 6. Loss functions: four distinct geometric objectives

## 6.1 Why distinguish them?

The thesis originally discussed weighted BCE for box embeddings and distance-based losses for towns. The interview developed two **new controlled objectives** from one signed-margin representation, then deliberately reinstated the **reference-inspired loss implementations** so comparisons can be traced to the literature. They are not four names for the same penalty. Record formula, origin, deviations and observed gradient behavior in the run manifest and report.

Define target `y∈{0,1}` and sign `t=2y−1`.

### A. `signed_bce` — box **and** town (primary)

`L = softplus(−t * s / τ)`  (BCEWithLogits with logit `s/τ`).  
**DEFAULT** temperature `τ=0.1`, positive, fixed within a run, configurable (not learned initially).

A continuous penalty rewards both correctly classified positive points well inside regions and negative points well outside. It approaches zero smoothly without becoming exactly zero for finite margins. Unlike a pure outside-distance score, a negative point **inside** a box can receive a useful gradient toward a boundary. The temperature changes score sharpness; tuning is deferred.

### B. `signed_distance` — box **and** town (primary)

`L = max(0, m − t * s)`  
**DEFAULT** margin `m=0.1`, nonnegative, configurable, not learned initially.

A hinge objective demands a clearance of at least `m` on the correct side and then stops penalizing that point. This differs from BCE **only in penalty behavior** over the same signed margin, permitting an interpretable controlled comparison. Its “distance” is the coordinate-clearance surrogate, not an exact L2 set distance.

### C. `reference_box_bce` — pure box **only** (additional)

Adapt the source `ChebiBox` construction:

1. Contract each class box inward by 20% of each axis width to `[l',r']`.
2. `v=Σ_i max(0,l'_i−p_i,p_i−r'_i)`.
3. `q=2−2 sigmoid(v)`.
4. Rescale/clamp the membership probability as in the documented reference implementation, transform into logits and apply BCE with the common target reducer.

When a negative point lies fully inside the contracted box, `v` is locally zero and its inside-point geometric gradient can vanish despite nonzero loss. This is a **characteristic of the reference construction**, not a bug to silently repair. The source used a different image/text encoder and parameterization; this is an **adapted loss reproduction**, not a full model replication. Preserve the relevant score construction and document required numeric adjustments.

### D. `reference_town_distance` — town **only** (additional)

Adapt the intended distance objective in `MGlauer/town-embeddings` (`convex_cone_geometry`): compute crisp frame/window/house membership plus separate inner and outer L2-style distances for body and windows, with the source's 10%-of-box-width margin construction; route false-negative/false-positive penalties according to membership, then apply the intended combination across windows/houses.

**Source-conformance gate (mandatory before implementation is accepted):** the prototype code includes a problematic `~belongs` applied to labels and ambiguous/fragile aggregation behavior. The implementing agent must first write a short **mathematical reference-port note** tracing each branch and reduction to the inspected source, state exactly how boolean inversion, target dtype, dimensions and aggregation are corrected, and provide small hand-computable cases. Do not invent a new closest-Euclidean-boundary optimizer while naming it “reference.” Corrections of executable defects are allowed and documented; any material change to the intended objective needs a separately named mode and author review. Preserve accepted **reference** behavior even where it has weaker gradients than the signed modes. Because the precise correction was not mathematically settled in the interview, it must **not be presented as already verified**.

### Compatibility

| Classifier | Standard BCE | signed BCE | signed distance | reference box BCE | reference town distance |
|---|---:|---:|---:|---:|---:|
| Neural baseline | **Yes** | — | — | — | — |
| One box/class | — | **Yes** | **Yes** | **Yes** | — |
| One town/class | — | **Yes** | **Yes** | — | **Yes** |

All geometric inference uses **exact `contains`** regardless of training loss; the shared `signed_margin` provides comparable soft indicators, and source-specific probabilities/distances are additional diagnostics. Different reference loss scales and gradient pathways are not directly comparable as scalar loss values.

## 6.2 Canonical class weighting and masked reduction

Dataset targets have `y∈{0,1}^C`, `known_mask∈{0,1}^C`. Unsupported/unseen classes are not made universally false; fixed `DomainThing` is excluded from the trainable objective. Every loss returns unreduced instance×class penalties. One common reducer owns masks, weights, and the agreed aggregation semantics—never implement separate averaging in each head.

**Weighting modes** (DEFAULT `none`):

- `none`: weight 1 for positive and negative examples.
- `inverse_sqrt`: class-positive weights proportional to `1/√(n_c^+)`.
- `effective_number`: weights proportional to `(1−β)/(1−β^(n_c^+))`, with configurable `0≤β<1`.

Here `n_c^+` counts **known-positive training observations**, separately for pizza and individual-component levels. Normalize resulting positive weights to a documented common scale and cap extremes (both parameters configurable); store the actual vector. The weight of a **known negative** remains exactly 1; an unknown contributes zero. If a required trainable class has zero known-positive training observations in a comparison run, fail the data-coverage gate rather than assign it infinite weight.

For an instance with active class set, define:

`L_instance = [Σ_c known_c * w(y_c,c) * L_c] / [Σ_c known_c * w(y_c,c)]`.

If its denominator is zero, it is **absent from the reduction**, not an observed zero-loss sample. Within one image, average losses of its supervised component instances; then average over **images with one or more supervised components**. Pizza loss averages valid pizza instance losses. Keep the detector's Torchvision loss terms individually logged and combined explicitly. The same reducer handles every model and geometric loss, with a zero-supervision case tested.

This per-instance normalization is a deliberate compromise: it avoids letting frequently topped pizzas dominate, but positive weight influence may be diluted by many negatives. Log class weights, active counts and gradient scales; do not change reducers mid-comparison.

## 6.3 Joint objective

For Phase 3:

`L_total = λ_D * L_detector + λ_C * L_component + λ_P * L_pizza`.

**DEFAULT:** all applicable coefficients `1.0`, configurable by phase and run. Phase 1 uses detector terms only; Phase 2 component/pizza only; Phase 3 all three. Equal numerical coefficients **do not mean equal gradient influence**; log per-term losses and relevant gradient norms. Geometry-specific volume/overlap penalties are initially off; optimizer weight decay is independent and configurable.

---

# 7. Three-phase training and exact gradient contract

## 7.1 Phase 1 — detect components

Train the Faster R-CNN backbone/FPN/RPN/ROI heads against **ground-truth boxes and 22 foreground detector categories**. Selectively unfreeze backbone stages according to config. Use the common original-pixel/processed-pixel coordinate contract. **Best checkpoint criterion:** validation COCO-style `mAP@0.50:0.95`; also record `AP@0.50`, category metrics, recall, small-object analyses and detector losses. Validation classes without instances are handled explicitly, never assigned arbitrary performance.

## 7.2 Phase 2 — learn ontology points and class representations with perfect ROIs

Start from the selected Phase-1 detector checkpoint. Put **all detector modules in eval mode**, freeze their parameters **and mutable state** (e.g., BatchNorm), and compute FPN/ROI features without a detector gradient graph. Use **ground-truth component boxes** for component ROIAlign, and the full-image ROI for global features. Train point heads, DeepSets/fusion as applicable, and the shared neural/box/town classifier using both component and pizza ontology targets. Even in global-only pizza mode, preserve component auxiliary supervision. Train a separate independently parameterized Phase-2 model per selected downstream configuration; start comparable modules from matched neural weights.

**Best checkpoint criterion (DEFAULT):** `0.5 * pizza_macro_F1 + 0.5 * component_macro_F1_on_GT_ROIs`. The latter describes classification with perfect GT crops, not end-to-end detection.

## 7.3 Phase 3 — jointly fine-tune on genuine predicted ROIs

Start from that run's Phase-2 checkpoint. Unfreeze allowed CNN/detector parameters. For each batch, do a **single shared backbone feature computation** and produce (a) detector losses using all GT annotations, (b) final selected predicted detections, (c) ROI embeddings from these **predicted** boxes, (d) pizza compositions over *all selected predicted component points*, and (e) component ontology loss on matched predictions only.

**Matching:** deterministic maximum-total-IoU one-to-one bipartite/Hungarian assignment between selected predictions and GT boxes, **regardless of detector predicted category**, with DEFAULT IoU acceptance threshold `0.5`. Handle ties deterministically, record matched/unmatched/missed counts and identities. The detector's category error is penalized by detector classification loss, while its correctly localized crop may still receive the correct ontology target. A missed GT region must **not** be inserted as a pseudo-predicted component for Phase-3 ontology training. Unmatched predicted boxes may influence pizza composition but receive **no fabricated component ontology label**.

**Gradient paths:** one combined backward pass per effective batch and one optimizer step (with accumulation as configured). Ontology losses backpropagate through ROI-sampled **FPN/CNN visual features** and all appropriate point/composition/class parameters. Predicted box coordinates are **detached** from ontology gradients; discrete selection, matching and NMS are not differentiated. Detector localization and class heads learn through their **own detector losses**. Phase 2 and 3 use exactly the same ROI→point architecture, merely changing box source and freeze state. Log/verify expected nonzero and deliberately zero gradients with an integration test; do not claim fully differentiable localization from ontology classification.

**Best checkpoint criterion (DEFAULT):** `0.5 * pizza_macro_F1 + 0.5 * component_E2E_macro_F1`. Matched-only component F1 is a diagnostic, not the checkpoint criterion, because a detector that finds very few easy components could otherwise be unfairly rewarded.

## 7.4 Optimization and stability

**DEFAULTS (all configurable and phase-specific):**

| Setting | Starting value or behavior |
|---|---|
| Optimizer | AdamW; interchangeable via validated config |
| Pretrained CNN learning rate | `1e−5` |
| Detection heads learning rate | `1e−4` |
| Point/composition MLPs learning rate | `1e−3` |
| Neural or geometric class head learning rate | `1e−3` |
| Scheduler | Short warmup followed by cosine decay; constant LR selectable |
| Physical batch | `4` images |
| Accumulation | `4` microbatches (target effective batch of 16 images) |
| Maximum epochs | `50` per phase (initial ceiling, not assumed required) |
| Early-stopping patience | `10` validation checks |
| Gradient clipping | Global norm `1.0` |
| Precision | Full float32 first; AMP opt-in only, geometry always float32 |
| Device | Single CUDA GPU primarily; CPU-supported tests/debug |
| Initialization | ImageNet-only backbone; matched common neural initial weights; broad label-agnostic box/town geometry |

Specify independently configurable weight decay for neural and geometric parameter groups, and record optimizer/scheduler steps. The **same phase-specific validation monitor** drives best-checkpoint selection **and early stopping**; training loss does not secretly provide a second stopping rule. Keep both latest resumable and best-validation checkpoints.

**Accumulation correctness:** do not simply average microbatch means if this changes the intended effective-batch reduction. Components are averaged within images and over images **with supervised matches**, while pizza targets are averaged over valid images. The accumulation mechanism must preserve these denominators across the full accumulation window. Tests should compare gradients from accumulated microbatches and a logically equivalent unaccumulated batch, allowing documented floating-point tolerance. Detection objective normalization requires separate consideration of Torchvision's own sampling semantics. Do not silently alter physical batch after out-of-memory failure.

Numerical failures (`NaN`/`Inf` in losses, gradients or physical geometric parameters) **stop the run** with an actionable log naming step, objective/parameter group and an attempt to preserve the latest valid checkpoint. No undocumented automatic optimizer resets, parameter clipping, skipped bad steps, or learning-rate adjustment.

## 7.5 Run identity, comparison quality and checkpoint lineage

Share a **single selected Phase-1 detector checkpoint per dataset/seed** across comparable downstream Phase-2 runs. Then independently copy/fine-tune for Phase 3. Match initial common neural weights across baseline/box/town, and pair actual initial independent/contained town geometry. Do not share already-trained downstream point weights unless an experiment explicitly says so. Common detector seed does not force all downstream run trajectories to match.

Support **exploratory** runs (one seed, warnings acceptable) and **comparison-quality** runs (at least three independent seeds per selected configuration, common seed list, identical dataset/split and controlled defaults). Report mean/variation as well as each run, rather than claiming one seed establishes a difference. Repeated seeds are not automated hyperparameter tuning.

A resumable checkpoint must include model parameters, optimizer and scheduler state, resolved config and hashes, RNG states, epoch/step, parent checkpoint IDs, and relevant metric/early-stopping state. Comparison-quality runs also record git commit and dirty status, dependency versions/lockfile, Python/Torch/Torchvision/CUDA, GPU type and determinism flags. A reproducible procedure is the requirement; bitwise equality across different hardware is **not** guaranteed.

---

# 8. Evaluation: predictive performance and ontology relations

## 8.1 Holdout discipline

Training samples can be inspected throughout training; validation samples can guide debugging, early stopping and configuration choices. **Test results and test-driven geometry inspection are reserved for an explicit final-evaluation action** after selection of the configuration. Dashboard views show split identity and hide test examples by default during development. A pre-training split-frequency audit may count test labels without exposing test model outcomes.

## 8.2 Multilabel pizza and component metrics

Record precision/recall/F1 with micro, macro and support-weighted aggregates, plus per-class support, confusion data and known-mask counts. Report exact-membership predictions for geometric models, sigmoid(logit) thresholds for neural baseline (DEFAULT 0.5), and common geometry soft indicators separately; calibrated thresholds are deferred. Predictions may contain zero, one or many classes. Fixed `DomainThing` is not part of headline metrics.

**Macro-F1 convention:** a class with **zero known-positive ground-truth observations in that evaluated split** is excluded from the macro average; its predictions, false positives and exclusion reason are **still reported**. Do not silently call its recall 0 or 1. Exclude classes whose labels are entirely unknown at that level, and use the authoritative scope/mask to decide which examples can contribute. This convention applies consistently to checkpoint scores and cross-model comparisons.

**Component metrics must distinguish:**

- `GT-ROI`: classify known GT components using annotated boxes (Phase-2 checkpoint monitor).
- `matched-only`: classify only matched predictions, with detection match rate reported; useful for diagnosis but **not** a sufficient end-to-end metric.
- `end-to-end`: under the validated **exhaustively annotated visible-components contract**, missed positive GT instances count as FN, unmatched predictions' positive ontology claims count as FP, and matched pairs contribute TP/FP/FN class by class. Duplicate predicted objects are therefore penalized. Instances with unknown class targets are masked, not treated as known negatives. A below-threshold predicted box and its missed GT can incur both FP and FN, consistently with the matching rule. The definition and denominator must be fixed across runs.

Track detector COCO-style mAP, AP@0.5, recall, category/count/small-object breakdown and component match coverage independently. An end-to-end component score is not merely classification accuracy conditioned on easy detections.

**Operational checkpoint monitors** superseding the earlier generic-validation-loss proposal:

| Phase | Best-checkpoint and early-stop monitor |
|---|---|
| 1 | Detection `mAP@0.50:0.95` |
| 2 | Half pizza macro-F1 + half component GT-ROI macro-F1 |
| 3 | Half pizza macro-F1 + half end-to-end component macro-F1 |

All alternatives can be explicitly configured; validation losses and other metrics remain recorded. If a monitor is undefined because of no supported positives, fail with a diagnostic instead of inventing a numeric score. Metrics affecting checkpoint selection must be computed under stable, documented filtering/matching/thresholds.

## 8.3 Geometric relation diagnostics (required capability; final formal protocol pending)

Given `A⊑B`, check whether learned `R_A⊆R_B`; for disjointness check `R_A∩R_B=∅`. Overlap and exclusion can also be inspected. Compare with ontology hierarchy/disjointness metadata, but **do not claim formal relation preservation from point classification alone**.

Implement a common relation-diagnostic interface with **exact** containment/intersection/disjointness checks when mathematically implemented and justified (straightforward for ordinary boxes), and **explicitly labeled sampled or bounded estimates** otherwise (towns are more complex). Every relation result must include verification method/scope, sample count when sampled, and counterexample point(s) when found. “No violation observed among 10,000 probes” is *not* equivalent to a proof of region containment. Do not label unknown OWL relations as logically false without an accepted evaluation convention. Full supervisor-approved RQ2 protocol, precision/recall/F1 over ontology pairs, and complete Boolean/quantified concept reasoning remain deferred as **methodological decisions**; data retention and relation diagnostics are **not** deferred.

---

# 9. Experiments, dashboard, visualization and artifacts

## 9.1 Local command line and dashboard

Provide one local interactive dashboard plus an independently usable CLI. Both consume the **same config schema, validation and experiment store**. The dashboard can edit draft configurations, show all hyperparameters, validate constraints, **launch a local training subprocess**, stream progress/logs, inspect current and completed runs, and compare configurations. It must support **graceful stop/cancel and resumption**, saving a resumable checkpoint when possible; if a process dies abruptly, mark it interrupted and explain what was preserved. These modest local controls do not imply distributed scheduling, cloud execution, a sophisticated job queue or automatic tuning.

Start with a portable file-based run directory (optional SQLite discovery index), e.g.:

```text
experiments/runs/<run-id>/
  resolved-config.yaml
  manifest.json
  metrics.csv  # or equivalent structured event log
  checkpoints/{latest,best,...}
  snapshots/{step-id}/
  diagnostics/
  environment.json
```

A run's resolved configuration and provenance are immutable after launch; mutable progress/events/checkpoints are appended/updated under a documented atomic-write policy. Distinguish `draft`, `running`, `completed`, `stopped`, `interrupted`, `failed` states. Avoid simultaneous processes overwriting checkpoints or silently competing for one GPU; use an explicit local concurrency rule and informative error. A database/MLflow backend is optional, not required.

## 9.2 Geometry viewer: exact slices, not misleading projections

For `d=2`, draw actual physical boxes/towns and points. For `d>2`, an **exact 2D slice** fixes all coordinates except chosen `i,j` at an anchor vector `a` and evaluates full-dimensional membership on `(a_1,...,x_i,...,y_j,...,a_d)`. Offer zero anchor, manually entered coordinates, and an **anchor through a selected sample's actual embedding**. Show closed-body/excluded-window boundaries faithfully, support overlapping houses and class selection, and allow inspecting membership/margin for selected points. A 2D slice can be **empty** when other fixed coordinates exclude every point of a class; this is meaningful, not a renderer failure.

Provide **separate PCA/UMAP views for point clouds** (pizzas and individual components), clearly labeled as projections and **not** exact representations of class-region containment. Do not overlay PCA coordinates on physical geometry slices as if they shared axes. For separate independently trained runs, show synchronized **side-by-side** panels for corresponding classes/samples and keep each latent coordinate system distinct. Do not directly superpose unrelated embedding coordinate frames. Alignment algorithms are optional later analyses, not the default representation of learned space.

## 9.3 Traceability and historical snapshots

A diagnostic trace should connect **sample ID → original RGB image → GT components → predicted boxes and scores/categories → one-to-one matches → 256D ROI features → projected component points → global/composed/fused pizza points → per-class hard prediction, margin, loss → responsible house/window**. Unsupervised hybrid intermediates must be marked as **features**, not claimed to be independently correct ontology predictions.

Use a **fixed reproducible train/validation reference panel** of pizza and GT component IDs. At configured intervals, save model class geometry, point representations under deterministic evaluation mode, margins/predictions, IDs and metadata. During Phase 3 provide **both** (a) stable GT-ROI reference embeddings to isolate encoder drift and (b) actual current predicted detection embeddings/matches to show inference-pipeline drift. Generate snapshots in evaluation mode without dropout/augmentation; restore training mode and relevant RNG state afterward. Avoid dumping every high-dimensional per-point/per-class/window diagnostic tensor at every training step; snapshots and chunking have configurable storage limits and frequencies.

The dashboard must display training curves and gradients, per-class support and weight vectors, detector and ontology metrics, sample/region exploration, snapshots/playback, inactive geometry warnings, loss-specific behavior, paired window-mode comparisons and cross-seed aggregates. Provide exports of plots, numeric results, geometry corners, matching records and reproducibility manifests. Every plot should identify split, phase, run, checkpoint, axes/projection and whether its relation claim is exact or sampled.

---

# 10. Testing and acceptance gates

A feature is not complete merely because it runs on one dataset. Tests must state required mathematical and interface behavior, including deliberate exceptions for reference methods.

## 10.1 Dataset and semantic tests

- Manifest determinism and complete declared ontology class/individual separation; correct concrete component self-class and superclass closure; example tests for `HamTopping`, `JalapenoPepperTopping`, `DeepPanBase`, and the four supported flags.
- Roundtrip framework-neutral target/mask serialization, justified negative versus unknown behavior, source `infer_pizzaiolo_label` consistency, invalid/corrupt country metadata, no-origin/no-flag and unsupported Germany detection-category cases.
- Topping integer counts and `hasTopping` cross-check; stable occurrence IDs; schema and provenance hashes; unique/disjoint split IDs and per-split supervision-frequency reports.

## 10.2 Mathematical geometry tests

- Every physical box stays within its domain and meets minimum width; every contained window is strictly inside its body with required clearance, even for extreme raw parameters and float32 precision.
- Paired window modes produce equivalent initial **physical corners** and identical initial hard membership/margin on test points; independent windows can move outside bodies without changing subtraction semantics.
- Closed body boundary included, closed window boundary excluded; overlapping houses can reinclude removed points; exact membership and zero signed margin intentionally disagree in *boundary inclusion* convention.
- Flat `[N,d]` and chunked geometries match, including gradients to physical parameters when expected; no shape/broadcasting errors for zero/one/many points, classes, houses and windows.
- Primary signed-margin losses give finite nontrivial gradients for representative positive-outside/negative-inside errors, including house/window examples. **Reference box BCE's known flat-inside negative case must be documented and tested, not “fixed” by changing its formula.** Corrected reference town loss must pass source-conformance hand examples before acceptance.
- DeepSets permutation invariance, multiplicity sensitivity where distinguishable, valid empty-set output, bounded point outputs and shared class head across point kinds.

## 10.3 Integration, optimization, reproducibility tests

- Single backbone computation supplies transformed FPN maps, detector losses and selected detections; pixel ↔ transformed ↔ ROI ↔ normalized original coordinate contracts and very small/overlapping boxes are correct.
- Phase 2 freezes detector params **and running state**. Phase 3 ontology gradients reach CNN features and point networks but **not detached selected box coords**; detection-specific loss gradients reach corresponding detection heads. A single component point participates in both classification and composition.
- Category-agnostic deterministic Hungarian matching, duplicate filtering, unmatched/zero-match cases, no ground-truth pseudo-detections in Phase 3, and valid end-to-end component FP/FN accounting.
- Masked losses ignore unknown labels and empty-supervision instances. Accumulation over heterogeneous component counts matches the specified full effective-batch objective and gradients. Four geometric loss modes are selectable only on compatible geometry.
- Train → validate → checkpoint → resume → produce deterministic geometry snapshot → load in dashboard/CLI on a tiny CPU fixture; corrupted config/checkpoint/hash fails clearly; graceful stop preserves usable state where feasible.
- Numerical failure detection, gradient clipping, stable RNG/snapshot behavior, no inadvertent test-set leakage; per-class metrics and zero-positive macro-F1 rules tested.

**Required acceptance gate:** a tiny, CPU-feasible **end-to-end smoke test** exercises dataset preparation/validation, detector forward/loss, GT and predicted ROI embeddings, all supported classifier/loss paths, backpropagation, checkpointing, evaluation, snapshot export, and dashboard loading. It need not produce a useful trained classifier. Full training must not be attempted before this gate passes.

---

# 11. Phased implementation plan, with rationale

The implementing agent may adjust code-level sequencing, but must preserve demonstrable deliverables and gates. This order minimizes expensive debugging of expensive training runs.

1. **Schema and dataset module first.** Bring over tested generation/label/split functions, correct component targets and flag semantics, produce versioned manifests and coverage reports. Rationale: every later head and metric assumes an authoritative class vocabulary and known/unknown masks.
2. **Geometry and loss as independent CPU-testable packages.** Build physical boxes, towns, exact membership, margins, four loss modes and reducers using small synthetic points—no CNN. Rationale: mathematical errors are otherwise difficult to distinguish from detector/optimizer issues.
3. **Detector adapter and ROI interfaces.** Pin Torchvision, verify 256-channel FPN, anchor presets, one-forward training/prediction integration, coordinates, 22-category labels. Rationale: incorrect crop alignment can produce seemingly plausible but invalid embeddings.
4. **Point projections and three composition modes.** Match common initial weights, ensure component point sharing and zero-detection handling. Assemble baseline, box and town around one output/geometry contract.
5. **Three training phases with loss/gradient tests.** Implement effective-batch reductions, phase-specific freeze/monitor logic, lineage, resume and deterministic snapshots. Run CPU smoke tests before GPU benchmarking.
6. **Local run store, dashboard and geometry explorer.** Configuration editing and launch should reuse the already-tested CLI/experiment APIs. Geometry views should be faithful to underlying membership calculations.
7. **First reference dataset and controlled pilot.** Generate/validate the 4,800-image reference, measure actual class frequencies, detector bbox sizes, GPU memory and convergence; adjust configurable provisional defaults only by creating auditable new runs. Execute the five core comparisons and paired window study as resources permit.
8. **Scientific analysis in cooperation with supervisor.** Compare classification/generalization (RQ1) and geometrical relation diagnostics (RQ2), document reference-loss deviations and capacity confounds, then finalize relation-level evaluation methodology and subsequent experiments.

This order intentionally decouples **specification of behavior** from **choice of internal Python organization**. Luna may select libraries for config validation, storage, UI and logging provided the stated contracts and acceptance tests remain true. If a required Torchvision integration or reference mathematical branch cannot be realized as specified, document the obstacle and proposed minimal change rather than silently changing experimental meaning.

---

# 12. Open risks, non-goals and controlled flexibility

## 12.1 Known scientific limitations to report

- **Predictive vs structural validity:** correct multilabel predictions do not prove global inclusion/disjointness; ontology relation testing needs independent methodology.
- **Capacity confound:** neural linear, pure box and multi-house town heads differ in expressivity/parameter count; shared encoders and seeds improve but do not eliminate confounding.
- **Small-object bottleneck:** 224px images, many small toppings/flags and 7×7 ROIAlign may affect recall and point quality; log AP/recall by class/size.
- **Synthetic shortcuts:** recipes, common visual assets and flags may permit image-level classification without learning logically general ontology structure. Distinguish visual prediction from logical reasoning.
- **Gradient starvation:** min/max town aggregation can leave some windows/houses inactive; containment parameterization cannot represent an exactly absent window; observe rather than auto-repair.
- **Distribution shift:** Phase 2 uses perfect ROI boxes; Phase 3 uses imperfect detector output and may have very few matched components. Monitor detection coverage and objective scales.
- **Resource pressure:** 4,800 images, at least five core configs, multiple seeds, multiple phases and snapshots can be expensive. All schedules/batches are defaults subject to measured resource constraints; do not claim measured performance before running anything.
- **OWL assumptions:** complete synthetic annotations justify selected closed-world negatives; that contract is narrower than arbitrary open-world ontology reasoning. The learned universal `DomainThing` covers only observed entity types.

## 12.2 Explicitly deferred (do not implement as hidden prerequisites)

Cone-based geometry; learned relations/roles and quantified ALC operators; explicit OWL axiom penalties; unrestricted OWL interpretation; adaptive house/window counts and trainable on/off gates; exact general Boolean concept construction; smooth min/max; attention pooling; hybrid intermediate pizza losses; component-supervision-off ablations; confidence-weighted pooling; label-informed geometric initialization; fixed-encoder and stronger-MLP baseline comparisons; ResNet/448px and COCO-detector transfer alternatives; image augmentation; unbounded point spaces; calibrated nonzero thresholds; geometry-specific regularization; alternate alternating-optimization curricula; automatic hyperparameter tuning and exhaustive sweeps; MLflow requirement; cloud/distributed training and sophisticated scheduling; automatic multilabel re-stratification; coordinate-frame alignment of different learned runs; final formal RQ2 metric design until supervisor consultation.

**Not deferred:** both window modes, all three pizza-composition modes, four geometric losses (native compatibility), corrected dataset targets/masks and validation, all three training phases, local dashboard launch/cancel/resume, exact slice viewer, relation diagnostics, environment manifests, and the required smoke test.

## 12.3 Assumptions and delegated details that require transparent resolution

These are not invitations to silently invent new research decisions. They are bounded implementation choices:

1. **Torchvision adapter/API:** verify supported release, anchor pyramid mapping, exact train/predict path and FPN map names. If unavoidable deviations occur, record them and demonstrate equivalent forward/gradient behavior.
2. **Numerical parameterization:** any stable differentiable transform is acceptable if physical corners, minimum widths and contained-window clearances meet the specified tests. Publish the actual transform in code/docs.
3. **Reference town loss correction:** requires explicit documented source-conformance mathematical note and hand-tested examples before coding can be considered faithful. If ambiguity persists, escalate for the author's approval rather than choose a new objective under the old name.
4. **Provisional defaults:** choose documented feasible values for unspecified minor activation, weight cap, effective-number `β`, anchor preset before bbox statistics, warmup duration, initialization width distributions, snapshot panel size/interval, source-conformance tolerances, dashboard run concurrency, and storage quota. They must be visible/configurable, not immutable research claims.
5. **Canonical town mode in first five-way run:** select explicitly and hold fixed; the required paired independent/contained comparison tests the effect of that separate choice. Do not compare baseline/box using meaningless window-mode flags.
6. **Exact town relation checks:** use exact only when proven correct, otherwise label probabilistic/sampled diagnostics. Do not invent proof claims.
7. **Operational cancellation:** graceful interruption is best effort; forced termination may prevent last-minute checkpoint writing. Correct run status and resume from last valid checkpoint are still required.

---

# 13. Acceptance checklist for author/agent

A reviewer should be able to verify all of the following without reading the original architecture interview:

- [ ] Dataset generation is independently usable, pinned, audited, versioned, and ontology-complete for its declared supervised scope.
- [ ] Component targets include concrete topping/base self-classes; flag identities remain individuals; only four supported countries become detector categories.
- [ ] One class manifest and one masked target contract govern every model, loss, checkpoint and metric.
- [ ] Only pixels and selected visual ROI/box-position information reach inference-time ontology points.
- [ ] Faster R-CNN shares one CNN/FPN feature computation for detection, ROI embeddings, global features and joint loss.
- [ ] Component points and complete-pizza points share one ontology class head/region set.
- [ ] Global, compositional DeepSets and hybrid pizza construction all work, including zero detections.
- [ ] Box and town geometry implement set membership and boundary conventions exactly, with validated stable coordinate transformations.
- [ ] Both town window modes begin with physically identical paired initialization and diverge under their respective constraints.
- [ ] Signed BCE and hinge use the same surrogate margin; box/town reference losses are separately documented faithful adaptations.
- [ ] Weighting, unknown masks, class coverage, reduction denominators and accumulated effective-batch gradients follow one tested implementation.
- [ ] Detector pretraining, frozen GT-ROI embedding training and joint predicted-ROI fine-tuning produce resumable, lineage-linked runs.
- [ ] Evaluation distinguishes GT-ROI, matched-only and end-to-end component performance, and reports unsupported validation classes explicitly.
- [ ] Dashboard can configure and launch/stop/resume runs, inspect hyperparameters, examine exact slices and historically trace samples/components.
- [ ] Geometry relation diagnostics clearly distinguish exact proofs from empirical evidence and do not assume OWL-axiom satisfaction.
- [ ] Test split remains a true final holdout; comparison runs use common detector/weights/dataset policy and at least three seeds.
- [ ] Required CPU-feasible smoke test, mathematical unit tests and numerical-gradient tests pass before expensive training.

## 13.1 Final interview amendments (Q188–Q191)

These last four approvals remove ambiguity left by earlier checkpoints:

- **Q188 — one monitor per phase:** the documented phase-specific validation metric controls **both** early stopping and best-checkpoint selection. The initial suggestion to stop on a different validation loss is superseded.
- **Q189 — shared detector lineage:** for each comparable dataset/seed, select one Phase-1 detector checkpoint and reuse it across independently trained Phase-2 model/loss configurations; every Phase-3 fine-tune starts from its own Phase-2 descendant.
- **Q190 — lightweight run lifecycle:** dashboard subprocess launch, graceful cancellation where possible, clear interrupted state and checkpoint-based resumption are required. A complex job scheduler is not.
- **Q191 — requirement tiers:** mathematical and scientific contracts are mandatory, numeric defaults are explicitly provisional/configurable, and implementation mechanics that preserve the contracts remain the agent's discretion.

---

# 14. References and provenance

**Author-provided/reviewed materials:**

- *Thesis Proposal-2.pdf* — **Representing Ontologies using Town Embeddings**, proposal dated 19 June 2026; research questions, methods and intended ontology-relation analysis.
- *pizzaiolo.xml* — supplied OWL ontology; class/individual structure, inherited concepts, defined concepts, named recipes, disjointness.
- *Pasted text(7).txt* — Luna's read-only audit of `fourth_attempt/` and a separate historical dataset. Its initial claim that flags are metadata-only was superseded by the subsequent source audit supplied in the conversation.
- `thesis_architecture_snapshot_v0_1.md`, `thesis_architecture_checkpoint_v0_2.md` — earlier consensus checkpoints; this document incorporates accepted decisions Q171–Q191 in addition.
- Source generator: https://github.com/SysReIC/pizzaiolo/tree/6c0bcb221a957c184012beaf103bebcde4b1f793
- Box reference code: https://github.com/adelmemariani/python-chebai/blob/Box4Chemi/chebai/models/electra.py (ChebiBox and BCE scoring).
- Town reference code: https://github.com/MGlauer/town-embeddings/blob/convex_cone_geometry/potemkin/models/town_model.py and https://github.com/MGlauer/town-embeddings/blob/convex_cone_geometry/potemkin/loss/town_loss.py
- Memariani et al. (2025), *Box embeddings for extending ontologies: a data-driven and interpretable approach*, *Journal of Cheminformatics*. The article PDF was provided in the conversation as `s13321-025-01086-1.pdf`.
- Glauer et al., *Extended Abstract: Embeddings for the Description Logic ALC*, town/house/window formalism, supplied during earlier architecture discussion.

**Interpretation note:** The specification documents an agreed design, not experimental findings. Any statements about expected training stability, relative performance, relation fidelity or memory costs are hypotheses/rationales until verified by measured runs.


---

# Appendix A. Complete decision-to-requirement crosswalk (Q1–Q191)

This crosswalk prevents compact thematic sections from obscuring an accepted decision. Each entry states the accepted requirement and *why it matters*. The main text above is authoritative when it gives an exact formula, interface or test. Entries marked **revised** or **updated** reflect later accepted changes; earlier snapshots do not override these. The entries paraphrase the interview and are not intended as exact question quotations.

## A.1. Initial research architecture and representation

- **Q001 — Whole pizzas and components are points in a common latent domain.** *Rationale:* The same class geometry should classify both parts and wholes.
- **Q002 — Inference uses RGB pixels, not recipe metadata.** *Rationale:* Derived labels are supervision, not inference shortcuts.
- **Q003 — Detector crops contribute visual features and locations.** *Rationale:* Geometry should classify grounded visual regions, not coordinates alone.
- **Q004 — Train in three stages and jointly fine-tune.** *Rationale:* Separate detector competence from geometry learning, then test realistic detection.
- **Q005 — Learn from multilabel ancestor targets without explicit OWL-axiom penalties.** *Rationale:* Whether ontology relationships emerge is a research question, not a training guarantee.
- **Q006 — Compare baseline BCE, boxes with BCE/distance, and towns with BCE/distance.** *Rationale:* A five-configuration core gives interpretable comparisons.
- **Q007 — Record predictive metrics and preserve future ontology evaluation data.** *Rationale:* The supervisor will approve final relation-evaluation methodology.
- **Q008 — Exclude cones from the required initial system.** *Rationale:* Keep the research scope on boxes versus towns.
- **Q009 — Support global, compositional, and hybrid pizza points.** *Rationale:* Test whether individually grounded components help whole-pizza classification.
- **Q010 — One shared class vocabulary/representation across pizza and component levels.** *Rationale:* Prevent unrelated output heads from undermining common-space semantics.
- **Q011 — Use a class-aware detector, but keep its predicted class labels out of ontology points.** *Rationale:* Detector identification is supervision/diagnostics; embeddings must use visual ROI evidence.
- **Q012 — Represent whole pizza by a full-image box for ROI extraction.** *Rationale:* The global branch has an unambiguous spatial input.
- **Q013 — Supervise each component with its own ontology memberships, pizza with pizza labels.** *Rationale:* Avoid assigning whole-pizza classes to individual toppings.
- **Q014 — Keep repeated topping occurrences distinct until aggregation.** *Rationale:* Multiplicity matters for cardinality-based pizza concepts.
- **Q015 — Use ROIAlign from the common CNN feature pyramid.** *Rationale:* Avoid a second crop encoder or duplicate backbone.
- **Q016 — Detector pretrain, frozen GT-ROI ontology training, then joint predicted-ROI tuning.** *Rationale:* Separates ideal-region learning from deployment-like detection errors.
- **Q017 — Share architectures and matching initialization, not necessarily trained weights.** *Rationale:* Isolate geometry effects while allowing independent representation adaptation.
- **Q018 — Implement all three pizza construction modes.** *Rationale:* They are configurable experimental factors, not alternate codebases.
- **Q019 — Use permutation-invariant DeepSets with multiplicity and location preserved.** *Rationale:* Component order is arbitrary but component count and placement are meaningful.
- **Q020 — Default latent dimension 16; include 2 and 8.** *Rationale:* Study expressivity and obtain exact 2D geometry where useful.
- **Q021 — Default towns have two houses and one window per house, fixed per class per run.** *Rationale:* Start with modest compositional capacity while permitting capacity ablations.
- **Q022 — Expose exact 2D slices and separate PCA/UMAP point views.** *Rationale:* Projected points must not imply projected-region containment.
- **Q023 — Save geometric history and fixed reference embeddings.** *Rationale:* Observe evolution rather than only final classification.
- **Q024 — Build a standalone local interactive geometry/dashboard application.** *Rationale:* Research requires inspection without an active training process.
- **Q025 — Trace evidence from image through detector, embeddings, geometry and losses.** *Rationale:* Make apparent predictions explainable and errors diagnosable.
- **Q026 — Defer learned object-role embeddings and quantified ALC semantics.** *Rationale:* Do not silently expand the objective into full description-logic reasoning.
- **Q027 — Use bounded latent domain [-1,1]^d.** *Rationale:* Stabilize valid class geometry and allow a finite universal reference domain.
- **Q028 — Use Faster R-CNN style two-stage CNN detector, not YOLO.** *Rationale:* Allow explicit FPN/ROI features in a transparent staged pipeline.
- **Q029 — Initialize the CNN backbone from pretrained visual weights by default.** *Rationale:* More reliable low-data visual representations; optional random-init ablation later.
- **Q030 — Use sum-based DeepSets, not attention, initially.** *Rationale:* Preserve invariance and reduce new architectural variables.
- **Q031 — Train separate component/global projection heads with common output space.** *Rationale:* Component crops and entire pizza images have different input statistics.
- **Q032 — Use GT ROIs in Phase 2, predicted ROIs in Phase 3.** *Rationale:* Reveal the effect of imperfect detection on ontology classification.
- **Q033 — One local experiment/dashboard interface with geometry viewer.** *Rationale:* Visualize runs and compare their scientific outputs.
- **Q034 — Give each experiment immutable identity, metadata and artifacts.** *Rationale:* Prevent loss of configuration/data/seed lineage.
- **Q035 — Use MobileNetV3-Large-FPN as default backbone.** *Rationale:* Reasonable cost for 224px synthetic images; stronger alternatives deferred.
- **Q036 — Native input 224×224, higher resolution configurable later.** *Rationale:* Stay compatible with the source generator for core comparisons.
- **Q037 — ROIAlign 7×7 on 256 FPN channels, spatial mean for component ROI.** *Rationale:* Produce fixed-size component visual vectors.
- **Q038 — Append normalized center-x, center-y, width, height to 256 features.** *Rationale:* Retain location and scale without leaking category labels.
- **Q039 — Derive global visual features via a full-image 7×7 ROI.** *Rationale:* Preserve full scene representation independent of detected components.
- **Q040 — Use ordered bounded-corner parameterization for all boxes.** *Rationale:* Invalid boxes should be impossible by construction.
- **Q041 — Require both independent and contained window parameterization modes.** *Rationale:* Directly test whether allowing windows to escape houses affects learning.
- **Q042 — Jointly train common visual features with detector and ontology losses.** *Rationale:* Ontology task should shape representations while detection remains supervised.
- **Q043 — Integrate adapted dataset module into one self-contained repository [revised].** *Rationale:* Independent responsibility need not require a separate maintained repository.
- **Q044 — Use declarative validated run configs and CLI overrides.** *Rationale:* Experiments should be repeatable and atomically composable.
- **Q045 — Component MLP 260→256→128→d with LayerNorm/ReLU and 0.1 dropout.** *Rationale:* Keep a fixed documented projection architecture across model families.
- **Q046 — Global projection 256×7×7→32 via 1×1 conv→1568→256→d.** *Rationale:* Preserve some spatial layout before global point prediction.
- **Q047 — Use two-layer phi:d→128, sum, two-layer rho:128→d.** *Rationale:* Initial composition should be simple, invariant and interpretable.
- **Q048 — Hybrid fusion concatenates global and compositional points, 2d→128→d.** *Rationale:* Let the model combine complementary scene and component evidence.
- **Q049 — Bound every final point-producing head with tanh.** *Rationale:* Match point outputs to the finite geometry domain.
- **Q050 — Baseline Linear(d,C) plus BCEWithLogits.** *Rationale:* Compare against a deliberately simple shared-space nongeometric classifier.
- **Q051 — Use bounded ordered-corner transform, later strengthened by min-width constraint.** *Rationale:* Valid physical geometry should not depend on post-step clipping.
- **Q052 — Window modes differ in parameterization, not semantic subtraction.** *Rationale:* Keep house/window class definitions identical across experiments.
- **Q053 — Match initial physical geometries across window-mode comparisons.** *Rationale:* Attribute changes to constraints, not different initial shapes.
- **Q054 — Separate geometry calculation from loss modules.** *Rationale:* Train/evaluate different objectives without duplicating membership code.
- **Q055 — Select individual experiments rather than exhaustively sweeping combinations.** *Rationale:* Limit compute while retaining future composability.
- **Q056 — Bodies closed, closed window boundaries removed.** *Rationale:* Exact membership needs unambiguous topological conventions.
- **Q057 — No learned window-activation or exact zero-width mask initially.** *Rationale:* Keep fixed-capacity geometry simple and track inactivity explicitly.
- **Q058 — Component auxiliary supervision remains active in global-only mode.** *Rationale:* Isolate pizza composition choice from component supervision choice.
- **Q059 — DeepSets of empty component list uses zero sum and rho(0).** *Rationale:* No detection must yield a valid reproducible representation.
- **Q060 — Give pizza and component losses independent configurable coefficients.** *Rationale:* Avoid implicit domination by one task; record gradient influence.

## A.2. Supervision, losses, repository and comparative controls

- **Q061 — Use compositional signed margin sB, sH=min(body,-windows), sT=max(houses).** *Rationale:* Create one common geometric score across pure boxes and towns.
- **Q062 — Primary losses should give useful gradients for incorrect inside and outside membership.** *Rationale:* Avoid the known false-positive flat-gradient pathology.
- **Q063 — Treat absent class as known negative only under validated completeness; otherwise unknown.** *Rationale:* Synthetic closed-world data is not arbitrary OWL open-world entailment.
- **Q064 — Support none, inverse-sqrt, and effective-number positive class weighting.** *Rationale:* Test rare-class sensitivity without assuming one scheme is best.
- **Q065 — Average component losses per image, then over images with valid component targets.** *Rationale:* Avoid rewarding heavily topped pizzas simply for having more objects.
- **Q066 — No initial geometry-specific regularization, but log degeneracy and reserve extension hooks.** *Rationale:* Observe whether geometry learns holes without imposing extra priors.
- **Q067 — Correct targets to include concrete component's own OWL class.** *Rationale:* The class region must learn from positive examples of that very concept.
- **Q068 — Keep full ontology manifest separate from supervised class vocabulary.** *Rationale:* Do not train unsupported classes as universally negative.
- **Q069 — Keep country names as individuals, not invented OWL classes.** *Rationale:* Respect ontology semantics while retaining detectable visual flags.
- **Q070 — Represent supervision with labels plus known/unknown masks.** *Rationale:* Prevent unjustified negative training on incomplete facts.
- **Q071 — Require a semantic/data-integrity audit before actual training.** *Rationale:* Avoid interpreting learning on inconsistent generated targets.
- **Q072 — Move semantic target derivation to integrated dataset module [revised].** *Rationale:* The model loader must not implement generator or ontology reasoning.
- **Q073 — Retain visually rendered flag instances with identity only for detector supervision.** *Rationale:* Flags are actual image evidence, while country names remain OWL individuals.
- **Q074 — Add original-style loss modes alongside signed-margin BCE and hinge.** *Rationale:* Relate controlled experiments to the scientific references.
- **Q075 — Use sigmoid(s/tau) BCE logits for primary geometric BCE.** *Rationale:* Penalize inside negatives and outside positives smoothly.
- **Q076 — Use hinge max(0,m-t*s) for primary geometric distance loss.** *Rationale:* Require a tunable separation clearance, then stop pushing once satisfied.
- **Q077 — Keep tau and margin configurable but not trainable initially.** *Rationale:* Avoid a moving score scale as a comparison confound.
- **Q078 — Use exact min/max for first town margin, not smooth approximations.** *Rationale:* Preserve compositional semantics and log possible gradient starvation.
- **Q079 — Expose per-instance/per-class margin, probabilities and loss diagnostics optionally.** *Rationale:* Enable traceable model failures without large routine overhead.
- **Q080 — Use 22 detector foreground categories plus background, from manifest.** *Rationale:* Represent 16 toppings, 2 bases and 4 flags without hardcoded ontology identities.
- **Q081 — Match predicted and GT component boxes one-to-one with IoU threshold.** *Rationale:* Supervise correct localized regions and avoid duplicate target assignment.
- **Q082 — Let every selected predicted region enter pizza composition, even unmatched.** *Rationale:* Match actual inference conditions rather than training only on perfect detections.
- **Q083 — Make score filtering, NMS and max detections configurable.** *Rationale:* Selection changes composition and must be a controlled variable.
- **Q084 — Disable augmentation for initial controlled runs.** *Rationale:* Avoid another source of variance, especially for tiny toppings and flags.
- **Q085 — Copy and adapt tested fourth_attempt data code into unified repository.** *Rationale:* Keep a single maintained source of truth without starting from scratch.
- **Q086 — Dataset artifacts must be model/framework independent.** *Rationale:* Allow inspection and re-use outside PyTorch.
- **Q087 — Immutable, reusable dataset versions separate from run identities.** *Rationale:* Training comparisons must not regenerate random images per model.
- **Q088 — Assign stable IDs to each component occurrence.** *Rationale:* Support matching and temporal evidence tracing.
- **Q089 — Dashboard can validate config and launch local CLI subprocess [revised].** *Rationale:* Lightweight execution does not require complex orchestration.
- **Q090 — Centralized validated experiment schema shared by CLI and UI.** *Rationale:* Reject conflicting or meaningless parameter combinations.
- **Q091 — Use portable local artifact files, optional SQLite index.** *Rationale:* Keep experiment history exportable without a tracking server.
- **Q092 — Test numerical invariants and module integration, not just code import.** *Rationale:* Geometry semantics are part of research correctness.
- **Q093 — Save model, optimizer, scheduler, RNG and phase state for resumption.** *Rationale:* Checkpoints must reproduce training state, not merely inference weights.
- **Q094 — Share one class head across all instance types.** *Rationale:* Class geometry must have the same interpretation for toppings and pizzas.
- **Q095 — Learn global per-class regions, not image-conditioned region parameters.** *Rationale:* Geometric ontology concepts should be stable across images.
- **Q096 — Treat DomainThing as fixed full restricted domain, not learned [narrow OWL scope].** *Rationale:* Do not optimize a universally positive class or claim full OWL universe coverage.
- **Q097 — Supervise final hybrid pizza point only, plus component points.** *Rationale:* Allow global/composed branches to specialize before fusion.
- **Q098 — Pad downstream component batches with explicit validity masks.** *Rationale:* Handle variable counts without polluting pooling or loss.
- **Q099 — Broad overlapping label-agnostic region initialization.** *Rationale:* Avoid initially excluding almost every high-dimensional point.
- **Q100 — Match common neural initialization across independently trained model runs.** *Rationale:* Reduce the influence of initial encoder randomness on comparisons.
- **Q101 — Use known negatives across entire trainable class vocabulary.** *Rationale:* Shared-space geometry needs separation between entity categories.
- **Q102 — Retain optional intermediate embedding diagnostics.** *Rationale:* Understand hybrid and compositional behavior without adding auxiliary labels.
- **Q103 — Include four compatible geometric losses with research interpretation [reconciled].** *Rationale:* Avoid falsely treating adapted losses as exact reproduction of sources.
- **Q104 — Restrict DomainThing's universal geometry to represented dataset entities.** *Rationale:* Explicitly separate synthetic interpretation from OWL semantics.
- **Q105 — Use hard geometry membership as principal geometric prediction.** *Rationale:* Calibrating soft-score thresholds must not redefine learned regions.
- **Q106 — Do not evaluate unsupervised hybrid intermediates as ontology predictions.** *Rationale:* They are features, not independent targets.
- **Q107 — Train matched-initialization encoders separately for each run.** *Rationale:* Different losses should be allowed to shape their point distributions.
- **Q108 — Normalize detector, pizza and component objectives explicitly.** *Rationale:* Prevent counts and vocab size changing objective scales by accident.
- **Q109 — Start joint loss group coefficients at 1.0 with overrides.** *Rationale:* Provide interpretable starting point and gradient diagnostics.
- **Q110 — Use AdamW and learning-rate groups for backbone, detector, projection and geometry.** *Rationale:* Different parameter groups need controlled update scales.
- **Q111 — Use capped epochs, phase-specific validation monitor and early stopping [updated].** *Rationale:* Avoid arbitrary stopping duration and inconsistent checkpoint criteria.
- **Q112 — Separate exploratory single-seed from comparison >=3-seed runs.** *Rationale:* Distinguish debugging from evidence of repeatability.
- **Q113 — Reference losses only on their original geometry types.** *Rationale:* Avoid undocumented invented cross-geometry objectives.
- **Q114 — Keep common CNN/latent architecture for reference-loss adaptations.** *Rationale:* Compare their loss ideas on a common visual task, not entire foreign models.
- **Q115 — All geometric inference independent of training-loss choice.** *Rationale:* Evaluate learned regions consistently by exact membership.
- **Q116 — Positive-only class weighting; negatives have weight 1.** *Rationale:* Increase rare-positive influence without magnifying all negative labels.
- **Q117 — Compute positive frequencies separately for pizza and component instances.** *Rationale:* Their statistical units and distributions differ.
- **Q118 — Fix DomainThing positive in the neural baseline as well.** *Rationale:* Avoid trivially inflating its measured performance.
- **Q119 — Choose best checkpoints via common validation macro-F1 [later phase-specific].** *Rationale:* Different losses have incomparable scalar values.
- **Q120 — Enforce float32 for geometry even with CNN AMP.** *Rationale:* Protect small widths, distances and boundary comparisons.

## A.3. Detector integration, geometry and evaluation

- **Q121 — One integrated Faster R-CNN forward exposes features, predictions and losses.** *Rationale:* Avoid duplicating expensive backbone computation.
- **Q122 — ImageNet-pretrained backbone only by default, not COCO detection classifier.** *Rationale:* Detector categories must match synthetic components.
- **Q123 — Select detector anchor presets using training-set box statistics.** *Rationale:* Tiny toppings and large bases require appropriate proposal scales.
- **Q124 — Initially fine-tune later three backbone stages, configurable.** *Rationale:* Control adaptation of pretrained visual representations.
- **Q125 — Embed final filtered predicted boxes, not all raw proposals.** *Rationale:* Ensure composition uses actual reported detections.
- **Q126 — Conservatively remove extreme-overlap cross-category duplicates, configurable.** *Rationale:* Prevent fake multiplicity without suppressing real overlap aggressively.
- **Q127 — Do not add missed ground-truth crops during predicted-ROI Phase 3.** *Rationale:* Preserve actual detector error distribution during training.
- **Q128 — Use detector confidence for selection only, not semantic point features/pooling weights.** *Rationale:* Avoid shortcutting through detector labels or scores.
- **Q129 — Use one combined backward pass for joint objectives.** *Rationale:* Make permitted shared parameters respond to all supervision simultaneously.
- **Q130 — Centralize original/transformed/ROI coordinate mappings.** *Rationale:* Avoid misaligned crops as resizing changes.
- **Q131 — Make region initialization dimension-aware.** *Rationale:* Width per axis can imply near-zero volume in high dimensions.
- **Q132 — Initialize windows to affect non-negligible high-dimensional volume.** *Rationale:* Prevent a town behaving like boxes because all holes are ineffective.
- **Q133 — Start houses with distinct seeded offsets.** *Rationale:* Avoid redundant symmetric houses and identical gradients.
- **Q134 — Choose label-agnostic rather than data-fitted initial regions.** *Rationale:* Let losses learn class placements rather than preload positive examples.
- **Q135 — Detect inactive/degenerate geometry and log it, without auto-fixing.** *Rationale:* Diagnose min/max gradient starvation as a research outcome.
- **Q136 — Initial reference dataset 4,800 with 16 balanced generator types.** *Rationale:* Ground the first experiment in one feasible reproducible sample set.
- **Q137 — Require training positives for all intended classes in comparison runs.** *Rationale:* Avoid scoring models on classes they cannot learn.
- **Q138 — Target single CUDA GPU, with CPU fixtures/debug available.** *Rationale:* Keep research implementation feasible on one workstation.
- **Q139 — Physical batch 4, accumulation 4 by default.** *Rationale:* Fit expensive detector features while targeting effective batch 16.
- **Q140 — Use warmup and cosine schedule, constant option.** *Rationale:* Make learning-rate changes controlled and reproducible.
- **Q141 — Use equal pizza/component macro-F1 provisional checkpoint score [updated].** *Rationale:* Balance both supervised levels, later use E2E components in Phase 3.
- **Q142 — Use five canonical baseline/box/town configurations first.** *Rationale:* Keep first comparison tractable before separate ablations.
- **Q143 — Match Phase-3 ROIs independent of predicted detector category.** *Rationale:* Localization is distinct from categorical detector errors.
- **Q144 — Use deterministic maximum-weight one-to-one IoU assignment.** *Rationale:* Prevent iteration order deciding component supervision.
- **Q145 — Detach selected box coordinates from ontology losses.** *Rationale:* Backpropagate through FPN features, not discrete detection choices.
- **Q146 — Freeze detector parameters and state in Phase 2.** *Rationale:* A frozen backbone must not drift through BatchNorm updates.
- **Q147 — Identical ROI-to-point architecture in Phases 2 and 3.** *Rationale:* Expose rather than hide distribution shift between GT/predicted boxes.
- **Q148 — Global branch independent of detection success.** *Rationale:* Global/hybrid should still function when no components are found.
- **Q149 — Skip nonexistent component supervision without reducing pizza loss.** *Rationale:* Missing matches must not manufacture zeros or invalidate whole-pizza labels.
- **Q150 — Share each component point between class membership and aggregation.** *Rationale:* Ensure compositional evidence is the same evidence classified as a part.
- **Q151 — Separate crisp region contains from signed score.** *Rationale:* Distinguish boundary topology from continuous training surrogate.
- **Q152 — Do not introduce arbitrary boundary epsilon into the loss.** *Rationale:* Preserve mathematically defined surrogates and handle exact membership separately.
- **Q153 — Guarantee minimum physical box width in float32.** *Rationale:* Avoid sigmoid saturation creating degenerate stored regions.
- **Q154 — Match physical town geometry between window-mode initializations.** *Rationale:* Make paired runs truly comparable at step zero.
- **Q155 — Give pure box and town a common geometry API.** *Rationale:* Keep losses and evaluators agnostic to specific region parameter structures.
- **Q156 — Geometry core accepts flat [N,d] points; caller handles padding.** *Rationale:* Keep mathematical primitives isolated from detector batch shapes.
- **Q157 — Expose complete house/window diagnostic data on demand.** *Rationale:* Explain which subregion creates membership and loss.
- **Q158 — Keep reference BCE's flat-gradient behavior visible.** *Rationale:* Preserve reference scoring, rather than quietly changing its limitations.
- **Q159 — Show true high-dimensional region slices, not geometric PCA projections.** *Rationale:* Prevent visually implied but false containment statements.
- **Q160 — Offer slice anchors through selected points.** *Rationale:* Inspect exact local class boundaries around a concrete example.
- **Q161 — Dataset manifest is authoritative for class ordering and IDs.** *Rationale:* Every head and checkpoint must agree on output column semantics.
- **Q162 — One sample structure across all model families.** *Rationale:* Geometry choice must not alter training target meaning.
- **Q163 — Centralize detector image preprocessing and transformations.** *Rationale:* Avoid normalization/resizing twice and inconsistent coords.
- **Q164 — Weight using positive known-training counts and validated formula.** *Rationale:* Ground rare-class effects in actual supervision statistics.
- **Q165 — All heads return unreduced losses to a common masked reducer.** *Rationale:* Keep weighting/normalization identical across experiments.
- **Q166 — Require end-to-end component precision/recall/F1.** *Rationale:* Matched-only success otherwise rewards low detector recall.
- **Q167 — Use E2E component macro-F1 for Phase-3 checkpointing [updated].** *Rationale:* Reward the whole pipeline, not only easy matched detections.
- **Q168 — Reserve test outputs for explicit final evaluations.** *Rationale:* Prevent repeated test inspection from directing model development.
- **Q169 — Track geometry with fixed train/validation reference panel.** *Rationale:* Compare the same samples and regions across checkpoints.
- **Q170 — Compare different learned spaces in synchronized side-by-side views.** *Rationale:* Independent latent coordinate frames are not directly overlayable.

## A.4. Numerical reproducibility and final amendments

- **Q171 — Exclude zero-known-positive classes from macro-F1 but report FP and exclusions.** *Rationale:* Avoid inventing recall for unsupported validation classes.
- **Q172 — Require exhaustive synthetic component annotations for E2E metric.** *Rationale:* A missed visible object can only count as FN under known completeness.
- **Q173 — Guarantee contained window minimum width and clearance from parent boundaries.** *Rationale:* Preserve strict interior geometry under float32 extremes.
- **Q174 — Preserve source-intended town reference branch/distance aggregation.** *Rationale:* Do not replace a problematic prototype with an unrelated objective.
- **Q175 — Allow equivalent chunked geometry evaluation and gradients.** *Rationale:* Bound memory without changing outcomes.
- **Q176 — Snapshot stable GT ROIs separately from current predicted ROIs.** *Rationale:* Disentangle encoder drift from detection selection drift.
- **Q177 — Default full float32; AMP only by opt-in.** *Rationale:* First verify numerical behavior without mixed-precision confounds.
- **Q178 — Stop and diagnose NaN/Inf instead of silent recovery.** *Rationale:* Keep experiment trajectories auditable.
- **Q179 — Choose detector checkpoint by mAP@0.50:0.95.** *Rationale:* Base detector selection on localization/classification quality, not training loss.
- **Q180 — Evaluate every supervised point against the full trainable class vocabulary.** *Rationale:* Avoid stochastic negative subsampling for a modest ontology.
- **Q181 — Use all matched components for ontology loss; detector samples proposals internally.** *Rationale:* Do not introduce another instance sampler into loss comparison.
- **Q182 — Match loss reductions across gradient-accumulation microbatches.** *Rationale:* Correct effective-batch weighting even when detections differ by image.
- **Q183 — Generate deterministic evaluation-mode snapshots and restore training RNG/state.** *Rationale:* Changes in plots should reflect learning, not dropout noise.
- **Q184 — Expose class-region containment and disjointness diagnostics.** *Rationale:* RQ2 concerns global ontology structure beyond point accuracy.
- **Q185 — Label relation evidence as exact or empirically sampled.** *Rationale:* No observed probe violation does not imply a proof of containment.
- **Q186 — Record software, hardware, commit and determinism provenance.** *Rationale:* Make comparison-quality runs independently reproducible.
- **Q187 — Require a tiny end-to-end smoke test before full training.** *Rationale:* Catch integration errors before expensive experiments.
- **Q188 — Use phase-specific metric for both best-checkpoint and early stopping.** *Rationale:* Avoid inconsistent monitors for a single training phase.
- **Q189 — Reuse one Phase-1 detector per dataset/seed across Phase-2 comparisons.** *Rationale:* Control detection initialization as a confound.
- **Q190 — Allow dashboard launch, graceful cancellation and checkpoint resume.** *Rationale:* Lightweight local process control is useful without a scheduler.
- **Q191 — Clearly separate required behavior, configurable defaults and agent freedom.** *Rationale:* Let implementation remain composable without silently changing scientific choices.

---

# Appendix B. Explicit public data, model and run contracts

This appendix makes the thematic main text actionable without mandating one Python architecture. Examples are *illustrative wire formats*, not a license to redefine the semantic fields. The authoritative field names and numeric storage formats must be fixed by the implementation schema, documented and schema-validated before the first comparison-quality run. This is an implementation task, not an additional research-architecture choice.

## B.1 Prepared dataset schema obligations

A prepared dataset version must be usable without importing the generator, an OWL reasoner, Torchvision or the dashboard. Its public contract must include:

- `schema_version`, `dataset_id`, `manifest_id`, source generator commit, preparation commit, ontology full-text hash, generation seed/config, class manifest, detector categories, split assignments, integrity report and asset checksums or equivalent integrity records.
- One **ontology class** vocabulary of fixed stable integer IDs and full names/IRIs, including flags for `trainable`, `fixed_positive`, `pizza_supervision`, `component_supervision`, and `unsupported`. A **separate detector-category** vocabulary includes 22 foreground types, with four distinct rendered flag identities. Named country individuals must not be silently included as class-region outputs.
- For each sample: immutable ID, RGB image path and dimensions, original-image GT boxes, pizza targets and known mask, component occurrences with immutable IDs and visual detector categories, ontology targets/masks. Raw recipe metadata may be retained in an audit-only payload but is never a model input.
- A documented conversion from the original generator's `[x,y,width,height]` to conventional `[x1,y1,x2,y2]`, including whether coordinates are inclusive, pixel-edge, float, or integer. All derived component IDs must be deterministic when several occurrences share one class.
- A documented negative/unknown rule: **validated generated no-origin/no-flag is known absence**; malformed or imported incompleteness is rejected or explicitly masked. The full ontology inventory is retained even though unsupported classes do not become trained negatives.
- Integrity gate: every referenced asset exists and every annotation/flag/topping-count and split relationship is checked. Never infer detector identity from a numeric ontology class ID.

Illustrative record (shortened, not an exhaustive required literal serialization):

```json
{
  "sample_id": "img_00042",
  "image": {"path": "images/img_00042.png", "width": 224, "height": 224},
  "pizza": {"positive_class_ids": ["..."], "unknown_class_ids": []},
  "components": [
    {
      "id": "img_00042/HamTopping/occurrence_001",
      "bbox_xyxy": [27, 44, 58, 77],
      "detector_category": "HamTopping",
      "ontology_positive_classes": ["HamTopping", "MeatTopping", "PizzaTopping", "Food", "DomainThing"],
      "ontology_unknown_classes": []
    }
  ],
  "split": "train"
}
```

The example box and IDs are invented solely to illustrate shape and naming. They are **not** observations from the user's dataset. A compliant implementation can store compact numeric IDs and binary arrays instead; it must supply the manifest and conversion utilities that make those values unambiguous.

## B.2 Model output and gradient-boundary contracts

The assembled model's public result should expose typed, named fields, rather than an undocumented positional tuple:

```text
ModelResult:
    sample_ids: B IDs
    predicted_detections: List[B] of (box_original, category, score, prediction_id)
    detection_losses: {loss_name: scalar_tensor}       # training with GT
    component_points: float32[B,K,d]                    # valid_mask[B,K] required
    component_valid_mask: bool[B,K]
    matched_gt_component_ids: optional IDs for supervised predictions
    global_points: optional float32[B,d]
    composed_points: optional float32[B,d]
    final_pizza_points: float32[B,d]
    component_class_outputs: float32[B,K,C] + known masks
    pizza_class_outputs: float32[B,C] + known masks
    geometry_diagnostics: optional separately requested structured result
```

`component_class_outputs` must distinguish unbounded **neural logits**, real-valued **geometric signed margins**, and exact **Boolean membership**; do not give all three the same ambiguous `scores` name. The fixed `DomainThing` output may be reconstructed at presentation/evaluation time and must not silently add a trainable head row. In Phase 2, detector features are frozen and computed without detector parameter/state updates. In Phase 3, selected predicted box *coordinates* are detached, but ROI feature-map sampling is differentiable with respect to FPN features. Missing detections do not remove pizza-supervision eligibility.

The geometry core remains independently callable without image tensors and is solely responsible for physical corner transforms, exact `contains`, `signed_margin`, reference-distance primitives and optional diagnostics. All supported shapes/membership results must have test fixtures with points inside bodies, in windows, on both boundaries and in overlaps of distinct houses.

## B.3 Loss reducer and accumulation obligation

There must be **one reducer** that accepts per-instance/per-class losses, target vectors, known masks, positive-class weight vectors, and image/component grouping. It must reproduce these semantics independent of model family:

1. Remove unsupported/fixed class outputs from trainable reductions; mask truly unknown targets.
2. For each supervised instance, compute the weight-normalized class loss: `sum(mask * weight * value) / sum(mask * weight)`.
3. Average supervised component instance losses **within their image**. Images with no supervised component instances are absent from the *component* image mean, not entered as zeros. Average valid pizza-instance losses over valid pizzas.
4. Apply configurable task coefficients; detector losses remain individually logged and have their source-defined reductions documented.
5. For gradient accumulation, calculate effective-batch denominators over the **whole accumulation window**. A sequence of independent microbatch means, each scaled by `1/num_microbatches`, is not generally equivalent when supervision counts vary. Tests must include heterogeneous matched component counts.
6. Fail clearly when an objective is undefined across an entire accumulation window; a skipped objective must be an explicitly documented behavior, not a silent fabricated zero scalar.

## B.4 Run configuration and experiment identity

Persist the **resolved** config and its schema version, not only partial CLI overrides. Every exposed setting must have type/range/default/validity rules and display through the dashboard. A run manifest must include dataset and ontology version IDs, class/category manifest hashes, seed and parent-detector/Phase-2 checkpoints, optimizer groups, freeze map, loss and weighting modes, phase-specific monitor definition, selection/filtering thresholds, snapshot policy and environment identity.

Example *illustrative* resolved core choice (unlisted parameters still must be resolved by the full config):

```yaml
experiment:
  purpose: exploratory
  seed: 42
  dataset_id: dataset-4800-example
model:
  family: town
  latent_dim: 16
  pizza_mode: hybrid
  town:
    houses_per_class: 2
    windows_per_house: 1
    window_mode: contained
  loss:
    name: signed_bce
    temperature: 0.1
detection:
  backbone: mobilenet_v3_large_fpn
  score_threshold: 0.25
  nms_iou: 0.5
  max_detections_per_image: 100
training:
  phase: 2
  physical_batch_size: 4
  gradient_accumulation_steps: 4
  precision: float32
```

Names here are suggestions, not a newly approved JSON/YAML schema. Invalid pairs—such as `reference_town_distance` with `box`, or a window mode attached to a pure neural model—must fail validation before execution. Changing any settled configuration after a run starts yields a new immutable experiment identity; a resumed run continues only from a compatible state.

## B.5 Expected acceptance artifacts

Before an expensive run, the implementation must produce accessible artifacts demonstrating: (i) a validated prepared tiny dataset and class/category manifests; (ii) exact-geometry unit and property tests, including window-mode initialization equality and edge boundaries; (iii) reference-box scoring and reference-town source-conformance test cases; (iv) a single-forward detector/ROI gradient trace; (v) an effective-batch reduction parity test; (vi) a CPU smoke-test report; (vii) checkpoint-resume and deterministic snapshot evidence; and (viii) a dashboard opening a saved run without any active training subprocess.

---

# Appendix C. Honest completeness boundaries and required handoff decisions

A long specification can still be incomplete. The following points must remain visible and be addressed **without silently changing the agreed model**:

1. **Reference town loss formula is not completely settled.** The original code has problematic target inversion and reduction details. The agreed contract is a source-faithful *corrected adaptation*, not permission to devise a new distance metric. Before accepting this loss, the agent must supply a branch-by-branch mathematical port and hand-computed cases, then obtain author approval for any material ambiguity. The source method cannot truthfully be called fully specified until that verification exists.
2. **Dataset schema is constrained, but its exact wire representation is not approved.** Appendix B specifies semantic content and stable IDs; the implementer may choose JSON/CSV/Parquet details, but must publish the schema, validate round trips, and demonstrate that it needs no generator reasoning at model runtime.
3. **Geometry parameter transform needs its exact implementation documented.** The original sigmoid proposal was superseded by finite-precision min-width/clearance invariants. We intentionally specified *mathematical validity* rather than final Python formulas; the implemented mapping needs tests at extreme raw values, near boundaries and under float32.
4. **Many hyperparameter values are provisional or intentionally unselected.** The architecture approves configurability and a few baselines; it does **not** settle beta, the positive-weight cap, anchor dimensions, geometry initial-width distributions, augmentation schedules (augmentation initially off), warmup fraction, snapshot period, retention budget or dashboard framework. Implementations must document actual chosen defaults and keep them editable; none is an empirical finding.
5. **Run-to-run fair comparison has limits.** Matched CNN initialization and shared detector data reduce confounds; the linear baseline and town are not parameter-capacity matched. Do not claim they are. Keep common procedures, parameter counts and per-seed outcome distributions visible.
6. **A full OWL/ALC interpretation and supervisor-approved RQ2 metric remain outside the settled first build.** The ontology inventory, hard region operations, exact/sampled relation diagnostics, comparison records and provenance are required; no unsupported assertion of formal relation preservation is allowed.
7. **Visual inspection requires split hygiene.** The dashboard can show train/validation history throughout development but should hide test predictions and test-driven geometry exploration until explicit final evaluation.
8. **No implementation is authorized by this checkpoint alone.** The author requested a human- and agent-readable justification-rich contract, with explicit approval before Luna starts substantive work. This expanded revision is ready for author review, not evidence that training or data correction has already occurred.

### How to review this handoff

Read sections 1–7 for the intent, formal model and train objectives, section 8 for evaluation semantics, sections 9–11 for run/visualization/test obligations, Appendix A to check each original Q-number, Appendix B for public module contracts, and Appendix C to locate remaining decisions instead of letting the agent silently invent them.
