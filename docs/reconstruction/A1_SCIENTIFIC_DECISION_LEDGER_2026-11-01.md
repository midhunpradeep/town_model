# A1 Scientific Decision Ledger — 70 Active Approvals

**Thesis:** Representing Ontologies Using Town Embeddings  
**Handoff date:** 2026-11-01  
**Authority:** Author-approved mathematical decisions F0.1–F10.16, not the historical implementation draft  
**Use:** Give this document to the separate agile reconstruction session together with the repository/teardown handoff.

## 0. Status, provenance, and authority

- **70 active scientific approvals:** F0 (1), F1 (1), F2 (8), F3 (11), F4 (4), F5 (6), F6 (5), F7 (8), F8 (5), F9 (5), F10 (16). Exactly one older decision, **F3.6**, is superseded by **F3.6-R** and is not an active approval.
- **Not equivalent to implementation authorization.** Approval of scientific meaning does not authorize arbitrary source changes, skipped review gates, or work on currently blocked scientific modules.
- **Repository:** https://github.com/midhunpradeep/town_model . Latest checked `main`: `b44304a09f387d767f55abe3db1b3f86cbae4c36`. R0 (#9) accepted/closed and its independent review records merged into `main`. Accepted T0–T5 removed the old classifier mathematics from active source; dataset, vision, image-to-point, generic supervised reducers, artifact identity, and historical inspection survive. Reconstruction remains distinct from older historical Phase 1–3 implementations.
- **Current open author decisions:** Issues [#2](https://github.com/midhunpradeep/town_model/issues/2) and [#3](https://github.com/midhunpradeep/town_model/issues/3) remain open; neither is a license to infer that the old model definitions are reapproved. Their GitHub bookkeeping may lag this chat's explicit decisions.
- **Historical scientific draft:** [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](https://github.com/midhunpradeep/town_model/blob/main/pizzaiolo_thesis_implementation_spec_v1_1_expanded.md), blob SHA `e2db0c3f764201ab55f4e2cd7df769f0970d219d`; about 990 lines, Q1–Q191. It is useful architecture and historical rationale, **not** a replacement for the active A1 ledger. Its closed-window exclusion, lower-corner transform, mandatory 4-step accumulation, bounded-only deferral, warmup/cosine default, and uniform geometry decay assumptions are superseded where they conflict with this ledger.
- **Two different sets of phases:** Historical *implementation milestones* Phase 1 = data preparation, Phase 2 = synthetic models, Phase 3 = pre-detector GT-ROI work. Future *scientific training stages* Stage 1 = train detector, Stage 2 = frozen visual detector + GT-ROI ontology learning, Stage 3 = joint predicted-ROI training. Never treat historical milestone Phase 3 as having completed scientific Stage 3.
- **Primary thesis questions:** RQ1 tests predictive quality of towns versus boxes; RQ2 tests whether learned global regions represent ontology relations (containment, exclusion/disjointness, overlap). These are independent: pointwise classification does not prove global region relations.

## 1. Complete authoritative decision ledger

### F0–F1 — Ontology meaning and known supervision (2)

- **F0.1 — Unary class predicates.** Interpret ontology classes as unary predicates/extensions in a declared interpretation. Preserve OWL open-world semantics: absence of assertion/entailment is not automatically false.
- **F1.1 — Scoped closed-world generator targets.** Within the generator's explicitly complete, validated supervised scope, assign known positives and negatives. Otherwise mark labels unknown with masks; do not make unsupported classes negative.

### F2 — Latent representation (8)

- **F2.1 — One observed entity maps to one latent point.** An observation `o` maps to `f(o)∈Δ`; distinct observations of the same underlying entity may have distinct points.
- **F2.2 — Membership-first semantics.** Class membership is the primary operation. No independent metric semantics or required proximity theorem is assumed.
- **F2.3 — One shared semantic domain.** Whole pizzas, component occurrences and class-region parameters use one shared latent coordinate space; distinct upstream projection heads are allowed.
- **F2.4 — Per-occurrence points.** Each observed component occurrence and whole pizza has one final latent point. Repeated components are distinct. Intermediate visual/compositional features may exist.
- **F2.5 — Visual inference, no target leakage.** Infer semantic points primarily from RGB/ROI visual evidence. Generator recipe/ontology labels and detector categories/confidences are not semantic point inputs; permitted ROI/image spatial coordinates derived from pixels are a distinct authorized visual feature.
- **F2.6 — Learned latent axes.** Coordinate axes have no prescribed ontology meaning.
- **F2.7 — Bounded/unbounded controlled comparison.** Support both `Δ=[-1,1]^d` and `Δ=ℝ^d` as scientifically distinct runs; do not disguise a finite bounded range as unbounded.
- **F2.8 — Configurable dimension.** Latent dimension `d` is configurable with controlled and documented dimensional comparisons; historical `d=16` is only a candidate starting value.

### F3 — Geometric class sets and capacities (11)

- **F3.1 — Global class regions.** Trainable class regions are shared across observations; relation satisfaction is not presumed from pointwise labels. Affine baseline class decisions correspond to halfspaces restricted to `Δ`.
- **F3.2 — Exact set membership.** Hard Boolean membership defines geometric predictions; differentiable measurements are not substitutes.
- **F3.3 — Membership-first learning.** No explicit ontology-axiom loss in primary training. Any emerged global structure must be tested separately.
- **F3.4 — Closed axis-aligned boxes.** Use boxes as fundamental geometric primitives.
- **F3.5 — Town construction.** Each house is a closed frame with window interiors excluded; each town is a union of houses, following F3.6-R.
- **F3.6-R — Open window interior, retained boundaries.** `H=F \ ⋃_w int_{ℝ^d}(W_w)` for closed frame `F`; the *ambient strict interior* of each window is excluded. Frame and window boundary points are retained unless excluded by another window's interior. Independent windows outside the frame do not remove out-of-frame points. This **supersedes F3.6**, whose closed-window exclusion is invalid.
- **F3.7 — Contained vs independent windows.** Both physical parameterizations are required with the *same* set semantics; paired tests start from matched physical initial coordinates, not just the same random seed.
- **F3.8 — Empty-set distinctions.** Empty, zero-measure, small, and no-observed-point regions differ. Full exact empty-set trainability is not required for every initial capacity, but diagnostics must be truthful.
- **F3.9 — Fixed per-run capacity.** `H≥1` houses and `W≥0` windows per house, fixed within a run but configurable between runs. No adaptive house/window gates or creation/deletion.
- **F3.10 — Positive trainable primitive widths.** Positive full axis widths; mathematical evaluators may separately test degenerate fixed corner fixtures where useful.
- **F3.11 — No-window equivalence.** `W=0` works; `H=1,W=0` yields exactly one box, without fake window state.

**Core mathematical semantics:** `B(l,r)={x∈Δ:∀j l_j≤x_j≤r_j}`; `H_h=F_h\⋃_w int(W_hw)`; `T=⋃_h H_h`.

### F4 — Classifier input/output (4)

- **F4.1 — Flat point classifier interface.** Consume `[N,d]` latent points and classifier state, independent of image/dataset/ontology source metadata.
- **F4.2 — Linear multilabel baseline.** Each trainable class has `z_c(x)=w_c·x+b_c`; hard prediction is `z_c≥0`.
- **F4.3 — DomainThing is fixed.** One always-positive `DomainThing` relative to the selected learned domain. No trainable parameter; not a claim about all OWL individuals in an unrestricted ontology universe.
- **F4.4 — Separately typed outputs.** Ordered Boolean `contains:[N,C_train]` and family-specific numerical measurements (baseline logits vs geometry signed clearances). Do not claim calibration or conflate scores with set membership.

### F5 — Signed clearance, gradients, responsibilities (6)

- **F5.1 — Box signed coordinate clearance.** `s_B(x)=min_j(x_j-l_j,r_j-x_j)`; explicitly not Euclidean distance.
- **F5.2 — Module separation.** Geometry owns physical sets and measurements; objective owns per-label penalties; reducer owns masking/weighting/aggregation; optimization owns updates; evaluation independently checks predictions/relations.
- **F5.3 — Exact `min`/`max` composition.** `s_H(x)=min(s_F(x),-s_W1(x),…,-s_WW(x))`, `s_T(x)=max_h s_Hh(x)`; for `W=0`, `s_H=s_F`; no soft replacements initially.
- **F5.4 — Representative gradient coverage.** For misclassified positives/negatives, primary objectives must show finite, useful corrective gradients through appropriate point and geometry parameters; not every parameter must be active on every case.
- **F5.5 — Framework AD tie subgradients.** Use documented automatic differentiation at exact `min`/`max` ties, audit tie and branch frequencies; no custom tie-backprop unless separately justified.
- **F5.6 — Raw physical units.** No implicit geometry-width/dimension normalization or undocumented score rescaling. Keep numerical meaning and scaling explicit.

### F6 — Primary and reference objectives (5)

- **F6.1 — Signed logistic and hinge controlled comparison.** Both box and town use the same signed clearance; vary only penalization behavior, with matched geometry/init where appropriate.
- **F6.2 — Fixed temperature and margin.** Logistic `τ>0` and hinge `m≥0` configurable per run, fixed rather than trainable in initial study.
- **F6.3 — Baseline BCE-with-logits.** Ordinary multilabel BCE on affine logits.
- **F6.4 — Two additional reference modes.** Source-inspired ChebiBox BCE for boxes and original-town-inspired distance for towns; supplementary to five-way core, not secretly equivalent to native losses.
- **F6.5 — Source-gated town reference correction.** An independent branch-by-branch source conformance note must reconcile original target inversion/reductions with F3.6-R, provide hand-computable examples and escalate material changes for author review. **Final source-corrected formula is not approved**.

`L_signed_BCE=softplus(-(2y-1)s/τ)`; `L_signed_hinge=max(0,m-(2y-1)s)`; baseline uses `BCEWithLogits(z,y)`.

**Primary five-way comparison:** affine+BCE, box+signed logistic, box+signed hinge, town+signed logistic, town+signed hinge. ChebiBox and town reference objectives are additional named studies.

### F7 — Raw-to-physical parameterization and numerical discipline (8)

- **F7.1 — Valid geometry by construction.** Raw parameters map to valid physical geometry; clipping coordinates after optimizer step is not the defining validity mechanism.
- **F7.2 — Center and positive half-width.** Use `l=c-h,r=c+h`, with width-first construction in bounded domains; historical ordered-low/width transform is not the selected coordinate basis.
- **F7.3 — Validated float32 invariant domain.** Geometry transforms/evaluation/loss in float32 (including when visual AMP is used), with explicit, tested finite numerical operating range rather than impossible all-real guarantees.
- **F7.4 — Boundary approach versus evaluation.** Bounded trainable faces can approach `Δ` boundaries without exactly attaining them; pure evaluator must accept valid exact-touch fixtures.
- **F7.5 — Separate min width and clearance.** Full primitive width floor `ε>0`, contained-window margin `δ>0`, each configured per run; enforce feasibility (e.g. frame full width enough for window + two clearances, and bounded capacity reserve). `W=0` does not inherit unnecessary window clearance conditions.
- **F7.6 — Matched physical meaning across domains.** Bounded and unbounded transforms differ but produce the same conceptual physical `(c,h)` and permit matched physical initial conditions; Jacobian differences are optimizer confounds to disclose.
- **F7.7 — Label-agnostic dimension-aware initialization.** Seeded broad regions, distinct houses, nontrivial window exclusions, matched physical starting configurations across window modes; diagnose initial volume/occupancy (`q^d` sensitivity); no label-informed fit.
- **F7.8 — Reject invalid optimization updates.** Log saturation and raw/physical gradients; stop/reject nonfinite or invalid proposed updates without silent physical-coordinate repair, parameter clipping or optimizer reset. Approved *gradient-norm* clipping under F9.5 is separate. Exact proposed-step transaction mechanics not settled.

Bounded boxes require `|c_j|+h_j≤1` and `2h_j≥ε`; contained windows require `|c_Wj-c_Fj|+h_Wj+δ≤h_Fj` for every coordinate. Concrete transforms/finite-precision reserves remain unresolved.

### F8 — Masked supervised reductions and effective batches (5)

- **F8.1 — Weighted known-class means; entity/image hierarchy.** Normalize each supervised entity's known class penalties by known weight sum. Within each eligible image, average supervised components; then average eligible images. Pizza entity mean separately across eligible images; zero-supervision cases absent, not fake zeros.
- **F8.2 — Unit weights primary; named variants.** Positives and negatives weight 1 in core. `inverse_sqrt` and `effective_number` positive-weight variants are calculated from known-positive TRAIN counts separately for component and pizza levels; known negatives remain 1, unknowns mask out; no surprise imbalance changes.
- **F8.3 — Fixed task coefficients.** Nonnegative, configurable per-stage task multipliers for detector, component and pizza objectives, fixed during run. Diagnose actual per-task losses/gradients; coefficients do not imply equal influence.
- **F8.4 — Undefined enabled tasks.** Omit and log unavailable task loss without renormalizing other coefficients; fail when every enabled objective in the step/window is undefined.
- **F8.5 — Accumulation optional only on measured need.** Physical batches default to `accumulation_steps=1`; introduce *genuinely memory-saving* accumulation only with measured necessity and tested effective-batch gradient parity, preserving denominator structure. Retaining all microbatch graphs until one backward is not a memory-saving implementation.

Entity reduction: `L_i=(Σ_c k_ic w_ic ℓ_ic)/(Σ_c k_ic w_ic)` if denominator positive. Phase-3 task combination: `L=λ_D L_D+λ_C L_C+λ_P L_P` over defined enabled tasks.

### F9 — Training phases and optimization (5)

- **F9.1 — Three scientific stages.** Stage 1 trains detector/backbone/FPN as selected. Stage 2 begins from selected detector, freezes all its visual parameters **and mutable state**, uses GT ROIs, trains points/composition/classifier. Stage 3 starts from corresponding Stage-2 checkpoint, selectively unfreezes, uses genuine selected predicted ROIs, joint detector and ontology losses; ROI feature-map gradients can reach visual features, but ontology gradients do NOT differentiate through detached predicted box coordinates, NMS, matching or discrete selections.
- **F9.2 — Functional optimizer groups.** Unique ownership of each trainable parameter; explicit pretrained backbone, FPN, detector, point projection, composition/fusion, and classifier groups where applicable; auditable groupwise diagnostics and phase-dependent active groups.
- **F9.3 — No raw geometric weight decay initially.** Configurable decay on neural weights, with bias/norm exclusions documented; physical geometry regularization is separate, optional and not disguised as raw AdamW decay.
- **F9.4 — AdamW with constant initial schedule.** Use constant-LR AdamW for early demonstrated learning; warmup/cosine remains selectable/configurable. Numerical LRs and exact phase-transition optimizer/scheduler behavior remain open.
- **F9.5 — Global gradient norm clipping.** Configurable global clipping with per-group pre/post diagnostics; it does not guarantee valid geometry or replace F7.8 numerical update validation. Historical norm `1.0` is a provisional default only.

### F10 — Mathematical adequacy, relations, and auxiliary experiments (16)

- **F10.1 — Global geometry primary for RQ2.** For subclass evaluate `R_A⊆R_B`, for disjointness `R_A∩R_B=∅`, over declared domain `Δ`; reachable-observation and distribution-relative interpretations remain separately identified, not conflated with global proof.
- **F10.2 — Isolate union versus window capacity.** Retain five-way main comparison and add focused capacity study across `H=1/H>1` and `W=0/W>0`; record capacity/parameter differences; no exhaustive Cartesian grid mandatory.
- **F10.3 — Supplementary fixed-point geometry benchmark.** Evaluate classifier regions/losses on common immutable latent points to distinguish representational capacity from learned-encoder adaptation; the main image pathway remains joint learning.
- **F10.4 — Raw-score scale diagnostics.** Keep F5.6 raw signed clearances, matched physical initial-scale controls, point/geometry drift and margin/temperature diagnostics; allow separate scale-sensitivity studies; no hidden score normalization.
- **F10.5 — Topology and finite-box-union theorem.** Finite houses/towns under F3.6-R are (relatively) closed and, for finite physical corners, compact; with freely chosen physical boxes and variable capacity, the abstract family equals finite unions of closed axis-aligned boxes of the admitted kind. Trainable width floors, clearance constraints and fixed capacities further restrict reachability; no general Boolean complement claim.
- **F10.6 — Sound relation certification.** Exact certificate where mathematically sound and tractable; otherwise explicit `unknown` with separately labeled sampled diagnostics. An exact counterexample is a violation witness; unsuccessful search is not proof of satisfaction.
- **F10.7 — Optional compactness bias only.** Primary geometry/classification objective remains unregularized. A separately identified, controlled physical region-size experiment is allowed; exact functional and coefficient require follow-up approval.
- **F10.8 — Optional auxiliary point organization only.** No auxiliary point attraction in primary training. Later separate label-conditioned and label-independent studies may be proposed; no implicit ontology-axiom penalty.
- **F10.9 — Three-state relation reference with provenance.** Reference relation states: ontology-entailed, witnessed violation *in a declared interpretation*, or unknown. Distinguish logical OWL entailment from generator-specific closed-world counterexamples. Never mark all non-entailed statements false.
- **F10.10 — Separate vacuity from formal truth.** True containment/disjointness may be vacuous through empty antecedents or universal consequents. Retain formal result while reporting geometric emptiness, empirical occupancy, support and nontriviality separately.
- **F10.11 — Predeclared RQ2 pair population and multidimensional metrics.** Stratify relation type/direct versus inferred edges, known cases and support; report satisfaction, violation, certification coverage/unknown, and nontriviality separately. No unjustified one-number F1 for unknown references.
- **F10.12 — Two distinct empirical violation distributions.** Evaluate distributional violations on held-out encoder points and on a declared latent geometric reference measure separately; neither substitutes for global proof, and no uniform probability measure exists on all `ℝ^d`.
- **F10.13 — Minimum-enclosure theorem as diagnostic.** For comparable known-positive point sets `P_A⊆P_B`, their coordinatewise minimal bounding boxes nest, including equal-margin expanded enclosures under feasibility. Use fixed-point analytic enclosures as diagnostic benchmark, not a mandatory trainable-class construction.
- **F10.14 — Box sum-of-widths compactness first.** First optional compactness study penalizes `Σ_j(r_j-l_j)` for a box; town compactness functional remains a separately reviewed choice. No change to primary loss.
- **F10.15 — Fixed-point compactness before co-adaptation.** First isolate physical geometry effects with fixed points, then study jointly trained point+region behavior; record geometry movement versus point movement, label accuracy and representation collapse indicators.
- **F10.16 — Weak point organization study sequence.** Optional identity-preserving visual-view consistency experiment before separate known-label-conditioned attraction. Exact augmentation/view, anti-collapse objective and coefficient remain undecided; not part of primary training.

## 2. Mathematical consequences — established facts, not additional approvals

1. **Boundary/score alignment in exact arithmetic:** For F3.6-R, `x∈T` iff `s_T(x)≥0`, including frame/window boundaries. Still implement Boolean `contains` independently to avoid finite-precision misinterpretation. Prior closed-window exclusion would *not* yield the same closed-box-union theorem.
2. **Finite-town decomposition:** Each `F\int(W)` is a union of at most `2d` closed axis-aligned boxes (allowing empty terms). Multiple windows can be distributed to obtain a finite union; a coarse bound is `(2d)^W` per house. This is a representational equivalence with *unbounded capacity*, not a promise of parameter/gradient equivalence.
3. **Minimum-enclosure monotonicity:** If the *same observation population and consistent complete positive labels* yield `P_A⊆P_B`, then `BBox(P_A)⊆BBox(P_B)`; equally expanded coordinate margins preserve inclusion. The result requires hard enclosure and does not automatically follow from finite-weight logistic/hinge training, width floors, or negatively supervised separation. Disjoint positive point sets can still have completely overlapping minimal boxes.
4. **Unbounded logistic scale freedom:** For coordinated positive scaling `k`, `s_{kT}(kx)=k s_T(x)` and membership stays unchanged. On separable samples, correctly signed logistic penalties tend toward zero with increasing `k` where architecture permits scaling; hinge behavior differs. An AdamW schedule cannot remove this mathematical non-identifiability by itself.
5. **Class-family comparison caveat:** A finite nonempty town with finite corners occupies a bounded set even in `Δ=ℝ^d`, whereas an affine baseline halfspace is generally unbounded. Disjointness, complements, and broad class interpretation depend on this domain/geometry mismatch.
6. **Global vs data claims:** Training instance memberships only constrains observed points. Global `R_A⊆R_B` is neither logically required nor empirically established by perfect pointwise multilabel prediction.
7. **Differing sources of regularization:** Raw parameter weight decay, physical box width, occupied town volume, auxiliary point attraction and an explicit ontology-axiom constraint are different mathematical interventions. Keep them distinct and separately named.

## 3. Accepted experiment and infrastructure context (historical, unless separately approved)

- **Retained data:** pinned Pizzaïolo synthetic images; ontology ID/order manifest; class-vs-individual distinctions; known labels/masks; original-pixel component bounding boxes; provenance and sealed split. The R0/T5 review records a 48-image example and 60 trainable class IDs, plus a fixed-positive and unsupported IDs; do not generalize example counts into every dataset.
- **Retained image pathway:** shared MobileNetV3-Large/FPN/ROI infrastructure, `[B,K,260]` component visual feature vectors (256 visual + 4 spatial coordinates), global ROI features `[B,256,7,7]`, `ImagePointModel`, `[B,K,d]` component points, `[B,d]` final pizza point, global/compositional/hybrid DeepSets modes, ragged/empty-component masks. These are retained engineering contracts, not renewed approval of removed classifiers.
- **Future stages:** detector training and selection, frozen GT ROI ontology learning from the selected detector checkpoint, predicted-ROI joint training with one shared backbone forward. Historical proposals include Hungarian max-IoU matching independent of detector class, controlled detector thresholds, and phase-specific validation criteria; revalidate/approve precise integration policy rather than assuming completed implementation.
- **Reproducibility/evaluation:** independent runs, matched initial common neural weights and detector checkpoint per dataset/seed, physically paired window initial conditions, phase lineage, exact class manifest, immutable configuration, known-mask metrics, reproducible storage, validation selection vs sealed test, independent review. Historical proposal suggests ≥3 seeds per comparison-quality configuration; do not mistake this for sufficient statistical proof.
- **Dashboard:** retain and progressively restore run configuration, learning curves, snapshot lineage, image→ROI→point→class trace, raw margins and hard membership, active house/window/face, geometry slices, relation certificate method and sampling diagnostics. Never overlay PCA coordinates onto physical 2D geometry slices as though aligned.
- **Five primary classifier/objective conditions and optional F10 studies must be presented as different experiment tiers.** Do not let exploratory reference/compactness/point-organization studies overwrite the primary five-way claims.

## 4. Open gates — no authorization to invent the answer

**P0: Needed for faithful base reconstruction / numerical validity**

1. Exact raw→center/half-width transform and inverse/initialization for bounded and unbounded domains; contained and independent window feasibility; float32 reserves, epsilon/delta defaults and operating range (F7).
2. Atomic optimizer-step validation and recovery of model **and optimizer state**, including AdamW moments and scheduler, under F7.8; precise failed-step/run behavior.
3. Phase-transition optimizer/scheduler state, unfreezing schedule, group LR/decay numeric baselines, clipping threshold and gradient failure behavior (F9).
4. Source-conformant formula and tests for town reference objective (F6.5); box reference implementation also needs source verification.
5. Exact classifier integration boundaries, point domain behavior when `Δ=ℝ^d`, typed class-order output and empty-batch / `W=0` behavior; do not derive from retired modules.

**P1: Needed before final RQ2 claims / comparison-quality experiments**

6. Formal ontology relation population, type/domain scope, valid pair/negative reference evidence, direct/transitive classifications, relation label source and supervisor-approved final aggregate protocol.
7. Practical certified town-relation algorithm and sound finite-precision certificate assumptions; resources/status for `unknown`; sampled reference measures in bounded/unbounded runs.
8. Final experiment matrix, dimensions/capacities/window mode, point-encoder controls, class weighting variants, detector/ROI matching policy and phase-specific checkpoint monitors; clear primary vs secondary studies.
9. Statistical unit/uncertainty: repeated generator assets/recipes, split independence, validation/test selection, seed variation, appropriate paired comparisons and finite-sample confidence.

**P2: Optional studies, explicitly not primary blockers**

10. Town compactness functional and coefficients, optional box sum-of-widths hyperparameters, post hoc controls and geometry/point drift diagnostics.
11. Point-view consistency augmentation design, anti-collapse/variance safeguards, class-conditioned attraction mechanism and coefficients.
12. Additional reference losses, volume/overlap statistics, broader capacity/scale/dimension ablations.

## 5. Foundational audit — newly identified unresolved scientific questions (NOT approved)

These should become future author-led discussion, NOT silently implemented changes. Priority names are suggested only, not official numbered decisions.

**A. Meaning of the common ontology domain (highest priority).** Is the interpretation's universe the disjoint union of observed pizza and component entities, and are class-region relations evaluated among all entity types or only meaningful typed scopes? For instance, a pizza point and a topping point live in one space but originate from different entity sorts; a class relation involving a pizza class should not be evaluated as though every unsupported component label were false. What makes a class globally empty or universal under `DomainThing` in this restricted universe? Distinguish typed applicability, unknown labels and semantic negation.

**B. What is actually being discovered? (highest priority).** Generator preprocessing expands instance labels using ontology and recipe rules. Consequently, a model can learn a hierarchy from training targets that already carry inherited superclass memberships, even though it receives no explicit axiom loss. Decide whether RQ2 calls this *emergent geometric organization from axiom-derived labels*, rather than independent ontology discovery; prevent test relations or negative labels from being reconstructed via leakage or the same evaluation artifact.

**C. Realizability versus optimization failure (highest priority).** For a fixed latent point set and fixed classifier capacity, can the training label patterns be simultaneously represented at all? Test feasibility separately from failure to optimize or failure to generalize. A town may realize structures unavailable to a one-box model at fixed points, but a sufficiently flexible learned encoder can remap points and conceal that difference. Disjoint point sets need not have disjoint bounding boxes.

**D. Coordinate identifiability and gauge freedom (high).** Common translations, axis permutations, reflections and positive coordinate scalings may leave membership relations unchanged (subject to domain and transform constraints) while changing scores, parameter distributions, regularizers, and training. What quantities are invariant enough to compare across independently learned runs? Region relation truth is invariant under a shared invertible coordinate map, but a point cloud's axis labels and raw parameter magnitudes are not. RQ2 should avoid assuming cross-seed coordinate alignment.

**E. Relation vocabulary (high).** Define mathematically what counts as `overlap` (nonempty intersection vs proper overlap without containment), `exclusion`, disjointness, equivalence, satisfiability and strict/proper subclass. Some are derived from others and some are not explicitly OWL-entailed; report reference provenance and logical consistency. Existing F10.9–F10.12 establish evidence rules, not the final relation vocabulary/pair selection.

**F. Soundness of geometric certificates in float32 (high).** Decide whether an `exact` certificate means exact real arithmetic applied to *the stored float32 values treated as exact dyadics*, validated rational interval arithmetic, an independently verified solver, or another sound scheme. Boundary touching under F3.6-R makes naive tolerance heuristics unsafe. A numerical optimizer's intended real transform need not match rounded stored corners.

**G. Geometry capacity versus class topology (medium-high).** Finite towns are finite closed-box unions, not arbitrary complements. With fixed `H,W` and positive widths, which empty, disconnected, annular and disjoint class patterns are realizable? A family containing mathematically closed regions does not imply a given parameterization can reach all of them. Distinguish *set-family expressivity* from *trainable parameter reachability*.

**H. Identifiability of class-scale/margin objectives (medium-high).** Raw clearance objectives are not invariant under common scaling; learned points and boxes co-adapt; comparison across `d`, bounded/unbounded `Δ`, temperature and width-floor settings needs clear physical scale policy. Record hard-metric versus loss-value differences; decide if extra scale-controlled benchmarks are needed.

**I. Generalization and statistical units (medium-high).** Decide independence and representativeness of synthetic image splits (shared source assets/recipes), epistemic versus sampling uncertainty of relation checks, and how to compare across seeds. The metric denominator must be specified in advance, not optimized after looking at validation/test relation outcomes.

**J. Role of visual composition in RQ2 (medium).** If pizza classes are inferred from components using generator logic but no roles are represented, is successful global pizza-region structure evidence about ontology relations or just memorization of visually compositional recipes? Distinguish supervised unary predicates from full ontology role/existential semantics; do not claim OWL/ALC reasoning.

## 6. Instructions to agile implementation session

1. Store this as the **author-scientific A1 ledger** alongside the accepted teardown handoff; preserve exact decision IDs and marked supersessions.
2. Update issue #2/#3 author reference only with approval-provenance appropriate to the user's direction; issue comments are not scientific replacements for this ledger. R0 is accepted and on `main`; inspect current HEAD again at task start.
3. Build narrowly scoped tasks from approved contracts with preconditions, module boundaries, no-go list, deterministic mathematical oracles, and independent review. Pure physical-corner Boolean membership and score modules can proceed without committing to unapproved raw-parameter transforms.
4. Do not reintroduce retired classifier implementations, silently change window boundary semantics, or infer training objectives from old source. Keep retained data/point/vision/reducer modules intact where verified.
5. Classify every upcoming task as `ready`, `blocked by mathematical decision`, `blocked by prior implementation`, `in independent review` or `accepted`. Never self-accept.
6. For undecided foundational questions above, send a focused mathematical decision request to this discussion. Do not choose a scientific interpretation to accelerate coding.
7. Keep the five-way primary comparison unchanged. F10 optional benchmarks, compactness biases and point-organization objectives remain independent, separately named experiments, not prerequisites for implementing primary classifiers.
8. Reconcile every future mathematical decision by ID and record affected implementation/tests. Scientific approval, implementation completion and independent review are distinct statuses.

**Next mathematical discussion recommendation:** Resolve (A) semantic universe/typing, (B) the provenance of apparent emergent ontology relationships, and (C) fixed-point realizability/learnability before returning to optimizer fine detail. F10 foundational approvals continue to govern relation reporting; final supervisor-dependent RQ2 protocol remains open.
