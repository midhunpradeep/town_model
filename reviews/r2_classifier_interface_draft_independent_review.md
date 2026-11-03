# Independent Review — R2 Classifier Interface Draft

## Verdict

**REQUEST DRAFT CORRECTIONS**

The draft is prominently unapproved, accurately separates approved scientific meaning from retained engineering and proposals, and is mostly technically coherent. Three corrections are needed before it is a complete author-review handoff: D1 lacks an enforceable checkpoint-to-class-column binding invariant; D2 does not elevate all unresolved empty/dimension choices into the author-decision summary; and the future fail-closed regression checklist omits the retained geometry-export route explicitly.

This verdict is a documentation review only. It does not approve D1–D5, scientific decisions, implementation, or route reconnection, and does not resolve issue #3.

## Review provenance and exact target

- Repository: `midhunpradeep/town_model`.
- Draft: `docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md`.
- Exact reviewed draft commit: `c48b4c7f275ea70d92462e7e20cda008039640aa`.
- Required starting base: `7ebc2b62c175288b0673524fbfae9f9d6f15a4ff`.
- The draft commit's direct parent is the stated base; the base is an ancestor of the draft.
- `origin/docs/r2-classifier-interface-contract` resolved to the exact reviewed commit during review. The review worktree/branch was created directly at that commit; this report is on a separate descendant branch.
- Reviewer: fresh Pi session `01a199f3-962c-713b-81eb-b5df19f10a47`; not the draft author and did not author the draft commit. The authenticated repository account is shared with the author, so independence is at the separate agent/session level, not separate human credentials. The harness did not provide parallel subagent dispatch; no two-subagent review is claimed.
- Environment: Python 3.12.0; PyTorch 2.5.1+cpu; CPU-only.

## Authority and lifecycle verification

The accepted repository frontier is three distinct, separately versioned scientific sources: original A1 (70 active F0–F10 decisions), F11 (12 author-reported additions), and F12 (12 author-reported additions), for 94 source-level approvals. F12.1–F12.8 are a grouped block without per-ID formula mapping. The reviewed draft is not a fourth scientific source.

The reviewed commit descends from the independently integrated R1 revision `100d04d4ecae4f58f525bbdac76b87bf28026d5c`, F11 revision `a050234ae490d650e28391f180670d1c38183192`, and F12 revision `5686d60bdb3e309924d37eb600aa75fb93855340`. Live issue state showed #10, #12, and #13 CLOSED with `workflow:accepted`, each for its separate documentation scope. Those acceptances do not prove the mathematics or authorize implementation.

Issue #14 was OPEN with `wayfinder:task`; #2 remained OPEN with `wayfinder:map`; #3 remained OPEN with `wayfinder:grilling`, pending explicit author confirmation. No issue state or decision was changed during this review.

## Changed-file and source-link verification

The exact base-to-draft diff is one file only:

```text
A docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md
```

The change is 319 added lines and no deletions. No A1/F11/F12 source, readiness/current-status/acceptance authority, production code, tests, dataset, artifact, checkpoint, or historical review record changed. `git diff --check <base>..<draft>` passed.

An independent parser reproduced the draft's reported Markdown-link check: **90 links checked, 0 missing or out-of-range local/GitHub-blob targets or line anchors**. The relevant GitHub issues and accepted revision states were also checked live.

## Source-fidelity assessment

The draft faithfully distinguishes:

- F4.1's flat `[N,d]` mathematical classifier boundary from image/dataset/ontology inputs;
- F4.2 affine logits and `z ≥ 0` membership;
- F4.3's fixed, parameter-free `DomainThing` from trainable columns;
- F4.4's ordered Boolean outputs from family-specific numerical measurements;
- F3.6-R's closed frame, excluded ambient open window interiors, retained boundaries, and common contained/independent-window set semantics;
- F5.1/F5.3 signed coordinate clearance and exact min/max composition from Boolean membership and objective/reducer ownership;
- F6's five unregularized primary conditions and F8's known-mask, weighting, hierarchical aggregation, and undefined-task rules;
- F11.1's shared-domain predicates and prohibition on automatic cross-type negatives; and
- F12 trainable-geometry/numerical approvals from the distinct, unresolved image-point-head behavior.

No historical classifier implementation is used as scientific authority. The 94-approval count is attributed to the three author-reported source documents, not to the R2 draft. The document keeps #3 open and marks itself **DRAFT — NOT APPROVED SPECIFICATION**.

## Retained engineering and test verification

Inspected the exact target's retained `contracts.py`, `image_points.py`, `losses.py`, `training/supervision.py`, `dataset/image_data.py`, `dataset/ontology.py`, `vision/features.py`, `models/__init__.py`, `classifier_backend.py`, and relevant dataset, point-model, reducer, and unavailable-backend tests.

Findings supporting the draft's inventory:

- `PreparedDataset.canonical_class_ids` is loaded from the canonical trainable manifest order; fixed-positive and unsupported IDs are separate, and `targets_in_trainable_order()` projects by those IDs. The class manifest hash is retained. `ImageBatch` carries target/mask columns but not a class-ID tuple.
- `ImageBatch.validate()` requires nonempty `B`, finite images/targets, matching target/mask dimensions, Boolean masks, and valid ragged component contracts. `collate_image_records()` independently rejects an empty record sequence. `K=0` remains supported for nonempty image batches.
- All point-producing terminal paths in retained `ImagePointModel` are `tanh`: component, global, DeepSets `rho`, and hybrid fusion. The point result carries identities/masks/points, not classifier outputs. Tests cover bounded ranges, identities, masks, composition and `K=0`.
- `reduce_supervision()` and `reduce_weighted_instances()` consume caller values; they do not turn logits/clearances into penalties. `aggregate_effective_batch()` reduces caller-supplied per-image values, preserves separate component/pizza aggregation, does not renormalize task coefficients, and raises when no contribution is eligible. These helpers do not alone establish end-to-end F8 conformance.
- All eight historical classifier/geometry/objective modules named by the draft are absent at the target. The package exports no retired classifier symbols. The central unavailable-backend exception and current regression tests remain present.

Focused executable verification, from the draft worktree:

```text
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
  tests/test_phase3_models.py tests/test_phase3_dataset.py \
  tests/test_classifier_backend_unavailable.py \
  tests/test_classifier_teardown_boundary.py
29 passed, 0 failed, 2 existing Pandas4Warning warnings (13.34s)
```

Independent probes, treated only as observations and not public contracts:

- A direct `ImagePointModel` call with `B=0` succeeded and emitted empty point outputs; there is no test establishing it as a supported interface. `ImageBatch.validate()` rejects `B=0`, and `collate_image_records()` rejects no records.
- Both generic reducers accepted `[N,0]` values and reported no eligible instances. This is not classifier or training approval; the F8 task remains undefined without supervision.
- `ImagePointModel` rejects `d=0`. That retained image-model behavior does not by itself decide a future pure evaluator's dimension policy.

No full test suite was run; this was documentation-only review, with focused tests/probes used to check contested factual claims.

## D1–D5 assessment

| Decision | Assessment |
|---|---|
| **D1 — class order / `DomainThing`** | Correctly identifies dataset-owned canonical trainable order as an existing fact, keeps fixed/unsupported classes out of trainable columns, and marks fixed-output representation as unresolved. **Blocking gap:** the recommended class-count-only kernel plus output adapter does not yet require that the parameter-row order be persisted and checked against training/inference/checkpoint identity. See Finding 1. |
| **D2 — empty axes / validity** | Correctly separates flat evaluator axes from `ImageBatch B=0`, distinguishes out-of-domain adapter rejection from bounded-set membership, and keeps no-supervision undefined under F8. **Blocking gap:** `N=0` is recommended but not consistently listed as an explicit pending author choice; `d=0` is given a proposed answer without an alternative/consequence or explicit approval request. See Finding 2. |
| **D3 — bounded/unbounded image points** | Technically coherent as a proposal: preserve the bounded `tanh` path; an identity terminal mapping can produce genuine unbounded outputs without clipping/scaling; include activation/domain in run/checkpoint identity; distinguish shape-compatible weights from semantically compatible checkpoints; and retain deferral as a viable alternative. This is not approved architecture, and no point-path change is authorized. |
| **D4 — image entity adapter** | Correctly selects only valid component occurrences, retains repeated occurrence identity and explicit row mapping, uses selected final pizza points, excludes padding, preserves differentiable selection, and keeps labels/categories/confidences out of point values. Combined versus separate calls can legitimately remain a later adapter-engineering decision. |
| **D5 — objective/reducer ownership** | Correctly describes the retained reducers as caller-value reducers, not classifiers/objectives, and does not claim full F8 conformity. The proposed future adapter covers masking, trainable projection, explicit positive weighting, per-entity/per-image/pizza aggregation, undefined tasks and fixed coefficients without hidden renormalization. |

### Finding 1 — D1 does not require reproducible checkpoint-to-column binding (**blocking**)

**Location:** Draft §§4.1 and 11 D1 (especially lines 100, 228–232), with the class-axis test in §13.

**Evidence:** F4.4 requires ordered `contains:[N,C_train]`; the retained dataset owns `canonical_class_ids` and a class-manifest hash, and target projection uses that order. The draft recommends that the pure kernel carry only `C_train` while an adapter attaches ordered IDs. It notes that a different mapping silently changes parameter meaning, but does not require the mapping used to train the parameter rows to be stored with the checkpoint/run or validated on restore/inference.

**Consequence:** A same-sized checkpoint trained with order A can be loaded under an adapter bound to order B. The adapter can attach B's IDs to unchanged A-ordered rows, so output metadata alone cannot detect the mismatch. A test comparing the current adapter's IDs to the current dataset manifest does not establish what order the checkpoint's rows learned.

**Minimum correction:** Keep the class-count-only mathematical kernel as an option, but make an explicit proposed D1 invariant require the exact ordered trainable-ID binding (and the relevant class-manifest/ontology identity) to travel with the bound model/run/checkpoint. Training, inference, and checkpoint restoration must validate exact binding equality and reject mismatch; they must not silently relabel/reorder rows. State this as a proposal requiring author review, not as an already-approved implementation detail, and include restoration/mismatch tests.

### Finding 2 — D2's author-decision summary omits unresolved cardinality choices (**blocking**)

**Location:** Draft §5 lines 126–135; §11 D2 lines 237–242; §14 D2 line 311.

**Evidence:** The source decisions do not specify pure classifier/evaluator behavior for `N=0`, `C_train=0`, or `d=0`. The draft labels `N=0` and `C_train=0` handling Proposed, and D2's exact-boundary list names all three axes, but D2's status names only `C_train=0`, out-of-domain queries, and affine dtype as unresolved. Its §14 summary asks the author to approve zero-class behavior and validity, while only recommending `N=0`; `d=0` is stated as invalid without an explicit author question or alternatives. Existing `ImagePointModel` rejecting `d=0` is not authority for a new pure evaluator.

**Consequence:** The author-facing register can be read as treating `N=0` and `d=0` as settled recommendations rather than open contract choices, despite those policies being outside the accepted source. That is inconsistent with the task's required decision completeness.

**Minimum correction:** Make the D2 status and §14 approval request explicitly enumerate the `N=0`, `C_train=0`, and `d=0` policies, as well as the bounded out-of-domain adapter/query distinction and proposed invalid-input rejection. Give each unresolved axis viable alternatives and consequences. Preserve the separate, existing `ImageBatch B=0` restriction and F8 no-supervision rule; do not decide any of these on the author's behalf.

### Finding 3 — future fail-closed test list omits geometry export (**blocking**)

**Location:** Draft §13 fail-closed row (line 304), read with §10's proposed verification rule.

**Evidence:** The current T5 boundary explicitly disables synthetic geometry export. The T5 report's unavailable-operation matrix names `export-geometry`, and `tests/test_classifier_backend_unavailable.py` exercises it at line 188. The draft's proposed future regression list names training, evaluation, resume, checkpoint, interpretation and relation operations, but not geometry export.

**Consequence:** A future pure evaluator could be added while a geometry-export route is accidentally reconnected, without violating the draft's enumerated check list. That would weaken the explicit T5 hold boundary.

**Minimum correction:** Add geometry export (including no artifact/file side effect) to the future fail-closed route regression list, or explicitly require the complete retained T5 unavailable-route matrix rather than the abbreviated list.

## Domain/head discrepancy and pure-evaluator independence

The domain/head analysis is accurate. F2.7 approves bounded and unbounded domains as distinct scientific conditions; F12 describes geometry parameterization, not image-point terminal heads. The retained encoder's terminal `tanh` layers mean it currently cannot emit a genuine unbounded image point. The proposed identity-head option is technically coherent and avoids an artificial finite domain. The draft accurately records semantic checkpoint incompatibility despite tensor-shape compatibility, preserves the bounded behavior, and offers deferral of unbounded image integration. No identity-head architecture is approved here.

The first three candidate slices in §12 are genuinely independent of D1–D3, trainable geometry parameterization, optimizer transactions, image-encoder changes, class-manifest binding, detector work, and dataset/training integration. Their physical corner/point inputs, domain/numeric preconditions, and deterministic tests are scoped. The later flat typed classifier, image adapter, reducer adapter and training routes are not declared ready while decisions remain open.

## Fail-closed and module-boundary assessment

The draft correctly says a direct-importable pure evaluator does not authorize CLI/API/dashboard/worker execution, checkpoint reconstruction, resume, interpretation, snapshots or relation queries. It requires no-side-effect checks and separate approval/review before reconnection. The geometry-export omission is the specific correction in Finding 3; current code/tests remain fail-closed.

Optional clarification (not a verdict blocker): §§2/12 cite F5.5 but do not spell out its approved automatic-differentiation tie-subgradient and tie/branch-frequency audit. Keep that outside the initial non-trainable evaluator if gradients are not in scope, but explicitly carry it into the later differentiable-clearance/objective acceptance contract. This preserves sequencing without making pure evaluator work wait on optimization.

## Final limitations and handoff

This report reviews the exact documentation commit and the contested retained-source behavior only. It does not validate mathematical proofs, implement or approve any classifier interface, settle any D1–D5 alternative, or establish training/scientific performance. Issue #14 should remain OPEN for draft correction and explicit author review; issue #3 remains OPEN pending explicit author confirmation. Do not merge the draft into `main`, change scientific authority, or begin implementation based on this review.
