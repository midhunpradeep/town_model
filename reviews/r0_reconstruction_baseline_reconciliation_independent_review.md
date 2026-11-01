# Independent Review — R0 Reconstruction Baseline Reconciliation

**Verdict: REQUEST CHANGES**

**Issue:** [#9 — R0: Reconcile reconstruction baseline and repository authority](https://github.com/midhunpradeep/town_model/issues/9)  
**Exact base SHA:** `3bd7252f2232d86e190f71c4e4a29de40dedb396`  
**Exact reviewed SHA:** `6e03e01d42b35c4bf69846c1cd4b55940075c30f`  
**Implementation branch:** `docs/r0-reconstruction-baseline-reconciliation`  
**Review record:** this file, committed separately from the implementation revision.

The submitted commit is a direct child of the claimed base. Most acceptance criteria and all requested runtime checks pass independently. Two blocking documentation defects remain: the current status contradicts the accepted Phase 3 authorization record about Phase 4, and the changed historical teardown map contains a dead branch link. Do not begin R1, R2, Phase 4, detector, classifier, or scientific training work as part of this remediation.

## 1. Review environment and repository identity

- Isolated review worktree: `/tmp/town-model-r0-review-6e03e01`, detached at the exact submitted SHA. It was clean at setup and remained clean after verification.
- The original `/home/midhun/Projects/ovgu/sem8/thesis/town_model` worktree remained on `main` at the base SHA and clean.
- `git show -s` confirms the reviewed commit's sole parent is exactly `3bd7252f2232d86e190f71c4e4a29de40dedb396`; `git merge-base` is the same base. The submitted remote branch resolves to the reviewed SHA; `origin/main` resolves to the base SHA.
- Python environment: Python 3.12.0; `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python`; PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu, pytest 8.4.2, Streamlit 1.66.1, Pandas 3.1.0, NumPy 2.5.4. CUDA unavailable.

### Full changed-file inventory

The actual `git diff --name-status BASE REVIEWED` contains exactly these eight Markdown files:

1. `AGENTS.md` — modified
2. `README.md` — modified
3. `docs/ACCEPTANCE_INDEX.md` — modified
4. `docs/CURRENT_PROJECT_STATUS.md` — modified
5. `docs/agents/DECISION_AUTHORITY.md` — modified
6. `docs/agents/EXECUTION_PROTOCOL.md` — modified
7. `docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md` — modified
8. `docs/reconstruction/R0_RECONCILIATION_REPORT.md` — added

No production source, tests, data, artifacts, scientific specification, historical review files, or retirement manifest changed. `git diff --check BASE REVIEWED` passes.

## 2. Ancestry and protected-history comparison

The base contains the accepted teardown implementation `19222424f5ad8347deddc6dfb2d7187bd85609c0` and the definitive T5 review/report commit `2b9df576aac2fd95f093bbb3b046813620a822a5` as ancestors. The T5 report identifies `19222424…` as the exact accepted teardown implementation; `2b9df576…` is its distinct review-record commit. The submitted docs preserve that distinction correctly.

There are nine commits after the T5 report commit on the path to the base. The actual intervening diff adds only the four T3/T4 independent review records; it contains no classifier implementation changes. The eight retired implementation modules are absent at the reviewed revision, and their explicit imports each raise `ModuleNotFoundError`:

- `src/town_model/models/classifiers.py`
- `src/town_model/models/geometry.py`
- `src/town_model/models/image_embeddings.py`
- `src/town_model/models/classifier_transport.py`
- `src/town_model/models/image_model_validation.py`
- `src/town_model/training/image_objectives.py`
- `src/town_model/experiments/synthetic.py`
- `src/town_model/dashboard/geometry.py`

`git diff BASE REVIEWED -- reviews` is empty. SHA-256 values match between base and reviewed trees for the Phase 1 final review, Phase 2 final review, Phase 3 definitive review, and T5 final acceptance report. `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` is byte-identical at both revisions (SHA-256 `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`). The Phase 1–3 reviews, teardown reviews, and manifest remain unchanged.

## 3. Documentation and authority findings

### Correct and recoverable boundaries

The new entrypoint/status material prominently identifies accepted Phase 1–3 functionality as historical and the accepted T0–T5 teardown as the current execution boundary. README, current status, and acceptance index distinguish:

- surviving dataset, ontology/known-mask, vision/FPN/ROI, point-only projection/composition, generic reduction, cache/storage, read-only historical inspection, and dataset CLI/dashboard infrastructure;
- removed neural/affine, box, town, membership, geometric-objective, and classifier-dependent execution paths, which are unavailable and must not publish model artifacts;
- the author's supplied handoff reporting 50 active A1 decisions from the still-pending R1 durable ledger reconciliation;
- the expanded v1.1 document as historical review material, not present implementation authority;
- unresolved choices as author-controlled; and
- detector/scientific training as unimplemented and on hold.

The old Phase 1–3 sections in `docs/CURRENT_PROJECT_STATUS.md` are clearly headed as a historical snapshot, and individual implementation/review sections are explicitly historical. The pre-T2 map now begins with a prominent historical-artifact notice directing readers to current status, acceptance index, and T5. Those markings materially reduce the risk of reading old classifier descriptions as executable instructions.

The scientific-authority language does not reconstruct the full A1 ledger, claim that R1 is complete, approve new math, or authorize classifier implementation. R1 (source-linked decision-ledger reconciliation) and the proposed R2 (later, separately scoped interface work grounded in R1) are distinguished; the execution protocol says R2 is not classifier math, objective/optimizer behavior, detector work, or training. The author remains the exclusive scientific decision authority.

### Blocking findings

#### R0-1 — Phase 4 authorization is stated inconsistently

- **Severity:** High — blocking.
- **Affected file/location:** `docs/CURRENT_PROJECT_STATUS.md:21`, contrasted with lines 44 and 80 and `reviews/phase_03_definitive_acceptance_review.md:87`.
- **Observed contradiction:** The new current-status paragraph says, “Phase 4 and detector/scientific training work remain on hold/unimplemented; **no historical Phase 1–3 acceptance authorizes them**.” The preserved historical section in the same file says Phase 4 is “Authorized, **on hold**,” and quotes the definitive Phase 3 review. That review's final authorization states: “Implementation Phase 4 is authorized.” The T5 report says that T5 itself does not approve Phase 4 implementation; it does not rescind the prior Phase 3 authorization.
- **Why this violates R0:** A fresh agent cannot reconcile the present hold with the separately recorded historical authorization. The current status should distinguish authorization from readiness/current hold, rather than denying a source record it preserves. This is precisely the current-versus-historical ambiguity R0 is supposed to remove.
- **Minimum correction:** Reword the current paragraph to acknowledge the Phase 3 review's historical Phase 4 authorization, state unambiguously that current author direction keeps Phase 4 on hold and no Phase 4/detector/scientific training may start now, and distinguish that authorization from acceptance of detector/training work. Do not add scientific decisions.
- **Required verification:** Read the corrected top-level current status alongside the cited Phase 3 review and T5 report; confirm both historical authorization and current hold are stated without contradiction and all eight changed-document links still resolve.

#### R0-2 — Historical teardown map links to a deleted branch

- **Severity:** Medium — blocking for the requested link/reference verification.
- **Affected file/location:** `docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md:11`.
- **Observed behavior:** The document says the accepted historical implementation remains at `https://github.com/midhunpradeep/town_model/tree/remediation/phase-3-last-two-findings`. GitHub's branch API returns 404 and the URL returns HTTP 404. The immutable commit URL for `2de9c5f29b436f727720e87132d0b4be25a40768` returns HTTP 200. The new banner does label the map historical, but it does not make the dead link usable.
- **Why this violates R0:** R0 changes this agent-facing map and explicitly asks that links/references and obsolete branch links be checked. A fresh agent following the claimed historical implementation reference cannot reach it from the document.
- **Minimum correction:** Replace the branch link with an immutable commit permalink (or a valid local acceptance/review link) while retaining the historical context.
- **Required verification:** Check the replacement URL returns successfully and run the local Markdown-target check plus `git diff --check` again.

## 4. GitHub issue reconciliation

Live GitHub state was independently checked:

- Issue #2 remains **OPEN**, unassigned, with `wayfinder:map`. Its provisional comment [6560741869](https://github.com/midhunpradeep/town_model/issues/2#issuecomment-6560741869) describes the accepted teardown and 50-A1 handoff, explicitly leaves R1 reconciliation pending, and does not resolve or close the planning map.
- Issue #3 remains **OPEN**, unassigned, with `wayfinder:grilling`. Its provisional comment [6560742108](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6560742108) summarizes only the supplied F3/F4 subset, explicitly says it is not a final resolution, and leaves exact-source reconciliation to R1. It retains its author-decision identity.
- Issue #9 was **OPEN**, `workflow:in-review` at review start. No issue was closed and no R1/R2 or scientific implementation was started.

The two provisional comments accurately represent the handoff boundary and do not silently approve additional choices. The stale #2/#3 bodies remain for R1 reconciliation as the implementation report states; the comments themselves do not claim final resolution.

## 5. Independent fresh verification

### Regression, collection, compilation, and diff integrity

From the isolated review worktree:

```text
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q
202 passed, 0 failed, 0 skipped; 7 existing Streamlit/Pandas Pandas4Warning warnings; 61.08 s.

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q
202 tests collected; exit 0.

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests
Passed.

git diff --check 3bd7252f2232d86e190f71c4e4a29de40dedb396 6e03e01d42b35c4bf69846c1cd4b55940075c30f
Passed.
```

The full suite independently exercises the existing dataset/dashboard tests, unavailable-backend tests, and teardown architecture/import boundaries.

### Dataset CLI and dashboard

Using the same explicit interpreter and `PYTHONPATH=src`:

- CLI root help, `generate --help`, and `dashboard --help` exited 0.
- `inspect examples/pizza-48 --json` exited 0: `current_pass`, 48 samples, split sizes train/validation/test = 16/16/16.
- `validate examples/pizza-48 --no-write-report --json` exited 0: valid, 0 errors, 178 warnings, 9 checks. The existing validation-report SHA-256 remained `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925` before/after.
- Dashboard tests in the full suite passed. A live dashboard launched from the reviewed worktree returned `/_stcore/health` HTTP 200 (`ok`) and `/` HTTP 200 (6,467 bytes). I deliberately interrupted the server with SIGINT after the checks; its process return code was `-2`, so I make no claim that it exited cleanly with code 0.

### Twelve unavailable classifier-dependent CLI operations

I independently ran each route as an actual `python -m town_model.cli` subprocess from the reviewed worktree. New outputs/config/snapshot paths were isolated under `/tmp`; existing image and synthetic runs contained manifest, checkpoint, snapshot, lock, and stop-request sentinels. For each operation I required exit 2, its operation-specific error, the explicit “intentionally disconnected” and “no reconstructed classifier backend is currently approved” text, no traceback, and equality of the complete sandbox file/directory inventory and file hashes before/after.

| Route | Result |
|---|---|
| `experiment fixed --output <new>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment mapping --output <new>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image preflight --config <missing>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image run --config <missing> --output <new>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image matrix --config <missing> --output <new>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image resume <sentinel-run>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image evaluate <sentinel-run>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image stop <sentinel-run>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `experiment image relations <missing-snapshot> … --output <new>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `evaluate <sentinel-run>` | Exit 2; explicit unavailable error; unchanged sandbox |
| `resume <sentinel-run> --additional-steps 1` | Exit 2; explicit unavailable error; unchanged sandbox |
| `export-geometry <sentinel-run> --output <new>` | Exit 2; explicit unavailable error; unchanged sandbox |

All 12 produced the intended rejection before reading missing config/snapshot paths or creating/publishing outputs. No traceback appeared, and the existing sentinels remained unchanged.

### Active-path imports and links

In a fresh process I imported 11 retained CLI/application/dashboard/inspection/point/vision/reducer paths; none loaded a removed module. Explicit imports of all eight retired module names failed with `ModuleNotFoundError`. An active `src/town_model` import scan found no import references to those removed modules.

I checked 65 local Markdown targets across the changed Markdown files; all resolve. Live GitHub issue-comment links used in the report exist. The historical branch URL in finding R0-2 is the exception and is verified dead (HTTP 404); its immutable commit URL is live (HTTP 200).

## 6. Acceptance-criteria matrix

| Criterion | Verdict | Independent evidence |
|---|---|---|
| **AC1 — Verified repository baseline and ancestry** | **PASS** | Reviewed commit is a direct child of the exact base; base and submitted refs verified; accepted teardown implementation/T5 report are ancestors; base-to-target diff is the actual eight-file list above. |
| **AC2 — Accurate current documentation** | **FAIL** | Current-status line 21 denies historical Phase 4 authorization that its own historical section and definitive Phase 3 review explicitly record. Changed teardown map also contains a dead branch URL. Findings R0-1 and R0-2. |
| **AC3 — Correct scientific-authority boundaries** | **PASS** | 50-A1 handoff is reported as supplied context only; full ledger remains R1 work; expanded draft is not implementation authority; unresolved decisions remain author-controlled; no classifier implementation is authorized. |
| **AC4 — Preserved infrastructure and credible verification** | **PASS** | Full suite 202/202, collection 202, compileall, dataset inspect/no-write validate, dashboard suite/live health and import boundaries independently verified; warnings and CPU-only/live-server termination limits disclosed. |
| **AC5 — Explicit unavailable operations and no artifact mutation** | **PASS** | All 12 subprocess routes returned expected exit 2 and explicit errors without traceback; isolated new paths stayed absent and complete sentinel inventories/hashes remained identical. |
| **AC6 — Strict R0 scope isolation** | **PASS** | Actual diff is eight documentation/report files only; no production, test, dataset, artifact, science spec, review, or retirement-manifest change. |
| **AC7 — Accurate provisional issue reconciliation** | **PASS** | #2/#3 comments accurately remain provisional; both issues remain open with their original Wayfinder labels/identities. #9 was open/in-review at review start. |
| **AC8 — Reproducible fresh-agent handoff** | **FAIL** | R1 vs proposed R2 and the classifier-free capabilities/blocks are described, but the conflicting Phase 4 authorization text prevents a fresh agent from reliably identifying the authority state, and the historical map's claimed implementation link returns 404. |

## 7. Limitations

- Verification was CPU-only; CUDA behavior is not established.
- No dataset generation, classifier/detector training, scientific experiment, or model evaluation was run.
- The live dashboard health/root checks passed, but the reviewer terminated the server by SIGINT (`-2`); no clean-exit claim is made.
- The available harness exposed no sub-agent dispatch interface. No parallel Standards/Spec sub-agent results are claimed; this is the direct independent review against the requested acceptance criteria.

## 8. Final verdict and required next action

**REQUEST CHANGES.** AC1, AC3, AC4, AC5, AC6, and AC7 pass. AC2 and AC8 fail for the concrete findings above. R0 is close to acceptable, but the operational-status contradiction and dead branch reference violate its documentation/recoverability requirements.

Minimum remediation is limited to:

1. Clarify in `docs/CURRENT_PROJECT_STATUS.md` that Phase 4 received historical authorization from the definitive Phase 3 review but is presently held by current author direction; explicitly keep detector/scientific training unimplemented and blocked. Do not infer new authority.
2. Replace the deleted branch URL in `docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md` with a live immutable commit/review link.
3. Rerun the local Markdown-link check and `git diff --check`, inspect the exact small diff, and resubmit a fixed SHA for independent review. Do not alter the reviewed SHA or protected history.

**Reviewed implementation SHA:** `6e03e01d42b35c4bf69846c1cd4b55940075c30f` (preserve as immutable evidence).  
**Issue action:** return #9 from `workflow:in-review` to `workflow:in-progress` after posting this linked verdict. Keep it open; do not close or merge to `main`. No R1/R2 or classifier work begins here.
