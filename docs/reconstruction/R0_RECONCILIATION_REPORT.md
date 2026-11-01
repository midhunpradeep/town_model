# R0 — Reconstruction Baseline Reconciliation

**Status:** Submitted for independent review; not accepted.

**Task:** [Issue #9](https://github.com/midhunpradeep/town_model/issues/9), child of planning initiative [#2](https://github.com/midhunpradeep/town_model/issues/2).

**Repository:** `midhunpradeep/town_model`

**Branch/worktree:** `docs/r0-reconstruction-baseline-reconciliation` / `/tmp/town-model-r0`

**Starting SHA:** `3bd7252f2232d86e190f71c4e4a29de40dedb396`

**Final review-target SHA:** Recorded explicitly in the final verification/review-submission comment on issue #9. The report is part of that immutable revision; do not infer or substitute a SHA from the historical baseline.

## Verified repository baseline

At claim, the worktree on `main` was clean. `git fetch origin main` left `origin/main` at the exact expected starting SHA `3bd7252f2232d86e190f71c4e4a29de40dedb396`; no intervening changes needed reconciliation. The accepted teardown implementation `19222424f5ad8347deddc6dfb2d7187bd85609c0` and definitive T5 review/report commit `2b9df576aac2fd95f093bbb3b046813620a822a5` are ancestors of starting `main`. The nine commits above the T5 report preserve T2–T4 review reports only; no classifier implementation changes intervene. All eight retired classifier modules named in the T5 report are absent from the starting tree.

Phase 1 dataset preparation, Phase 2 synthetic classifiers, and Phase 3 pre-detector GT-ROI embeddings remain accepted historical milestones for their original scopes. The later accepted T0–T5 teardown is the current execution boundary: classifier implementations, geometric membership algorithms, classifier-specific objectives, and model-dependent execution paths have been removed or made unavailable. Historical acceptance reviews, teardown reports, and `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` are preserved unchanged.

## Current retained boundaries

Retained classifier-independent infrastructure includes dataset preparation/validation and ontology targets; explicit known/unknown masks and stable class ordering; canonical preprocessing, MobileNetV3-Large/FPN features, ground-truth ROIs and component identity; `VisionFeatureBatch`; point-only `ImagePointModel`/`ImagePointResult` and global, compositional, and hybrid pathways; generic known-mask reducers; visual-feature caching, run identity/storage; validated read-only historical metadata/snapshot inspection; and dataset CLI/dashboard surfaces.

Protected tensor contracts remain component visual features `[B,K,260]`, global spatial features `[B,256,7,7]`, component latent points `[B,K,d]`, and final pizza points `[B,d]`. No R0 production model, vision, dataset, training, inference, or tensor/ontology contract changes were made.

The historical affine/neural, box, and town classifiers and their executable mathematics are unavailable. Classifier-dependent CLI/API/dashboard operations fail explicitly with `ClassifierBackendUnavailableError`; no replacement backend exists. The twelve CLI routes tested below do not publish or mutate model artifacts.

## Scientific authority and task boundary

The author-approved mathematical handoff supplied for R0 reports **50 active A1 decisions**. R0 records this count and the process boundary only; it does not reconstruct, enumerate, or independently declare the full ledger authoritative. R1 must durably reconcile exact approved definitions and sources, superseded choices, and unresolved details. The expanded v1.1 specification remains a historical review draft, not current reconstruction authority. Unresolved scientific decisions remain author-controlled; a summary comment does not create new approval.

R2 is a separate proposed interface-development task after R1 reconciliation and coordinator scoping. Its boundary should be limited to classifier-facing interface work grounded in R1's exact approved F4 decisions, preserving the existing point-only and tensor/ontology contracts. R2 is not classifier mathematics, geometry/membership, objectives, optimizer behavior, detector work, or scientific training. No R1 or R2 work began here.

## Documentation and workflow changes

| File | Reconciliation |
|---|---|
| `docs/CURRENT_PROJECT_STATUS.md` | Added the current T0–T5 classifier-free baseline, exact verified starting `main` and teardown ancestry, retained tensor contracts, A1/R1 authority boundary, and separate R2/hold status. Clearly labels the 2026-10-31 Phase 1–3 text below as a historical snapshot instead of rewriting it. |
| `docs/ACCEPTANCE_INDEX.md` | Preserved Phase 1–3 accepted implementation/review pairs as historical evidence; added the accepted T0–T5 teardown and current unavailable/retained boundaries; corrected stale branch/authority statements. |
| `README.md` | Removed the false “T2 awaiting review” status and clarified that historical model code is removed, T5 is accepted, retained paths are classifier-free, and detector/scientific training remains unimplemented. |
| `AGENTS.md` | Directs agents to the definitive T5 review and distinguishes the supplied 50-decision handoff from pending R1 reconciliation and unresolved author-owned decisions. |
| `docs/agents/DECISION_AUTHORITY.md` | Replaced the expanded draft as current authority with explicit Phase 1–3/T0–T5 and A1/R1 boundaries; kept author authority and non-inference rules. |
| `docs/agents/EXECUTION_PROTOCOL.md` | Added reconstruction-specific R1 sequencing and a planning-only R2 interface boundary; neither task is created, readied, or implemented by this note. |
| `docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md` | Added a top-level historical-artifact notice; pre-T2 architecture and tracker statements are preserved as historical evidence. |
| `docs/reconstruction/R0_RECONCILIATION_REPORT.md` | This reproducible baseline, verification, tracker, acceptance mapping, limitations, and next-task handoff. |

No historical acceptance review/report or retirement-manifest content was edited.

## Tracker reconciliation

- [Issue #2 status comment](https://github.com/midhunpradeep/town_model/issues/2#issuecomment-6560741869): states T0–T5 acceptance/current `main`, the 50-decision handoff, R0 engineering baseline, R1 ledger reconciliation, atomic-task organization, and author authority over unresolved decisions. Issue #2 remains open with `wayfinder:map`.
- [Issue #3 status comment](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6560742108): records only the supplied F3.2, F3.5, F3.6-R, F3.11, and F4.1–F4.4 handoff summary; explicitly says dependencies may remain unsettled and R1 must reconcile exact sources. Issue #3 remains open with `wayfinder:grilling` and retains its author-decision identity.
- Issue #9 is the scoped R0 implementation task associated as a sub-issue of #2. It remains open and will be moved to `workflow:in-review`, not accepted or closed, after the fixed SHA is pushed and the final review handoff is posted.

## Fresh verification

### Environment

All Python checks used `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python` from the original repository because the isolated worktree has no `.venv`. Versions: Python 3.12.0; PyTorch 2.5.1+cpu; Torchvision 0.20.1+cpu; pytest 8.4.2; Streamlit 1.66.1; Pandas 3.1.0; NumPy 2.5.4; GitHub CLI 2.55.0. CUDA was unavailable (`torch.cuda.is_available() == False`).

An unqualified `python` helper invocation hit an unconfigured pyenv shim (`pyenv: python: command not found`); commands were rerun with the explicit project interpreter. This did not affect repository CLI results. No dataset was generated and no scientific experiment was run.

### Full regression and preserved-boundary tests

| Check | Exact command | Result |
|---|---|---|
| Starting-revision baseline suite | `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` | **202 passed, 0 failed, 0 skipped**; 7 non-failing Streamlit/Pandas `Pandas4Warning` warnings; 61.44 s. |
| Post-documentation suite | Same command on the R0 worktree | **202 passed, 0 failed, 0 skipped**; same 7 warnings; 54.18 s. |
| Final post-report suite | Same command after the R0 report was added | **202 passed, 0 failed, 0 skipped**; same 7 warnings; 48.52 s. |
| Collection | `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q` | **202 collected**, exit 0 at baseline, after documentation edits, and after report creation. |
| Compilation | `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` | Passed at baseline and after report creation. |
| Teardown/import/unavailable focused tests | `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q tests/test_classifier_backend_unavailable.py tests/test_classifier_teardown_boundary.py tests/test_phase3_architecture.py tests/test_t3_architecture.py` | **22 passed**, 2 non-failing `Pandas4Warning` warnings; 19.70 s. |
| Dataset/dashboard page tests | `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q tests/test_dashboard_app.py` | **5 passed**, 5 non-failing `Pandas4Warning` warnings; 5.60 s. |
| Diff whitespace | `git diff --check` | Passed before and after documentation edits and after report creation. |

The full suite covers retained dataset/ontology/supervision, vision/ROI, point, cache/storage, snapshot, CLI/dashboard, unavailable-operation, and teardown boundaries. The explicit focused suite also checks removed module imports and package exports.

### Supported dataset CLI and dashboard

Executed from the isolated R0 worktree with `PYTHONPATH=src` and the explicit interpreter:

- `-m town_model.cli --help`, `generate --help`, and `dashboard --help` — exit 0. Dataset generation was not run (R0 does not generate datasets).
- `-m town_model.cli inspect examples/pizza-48 --json` — exit 0; dataset `pizza-dcc148789aecf8af86a3`, `current_pass`, 48 samples, train/validation/test each 16.
- `-m town_model.cli validate examples/pizza-48 --no-write-report --json` — exit 0; valid, 0 errors, 178 support warnings, 9 checks. `examples/pizza-48/reports/validation.json` SHA-256 remained `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925` before/after.
- Dashboard AppTests passed all 5 existing relevant page tests (above). A live `town dashboard --address 127.0.0.1 --port 59051 --headless` server returned `/_stcore/health` HTTP 200 body `ok` and `/` HTTP 200; the CLI/server process exited 0 after shutdown.

### Twelve unavailable classifier CLI routes and artifact safety

Each route was invoked as a subprocess using `python -m town_model.cli …` in an isolated temporary sandbox. Every command exited **2**, printed its operation-specific unavailable error with “intentionally disconnected” / “no reconstructed classifier backend is currently approved”, and emitted no traceback:

1. `experiment fixed --output <temporary-output>`
2. `experiment mapping --output <temporary-output>`
3. `experiment image preflight --config <missing-config>`
4. `experiment image run --config <missing-config> --output <temporary-output>`
5. `experiment image matrix --config <missing-config> --output <temporary-output>`
6. `experiment image resume <existing-sentinel-run>`
7. `experiment image evaluate <existing-sentinel-run>`
8. `experiment image stop <existing-sentinel-run>`
9. `experiment image relations <missing-snapshot> --source-class-id 1 --target-class-id 2 --relation containment --output <temporary-result>`
10. `evaluate <existing-sentinel-run>`
11. `resume <existing-sentinel-run> --additional-steps 1`
12. `export-geometry <existing-sentinel-run> --output <temporary-result>`

The temporary tree inventory and hashes were identical after each rejection. The nonexistent output/config/snapshot paths remained absent; an existing run's manifest, checkpoint sentinel, snapshot, lock, and stop-request sentinels were unchanged. Direct API unavailable-boundary and nonmutation checks also passed in the focused tests. No model output was published or changed.

### Protected history and documentation

The eight removed classifier modules are absent from the verified starting tree. The teardown/import architecture tests pass; no production source or tests were modified. The accepted T5 review, historical review documents, and `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` are unchanged in the R0 diff. Local Markdown links in all changed Markdown files were checked after report creation; all resolve. Final changed paths and `git diff --check` are recorded with the submitted revision.

## Acceptance-criteria evidence

| Criterion | Evidence |
|---|---|
| **AC1 — Repository identity** | Verified `main`/`origin/main` `3bd7252f2232d86e190f71c4e4a29de40dedb396`; accepted teardown `19222424…` and T5 report commit `2b9df576…` are ancestors; no material intervening code change. |
| **AC2 — Documentation accuracy** | Status/acceptance index, README, AGENTS, decision authority, execution protocol, and the pre-teardown map now distinguish historical classifier acceptance from accepted current teardown/unavailable behavior. |
| **AC3 — Scientific authority** | Records the author-supplied 50-active-A1 handoff, pending R1 durable ledger, historical expanded draft, and unresolved author-controlled details without reconstructing the ledger. #2/#3 comments are explicitly provisional. |
| **AC4 — Preserved infrastructure** | Full suite and collection, compilation, CLI inspect/no-write validation, 5 dashboard tests/live health, and retained-boundary focused suite pass with warnings disclosed. |
| **AC5 — Legacy-operation safety** | All 12 historical CLI routes reject with exit 2 and explicit errors; temporary new-output paths remain absent and existing sentinel artifacts are byte/hash unchanged. |
| **AC6 — Scope isolation** | Only approved documentation/workflow/report files changed. No production science, model/tensor/ontology code, tests, data, or protected historical acceptance/manifest evidence changed. |
| **AC7 — Issue consistency** | Provisional comments linked above; #2 stays open `wayfinder:map`, #3 stays open `wayfinder:grilling`; neither is converted into an implementation ticket or closed. |
| **AC8 — Fresh-agent recoverability** | Current status, acceptance index, agent entrypoints, report, issue journal, exact baseline, tested/unavailable capabilities, R1 authority boundary, R2 proposal, and unresolved limitations are linked and documented. |

## Remaining inconsistencies, limits, and next work

- The historical remainder of `CURRENT_PROJECT_STATUS.md`, original Phase 1–3 reports, and the preserved pre-teardown map contain statements that were true in their historical context. They are labeled as historical rather than rewritten. R1 should reconcile the remaining issue #2/#3 body text with the supplied A1 handoff; the R0 comments do not finalize that record.
- The R0 handoff supplied the count and a bounded subset of A1 topics, not a repository-ready complete ledger with every exact source, supersession, and open dependency. R1 must establish that ledger from author-approved sources and request author input where approval is absent or ambiguous. R0 does not claim all dependencies are settled.
- CPU-only verification is recorded; CUDA behavior is not established. No detector, scientific training, data generation, or new scientific experiment was run. Seven full-suite Streamlit/Pandas deprecation warnings remain non-failing. Bare `python` is not configured in this environment; the explicit project interpreter is available.

**Next task R1:** reconcile the exact author-approved A1 decision ledger and links as a documentation/authority task; preserve unresolved decisions as author-owned and do not implement classifiers. **Proposed R2 boundary:** after R1, create a separate reviewed task for approved classifier-facing interface development only, preserving point/tensor/ontology contracts and excluding classifier math, loss/optimizer behavior, detector, and training. R0 does not start either task.

## Non-goals confirmed

No classifier was implemented/restored; no geometry/membership semantics, objective, optimizer, defaults, model/dataset/training/inference code, tensor/ontology contract, run identity, historical acceptance report, or retirement-manifest content was changed. No detector/scientific training or R1/R2 work began. The implementer has not accepted or closed this task.

## Independent-review handoff

Issue #9 is the authoritative task tracker and execution journal. The final review-submission comment contains the exact fixed implementation SHA, push/branch state, complete file inventory, test commands/results, AC mapping, limitations, and reviewer request. Review that exact SHA against all eight criteria and inspect actual changed files/protected history; the task remains open and in review until a separate reviewer acts. Do not modify the submitted revision while under review.
