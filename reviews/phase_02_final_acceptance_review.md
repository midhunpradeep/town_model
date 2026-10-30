# Final Independent Phase 2 Acceptance Review

## Verdict: **ACCEPT PHASE 2**

F4 is independently verified closed. Invalid composed configurations are rejected before fixture construction, training, or filesystem side effects. Geometry feasibility, valid three-family training/evaluation, and the complete regression suite pass. **Phase 3 implementation is authorized.**

## Revision, ancestry, and scope

- Repository branch: `remediation/phase-2-review-findings`
- **Implementation SHA reviewed:** `087b116c4d06cd08af65d775136986a413cc5eff`
- The fetched remote branch HEAD matched this SHA. The accepted Phase 1 baseline (`bf2e308`), prior remediation (`19fdc76`), and prior closure review (`21dc685`) are all ancestors; `087b116` is the only commit after the prior closure review.
- Reviewed the complete `21dc685..087b116` diff (five files). Changes are limited to shared minimum-width validation, composed-config preflight, its export/tests, and this addendum in the remediation report. No loss, membership, class-order, checkpoint-schema, resume, visualization, or dataset changes were introduced.
- Environment: repository `.venv`, Linux, Python 3.12.0, PyTorch 2.5.1+cpu; CPU-only (`torch.cuda.is_available() == False`).
- This report is a separate, report-only commit. Its commit SHA is identified in the final review response; it is distinct from the implementation SHA above.

## F4 — composed-run preflight: **CLOSED**

The single `validate_min_width()` rule in `src/town_model/models/geometry.py` is used by both direct geometry constructors and `SyntheticConfig.validate()`. The composed harness passes its actual family tuple (`baseline`, `box`, `town`), so it applies the stricter town constraint. `run_experiment()` validates the configuration before calling `make_fixture()`, resolving/creating the output run directory, or invoking family training.

For `R = 16 * torch.finfo(torch.float32).eps = 1.9073486328e-6`, observed limits are strict:

- Box: `0 < min_width < 2 - 2R` (`1.9999961853`).
- Contained town: `0 < min_width < (2 - 5R)/3` (`0.6666634878`).

### Required invalid-run reproduction and side effects

Executed the requested `run_experiment("fixed", "simple", output, config=SyntheticConfig(seed=2, points=30, steps=1, min_width=0.7))` with `output` initially nonexistent and spies on fixture construction and family training.

- Raised actionable `ValueError`: `min_width ... less than 0.666663488 ... contained town frames and clearance feasibility`.
- Fixture construction calls: **0**.
- Family-training calls: **0**.
- Output root created: **no**.
- Run/status/metrics files or checkpoints: **none**.

Thus baseline, box, or town training cannot partially run for this invalid composed configuration.

### Independent numerical and type probes

Probed defaults, `1e-8`, `0.7`, `1.999999`, zero, negative values, NaN, infinities, booleans, strings, `None`, complex values, tensor scalars, and NumPy scalar types. Invalid values fail validation; positive `numpy.float64` values work. `0.7` is valid for a direct pure box but invalid for a contained town and for the composed run.

For each family, the immediately representable float64 value below its limit (`math.nextafter(limit, -inf)`) was accepted; the exact limit and immediately-above value were rejected. The composed configuration accepts the value just below the town limit and rejects the boundary and above.

For default, small-positive, and accepted near-limit widths, independently forced all sign combinations of extreme raw parameters (±`1e30`) and checked actual float32 geometry. Box and town corners remained in `[-1,1]`; boxes did not collapse; frame and window widths and both contained-window clearances met the requested minimum. Town windows remained strictly inside their frames. No silent physical-corner clipping was added.

## Valid experiment and regression evidence

A disposable fixed/simple run (seed 22, 30 points, one step) completed **baseline, box, and town** training. Each produced metrics and a checkpoint; each checkpoint loaded and evaluated successfully. Reloaded validation loss, known-label accuracy, macro-F1, counts, per-class metrics, and objective exactly matched recorded results. Checkpoint configuration hashes matched each saved resolved configuration; class order and shared fixture/split identity were preserved. The existing schema-2 identity/checkpoint format required no migration.

Trainable-point mapping regression:

```text
.venv/bin/pytest -q tests/test_phase2_experiments.py::test_trainable_mapping_has_matched_initial_encoders_and_gradient_paths
1 passed
```

The complete diff is narrowly limited to preflight/rule reuse. Earlier independently closed findings F1, F2, F3, F5, F6, and F7 are unaffected; no regression or new defect was found.

## Full regression suite and Phase 1 sample

- `.venv/bin/pytest -q` — **143 passed, 0 failed, 0 skipped** (29.38s).
- `.venv/bin/python -m compileall -q src tests benchmarks` — passed.
- `.venv/bin/python -m pip check` — no broken requirements.
- `.venv/bin/town validate examples/pizza-48 --no-write-report --json` — valid, 0 errors, 178 low-class-support warnings.
- `.venv/bin/town inspect examples/pizza-48 --json` — `integrity_status=current_pass`.
- Dataset comparison against the accepted Phase 1 baseline — unchanged.

Five nonblocking Streamlit/Pandas `Pandas4Warning` warnings report the deprecated `DataFrame.reset_index(inplace=...)` use in Streamlit's chart helper. CUDA was unavailable; no CUDA behavior is claimed.

## F1–F7 closure matrix

| Finding | Status | Basis |
|---|---|---|
| F1 — Overlap plotting | **CLOSED** | Prior independent closure; final diff does not touch rendering. |
| F2 — Zero-margin persistence | **CLOSED** | Prior independent closure; final diff does not touch objectives/evaluation/resume. |
| F3 — High-dimensional initialization | **CLOSED** | Prior independent closure; initialization policy is unchanged by this preflight-only fix. |
| F4 — Physical-domain/config preflight | **CLOSED** | Shared feasibility helper, strict boundary probes, invalid-run side-effect reproduction, and float32 geometry checks above. |
| F5 — Snapshot diagnostics | **CLOSED** | Prior independent closure; final diff does not touch snapshots/dashboard. |
| F6 — Corrupt-checkpoint handling | **CLOSED** | Prior independent closure; final diff does not touch artifact loading. |
| F7 — README consistency | **CLOSED** | Prior independent closure; final diff does not touch README. |

## Findings and remaining limitations

**New findings:** none. No acceptance blockers remain. Nonblocking limitations include CPU-only verification, the existing low-support warnings in the 48-image sample, and the previously documented deferred image/detector pipeline and broader research work; none is introduced by this correction or blocks Phase 2 acceptance.

## Phase 3 authorization

**Phase 2 is accepted. Phase 3 implementation is explicitly authorized to begin.**
