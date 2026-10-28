# Phase 1 remediation report

**Status:** Remediation implemented and staged for handoff. Phase 1 is **not approved**; this report makes no approval recommendation.

**Repository:** <https://github.com/midhunpradeep/town_model>

**Working branch:** `remediation/phase-1-findings`

**Base commit:** `e6f4050800e32b4c6d85b076b5a406a60f39398` (published `main` at start; includes the independent review)

**Implementation baseline:** `220318f914668cba914b504596dda49a298e006f`
**Remediation commit:** Not created, per the explicit handoff instruction to stage changes and not commit. A remediation SHA will exist only after the user commits.

## Executive summary

All four findings have a correction and regression evidence:

| Finding | Status | Result |
| --- | --- | --- |
| P1-C1 — malformed origin normalization | **Resolved** | Only a blank source CSV cell (or in-memory `None`) means absent origin. Text sentinels, whitespace, and unsupported identities fail. The generator's separate literal `None` type convention remains context-specific. |
| P1-Major — stale inspection/dashboard reports | **Resolved** | Inspection validates current artifacts, reports current and saved validation separately, withholds support counts on failure/unavailability, and disables browsing of invalid data. A file-state cache avoids repeated full validation on unchanged dashboard reruns. |
| P1-M1 — contours/segmentation validation | **Resolved** | Checks are derived from the pinned generator's actual OpenCV contour, concepts-map, and text segmentation formats. Generated example output remains valid. |
| P1-M2 — unreachable provenance commit | **Resolved via documented source-equivalent reference** | The original runtime SHA is preserved as historical evidence. A reachable commit with byte-identical preparation-source trees is recorded and was resolved from a fresh GitHub clone. The historical runtime SHA itself remains unreachable. |

## Findings, root causes, and corrections

### P1-C1 — malformed origin values no longer imply known negatives

**Root cause.** `_normalize_optional()` previously mapped the nonempty strings `NaN`, `nan`, `None`, and `none` to Python `None`. Origin-dependent labels use a closed-world interpretation for complete generated rows, so malformed input could become a valid no-origin known-negative observation.

**Pinned source evidence.** At Pizzaïolo commit `6c0bcb221a957c184012beaf103bebcde4b1f793`, `libs/pizzaiolo.py` writes the country value as Python `None` when absent and passes it through pandas `DataFrame.to_csv`, producing a genuinely empty CSV field. Preparation uses `read_csv(..., dtype=str, keep_default_na=False)`, so the blank remains distinguishable from text. In contrast, anonymous/random `type` is explicitly written as the literal string `None` by the generator. The code now handles that one representation only in the `type` column; generic optional normalization no longer turns text into absence and does not strip whitespace into a blank.

**Correction.** `_normalize_optional()` accepts only `None` or the empty string as absence. Unsupported country identities and all nonempty sentinels fail before label inference/box reconciliation. Existing no-origin masks remain known negatives; a rendered flag with blank origin and an origin/flag mismatch fail validation.

**Evidence.** Regression cases cover blank origin; `NaN`, `nan`, `None`, `none`, whitespace, and Germany in preparation and validation; metadata/flag inconsistency; America-positive and blank-origin-negative targets/masks; and preserved random-type handling. All assert actionable preparation errors or named validation diagnostics, not helper return values.

### P1-Major — inspection trusts current artifacts, not saved PASS/statistics

**Root cause.** `inspect_dataset()` returned manifest statistics and the saved validation report without checking current sample and annotation files. The dashboard explorer loaded samples independently of validation. Changing a target while leaving manifest/report intact therefore left historical support and PASS visible.

**Correction and trust policy.** The common application inspection service now runs the complete validator against current files and exposes:

- `integrity_status`: `current_pass`, `current_fail`, or `unavailable`;
- `validation`: the current validation result and its `current_artifact_state` scope;
- `saved_validation`: the on-disk report, explicitly scoped as `historical_only`.

Only current passing validation supplies summary support/count fields. CLI inspection exits nonzero on current failure/unavailability and retains actionable JSON diagnostics. The dashboard labels current status, marks saved status historical, withholds manifest statistics after failure, and disables the image explorer unless current validation passes. Explicit validation always runs the full gate. For dashboard reruns, validation is cached only within the process and keyed by the manifest plus device/inode/size/mtime/ctime for every declared asset; any observed state change triggers full validation. A new CLI process performs its own full validation.

**Regression evidence.** Tests mutate a sample target without changing manifest or saved validation, verify explicit validation fails, verify CLI inspection returns failure and no current support counts, and verify dashboard status is FAIL and explorer data is unavailable. Separate tests prove unchanged state reuses validation and changed state invalidates the cache. The committed example demonstrates current validator `1.1.0` versus its historical saved report `1.0.0`.

### P1-M1 — validation follows pinned contour and segmentation formats

**Source-derived invariants.** The pinned generator serializes each contour category as a list of OpenCV contours with integer `N×1×2` points. It derives bounding boxes only from root/external contours, while the contour file also retains nested contours (including holes); consequently contour count must be **at least** box count, not equal. Coordinates are image pixel coordinates. Segmentation is a `uint8` image-sized matrix written as whitespace-separated integers by `numpy.savetxt`; `concepts.json` maps ID `0` to `None`/background and the nonzero IDs to the 22 rendered detector identities.

**Correction.** Validation now checks the concepts map; contour category equality with boxes/components; per-category contour count against box occurrences; OpenCV point-array structure, integer coordinates, and image bounds; segmentation parseability, dimensions, permitted IDs, and consistency of used non-background categories with that sample's components. Missing, malformed, truncated, or inconsistent assets fail.

**Mask/occlusion limits.** No contour is required to match a box edge. No per-category segmentation pixel or pixel-for-pixel contour/box equality is required: upstream masks overlap, are clipped, and are composited with overwrite semantics. These limitations are documented in `docs/DATASET_SCHEMA.md`. Source annotation files are not rewritten.

**Regression evidence.** Fixtures cover the reviewer's `{}` contour document with coherently updated checksums/manifest/dataset IDs, insufficient contour counts, malformed points, out-of-bounds coordinates, invalid segmentation IDs, wrong dimensions, and a segmentation category absent from the sample. A regression validates the committed 48-image pinned-generator dataset, and an actual newly generated dataset also passes.

### P1-M2 — independently verifiable preparation code

**Root cause.** The recorded preparation-time HEAD `99956608ac88fd5ab3011d28d57937c7b5e32983` exists in the original local object database but is not reachable from published refs. It cannot be fetched by a clean clone.

**Correction.** The manifest preserves the original `preparation_commit` record and explicitly says it is historical/unreachable. It adds `preparation_source_reference_commit=220318f914668cba914b504596dda49a298e006f` and the source tree object ID `40aee3f823d20145f022bd65eb8b2eaf5fde490a`, with the reference's role explained. No data assets or saved validation report were regenerated. The manifest ID was recomputed for the added provenance metadata; the dataset ID remains `pizza-dcc148789aecf8af86a3`.

**Verification evidence.** A fresh clone of `git@github.com:midhunpradeep/town_model.git` checked out the recorded reference. The clone resolved commit `220318f914668cba914b504596dda49a298e006f`; its `src/`, `tests/`, and `pyproject.toml` tree/blob IDs were respectively `40aee3f823d20145f022bd65eb8b2eaf5fde490a`, `328b792b3d2eb591a4f25cdfd2e7aa25df557f5d`, and `9ed9f791aa506d51ced4d65b46cb11036dcc3133`. `README.md` also matches (`6f842851f8962746ad4171f9ac516a1bb7cfb504`). These IDs match the locally available historical commit's corresponding objects. The original `999566…` commit was absent from the fresh clone, as expected. Thus the fetchable reference identifies the exact preparation source tree, without claiming it was the historical runtime HEAD.

## Changed files

1. `docs/DATASET_SCHEMA.md`
2. `docs/PHASE_1_REPORT.md`
3. `docs/PROVENANCE.md`
4. `docs/PHASE_1_REMEDIATION_REPORT.md` (new)
5. `examples/pizza-48/manifest.json`
6. `src/town_model/application.py`
7. `src/town_model/cli.py`
8. `src/town_model/dashboard/app.py`
9. `src/town_model/dashboard/data.py`
10. `src/town_model/dataset/constants.py`
11. `src/town_model/dataset/preparation.py`
12. `src/town_model/dataset/validation.py`
13. `tests/conftest.py`
14. `tests/test_application_cli_dashboard.py`
15. `tests/test_dashboard_app.py`
16. `tests/test_dataset.py`

The original review at `reviews/phase_01_independent_review.md` was not modified. No changes were made to `fourth_attempt/`, the pinned upstream generator, model/training functionality, or later-phase geometry/detector code.

## Tests and verification performed

| Command / exercise | Result |
| --- | --- |
| Baseline `pytest -q` before implementation | **37 passed** |
| `python -m compileall -q src tests` | Passed |
| Fresh environment: `python3 -m venv /tmp/town-phase1-remediation-venv`; `pip install -e '.[test]'` | Passed; editable package and test dependencies installed |
| Fresh-environment `python -m pip check` | Passed: no broken requirements |
| Fresh-environment `python -m pytest -q` | **62 passed**, 3 Streamlit/Pandas deprecation warnings |
| JSON Schema Draft 2020-12 schema check and example-record validation | Passed; 0 schema errors |
| `town validate --json examples/pizza-48` | Valid; 0 errors, 178 low-support warnings |
| `town inspect --json examples/pizza-48` | `current_pass`; 48 samples; train/validation/test = 16/16/16 |
| Real pinned-source generation: 3 `random` samples, split seed 42 | Passed; 1 sample in each split; validation 0 errors; CLI inspection `current_pass` |
| Repeat the same real generation to a separate output | Same dataset ID and identical immutable asset-checksum map |
| Pinned upstream checkout after generation | Clean at `6c0bcb221a957c184012beaf103bebcde4b1f793` |
| Streamlit `AppTest` valid explorer, corrupted-data status/blocking, and generation page | Passed in the full suite; no app exceptions |
| `town dashboard --headless --address 127.0.0.1 --port 18517` and `/_stcore/health` | Passed; endpoint returned `ok` |
| Fresh-clone provenance check | Passed for source-equivalent commit; original historical SHA absent from clone |

No 4,800-image run, training experiment, or manual graphical-browser session was performed. Automated dashboard tests and the HTTP health check were used instead.

## Before/after behavior

| Finding | Before | After |
| --- | --- | --- |
| P1-C1 | Literal origin sentinels normalized to absence and could create known negatives. | Only blank/`None` is absent; malformed nonempty text and unsupported identities fail. Blank-origin known negatives and valid origin-positive labels are preserved. |
| P1-Major | CLI/dashboard exposed stale saved PASS/support after file mutation. | Current validation runs on first/changed file state; current/saved reports are distinct; stale counts are withheld and corrupt explorer access is blocked. |
| P1-M1 | Coherently rehashed `{}` contours and weakly checked segmentation could pass. | Required category/count/point/dimension/ID/cross-file invariants fail malformed fixtures; pinned generated artifacts still pass. |
| P1-M2 | Manifest/report named an unresolvable runtime SHA with no published source ref. | Historical identifier retained transparently; a clean-fetchable, source-tree-identical revision is recorded and verified. |

## Compatibility, limitations, and handoff

- Sample wire schema remains `1.0.0`; no sample fields, ontology labels, categories, identities, splits, or dataset assets changed. The validator version is now `1.1.0`; the committed saved report remains `1.0.0` and is explicitly historical. The example manifest provenance fields changed and its `manifest_id` was recomputed; `dataset_id` and data assets are unchanged.
- `DatasetSummary` adds current integrity/saved-report scope fields, and `sample_count` is `None` when current validation does not pass. `town inspect` now returns a nonzero status on failure/unavailability. Consumers of the CLI summary should read `integrity_status` before using counts.
- The process-local inspection cache uses filesystem change metadata to avoid rehashing/revalidating every declared asset on every dashboard interaction. It is intended for ordinary changes, not an attacker able to alter bytes while defeating filesystem metadata. Explicit validation and each new CLI process perform full validation.
- The historical execution commit `999566…` itself remains unreachable. The alternate provenance is a verified source-equivalent reference and does not prove the exact runtime checkout beyond the identical source trees.
- No later-phase functionality was added. Phase 1 remains pending independent review; this report does not declare it approved.

**Suggested commit message:** `fix: remediate Phase 1 validation and provenance findings`
