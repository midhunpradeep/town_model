# Phase 1 — Second Remediation Report

**Status:** D1–D5 implementation complete and staged for independent review. Phase 1 is **not approved**, and this report does not approve the implementation.

**Repository:** <https://github.com/midhunpradeep/town_model>

**Branch:** `remediation/phase-1-findings`

**Base commit / latest review HEAD:** `a82b96cb89a2f6a4788c5d534182d72f42044baf` (`docs: add independent Phase 1 remediation review`)

**Last reviewed implementation:** `54a16c57280afed6accaf3837a7db8d112b0cd1b` (`fix: remediate Phase 1 validation and provenance findings`)

**Implementation commit:** Not created, per the user's explicit instruction to stage changes without committing. The commit SHA will be assigned when the author commits. The complete implementation and this report are staged on the branch.

The first remediation commit, its independent review commit, and the new second-remediation work are distinct. The historical preparation SHA `99956608ac88fd5ab3011d28d57937c7b5e32983` remains an unreachable historical identifier; the reachable source-equivalent reference `220318f914668cba914b504596dda49a298e006f` has matching `src/` tree `40aee3f823d20145f022bd65eb8b2eaf5fde490a`. The reference is not represented as the runtime revision. The first remediation's provenance claims are preserved.

## Summary of D1–D5

| Finding | Correction |
|---|---|
| **D1 — validation/read consistency** | Bind validation and explorer loading to one declared artifact snapshot; verify bytes consumed and the selected image/annotation assets; fail closed and invalidate on change. |
| **D2 — blank generator type** | Accept exact upstream `"None"` for random and the pinned named types only; reject blank, whitespace, sentinel, and unsupported values in preparation and validation. Cross-check stored audit against the preserved raw CSV cell. |
| **D3 — rerender cost** | Replace separate validation-only caching with one bounded process-local validated explorer bundle and Linux inotify invalidation; explicit validation remains a full integrity pass. |
| **D4 — test isolation** | Exercise the historical/current validation distinction on the disposable `prepared_dataset` fixture, never the committed example report. |
| **D5 — stale history** | Correct the first-remediation account and record the first-remediation and independent-review SHAs without changing the historical preparation provenance. |

No schema version or committed example dataset asset changed. Previously accepted origin, target, ontology, contour/segmentation, split, immutability, and provenance behavior was retained.

## Root causes and design

### D1 — validation and consumed-data consistency

**Root cause.** The first remediation validated through `inspect_dataset()` and then separately loaded vocabularies and JSONL records. A successful validator return did not bind those later reads to the bytes checked by the validator. The selected image was also opened directly from its path after validation. A mutation in either gap could pair a current PASS with different consumed data.

**Correction.** `application.py` now owns a single cache entry containing the validation report, manifest bytes/object, summary, and (on successful load) the parsed vocabularies and records. A bounded two-attempt validation window is guarded by a dependency-free Linux inotify watcher over dataset files and their directories. Changes during validation trigger one retry; repeated changes produce an unavailable status. A fallback state comparison is used where inotify is unavailable.

Explorer construction reads and hashes `manifest.json`, both vocabulary files, and `samples.jsonl`; vocabularies and records are parsed from the exact verified byte buffers, not reopened paths. A changed/missing artifact invalidates the cache and causes a bounded reload/revalidation. Failed validation never exposes records or leaves an earlier successful explorer cache entry accessible. Explicit validation discards the old entry and performs the complete validation gate.

The dashboard now renders manifest metadata from the validated explorer snapshot rather than reopening `manifest.json`. For a selected image, the shared application service verifies the manifest-declared SHA-256 of the image and all three source annotations referenced by that record, reads the image into a buffer, checks for watcher events across the read, and opens that same buffer. A mismatch rejects the display, invalidates the cache, and gives an actionable message. Image-open changes are rejected immediately; the next rerun revalidates.

**Consistency guarantee.** A current PASS and its displayed JSONL records/vocabularies are bound to bytes matching the manifest checksums that passed validation. A displayed image and its associated annotation files are independently checked against those same declared checksums when selected. Relevant local changes observed during validation/loading/opening invalidate the bundle; `load_explorer()` makes at most two outer attempts, while each validation window permits at most two integrity passes. No partial data is returned as a successful explorer.

This preserves the dataset's immutable-artifact/checksum contract; it does **not** claim an OS-level atomic filesystem snapshot or treat metadata/event checks as cryptographic protection. Linux inotify is best-effort local change notification. Remote/network filesystem writes, notification failures, or an actor able to defeat both notification and declared checksums are limitations. On platforms without inotify, the fallback checks metadata for every declared artifact on access (safer but linear in asset count). Use explicit validation where filesystem notifications are unreliable.

### D2 — source-faithful type metadata

**Root cause.** Preparation used optional-value normalization for the generator `type` field, so an empty CSV cell became `None` and was then mapped to random. Validation also accepted `"unspecified"` as random and did not whitelist stored source type values.

**Correction.** Preparation accepts only the pinned literal `"None"` (normalized in the sample audit to Python/JSON `null`) or a named type from `DEFAULT_PIZZA_TYPES` other than `random`. Empty, whitespace-only, `"none"`, `"unspecified"`, and unsupported values fail. Validation enforces the same stored-audit domain and checks the sample's normalized value against the original `type` cell in `audit/generator_master.csv`. This keeps source-level `"None"` auditable and prevents malformed stored values from being reinterpreted. Type-stratified splitting remains unchanged.

### D3 — responsive unchanged-data interactions

**Root cause.** The prior cache avoided repeat validation but built a stat key by scanning every declared asset on each call; `load_explorer()` then reparsed all sample JSONL records and rebuilt the explorer object on every Streamlit rerun.

**Correction and tradeoff.** A single LRU holds at most three dataset entries, each with its validated summary/report, manifest, watcher, and lazily loaded parsed explorer bundle. On Linux, routine accesses drain the nonblocking watcher queue rather than statting or hashing all assets. An unchanged `load_explorer()` returns the same bundle. A selected-image interaction checks only that sample's three source annotations and image. Any observed dataset event invalidates the entire entry and causes full revalidation/reload; the explicit validation control also forces the full gate. Each CLI process validates independently. Cache size is bounded and watcher file descriptors are closed on eviction.

Initial watcher installation and initial/full invalidation validation remain dataset-sized by design. The fallback path on platforms/filesystems where inotify is unavailable also scans declared-file metadata per access. The cache is process-local; it is not a distributed freshness mechanism.

### D4 — test independence

**Root cause.** A test required the tracked `examples/pizza-48/reports/validation.json` to retain validator `1.0.0`, but the documented CLI command can rewrite that mutable report to `1.1.0`.

**Correction.** The regression now changes the saved validator version only on the disposable `prepared_dataset` fixture and asserts that current validation is `1.1.0` while the saved report is marked historical. No test mutates committed example assets.

### D5 — historical handoff accuracy

`docs/PHASE_1_REMEDIATION_REPORT.md` now records the verifiable first-remediation commit `54a16c57280afed6accaf3837a7db8d112b0cd1b`, independent-review commit `a82b96cb89a2f6a4788c5d534182d72f42044baf`, original implementation baseline `220318f914668cba914b504596dda49a298e006f`, and the original preparation SHA/source-equivalent provenance distinction. It states that the second implementation commit does not yet exist because the user requested no commit.

## Performance verification

**Reproduction:** `python benchmarks/phase1_dashboard_cache.py --iterations 3` from the repository root. The script copies the committed 48-image example to a temporary directory and builds a disposable, validator-compatible 4,800-sample scale fixture. The larger fixture has 4,800 distinct image files and 14,400 distinct annotation files, approximately 19,200 per-sample artifact paths. It duplicates the example's 48 rows 100 times with new IDs; its dimension-valid segmentation matrices are background-only to limit fixture size. It is **synthetic scale data, not 4,800 Pizzaïolo generation**. The original example is never modified. Fixture creation/reset leaves the process-local dataset cache cold; OS filesystem data may be cached.

Measurements include validation and watcher setup for cold inspection; cold explorer load includes full validation plus parsing. Repeated timings are medians of three unchanged calls. Image selection includes cached `load_explorer()` plus verification/opening of the selected image and its annotations. Invalidation appends a newline to disposable `samples.jsonl`, then times watcher-driven invalidation and full revalidation (expected `current_fail`). Timings exclude Streamlit browser/network rendering.

Environment: Python 3.12.0; Linux x86_64, kernel 5.15.0-134-generic, glibc 2.35; Streamlit 1.66.1, pandas 3.1.0, NumPy 2.5.4, OpenCV headless 4.14.0.94, Pillow 12.3.0, pytest 8.4.2. No full reference dataset was generated.

| Fixture | Files / sample count / logical bytes | Initial inspection | Repeated inspection | Initial explorer load | Repeated explorer load | Image selection interaction | Change invalidation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 48-image example copy | 204 / 48 / 14,397,957 B | 0.445 s | 1.16 ms | 0.498 s | 1.23 ms | 3.20 ms | 0.444 s |
| Synthetic 4,800 scale fixture | 19,211 / 4,800 / 1,397,740,519 B | 41.536 s | 0.118 ms | 44.890 s | 0.081 ms | 2.00 ms | 41.276 s |

The large initial/invalidation times are full validation costs on this synthetic fixture, not a performance target. The key D3 result is that unchanged inspection/explorer calls and ordinary image selection no longer perform dataset-sized JSON parsing, explorer reconstruction, artifact validation, or declared-asset scans on the Linux watcher path. The benchmark is reproducible through the committed script and does not claim complete-reference generation or end-to-end browser latency.

## Regression tests and results

New/adjusted regression coverage includes:

- **D1:** mutation before validation; mutation after successful validation but before explorer loading; mutation between vocabulary and sample loading; manifest and vocabulary mutations; mutation before selected-image opening; failed explicit validation after a successful cached load; unchanged bundle identity and targeted selected-asset reads.
- **D2:** preparation rejects empty, whitespace, and unsupported source types; validation rejects malformed stored audit values and mismatches against the original raw CSV `type`; literal `"None"` random and named types remain valid.
- **D4:** current-versus-historical validation test operates on a temporary dataset fixture.

Full suite, including all existing Phase 1 tests: **84 passed**, no skips; three existing Streamlit/Pandas deprecation warnings. `python -m compileall -q src tests benchmarks` passed. `python -m pip check` passed.

## Installation and integration verification

- Created a clean Python 3.12 virtual environment and installed `pip install -e '.[test]'`; installation succeeded and `pip check` reported no broken requirements.
- Ran the full test suite from that clean environment: **84 passed**.
- Ran the documented `town validate --json` operation against a disposable copy of `examples/pizza-48`; it updated only the temporary saved report to validator `1.1.0`. The full suite then passed again, demonstrating command-order independence. The committed example report/assets remained unchanged.
- Generated a real three-sample random dataset using the clean pinned upstream checkout `SysReIC/pizzaiolo@6c0bcb221a957c184012beaf103bebcde4b1f793`. Dataset preparation completed; CLI validation returned valid with 0 errors (323 expected low-support warnings); CLI inspection returned `current_pass`, 3 samples, and 1/1/1 splits.
- On a disposable corrupted copy, `town inspect` returned `current_fail`, withheld sample counts, and exited 1.
- Started the dashboard with the clean environment; `/_stcore/health` returned `ok`. Streamlit `AppTest` rendered the valid generated sample/image and rejected/disabled browsing of the corrupted copy. These are automated interactions, not a manual graphical-browser session.
- Cache identity, image checksum guarding, validation failure invalidation, all four required deterministic D1 race windows, and malformed D2 inputs were exercised by regression tests.
- Confirmed the pinned upstream checkout remained clean at the pinned SHA. No full 4,800-image Pizzaïolo generation, model training, manual browser session, or non-Linux watcher verification was performed.

## Modified files and observable behavior

1. `src/town_model/application.py` — bounded snapshot/explorer cache, force-validation invalidation, verified-byte loading, selected-image service.
2. `src/town_model/dataset/watch.py` — dependency-free Linux inotify watcher and conservative cache-change notification.
3. `src/town_model/dataset/schema.py` — JSONL parser for already-verified byte buffers; path API retained.
4. `src/town_model/dataset/preparation.py` — exact source-type acceptance.
5. `src/town_model/dataset/validation.py` — strict stored type validation and raw-CSV correspondence diagnostics.
6. `src/town_model/dashboard/app.py` — use validated manifest snapshot and verified image open.
7. `tests/test_application_cli_dashboard.py` — deterministic snapshot races, cache, historical-report isolation.
8. `tests/test_dataset.py` — type metadata regressions.
9. `docs/DATASET_SCHEMA.md` — generator type contract and snapshot/cache guarantees/limitations.
10. `docs/PHASE_1_REMEDIATION_REPORT.md` — corrected historical commit account.
11. `benchmarks/phase1_dashboard_cache.py` — reproducible 48/4,800 cache benchmark.
12. `docs/PHASE_1_REMEDIATION_2_REPORT.md` — this handoff.

Externally visible differences: malformed/blank type metadata now fails preparation/validation; validation reports include a `generator_type_metadata` check/diagnostics; explorer snapshots expose a cache token and validated manifest; image display can now reject a file changed after validation instead of rendering it. The sample wire schema remains `1.0.0`; no dataset/example payload, target semantics, split policy, or Phase 2 feature changed.

## Risks, limitations, and deviations

- No implementation commit SHA exists yet because this handoff explicitly forbids committing. The implementation and report are staged on the existing remediation branch; the author must create and push the commit before the branch URL resolves to this report.
- Inotify is best-effort and Linux-specific; non-Linux or failed watcher initialization falls back to linear declared-file metadata checks. Network/remote filesystem notifications may be incomplete. Content hashes are checked for all explorer bytes and selected sample assets, but this is not a general atomic-filesystem or adversarial-security guarantee.
- A full validation or detected large-dataset change is still expensive (about 41 s for this synthetic 4,800 fixture in this environment); unchanged interactions are fast. OS, filesystem, and synthetic-content differences affect absolute timings.
- The larger benchmark is an explicitly synthetic scale fixture with duplicated source rows and background-only segmentation; it is not actual Pizzaïolo generation and does not establish full-reference generation or dashboard browser latency.
- No manual GUI/browser, non-Linux watcher, full reference dataset, detector, training, or Phase 2 functionality was exercised or added.

No Phase 2 work was started, no historical provenance claim was rewritten, and this report does not declare Phase 1 accepted.
