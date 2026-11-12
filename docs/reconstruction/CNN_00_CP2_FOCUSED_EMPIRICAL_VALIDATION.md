# CNN-00 CP2 — Focused Empirical Validation

**Checkpoint:** CP2 only; this is not CP3, a classifier reconstruction, a training result, or an architecture decision.

## Authorization and execution identity

- **Repository/source base:** `midhunpradeep/town_model`, exact audited source SHA `404d76112a9d53282b3f55698b4decb705a5601c`. Live `main` matched this SHA at execution start; the relevant source blob IDs are recorded in [`environment.json`](evidence/cnn00_cp2/environment.json). CP1 report `fc08f11e98aeef0bcc61036228a26f13786027cc` was independently accepted for CP1 only; see [CP1 verdict](https://github.com/midhunpradeep/town_model/issues/25#issuecomment-6813001736).
- **Authority:** thesis-author CP2 execution authorization [#25 comment 6813353068](https://github.com/midhunpradeep/town_model/issues/25#issuecomment-6813353068), applying the complete P6/P7/P4/P5 fixture/tolerance/evidence specification [comment 6813267997](https://github.com/midhunpradeep/town_model/issues/25#issuecomment-6813267997).
- **Isolated worktree/branch:** `/tmp/town-model-cnn00-cp2`, `investigation/cnn-00-cp2-focused-validation`, created clean from the exact source SHA above. No production source, existing tests, configuration, scientific decision, accepted report, classifier, BOX/TOWN worktree, or `main` was modified.
- **Environment:** Python `3.12.0`; torch `2.5.1+cu124`; torchvision `0.20.1+cu124`; NumPy `2.5.4`; Pillow `12.3.0`; Linux Mint x86_64, kernel `5.15.0-134-generic`, Intel Core i5-9300H. All probes explicitly ran on **CPU**, float32, seed `2026`, one Torch intra-op and inter-op thread, with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`. The installed PyTorch build reports CUDA 12.4 and `torch.cuda.is_available() == true`, but no operation was sent to CUDA and no CUDA result is claimed.
- Each visual factory config used `pretrained_imagenet=False`; no pretrained weights were downloaded. Fixtures were deterministic, synthetic, generated in memory; no dataset or split was accessed. No optimizer, parameter update, objective/loss, classifier, detector, or training job was used.

## Overall results

| Probe | Result | Invariant summary |
|---|---|---|
| **P6 — Gradient connectivity** | **PASS** | All six separate point/final-output contractions had the expected active parameter/input graph, expected disconnected paths (`None`), finite observations, and exactly zero masked-input gradient where applicable. No connected active group was wholly zero; no terminal output approached the recorded `abs(value) >= 0.99` diagnostic threshold. |
| **P7 — Frozen-state correctness** | **PASS** | All 5,624,112 visual parameter values across 186 parameter tensors were exactly unchanged by one forward for each policy. Frozen-module buffers remained unchanged. The selected trainable block’s executed BatchNorm counter advanced once and its running buffers changed as recorded; this is expected trainable-state characterization, not a frozen-state failure. |
| **P4 — ROI isolation** | **PASS** | A-repeat control passed; normalized component coordinates matched exactly; all component/global features were finite. Global spatial tensors were bit-identical across A, A-repeat, B, and empty providers. Component pooled features differed between A and B and are reported independently. |
| **P5 — DeepSets composition** | **PASS** | Masked padding changed neither final `rho` output nor valid gradients; masked-input gradients were exactly zero. The controlled duplicate witness changed the final output as specified. This is a witness, not a universal learned-representation guarantee. |

The machine-readable invariant-by-invariant record, complete per-parameter gradient summaries, every named parameter’s before/after hashes and equality result, all module modes/`requires_grad` flags, buffer and BatchNorm records, and raw tensors are in [`observations.json`](evidence/cnn00_cp2/observations.json) and [`raw_observations.npz`](evidence/cnn00_cp2/raw_observations.npz).

## P6 — Gradient connectivity

The input was a direct CPU float32 `VisionFeatureBatch`: component vectors `[2,3,260]`, global spatial features `[2,256,7,7]`, seed-2026 Gaussian inputs scaled by `0.05`, mask `[[T,T,F],[T,F,F]]`, and matching occurrence IDs/`None` padding. `ImagePointModel(dimensions=4, seed=2026).eval()` was tested in global, compositional, and hybrid modes. For each mode, separate fresh forwards were contracted using fixed non-symmetric coefficients over valid component points and over selected final pizza points. These are scalar diagnostics, not losses.

| Mode / contraction | Connected parameter groups (combined gradient L2; max abs) | Disconnected groups (`None` for every parameter) | Input gradient L2 / max abs | Masked component gradient nonzeros |
|---|---|---|---|---:|
| global / component points | `component_projection` 29.7381 / 4.14377 | `global_projection`, `composition`, `fusion` | component 6.54172 / 0.852185; global `None` | 0 |
| global / final pizza | `global_projection` 1.42938 / 0.360013 | `component_projection`, `composition`, `fusion` | component `None`; global 0.248364 / 0.00645822 | N/A |
| compositional / component points | `component_projection` 29.7381 / 4.14377 | `global_projection`, `composition`, `fusion` | component 6.54172 / 0.852185; global `None` | 0 |
| compositional / final pizza | `component_projection` 0.636813 / 0.0894921; `composition` 1.45105 / 0.359865 | `global_projection`, `fusion` | component 0.216219 / 0.0242317; global `None` | 0 |
| hybrid / component points | `component_projection` 29.7381 / 4.14377 | `global_projection`, `composition`, `fusion` | component 6.54172 / 0.852185; global `None` | 0 |
| hybrid / final pizza | `component_projection` 0.143254 / 0.0215994; `global_projection` 0.260791 / 0.0924212; `composition` 0.371178 / 0.0518149; `fusion` 1.16478 / 0.359575 | none | component 0.0358758 / 0.00504677; global 0.0434057 / 0.00131894 | 0 |

**Invariant outcomes:** expected active paths connected — **PASS**; expected inactive paths disconnected (`None`, not a connected zero tensor) — **PASS**; feature-input graph paths match each mode — **PASS**; padded component input gradients exactly zero — **PASS**; all observed gradients/outputs finite — **PASS**. Every connected active group had nonzero elements. The JSON/NPZ retains per-parameter `None`, exact-zero/nonzero counts, max-absolute values, L2 norms, and input gradient arrays.

Terminal pre-tanh ranges were component `[-0.540734, 0.115296]`, global `[-0.0171955, 0.0465343]`, composition ` [0.0209025, 0.123931]`, and hybrid fusion `[-0.111446, 0.0960442]`. Across all four returned point tensors, the fraction with `abs(value) >= 0.99` was `0`. ReLU positive fractions were component projection `0.4954` and `0.5326`, composition `phi` `0.5052` and `0.4753`, and `rho` `0.5000`; no observed ReLU was wholly inactive. These diagnose this fixed seeded fixture only. Gradient connectivity does **not** demonstrate useful optimization, learning, or scientific performance.

## P7 — Frozen and selectively trainable visual state

Each case used the supported validated `build_visual_feature_extractor(ImageExperimentConfig(...))` factory, `pretrained_imagenet=False`, resolution 64, seed 2026, and the same two-image synthetic `ImageBatch` with empty component ROIs. The extractor was set to train mode and forwarded exactly once under `torch.inference_mode()`; there was no optimizer/update. The JSON records per-name parameter hashes, exact before/after equality and delta counts, all named `requires_grad` flags and module modes before/after, and raw before/after buffer tensors. The raw NPZ stores the complete shared initial parameter arrays once (all three factory constructions had identical per-name hashes); every policy’s after-state was checked `torch.equal` to its before-state.

| Policy | Trainable parameters | Parameter changes | Frozen buffers | BatchNorm behavior | Result |
|---|---:|---:|---:|---|---|
| `frozen` | 0 / 186 tensors | 0 of 5,624,112 values | 138/138 unchanged | All 46 BatchNorm modules were eval-mode; all buffers unchanged | **PASS** |
| `trainable_fpn` | 16 / 186 tensors (FPN) | 0 of 5,624,112 values | 138/138 unchanged | MobileNet BatchNorm modules were eval-mode; FPN has no changing tracked BatchNorm state | **PASS** |
| `trainable_fpn_last_blocks`, block `(16,)` | 19 / 186 tensors (FPN plus block 16) | 0 of 5,624,112 values | 135/135 unchanged | `backbone.mobile_features.16.1` was train-mode and invoked once: `num_batches_tracked +1`; `running_mean` 960 values changed, max abs delta `1.72029e-10`; `running_var` 960 values changed, max abs delta `0.00999999046` | **PASS** |

All policies preserved every parameter’s value exactly and retained their configured train/eval and `requires_grad` policies across the forward. The only mutable-buffer deltas were within the selected trainable block’s BatchNorm; its tracked counter advanced exactly once. Other frozen-module buffers remained exact. Component features were `[2,0,260]`; global features were finite `[2,256,7,7]` in all policies.

## P4 — ROI isolation

A fixed extractor built through the same factory with `pretrained_imagenet=False` was held in eval mode. The two 64×64 RGB images were generated by the recorded pixel formulas in `observations.json`, passed through `ImageNetLetterbox(64)`, and used with all-unknown dummy targets and empty GT component lists. Explicit provider A used `[4,4,28,28]`, `[36,32,60,60]`; B used `[0,0,20,24]`, `[24,8,52,40]`; the empty provider supplied no boxes. Each provider was run under inference mode with diagnostics; A was repeated first as the determinism control.

- **Coordinates:** A and B each returned `[2,2,4]` coordinates matching `(cx/W, cy/H, w/W, h/H)` exactly (max absolute error `0`, tolerance `atol=1e-7, rtol=0`). Empty returned `[2,0,4]` coordinates.
- **Component path:** A/B pooled means were `[2,2,256]`, finite; A-vs-B max absolute difference `0.0965822`, L2 `0.938824`. The component vectors were `[2,2,260]`. Empty returned pooled features `[2,0,256]` and component vectors `[2,0,260]`.
- **Global path:** all outputs were finite `[2,256,7,7]`. A-repeat, A-vs-B, and A-vs-empty comparisons each had max absolute error `0`, L2 error `0`, satisfying `atol=1e-6, rtol=1e-5`. Repeat determinism control — **PASS**; global independence from component-provider changes — **PASS**.

The observed A/B component-feature difference is descriptive ROI sensitivity only; no minimum feature difference or semantic-quality claim is made. Full raw coordinates, pooled features, global tensors, image pixels, shapes, and comparison records are preserved in the evidence files.

## P5 — DeepSets masked composition and multiplicity witness

`DeepSetsComposition(dimensions=1).eval()` was hand-set to the non-saturating map `tanh(sum(max(x,0)))`: all parameters/biases zero except unit weights at `phi[0]`, `phi[2]`, `rho[0]`, and `rho[2]` `[0,0]`. The valid values were `0.2, 0.4`; padding payloads changed from `-0.7, 0.9` to `0.8, -0.6` with the valid mask fixed.

- Masked-padding output was `0.5370495915412903` in both cases; difference `0`, within `atol=1e-7, rtol=0` — **PASS**.
- Both masked input gradients had exactly `0` nonzero entries. The two valid gradients were finite and nonzero (`0.7115777134895325` each) — **PASS**.
- Through the complete `phi`/sum/`rho` path, one `0.25` occurrence produced `0.24491865932941437 = tanh(0.25)`; two duplicate `0.25` occurrences produced `0.46211716532707214 = tanh(0.5)`. Both expected-value errors were `0` at `atol=1e-7, rtol=0`; output difference was `0.21719850599765778` — controlled multiplicity witness **PASS**.

This only establishes that multiplicity can affect the final output for this hand-set witness; it is not a universal guarantee about learned representations or every multiset.

## Reproducibility and evidence inventory

From the CP2 worktree, the corrected execution command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
tools/cnn00_cp2/run_probes.py --output docs/reconstruction/evidence/cnn00_cp2
```

The command was run twice: after the first run exposed a metadata-capture defect, only that recorder command was corrected, then the same four probes/fixtures/tolerances were rerun once. The first attempt’s log/raw evidence and original diagnostic script are retained under `evidence/cnn00_cp2/attempt_1/`; its malformed `git status` option caused only the environment recorder to emit an error and leave `environment.json` source/status null. That attempt still recorded P6/P7/P4/P5 as PASS. The corrected run records the full source/environment metadata. Both NPZ files have the identical SHA-256 `f4f1939d699632f59ea0454f3b4bbbce1858df807760affd248ae257041d3a71`; no observed value, fixture, or tolerance changed.

- [`run_probes.py`](../../tools/cnn00_cp2/run_probes.py): isolated executable harness; SHA-256 `1c624cf854789d4184843e0c8e4b831fdae806588e5e670bd2c7d19d5ff6e706`.
- [`observations.json`](evidence/cnn00_cp2/observations.json): per-invariant results and summaries.
- [`raw_observations.npz`](evidence/cnn00_cp2/raw_observations.npz): 1,189 raw fixture/output/gradient/parameter/buffer arrays; SHA-256 above.
- [`environment.json`](evidence/cnn00_cp2/environment.json): exact source SHA/blob IDs, Python/platform/device/thread/dependency metadata, script hash and commands.
- [`dependencies.txt`](evidence/cnn00_cp2/dependencies.txt) and [`python_version.txt`](evidence/cnn00_cp2/python_version.txt): captured `pip freeze --all` and interpreter version.
- [`execution.log`](evidence/cnn00_cp2/execution.log): corrected run summary.
- [`attempt_1/`](evidence/cnn00_cp2/attempt_1/): preserved first-attempt evidence and recorder error; its original harness SHA-256 is `a59867451211734bf05eaa8f4ff8f1fc2ec0fb75a455d35d942d614f89dc57de`.
- [`SHA256SUMS.txt`](evidence/cnn00_cp2/SHA256SUMS.txt): checksums for the final report, harness, environment, command logs and raw evidence.

No pytest suite or other model experiment was run. The harness’s exact observed results are reproducible with the command and environment above.

## Limits and acceptance mapping

| Authorized criterion | Evidence | Result |
|---|---|---|
| P6: separate component/final contractions in all modes; active/inactive parameter paths; input and masked-input gradients; finite/zero/`None` distinctions; activation/saturation diagnostics | Six cases in `observations.json`; full raw gradients/inputs/outputs in NPZ | **PASS** |
| P7: configured frozen/FPN/last-block factory modes; parameter values, `requires_grad`, modes, and buffers before/after; no optimizer | All named tensor equality/hash records and raw parameter/buffer arrays; BN invocation/counter records | **PASS** |
| P4: deterministic two-image fixture; A/A-repeat/B/empty providers; coordinates, pooled component features, and global features separately compared with specified tolerances | Provider-specific raw tensors and comparison errors in JSON/NPZ | **PASS** |
| P5: masked padding output invariance and exact masked-gradient exclusion; controlled duplicate witness through final `rho` | Inputs, hand-set witness, outputs, gradients, expected-value errors in JSON/NPZ | **PASS** |

**Limitations/outstanding questions:** These are small deterministic CPU diagnostics on synthetic inputs and random (not pretrained) visual weights. They do not establish real-image feature quality, scientific validity, optimization behavior, learning, generalization, or model performance. The P5 result is specific to the hand-set witness. P7 characterizes only the requested policies/one forward; the trainable block’s BatchNorm mutation is expected for that train-mode path, while all observed frozen state remained unchanged. No architecture or scientific decision is made. CP3, CNN-01, production repair/reconstruction, detector development, training, and any broader investigation remain unauthorized and unstarted.

**Review handoff:** request a fresh independent reviewer to verify this fixed CP2 revision, reproduce the four probes as needed, and record a **CP2-only verdict** on issue #25. Leave #25 OPEN; do not merge to `main`. Stop after this handoff.
