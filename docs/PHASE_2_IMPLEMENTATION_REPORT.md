# Phase 2 implementation report

**Branch:** `implementation/phase-2-geometry-classifiers`

**Implementation base:** accepted Phase 1 commit `bf2e3080a07734c50b9c679aa812d1e1e933db91`
**Authority:** [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md), unchanged. See also [Phase 2 scope and explicit deferrals](PHASE_2_SCOPE.md).

## 1. Delivered architecture

### Shared classifier contract

`town_model.models.classifiers.PointClassifier` accepts flat float32 points `[N,d]`, preserves an explicit tuple of trainable class names, and produces a `ClassifierOutput` containing exact Boolean predictions, either logits or signed margins, optional diagnostics, and separate fixed-positive outputs. `DomainThing` is supplied as a constant `True` output for every supported entity point; it is not a trainable class row or a loss target. Only caller-declared trainable classes receive parameters; unsupported classes are not made-up negatives.

The three implementations share the class order, point convention, known-mask reducer, training/evaluation helpers, and checkpoint metadata:

- **Baseline:** one `Linear(d,C_train)` head. Logits are returned; prediction is `logit >= 0`. Training penalties are `binary_cross_entropy_with_logits(..., reduction="none")`.
- **Box:** one `BoxGeometry` per trainable class. Exact membership uses closed comparisons. Margins remain differentiable.
- **Town:** one `TownGeometry` vocabulary over all classes and flat points, with default two houses and one contained window per house. It is not assembled from independently trained binary modules.

A synthetic-only `SyntheticPointEncoder` maps raw features through a small MLP ending in `tanh`. It is independent of model families and is not an image encoder. In mapping experiments its initial state is copied byte-for-byte (verified by a state hash); copies are optimized independently afterward.

### Exact geometry and signed-distance objective

For a box `B(l,r)` in `[-1,1]^d`:

` s_B(x) = min_i { x_i-l_i, r_i-x_i } `

For a house and town:

` s_H(x) = min(s_F(x), -s_W1(x), ..., -s_Wk(x)) `

` s_T(x) = max_h s_Hh(x) `

Set membership is computed separately:

`contains_box = all(l <= x and x <= r)`

`contains_house = contains_frame AND NOT any(contains_window)`

`contains_town = any(contains_house)`

Therefore frame boundaries are included; window boundaries are excluded from that house; another house can reinclude a removed point. A zero signed margin never overrides the Boolean result. `evaluate_town_corners` independently tests the literal set operation and also accepts windows that are not contained, while the trainable Phase 2 representation is strictly contained.

Box and town training both use the same unreduced signed-distance penalty:

`L(x,y) = relu(m - (2y-1) * s(x))`, with default `m=0.1` and finite `m >= 0`.

The margin is signed coordinate clearance, not Euclidean distance. There is no smooth-min, soft union, or membership epsilon. The neural model uses ordinary BCE and its numerical loss is not compared to hinge magnitudes as though the objectives were equivalent.

### Stable parameterization and initialization

Raw corner logits are clamped to `[-12,12]` before sigmoid, keeping transforms finite and leaving representable slack at extreme raw values. Box low corners are bounded; high corners are formed from the remaining domain width and a positive minimum. Town frame width reserves a window plus clearance on both sides. A contained window is parameterized using a positive width and a position within the remaining interior slack.

The configured minimum is `1e-5`; the implementation adds a 16-float32-epsilon safety reserve to the numerical floor. Thus actual float32 widths/clearances are slightly greater than the request. This reserve is in the parameter transform only; it does not alter membership. Physical corners are never repaired by optimizer-step clipping.

Initialization is seeded, broad, and independent of targets. Initial box/frame widths are roughly 1.82–1.94 per axis before fitting within the domain. Houses have deterministic seeded offsets rather than coincident initialization. A window initially occupies about 72% of each available interior dimension, giving a nontrivial exclusion. Initial centers and widths, seed, mode, final physical corners, occupancy, house/window activity, and gradient zero fractions are persisted.

For exact `min`/`max` ties, PyTorch autograd's subgradient is used (equal minima/maxima share the reduction gradient; binary membership is nondifferentiable and never used as the training score). Inactive min/max branches and points beyond the hinge margin can have zero gradients. Extreme raw values beyond the raw-logit clamp are finite but locally saturated. These behaviors are measured/reported, not auto-reset.

### Shared supervision reduction

`reduce_supervision` receives unreduced `[N,C]` penalties, targets, and a Boolean known mask. It supports a positive-class weight vector (known negatives remain weight one) and a class/trainable mask. For instance `n` it computes:

`sum_c(mask[n,c] * weight[n,c] * penalty[n,c]) / sum_c(mask[n,c] * weight[n,c])`.

Only instances with a positive denominator enter the final instance mean. All-unknown batches produce a connected zero with zero supervised gradient. Phase 2 uses the default unit weights. Component-level image aggregation and alternate weighting-policy construction are not claimed as implemented.

## 2. Synthetic experiments and persistence

### Experiment A — fixed points

The deterministic generator makes bounded points once, uses one train/validation split and known mask for all families, and trains classifier parameters only. The `simple` fixture has linearly representable right-half and diagonal classes. The `window` fixture has a frame-minus-closed-window class and a disconnected two-lobe class. Labels/targets are supervision only. Resolved settings record class weighting `none`, no explicit positive-weight vector, and the shared known-trainable-class per-instance reducer.

### Experiment B — trainable point mapping

Raw synthetic features enter a shared-architecture, `tanh`-bounded point encoder. A private deterministic teacher transform generates fixture targets, but teacher points/labels are not model inputs. The same features, labels, masks, splits, and class order are used for all families. Each family gets an independently optimized encoder copy with identical initial weights.

### Run artifacts

Each experiment creates a group under `artifacts/phase2/runs/<group>/` with a shared `fixture.npz`/`fixture.json`, common resolved config, status and comparison files, then one subdirectory per model family. The fixture manifest stores a SHA-256 over all feature, target, known-mask, and split arrays; loading rejects a mutated fixture. Each model run includes:

- `resolved_config.json`, `environment.json`, `metrics.json`, `history.jsonl`;
- `initial_state.pt`, `checkpoint.pt` (model, optional encoder, initial state, optimizer state, config/fixture/class identity);
- initial and interval/final model/point/prediction/margin snapshots.

JSON and checkpoint writes use temporary-file replacement. `town evaluate` reloads a checkpoint and reproduced the saved validation loss exactly in the verification runs. `town resume` restores Adam after validating the resolved-config hash, class order, and fixture identity; it appends steps/history/snapshots without modifying the resolved configuration.

## 3. CLI and dashboard changes

Existing Phase 1 `generate`, `validate`, `inspect`, and `dashboard` commands remain available. New commands:

```bash
town experiment fixed --output artifacts/phase2/runs --seed 42 --points 320 --steps 60
town experiment mapping --output artifacts/phase2/runs --seed 42 --points 320 --steps 60
town evaluate artifacts/phase2/runs/<group>/town --split validation
town inspect-run artifacts/phase2/runs/<group>/town
town export-geometry artifacts/phase2/runs/<group>/town --output geometry.json
town resume artifacts/phase2/runs/<group>/town --additional-steps 10
```

`fixed` trains all three models on both fixed fixtures. `mapping` trains all three models on the selected mapping fixture (default `window`). Choose a fresh output root or configuration; completed group paths are immutable and are not overwritten.

The existing Streamlit app now includes **Geometry Laboratory** without replacing Phase 1 dataset overview, image explorer, or dataset generation. Controls select a saved run group, family, class, split, sample, snapshot, anchor, and (for `d>2`) plotted dimensions. It loads persisted model state and never trains on widget changes.

- `d=2`: evaluates actual membership on a grid, draws physical frame/window boundaries, shows train/validation positive/negative/unknown points, and draws the neural zero-logit contour from actual logits.
- `d>2`: evaluates the full model at every grid point while fixing all other coordinates to zero or the selected sample embedding. Empty slices are explicitly reported. Point markers are labeled projections and are not used to imply high-dimensional containment.
- Side-by-side views keep independently learned coordinate systems separate. History panels distinguish train/validation and each model's objective. Tables show validation metrics, support, parameters, gradient/activity diagnostics. Snapshot playback plots actual point/geometry parameter motion.

Launch with `town dashboard`; the default run root is `artifacts/phase2/runs`. For a custom root, use the **Synthetic run directory** sidebar field.

## 4. Verification and reproduced outcomes

### Test and environment verification

Environment used for implementation and experiments: Python 3.12.0, PyTorch `2.5.1+cpu`, CPU-only (`torch.cuda.is_available() == False`), NumPy 2.5.4, matplotlib 3.11.2, Streamlit 1.66.1, pandas 3.1.0, scikit-learn 1.9.1.

Commands:

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src tests benchmarks
.venv/bin/python -m pip check
```

Result: **120 passed, 0 failed** (including the accepted Phase 1 regression suite), `compileall` passed, and `pip check` reported no broken requirements. Four Streamlit/Pandas deprecation warnings remain; no test was skipped.

Tests cover exact box/frame/window boundaries, overlapping-house reinclusion, multiple classes/windows, signed margins and zero-margin membership, output shapes/dtypes, empty batches and invalid inputs, extreme raw float32 values, width/clearance bounds, seeded initialization, chunked output/gradient parity, loss masks/weights/all-unknown handling, point/box/frame/window/encoder gradients, high-dimensional finite gradients, paired experiments, checkpoint reload, fixture-content hash rejection, matched mapping initialization, Adam resume/config rejection, CLI, snapshot loading, dashboard geometry/comparison data, invalid runs, and Streamlit Geometry Laboratory controls. Phase 1 dataset/dashboard tests also pass.

### Saved sample runs (diagnostic, seed 42)

Reproduction settings: 320 points, 224 train / 96 validation, 60 full-batch Adam steps, learning rate 0.03, `m=0.1`, 4% unknown-label probability, snapshot interval 10, `d=2`. Group identifiers produced:

- `fixed-simple-seed42-d2-b4553830`, fixture `17479536fdf85613`;
- `fixed-window-seed42-d2-b4553830`, fixture `d57992fb853e1e15`;
- `mapping-window-seed42-d2-b4553830`, fixture `d72808581ad04e5b`.

Known/positive/negative/unknown observation counts were identical across models within each fixture:

| Fixture / split | Known | Positive | Negative | Unknown |
|---|---:|---:|---:|---:|
| fixed simple / train | 431 | 193 | 238 | 17 |
| fixed simple / validation | 184 | 88 | 96 | 8 |
| fixed window / train | 431 | 175 | 256 | 17 |
| fixed window / validation | 184 | 93 | 91 | 8 |
| mapping window / train | 433 | 191 | 242 | 15 |
| mapping window / validation | 189 | 67 | 122 | 3 |

Validation results (loss objective is stated explicitly; BCE and hinge magnitudes are not cross-compared):

| Experiment | Family / objective | Train loss | Validation loss | Validation known-label accuracy | Validation macro-F1 |
|---|---|---:|---:|---:|---:|
| fixed simple | baseline / BCE | 0.3192 | 0.3307 | 0.9239 | 0.9134 |
| fixed simple | box / signed distance | 0.1173 | 0.1133 | 0.6304 | 0.6611 |
| fixed simple | town / signed distance | 0.0426 | 0.0472 | 0.8043 | 0.8203 |
| fixed window | baseline / BCE | 0.6393 | 0.6762 | 0.6522 | 0.3974 |
| fixed window | box / signed distance | 0.1292 | 0.0980 | 0.7337 | 0.7535 |
| fixed window | town / signed distance | 0.0296 | 0.0332 | 0.9022 | 0.8912 |
| mapping window | baseline / BCE | 0.4210 | 0.4708 | 0.7937 | 0.5636 |
| mapping window | box / signed distance | 0.0645 | 0.0659 | 0.6825 | 0.3336 |
| mapping window | town / signed distance | 0.0352 | 0.0504 | 0.7831 | 0.6688 |

Counts of trainable classifier-plus-encoder parameters were 6/8/32 for the fixed baseline/box/town and 120/122/146 for mapping. The mapping experiment persisted the same initial encoder SHA-256 for all families (`9f92b95ef75a24cd0d0b4e731a15476076fd75585e9ec0d2caea20f595d3993c`). Final-step gradient norms were finite and nonzero for encoder and relevant geometry groups; e.g. mapping encoder norms were 0.00852/0.00707/0.01650 for baseline/box/town, box geometry 0.00462, and town frames/windows 0.00135/0.00446. All town windows had at least one observed window-membership point in these runs; some parameters still had zero-gradient fractions from inactive min/max/hinge branches.

These results are outcomes of one small synthetic seed, not a predetermined claim. The baseline was competitive on the simple fixture; the town had the strongest metrics on the nonconvex fixed fixture; the box/mapping result illustrates that optimization outcomes are not guaranteed by representational capacity. Raw objective magnitudes are not comparable across BCE and hinge. None of these results establishes image generalization or resolves a thesis research question.

### Dashboard/run reload verification

The Geometry Laboratory was exercised through Streamlit `AppTest` against generated saved groups: model/class/split/snapshot controls loaded, selected-point membership and score were shown, and all four tabs (selected model, three-model comparison, learning/activity, motion) were present. Pure visualization tests verify side-by-side figures, actual baseline logits, box/town state, high-dimensional sample anchors, history, motion, and invalid/missing-run handling. The actual saved run groups were also reloaded through the checkpoint evaluator and geometry viewer without training. A live `town dashboard --headless --port 8765` server returned `ok` from Streamlit's health endpoint.

## 5. Limitations and handoff

- No CNN, detector, ROIAlign, pizza image encoder, component/image-level training, full composition architecture, or actual Pizzaïolo experiment was implemented.
- Only `signed_distance` is implemented for geometric models. The deferred signed BCE and reference losses remain deferred; the reference town loss was not renamed or approximated.
- Only contained trainable windows are implemented. Window/houses are fixed-capacity and always active; diagnostic inactivity does not create/delete regions.
- The reduction primitive handles one flat instance batch. Component-instance-to-image aggregation and the eventual two-level pizza/component objectives remain future work.
- Synthetic runs have train/validation splits only. There is no final test protocol, hyperparameter search, dashboard subprocess launcher/cancel, experiment queue, or handling that converts an abruptly interrupted CLI group from `running` to `interrupted` automatically.
- The dashboard defaults to train/validation and exposes no test split. No PCA/UMAP view or ontology relation proof/diagnostic is claimed.
- Small CPU experiments do not establish pizza-image generalization. A geometric town has more capacity than the linear baseline; their parameter counts and architecture are reported, not capacity-matched.

No unresolved blocker remains for the bounded Phase 2 completion criterion. The items above are deliberate deferrals and limitations, not completed Phase 3 requirements. The implementation branch is intentionally not merged into `main`.
