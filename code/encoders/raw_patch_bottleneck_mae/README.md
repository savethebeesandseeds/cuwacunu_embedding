# Raw Patch Bottleneck MAE (RPB-MAE)

RPB-MAE implements a compact raw-patch time-series encoder alongside the existing
[MTF-JEPA-MAE-VICReg encoder](../mtf_jepa_mae_vicreg/README.md). Its initial
candidate is a masked autoencoder whose reconstruction loss passes through the
served embedding. The baseline API, executable, archive reader and checkpoint
format remain separate.

The latest [RPB-v13 experiment](VISIBLE_DIFFERENCE_DIAGNOSTIC.md) adds currently
visible adjacent differences to early mixing on **TEMPO-3**, complexity **4/5**.
Five new CUDA runs retain native32, .15 context deletion, batch 8, 512 updates and
the fixed heads. Mean TRAIN/VALIDATION query MAE improves to 0.622320 / 0.664399,
but intact linear/neural accuracy is 53.28% / 52.03%; under extra 30% deletion it
is 52.81% / 50.52%, all at 100% coverage. This is not a consistent improvement
over the reused **RPB-v10.alt-05** control. Initial common state and all 15
initial exports matched the saved early model exactly; the added 3,072 values
raise capacity to 228,877. The [saved-TRAIN diagnosis](STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md)
is also complete. The [next comparison](../../../doc/CONTINUATION_2026-10-10_AFTER_VISIBLE_DIFFERENCE.md)
will test more independent TRAIN sources with a shared fresh validation set.
Original saved **RPB-v7**, formal **RPB-v4**, TEMPO-1 **RPB-v10.alt-03** and all
historical evidence remain protected. No promotion; gain **RPB-v11** and pooled
**RPB-v12** remain stopped. The frozen [card](../../evaluation/cards/visible_difference_v1.md)
and separately preserved repaired reader document this measured variation.

**Active research: RPB-v4 — Learned global bottleneck**, through
[learned_global.conf](config/learned_global.conf). Its native 32-number global
export is the sole reconstruction signal. The completed
[native curve advance](NATIVE_CURVE_ADVANCE.md) gives fresh timing TEST means of
94.01% linear-head and 96.09% neural-head accuracy, versus 51.82% / 79.51% for
standalone raw PCA32, all at 100% coverage. Native linear VALIDATION selected 512
updates before TEST; 2048 improved reconstruction but reduced classification
validation accuracy. Consistency and moderate missingness remain measured gaps.
The [native baseline comparison](NATIVE_BASELINE_COMPARISON.md) preserves the
separate archived TRAIN/VALIDATION milestone. Use the
[version registry](../../../doc/EMBEDDING_VERSIONS.md),
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md), and
[next advance plan](NEXT_ADVANCE.md). Primary comparisons apply no PCA
after an encoder; PCA remains a standalone raw-data baseline. Historical drivers
and the validation chronology below retain their original recipes. Mode-0
loader/configuration defaults remain compatible, and this research designation
does not assert consumer acceptance.

The [current research journal](../../../doc/RESEARCH_SESSION_2026-10-09.md)
tracks the working RPB-v7 direction and preserved original five-instance group.
The completed [amplitude transfer diagnostic](FROZEN_AMPLITUDE_TRANSFER_DIAGNOSTIC.md)
uses unchanged v4.alt-01/v7.alt-01 encoders, original timing scalers and new
amplitude classifiers. A feature-only CUDA adapter exposes the shared evaluation
contract without retraining or reconstruction callbacks. The frozen
[RPB-v10 card](../../evaluation/cards/early_mixer_reliability_v1.md) moves the
existing aligned mixer before temporal blocks while retaining an independent
local path, native32 and all225,805 parameters. It compares a separate fresh
RPB-v7.alt-02 group at the same training budget. New checkpoints carry an explicit
architecture identity; ordinary defaults and historical loaders remain intact.
The [verified diagnostic](EARLY_MIXER_RELIABILITY_DIAGNOSTIC.md) improves mean
and worst-cohort timing accuracy, but two timing cohorts and amplitude deletion
accuracy worsen. V10 remains experimental. The completed
[paired learning curve](EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md) uses fresh
RPB-v7.alt-03/RPB-v10.alt-01 groups, retaining every 0/512/1024/2048 point.
At 2048, late/early timing linear means are 92.1875%/92.65625% intact and
87.8125%/91.71875% under additional deletion, all at 100% coverage. Reconstruction
improves in both groups as training continues, but timing classification does
not improve monotonically; early mixing also loses amplitude-transfer quality.
The audit passed 141,317,815 checks. Fixed heads, one-time quality exports and
all five pairs remain preserved, with no best-point selection or promotion.
Existing saved groups remain unchanged.

The completed [matched-target gain diagnostic](MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
compares fresh RPB-v10.alt-02/RPB-v11 at fixed 512. Timing linear means fall
from 96.40625% to 95.3125% intact and 92.5% to 84.6875% under extra deletion;
all coverage remains 100%. All six numeric guards fail and all five original-query
MAEs worsen. The gain recipe is stopped without range/rate/budget/head rescue.
The completed [pooled-context diagnostic](POOLED_CONTEXT_DIAGNOSTIC.md)
compares the compact early RPB-v10.alt-03 with RPB-v12 at the same 512 budget.
Compact/pooled timing linear means are 98.28125%/90.9375% intact and
97.03125%/83.125% under additional deletion; every paired cohort loses in both
views. Both mean reconstruction guards pass, all four timing guards fail, and
coverage stays 100%. Strong separate untrained amplitude controls and all four
seven-method panels remain explicit. Route and first-layer capacity change
together; this does not identify an isolated projection-loss cause. The sole
audit passed 96,170,207 checks. Stop wider pooled width and gain without rescue.
The fresh fixed 512 compact early-v10 confirmation selected by this milestone
is now complete as linked above. The [next direction](NEXT_ADVANCE.md) keeps
both designs for the separately engineered harder timing benchmark, with no
harder encoder result or promotion claimed. See
the [historical pooled continuation](../../../doc/CONTINUATION_2026-10-09_AFTER_POOLED_CONTEXT.md)
and [portable source tools](../../evaluation/tools/pooled_context_v1/README.md).

The [RPB-v5 comparison](PAIRED_POOLING_ADVANCE.md) and
[optimization diagnostic](OPTIMIZATION_DIAGNOSTIC.md) are complete. Direct patch
pooling did not improve the fixed-budget linear primaries; additional training
improved reconstruction without improving mean linear validation accuracy.
The measured candidate is **RPB-v6 — Global bottleneck with context deletion**.
It keeps the RPB-v4 architecture and serves the same native32 vector, while
training with fewer visible coordinates and unchanged reconstruction targets.
Its original audited [measured advance](CONTEXT_DELETION_ADVANCE.md) improves both linear accuracy
primaries with full coverage, but reconstruction worsens and promotion remains
blocked by the declared guard. Its [separate card](../../evaluation/cards/context_deletion_v1.md)
fixed the recipe before scoring. Policy-tagged checkpoints remain normal inference
checkpoints; ordinary training rejects their resume to prevent losing the
training view silently. The dedicated experiment currently starts fresh.

Its completed [context optimization diagnostic](CONTEXT_OPTIMIZATION_DIAGNOSTIC.md)
replays the audited 512 state and continues the same training policy to 1024
and 2048 on TRAIN and known VALIDATION only. No measured budget both reaches
the v4 fixed-512 reconstruction reference and preserves v6's 512-update linear
accuracy in intact and fixed-deletion validation views. At 2048, mean
TRAIN/VALIDATION MAE is 0.035736/0.035845, while linear accuracy falls to
96.875% intact and 95.3125% with additional 30% deletion. No TEST was opened and
RPB-v4 remains active.

The completed [five-master replication](CONTEXT_REPLICATION_ADVANCE.md) pairs
fresh v4/v6 training at 512 updates on 4404/5505/6606/7707/8808. Mean linear gains
are +1.71875 percentage points intact and +8.125 under additional 30% coordinate
deletion, with equal 100% coverage. Mean TRAIN and VALIDATION reconstruction
guards still fail. Independent archive audit v3 passed 103,787,020 checks;
failed v1/v2 reader attempts and their corrections remain preserved. This
repeats the accuracy/reconstruction tradeoff rather
than promoting v6.

**RPB-v7 — Global bottleneck with lighter context deletion** keeps the same
mode2/mixer1/native32 architecture, optimizer, loss and heads while reducing the
extra TRAIN request to 0.15 at 512 updates. The completed
[five-master validation diagnostic](CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md)
improves mean linear accuracy over v6 by 2.65625 points intact and 1.25 with
additional deletion, at 100% coverage. Mean reconstruction improves over v6,
but remains worse than v4 in every master. Neural means decline, and master
8808 loses deletion accuracy. Both reconstruction guards fail; v7 is not
promoted and v4 remains active. Independent archive audit v3 passed 67,097,380 checks.

The frozen [lighter-policy card](../../evaluation/cards/context_lighter_validation_v1.md)
and explicit `rpb-training-context-deletion-015-v1` identity govern these known
TRAIN/VALIDATION results. The encoder owns the training recipe; shared evaluation
owns fixed fits, views, support and scores. CUDA admission also passes exact
live-trainer continuation across saved snapshots and unchanged v6 default/replay.
No TEST/stress or checkpoint selection occurred. Save the measured milestone
and passed audit before the next bounded training-view change in the
[next plan](NEXT_ADVANCE.md).

**RPB-v8 — Global bottleneck with balanced context views** is implemented and
measured under the frozen [balanced-view card](../../evaluation/cards/context_balanced_validation_v1.md).
It alternates 256 ordinary and 256 deletion-0.30 attempts at 512 completed updates,
with unchanged mode2/mixer1/native32 architecture, original targets/loss and heads.
The [five-master diagnostic](CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md) reports
mean native linear accuracy of 95.93750% intact and
91.71875% under the exact saved v7 additional-deletion view,
at 100% coverage. Mean TRAIN/VALIDATION MAE is
0.085022/0.087620.
The predeclared joint development guard **failed**. Independent archive audit v2
passed 67,099,959 checks; actual CUDA admission and prior-policy regressions passed.
V8 is not promoted, and RPB-v4 remains active. No TEST/stress, checkpoint selection,
head tuning or post-encoder PCA occurred. Five new v8 trajectories use ten
retained v4/v7 references; all five masters and both heads remain in the report.

The completed [TRAIN objective diagnostic](TRAINING_OBJECTIVE_DIAGNOSTIC.md)
uses fifteen retained v4/v7/v8 instances, with zero encoder updates and zero
head refits. An all-pair residual-difference loss helps the local fixed-Ridge
gradient direction on two v7 masters and worsens it on three, failing its fixed
rule. CUDA admission and independent saved-arithmetic audit passed; do not
train or tune that auxiliary. These are TRAIN mechanism results, with no new
quality score. The failed reader and its CPU/GPU correction remain recorded.

**RPB-v9 — Global bottleneck with native view agreement** is implemented and
measured in the [five-run diagnostic](NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md).
It adds same-row native32 agreement and a variance safeguard to ordinary
reconstruction, with unchanged inference and fixed shared heads. Actual CUDA
admission and the independent audit passed, but all six numeric guards failed:
linear means are 59.21875% intact / 57.03125% with extra deletion; mean standardized
TRAIN/VALIDATION MAE is 29.194490 / 29.371655. One run is especially unstable,
and the other four also reconstruct much worse than v4. RPB-v4 remains active.
The [prospective plan](NATIVE_VIEW_AGREEMENT_PLAN.md) and quality card remain the
frozen design record. Stop this mechanism without weight/rate/budget rescue.
The original [continuation](../../../doc/CONTINUATION_2026-10-08.md) is preserved.
Its [saved-TRAIN scale diagnosis](NATIVE_VIEW_LOSS_SCALE_TRAIN_DIAGNOSTIC.md) is
complete, with no scale-floor hits and later agreement growth in all five runs.

Current work continues from RPB-v7. The [decoder-calibration diagnosis](V7_DECODER_CALIBRATION_DIAGNOSTIC.md)
freezes its encoder/scaler and trains only the existing decoder for 128 CUDA
updates. Mean TRAIN/known-VALIDATION MAE is 0.054039/0.056989, below each paired
v4 reference, with exact unchanged native32 outputs. Cached classification is
retained without refits. Actual CUDA admission and independent arithmetic audit
passed. This creates no new embedding tag or promotion. Its
[dated continuation](../../../doc/CONTINUATION_2026-10-08_AFTER_DECODER_CALIBRATION.md)
preserves the prospective equal-budget plan.

The completed [fresh v4/v7 replication](FRESH_DECODER_REPLICATION_DIAGNOSTIC.md)
uses five new paired TRAIN/VALIDATION cohorts, encoder512 and frozen decoder128
for both policies. V7/v4 linear means are 92.96875%/92.34375% intact and
89.84375%/82.03125% under extra deletion, with 100% coverage. V7's intact
worst-master score is lower and its intact mean effect interval crosses zero.
The equal-decoder MAE guards also fail, although recovery against pre-calibration
v4 passes. Independent saved-evidence audit passed 67,932,331 checks. Native
exports remain exact across calibration; classification fits and scores are
reused afterward. No new tag, TEST/stress access or promotion. The
[historical continuation](../../../doc/CONTINUATION_2026-10-08_AFTER_FRESH_DECODER_REPLICATION.md)
prescribed saved TRAIN reliability across all ten instances, now completed before
the later comparisons above. Generalized frozen-decoder calibration stays in this encoder; the new
tensor-only fixed-head evaluator stays in shared code.

The [embedding evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md)
governs shared protocols, reporting and acceptance. The
[architecture and evaluation specification](RPB_MAE_architecture_and_evaluation_spec.md)
states the research hypotheses; the
[implementation guidelines](RPB_MAE_implementation_guidelines.md) define tensor,
support, precision and extension contracts.

**Validation chronology:** core/workflow checks and archived protocol-v1 comparisons
are recorded in [validation evidence](IMPLEMENTATION_STATUS.md). The feature
evaluator refactor passed its container isolation, adapter, repeated-card and
regression checks; see the [shared review](../../../doc/EVALUATION_IMPLEMENTATION_REVIEW.md).
The v1 scores do not establish protocol-v2 results, and the short archived
training run did not establish an advantage over its untrained control.
The [held-out reconstruction track](DECODER_RELIANCE.md) passed container checks
and completed 12 runs with matched 128-update model/control budgets. Every run
beat metadata-only decoding and degraded under source-group shuffling, with
positive paired intervals. This supplies synthetic decoder reliance evidence;
it does not establish frozen-feature quality or consumer acceptance.
The [stronger frozen-probe comparison](FROZEN_PROBE_ADVANCE.md) completed 12
development runs with 128 updates. Compact lag-sign ridge gains remain
inconclusive across the three seeds; native channel concatenation exposes lag
information to the fixed nonlinear probe. These results motivate a separate
contextual channel-mixer ablation and do not promote the encoder.
The [completed mixer ablation](CHANNEL_MIXER_ADVANCE.md) improves compact lag-sign
ridge accuracy to 90.63%, 76.56% and 63.28% versus the same independent control's
52.34%, 48.44% and 52.34%. All three nominal within-run paired intervals favor
the mixer. Tests cover both modes, legacy checkpoints and CPU/CUDA CLI operation;
nonlinear results vary, and confirmation, costs and consumer acceptance remain open.
The [five fresh seeds and fixed-readout sweep](FRESH_SEEDS_AND_STRESS.md) add
development replication and shared missingness evaluation. Compact lag ridge
means are 79.69% versus 48.75%, with four positive intervals and one crossing
zero; heavy corruption erodes the benefit. Both encoders pass the same shared path.
The [GPU learning curve](LEARNING_CURVE.md) completes continuous paths at
128/512/2,048 updates for three new masters. Validation selects 128 updates;
fresh compact lag ridge means are 72.92% versus 51.82%. Longer training improves
reconstruction and some native-width nonlinear probes, while compact/global lag
readouts remain limited. This is development evidence, not consumer acceptance.
The [compression and global bottleneck experiments](GLOBAL_BOTTLENECK_ADVANCE.md)
add optional mean/learned global reconstruction modes and independent shared
diagnostics. The first fresh comparison improves learned-global lag readouts,
with uneven seed evidence and worse reconstruction. The completed fixed-budget
source matrix reaches 100% native-global / 98.70% PCA12 nonlinear lag accuracy
for learned pooling with 128 training pairs. PCA12 linear accuracy remains 76.56%.
Matched-size probe refits show that additional encoder sources alone do not
uniformly explain the gain. The record retains all comparisons and tradeoffs;
the learned configuration stays opt-in.

## Implemented model

The smoke configuration uses `[B,C,32,F]` histories, non-overlapping eight-step
patches, encoder width 64, three pre-normalized temporal attention blocks with
four heads, and export width 32. Values and visibility indicators preserve the
time-major `[P,F]` order. Visible patches retain their original patch indices,
so removing a target does not compress its time coordinates.

The temporal encoder shares weights across channels and processes each channel
independently. Position-aware pooling produces `z_local [B,C,D]`; an equal mean
over valid local channel vectors produces `z_global [B,D]`. Outputs include
validity, visible observation/patch counts and semantic channel IDs. An absent
channel has a finite zero vector and false validity; an entirely absent sample
also has a zero/invalid global vector.

With the default `global_bottleneck_mode=0`, a small two-layer decoder receives
only `z_local` and target position/channel metadata. It reconstructs hidden
observed cells with hierarchical masked Huber:
average within each eligible channel, then within each eligible example, then
across eligible examples. Training requires two visible observed patches and one
observed target patch per eligible channel. Inference can serve a channel with
one observed patch and reports its support.

An optional aligned channel mixer produces separately named `z_contextual` and
`z_contextual_global` exports. It attends across observed channels at original
patch positions, then uses the shared temporal pooling/export projection.
Set `channel_mixer_layers=1` through [channel_mixer.conf](config/channel_mixer.conf).
The default is zero, preserving the independent model's modules and random
initialization. Missing channels remain zero/invalid; contextual outputs do not
infer support from peers. Reconstruction trains through the exact contextual
export when enabled. The pre-mixer local branch retains independent computation
but shares the contextually trained parameters and has no separate loss.
See [the architecture ablation](CHANNEL_MIXER_ADVANCE.md) for lessons from the
first encoder, contracts and the fixed comparison.

Optional `global_bottleneck_mode=1` uses the valid-channel mean as the sole
reconstruction vector. Mode 2 learns a semantic-ordered, support-aware global
pooling MLP. Both decode only from the exact served `[B,D]` global export and
fixed target metadata. Local/contextual channel exports remain distinct
diagnostic surfaces. The mixer configurations are
[mean_global.conf](config/mean_global.conf) and
[learned_global.conf](config/learned_global.conf); defaults and mode-0 checkpoints
retain their previous behavior. See the
[global bottleneck experiments](GLOBAL_BOTTLENECK_ADVANCE.md).

Multiscale/frequency branches, forecasting, JEPA/EMA, VICReg, codebooks and
specialist banks remain future experiments. The local surface has different
semantics from the baseline's contextualized per-channel output, even when their
tensor shapes match.

## Source layout

Headers use the namespace `embedding::encoders::raw_patch_bottleneck_mae` and
include prefix `embedding/encoders/raw_patch_bottleneck_mae/`.

```text
code/encoders/raw_patch_bottleneck_mae/
  include/embedding/encoders/raw_patch_bottleneck_mae/
    config.h types.h preprocessing.h tokenization.h masking.h
    encoder.h model.h objectives.h workflow.h training_utils.h
    evaluation_adapter.h reconstruction_adapter.h learning_curve_adapter.h
  src/
    main.cpp workflow.cpp evaluation_adapter.cpp reconstruction_adapter.cpp
    learning_curve_adapter.cpp
  config/
    smoke.conf evaluation.conf channel_mixer.conf mean_global.conf learned_global.conf
  tests/
    preprocessing_test.cpp numerics_test.cpp masking_test.cpp model_test.cpp
    workflow_test.cpp evaluation_adapter_test.cpp reconstruction_adapter_test.cpp
    learning_curve_adapter_test.cpp legacy_checkpoint_parity_test.cpp
    cli_smoke.sh periodic_checkpoint.sh
```

The model and preprocessing core are header-only. Workflow code owns raw/scaler
archives, checkpoints and CLI operations. Its evaluation adapter implements the
[shared provider interface](../../shared/include/embedding/shared/feature_evaluation.h).
Comparison orchestration and CLI dispatch live in the shared engine and
[independent evaluation integration](../../evaluation/README.md); the standalone
minimum harness does not link the RPB-MAE model.
The reconstruction adapter supplies exact local exports, raw predictions and a
separately fitted mask/metadata control through the
[shared reconstruction interface](../../shared/include/embedding/shared/reconstruction_evaluation.h).
Shared code owns target masks, export interventions, metric scaling and reports;
the integration CLI owns the separate `reconstruct` command.
The CUDA learning-curve adapter supplies a continuous trainer and immutable
checkpoint snapshots through the
[shared learning-curve interface](../../shared/include/embedding/shared/learning_curve.h).
Its separate `embedding_learning_curve` integration owns the development card,
GPU gate and final testing. It does not change existing adapters' CPU behavior.
The shared `projection_diagnostic` fits CPU readouts on explicit archived
training/validation features without encoder linkage or retraining. The shared
`global_bottleneck_experiment` compares continuous CUDA trainers and immutable
snapshots through `embedding_global_bottleneck`. Their integration sources are
`code/evaluation/src/projection_diagnostic_main.cpp` and
`code/evaluation/src/global_bottleneck_main.cpp`.
Shared utilities live in [code/shared](../../shared/README.md), with include
directory `code/shared/include` and include prefix `embedding/shared/`.

## Container build and checks

Use the existing managed development container and the
[environment workflow](../../../doc/ENVIRONMENT.md). From the project root on
Windows, build and run through the named `rpb-mae` session:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', '-j2', 'rpb-mae', 'evaluation', 'feature-harness', 'test-rpb-mae', 'test-feature-harness')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', 'smoke-rpb-mae')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', 'smoke-rpb-mae-cuda')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', 'periodic-rpb-mae')
```

The corresponding commands in a container shell at `/embedding` are:

```bash
bash code/scripts/task.sh rpb-mae -j2 rpb-mae evaluation feature-harness test-rpb-mae test-feature-harness
bash code/scripts/task.sh rpb-mae smoke-rpb-mae
bash code/scripts/task.sh rpb-mae smoke-rpb-mae-cuda
bash code/scripts/task.sh rpb-mae periodic-rpb-mae
```

`test-rpb-core` runs only the preprocessing, numerical, masking and model tests.
`test-rpb-mae` also runs workflow tests. The periodic test interrupts a real
training process after an atomic checkpoint appears, then resumes and exports
embeddings. The CUDA smoke also loads the GPU-trained checkpoint on CPU.
`test-rpb-feature-adapter` checks actual training seeds/settings, separate
untrained/scaler provenance, frozen assets and held-out extraction isolation.
Model, workflow, feature and reconstruction adapter tests cover mixer-off and
mixer-on behavior.
`RPB_CONFIG` selects an explicit configuration for the CLI smoke script.
Build the separate duration diagnostic with `learning-curve`; its shared and CUDA
adapter checks are `test-learning-curve` and `test-rpb-learning-curve`. The
[learning-curve record](LEARNING_CURVE.md) retains the actual GPU-gate and run
invocations. Complete the gate before launching the full experiment.
`projection-diagnostic` and `global-bottleneck` build the separate diagnostic
executables; `test-projection-diagnostic` and `test-global-bottleneck` check their
shared protocols. `test-rpb-legacy-checkpoint` loads six archived mode-0
checkpoints and compares 24 exact training/validation feature surfaces, scaler
and settings identities, and preserved artifact bytes. CUDA continuation checks
cover mixer-off/on and all three global modes.

This session produces
`/opt/cuwacunu_embedding/build/rpb-mae/embedding_raw_patch_bottleneck_mae`,
`/opt/cuwacunu_embedding/build/rpb-mae/embedding_evaluate` and
`/opt/cuwacunu_embedding/build/rpb-mae/feature_harness`. Objects and temporary
files stay inside the container. Smoke/evaluation runs receive unique directories
under `output/runs/rpb-mae/`. Use a separate session such as `baseline` for the
existing encoder. Session locks protect runner operations; coordinate direct
binary use, shared source edits and GPU capacity separately.

## Raw data and preprocessing

The versioned raw archive stores float32 or float64 `data [B,C,H,F]`, boolean
`observed` of the same shape, unique semantic `channel_ids [C]` or `[B,C]`,
float64 `endpoints [B]`, a positive uniform sampling interval and feature units.
It carries an encoder tag, artifact kind, schema identity and dataset identity.
The shared legacy float32 archive is preserved and is not accepted implicitly.
Raw dtype, dimensions, units, channel schema and interval contribute to the
schema identity; a different source-precision schema requires an explicit adapter.

`prepare` fits a frozen scaler from its supplied training archive. Statistics
are stored per semantic channel and feature in float64, including observation
counts and applied scale floors. Centering/scaling happens before conversion to
float32 model coordinates. Already quantized float32 observations cannot recover
lost variations. Naturally absent stored values are safely removed before
arithmetic; observed NaN/Inf or overflow is rejected.

Archives contain histories already cropped for each endpoint. The core C++
`prepare_endpoint_history` helper separately accepts a longer raw stream,
timestamps and endpoints, validates uniform ordered sampling, and selects the
last H observations ending at an on-grid endpoint. It rejects off-grid endpoints.
The CLI `prepare` command fits a scaler; it does not crop a longer stream.
Bidirectional attention inside a cropped history serves endpoint embeddings.
Per-timestep causal sequence outputs are outside the current API.

## CLI training and export

After the build finishes, run the following in the container at `/embedding`.
The example allocates a new directory and keeps the original training archive:

```bash
rpb_bin=/opt/cuwacunu_embedding/build/rpb-mae/embedding_raw_patch_bottleneck_mae
rpb_config=code/encoders/raw_patch_bottleneck_mae/config/smoke.conf
mkdir -p output/runs/rpb-mae
rpb_run="$(mktemp -d "$PWD/output/runs/rpb-mae/manual-XXXXXX")"
"$rpb_bin" synthetic --config "$rpb_config" --output "$rpb_run/raw.pt" --samples 32
"$rpb_bin" prepare --config "$rpb_config" --input "$rpb_run/raw.pt" --output "$rpb_run/scaler.pt"
"$rpb_bin" train --config "$rpb_config" --input "$rpb_run/raw.pt" --scaler "$rpb_run/scaler.pt" \
  --checkpoint "$rpb_run/model.pt" --steps 8 --attempt-limit 1000 --checkpoint-every 4 --device cpu
"$rpb_bin" train --resume "$rpb_run/model.pt" --input "$rpb_run/raw.pt" \
  --checkpoint "$rpb_run/resumed.pt" --steps 2 --device cpu
"$rpb_bin" embed --checkpoint "$rpb_run/resumed.pt" --input "$rpb_run/raw.pt" \
  --output "$rpb_run/embeddings.pt" --batch-size 4 --device cpu
```

`synthetic` creates a deterministic float64 fixture and accepts `--seed`,
`--samples` and comma-separated `--units`. Training without `--input` uses a fixed
32-example fixture. Without `--scaler`, training fits its scaler on that training
dataset. `scaler-fit` is an alias for `prepare`.

`--steps` means additional completed optimizer updates. Every attempted batch
advances the absolute attempt counter and its deterministic row/mask/dropout seed
policy. An ineligible batch performs no optimizer update or weight decay.
`--attempt-limit` bounds additional attempts; exhaustion saves a recovery
checkpoint and returns an error. `--checkpoint-every` counts attempts, including
skips. CPU continuation targets exact parity under unchanged execution settings.
Resume requires the same dataset identity and training batch size; configuration,
seed and scaler come from the checkpoint. `--device cpu|cuda` selects execution.

Checkpoints contain explicit encoder/format/output-semantic IDs, canonical
settings, model and AdamW state, frozen scaler, schema/dataset identities and both
counters. Scaler-fit dataset provenance is recorded separately from the model's
pretraining dataset. Compile-time source fingerprints and Git metadata accompany
the state. Baseline checkpoints are rejected before loading model weights.
Periodic saves replace the destination atomically; CLI output paths cannot alias
their input files, including hardlinks.

Exports store `z_local`, `z_global`, validity masks, visible observation/patch
counts, expanded channel IDs, endpoints and interval. They retain schema,
preprocessing and data identities, plus distinct checkpoint-producing and inference
source fingerprints. Mixer checkpoints also export separately tagged observed
contextual vectors. Global modes carry their reconstruction-path semantic tags;
missing channels retain zero/invalid local and contextual vectors.

## Shared feature evaluation

The encoder-owned `evaluation_adapter.h/.cpp` maps exact RPB-MAE model/scaler
state to the [shared provider contract](../../shared/include/embedding/shared/feature_evaluation.h).
Its fit callback receives permitted training observations and declared metadata;
it receives no labels, hidden clean signals or held-out observations. The
[shared engine and integration guide](../../evaluation/README.md) owns cards,
splits, normalization/PCA/probes, paired populations, scores and reports. See
[the policy review](../../../doc/EVALUATION_IMPLEMENTATION_REVIEW.md) for remaining
scope. The ordinary RPB executable owns synthetic/prepare/train/embed and does
not link the evaluator or baseline; it has no evaluation command. Run comparisons
through `embedding_evaluate`.

The `evaluation` target builds that executable with baseline and RPB adapters.
`feature-harness` builds a minimum executable with the baseline adapter only;
`test-feature-harness` also runs the shared engine's dummy-provider tests.

```bash
bash code/scripts/task.sh rpb-mae evaluation feature-harness test-feature-harness
eval_bin=/opt/cuwacunu_embedding/build/rpb-mae/embedding_evaluate
mkdir -p output/runs/rpb-mae
rpb_results="$(mktemp -d "$PWD/output/runs/rpb-mae/feature-XXXXXX")"
"$eval_bin" --output "$rpb_results/results" --encoders baseline,rpb --seeds 101 \
  --channels 3 --history 32 --features 3 --train-pairs 32 --validation-pairs 16 --test-pairs 32 \
  --baseline-steps 4 --rpb-config code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf \
  --rpb-steps 4 --compare rpb_trained_global,rpb_untrained_global,matched_global
```

The [controlled-pairs-v2 recipe](../../evaluation/cards/controlled_pairs_v2.md)
defines development evidence. Each run saves its instantiated `evaluation-card.json`
before generation/fitting; that card fixes C/H/F, IDs/units, tasks, budgets, matched
widths, seeds and explicit compared-method pairs. Encoder configuration validates
against the card rather than choosing evaluator geometry. The four current tasks
are reversal, level, amplitude and lag sign. This is not consumer acceptance.

RPB options are `--rpb-config FILE`, `--rpb-steps N` for fresh training on permitted
observations only, and `--rpb-checkpoint FILE` for frozen model/scaler extraction
without refitting. A frozen checkpoint cannot be combined with config overrides
or fresh training. Global features require any observed-valid channel;
`rpb_*_channel_concatenation` contains the active local or contextual vectors in
declared semantic order and requires every channel to be observed-valid. Global
decoding modes retain these distinct cotrained vectors as diagnostic features.

Matched tiers always apply train-fit PCA, including when native width already
equals the requested width. Insufficient valid support or numerical rank is
reported as an unsupported fit. Every declared pair has its own fixed common-valid
population; adding unrelated providers cannot change that pair. Reports retain
coverage and grouped effect uncertainty as well as conditional scores.

Raw/mask controls, descriptors and untrained/fresh-frozen baseline controls remain
available. Fresh baseline training uses only permitted training observations and
explicit identity float64-to-float32 preparation. No verified historical
checkpoint/preprocessing association is supplied by this driver.

`bash code/scripts/task.sh rpb-mae evaluate-rpb-mae` selects the separate evaluator
and allocates a unique run directory. `evaluate-minimum` selects the minimum
harness. The [archived v1 reports](IMPLEMENTATION_STATUS.md) retain their original
scores and source identity; reproducing them requires archived v1 source/scripts.
Current v2 commands produce a different protocol and cannot recreate those scores.
Refactor validation is recorded in the status record. Development decoder reliance
and synthetic missingness have been measured. Consumer confirmation on domain
data, domain outage behavior, comparative costs and promotion gates remain open.

## Fresh seeds and fixed-readout missingness

The [fresh-seed record](FRESH_SEEDS_AND_STRESS.md) completes five further
development seeds with unchanged 128-update recipes. Compact PCA36 lag ridge
accuracy averages 79.69% for the contextual mixer versus 48.75% for independent
RPB; four of five within-run intervals are positive and seed704 crosses zero.
This result concerns concatenated channel vectors; pooled-global evidence is
weaker and heavy deletion/temporal gaps erode the benefit.

Add `--stress-sweep fixed-readout-v1` to the separate feature evaluator to reuse
each encoder and its ordinary fitted normalization/PCA/probes across testing-only
missingness cases. The shared engine owns masks, source populations, coverage,
conditional/full-population scores and sidecars. It needs no encoder-specific
stress implementation. Ordinary reports/assets remain exact with stress enabled,
and the original encoder passes the same path. Full recipe, results and
validation are linked in the record. These remain synthetic development evidence.

## Held-out decoder reliance

The separate [reconstruction protocol](../../evaluation/cards/controlled_reconstruction_v1.md)
measures the bottleneck's held-out reliance on trajectory information. See
[DECODER_RELIANCE.md](DECODER_RELIANCE.md) for the complete recipe and interpretation.
It adds no mixer, objective or checkpoint format. Every task/seed trains fresh
assets using only its permitted training observations.

Build and test through a separate named session inside the managed container:

```bash
bash code/scripts/task.sh reconstruction -j2 evaluation test-reconstruction-evaluation test-rpb-reconstruction
EVALUATION_BIN=/opt/cuwacunu_embedding/build/reconstruction/embedding_evaluate \
  EMBEDDING_RUN_ROOT="$PWD/output/runs/reconstruction" \
  bash code/scripts/reconstruct-rpb-mae.sh --seeds 101 --tasks reversal \
    --train-pairs 8 --validation-pairs 4 --test-pairs 8 \
    --rpb-steps 4 --metadata-steps 4 --batch-size 8
```

This is a wiring-sized recipe. The `reconstruct-rpb-mae` Make target runs the
default development recipe through the same
unique-directory wrapper: four tasks, three seeds, 32/16/64 source pairs,
128 completed updates and minibatches of eight. Invoke it with
`bash code/scripts/task.sh reconstruction reconstruct-rpb-mae`. Direct use is
`embedding_evaluate reconstruct --output NEW_DIRECTORY`; `--help` lists card
geometry/IDs/units, patch length, source counts, tasks/seeds, bootstrap budget,
`--rpb-config`, `--rpb-steps`, `--metadata-steps` and `--batch-size`.
Omitting `--metadata-steps` matches the main update budget. This fresh-only command
accepts no checkpoint input. The ordinary RPB executable has no reconstruction
command, and the minimum harness does not register this track.

The saved `reconstruction-card.json` freezes the resolved adapter recipe before
fitting. Fully observed histories support exact cross-source swaps; all original
patch positions are enumerated as held-out targets. The shared engine compares
intact, shuffled and separately zeroed local exports, mask/metadata-only
predictions, an untrained model and the training-mean reference. It scores
training-standardized masked MAE, with Huber as secondary, and bootstraps complete
two-source exchange blocks containing paired variants and repeated patch trials.

The stronger mask/metadata control starts with identical encoder/decoder weights,
freezes the original encoder, and encodes normalized zeros with the actual
visible mask, semantic IDs and positions. It independently optimizes only
`decoder_*` parameters under the same sampled batches, masks, optimizer settings
and default update budget. Its encoder stays in evaluation mode with gradients
and dropout disabled. This control receives no signal values; it differs from
zeroing the trained decoder's input.

Each `provider-assets/` directory saves ordinary resumable `model.pt`,
versioned `training-raw.pt` and `scaler.pt`, plus distinct
`metadata-decoder.pt`, `untrained-model.pt` and `training-provenance.pt` artifacts.
The main checkpoint retains its existing format and AdamW/attempt continuation.
Its `source_fingerprint` identifies the core checkpoint writer; the companion
provenance and `provider-audit.json` separately identify the adapter training
producer with `EVALUATION_SOURCE_ID`, resolved recipe and source manifests.
That producer identity is required to identify how these weights were trained.
Metadata/untrained assets are not ordinary training checkpoints.

Callbacks encode only the supplied visible observations and decode the supplied
compact vectors into raw CPU float64 predictions. Neural decoding remains
float32; inverse scaling runs in float64. Shared evaluation owns its independent
train-fit metric scaler and retains all fixed target support. The
[measured results](DECODER_RELIANCE.md) show synthetic decoder reliance; this
development track supplies no consumer acceptance or frozen-feature quality claim.
