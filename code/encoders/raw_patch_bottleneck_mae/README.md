# Raw Patch Bottleneck MAE (RPB-MAE)

RPB-MAE implements a compact raw-patch time-series encoder alongside the existing
[MTF-JEPA-MAE-VICReg encoder](../mtf_jepa_mae_vicreg/README.md). Its initial
candidate is a masked autoencoder whose reconstruction loss passes through the
served embedding. The baseline API, executable, archive reader and checkpoint
format remain separate.

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

The [RPB-v5 comparison](PAIRED_POOLING_ADVANCE.md) and
[optimization diagnostic](OPTIMIZATION_DIAGNOSTIC.md) are complete. Direct patch
pooling did not improve the fixed-budget linear primaries; additional training
improved reconstruction without improving mean linear validation accuracy.
The audited candidate is **RPB-v6 — Global bottleneck with context deletion**.
It keeps the RPB-v4 architecture and serves the same native32 vector, while
training with fewer visible coordinates and unchanged reconstruction targets.
Its [measured advance](CONTEXT_DELETION_ADVANCE.md) improves both linear accuracy
primaries with full coverage, but reconstruction worsens and promotion remains
blocked by the declared guard. Its [separate card](../../evaluation/cards/context_deletion_v1.md)
fixed the recipe before scoring. Policy-tagged checkpoints remain normal inference
checkpoints; ordinary training rejects their resume to prevent losing the
training view silently. The dedicated experiment currently starts fresh.

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
