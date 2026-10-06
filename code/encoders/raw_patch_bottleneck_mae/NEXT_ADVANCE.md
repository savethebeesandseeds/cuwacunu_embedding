# RPB v4 evaluation and development plan

Date: 2026-10-06

Advance **RPB-v4 — Learned global bottleneck**, using its native 32-number
embedding and fixed classifier heads. PCA is a standalone raw-data baseline.
Older encoder designs remain archived references and compatibility inputs;
routine research does not retrain every historical variant.

Use [the reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md) and
[the version registry](../../../doc/EMBEDDING_VERSIONS.md). Shared evaluation
stays separate from encoder training and decoding.

## Completed milestone

No new experiment. The [saved report](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/diversity-128/report.json)
uses three synthetic development masters 1701/1802/1903. Each run has 128 training
source pairs (256 examples), 64 validation pairs (128 examples), and 64 fresh
test pairs (128 examples). Encoder training completes 512 updates at batch size
8 on the verified RTX A2000 CUDA device. The linear head is ridge with penalty
1; the neural head is tanh-16 with Adam 0.01 for 100 updates. Heads have identical
recipes and paired initialization policies; their weights fit separately.

Timing means across the three runs, with no PCA on encoder exports:

RPB-v2 uses channel mixing and channel reconstruction. RPB-v4 uses a learned
global bottleneck.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | — | — | — |
| PCA only — no encoder | 32 | — | — | — |
| RPB-v2 | 32 | 69.3 | 62.5 | 100 |
| RPB-v4 | 32 | 94.8 | 100 | 100 |

The two baseline rows are unmeasured **under these conditions**. Older raw/PCA
measurements used fewer training examples and cannot fill these cells. Analytic
raw rules score 100%, but are separately labelled solvability checks.
RPB-v4 linear scores by master are 98.44%, 99.22%, and 86.72%; its neural head
is correct on all 384 timing test examples. Three source-group conditional
intervals are not an uncertainty estimate across encoder retraining.

The [validation record](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/diversity-128/validation-report.json)
also provides fixed-query standardized MAE and synchronized encoder time:

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v2 | 512 | 0.0421 | 0.0454 | 16.4 |
| RPB-v4 | 512 | 0.0693 | 0.0730 | 15.8 |

RPB-v2 reconstruction consumes three contextual channel vectors (96 signal
numbers); RPB-v4 uses only the exact global 32-number vector. This is a different
reconstruction capacity, not two decoders reading the same bottleneck. Timing
ranges overlap and do not establish a speed advantage. Encoder and head budgets
are separate; 4096 sampled row presentations are 16 equivalent passes, not 16
guaranteed full epochs.

RPB-v4's separate direction/level runs score 100% with either native head.
Its amplitude runs score 100% linear and 99.48% neural. Each task currently trains
its own model; this does not establish one universal embedding across tasks.
Severe missingness remains a measured weakness. These are development findings,
not consumer acceptance.

## Phase 1 Establish the no encoder comparison

**Completed:** the [native baseline comparison](NATIVE_BASELINE_COMPARISON.md)
implements this shared archive protocol and reports all four tasks. Timing
validation means are 48.7%/52.8% raw, 51.3%/69.6% PCA-only, and 93.8%/99.9%
RPB-v4 (linear/neural), all with 100% coverage. Encoder checkpoints are unchanged.
The definitions below record the frozen phase-1 procedure; phase 2 is next.

The small generic archive readout protocol lives in shared evaluation.
It consumes legal raw TRAIN/VALIDATION observations and archived native feature
surfaces, labels, source IDs, and support. It does not instantiate an encoder or
retrain one. Reuse the existing training-only normalizers, ridge, tiny head,
PCA and grouped scoring primitives. Build/test/run only in the managed container.

Freeze a new card before any fitting. Use the saved diversity-128 cohorts and
RPB-v4 milestone-512 native global TRAIN/VALIDATION features for all four tasks
and three masters. Validation-only results remain development diagnosis. No
TEST archive is opened for this phase and no historical report is overwritten.

Compare these rows:

1. Raw data: 288 observed values plus 288 visibility flags, training-only scaling
   and zero filling of missing values. This is a 576-number reference.
2. PCA only — no encoder: the same legal raw representation compressed to 32
   components using training data only.
3. RPB-v4: exact saved native 32-number global export, with no projection.

Use the same declared training rows, labels, split/source manifests, fixed head
recipes and readout budgets. Pair initialization seeds for the two 32-number
methods; both have ridge 66/neural 562 fitted parameters. Raw576 has a larger
head and is reported as a larger-input reference. Freeze the observation scaler,
PCA input definition, supports, seed policy, and repetitions before scores.
Do not use hidden clean targets or labels to fit raw preprocessing/PCA.

Save fitted assets, predictions, valid/total counts, per-master paired effects,
conditional source-group intervals, and immutable input hashes. Preserve the
complete declared seed set; do not select the most favorable repetition.
Retain RPB-v2's existing milestone as a named historical comparison rather than
adding another encoder training path.

## Phase 2 Re-evaluate RPB v4 on fresh development data

Create a separate native-only training/selection protocol before another curve.
The historical global driver requires three variants, computes compressed tiers,
and selects a budget with PCA12. Changing the displayed table cannot change that
protocol. Leave its cards, executable semantics and archived results intact.

The new protocol registers RPB-v4 as its sole trained encoder. Use fresh declared
development masters/source cohorts, native global32, and the same head recipes.
This is development re-evaluation; formal consumer confirmation still requires
the separate evaluation-policy contract and acceptance thresholds.
Carry a matched untrained RPB-v4 control where the claim needs training evidence;
it is not another older architecture. Raw/PCA-only controls fit on the same legal
training data and stay fixed across checkpoint milestones.

For the timing task, freeze milestones 0/128/512/2048 and one common positive
checkpoint budget selected by mean **native32 linear validation accuracy** over
every declared master. Zero is initialization diagnosis only. An unsupported
primary fit excludes that budget; exact ties choose the smaller positive budget.
Use no PCA after the encoder, including in selection. Other task scope and
fixed sanity-check budgets are declared in the card before generation.

Verify actual CUDA weights, inputs, loss, finite gradients, changed weights,
and exact checkpoint/export reconstruction before long training. Preserve one
continuous optimizer and immutable milestone checkpoints. Persist selection
before generating independent fresh test sources. Restore the selected model
and fitted readouts without test refitting. The neural head remains secondary;
its result cannot replace the primary after scores are known.

Report native RPB-v4 versus PCA-only32 and raw576, full coverage/counts, paired
effects and uncertainty. Repeat the fixed-readout stress protocol with identical
corruptions and no fit under stress. Do not routinely train RPB-v1/v2/v3-mean.

## Phase 3 Change the encoder only for a measured gap

If native linear accessibility or missingness remains weak relative to the
declared controls, register one new version and change one encoder mechanism.
A candidate hypothesis is learned global pooling directly over aligned
patch/channel states, before information is compressed into separate channel
vectors. It aims to preserve joint temporal relationships at the served
32-number bottleneck. Its benefit is a hypothesis, not an established result.

Keep the reconstruction objective, data, classifier recipes and selection budget
fixed for that comparison. Use RPB-v4 as the relevant reference for this one
advance; do not bring every older variant back. Do not add auxiliary losses or
tune classifier capacity without a separate, justified experiment.

If raw/PCA controls and RPB-v4 already solve the simple synthetic tasks, move to
harder declared tasks or the intended consumer dataset instead of optimizing a
saturated score. Consumer confirmation requires the dataset/task, chronology,
support, thresholds and cost contract to be defined before acceptance.
