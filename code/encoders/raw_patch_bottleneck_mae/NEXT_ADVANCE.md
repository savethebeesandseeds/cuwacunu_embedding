# RPB v4 evaluation and development plan

Date: 2026-10-07

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
The definitions below record the frozen phase-1 procedure. Phase 2 is also
complete; its fresh TEST results are kept separate from these validation scores.

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

**Completed:** the [native curve advance](NATIVE_CURVE_ADVANCE.md) implements
the separate shared native-only driver and its frozen card. CUDA correctness
and focused container tests passed, followed by an independent artifact audit.
Three timing trainers continued through 0/128/512/2048. Native linear VALIDATION
selected 512 (94.79%), ahead of 128 (92.19%) and 2048 (90.10%), before any TEST
generation. Fresh timing TEST means are 94.01% linear / 96.09% neural, versus
51.82% / 79.51% PCA-only and 46.61% / 50.61% raw; coverage is 100% throughout.
Direction, level and amplitude each reach 100% with trained native32 heads.

Reconstruction improves through 2048, but native linear validation does not.
The remaining timing gap is consistency across trained instances and moderate
additional missingness. The exact untrained timing controls score 55.47% linear
/ 65.19% neural, showing a substantial training contribution. The procedure
below records the completed protocol; historical compressed drivers stay intact.

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

**Completed:** the [paired pooling record](PAIRED_POOLING_ADVANCE.md) reports
RPB-v5 versus the frozen RPB-v4 reference at512. Intact timing linear accuracy
is84.38% versus94.53%; additional30% deletion is66.15% versus85.16%, with equal
100% coverage. V5 TRAIN/VALIDATION fixed-query MAE also worsens. The claimed
advance is rejected at this fixed budget; RPB-v4 stays active. The unchanged
card and full independent audit preserve this measured result rather than
universally rejecting the architecture.

The following conditions record that completed experiment: learned global
pooling directly over aligned patch/channel states, before separate channel
summaries compress them.
RPB-v4 already mixes aligned channel states; the proposed pool bypasses its
subsequent per-channel compression. It aims to preserve joint temporal
relationships more consistently at the served 32-number bottleneck. These measurements
do not establish that compression caused the remaining errors.

Keep the temporal blocks, channel mixer, legal preprocessing, reconstruction
objective, exact served 32-number decoder input, sampling/mask policy, fixed heads and
512 encoder updates. Freeze pooling support/position/channel rules and pair
common-module initialization with RPB-v4's point-zero assets. Record added
parameters and training cost; equal export width does not mean equal encoder
capacity. All-absent exports remain zero and invalid.

Use only the selected RPB-v4 instances and their original TRAIN-fitted assets
as the relevant reference. Do not bring every older variant back, add auxiliary
losses, tune classifier capacity, or fit under stress. Declare intact timing
and the existing 30% additional-deletion case before scores; its legal raw rule
retains full accuracy and coverage. Primary evidence is native linear accuracy,
per-master spread and within-master paired effects, with neural results secondary.

Freeze a candidate card and durable fixed-budget manifest before generating a
new TEST namespace. The opened phase-2 TEST data are now development evidence.
Reuse legal TRAIN/VALIDATION cohorts for paired development if needed, and apply
candidate and frozen RPB-v4 to common new TEST sources and corruption masks,
without TEST refitting. Preserve every declared master/head and point-zero
diagnostic. No across-retraining confidence interval is claimed from three masters.

If raw/PCA controls and RPB-v4 already solve the simple synthetic tasks, move to
harder declared tasks or the intended consumer dataset instead of optimizing a
saturated score. Consumer confirmation requires the dataset/task, chronology,
support, thresholds and cost contract to be defined before acceptance.

## Completed diagnostic: continue v5 on TRAIN and VALIDATION only

The [optimization diagnostic](OPTIMIZATION_DIAGNOSTIC.md) is complete and
independently audited. Mean linear VALIDATION accuracy was84.38% at512,
84.38% at1024 and82.81% at2048, while reconstruction error improved at both
new budgets. Longer training did not improve mean linear access under the fixed
heads. This does not change the fixed512 rejection or establish universal
failure at every possible budget. The following records its frozen procedure.

Save and commit the fixed512 milestone before further source work. A separate
prospective recipe may resume each exact saved v5 point512 model, AdamW state,
scaler and counter stream through absolute1024 and2048 updates on the same
TRAIN observations. Keep all three masters, native32 and the fixed readout
recipes; candidate heads at new points fit only TRAIN. Record any settings
step-ceiling override and cumulative cost without resetting initialization,
optimizer, preprocessing or sampling/mask progression.

This is an optimization diagnosis using known VALIDATION, fixed-query
TRAIN/VALIDATION reconstruction and native linear access. Do not open or
rescore TEST/stress, introduce a replacement seed, alter the fixed512 rejection,
or promote v5 based on it. Its own card was frozen before running. Any later trained-budget comparison needs its own
declared reference/budget and new TEST namespace before unseen scoring.

## Completed bounded candidate: RPB-v6 context deletion

The [context-deletion advance](CONTEXT_DELETION_ADVANCE.md) is complete and
independently audited. Timing TEST means are98.96%/100% intact and97.14%/99.65%
with30% additional deletion (linear/neural), versus93.49%/96.35% and86.72%/93.06%
for frozen RPB-v4, with full coverage. The quality guards pass, but mean original
fixed-query TRAIN/VALIDATION MAE worsens to.08552/.08577 from.06160/.06349.
This unresolved reconstruction tradeoff prevents promotion. The following
records the completed card.

Keep the RPB-v4 inference architecture, all225,805 parameters and exact served
32-number decoder input. Change only its training view: remove30% of originally
visible coordinates through a separate deterministic counter stream, repairing
only extra deletion to retain two original visible patch groups for each target
eligible channel. Keep the original target queries, loss, scaler, batches,
optimizer,512-update budget and classifier recipes.

Freeze [context-deletion-v1](../../evaluation/cards/context_deletion_v1.md)
before measurement. Reuse the three retained TRAIN/VALIDATION cohorts, paired
point-zero weights and frozen RPB-v4/control readouts. Preserve enabled-context
CUDA test evidence and source identity before the new TEST namespace is opened.
Ordinary training must reject resuming a checkpoint carrying the new training
policy; ordinary inference may serve its normal native32 embedding.

Both mean linear primaries (intact and30% additional deletion) must improve,
with equal coverage, no lower worst-master primary score and no worse mean
TRAIN/VALIDATION fixed-query reconstruction error. Neural and other stress
results stay secondary. Preserve every seed and audit the artifacts before any
explicit promotion.

## Next diagnostic: RPB-v6 optimization with the same training view

Freeze a separate TRAIN/known VALIDATION-only card for absolute512/1024/2048.
Keep the exact30% context policy, native32, all three cohorts and fixed heads.
Use the existing continuous trainer with ceiling2048: deterministically replay
to512, verify exact weights/buffers/AdamW/scaler/absolute counters and context
deletion counts against the saved C checkpoints, then advance the same live
optimizer. Record replay cost separately and retain every budget/seed.

Measure whether reconstruction reaches the frozen v4 fixed512 reference while
intact and moderate-missingness native linear VALIDATION access is preserved.
Use an explicitly declared VALIDATION corruption diagnostic; preserve the
existing TEST-only stress protocol's semantics. No TEST reopening, checkpoint
selection, head tuning or promotion occurs in this diagnostic. The fixed512
record and its reconstruction-guard failure remain unchanged.
