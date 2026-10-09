# Project instructions

## Results and communication

Follow [the reporting standard](doc/RESULTS_REPORTING_STANDARD.md) in final
responses: results tables, a short explanation of what changed, and what to do
next. State whether an experiment is new, reused, planned, or not measured.
Use plain words and the stable model tags in
[the version registry](doc/EMBEDDING_VERSIONS.md); do not name models "previous"
and "current".

The main quality table uses Method, Size, Linear head %, Neural head %, and
Coverage %. The training table uses Encoder, Updates, Train error, Validation
error, and GPU training seconds. Identify the task, split sizes, batch size,
device, repetitions, head recipes, checkpoint budget, and evidence stage.
Use only the registered model label in encoder table cells, including an
alternative suffix such as `RPB-v7.alt-01` when applicable. Put short embedding descriptions
in prose immediately beside the table, rather than inside its cells.
Do not average different tasks into an unspecified encoder accuracy.

Evaluate native encoder exports without PCA afterward. PCA is permitted as a
standalone raw-data baseline labelled exactly "PCA only — no encoder".
Keep classifier architectures, training budgets, and declared seed policies
fixed across encoder comparisons; fit their weights separately on training data.

## Active research version

The working saved v7 models are pinned as **RPB-v7** in the version
registry: masters 4404/5505/6606/7707/8808 in the original lighter-policy capsule.
Preserve their checkpoints, scalers, fitted readouts and reports. The newer
**RPB-v7.alt-01** contains separately trained instances on different
sources/seeds and does not replace this frozen reference. Use these short labels
when comparing the groups; do not describe replication scores as a decline
of the original saved encoder. Any architecture, objective or training recipe
change gets a new design tag and separate outputs, leaving the original v7 intact.

RPB-v4, the learned global bottleneck, is the active experimental encoder.
RPB-v5, the direct patch global bottleneck, failed the fixed512 comparison under
paired-pooling-v1 and is not promoted. This rejects that measured advance, not
the design under every budget. Its separately frozen TRAIN/VALIDATION-only
continuation at1024/2048 improved reconstruction without improving mean native
linear accuracy. It did not reopen TEST/stress or change the fixed512 disposition.
RPB-v6's audited context-deletion comparison improves both linear timing primaries,
with equal coverage and improved worst-master scores, but fails the no-worse
TRAIN/VALIDATION reconstruction guard. It is not promoted. Its completed
[context optimization diagnostic](code/encoders/raw_patch_bottleneck_mae/CONTEXT_OPTIMIZATION_DIAGNOSTIC.md)
replays the exact saved 512 state and continues the unchanged policy through
1024 and 2048 on TRAIN and known VALIDATION only. No measured budget combines
the original v4 fixed-512 reconstruction reference with preserved v6 fixed-512
linear accuracy in both validation views. At 2048, mean TRAIN/VALIDATION MAE is
0.035736/0.035845, while intact/deletion linear accuracy is 96.875%/95.3125%.
This diagnostic accessed no TEST and did not change the fixed-512 disposition.
RPB-v4 remains the active reference until an explicit decision.
The completed [five-master replication](code/encoders/raw_patch_bottleneck_mae/CONTEXT_REPLICATION_ADVANCE.md)
pairs fresh v4/v6 training at 512 updates on 4404/5505/6606/7707/8808. Mean linear
gains are +1.71875 percentage points intact and +8.125 with additional 30%
coordinate deletion, at equal 100% coverage. Mean TRAIN and VALIDATION
reconstruction guards still fail. Independent archive audit v3 passed
103,787,020 checks; failed v1/v2 reader attempts and their corrections are
preserved. RPB-v4 remains active.
RPB-v7 — Global bottleneck with lighter context deletion — is implemented and
measured on five known TRAIN/VALIDATION masters at 512 updates under the frozen
[lighter-policy card](code/evaluation/cards/context_lighter_validation_v1.md).
Its separate policy is `rpb-training-context-deletion-015-v1`; inference is
unchanged. The [diagnostic record](code/encoders/raw_patch_bottleneck_mae/CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md)
reports linear means of 97.96875% intact and 95.15625% with additional 30%
VALIDATION deletion, at 100% coverage. Both exceed paired v6 means, but neural
means fall, master 8808 loses deletion accuracy and both mean reconstruction
guards against v4 fail. Every master's v7 reconstruction exceeds its v4
reference. No TEST/stress, selection or promotion occurred. Independent archive
audit v3 passed 67,097,380 checks; save the measured milestone and its passed audit before
further production changes. Preserve the exact v6 default/replay, explicit
recipe binding, ordinary tagged-resume rejection and all historical artifacts.
The completed [RPB-v8 balanced-view diagnostic](code/encoders/raw_patch_bottleneck_mae/CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md)
fails the fixed joint guard; its independent audit passed 67,099,959 checks.
The subsequent [TRAIN objective diagnostic](code/encoders/raw_patch_bottleneck_mae/TRAINING_OBJECTIVE_DIAGNOSTIC.md)
uses fifteen saved instances with zero encoder updates and zero head refits.
The proposed all-pair residual-difference loss improves the fixed-head gradient
direction on two v7 masters and worsens it on three, failing its prospective rule.
Do not train that auxiliary or tune its weights or pair subsets. Independent
saved-arithmetic audit v3 passed 16,119,003 checks; the failed v2 reader and
cross-backend correction are preserved. This is local TRAIN evidence, not a
new accuracy result. Preserve the diagnostic sources, card and captured evidence.
The completed [RPB-v9 native view agreement diagnostic](code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md)
fails all six numeric guards at fixed 512. Native linear means are 59.21875%
intact / 57.03125% under extra deletion; mean TRAIN/VALIDATION MAE is
29.194490 / 29.371655, including one especially unstable run. Actual CUDA
admission and independent audit passed 74,721,050 checks. Stop this frozen
mechanism without coefficient/rate/budget rescue. RPB-v4 remains active;
this v9 mechanism remains stopped. The original
[continuation note](doc/CONTINUATION_2026-10-08.md) prescribed the saved TRAIN
loss-scale/component diagnosis, now complete as recorded below. No TEST/stress
or automatic promotion.
The completed [early-mixer learning curve](code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md)
retains every 0/512/1024/2048 point for fresh RPB-v7.alt-03/RPB-v10.alt-01.
Both reconstruct more accurately without monotone timing classification benefit;
no best point was selected. The latest [matched-target gain diagnostic](code/encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
compares fresh RPB-v10.alt-02/RPB-v11 at fixed 512. All six numeric guards fail,
all five original-query MAEs worsen, and equal 100% coverage remains. Audit
passed 88,154,488 checks. Stop this gain recipe without range/rate/budget/head
rescue. Preserve original v7, active v4, all earlier groups and every curve point.
The [next direction](code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md)
is engineering-first review of one pooled-context information path with a
matched compact control; it has no measured result or new registered tag yet.
The user's current working direction is RPB-v7, with
RPB-v4 retained as the active reference. The separate
[saved-TRAIN v9 scale lesson](code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_LOSS_SCALE_TRAIN_DIAGNOSTIC.md)
and [v7 decoder calibration](code/encoders/raw_patch_bottleneck_mae/V7_DECODER_CALIBRATION_DIAGNOSTIC.md)
are complete. Decoder-only 128 on five frozen v7@512 instances gives mean
TRAIN/known-VALIDATION MAE 0.054039/0.056989; all five beat their own v4 reference.
Encoder/native/scaler exactness and independent saved-arithmetic audit passed.
No classification refits, new embedding tag, TEST/stress or promotion occurred.
The [dated continuation](doc/CONTINUATION_2026-10-08_AFTER_DECODER_CALIBRATION.md)
preserves the prospective equal-budget replication plan. That
[fresh replication](code/encoders/raw_patch_bottleneck_mae/FRESH_DECODER_REPLICATION_DIAGNOSTIC.md)
is now complete on masters9109/10210/11311/12412/13513. V7/v4 intact linear
means are 92.96875%/92.34375%; extra-deletion means are 89.84375%/82.03125%,
all at 100% coverage. The intact effect interval crosses zero and v7's
worst intact score is lower. Recovery versus pre-calibration v4 passes, but both
equal-decoder reconstruction guards fail; the joint guard fails three of six
numeric conditions. Independent saved-evidence audit passed 67,932,331 checks.
No TEST/stress, selection, promotion or new tag occurred. Preserve the fresh
card, measured sources, all five pairs, failed admission fixture evidence and
sealed passed audit. Do not rewrite them to improve a result. Continue from v7
with v4 as reference under the
[current continuation](doc/CONTINUATION_2026-10-08_AFTER_FRESH_DECODER_REPLICATION.md):
first diagnose saved TRAIN representation reliability across all ten instances,
with no encoder/decoder updates, head refits or VALIDATION/TEST/stress analysis
inputs. Do not tune the heads, deletion rate, decoder budget or v9 coefficients.

Older versions remain available for archived evidence, explicit comparisons,
and compatibility tests. Do not routinely retrain all older architectures.
Model tags identify designs; exact configurations, source revisions, datasets,
scalers, seeds, update counts, and checkpoint hashes identify trained instances.

Preserve historical cards, checkpoints, reports, and their original protocol
semantics. Existing drivers that select using compressed features retain their
original rules. The separately implemented native-only protocol governs new
results under this standard.
The [next advance plan](code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md)
records that implementation boundary. Active research status does not change
legacy loader defaults or assert consumer acceptance.

## Execution

Follow [the environment instructions](doc/ENVIRONMENT.md). Build, test, and run
project code in the existing managed development container using named sessions.
Verify actual CUDA parameters, inputs, loss, gradients, and weight updates before
expensive encoder training. Preserve all existing artifacts and container data.

Train each declared encoder run once on CUDA. CPU work may fit the fixed
classifier heads and raw-data PCA, or verify calculations from saved tensors.
It must not repeat encoder training. Reuse saved native feature archives for
routine scoring; frozen inference is for extracting needed features or an
explicit correctness/parity witness. Do not rerun encoders merely to repeat
already captured measurements. Keep training, inference, head fitting and
archive-audit time separate in reports.
