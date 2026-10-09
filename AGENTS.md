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

Every new results table needs an adjacent dataset legend with its codename,
short signal recipe and numeric designed complexity, using the
[dataset registry](doc/DATASET_REGISTRY.md). TEMPO-1 is original fixed timing1/5,
TEMPO-2 is variable-delay timing2/5, TEMPO-3 is structured-hard timing4/5
(variable delays, gains/offsets and 3-tick channel gaps), and AMP-1 is separate
amplitude1/5. Levels are ordinal design
labels, not mathematical measurements or accuracy-derived rankings. State the
view/split and keep head/update budgets independent. Add a Dataset column only
when a table mixes datasets; otherwise retain the standard five columns.
These rules apply prospectively; preserve all frozen historical reports/cards.

Evaluate native encoder exports without PCA afterward. PCA is permitted as a
standalone raw-data baseline labelled exactly "PCA only — no encoder".
Keep classifier architectures, training budgets, and declared seed policies
fixed across encoder comparisons; fit their weights separately on training data.

## Active research version

The fixed 512
[TEMPO-3 comparison](code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_DIAGNOSTIC.md)
for fresh RPB-v7.alt-05/RPB-v10.alt-05: structured harder timing, designed
complexity 4/5, near-chance fixed-head scores and 100% coverage. Raw fixed-head
scores are also near chance; the legal-observation analytic rule solves all
saved quality cohorts. Do not infer encoder information loss from accuracy alone.
Neither group is promoted. The [saved-TRAIN diagnosis](code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md)
is now complete, with no model execution or head refits. Early-v10 neural
accuracy is 82.06% on TRAIN and54.90% on intact VALIDATION; raw neural accuracy
is99.69%/50.73%. Same-support query timing is 100% in targets,68.28% in late
reconstruction and61.02% in early reconstruction. The four8-tick query banks
use different masked contexts; these scores do not establish information loss
in a single full-context native32 vector. The
[RPB-v13 visible-difference experiment](code/encoders/raw_patch_bottleneck_mae/VISIBLE_DIFFERENCE_DIAGNOSTIC.md)
is now measured and audited at fixed native32/.15/B8/512 and heads. It adds 3,072
zero-initialized values while preserving the original 225,805 common
initialization and all 15 initial features. Intact linear/neural means are
53.28%/52.03%; deleted means 52.81%/50.52%, all at 100% coverage. Mean query
TRAIN/VAL MAE improves to 0.622320/0.664399, but timing does not improve
consistently. The audit passed 17,152,339 checks / 280 CPU witness archives.
The original sealed reader and failed v1 audit remain preserved; the additive
reader-v2 corrects only a query artifact basename and binds its exact original
source. No CPU model execution, old model forwards or control head refits.
The user redirected work toward architecture changes and asked to limit
comparisons of weak candidates. The [measured architecture screen](code/encoders/raw_patch_bottleneck_mae/TEMPORAL_ARCHITECTURE_SCREEN.md)
finds RPB-v14's generic relation bank much stronger on TEMPO-3:97.03%/95.10%
intact and72.97%/75.21% deleted across five known cohorts, at100% coverage.
Its untrained prior is stronger; credit the architectural relation calculation.
RPB-v15 stays near52% and stops after two cohorts. The [separate v16 screen](code/encoders/raw_patch_bottleneck_mae/PARTITIONED_TEMPORAL_RELATION_SCREEN.md)
reaches100% intact with20 shape+12 grouped odd coordinates, but master76373's
69.53% deleted neural score fails its75% gate; stop that frozen recipe.
[RPB-v17](code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_TEMPORAL_RELATION_SCREEN.md)
freezes only the generic432 odd weights, while learning shape/backbone/decoder.
Five-cohort intact accuracy is100%/100%; deleted accuracy is85.94%/99.84%,
at100% coverage. Passed saved checks verify30,720 exact odd coordinates at0/512;
actual CUDA admission checks immutable load and trainable gradients.
Keep shared heads, native32, CUDA-only encoder execution and original masked-query
MAE. Use each frozen card's two-cohort75%/100%-coverage gate before remaining
known cohorts; known validation is development evidence, not unseen confirmation.
The [current continuation](doc/CONTINUATION_2026-10-10_ARCHITECTURE_SCREEN.md)
records fresh-data verification as the next step; the earlier data-support2×2
plan is deferred and preserved. No phase/lag labels, head tuning, stopped-recipe
rescue or automatic promotion. Preserve original v7 and formal v4.

The earlier [confirmation](code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_CONFIRMATION_DIAGNOSTIC.md)
retains both worst-cohort regressions, including master65262, without promotion.

The working saved v7 models are pinned as **RPB-v7** in the version
registry: masters 4404/5505/6606/7707/8808 in the original lighter-policy capsule.
Preserve their checkpoints, scalers, fitted readouts and reports. The newer
**RPB-v7.alt-01** contains separately trained instances on different
sources/seeds and does not replace this frozen reference. Use these short labels
when comparing the groups; do not describe replication scores as a decline
of the original saved encoder. Any architecture, objective or training recipe
change gets a new design tag and separate outputs, leaving the original v7 intact.

RPB-v4, the learned global bottleneck, is the active experimental encoder.
RPB-v5, the direct patch global bottleneck, failed the fixed 512 comparison under
paired-pooling-v1 and is not promoted. This rejects that measured advance, not
the design under every budget. Its separately frozen TRAIN/VALIDATION-only
continuation at1024/2048 improved reconstruction without improving mean native
linear accuracy. It did not reopen TEST/stress or change the fixed 512 disposition.
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
no best point was selected. The completed [matched-target gain diagnostic](code/encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
compares fresh RPB-v10.alt-02/RPB-v11 at fixed 512. All six numeric guards fail,
all five original-query MAEs worsen, and equal 100% coverage remains. Audit
passed 88,154,488 checks. Stop this gain recipe without range/rate/budget/head
rescue. Preserve original v7, active v4, all earlier groups and every curve point.
The completed [pooled-context diagnostic](code/encoders/raw_patch_bottleneck_mae/POOLED_CONTEXT_DIAGNOSTIC.md)
compares RPB-v10.alt-03/RPB-v12 at fixed 512. All four timing numeric guards
fail and all five cohorts lose timing Ridge in both views; both mean MAE guards
and coverage pass. The sole archive audit passed 96,170,207 checks. Stop wider
pooled width and gain without rate/budget/head rescue. Compact early v10 remains
an investigation candidate, not a promoted reference. The [next direction](code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md)
was the fresh fixed 512 confirmation against late v7, now completed as linked
above. Preserve every group and curve point, original working v7 and active v4.
The [historical pooled continuation](doc/CONTINUATION_2026-10-09_AFTER_POOLED_CONTEXT.md)
records that decision and its container-only gates.
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
sealed passed audit. Do not rewrite them to improve a result. The
[historical continuation](doc/CONTINUATION_2026-10-08_AFTER_FRESH_DECODER_REPLICATION.md)
prescribed saved TRAIN reliability analysis across all ten instances, with no
encoder/decoder updates, head refits or held-out analysis inputs. That diagnosis
and the subsequent milestones are complete; the current next action is recorded
in the TEMPO-3 continuation above. Do not tune the heads, deletion rate,
decoder budget or stopped v9 coefficients.

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
