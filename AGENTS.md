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
Use only the model tag in encoder table cells. Put short embedding descriptions
in prose immediately beside the table, rather than inside its cells.
Do not average different tasks into an unspecified encoder accuracy.

Evaluate native encoder exports without PCA afterward. PCA is permitted as a
standalone raw-data baseline labelled exactly "PCA only — no encoder".
Keep classifier architectures, training budgets, and declared seed policies
fixed across encoder comparisons; fit their weights separately on training data.

## Active research version

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
The next bounded question is a fixed ordinary/deleted training-view mixture,
documented separately before implementation; do not search rates or budgets.
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
