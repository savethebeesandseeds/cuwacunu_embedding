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
Older versions remain available for archived evidence, explicit comparisons,
and compatibility tests. Do not routinely retrain all older architectures.
Model tags identify designs; exact configurations, source revisions, datasets,
scalers, seeds, update counts, and checkpoint hashes identify trained instances.

Preserve historical cards, checkpoints, reports, and their original protocol
semantics. Existing drivers that select using compressed features require a
separate native-only protocol before new results can follow this standard.
The [next advance plan](code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md)
records that implementation boundary. Active research status does not change
legacy loader defaults or assert consumer acceptance.

## Execution

Follow [the environment instructions](doc/ENVIRONMENT.md). Build, test, and run
project code in the existing managed development container using named sessions.
Verify actual CUDA parameters, inputs, loss, gradients, and weight updates before
expensive encoder training. Preserve all existing artifacts and container data.
