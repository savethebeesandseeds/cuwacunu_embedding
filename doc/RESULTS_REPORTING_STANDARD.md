# Embedding results reporting standard

Standard version: 1.0

Agreed: 2026-10-06

Report what the encoder contributes through fixed classifier heads. Show task
performance separately from reconstruction progress and training cost. Use
[stable model tags](EMBEDDING_VERSIONS.md) with short descriptions.

## Report structure

Each final update contains results, what changed, and what to do next. Say
"No new experiment" when reusing measurements. For a planning-only update,
identify planned measurements without inventing scores.

Before the tables, identify the task and evidence stage, training/validation/test
examples **per run**, independent source groups, number of encoder runs, batch
size, device, encoder update budget, classifier recipes, and evaluation protocol.
Paired transformations are two examples from one source; three readout fits on
one checkpoint are not three independently trained encoders.

Use one task per quality table. Do not average direction, level, amplitude, and
timing into a single encoder score. Longer reports may put detailed per-run
scores, uncertainty, and provenance in linked artifacts.

## Quality table

Use only the registered model tag in encoder rows. Place its short embedding
description in prose immediately above or below the table.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | Declared | — | — | — |
| PCA only — no encoder | Declared | — | — | — |
| Registered model tag | Declared | — | — | — |

- **Method:** the representation-producing method. Head choice is a column.
- **Size:** numbers supplied to a head, including appended masks or metadata.
  This is not the hidden width of the encoder or decoder.
- **Linear head %:** correct linear/ridge predictions divided by valid evaluated
  examples, multiplied by 100. This is conditional accuracy when support differs.
- **Neural head %:** the same accuracy for the fixed small neural classifier.
- **Coverage %:** valid evaluated examples divided by the entire declared scoring
  population (validation or test), multiplied by 100. It is not confidence or
  accuracy.
- **—:** not measured under the stated conditions. **N/A:** not applicable.
  Unsupported fits are named explicitly rather than scored as zero.

Keep head architectures and training recipes fixed. The present linear recipe
is ridge with penalty 1. The neural recipe is 16 tanh hidden units, Adam at 0.01,
and 100 updates. Fit each head's weights and normalizer only on its permitted
training data. Use paired declared initialization seeds for equal-size inputs;
never choose the best head seed or retune a head after seeing final scores.
At size 32 these heads have 66 and 562 fitted parameters respectively; larger
raw inputs have larger heads, which must be disclosed.

The primary encoder row uses the exact native served export, with training-only
normalization permitted. No PCA or random projection follows an encoder in this
comparison. The PCA-only baseline fits its normalization and PCA on raw training
observations and then sends those components to the same heads. Set its width
equal to the active encoder's native width for the main compact comparison.
Raw data remains a larger-input reference; an analytic rule is a separately
labelled solvability check, not a trained classifier result.

With unequal support, report each method's coverage, the common population for
the paired comparison, and full-population correctness or the card's declared
abstention utility. Include valid/total counts when coverage differs. A high
conditional accuracy on a small retained subset is not a full-population result.
For multiple encoder runs, report the mean and relevant spread. Conditional
source-group intervals and variation across retraining are different quantities.

## Training table

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| Registered model tag | — | — | — | — |

- **Updates:** completed optimizer weight updates, with attempts/skips retained
  separately. Report the encoder and classifier budgets separately.
- **Train error:** mean absolute reconstruction error on a fixed hidden-target
  query over training observations.
- **Validation error:** the same error on held-out validation observations,
  using training-fitted scales and a fixed query. Lower is better.
- **GPU training seconds:** synchronized cumulative encoder training time;
  excludes fitting classifiers, extraction, scoring, stress tests, and artifact
  writing. Times are descriptive unless a repeated cost benchmark is declared.

Name the error metric and units. The RPB track currently uses standardized MAE.
Retain the actual optimization loss, its name/components, and its trace in the
training artifacts. A final randomly masked minibatch loss is not the fixed
train error. Different objectives or scalers cannot be ranked by one generic
"loss" number.

Reconstruction error measures the encoder/decoder training route. Describe its
capacity when it changes: RPB-v2 reconstructs through three 32-number channel
vectors, while RPB-v4 reconstructs solely through its 32-number global export.
A lower reconstruction error does not establish a better classification embedding.
Other encoder families need their own applicable optimization diagnostics.

Use updates as the main training count. RPB samples rows with replacement and
fresh masks, so it has no exact full-pass epoch counter. If useful, report
equivalent presentations as sampled rows divided by available training rows.
For 512 completed, unskipped updates at batch size 8 and 256 training examples,
this is 4096 / 256 = 16 equivalent passes, not 16 guaranteed complete epochs.

## Changes and next action

Describe the encoder/configuration change, why it was made, and what the tables
support. Distinguish a new architecture, a new training run, a reporting change,
and reused evidence. State unresolved tradeoffs and the experimental/default
status. Identify the next concrete comparison before proposing another loss or
architecture. Parameter count, memory, and inference time accompany decisions
where model cost matters; they need not widen every routine table.

Keep older versions out of routine retraining. A frozen reference can appear in
a named historical comparison, but is not a newly matched training control for
another dataset. Every final number remains traceable to a protocol, checkpoint,
resolved configuration, source revision, and run artifact.
