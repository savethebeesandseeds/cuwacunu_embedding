# Current repository status

Updated on 13 October 2026 after the completed content-only ordered relation
experiment. RPB-v22.alt-01 passes the two-cohort screen and third development
check. It retains useful learned timing and complete-embedding missing-value
protection. Selected v20 settings, original checkpoints and all earlier
measurements remain preserved; there is no automatic serving default change.

## Implemented code and measured evidence

| Track | Status | Entry point |
| --- | --- | --- |
| RPB-v22.alt-01 | Implemented content-only relation candidate; two-cohort screen and third development check pass | [Report](CONFIGURABLE_CONTENT_RELATION_V1.md), [module](../code/encoders/raw_patch_bottleneck_mae/CONTENT_ORDERED_RELATION.md) |
| RPB-v21.alt-01 | Implemented learned relation candidate; completed two-cohort screen fails deletion guards and is stopped | [Report](CONFIGURABLE_LEARNED_RELATION_V1.md), [module](../code/encoders/raw_patch_bottleneck_mae/LEARNED_ORDERED_RELATION.md) |
| RPB-v20 | Implemented configurable spectral API, completed capacity experiment and saved-coordinate diagnostic | [API](../code/encoders/raw_patch_bottleneck_mae/CONFIGURABLE_SPECTRAL.md), [capacity](CONFIGURABLE_CAPACITY_V1.md) and [coordinate diagnostic](CONFIGURABLE_BRANCH_V1.md) |
| RPB-v19.alt-01 | Preserved fresh synthetic confirmation of unchanged RPB-v19 | [Confirmation report](SPECTRAL_CONFIRMATION_V1.md) and [saved summary](results/spectral_confirmation_v1.json) |
| RPB-v4 | Formal evaluated reference, retained under its original protocol | [Native curve record](../code/encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md) |
| RPB-v7 | Original five-instance reference bundle, preserved with its scalers and readouts | [Version and instance registry](EMBEDDING_VERSIONS.md) |
| MTF-v1 | Retained multiscale JEPA/MAE/VICReg encoder and historical CLI | [Encoder guide](../code/encoders/mtf_jepa_mae_vicreg/README.md) |

The latest completed quality experiment is
[configurable-content-relation-v1](CONFIGURABLE_CONTENT_RELATION_V1.md).
It removes additive observation-density projection inputs, while retaining
counts for normalization/validity and the exact v21 odd/even content moments.
All three known TEMPO-5 cohorts and all fixed head repetitions are retained;
both prospective stages pass. Native32 and its diagnostic learned20 slice,
original-query errors and separate GPU costs are recorded in the report.
This is development evidence, not fresh confirmation, TEST or market quality.
Follow the [content-only continuation](CONTINUATION_2026-10-13_AFTER_CONTENT_RELATION.md).

For the strongest complete encoder in this matched TEMPO-5 development
comparison, read [RPB-v22 explained](RPB_V22_ENCODER_GUIDE.md). It documents
rank-4 values/masks `[B,C,H,F]`, one rank-2 native embedding `[B,D]` per window,
the learned20/fixed12 composition at C3/D32, training reconstruction and
the separate classifier evaluation. RPB-v22.alt-01's three-cohort intact
Linear/Neural means are 97.14%/95.96%, with 100% coverage; these reuse the
completed report, not a new measurement or universal encoder ranking.

The optional portable v22 API has also been synced into sibling `cuwacunu`
and passed its focused artificial CUDA integration check. Selected v20 settings
remain unchanged and no synthetic trained checkpoint is selected for market
use. The sibling [sync manifest](../../cuwacunu/doc/CONTENT_ENCODER_SYNC.json)
records that engineering evidence separately from synthetic quality.

The preceding [configurable-learned-relation-v1](CONFIGURABLE_LEARNED_RELATION_V1.md)
screen remains stopped.
On the same two known TEMPO-5 cohorts, its complete32 intact Linear/Neural
means improve to 97.85%/96.48%, and learned20 reaches 62.50%/61.00%.
Extra-deletion complete32 falls to 65.43%/78.06%, compared with matched saved
v20 controls 88.67%/87.89%; all coverage is 100%. Nine of 54 declared conditions
fail. No third cohort or default replacement follows. The preceding
[TRAIN-only diagnosis](LEARNED_TRAIN_V1.md) found varying learned coordinates
with weak paired timing separation, without any model execution or head fit.

The preceding capacity record is [configurable-capacity-v1](CONFIGURABLE_CAPACITY_V1.md)
on TEMPO-5 slow lead/lag beneath fast rhythms, nine grouped features and structured
distractors, at designed complexity 5/5. Three fresh source cohorts compare
RPB-v20.alt-01 (W64/D32), alt-02 (W128/D32) and alt-03 (W64/D64), all at
C3/H128/F9/P8 and the same fixed 512-update budget. W is the internal encoder width.
Alt-01 is retained as the compact continuation configuration; larger profiles
show no consistent final classification benefit at that budget. Initial controls
are already strong. The completed [saved-coordinate diagnostic](CONFIGURABLE_BRANCH_V1.md)
finds fixed12 alone gives95.18%/94.66% intact and89.97%/89.54% deleted
Linear/Neural means, while learned20 alone remains near50% before and after
training, all at100% coverage. Complete32 scores are reused unchanged. This
measures usefulness under the fixed heads, not intrinsic information or a perfect
causal attribution. The same RPB-v20.alt-01 native32 encoder remains preserved;
no subset becomes a new serving model or design tag.
The [instance registry](configurable_capacity_embedding_versions.json) and
[saved summary](results/configurable_capacity_v1.json) identify the exact bundles.

RPB-v19 combines learned shape coordinates with a fixed two-band spectral prior.
Its confirmed quality is specific to TEMPO-4 slow-component timing and AMP-2
component balance, each at designed complexity 5/5. Most strength comes from the
prior; the saved report distinguishes initial controls from trained instances.
Those v19 scores do not measure the separate v20 capacity profiles. Neither
experiment establishes market representation quality, H256 encoder quality or
arbitrary geometry/capacity. No reference promotion or CLI default change follows.

## RPB-v20 contract

Input values and observation masks have shape `[B,C,H,F]`; the served native global
embedding has shape `[B,D]`. B is the batch size, C the semantic channel count,
H the history length, F the features per tick, and D the embedding width.
The checked-in settings use C3/H128/F9/P8/D32, where P is the patch length.
These are model defaults, not a chosen market dataset or candle interval.

C/H/F/D and channel IDs are fixed per constructed model and checkpoint; B can
vary. H must be at least 17 and may be nondivisible by P. The learned backbone
pads privately to complete patches, while the fixed harmonics retain original H
and reconstruction is cropped back to H. Four fixed spectral coordinates are
allocated per semantic channel pair, so D must exceed `4*C*(C-1)/2`; the rest are
learned. All F features enter the learned path and all F² feature pairs enter
the fixed bank. Model execution requires CUDA float32.

Fit the configurable scaler only on TRAIN, then reuse it for validation and
serving. It currently fits from a complete supplied tensor; bounded streaming
statistics remain future work. Reconstruction eligibility requires three observed
patches per channel; skip batches with no eligible examples.

The typed inference checkpoint binds full resolved geometry, semantic IDs,
sampling interval, scaler and completed update count. It excludes optimizer/RNG
resume. Saves require a new filename and publish atomically; existing files are
protected, and malformed/nonfinite model state is rejected. Historical CLI
checkpoint/resume formats have their own semantics and are not this API.

The API and [model-only settings](../code/encoders/raw_patch_bottleneck_mae/config/configurable_spectral.conf)
are separate from the historical `embedding_raw_patch_bottleneck_mae` CLI.
The `rpb-v20` Make target builds four engineering check executables;
`test-rpb-v20` runs them. Neither target creates a general v20 training/serving CLI.
The separate [capacity protocol](../code/evaluation/protocols/configurable_capacity_v1/README.md)
owns its experiment-specific CUDA producer and saved-feature evaluator.

## Repository boundary and validation

`cuwacunu_embedding` owns reusable representation models, core settings and
evaluation research. Its sibling `../cuwacunu` owns the market loader,
CSV/cache/mmap storage, time-grid alignment and application adapter.
Reusable encoder improvements return here; market ingestion stays there.

The initial backport is `37c7a3b`; the stability backport is `9123891`.
Recorded checks cover strict config parsing, configurable CUDA shapes/masks,
exact old-shape v19 parity, typed model/scaler checkpoint round trip and immutable
checkpoint integrity. Tested shapes are correctness witnesses, not a maximum
input size or measured throughput/VRAM capacity.
The sibling [backport manifest](../../cuwacunu/doc/CONFIGURABLE_ENCODER_BACKPORT.json),
[engineering validation](../../cuwacunu/doc/CONFIGURABLE_ENCODER_VALIDATION.md) and
[stability review](../../cuwacunu/doc/STABILITY_REVIEW_2026-10-12.md) retain receipts.
Those sibling links require this workspace layout; the standalone API guide is
self-contained.

For focused engineering checks, use the existing managed container and a named
session, as specified in [ENVIRONMENT.md](ENVIRONMENT.md):

```powershell
.\container.ps1 status
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'docs-v20', '-j2', 'test-rpb-v20')
```

These are reproduction instructions, not checks rerun for this documentation
refresh. Preserve completed feature archives and use them for routine reporting.
Research model execution runs on CUDA; CPU work can prepare data, fit fixed
heads/PCA baselines, or audit saved arithmetic inside the managed container.

## Open work

Follow the [latest 13 October continuation](CONTINUATION_2026-10-13_AFTER_CONTENT_RELATION.md):
declare an unchanged-v22 fresh synthetic confirmation before allocating new
quality sources. The content-only architecture passes all known development
cohorts; preserve its complete record and the stopped v21 recipe. Keep the fixed
prior and classifiers. Matched H128/H256 quality work and a feature-specific
challenge remain later prospective steps. The capacity experiment already
records descriptive costs at its three configurations; it does not establish
maximum input size, controlled performance scaling or H256 encoder quality.
Market quality and bounded streaming scaler fitting remain unmeasured/unimplemented,
respectively. Record each new experiment with its frozen card, source/config
receipts, instance tags and saved report before updating status or registry pointers.

The harmonic linear decoder in the
[10 October continuation](CONTINUATION_2026-10-10_AFTER_SPECTRAL_CONFIRMATION.md)
is a preserved prospective architecture idea. It has no implementation or measured
quality and is not the default next action after the v20 integration.

Market preparation continues separately in the sibling
[integration continuation](../../cuwacunu/doc/CONTINUATION.md): select sources and
a common cadence, declare chronological splits, implement streaming TRAIN
statistics, measure RAM/GPU use and run a bounded pilot. This status refresh
does not launch those experiments.

## How to read and maintain the docs

Start with this page and the [documentation index](README.md). Encoder API guides
describe supported code; the [version registry](EMBEDDING_VERSIONS.md) maps designs
and exact instance bundles, including its additive capacity registry. Dated
continuations, journals, measured reports,
cards and saved JSON receipts retain their original stage and protocol.
An old document's "next" or "current" wording is local to its recorded date.

Update this page when implementation or completed evidence changes, then update
the live guide links and machine-readable registry pointers. Preserve frozen
historical reports/cards, artifact hashes and original model instances. Proposed
work needs an explicit status until the required admission and measurements finish.
