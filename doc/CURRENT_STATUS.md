# Current repository status

## Latest completed task: causal conversion, 15 October 2026

**CPB-v1.market-01** is the first verified causal reference, derived from the
exact saved V30 tensors with zero new encoder training or input/bank fits. Read
[the guide](CAUSAL_PATCH_BOTTLENECK_MAE_V1.md),
[the family entry](../code/encoders/causal_patch_bottleneck_mae/README.md) and
[the complete integration report](../../cuwacunu_dev/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md).
Physical `causal_patch_bottleneck_mae/versions/v1/` owns the complete source,
configuration and tests; future versions own full independent copies.

Actual CUDA proof passed128 cutoffs/2350 suffix interventions/30 internal
temporal cases, exact frozen/imported state and typed roundtrip. Same fixed
heads pass market9/9 and strict8/8; neural trend is positive in2/3 repetitions.
TEMPO-5 remains strong, with100% coverage, but only3/8 exact native36 retention
floors pass across both scaler policies and intact/deleted views. No overall
replacement follows: keep RPB-v22 primary, native36 balanced, V30's old market
result and every historical/scoped reference. Checkpoints remain local/ignored.
The previous continuous search was stopped by the user; do not resume it or
start causal training automatically. Next propose separately declared causal
training with robustness and retention guards. Source is mirrored; only the
existing integration GPU container ran proof/quality. No commit/push occurred.

The following sections preserve the earlier search history.

Active goal, 15 October 2026: preserve every balanced/task reference while
searching for the fully verified successor. The mirrored
[search ledger](ENCODER_SEARCH_LEDGER.md) and [exact index](encoder_search_ledger.json)
retain all eleven measured candidate records and 55 available, ignored checkpoint
pins. RPB-v22 stays primary; RPB-v24.market-01.alt-01 stays the balanced research
reference pending V30's separately declared frozen synthetic transfer.

Latest completed market experiment: **RPB-v30.market-01** passes **9/9 original
conditions, 8/8 strict native36 floors, all three positive neural trend fits and
100% coverage**. Final linear/mean-neural R² are volatility0.953913/0.588277,
trend0.794366/0.320210, activity0.865732/0.613572 and relation0.487476/0.406836.
These are four separate past-window tasks on MKT-HIST-2, complexity4/5, with
112 TRAIN and32 known chronological VALIDATION probes. One trained encoder and
three fixed neural fits do not establish unseen confirmation or forecasting.

V30 sets all four linear-head maxima and the neural-volatility maximum in the
recorded matched scope. Preserve v25 neural trend0.4884646605379312 and
relation0.4071495455067296, native40 neural activity0.6781894013517937, and all
displaced v26/v29 records. The initial neural activity control0.7246054298126288
is reused initialization scoring, not a trained-final leader. No universal
encoder ranking follows.

The Gaussian waveform location/scale objective keeps the unmodulated parent-
plus-change mean and adds zero-start variance matrix108/bias27. Its anchored
softplus link gives actual initial variance1. Targets/mean/variance are cast to
CUDA F64 before NLL arithmetic; legitimate negative losses remain valid.
Total/updated/protected parameters are297560/15039/282521. Native36 retains
exact parent32 plus learned4, the raw-relative bank and unchanged V28 normalizer.
Fresh saved v24 initial weights are used, never trained v28/v29 weights. Exact
saved-v28 full initial36 parity permits zero initial head refits. Only final
one ridge pipeline/four targets and three neural fits are new; no calibration
fits/forwards or old encoder replays occur.

One CUDA trajectory completed2048 new updates/B8 after2048 inherited updates on
the RTX A2000 8GB. GPU segment time131.75605979s includes checkpoint/witness/
loader-parity I/O. Initial/final feature I/O1.1259591679999998/0.782120961s and
final saved-head fitting/scoring/I/O2.177286234s stay separate. The reused
normalizer's historical fit2.929513154s is not new cost. Per-batch raw-loader
checks and CPU parent preprocessing prevent a fair speed comparison.

Fixed-query TRAIN/VALIDATION mean MAE changes from0.27862535204206196/
0.39052824676036835 to0.28209757645215305/0.3901873007416725, so it does not
improve uniformly. NLL first/last is0.3399173832780379/−0.27151144758246093;
1449 of2048 minibatches have valid negative NLL. Learned variance includes
mean error, not realized market volatility. The objective/distribution both
change, so this does not isolate a causal explanation of past failures.

All99 reusable pairs match locally:91 preserved plus8 new V30 pairs. The
integration CUDA fixture proves masking, actual mean/variance/projection
learning, exact parent32 and exact typed next-update optimizer/state resume.
Both fixture compiles passed; the first formatting warning was corrected
before the one passed GPU invocation. No standalone fixture/quality execution
is claimed. Independent saved arithmetic passes941,513 checks; report
verification binds198 assets,570 durable members and96 task rows:88 unchanged
historical,4 explicitly reused initial and4 new final. All55 checkpoint files
were rehashed and remain available, local, ignored and untracked.

Read the [V30 portable guide](LOCATION_SCALE_DECODER_V30.md),
[exact registry](location_scale_decoder_embedding_versions.json) and
[frozen market result](../../cuwacunu_dev/doc/MARKET_LOCATION_SCALE_DECODER_RESULTS.md).
Preserve the stopped V25–V29 mechanisms, their failed attempts, every checkpoint
and every original result/card/source capsule. The V29 reporting-only repair
remains historical; V30's producer/report each passed on their first invocation.

Next: the [separate frozen TEMPO-5 transfer card](../../cuwacunu_dev/doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md)
is declared, pending admission/extraction/scoring. It uses the exact saved V30
final inference checkpoint, zero encoder updates, both saved scaler policies,
the eight native36 head/view floors and full coverage. TEMPO-5 is the compatible
C3/H128/F9 slow lead/lag signal family, designed complexity5/5. V30 synthetic
quality is unmeasured. Keep RPB-v22 primary and native36 balanced until transfer
is verified. All later work remains local and unpublished; no primary promotion,
commit, push or publication is implied.

## Historical milestones and their original proposed next steps

The records below retain the evidence and status wording from their dates.
Their proposed auxiliary was completed and stopped as v25; v26 and v27 are
separate later experiments. These historical proposals are not instructions
to rerun any stopped recipe.

Latest completed diagnosis, 14 October 2026: saved TRAIN arithmetic explains
the joint-ridge fitting mechanism, without establishing the cause of worse
held-out generalization. Parent32 and its old predictor remain exact. Native36
TRAIN volatility linear R² improves0.767220→0.832289 despite its known validation
failure, and prefix coefficients shift L2 0.602036. Strong pairwise level/change
duplication is not observed (maximum|r|0.320435); full volatility contributions
reinforce overall. First56 TRAIN R² improves0.654690→0.772730; last56 falls
0.932627→0.910444. Both halves fitted the same head and are descriptive.

The [diagnosis guide](MARKET_TRAIN_VOLATILITY_DIAGNOSIS.md) links all four task
tables and the exact22-input evidence. Independent arithmetic passes15,428
checks;1,249 historical files remain unchanged. There were zero encoder calls,
updates, head/scaler fits or held-out tensor diagnostics. Keep RPB-v22 primary,
the native36 leading research candidate and its8/9 market disposition, all
checkpoints and frozen transfer evidence. No new encoder arithmetic was backported.

**Next proposed experiment:** preserve the existing signed-change and RMS/energy
moments while learning the same four-coordinate branch, by adding a balanced
moment reconstruction auxiliary to the current waveform objective. The
[next-direction note](../../cuwacunu_dev/doc/MARKET_CHANGE_ENERGY_NEXT_DIRECTION.md)
specifies scope and strict protection of the other market and TEMPO-5 gains.
It is not implemented or measured. Its exact recipe needs a separate instance,
card and typed-state/GPU checks before execution. Generic decorrelation is not
established as the remedy. These new diagnosis docs/evidence remain local;
earlier paragraphs preserve their milestone-local status and proposals.

Saved milestone, 14 October 2026: **RPB-v24.market-01.alt-01 native36 is the
leading research candidate; RPB-v22 remains the primary encoder.** Read
[the milestone](MILESTONE_2026-10-14_NATIVE36_TRANSFER.md) for completed evidence,
verified local checkpoints and the next saved-TRAIN-only volatility diagnosis.
The user requested documentation, commits and pushes of this completed work to
both existing remotes, including this public repository. That authorization
supersedes earlier local-only wording for this payload; licenses, visibility,
primary selection and ignored checkpoint storage stay unchanged.
No new experiment was run while saving this milestone.

Latest completed milestone, 14 October 2026: the exact frozen market-trained
**RPB-v24.market-01.alt-01**, native36, performs strongly on **TEMPO-5,
complexity 5/5**. With the original saved synthetic TRAIN scalers, mean
linear/neural accuracy is 98.96%/97.74% intact and 93.36%/92.53% under the saved
additional 30% erasure. Both heads and views exceed the reused synthetic-trained
RPB-v22.alt-01 results. The separately declared unchanged market-scaler view
reaches 100%/99.52% intact and 96.09%/95.05% deleted. Coverage is 100% throughout.
Three known cohorts each contain 512 TRAIN/256 VALIDATION examples; one exact
market checkpoint supplies B8 CUDA features, with three unchanged head repetitions.
There were zero encoder updates, scaler refits or old control reruns; all
model values, 91 protected files and 15 saved feature files remain unchanged.
Read the [transfer guide](MARKET_ENCODER_SYNTHETIC_TRANSFER.md) and
[full results](../../cuwacunu_dev/doc/MARKET_SYNTHETIC_TRANSFER_RESULTS.md).
Keep RPB-v22 primary, all checkpoints and the separate market volatility
failure. TEMPO-4/AMP-2 remain unmeasured for this checkpoint because their
C3/H32/F3 geometry differs. The next priority is the saved-TRAIN volatility
diagnosis; a new compatible hard family remains a separate future option.
Earlier sections preserve the evidence and next-action wording of each milestone.

Latest completed variation, 14 October 2026: **RPB-v24.market-01.alt-01**
preserves the trained v22 native32 prefix exactly and appends four learned
change coordinates, native36. Its [market result](../../cuwacunu_dev/doc/MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md)
reaches trend R² 0.655 linear / 0.134 mean neural, with all three neural fits
positive. Eight of nine unchanged conditions pass; only linear volatility
retention fails (0.446 versus v22's 0.728). Keep RPB-v22 primary and every
v22/v23/native40/native36 checkpoint. This is a useful trend reference, not an
overall winner or promotion, forecasting, TEST or fresh cohort confirmation.

One CUDA trajectory completed 2048 new updates/B8 after 2048 parent updates,
with the same MKT-HIST-2 bank/scaler/targets. New initial/final saved banks use
the predeclared D36 fixed heads; all older results are reused. Training took
93.377385897 GPU seconds including checkpoint I/O. All 45 portable pairs
match locally, with one integration CUDA fixture proving masking, exact prefix,
actual learning and typed resume. No standalone fixture execution is claimed;
the later frozen synthetic transfer is recorded above. The [native36 guide](SEPARATED_CHANGE_NATIVE36.md) and
[new instance registry](separated_change_native36_embedding_versions.json)
record geometry, settings, evidence and local checkpoint hashes; the original
[native40 registry](separated_change_embedding_versions.json) remains intact.

**Next proposed work:** saved-TRAIN-only diagnosis using existing level/change
covariance and fitted linear contribution/prediction archives to investigate
the linear volatility loss. No encoder updates, head refits, old replays or
VALIDATION/TEST diagnostic inputs. Declare a separate new objective or branch
design only after that diagnosis; no width, rate, budget or head rescue.
Fresh chronology follows a useful balanced candidate. This diagnosis is not
measured or scheduled. The earlier next-action wording below is milestone-local;
follow [integration continuation](../../cuwacunu_dev/doc/CONTINUATION.md).
The save authorization above covers the completed source/docs/evidence payload;
checkpoint weights remain local and ignored.

For a new session, read the [14 October handoff](HANDOFF_2026-10-14.md) and its
full integration counterpart before choosing work or loading checkpoints.

Latest completed attempt,14 October2026: **RPB-v24.market-01** appends8 learned
change coordinates and a separate residual decoder to an exact frozen trained
v22 native32 prefix, yielding native40. Its [market result](../../cuwacunu_dev/doc/MARKET_SEPARATE_CHANGE_RESULTS.md)
improves linear trend to0.599 and neural mean to0.066, exceeding the useful
v23 linear-trend reference0.387. Activity improves. Volatility/relation miss
retention;5of9 preferences pass, so keep RPB-v22 primary and all instances
intact. Training improves trend markedly versus the initialized new block.
D40 changes head counts/width-derived seeds; initial/final share those settings.
No new TEST, forecasting, synthetic quality or cohort confirmation.

Six portable headers, settings and the focused fixture match integration;
its CUDA gate checks exact parent32/native40, masking/semantic order, actual
branch learning and exact typed resume. No second standalone fixture run is
claimed. The [portable guide](../code/encoders/raw_patch_bottleneck_mae/SEPARATED_CHANGES.md)
and [instance registry](separated_change_embedding_versions.json) identify
contracts and checkpoints. Next propose a separately declared smaller change
block, retaining the saved v24 trend result; follow integration continuation.
This is not implemented or scheduled. Earlier completed records follow.

Latest completed variation,14 October2026: **RPB-v23.market-01.alt-01** holds
trained RPB-v22.market-02's level/relation encoder and decoder exactly fixed
while learning only its new change projection for2048 additional CUDA updates.
The [result](../../cuwacunu_dev/doc/MARKET_FROZEN_CHANGE_RESULTS.md) misses the
development goal:4of8 conditions pass; trend remains negative and linear
volatility/relation fall. Activity/neural relation improve. Keep v22 primary.
All parent/previous artifacts and native32 contracts remain preserved.

The new [portable freeze helper](../code/encoders/raw_patch_bottleneck_mae/FROZEN_LEVEL_TRAINING.md)
and fixture match integration; the focused CUDA protection/resume gate passes.
No second standalone fixture or synthetic confirmation was run. Its
[instance registry](frozen_change_embedding_versions.json) records the parent
and new checkpoint hashes. Next propose separate level/change coordinates,
preserving the old32 block exactly, with a declared larger dimension and
independent reconstruction path. Follow the integration continuation; this
architecture is not implemented. Earlier completed records follow.

Latest completed candidate,14 October2026: **RPB-v23.market-01** adds a
zero-start observed consecutive-change bank to unchanged v22 paths. The
[matched market experiment](../../cuwacunu_dev/doc/MARKET_CONSECUTIVE_CHANGE_RESULTS.md)
improves linear trend−0.384→0.387 and both relation heads; activity stays close.
Neural trend remains negative and volatility loses more than the predeclared
.05 allowance, so the full development preference fails and primary RPB-v22
remains selected. Six reusable headers, the CUDA fixture and settings match
integration; [portable guide](../code/encoders/raw_patch_bottleneck_mae/CONSECUTIVE_CHANGES.md)
records the optional API and checked contracts. Next protect useful level/
volatility information while retaining this change information under a new
prospective recipe. No old weights, data or head fits were replaced.

Latest completed milestone,14 October2026: integration's
[market duration/dimension study](../../cuwacunu_dev/doc/MARKET_TRAINING_STUDY_RESULTS.md)
finished five CUDA segments with independent saved-feature readouts. A512→2048
continuation did not solve trend and weakened volatility/relation. On the
separate matched MKT-HIST-2 panel, W128 improves several linear readouts;
D64/H256 show no consistent gain, and every trained configuration still has
negative trend R². No original instance or primary architecture changed.
Next investigate observed consecutive-candle changes while preserving the
useful level/activity paths and native32, under a new candidate tag and fixed
heads. The [integration continuation](../../cuwacunu_dev/doc/CONTINUATION.md)
defines that direction. The [25-pair check](../../cuwacunu_dev/doc/MARKET_STUDY_BACKPORT_CHECK.json)
passes; the new application runner/data helper stays outside this repository.

Latest additive milestone,13 October 2026: bounded streaming TRAIN statistics
and distinct v22 model/scaler/AdamW/counter resume headers/tests are locally
mirrored from `cuwacunu_dev`; the [portable support guide](STREAMING_TRAINING_SUPPORT.md)
records passing checks and exact pairs. The integration's first
[MKT-HIST-1 market pilot](../../cuwacunu_dev/doc/MARKET_HISTORY_PILOT_RESULTS.md)
is complete, without changing the v22 architecture or synthetic records.
RPB-v22.market-01 is a separate market instance: good linear volatility/activity
results, weak past trend, mixed neural heads,100% coverage. Its loader, targets,
data and runner stay in integration; no post-encoder PCA or new market default.
These private additions are synchronized locally and not published to the
public embedding remote. The next encoder candidate should focus on observed
changes while preserving the good level/activity representation.

Updated on 13 October 2026 after the completed content-only ordered relation
experiment and the user's application selection. RPB-v22.alt-01 passes the
two-cohort screen and third development check. The user has selected its RPB-v22
architecture as the primary encoder in sibling `cuwacunu`; v20 remains an
explicit reference. Original checkpoints and all earlier measurements remain
preserved. This application choice selects no trained market checkpoint.

## Implemented code and measured evidence

| Track | Status | Entry point |
| --- | --- | --- |
| RPB-v22.alt-01 | Content-only relation API selected as primary in `cuwacunu`; completed screen and third development check pass | [Report](CONFIGURABLE_CONTENT_RELATION_V1.md), [module](../code/encoders/raw_patch_bottleneck_mae/CONTENT_ORDERED_RELATION.md) |
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

The portable v22 API is synced into sibling `cuwacunu` and has passed its
focused artificial CUDA integration check. The user's subsequent primary
selection routes the public application API and `config/encoder.conf` to RPB-v22.
The original v20 settings are retained byte-for-byte as `config/encoder_v20.conf`;
the explicit `config/encoder_v22.conf` is also retained. The focused primary-entry
GPU check passed in the managed `cuwacunu` container: full W64/H128 mmap initialized
inference and typed reload, H129 with one actual CUDA update, and five preserved
v20/config/integrity checks. All 21 portable file pairs remain byte-exact. See the sibling
[primary selection record](../../cuwacunu_dev/doc/PRIMARY_ENCODER_SELECTION.md)
and [API sync manifest](../../cuwacunu_dev/doc/CONTENT_ENCODER_SYNC.json).
No synthetic trained checkpoint is selected for market use. Architecture and
saved quality are unchanged; there is no new confirmation or TEST measurement.

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

## Shared configurable v20/v22 geometry

Input values and observation masks have shape `[B,C,H,F]`; the served native global
embedding has shape `[B,D]`. B is the batch size, C the semantic channel count,
H the history length, F the features per tick, and D the embedding width.
Both portable model settings use C3/H128/F9/P8/D32, where P is the patch length.
The sibling application's primary settings now select the v22 architecture at
that geometry. These are model settings, not a chosen market dataset or candle
interval; the standalone v20 settings and historical CLI defaults stay fixed.

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
evaluation research. Its sibling `../cuwacunu_dev` owns the market loader,
CSV/cache/mmap storage, time-grid alignment and application adapter.
Reusable encoder improvements return here; market ingestion stays there.

The initial backport is `37c7a3b`; the stability backport is `9123891`.
Recorded checks cover strict config parsing, configurable CUDA shapes/masks,
exact old-shape v19 parity, typed model/scaler checkpoint round trip and immutable
checkpoint integrity. Tested shapes are correctness witnesses, not a maximum
input size or measured throughput/VRAM capacity.
The sibling [backport manifest](../../cuwacunu_dev/doc/CONFIGURABLE_ENCODER_BACKPORT.json),
[engineering validation](../../cuwacunu_dev/doc/CONFIGURABLE_ENCODER_VALIDATION.md) and
[stability review](../../cuwacunu_dev/doc/STABILITY_REVIEW_2026-10-12.md) retain receipts.
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
Market evidence is now limited to the separate MKT-HIST-1 pilot, and bounded
streaming scaler fitting is implemented. Record each new experiment with its frozen card, source/config
receipts, instance tags and saved report before updating status or registry pointers.

The harmonic linear decoder in the
[10 October continuation](CONTINUATION_2026-10-10_AFTER_SPECTRAL_CONFIRMATION.md)
is a preserved prospective architecture idea. It has no implementation or measured
quality and is not the default next action after the v20 integration.

Market work continues separately in the sibling
[integration continuation](../../cuwacunu_dev/doc/CONTINUATION.md). Source/split
selection, streaming statistics, RAM/GPU measurements and a bounded pilot are
complete there; preserve the admitted market control and target its weak trend
information in a separately specified candidate.

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
Current search stage,15 October2026: integration's TRAIN scale diagnosis is
verified with zero model calls/fits and3,263,004 independent checks. See
`../cuwacunu_dev/doc/MARKET_CHANGE_SCALE_DIAGNOSIS_RESULTS.md` relative to the
repository root. RPB-v26.market-01 is prospectively declared in integration's
`doc/MARKET_CONDITIONED_CHANGE_CARD.md`: frozen TRAIN coordinate mean/std
conditioning of the existing change bank. Admission/quality are pending;
reusable new files are being mirrored locally. Keep RPB-v22 primary, native36
the8/9 balanced reference, and the mirrored search ledger's individual leaders.
The continuing goal has no verified9/9 winner yet. Later work remains local.

Latest stage,15 October2026: [RPB-v26.market-01](TRAIN_CONDITIONED_CHANGE_V26.md)
is measured and stopped: original8/9, strict5/8, all trend repetitions and
coverage pass. Preserve its neural-volatility reference R²0.41926036121396154;
linear volatility and strict linear activity/neural relation fail. All7 ledger
entries and35 checkpoint pins are preserved, with64 portable pairs matched.
RPB-v22 stays primary; native36 stays the balanced reference. New source and
evidence remain local and unpublished.

The next [RPB-v27.market-01 card](../../cuwacunu_dev/doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md)
declares an unlabeled TRAIN parent-to-bank conditional map before the learned4
branch, reusing v26's frozen standardizer. Same budget/data/heads and protection
floors; necessary new frozen parent calibration forwards are explicit. Admission
and quality are pending. The mirrored [search ledger](ENCODER_SEARCH_LEDGER.md)
retains balanced and individual task/head leaders. Earlier notes below describe
the prior stages, not another primary selection.
