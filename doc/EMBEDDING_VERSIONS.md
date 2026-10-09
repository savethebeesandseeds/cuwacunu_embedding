# Embedding version registry

RPB-v4 is the active research reference. RPB-v5's claimed advance failed the
fixed512 comparison. Retained versions provide historical evidence and compatible
loaders; routine experiments need not retrain all of them.
RPB-v6 keeps the v4 inference architecture with a fixed training policy that
removes some visible context while preserving targets. Its audited fixed512
comparison improves both linear accuracy primaries, but worsens reconstruction.
The declared reconstruction guard prevents promotion while that tradeoff remains.
Its completed [context optimization diagnostic](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_OPTIMIZATION_DIAGNOSTIC.md)
keeps that policy at 512, 1024 and 2048 updates on the same TRAIN and known
VALIDATION cohorts. No measured budget reaches the v4 fixed-512 reconstruction
reference while preserving v6's 512-update linear accuracy in both validation
views. At 2048, mean TRAIN/VALIDATION MAE is 0.035736/0.035845, but intact/deletion
linear accuracy falls to 96.875%/95.3125%. This diagnostic accessed no TEST and
made no promotion decision.

The completed [five-master replication](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_REPLICATION_ADVANCE.md)
measures fresh v4/v6 training at equal 512 updates. Mean linear gains are
+1.71875 percentage points intact and +8.125 under additional 30% coordinate
deletion, with equal 100% coverage. Both mean TRAIN/VALIDATION reconstruction
guards still fail; RPB-v4 remains active. Independent archive audit v3 passed
103,787,020 checks. Failed v1/v2 reader attempts and their corrections remain
preserved alongside the successful result.

| Tag | Very short description | Status | Configuration template |
| --- | --- | --- | --- |
| MTF-v1 | Multiscale JEPA and reconstruction | Retained first encoder | [MTF default](../code/encoders/mtf_jepa_mae_vicreg/config/default.conf) |
| RPB-v1 | Independent channel summaries | Archived reference | [RPB evaluation](../code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf) |
| RPB-v2 | Channel mixing and channel reconstruction | Archived reference | [Channel mixer](../code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf) |
| RPB-v3-mean | Averaged global bottleneck | Ablation reference | [Mean global](../code/encoders/raw_patch_bottleneck_mae/config/mean_global.conf) |
| **RPB-v4** | **Learned global bottleneck** | **Active experimental version** | [Learned global](../code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf) |
| RPB-v5 | Direct patch global bottleneck | Advance rejected at fixed 512, not promoted | [Direct patch global](../code/encoders/raw_patch_bottleneck_mae/config/learned_patch_global.conf) |
| RPB-v6 | Global bottleneck with context deletion | Accuracy gains, reconstruction tradeoff; not promoted | [Context deletion card](../code/evaluation/cards/context_deletion_v1.md) |
| RPB-v7 | Global bottleneck with lighter context deletion | Measured validation gains, unresolved reconstruction tradeoff; not promoted | [Lighter-policy diagnostic](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md) |
| RPB-v8 | Global bottleneck with balanced context views | Measured and audited; joint guard failed; not promoted | [Balanced-view diagnostic](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md) |
| RPB-v9 | Global bottleneck with native view agreement | Measured and audited; severe fixed512 failure; not promoted | [Native view agreement diagnostic](../code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md) |
| RPB-v10 | Global bottleneck with earlier channel mixing | Measured at 512 and in a continuous 2048-update comparison; experimental, not promoted | [Learning-curve diagnostic](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md) |
| RPB-v11 | Early mixer with matched-target TRAIN gain | Measured; all six numeric guards fail; recipe stopped | [Gain diagnostic](../code/encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md) |
| RPB-v12 | Pooled temporal summaries before global bottleneck | Measured; all four timing numeric guards fail; wider route stopped | [Pooled-context diagnostic](../code/encoders/raw_patch_bottleneck_mae/POOLED_CONTEXT_DIAGNOSTIC.md) |

The [machine-readable registry](embedding_versions.json) records the same mapping.
`RPB-v4.alt-01` names the fresh v4 controls paired with `RPB-v7.alt-01` on
masters9109/10210/11311/12412/13513. It is an instance-group alias for the
unchanged v4 design, and does not rename or replace a historical checkpoint.
Use only the tag in result table cells; place its short description in adjacent
prose. The number identifies a design milestone; it is not a quality score or
the checkpoint format version.

## Exact interpretation

The RPB templates share a native 32-number global export. Their training paths
differ:

| Tag | Channel mixer layers | Global bottleneck mode | Reconstruction signal | Historical report name |
| --- | ---: | ---: | --- | --- |
| RPB-v1 | 0 | 0 | Independent channel vectors | Independent RPB |
| RPB-v2 | 1 | 0 | Contextual channel vectors | `current_mixer` in global comparison |
| RPB-v3-mean | 1 | 1 | Exact valid-channel mean global vector | `mean_global` |
| RPB-v4 | 1 | 2 | Exact learned global vector | `learned_global` |
| RPB-v5 | 1 | 3 | Exact learned global vector from original patch states | N/A |
| RPB-v6 | 1 | 2 | Same exact learned global vector; context deletion during training | N/A |
| RPB-v7 | 1 | 2 | Same exact learned global vector; lighter training context deletion | N/A |
| RPB-v8 | 1 | 2 | Same learned global vector; alternating ordinary/deleted training views | N/A |
| RPB-v9 | 1 | 2 | Same learned global vector; ordinary reconstruction plus native view agreement | N/A |

With three channels, RPB-v1/v2 supply 96 signal numbers to reconstruction; their
32-number global mean is a diagnostic export. RPB-v3-mean/v4 reconstruct solely
from the exact 32-number global export plus fixed channel/patch metadata.
Channel concatenations are distinct diagnostic surfaces, not the main global
embedding. MTF-v1 dimensions depend on its resolved configuration.

RPB-v5 pools original aligned temporal/channel states of width64 and visibility
bits directly into32 numbers. It bypasses the separately compressed channel
summaries while retaining the same channel mixer and temporal blocks. Those
channel-summary projection/pooling modules remain diagnostic and receive no
reconstruction gradient in mode3. At C3/H32/F3/patch8, it has269,389 parameters,
versus225,805 for RPB-v4. The
[paired pooling experiment](../code/encoders/raw_patch_bottleneck_mae/PAIRED_POOLING_ADVANCE.md)
completed: v5 failed both declared Ridge primaries and TRAIN/VALIDATION
reconstruction at512. Its claimed advance is rejected under that frozen card;
this is not a rejection of the design under every possible budget.

The configuration files are templates. GPU/device, data schema, dimensions,
seeds, and update overrides must be recorded from the actual run. Choosing the
active research version does not change legacy loader defaults or invalidate
earlier checkpoints. In particular, an omitted global mode still means mode 0.

## Design tags and trained instances

The frozen working bundle is **RPB-v7**: the five original saved
instances on masters 4404/5505/6606/7707/8808, with their original checkpoints,
scalers and readouts. Their earlier intact timing VALIDATION means remain
97.96875% linear and 98.95833% neural. Optional decoder-calibration assets live
in a separate capsule and do not replace those checkpoints or encoder weights.

**RPB-v7.alt-01** names the five separately trained instances on new
sources/seeds9109/10210/11311/12412/13513. Their 92.96875%/95.26042% intact
means are results of another experiment, not changed scores for the saved
reference. The dataset and seeds differ, so these two result sets do not measure
the effect of a code change on the same weights and evaluation population.
The `.alt-NN` suffix identifies a separate run group of the same design;
both groups use the RPB-v7 architecture and recipe. The registry pins their
separate paths and inventory hashes. Use these short labels in comparisons,
with descriptions beside the table. Further repetitions use `.alt-02`, etc.

Keep the original RPB-v7 frozen while developing candidates. Any change to its
architecture, objective or training recipe requires a new design tag and new
output paths. Repeating its unchanged recipe uses a separate instance-bundle
identity and must never overwrite or silently replace the working reference.

A tag names the encoder design or a frozen training-view milestone. A trained instance also records the exact
resolved configuration/hash, source revision/fingerprint, exported surface,
training dataset and scaler IDs, seed, completed updates, and checkpoint
path/hash. A new seed, dataset, training budget, or head fit is another run of
the same design, not automatically a new architecture version. State dimension
or other configuration variants explicitly.

Register a new structural design or explicitly frozen training-view mechanism before measuring it, with a new tag and short
description. Never reuse an existing tag for a different pooling or reconstruction
path. Record promotion and archival decisions explicitly. Preserve old source,
configuration and checkpoint identities; labels are aliases rather than archive
format migrations or Git release tags.

RPB-v7 is an implemented and measured training-view milestone. It retains
v4/v6's mode2/mixer1/native32 inference and changes only the extra TRAIN deletion
request from 0.30 to 0.15. The frozen
[lighter-policy card](../code/evaluation/cards/context_lighter_validation_v1.md)
uses the separate identity `rpb-training-context-deletion-015-v1`. CUDA admission
preserves the v6 default/replay, explicit recipe binding and ordinary
tagged-resume rejection. The [diagnostic](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md)
is measured on five known TRAIN/VALIDATION masters; independent archive audit v3
passed 67,097,380 checks. Linear means improve over v6 by 2.65625 points intact and 1.25
under additional deletion, at 100% coverage. Both reconstruction guards against
v4 fail, neural means fall and one master loses deletion accuracy. V7 is not
promoted; no TEST/stress, selection or consumer acceptance is claimed.

RPB-v4 remains the active research reference; RPB-v5 is retained as a measured
fixed512 result and a completed separately frozen TRAIN/VALIDATION-only
continuation diagnostic at1024/2048. That follow-up did not reopen TEST/stress,
change the fixed512 disposition or promote the candidate. Retain RPB-v2's frozen evidence
for this completed milestone and existing compatibility tests. Later advances
compare against a named relevant checkpoint of RPB-v4, without automatically
bringing every older design back into training.

**RPB-v8 — Global bottleneck with balanced context views** is implemented and
measured under the frozen [balanced-view card](../code/evaluation/cards/context_balanced_validation_v1.md).
It alternates 256 ordinary and 256 deletion-0.30 attempts at 512 completed updates,
with unchanged mode2/mixer1/native32 architecture, original targets/loss and heads.
The [five-master diagnostic](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md) reports
mean native linear accuracy of 95.93750% intact and
91.71875% under the exact saved v7 additional-deletion view,
at 100% coverage. Mean TRAIN/VALIDATION MAE is
0.085022/0.087620.
The predeclared joint development guard **failed**. Independent archive audit v2
passed 67,099,959 checks; actual CUDA admission and prior-policy regressions passed.
V8 is not promoted, and RPB-v4 remains active. No TEST/stress, checkpoint selection,
head tuning or post-encoder PCA occurred. Five new v8 trajectories use ten
retained v4/v7 references; all five masters and both heads remain in the report.

The balanced-view implementation plan remains the prospective design record.
This diagnostic completes its fixed question; stop local schedule/rate/budget
tuning. The subsequent [TRAIN objective diagnosis](../code/encoders/raw_patch_bottleneck_mae/TRAINING_OBJECTIVE_DIAGNOSTIC.md)
rejects a weight1 all-pair residual-difference auxiliary: two v7 masters improve
the local fixed-head gradient direction and three worsen it. No encoder update
or head refit occurred; CUDA admission and independent saved-arithmetic audit
passed. This is not a new quality score or model version.

**RPB-v9 — Global bottleneck with native view agreement** is implemented and
measured in its [separate diagnostic](../code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md).
Its [plan](../code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_AGREEMENT_PLAN.md)
and quality card remain frozen prospective records. Five fresh CUDA runs at512
use the same v4 inference architecture, ordinary reconstruction and fixed heads.
All six numeric guards fail: mean native linear accuracy is 59.21875% intact /
57.03125% under extra deletion; mean TRAIN/VALIDATION MAE is
29.194490 / 29.371655. Master4404 is especially unstable; all other masters also
reconstruct substantially worse than v4. Actual CUDA admission and independent
audit passed 74,721,050 checks, establishing the recorded computations rather
than useful optimization. Stop this recipe without coefficient/rate/budget
rescue. RPB-v4 remains active; this RPB-v9 mechanism remains stopped.
The [original continuation note](CONTINUATION_2026-10-08.md) records the pre-diagnosis state.

Current work continues from RPB-v7. Its separate
[decoder calibration](../code/encoders/raw_patch_bottleneck_mae/V7_DECODER_CALIBRATION_DIAGNOSTIC.md)
freezes all encoder parameters, buffers and scaler, then optimizes only its
existing decoder for 128 CUDA updates. Mean TRAIN/known-VALIDATION MAE is
0.054039/0.056989; every run is below its paired v4 fixed512 reference.
Independent arithmetic audit and exact native32 invariants passed. Classification
scores are reused without fitting. This leaves the RPB-v7 embedding unchanged,
creates no new design tag, and does not revise historical fixed512 dispositions
or promote it. RPB-v4 remains the active reference. The new
[dated continuation](CONTINUATION_2026-10-08_AFTER_DECODER_CALIBRATION.md)
defines the prospective fresh-source replication; dated notes and reports remain
historical.

The completed [fresh equal-budget replication](../code/encoders/raw_patch_bottleneck_mae/FRESH_DECODER_REPLICATION_DIAGNOSTIC.md)
compares five new paired v4/v7 instances at encoder512 plus frozen decoder128.
V7/v4 intact linear means are 92.96875%/92.34375%, while additional-deletion
means are 89.84375%/82.03125%, all at 100% coverage. The intact effect interval
crosses zero, and v7's worst intact run is 80.46875% versus v4's 89.0625%.
Post-calibration v7 TRAIN/VALIDATION MAE is 0.056375/0.059200; v4 is lower at
0.053609/0.056835 with the same decoder budget. Recovery versus v4 before
calibration passes, but the joint development guard fails three of six numeric
conditions. Independent audit passed 67,932,331 checks. Encoder/native outputs
are unchanged by calibration; there is no new design tag or automatic promotion.
V7 remains the working direction and v4 the active reference. The
[historical continuation](CONTINUATION_2026-10-08_AFTER_FRESH_DECODER_REPLICATION.md)
prescribed the all-master saved-TRAIN reliability diagnosis completed below.

That [saved-TRAIN diagnosis](../code/encoders/raw_patch_bottleneck_mae/SAVED_NATIVE_RELIABILITY_DIAGNOSTIC.md)
is complete and independently verified: v4/v7 alternative-group linear means
94.92188%/94.45313%, neural means99.89583%/98.35938%, full coverage. The weak v7
master is already weaker on TRAIN; mixed geometry/trace differences do not
identify a common architecture defect. No encoder update, forward, head refit
or held-out analysis occurred. Its next step was the separately frozen new-source
amplitude-transfer check using frozen encoders and original timing scalers.

That [amplitude transfer check](../code/encoders/raw_patch_bottleneck_mae/FROZEN_AMPLITUDE_TRANSFER_DIAGNOSTIC.md)
is now complete, with38,784,404 independent audit checks. Both trained groups
preserve amplitude well; the intact untrained control already reaches99.375%
linear accuracy. With extra30% deletion, v4.alt-01/v7.alt-01 linear means are
96.875%/93.4375%, versus89.6875% untrained, with full coverage. These are new
data and head fits on reused encoders, with zero encoder updates. They do not
identify a timing mechanism or promote a model.

**RPB-v10 — Global bottleneck with earlier channel mixing** is measured in the
[early-mixer comparison](../code/evaluation/cards/early_mixer_reliability_v1.md).
Its existing aligned mixer acts before temporal blocks; independent local
exports retain a separate unmixed pass. Native32, parameter count, objective,
coordinate15 training policy and classifier recipes stay fixed. More compute
is measured separately. **RPB-v7.alt-02** is its fresh matched control
group on masters19119/20220/21321/22422/23523. The independently verified
[diagnostic](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_RELIABILITY_DIAGNOSTIC.md)
reports timing linear means 94.375%/95.78125% intact and 90.625%/94.21875%
with additional deletion, late/early respectively. Early mixing improves the
worst timing cohort but two cohorts worsen; amplitude deletion accuracy also
declines. Mean TRAIN/VALIDATION MAE is 0.071024/0.073853 for v10 versus
0.072575/0.074894 for the matched late group. Audit passed 96,352,345 checks.
V10 remains experimental without promotion; no existing group is replaced.

The [paired learning curve](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md)
is complete for fresh **RPB-v7.alt-03** and **RPB-v10.alt-01** groups. Each live
AdamW controller retains 0/512/1024/2048 points on all five timing cohorts.
Late/early intact linear means are 94.84375%/92.5% at 512,
93.28125%/94.375% at 1024 and 92.1875%/92.65625% at 2048. Additional-deletion
means are 89.375%/92.1875%, 87.1875%/92.96875% and 87.8125%/91.71875%,
respectively. Coverage is 100%. Both groups reconstruct more accurately with
longer training, without monotone timing classification improvement. At 2048,
late/early TRAIN MAE is 0.036205/0.036075 and VALIDATION MAE is
0.037305/0.036304. Final amplitude intact/deletion linear means are
99.84375%/97.65625% late and 97.5%/92.1875% early.

The independent audit passed 141,317,815 checks and 1,675 CPU archive decodes.
All 40 retained points, 270 pipelines/540 heads, 180 native exports and 60 query
writers are preserved in capsule FDv5b0. Fixed controls fit once per task/cohort;
quality exports occur once after continuation. Earlier fixed-512 evidence and
the original v7 group remain unchanged. No best-point selection, TEST/stress or
promotion occurred; v4 remains active.

The completed [matched-target gain comparison](../code/encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
uses **RPB-v10.alt-02** against **RPB-v11**, with the same early architecture and
fixed512 budget. Timing intact/deletion linear means are 96.40625%/92.5% for the
control and 95.3125%/84.6875% for gain; coverage is 100%. All six numeric guards
fail, and all five gain instances have worse original-query TRAIN/VALIDATION MAE.
Audit passed 88,154,488 checks. This fixed gain recipe is stopped, with no
range/rate/budget/head rescue. No prior group or curve evidence is replaced.

Half-mixer exploration remains skipped. The gain milestone's
[historical continuation](CONTINUATION_2026-10-09_AFTER_MATCHED_TARGET_GAIN.md)
planned the separate pooled-context feasibility step, subsequently completed
with its own frozen card and gates as recorded below.

## Pooled-context fixed512 milestone

The latest [pooled-context diagnostic](../code/encoders/raw_patch_bottleneck_mae/POOLED_CONTEXT_DIAGNOSTIC.md)
and [durable JSON](results/pooled_context_v1.json) register fresh **RPB-v10.alt-03**
compact early controls and **RPB-v12** pooled-width candidates on
53151/54252/55353/56454/57555. Both serve native32 and keep the original .15
training view. V12 pools contextual temporal W64 summaries before the global
bottleneck rather than the compact D32 projection. Route and first-layer
capacity change together: 231,949 registered values versus 225,805; 2,080
projection values remain inactive. The reconstructed route reaches 229,869
values, 4,064 more than the control. This comparison does not isolate a causal
effect of projection loss.

At512, compact/pooled timing linear means are 98.28125%/90.9375% intact and
97.03125%/83.125% with deletion, at 100% coverage. Every cohort loses timing
Ridge in both views. Worst-cohort scores are 93.75%/73.4375% intact and
89.0625%/53.125% deleted. Both mean original-query TRAIN/VALIDATION MAE guards
pass, while all four timing numeric guards fail. All amplitude panels and both
strong separate untrained controls are retained in the report, rather than
combined into a generic encoder score. Stop this wider route; no promotion.

The sole independent audit passed 96,170,207 checks/1,235 CPU archive decodes.
Every old version and instance-group object, original v7, active v4 and every
continuous curve point remain unchanged. Compact early v10 is the next
investigation candidate for one fresh fixed512 confirmation against late v7;
that direction has no new result or tag yet. The
[current continuation](CONTINUATION_2026-10-09_AFTER_POOLED_CONTEXT.md) records
the decision, and [portable SOURCE tools](../code/evaluation/tools/pooled_context_v1/README.md)
provide a fresh blocked reader/stager command for a clean checkout.
