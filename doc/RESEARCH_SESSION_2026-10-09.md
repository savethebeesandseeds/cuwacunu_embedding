# RPB research session, 9 October 2026

Authorized work window: approximately 00:05–08:05 Asia/Dubai
(8 October20:05 through 9 October04:05 UTC). The working direction is RPB-v7;
the original five saved instances and all historical evidence remain frozen.

The first milestone is the prospective all-master saved-TRAIN reliability
diagnosis under [its card](../code/evaluation/cards/saved_native_reliability_v1.md).
It uses only the 85 declared TRAIN roles from RPB-v4.alt-01/RPB-v7.alt-01.
There are zero new encoder/decoder updates, zero head refits and zero held-out
analysis roles. CPU saved arithmetic runs in the existing managed container.

After that result, choose one bounded information-path variation if supported,
or a bounded new-source amplitude check if the diagnosis is inconclusive.
Register its separate design/instance identity and freeze its evaluation before
measurement. Admit actual CUDA execution before training. Keep native32,
fixed ridge/tanh16 heads, equal comparator budgets and clearly reported costs.
Preserve failed experiments and commit/push useful milestones. No deletion-rate
grid, head tuning, post-encoder PCA or rescue of the stopped v9 recipe.

The existing GPU container has been inspected and reused: NVIDIA RTX A2000 8GB
Laptop GPU, immutable container1e8de9cfec33. Build/toolchain objects remain in
the container's named `rpb-paired-pooling` build session. Shared evaluation
components have no encoder implementation dependency.

First milestone complete at approximately00:47 Dubai: all10 saved TRAIN
instances analyzed and independently verified (1,693,154 checks/130 archives).
The [diagnostic](../code/encoders/raw_patch_bottleneck_mae/SAVED_NATIVE_RELIABILITY_DIAGNOSTIC.md)
records all five pairs and preserved engineering failures. The weak fresh v7
run already has weaker TRAIN accessibility; mixed geometry/trace differences
do not identify a consistent architecture defect. Next is one separately frozen
new-source amplitude-transfer check with existing encoders/scalers on CUDA.

Second milestone complete at approximately01:45 Dubai under the frozen
[amplitude card](../code/evaluation/cards/frozen_amplitude_transfer_v1.md).
The [diagnostic](../code/encoders/raw_patch_bottleneck_mae/FROZEN_AMPLITUDE_TRANSFER_DIAGNOSTIC.md)
and [durable summary](results/frozen_amplitude_transfer_v1.json) retain all five
new cohorts, six methods and both views. Independent saved-arithmetic audit
passed38,784,404 checks; report metadata QA passed15,147 comparisons. All methods
have full coverage. Intact amplitude linear means are99.375% untrained,
99.21875% v4.alt-01 and98.59375% v7.alt-01; extra-deletion means are89.6875%,
96.875% and93.4375%. Training receives no intact amplitude-gain claim. No new
encoder/decoder updates, old classifier reuse, CPU model forward, TEST or stress.

The next bounded experiment is
[early-mixer-reliability-v1](../code/evaluation/cards/early_mixer_reliability_v1.md):
RPB-v10 moves aligned channel mixing before temporal blocks, compared with fresh
RPB-v7.alt-02 on five new timing cohorts. Both retain native32,225,805 parameters,
512 CUDA updates at batch8 and unchanged classifiers. Two distinct untrained
controls and a separate amplitude transfer task expose architecture and training
contributions. The ordinary original-patch query measures timing reconstruction;
there is no extra decoder optimization. Source/admission/reader gates precede
quality generation. Earlier mixing adds an unmixed temporal pass for independent
local exports, so added computation is measured. No result or promotion exists
for these planned groups yet.

Third milestone completed at approximately 02:51 Dubai. The fresh
[early-mixer diagnostic](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_RELIABILITY_DIAGNOSTIC.md)
passed 96,352,345 independent checks and 1,215 CPU archive decodes. Timing
linear means improve from 94.375% to 95.78125% intact and from 90.625% to
94.21875% with additional deletion. Worst-cohort timing also improves, but two
cohorts worsen and amplitude deletion accuracy declines. Reconstruction means
improve slightly while three individual cohorts worsen. V10 stays experimental;
there is no promotion, TEST, head tuning or post-encoder PCA. All ten fresh
encoders trained once on CUDA; 210 pipelines/420 heads fit once. Byte checks
preserve all 1,710 original V7 and calibration files unchanged.

Next is the separately frozen
[paired learning curve](../code/evaluation/cards/early_mixer_learning_curve_v1.md):
fresh RPB-v7.alt-03 and RPB-v10.alt-01 on masters 30130/31231/32332/33433/34534.
Each live optimizer saves 0/512/1024/2048 without reload or altered sampling,
loss, deletion policy or heads. Timing is scored at every positive budget;
amplitude uses only initial and final points. Controls fit once per task/cohort,
and each quality feature/query is extracted once after training. The design
tests repeatability and optimization progress without selecting the best point.

Before the learning-curve measurement, the existing pinned LibTorch 2.6.0+cu124
bundle was copied from its preserved transfer folder into the managed container
at `/opt/cuwacunu_embedding/libtorch`. All 8,821 regular files, totaling
4,633,956,070 bytes, were verified against the source. The installer records
inventory/proof hashes and preserves the original generated runtime profile.
Make now uses the internal SDK and captures its installer plus `setup.sh` in
source provenance. Actual CUDA admission and runtime dependency inspection were
repeated for this environment change; no quality experiment was repeated.
The new blocked independent reader v3 narrowly admits the captured setup script;
the earlier reader versions and admissions remain preserved.

The first encoder also passed its documented nine-row CUDA smoke sequence in
the fresh `mtf-sdk-cuda-20261009` build session: eight training updates, two
resumed updates and CUDA global/per-channel export. Runtime inspection resolves
the same internal SDK. The existing smoke target's extra CPU inference pass was
omitted. These ten artificial engineering updates are separate from the RPB
quality experiment; existing MTF builds and checkpoints were preserved.

Fourth milestone completed: the [continuous early-mixer learning curve](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md)
retains all 0/512/1024/2048 points for five new RPB-v7.alt-03/RPB-v10.alt-01
pairs. The [durable result](results/early_mixer_learning_curve_v1.json) records
every cohort, fixed-head repetition, conditional interval and complete training
trace. Timing linear means for late/early are 94.84375%/92.5% intact at 512,
93.28125%/94.375% at 1024 and 92.1875%/92.65625% at 2048. Additional-deletion
means are 89.375%/92.1875%, 87.1875%/92.96875% and 87.8125%/91.71875%.
All coverage is 100%. Final amplitude intact means are 99.84375%/97.5% and
deletion means 97.65625%/92.1875%. These tasks are reported separately.

Both groups reconstruct more accurately with longer training, without monotone
timing classification benefit. At2048, late/early TRAIN MAE is
0.0362045112/0.0360753016 and VALIDATION MAE is0.0373053915/0.0363039583.
Cumulative synchronized training-loop means are 54.1306847/56.9401724 seconds;
these loops include CPU trace capture and exclude checkpoint/state writing,
quality features/queries, verification and CPU classifier work. Earlier mixing
adds the independent unmixed temporal pass. Equal parameter count does not mean
equal compute. There was no separate decoder-only calibration.

The sole independent saved-arithmetic audit passed 141,317,815 checks and 1,675
CPU archive decodes in 548.2863 seconds. It verified 40 saved points,
270 pipelines/540 heads, 180 native exports,60 reconstruction writers/240
necessary query forwards and 163,840 sampled row exposures. One live AdamW and
scaler continue each trajectory; earlier bytes and state/trace prefixes are
rechecked after 2048 before one-time quality exports. All declared controls fit
once per task/cohort. No TEST/stress, best-point selection, promotion or
replacement of the original v7 group occurred; v4 remains active.

Capsule `early-mixer-learning-curve-FDv5b0` has inventory SHA
`b153296974bb15d03d181c455f67a79b486d980631ebc2484e5c6ffe5b345d85`.
Enclosing source/curve-adapter SHA is
`7bfda3c2d56716dc7d40c8835fb9f64f1df117969cd4603de3bfa846eb9632f8`;
the distinct core and learner identities remain in the durable JSON. Audit SHA
is `4bcd70194986a0207abcd2e721b6e5bac5cfa62c6b22bf47106d86a09858ebbe`.
Report SHA is `3fa32d02bdf92dc084a5a1c78a9d06c1d3412abaf11935195d740fc079833372`
and durable JSON SHA is
`2f661e1d74d5f13d9b3851149c05e119844d3085075dac78cfa1eb57a5f3409f`.
The prequality false-flag reader and earlier admissions remain preserved;
release changes only its two authorization flags after completed inventory.
The internal-SDK CUDA admission and the separate first-encoder CUDA smoke above
passed before this milestone was saved. No CPU model fallback or old-artifact
replacement was used.

Next, skip the half-mixer variation. Plan one fixed 512 comparison of the same
early architecture: RPB-v10.alt-02 as a new matched control and RPB-v11 with
TRAIN-only matched-target gain drawn log-uniformly over 0.5–2. Timing seeds are
41140/42241/43342/44443/45544; amplitude seeds are
46645/47746/48847/49948/51049. This is a planned direction, with no frozen card
or measured result yet. Its exact recipe and source/admission/reader gates must
precede new generation. Preserve all historical evidence and every curve point.

Fifth milestone completed at approximately 05:58 Dubai: the [matched-target gain
diagnostic](../code/encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
and [durable summary](results/matched_target_gain_v1.json) preserve all five new
RPB-v10.alt-02/RPB-v11 pairs at 512. Timing linear means fall 96.40625%→95.3125%
intact and 92.5%→84.6875% under additional deletion, with equal 100% coverage.
All six numeric guards fail; all five original-query TRAIN/VALIDATION errors
worsen. Mean MAE rises 0.075181/0.076729→0.216641/0.214659. Amplitude deletion
linear accuracy falls 97.5%→86.25%; strong untrained intact controls remain explicit.
Stop this gain recipe without range/rate/budget/head rescue; no promotion.

Actual CUDA admission admission-nrpfpL passed before quality generation. The sole
independent saved-arithmetic audit passed 88,154,488 checks/1,165 CPU archives in
358.420464 seconds. Emission peer metadata QA passed 169,228 comparisons; one
reviewed flag-only saver copied the exact report and JSON once, preserving all
eight approved metadata inputs. There were 180 pipelines/360 heads, 20 retained
points, 120 full native callbacks (90 quality + 30 initial counterpart witnesses),
20 query writers/80 necessary masked CUDA forwards and 40,960 sampled exposures.
Loop means 14.177860/14.335223 seconds include CPU evidence capture and candidate
target copies; they are not pure kernel timing. No extra decoder calibration,
CPU model replay, historical quality inputs, TEST/stress or best-point selection.

Capsule YmAtKW inventory is `e653ab95509af9eb321a0ff90e50f16f4566051894a3e4df4d162f55dda0f4f4`;
enclosing source is `1b734e72069f46aef80ecf9270edfa186aa50557b715ae3f101380b80f519215`.
Passed audit SHA is `9807e8439c60c3beefeb83431cd7660667a10aae449fbdc4b281c98b1b631682`.
Report SHA is `f88af56f6d7d9a4692d622b479613e76af2cb58f53ef02ff9b91767da21a7217`;
JSON SHA is `6afdc72e738b915c50043c92d436f9a4c1006c38e92f1746b4e7bbb5d8b7112c`.
The original false-flag reader/renderer/saver sources and source-only stale-pin
rejection remain preserved. No numeric acceptance requirement was changed.

The next bounded direction is one pooled-context information-path feasibility
step with a matched compact control, starting with source/card/CUDA engineering
gates. This milestone registers no new wider architecture or quality result.
The [end-window continuation](CONTINUATION_2026-10-09_AFTER_MATCHED_TARGET_GAIN.md)
records the protected references and explicit decision boundary.

Sixth milestone completed: the [pooled-context diagnostic](../code/encoders/raw_patch_bottleneck_mae/POOLED_CONTEXT_DIAGNOSTIC.md)
and [exact durable JSON](results/pooled_context_v1.json) preserve five fresh
RPB-v10.alt-03/RPB-v12 pairs at fixed512. Compact/pooled timing linear means
are 98.28125%/90.9375% intact and 97.03125%/83.125% deleted. Every cohort loses
in both views; all four timing numeric guards fail. Both mean reconstruction
guards pass: TRAIN 0.079615491/0.075831372 and VALIDATION
0.080886286/0.078075544. Coverage is100%. Amplitude favors the wider route on
these fresh cohorts, with strong untrained intact controls; all four panels,
all five cohorts and conditional intervals remain in the report. The joint
direction fails. Stop pooled width and gain without head/rate/budget rescue.

The route and first-layer capacity change together. The candidate registers
231,949 values versus225,805, with219,469 common values copied before AdamW,
2,080 inactive projection values and229,869 reconstruction-reachable values.
These results do not prove an isolated projection-loss mechanism. Ten joint
encoder/decoder trajectories complete512 at batch8 on RTX A2000 8GB CUDA;
there is zero extra decoder calibration. Synchronized loop means
13.827282377/13.886893782 seconds include CPU evidence capture and exclude
checkpoint/state writing, quality inference and CPU head/interval work.

Admission LyXM0F passed before the sole quality run. The independent saved-
arithmetic audit passed96,170,207 checks/1,235 CPU archives in402.684740278s;
emission metadata QA passed347,585 comparisons. The exact saver copied the
reviewed report/JSON once and preserved all eight admitted metadata inputs.
It verified210 pipelines/420 heads,120 distinct native exports,20 query
writers/80 necessary masked CUDA forwards,20 retained points and40,960 row
exposures. No historical payload, TEST/stress, CPU encoder/optimizer/PCA/head
rerun, best-point selection or promotion occurred.

Capsule Na1nEQ inventory SHA is
`66a86efd911b5c98485b8540120c348105fb7be6853e9180b9733c53e4b3d700`;
enclosing SOURCE SHA is
`be129553c8712415d3f009dce8b2131c88123ff2e39f3822baf3bb55e0310059`.
Audit SHA is `6820cd40ec4ff79a715ff7521a4ff1315c78908ae81c8d2fa8e27395b7f23964`.
Report SHA is `3fc19814bfb5976a04e7335df2f2c5cf6b5d1f4b38022496ab2ff73814378944`;
JSON SHA is `196c4197e7ab3b34ad91d4df0df58fcfb6ca50197a70c8f0156b4ef3515142e1`.
The original false-flag reader, reviewed fixture/seal, captured code and all
historical registry objects remain byte/digest exact. Source portability copies
are outside the174-path compiled closure: seven exact tools plus the reviewed
stager/manifest/README/proofs. One separate managed-container SOURCE stager
smoke created a fresh FALSE triple and passed27,229 artificial checks, with
zero payload/archive/model/head execution; it did not replace any current
reader pointer, reuse old fixtures or rerun the quality audit.

Root's next bounded direction is one fresh fixed512 compact early-v10
confirmation against late-v7, with a separate prospective card and matched
sources. Keep original v7 protected and v4 active. The
[current continuation](CONTINUATION_2026-10-09_AFTER_POOLED_CONTEXT.md) records
the stopped mechanisms and exact container-only setup; no next-run results
or new tag are invented here. Root's once-only final
[protected-capsule check](../output/runs/rpb-matched-target-gain/original-v7-preserved-20261009-end-window.json)
passed: all 1,710 files / 678,396,765 bytes are unchanged, including 1,483 original
v7 files / 547,772,582 bytes and 227 decoder-calibration files / 130,624,183 bytes.
Both original inventory identities remain exact. Proof SHA is
`46f97eea182aeb1d1e1bb44e572033545f66ea455eca7b079c9cca44e6773c84`.
There were zero tensor decodes, payload mutations, model/head executions or
original roles used for quality analysis. This closes preservation verification
without repeating encoder inference, fitting or an archive audit.
