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
