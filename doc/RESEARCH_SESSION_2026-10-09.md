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
