# Embedding version registry

RPB-v4 is the active research reference. RPB-v5's claimed advance failed the
fixed512 comparison. Retained versions provide historical evidence and compatible
loaders; routine experiments need not retrain all of them.
RPB-v6 keeps the v4 inference architecture with a fixed training policy that
removes some visible context while preserving targets. Its audited fixed512
comparison improves both linear accuracy primaries, but worsens reconstruction.
The declared reconstruction guard prevents promotion while that tradeoff remains.

| Tag | Very short description | Status | Configuration template |
| --- | --- | --- | --- |
| MTF-v1 | Multiscale JEPA and reconstruction | Retained first encoder | [MTF default](../code/encoders/mtf_jepa_mae_vicreg/config/default.conf) |
| RPB-v1 | Independent channel summaries | Archived reference | [RPB evaluation](../code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf) |
| RPB-v2 | Channel mixing and channel reconstruction | Archived reference | [Channel mixer](../code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf) |
| RPB-v3-mean | Averaged global bottleneck | Ablation reference | [Mean global](../code/encoders/raw_patch_bottleneck_mae/config/mean_global.conf) |
| **RPB-v4** | **Learned global bottleneck** | **Active experimental version** | [Learned global](../code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf) |
| RPB-v5 | Direct patch global bottleneck | Advance rejected at fixed512, not promoted | [Direct patch global](../code/encoders/raw_patch_bottleneck_mae/config/learned_patch_global.conf) |
| RPB-v6 | Global bottleneck with context deletion | Accuracy gains, reconstruction tradeoff; not promoted | [Context deletion card](../code/evaluation/cards/context_deletion_v1.md) |

The [machine-readable registry](embedding_versions.json) records the same mapping.
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

RPB-v4 remains the active research reference; RPB-v5 is retained as a measured
fixed512 result and a completed separately frozen TRAIN/VALIDATION-only
continuation diagnostic at1024/2048. That follow-up did not reopen TEST/stress,
change the fixed512 disposition or promote the candidate. Retain RPB-v2's frozen evidence
for this completed milestone and existing compatibility tests. Later advances
compare against a named relevant checkpoint of RPB-v4, without automatically
bringing every older design back into training.
