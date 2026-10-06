# Embedding version registry

RPB-v4 is the active research version. Its learned global embedding is the next
development target. Retained versions provide historical evidence and compatible
loaders; routine experiments need not retrain all of them.

| Tag | Very short description | Status | Configuration template |
| --- | --- | --- | --- |
| MTF-v1 | Multiscale JEPA and reconstruction | Retained first encoder | [MTF default](../code/encoders/mtf_jepa_mae_vicreg/config/default.conf) |
| RPB-v1 | Independent channel summaries | Archived reference | [RPB evaluation](../code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf) |
| RPB-v2 | Channel mixing and channel reconstruction | Archived reference | [Channel mixer](../code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf) |
| RPB-v3-mean | Averaged global bottleneck | Ablation reference | [Mean global](../code/encoders/raw_patch_bottleneck_mae/config/mean_global.conf) |
| **RPB-v4** | **Learned global bottleneck** | **Active experimental version** | [Learned global](../code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf) |

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

With three channels, RPB-v1/v2 supply 96 signal numbers to reconstruction; their
32-number global mean is a diagnostic export. RPB-v3-mean/v4 reconstruct solely
from the exact 32-number global export plus fixed channel/patch metadata.
Channel concatenations are distinct diagnostic surfaces, not the main global
embedding. MTF-v1 dimensions depend on its resolved configuration.

The configuration files are templates. GPU/device, data schema, dimensions,
seeds, and update overrides must be recorded from the actual run. Choosing the
active research version does not change legacy loader defaults or invalidate
earlier checkpoints. In particular, an omitted global mode still means mode 0.

## Design tags and trained instances

A tag names the encoder design. A trained instance also records the exact
resolved configuration/hash, source revision/fingerprint, exported surface,
training dataset and scaler IDs, seed, completed updates, and checkpoint
path/hash. A new seed, dataset, training budget, or head fit is another run of
the same design, not automatically a new architecture version. State dimension
or other configuration variants explicitly.

Register a new structural design before measuring it, with a new tag and short
description. Never reuse an existing tag for a different pooling or reconstruction
path. Record promotion and archival decisions explicitly. Preserve old source,
configuration and checkpoint identities; labels are aliases rather than archive
format migrations or Git release tags.

Only RPB-v4 is the routine research candidate. Retain RPB-v2's frozen evidence
for this completed milestone and existing compatibility tests. Later advances
compare against a named relevant checkpoint of RPB-v4, without automatically
bringing every older design back into training.
