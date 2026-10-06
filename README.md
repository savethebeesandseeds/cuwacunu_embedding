# cuwacunu_embedding

A standalone C++20/LibTorch representation workspace extracted from the segmented
MTF-JEPA-MAE-VICReg implementation. It converts masked multichannel histories into
global and per-channel embeddings and trains without the parent project's runtime,
DSL, registries, graph orchestration, forecasting, trading, or interface components.

The retained baseline uses multiscale time descriptors and windowed frequency
magnitudes, a shared token encoder, JEPA context/target masking and latent
prediction, an EMA teacher, auxiliary reconstruction, time/frequency alignment,
and global/channel VICReg stabilization. Experimental serving pools, alternative
mask policies, debug captures, and outer augmentation configuration are removed.
Pooling is the masked mean over all valid tokens; VICReg uses independent weak
views. JEPA retains the default legacy soft-overlap masking behavior. An optional
`strict_jepa_support=true` policy excludes all same-channel context windows that
intersect target support across scales and domains; it reduces or resamples
targets rather than relaxing the separation. This project does
not depend on or modify `cuwacunu_torch` at runtime.

The independent [Raw Patch Bottleneck MAE](code/encoders/raw_patch_bottleneck_mae/README.md)
now has its own model, configuration, CLI, checkpoint format and tests. It shares
archive utilities and frozen-feature evaluation components with the baseline.
The active research version is **RPB-v4 — Learned global bottleneck**. It trains
reconstruction through its exact native 32-number global embedding. The fresh
native-only timing TEST comparison gives 94.01% linear-head and 96.09% neural-head
accuracy, versus 51.82% / 79.51% for standalone raw PCA32, all at 100% coverage;
see the [native curve advance](code/encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md).
CUDA training and an independent artifact audit passed. Native linear VALIDATION
selected 512 updates before TEST; longer training improved reconstruction without
improving classification validation. The earlier
[native baseline comparison](code/encoders/raw_patch_bottleneck_mae/NATIVE_BASELINE_COMPARISON.md)
remains a separate archived TRAIN/VALIDATION record.
These are synthetic development results, not consumer acceptance.
Older designs remain available as archived references. See the
[version registry](doc/EMBEDDING_VERSIONS.md),
[results reporting standard](doc/RESULTS_REPORTING_STANDARD.md), and
[next advance plan](code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md).

Its [held-out reconstruction track](code/encoders/raw_patch_bottleneck_mae/DECODER_RELIANCE.md)
tests whether the compact vector carries trajectory information beyond
masks/positions/channel IDs. Targets, interventions and scoring belong to a
reusable shared engine; RPB training and decoding stay in its adapter.

## Start and build

From PowerShell in this directory:

```powershell
.\container.ps1 up
.\container.ps1 exec bash setup.sh
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Action exec -Command @('/opt/cuwacunu_embedding/build/baseline/embedding', '--help')
```

The managed container is `cuwacunu_embedding`, using the pinned Debian 12 image
digest in `container.ps1`, command `/bin/bash`, and this directory mounted at
`/embedding`. It has all GPUs available, 1 GiB shared memory, no published ports,
no named volumes, and restart policy `no`. Docker Desktop must have Linux
containers and NVIDIA GPU support available. Existing matching containers are
reused, including stopped ones; a conflicting container is preserved and rejected.

`setup.sh` installs the pinned Debian and NVIDIA dependency versions in
`dependencies.lock` with `--no-install-recommends` and checks LibTorch dependencies.
Its installation steps are adapted from `cuwacunu_torch/setup.sh`: C++ build/debug
tools (GCC, GDB, Valgrind, ccache, mold, clang-format), Git/cURL/shell tools,
CUDA toolkit 12.4.1 and cuDNN 9 from NVIDIA's Debian 12 repository.
It does not build, train, or manage containers. No Dockerfile is used.
`container.ps1 status`, `stop`, and `shell` provide the other lifecycle operations.
After entering the shell, commands below run directly from `/embedding`.
For arguments that PowerShell treats as its own options, pass an explicit array:
`./container.ps1 -Action exec -Command @('make', '-j12')`.

Build products and task temporary files live inside the container under
`/opt/cuwacunu_embedding/build/<session>`. Objects and test binaries follow the
source layout under its `code/` subdirectory; the CLI remains at
`/opt/cuwacunu_embedding/build/<session>/embedding`. Use `bash code/scripts/task.sh baseline`
followed by make arguments for this model; another session name isolates its
objects and executable. A nonblocking same-session lock refuses overlapping
tasks. Source files and GPU capacity are shared. Generated data, reports,
checkpoints and exports persist under `output/runs/<session>`; smoke and
evaluation runners create a fresh subdirectory for each invocation. Existing
`.build`, `.external` and `output` contents are retained. `.external/libtorch` is
the shared staged Linux library input, and `.build/reference` is the optional
read-only extraction-audit input. See [environment coordination](doc/ENVIRONMENT.md) for details.

Choose the model with an explicit Make target; session names only select build
and output directories. Inside the container, these can run in distinct chats:

```bash
bash code/scripts/task.sh baseline -j2 all test
bash code/scripts/task.sh rpb-mae -j2 rpb-mae test-rpb-mae
bash code/scripts/task.sh rpb-mae smoke-rpb-mae smoke-rpb-mae-cuda periodic-rpb-mae
bash code/scripts/task.sh comparison -j2 evaluation feature-harness test-feature-harness
```

The RPB binary is `embedding_raw_patch_bottleneck_mae`. The `evaluation` target
builds `embedding_evaluate`, which registers both encoder adapters; `feature-harness`
builds `feature_harness`, which registers only the baseline and minimum controls.
Executables live under the selected `/opt/cuwacunu_embedding/build/<session>`
directory. RPB prepare/train/embed has no evaluator dependency. Existing `all`,
`test`, `smoke` and `evaluate` targets continue to select the historical baseline.

The independent LibTorch `2.6.0+cu124` C++11 ABI bundle is staged in
`.external/libtorch`; it was copied from the existing local bundle. To transfer
this project, also stage that same Linux bundle at this path before setup. It
includes the CUDA/cuDNN runtime libraries used by the model. The system CUDA
toolkit and cuDNN packages additionally provide `nvcc` and development tools.
Bundled LibTorch libraries take precedence over system
cuDNN to keep runtime versions consistent. Interactive container shells load the
CUDA path from `/etc/profile.d/embedding.sh`; for direct execution without a shell,
use `/usr/local/cuda-12.4/bin/nvcc`. `.external`, `.build`, and generated archives are excluded from Git.
See `THIRD_PARTY_NOTICES.md` for binary license requirements.

## Train and generate embeddings

The small default configuration has 3 channels, 32 history steps, 3 input features,
and 12-dimensional embeddings. Synthetic data mixes periodic waves, changing
frequencies, shared channel signals, trends, noise, and missing features; it is a
training smoke fixture, not a claim of downstream representation quality.

```bash
# Generate a reusable dataset, then train on random minibatches from it.
/opt/cuwacunu_embedding/build/baseline/embedding synthetic --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --output output/data.pt --samples 32
/opt/cuwacunu_embedding/build/baseline/embedding train --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --input output/data.pt --checkpoint output/model.pt

# Export both embedding surfaces using the checkpoint's saved architecture.
/opt/cuwacunu_embedding/build/baseline/embedding embed --checkpoint output/model.pt --input output/data.pt --output output/embeddings.pt

# Restore weights, teacher, optimizer moments and completed-step counter.
/opt/cuwacunu_embedding/build/baseline/embedding train --resume output/model.pt --input output/data.pt --checkpoint output/continued.pt --steps 8

# Save atomically every 100 completed updates during a longer run.
/opt/cuwacunu_embedding/build/baseline/embedding train --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --input output/data.pt --checkpoint output/long.pt --steps 1000 --checkpoint-every 100

# Alternatively, train on newly generated synthetic batches, or use the GPU.
/opt/cuwacunu_embedding/build/baseline/embedding train --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --checkpoint output/synthetic.pt --steps 8
/opt/cuwacunu_embedding/build/baseline/embedding train --config code/encoders/mtf_jepa_mae_vicreg/config/default.conf --input output/data.pt --checkpoint output/cuda.pt --device cuda
```

`--steps` always means additional updates for that invocation. `--batch-size`
overrides minibatch size. Fresh training accepts `--seed`; resume restores its
saved seed and rejects `--config`/`--seed` to avoid an accidental architecture or
optimizer change. Pass the same `--input` when continuing a dataset; data is not
embedded in checkpoints. Omitting it deliberately selects synthetic training,
which requires `input_width=3`. Synthetic generation supports other channel counts
and history lengths.

The CLI seeds Torch with `seed + absolute_step` before each training update. This
makes CPU continuation reproducible for the same build, settings and data without
serializing opaque RNG state. It does not promise identical results across
devices or LibTorch versions. CPU is the default for embedding and resume; request
`--device cuda` explicitly for those commands. New training uses the config's
device. Training uses AdamW over trainable parameters, checks finite loss and
gradient norm, clips gradients when enabled, then updates the frozen teacher by
EMA. Checkpoints are written after the requested updates and, when
`checkpoint_every` is positive, at that interval of absolute completed steps.
Each periodic save atomically replaces the requested checkpoint path. Logging
includes gradient norm, context/target counts, and samples with no JEPA targets.

## Configuration

Configuration is plain `key=value` text with `#` comments, strict key names,
`true`/`false` booleans, and comma-separated integer lists. Unknown or duplicate
keys and invalid values fail early. `code/encoders/mtf_jepa_mae_vicreg/config/default.conf` is the readable small
baseline; omitted keys use `default_settings()` in `code/encoders/mtf_jepa_mae_vicreg/src/workflow.cpp`.

The model's scalar fields in
`code/encoders/mtf_jepa_mae_vicreg/include/embedding/encoders/mtf_jepa_mae_vicreg/config.h`
are accepted using their C++ names, along with `time_scales`, `scale_strides`,
and `device=cpu|cuda`.
`dtype` is fixed to float32 in the CLI. An empty `scale_strides=` derives strides
as half each time scale, with a minimum of one. `latent_dim` must be divisible by
`num_heads`. Architecture dimensions C/H/F must match the dataset exactly.

Training settings are:

| Key | Default | Meaning |
| --- | --- | --- |
| `steps` | 8 | Additional optimizer updates |
| `batch_size` | 4 | Training minibatch or embedding chunk size |
| `seed` | 101 | Nonnegative seed; per-step randomness uses seed + step |
| `threads` | 1 | Torch CPU worker threads |
| `log_every` | 1 | Print scalar losses every this many completed steps |
| `checkpoint_every` | 0 | Save every this many absolute completed steps; 0 saves only at completion |
| `learning_rate` | 0.001 | AdamW learning rate |
| `weight_decay` | 0.0001 | AdamW weight decay |
| `gradient_clip_norm` | 1.0 | Maximum gradient L2 norm; 0 disables clipping |

## Tensor archives and C++ API

Archives use LibTorch `torch::serialize::OutputArchive` / `InputArchive`, rather
than Python pickle dictionaries, CSV, or a custom dataset format. All public data
and output tensor fields are buffers (`write`/`read` third argument `true`).

Input keys:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `data` | float32 `[B,C,H,F]` | Batch, channel, history, feature |
| `feature_mask` | bool `[B,C,H,F]` | True for an observed feature |

Observed values must be finite. Masked values are replaced with zero before model
use; changing them cannot change the embeddings. Each training sample must contain
an observation. Embedding permits entirely missing samples/channels and reports
their validity separately. The loader keeps the dataset on CPU; the model moves
each minibatch to its configured device. Minibatches are sampled with replacement.
This deliberately small loader holds the dataset in memory.

For real sensors, center and scale each feature using statistics from the training
split only, and reuse those statistics for validation and serving. The ordinary
`train`/`embed` commands consume the supplied values without fitting preprocessing.
Descriptor reductions use float64 accumulation before returning the model dtype;
this avoids float32 offset cancellation and intermediate squared-value overflow.
Neural layers still use float32 and require sensible input ranges. `encode` fails
explicitly if neural computation produces nonfinite embeddings.

Example of writing your own data from C++:

```cpp
#include <torch/torch.h>

torch::serialize::OutputArchive archive;
archive.write("data", data.to(torch::kCPU).to(torch::kFloat32), true);
archive.write("feature_mask", observed.to(torch::kCPU).to(torch::kBool), true);
archive.save_to("input.pt");
```

Embedding output keys, all stored on CPU and in the original sample order:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pooled_embedding` | float32 `[B,D]` | Global embedding |
| `pooled_by_channel` | float32 `[B,C,D]` | Per-channel embeddings |
| `sample_valid_mask` | bool `[B]` | At least one valid token |
| `channel_valid_mask` | bool `[B,C]` | At least one valid token for this channel |

Read them with `archive.load_from(path, torch::kCPU)` followed by
`archive.read("pooled_embedding", tensor, true)`, and likewise for other keys.

For direct integration, include
`embedding/encoders/mtf_jepa_mae_vicreg/model.h` and construct `Model(config)`
using `Config` from `embedding::encoders::mtf_jepa_mae_vicreg`.
`encode(data, mask)` returns
token embeddings/masks/metadata as well as the two pooled surfaces. `forward`
returns the training losses; follow optimizer updates with
`update_target_network()`. Use `model->eval()` and `torch::NoGradGuard` for
inference. The model owns a copy of the config; choose its device before
construction. `embedding/encoders/mtf_jepa_mae_vicreg/workflow.h` supplies this
encoder's settings and checkpoint helpers. Generic inputs and archives live in
`embedding/shared/types.h` and `embedding/shared/data.h`. Link the encoder's
`src/workflow.cpp` and `src/evaluation.cpp` with `code/shared/src/data.cpp` and
`code/shared/src/evaluation.cpp` when using its workflow API; its `src/main.cpp`
owns the standalone CLI.

Checkpoint format version 1 contains the canonical settings as UTF-8 bytes,
completed-step counter, nested module state (including EMA teacher), and nested
AdamW state. `load_checkpoint` reconstructs the saved architecture and maps it to
the requested device. `load_optimizer` restores optimizer state after constructing
AdamW over the same ordered trainable parameters. Existing parent-project
checkpoints are not a supported import format.

## Source layout and verification

The root separates implementation in `code/`, documentation in `doc/`, and
generated artifacts in `output/`. Source within `code/` is organized by encoder,
with reusable code kept separately:

```text
code/
  shared/
    include/embedding/shared/
    src/
    tests/
  encoders/
    mtf_jepa_mae_vicreg/
      include/embedding/encoders/mtf_jepa_mae_vicreg/
      src/
      config/
      tests/
      scripts/                  # Original encoder's evaluation runner
    raw_patch_bottleneck_mae/
      include/embedding/encoders/raw_patch_bottleneck_mae/
      src/
      config/
      tests/
      README.md                 # Architecture, commands and validation links
  evaluation/
    src/main.cpp                # Separate comparison CLI and encoder registry
    cards/                      # Versioned controlled evaluation protocols
  scripts/                      # Shared container task dispatch
doc/                            # Paper sources and environment documentation
output/                         # Persisted generated artifacts
Makefile                        # Root build entry point
container.ps1                   # Managed container launcher
setup.sh                        # Dependency installation
dependencies.lock               # Pinned dependency versions
```

| Path | Responsibility |
| --- | --- |
| `code/shared/` | Reusable tensor operations, losses, archives, diagnostics, controlled protocols, provider/card contracts, fitting/scoring, comparison engine and tests |
| `code/encoders/mtf_jepa_mae_vicreg/` | Existing model, configuration, training/checkpoint/evaluation adapters and encoder regression tests |
| `code/encoders/raw_patch_bottleneck_mae/` | Independent RPB-MAE model, versioned raw/scaler/checkpoint/export workflow, evaluation adapter, configuration, tests and [implementation guidelines](code/encoders/raw_patch_bottleneck_mae/RPB_MAE_implementation_guidelines.md) |
| `code/evaluation/` | Separate comparison CLI, encoder-adapter registry and versioned protocol cards |
| `code/scripts/` | Shared container task dispatch; each encoder owns its CLI and end-to-end checks |
| `doc/` | Paper sources and environment coordination instructions |
| `output/` | Generated datasets, checkpoints, exports and reports |

Shared sources and tests compile without encoder include paths. Each encoder's
public includes begin with `embedding/encoders/<encoder>/`, and its C++ API lives
in the matching `embedding::encoders::<encoder>` namespace. Add
`code/shared/include` and the selected encoder's `include` folder to a consumer's
include directories. The flat forwarding headers and root-level encoder aliases
have been removed; C++ callers must use the explicit encoder API. Checkpoint
formats retain their existing versions and parameter registration names.
See [shared components](code/shared/README.md) and the
[encoder folder](code/encoders/mtf_jepa_mae_vicreg/README.md).

`make test` builds/runs the standalone model and workflow tests. `make baseline`
is an optional extraction audit against the original header copied into the local
`.build/reference` fixture; it is not needed for normal build, testing, or use.

`make smoke` exercises the complete CPU CLI workflow with the checked-in config:
generate data, train eight steps, resume two more, and export embeddings. Artifacts
are written to a fresh `output/runs/<session>/smoke-cpu-*` directory. `make smoke-cuda`
does the same on the GPU and also loads that checkpoint on CPU, writing to a fresh
`output/runs/<session>/smoke-cuda-*` directory.

Verified on 2026-09-13 with Debian 12, GCC 12.2.0, LibTorch 2.6.0+cu124,
and an NVIDIA RTX A2000 8 GB Laptop GPU:

- `make test`: shapes, missing inputs, masks, finite gradients, frozen teacher,
  EMA, checkpoint round-trip, strict config, optimizer continuation, and batched
  embedding export passed. The fixed-batch 12-step check reduced loss from
  1.83091 to 1.56688.
- `make baseline`: seeded CPU float32 parameters, embeddings, masks, losses,
  and gradients matched the original header on the deterministic test fixture.
  This checks the preserved baseline, not every possible model configuration.
- `make smoke smoke-cuda`: both devices completed eight training steps plus
  two resumed steps, then exported embeddings for nine inputs. The CUDA-trained
  checkpoint also loaded and generated embeddings on CPU.

The earlier encoder/shared split was verified on 2026-10-04 in container session
`structure`. The model and shared test suites, extraction audit, CPU/CUDA CLI
smoke checks, and interrupted periodic-checkpoint recovery passed. Exports from
preserved CPU/CUDA checkpoints, including CUDA checkpoints served on CPU,
matched all tensor buffers and archive metadata exactly. A CPU update resumed
from a preserved checkpoint also matched the pre-refactor executable's export
exactly. The evaluation runner passed with the relocated default configuration
and with a preserved checkpoint's original normalization archive. New artifacts
are retained under `output/runs/structure`.

The root `code/` and `doc/` layout was verified on 2026-10-04 in session
`root-layout`. All 43 C++ and configuration files matched their pre-move hashes.
Model/shared tests, extraction parity, CPU/CUDA smoke workflows and periodic
checkpoint recovery passed. The moved runners worked from another directory,
and the task runner refused overlapping work in the same session. Evaluation
passed with both the default config and a preserved checkpoint's normalization.
A preserved checkpoint's export matched its original schema, tensor metadata
and all four tensor buffers exactly. Artifacts are retained under
`output/runs/root-layout`; the move manifest is under `output/runs/reorganization`.

The subsequent source ownership cleanup on 2026-10-05 removed the forwarding
headers and migrated the original encoder's C++ namespace and callers. Final
encoder/shared regression suites and original-header extraction parity passed in
the managed container. The relocated baseline CPU/CUDA workflows, checkpoint
recovery and evaluation runner also passed. September CPU/CUDA checkpoints still
load: export schema and validity masks match exactly, while embedding values
agree within `rtol=1e-5, atol=1e-6` (maximum absolute difference `2.4e-7`).
See the [validation record](output/runs/source-ownership/validation.json) for
sessions, checks and retained artifacts. This is source-refactor evidence, not
representation-quality evidence.

The original `cuwacunu_torch` tracked tree remains clean, and its representation
header matches the unmodified extraction reference.

## Representation evaluation

The [embedding evaluation policy](doc/EMBEDDING_EVALUATION_POLICY.md) defines
shared evidence, protocol, reporting, and acceptance requirements for every
encoder. The workflow and historical measurements below describe the existing
baseline's protocol; they do not by themselves satisfy a new acceptance claim.

The separate [evaluation integration](code/evaluation/README.md) uses a shared
engine with encoder-owned adapters. Its instantiated versioned card owns input
geometry, channel IDs/units, tasks, splits, matched dimensions and compared-method
pairs. Adapter fitting receives training observations and permitted metadata;
labels, hidden clean signals and held-out observations stay inside evaluation.
Typed surfaces declare global, channel-concatenation or control semantics and
validity. Baseline channel constituents are contextual; RPB local constituents
are independent.

Build the `comparison` session above, then run inside the managed container:

```bash
bash code/scripts/task.sh comparison evaluate-minimum
bash code/scripts/task.sh comparison evaluate-rpb-mae
/opt/cuwacunu_embedding/build/comparison/embedding_evaluate --output output/new-comparison \
  --encoders baseline,rpb --baseline-steps 8 --rpb-steps 20 \
  --compare rpb_untrained_global,untrained_baseline_global,matched_global
```

The direct CLI requires a new output directory; runners allocate unique ones.
Each run persists its actual `evaluation-card.json`, report, fitted readouts and
adapter assets. Protocol v2 reports pair-specific common-valid populations and
unsupported fits. It provides development evidence, with incomplete acceptance,
cost and robustness requirements documented in the
[implementation review](doc/EVALUATION_IMPLEMENTATION_REVIEW.md). Earlier v1
reports retain their original interpretation.

The baseline's original `embedding evaluate` workflow follows:

The independent quality workflow is separate from the small synthetic smoke
fixture. Its regime labels and missingness are generated independently, and its
training and held-out histories use separate seeded trajectories. It compares
standardized raw inputs, time/frequency descriptors, a mask-only baseline, an
untrained encoder, and the trained encoder with the same ridge-probe protocol.

```bash
bash code/scripts/task.sh baseline evaluate
/opt/cuwacunu_embedding/build/baseline/embedding evaluate --config code/encoders/mtf_jepa_mae_vicreg/config/evaluation.conf --output output/evaluation/report.json --seeds 101,202,303 --steps 100
```

The JSON report records seeds, split provenance, training-only preprocessing,
individual training losses, held-out probe results, embedding diversity, validity
coverage, missing-sensor robustness, and conservative JEPA support overlap. It
measures the exported global and per-channel embeddings as well as projector
outputs. Small or singleton valid batches do not establish representation
diversity: the VICReg result exposes `statistics_supported=false` below two valid
rows, and differences between channel identities do not prove variation between
histories.

Fresh evaluation runs save a resumable checkpoint and normalized training/test
archives per seed beside the report. Input-normalization statistics are saved
with each checkpoint and must be reused when serving it. Evaluation of an existing
checkpoint uses its saved normalization archive; it does not refit the encoder's
input transform on the new evaluation histories. Training-data independence for
an externally supplied checkpoint remains the caller's responsibility.

```bash
/opt/cuwacunu_embedding/build/baseline/embedding evaluate --checkpoint output/evaluation/report.json.seed-101.pt --output output/evaluation/checkpoint-review.json --seeds 101
```

The default normalization path is `CHECKPOINT.normalization.pt`; use
`--normalization PATH` to supply the saved archive explicitly. The normalized
training archive saved by a fresh evaluation run can also be passed to ordinary
`train --resume` to continue that exact dataset.

Strict support masks leave other channels available unless
`mask_same_channel_block=true`. Audits measure window intervals per sample and
channel, merging repeated scales and domains. Because metadata lacks per-feature
observation support, these are conservative overlap estimates for sparse inputs.
Reports include target retention and samples with no feasible targets; a zero
overlap score must be interpreted together with those counts.

Time means/stds discard within-window ordering and frequency magnitudes discard
phase. The evaluator includes an explicit single-window information-retention
check; this is a structural demonstration, not a robotics performance result.
Use held-out probes and time-only/objective/masking ablations to decide whether a
richer raw-patch token branch is justified.

For real data, split by trajectory or session before constructing histories so no
raw observation occurs in two splits. Chronological splits also require a gap
covering the history and prediction horizons. Synthetic probe scores establish
only behavior on the controlled generator; they do not validate a real world model
or control system.

In the September 13 setup verification, after adding the original setup's
development tools, CUDA toolkit 12.4.1 and
system cuDNN 9.26.0.51, every package pin was verified and `dpkg --audit` was
clean. A fresh three-step CUDA training run and nine-sample embedding export
passed under the configured shell environment. These artifacts are in
`output/setup-gpu-check`; LibTorch continues to load its own bundled cuDNN.

Before the code/doc layout move, verification on 2026-10-04 passed all encoder/shared tests, the original-header
extraction audit, CPU/CUDA smoke and resume/export, periodic interrupted
checkpoint recovery, and evaluation with the relocated configuration and saved
normalization. Shared code built without encoder headers. Pre-move CPU/CUDA
checkpoint exports (including CUDA checkpoints served on CPU) and a one-update
CPU continuation matched the pre-move executable's tensor buffers exactly.
Evidence is retained under `output/runs/structure/`.

## Measured synthetic baseline (2026-10-04)

The quality benchmark used three seeds (101, 202, 303), 128 training and 64
held-out trajectories per seed, four balanced regimes, 100 optimizer updates,
batch size 32, and CPU execution with one Torch thread. Each comparison used
the same split, training-only input scaler and initial parameter fingerprints.
All probes used train-fitted coordinate scaling and the same fixed ridge penalty
of one. These are exploratory synthetic results; the ablations have exposed the
test set, so configuration selection needs a validation split and confirmation
on fresh test seeds.

| Configuration | Effective override | Global accuracy | Channel accuracy |
| --- | --- | ---: | ---: |
| Full objective | Evaluation defaults | 68.75% | 68.23% |
| Strict support | `strict_jepa_support=true` | 69.79% | 67.71% |
| Time only | `use_frequency_tokens=false` | 64.58% | 67.71% |
| Without JEPA | `use_jepa_loss=false` | 66.67% | 70.83% |
| Without MAE | `use_mae_decoder=false` | 70.31% | 68.75% |
| Without alignment | `use_tf_align_loss=false` | 69.27% | 71.35% |
| Without VICReg | `use_vicreg_loss=false` | 72.92% | 74.48% |
| Untrained full encoder | Same initial weights, no updates | 72.92% | 75.52% |

Values are means across seeds. The full trained global score has population
standard deviation 5.56 percentage points, versus 1.47 for its untrained control.
Removing VICReg improves channel scores relative to the full objective at all
three seeds, but still trails the untrained channel encoder. It ties the
untrained global mean. No tested condition establishes a consistent learned
advantage. Time-only also removes frequency reconstruction and time/frequency
alignment, making it a composite domain/objective ablation.

Raw descriptors scored 99.48%, standardized raw inputs 26.04%, and masks alone
23.44% (chance is 25%). Descriptors have 3,168 reported coordinates, including
zero padding/domain slots, versus 12 global or 36 concatenated channel
coordinates. This reveals available regime information, but is not a
dimension-matched compression comparison. Random phase also makes the raw-input
linear probe a limited measurement of information in the original histories.

The full model's served global effective ranks decrease from 1.63–2.13 before
training to 1.20–1.75 afterward, while standard deviations increase. This is
greater concentration rather than constant-vector collapse. More than 99% of
its flattened served channel-row population variance is between channel means;
within-channel variation across trajectories is much smaller. Global projector
mean dimension standard deviations are only 0.048–0.063 against a configured
variance floor of one. These findings motivate reviewing regularization
statistics; they do not justify simply increasing the existing VICReg weight.

Across three seeds, eight mask trials and 64 clean held-out rows per trial,
legacy masking overlapped context support for 63.74% of target tokens. Strict
masking produced zero measured same-channel overlap, retaining 60.84% of target
tokens and 71.24% of target temporal support, with no valid row losing every
target in this fixture. This verifies direct support separation, not improved
downstream utility. Different channels can remain statistically dependent,
full-window support is conservative for sparse observations, and fixed update
budgets expose strict training to fewer targets. Strict selection is heuristic
and may return no targets on other geometries when separation and minimum
context conflict. The legacy default remains appropriate for compatibility.

Outage probes reuse the original training-fitted probe on the clean version of
the held-out histories or its corruptions:

| Input condition | Trained global | Untrained global | Raw descriptors |
| --- | ---: | ---: | ---: |
| Clean, fully observed | 68.23% | 74.48% | 100.00% |
| 60% random dropout | 45.83% | 39.06% | 86.98% |
| 25% contiguous outage | 31.25% | 40.10% | 80.21% |
| 50% contiguous outage | 29.17% | 38.54% | 59.90% |
| One of three channels missing | 24.48% | 19.79% | 100.00% |

All these cases retain 100% sample validity; the gap concerns utility, not
whether the model produces an output. All-missing inputs correctly have zero
coverage and null scores. Training helps at 60% random dropout but structured
outages remain weak. The generator supplies redundant regime signals across
channels; this controlled stress test does not establish recoverability of
unique information from a missing real sensor.

The next bounded experiments should:

1. Add train-fitted descriptor PCA/random-projection controls at 12 and 36
   dimensions, and a validation split for probe/training-duration selection.
2. Compare current VICReg with statistics across trajectories within each
   channel, retaining the architecture, objective coefficients and training
   budget. Confirm candidates on fresh seeds, and keep served-space diversity
   diagnostics.
3. Diagnose the change in pooled embeddings when channels disappear, and test
   outage-matched training before selecting a different pooling rule.

The single-window reversal collision proves an ordering/phase limitation in
that deliberately restricted tokenizer fixture. It does not prove that the
default overlapping multiscale model is reversal-invariant or explain the
current regime gap. A richer raw-patch branch should be evaluated on tasks
requiring that missing information; a larger alternative encoder remains
deferred.

Reproduce the full benchmark inside the managed container:

```bash
bash code/scripts/task.sh baseline -j2 all test baseline
bash code/encoders/mtf_jepa_mae_vicreg/scripts/evaluate.sh --config code/encoders/mtf_jepa_mae_vicreg/config/evaluation.conf \
  --seeds 101,202,303 --train-samples 128 --test-samples 64 --steps 100 --device cpu
```

For an ablation, copy the evaluation configuration and change the indicated
effective setting, preserving every other setting and the same CLI arguments.
The runner creates a fresh output directory. The original 21 runs completed
with finite losses and retain 84 checkpoint/data/scaler archives. Their local
[summary](output/evaluation/goal-01a107a5/summary.json), per-variant reports under
`output/evaluation/goal-01a107a5/`, and
[source hashes](output/evaluation/goal-01a107a5/source-sha256.txt) preserve the
evidence. Those runs preceded the folder refactor; the hash snapshot records
the source paths at that time. Generated reports and archives are excluded from
Git. The earlier encoder/shared folder move was separately validated against the original extraction
and checkpoints.
