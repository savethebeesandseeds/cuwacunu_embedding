# cuwacunu_embedding

A standalone C++20/LibTorch representation model extracted from the segmented
MTF-JEPA-MAE-VICReg implementation. It converts masked multichannel histories into
global and per-channel embeddings and trains without the parent project's runtime,
DSL, registries, graph orchestration, forecasting, trading, or interface components.

The retained baseline uses multiscale time descriptors and windowed frequency
magnitudes, a shared token encoder, JEPA context/target masking and latent
prediction, an EMA teacher, auxiliary reconstruction, time/frequency alignment,
and global/channel VICReg stabilization. Experimental serving pools, alternative
mask policies, debug captures, and outer augmentation configuration are removed.
Pooling is the masked mean over all valid tokens; VICReg uses independent weak
views. JEPA retains the default legacy soft-overlap masking behavior; the
experimental support-separated mask repair is not included. This project does
not depend on or modify `cuwacunu_torch` at runtime.

## Start and build

From PowerShell in this directory:

```powershell
.\container.ps1 up
.\container.ps1 exec bash setup.sh
.\container.ps1 exec make -j12
.\container.ps1 exec make test
.\container.ps1 exec .build/embedding --help
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
.build/embedding synthetic --config config/default.conf --output output/data.pt --samples 32
.build/embedding train --config config/default.conf --input output/data.pt --checkpoint output/model.pt

# Export both embedding surfaces using the checkpoint's saved architecture.
.build/embedding embed --checkpoint output/model.pt --input output/data.pt --output output/embeddings.pt

# Restore weights, teacher, optimizer moments and completed-step counter.
.build/embedding train --resume output/model.pt --input output/data.pt --checkpoint output/continued.pt --steps 8

# Alternatively, train on newly generated synthetic batches, or use the GPU.
.build/embedding train --config config/default.conf --checkpoint output/synthetic.pt --steps 8
.build/embedding train --config config/default.conf --input output/data.pt --checkpoint output/cuda.pt --device cuda
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
EMA. Checkpoints are written after the requested updates; there is no background
scheduler or periodic checkpoint manager.

## Configuration

Configuration is plain `key=value` text with `#` comments, strict key names,
`true`/`false` booleans, and comma-separated integer lists. Unknown or duplicate
keys and invalid values fail early. `config/default.conf` is the readable small
baseline; omitted keys use `default_settings()` in `src/workflow.cpp`.

The model's scalar fields in `include/embedding/config.h` are accepted using their
C++ names, along with `time_scales`, `scale_strides`, and `device=cpu|cuda`.
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

For direct integration, include `embedding/model.h` and construct
`embedding::Model(config)` using `embedding::Config`. `encode(data, mask)` returns
token embeddings/masks/metadata as well as the two pooled surfaces. `forward`
returns the training losses; follow optimizer updates with
`update_target_network()`. Use `model->eval()` and `torch::NoGradGuard` for
inference. The model owns a copy of the config; choose its device before
construction. `embedding/workflow.h` supplies settings, synthetic data, archive,
and checkpoint helpers. Link `src/workflow.cpp` when using those helpers.

Checkpoint format version 1 contains the canonical settings as UTF-8 bytes,
completed-step counter, nested module state (including EMA teacher), and nested
AdamW state. `load_checkpoint` reconstructs the saved architecture and maps it to
the requested device. `load_optimizer` restores optimizer state after constructing
AdamW over the same ordered trainable parameters. Existing parent-project
checkpoints are not a supported import format.

## Source layout and verification

| Path | Responsibility |
| --- | --- |
| `include/embedding/config.h`, `types.h` | Validated model settings and tensor contracts |
| `tokenization.h`, `masking.h` | Time/frequency views and JEPA context/target selection |
| `encoder.h`, `objectives.h`, `tensor_ops.h` | Encoder, context attention, and loss/pooling helpers |
| `model.h` | Representation forward pass, encoding, EMA teacher |
| `workflow.h`, `src/workflow.cpp`, `src/main.cpp` | Data archives, configuration, checkpointing, CLI |
| `config/default.conf` | Small editable training baseline |
| `tests/` | Focused model, archive, resume and extraction checks |

`make test` builds/runs the standalone model and workflow tests. `make baseline`
is an optional extraction audit against the original header copied into the local
`.build/reference` fixture; it is not needed for normal build, testing, or use.

`make smoke` exercises the complete CPU CLI workflow with the checked-in config:
generate data, train eight steps, resume two more, and export embeddings. Artifacts
are written to `output/smoke-cpu`. `make smoke-cuda` does the same on the GPU and
also loads that checkpoint on CPU, writing to `output/smoke-cuda`.

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

The original `cuwacunu_torch` tracked tree remains clean, and its representation
header matches the unmodified extraction reference.

After adding the original setup's development tools, CUDA toolkit 12.4.1 and
system cuDNN 9.26.0.51, every package pin was verified and `dpkg --audit` was
clean. A fresh three-step CUDA training run and nine-sample embedding export
passed under the configured shell environment. These artifacts are in
`output/setup-gpu-check`; LibTorch continues to load its own bundled cuDNN.
