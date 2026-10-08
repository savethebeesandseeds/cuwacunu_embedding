// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kEarlyMixerProtocol = "early-mixer-reliability-v1";
inline constexpr const char *kEarlyMixerFitProtocol = "early-mixer-reliability-v1/lag_sign";
inline constexpr const char *kEarlyMixerFixtureFitProtocol =
    "early-mixer-reliability-engineering-v1/lag_sign";
inline constexpr const char *kEarlyMixerSnapshotArtifact = "rpb_early_mixer_cuda_snapshot_v1";
inline constexpr const char *kEarlyMixerSnapshotAuditFile = "early-mixer-snapshot-audit.pt";

// The quality scope admits only point0/512. Artificial CUDA engineering fixtures
// have a different namespace and explicitly admit only point0/2/4.
enum class EarlyMixerScope { quality, engineering };

struct EarlyMixerSnapshotOptions {
  std::string checkpoint_path;
  int64_t expected_channel_mixer_placement{-1};
  int64_t expected_completed_updates{-1};
  EarlyMixerScope scope{EarlyMixerScope::quality};
  std::string expected_core_source_fingerprint;
  std::string expected_training_producer_source_fingerprint;
};

// Fixed mode2/mixer1/native32/dropout0, unchanged coordinate15 training. Captures
// settings by value and delegates the existing live AdamW/counter training loop.
// Every callback preserves ambient RNG/threads. Snapshots use the new CUDA-only
// path; no historical CPU feature provider is constructed.
embedding::evaluation::CurveTrainerFactory make_early_mixer_trainer(
    const Settings &settings, EarlyMixerScope scope = EarlyMixerScope::quality);

// Exact original label-free timing TRAIN/scaler/source binding. Native features
// and ordinary original-Q reconstruction execute on one independently loaded,
// immutable CUDA model, returning detached CPU evidence. No inference erasure,
// optimizer, fitting, checkpoint writing or ordinary tagged resume is exposed.
embedding::evaluation::CurveSnapshot make_early_mixer_snapshot(
    const EarlyMixerSnapshotOptions &options,
    const embedding::evaluation::ProviderFitInput &original_training);

// Compare all initialized named parameters/buffers and frozen TRAIN scaler/data
// between declared late0 and early1 point0 snapshots. Features need not be equal.
std::map<std::string, std::string> audit_early_mixer_initialization(
    const EarlyMixerSnapshotOptions &late_point0,
    const EarlyMixerSnapshotOptions &early_point0,
    const embedding::evaluation::ProviderFitInput &original_training);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
