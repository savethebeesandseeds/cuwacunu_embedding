// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_harness.h"
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation {

struct FixedFeatureMethod {
  std::string name;
  FeatureSurface training, validation_intact, validation_deleted;
  // True only for caller-prepared TRAIN-normalized raw PCA components. The
  // caller persists that raw normalizer/PCA. The fixed probe still fits its own
  // TRAIN normalizer; this helper never instantiates PCA or an encoder.
  bool inputs_train_prepared{false};
  // Optional caller-owned preprocessing failure (e.g. raw PCA rank bound).
  // Preserve declared-width surfaces/coverage; fit neither head in this case.
  std::string preparation_unsupported_reason;
};

struct FixedFeatureReadoutRun {
  std::string output_directory; // New exclusive leaf; parent may already exist.
  uint64_t master_seed{0};
  torch::Tensor training_labels, validation_labels; // CPU int64, binary [B].
  std::vector<std::string> training_source_ids, validation_source_ids;
  std::vector<FixedFeatureMethod> methods;
};

// Fixed recipe: outer TRAIN FeatureNormalizer (unless already prepared), then
// RidgeProbe penalty1 and TinyProbe tanh16/100 Adam0.01. Repetitions2701/2802/
// 2903 use stream_seed(repetition,width), identical across equal-size methods.
// Fit once; save fit plus TRAIN/intact/deleted predictions; retained fit tensors
// remain exact. JSON retains unsupported fits/coverage and within-master source
// intervals. When named methods native_v4/native_v7 are both present, include
// common-valid v7-minus-v4 Ridge/Tiny paired intervals for each VALIDATION view.
std::string run_fixed_feature_readouts(const FixedFeatureReadoutRun &run);

} // namespace embedding::evaluation
