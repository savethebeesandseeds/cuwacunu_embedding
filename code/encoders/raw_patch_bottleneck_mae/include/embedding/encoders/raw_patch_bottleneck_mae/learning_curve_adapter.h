// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/shared/learning_curve.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kLearningCurveContinuationSuffix = ".continuation.pt";
inline constexpr const char *kLearningCurveContinuationArtifact =
    "rpb_early_mixer_curve_continuation_state_v1";
struct LearningCurveStateWitnessOptions {
  bool enabled{false};
};

// Captures resolved CUDA settings once. The label-free fit callback clones only
// permitted training observations and fits one frozen scaler. Completed budgets
// are absolute; optimizer state and attempted counter streams are continuous.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings);

// Fresh continuous TRAIN only. Enabled defaults to fixed RPB-v6 .30; explicit
// coordinate15_v1 selects RPB-v7 .15 on the same mode2/mixer1/native32 architecture.
// balanced30_v1 selects RPB-v8: absolute even attempts are ordinary forward,
// odd attempts use the original .30 deletion. Ineligible attempts abort this recipe.
// Checkpoints retain a policy tag; ordinary workflow resume must reject them.
// No augmented resume API is exposed in this bounded experiment.
// Placement1 is additionally bound to the new early-mixer timing protocol and
// carries RPB-v10 architecture provenance; its historical snapshot callback
// rejects placement1. Use make_early_mixer_trainer for CUDA-only snapshots.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options);

// Additive live-state witness for the separately bound early-mixer curve only.
// Old overloads keep this disabled and preserve their saved artifact schemas.
// No optimizer is constructed or reloaded when capturing the witness.
embedding::evaluation::CurveTrainerFactory make_learning_curve_trainer(
    const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
