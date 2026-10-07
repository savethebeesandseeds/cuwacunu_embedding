// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/shared/learning_curve.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Captures resolved CUDA settings once. The label-free fit callback clones only
// permitted training observations and fits one frozen scaler. Completed budgets
// are absolute; optimizer state and attempted counter streams are continuous.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings);

// Fresh continuous TRAIN only. Enabled defaults to fixed RPB-v6 .30; explicit
// coordinate15_v1 selects RPB-v7 .15 on the same mode2/mixer1/native32 architecture.
// Checkpoints retain a policy tag; ordinary workflow resume must reject them.
// No augmented resume API is exposed in this bounded experiment.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
