// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/learning_curve.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Captures resolved CUDA settings once. The label-free fit callback clones only
// permitted training observations and fits one frozen scaler. Completed budgets
// are absolute; optimizer state and attempted counter streams are continuous.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
