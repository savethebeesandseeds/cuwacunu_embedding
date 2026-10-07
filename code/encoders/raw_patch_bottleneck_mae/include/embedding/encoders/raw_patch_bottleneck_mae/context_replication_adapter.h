// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/paired_pooling.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Fresh native-development-v1/lag_sign only. The reference is a newly trained
// equal-budget v4, not a historical selected checkpoint. Verify the entire v4/
// v6 point-zero architecture, scaler, TRAIN association and original streams;
// separately verify the candidate's unchanged training-only context policy.
// No fitting, training, TEST access or input-hash claim: the shared driver owns
// explicit input byte protection. Inspection restores ambient RNG/thread state.
std::map<std::string, std::string> audit_context_replication_initialization(
    const std::string &candidate_point_zero,
    const embedding::evaluation::RetainedPoolingCohort &reference,
    const embedding::evaluation::ProviderFitInput &training_only,
    int64_t expected_reference_updates = 512);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
