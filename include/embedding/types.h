// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/config.h"

namespace embedding {

struct mtf_input_t {
  torch::Tensor data{};         // [B,C,Hx,Dx]
  torch::Tensor feature_mask{}; // [B,C,Hx,Dx], bool
};

struct mtf_token_metadata_t {
  torch::Tensor start_index{}; // [N], int64
  torch::Tensor width{};       // [N], int64
  torch::Tensor scale_id{};    // [N], int64
  torch::Tensor channel_id{};  // [N], int64
  torch::Tensor domain_id{};   // [N], int64, 0=time, 1=frequency
};

struct mtf_token_batch_t {
  torch::Tensor tokens{};                           // [B,N,d_model]
  torch::Tensor reconstruction_targets{};           // [B,N,d_model], fallback
  torch::Tensor time_reconstruction_targets{};      // [B,N,2*Dx]
  torch::Tensor frequency_reconstruction_targets{}; // [B,N,K*Dx]
  torch::Tensor time_reconstruction_mask{};         // [B,N,2*Dx], bool
  torch::Tensor frequency_reconstruction_mask{};    // [B,N,K*Dx], bool
  torch::Tensor token_mask{};                       // [B,N], bool
  mtf_token_metadata_t metadata{};
};

struct jepa_context_target_mask_t {
  torch::Tensor context_mask{}; // [B,N], bool
  torch::Tensor target_mask{};  // [B,N], bool
  torch::Tensor valid_mask{};   // [B,N], bool
};

struct mtf_jepa_mae_vicreg_encode_output_t {
  torch::Tensor embeddings{};         // [B,N,latent_dim]
  torch::Tensor pooled_embedding{};   // [B,latent_dim]
  torch::Tensor pooled_by_channel{};  // [B,C,latent_dim]
  torch::Tensor pooled_time{};        // [B,latent_dim]
  torch::Tensor pooled_frequency{};   // [B,latent_dim]
  torch::Tensor token_mask{};         // [B,N]
  torch::Tensor sample_valid_mask{};  // [B]
  torch::Tensor channel_valid_mask{}; // [B,C]
  mtf_token_metadata_t metadata{};
};

struct mtf_jepa_mae_vicreg_output_t {
  torch::Tensor embeddings{};
  torch::Tensor pooled_embedding{};
  torch::Tensor pooled_by_channel{};
  torch::Tensor pooled_time{};
  torch::Tensor pooled_frequency{};
  torch::Tensor loss{};
  torch::Tensor loss_jepa{};
  torch::Tensor loss_mae{};
  torch::Tensor loss_mae_time{};
  torch::Tensor loss_mae_frequency{};
  torch::Tensor loss_tf_align{};
  torch::Tensor loss_vicreg{};
  torch::Tensor loss_vicreg_global{};
  torch::Tensor loss_vicreg_channel{};
  torch::Tensor jepa_target_mask{};
  torch::Tensor jepa_context_mask{};
  torch::Tensor sample_valid_mask{};
  torch::Tensor channel_valid_mask{};
};

struct mae_decoder_output_t {
  torch::Tensor projected{};
  torch::Tensor time{};
  torch::Tensor frequency{};
};

struct vicreg_stability_loss_options_t {
  double invariance_weight{25.0};
  double variance_weight{25.0};
  double covariance_weight{1.0};
  double variance_floor{1.0};
  double eps{1e-4};
};

struct vicreg_stability_loss_result_t {
  torch::Tensor loss{};
  torch::Tensor invariance_loss{};
  torch::Tensor variance_loss{};
  torch::Tensor covariance_loss{};
  int64_t valid_rows{0};
};

struct tf_alignment_result_t {
  torch::Tensor loss{};
  int64_t pair_count{0};
  int64_t pair_valid_count{0};
};

struct vicreg_branch_loss_result_t {
  torch::Tensor loss{};
  torch::Tensor global_loss{};
  torch::Tensor channel_loss{};
};

using EncodeOutput = mtf_jepa_mae_vicreg_encode_output_t;
using TrainingOutput = mtf_jepa_mae_vicreg_output_t;

} // namespace embedding
