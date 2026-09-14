// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/types.h"

namespace embedding {

namespace detail {

inline mtf_input_t
canonicalize_input(const torch::Tensor &x, const torch::Tensor &feature_mask,
                   const mtf_jepa_mae_vicreg_config_t &config) {
  TORCH_CHECK(x.defined(), "[embedding] input is undefined");
  TORCH_CHECK(x.is_floating_point(),
              "[embedding] input must be floating point");
  torch::Tensor data = x;
  torch::Tensor mask = feature_mask;
  if (data.dim() == 3) {
    TORCH_CHECK(config.channel_count == 1,
                "[embedding] rank-3 input requires "
                "channel_count=1");
    data = data.unsqueeze(1);
    if (mask.defined()) {
      mask = mask.unsqueeze(1);
    }
  }
  TORCH_CHECK(data.dim() == 4,
              "[embedding] input must be [B,Hx,Dx] or "
              "[B,C,Hx,Dx]");
  TORCH_CHECK(data.size(1) == config.channel_count &&
                  data.size(2) == config.history_length &&
                  data.size(3) == config.input_width,
              "[embedding] input shape does not match config");
  data =
      data.to(torch::TensorOptions().dtype(config.dtype).device(config.device));
  if (mask.defined()) {
    TORCH_CHECK(mask.sizes() == data.sizes(),
                "[embedding] feature_mask shape mismatch");
    mask = mask.to(
        torch::TensorOptions().dtype(torch::kBool).device(config.device));
  } else {
    mask = torch::ones(
        data.sizes(),
        torch::TensorOptions().dtype(torch::kBool).device(config.device));
  }
  mask = mask.logical_and(torch::isfinite(data));
  data = torch::where(mask, data, torch::zeros_like(data));
  return {data, mask};
}

inline torch::Tensor masked_mean(const torch::Tensor &x,
                                 const torch::Tensor &mask) {
  TORCH_CHECK(x.dim() == 3, "[embedding] x must be [B,N,D]");
  TORCH_CHECK(mask.dim() == 2, "[embedding] mask must be [B,N]");
  const auto mask_f = mask.to(x.dtype()).unsqueeze(-1);
  const auto denom = mask_f.sum(/*dim=*/1).clamp_min(1.0);
  return (x * mask_f).sum(/*dim=*/1) / denom;
}

inline torch::Tensor masked_mse(const torch::Tensor &a, const torch::Tensor &b,
                                const torch::Tensor &mask) {
  const auto mask_f = mask.to(a.dtype()).unsqueeze(-1);
  const auto denom =
      (mask_f.sum() * static_cast<double>(a.size(-1))).clamp_min(1.0);
  return ((a - b).pow(2) * mask_f).sum() / denom;
}

inline torch::Tensor masked_weighted_mse(const torch::Tensor &a,
                                         const torch::Tensor &b,
                                         const torch::Tensor &token_mask,
                                         const torch::Tensor &descriptor_mask) {
  TORCH_CHECK(a.sizes() == b.sizes(),
              "[embedding] weighted MSE tensor shape mismatch");
  TORCH_CHECK(token_mask.dim() == 2 && descriptor_mask.dim() == 3,
              "[embedding] weighted MSE mask rank mismatch");
  TORCH_CHECK(token_mask.size(0) == a.size(0) &&
                  token_mask.size(1) == a.size(1),
              "[embedding] weighted MSE token mask shape mismatch");
  TORCH_CHECK(descriptor_mask.sizes() == a.sizes(),
              "[embedding] weighted MSE descriptor mask mismatch");
  const auto weights = token_mask.to(torch::kBool)
                           .unsqueeze(-1)
                           .logical_and(descriptor_mask.to(torch::kBool))
                           .to(a.dtype());
  return ((a - b).pow(2) * weights).sum() / weights.sum().clamp_min(1.0);
}

inline torch::Tensor domain_mask(const mtf_token_metadata_t &metadata,
                                 const torch::Tensor &token_mask,
                                 int64_t domain) {
  auto domain_ids = metadata.domain_id.to(token_mask.device());
  return token_mask.logical_and(domain_ids.eq(domain).unsqueeze(0));
}

inline torch::Tensor
metadata_features(const mtf_token_metadata_t &metadata,
                  const mtf_jepa_mae_vicreg_config_t &config) {
  TORCH_CHECK(metadata.start_index.defined(),
              "[embedding] metadata is undefined");
  const auto opts =
      torch::TensorOptions().dtype(config.dtype).device(config.device);
  const auto denom_t =
      static_cast<double>(std::max<int64_t>(1, config.history_length));
  const auto denom_scale =
      static_cast<double>(std::max<int64_t>(1, config.time_scales.size() - 1));
  const auto denom_channel =
      static_cast<double>(std::max<int64_t>(1, config.channel_count - 1));
  const auto start = metadata.start_index.to(opts) / denom_t;
  const auto center =
      (metadata.start_index.to(opts) + 0.5 * metadata.width.to(opts)) / denom_t;
  const auto width = metadata.width.to(opts) / denom_t;
  const auto scale = metadata.scale_id.to(opts) / denom_scale;
  const auto channel = metadata.channel_id.to(opts) / denom_channel;
  const auto domain = metadata.domain_id.to(opts);
  return torch::stack({start, center, width, scale, channel, domain},
                      /*dim=*/1);
}

inline torch::Tensor finite_or_zero(const torch::Tensor &value) {
  return torch::where(torch::isfinite(value), value, torch::zeros_like(value));
}

inline torch::Tensor channel_mask(const mtf_token_metadata_t &metadata,
                                  const torch::Tensor &token_mask,
                                  int64_t channel) {
  auto channel_ids = metadata.channel_id.to(token_mask.device());
  return token_mask.logical_and(channel_ids.eq(channel).unsqueeze(0));
}

inline torch::Tensor
channel_valid_mask(const mtf_token_metadata_t &metadata,
                   const torch::Tensor &token_mask,
                   const mtf_jepa_mae_vicreg_config_t &config) {
  std::vector<torch::Tensor> masks;
  masks.reserve(static_cast<std::size_t>(config.channel_count));
  for (int64_t c = 0; c < config.channel_count; ++c) {
    masks.push_back(channel_mask(metadata, token_mask, c).any(/*dim=*/1));
  }
  return torch::stack(masks, /*dim=*/1);
}

inline torch::Tensor
pooled_by_channel(const torch::Tensor &embeddings, const torch::Tensor &mask,
                  const mtf_token_metadata_t &metadata,
                  const mtf_jepa_mae_vicreg_config_t &config) {
  std::vector<torch::Tensor> pooled;
  pooled.reserve(static_cast<std::size_t>(config.channel_count));
  for (int64_t c = 0; c < config.channel_count; ++c) {
    pooled.push_back(masked_mean(embeddings, channel_mask(metadata, mask, c)));
  }
  return torch::stack(pooled, /*dim=*/1);
}

} // namespace detail

} // namespace embedding
