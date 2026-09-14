// SPDX-License-Identifier: MIT
#pragma once

#include <set>
#include <utility>
#include "embedding/types.h"

namespace embedding {

class JEPAContextTargetMasker {
public:
  explicit JEPAContextTargetMasker(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
  }

  [[nodiscard]] jepa_context_target_mask_t
  create_masks(const mtf_token_batch_t &batch) const {
    TORCH_CHECK(batch.token_mask.defined() && batch.token_mask.dim() == 2,
                "[embedding] token_mask must be [B,N]");
    const int64_t B = batch.token_mask.size(0);
    const int64_t N = batch.token_mask.size(1);
    auto valid_cpu = batch.token_mask.to(torch::kCPU).contiguous();
    auto target_mask_cpu = torch::zeros(
        {B, N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto context_mask_cpu = torch::zeros_like(target_mask_cpu);
    auto domain_cpu = batch.metadata.domain_id.to(torch::kCPU).contiguous();
    auto channel_cpu = batch.metadata.channel_id.to(torch::kCPU).contiguous();
    auto scale_cpu = batch.metadata.scale_id.to(torch::kCPU).contiguous();
    auto start_cpu = batch.metadata.start_index.to(torch::kCPU).contiguous();
    auto width_cpu = batch.metadata.width.to(torch::kCPU).contiguous();
    auto valid_acc = valid_cpu.accessor<bool, 2>();
    auto target_acc = target_mask_cpu.accessor<bool, 2>();
    auto context_acc = context_mask_cpu.accessor<bool, 2>();
    auto domain_acc = domain_cpu.accessor<int64_t, 1>();
    auto channel_acc = channel_cpu.accessor<int64_t, 1>();
    auto scale_acc = scale_cpu.accessor<int64_t, 1>();
    auto start_acc = start_cpu.accessor<int64_t, 1>();
    auto width_acc = width_cpu.accessor<int64_t, 1>();


    for (int64_t b = 0; b < B; ++b) {
      std::vector<int64_t> valid_indices;
      std::vector<int64_t> time_indices;
      std::vector<int64_t> freq_indices;
      valid_indices.reserve(static_cast<std::size_t>(N));
      for (int64_t n = 0; n < N; ++n) {
        if (!valid_acc[b][n]) {
          continue;
        }
        valid_indices.push_back(n);
        if (domain_acc[n] == 0) {
          time_indices.push_back(n);
        } else {
          freq_indices.push_back(n);
        }
      }

      const int64_t valid_count = static_cast<int64_t>(valid_indices.size());
      if (valid_count == 0) {
        continue;
      }
      const int64_t min_context =
          std::max<int64_t>(1, static_cast<int64_t>(std::ceil(
                                   config_.min_context_ratio * valid_count)));
      const int64_t max_target =
          std::max<int64_t>(0, valid_count - min_context);
      std::set<int64_t> target_set;

      if (!time_indices.empty() && max_target > 0) {
        const int64_t desired_time = std::max<int64_t>(
            1, static_cast<int64_t>(std::llround(config_.mask_ratio_time *
                                                 time_indices.size())));
        const int64_t block_len = std::min<int64_t>(
            desired_time, static_cast<int64_t>(time_indices.size()));
        const int64_t max_start =
            static_cast<int64_t>(time_indices.size()) - block_len;
        int64_t block_start = 0;
        if (max_start > 0) {
          block_start =
              torch::randint(0, max_start + 1, {1},
                             torch::TensorOptions().dtype(torch::kInt64))
                  .item<int64_t>();
        }
        for (int64_t i = 0; i < block_len; ++i) {
          target_set.insert(
              time_indices[static_cast<std::size_t>(block_start + i)]);
        }
      }

      if (!freq_indices.empty() && max_target > 0 &&
          config_.mask_ratio_frequency > 0.0) {
        const int64_t desired_freq = static_cast<int64_t>(
            std::llround(config_.mask_ratio_frequency * freq_indices.size()));
        add_random_subset(freq_indices, desired_freq, target_set);
      }

      if (config_.mask_ratio_channel > 0.0 && max_target > 0) {
        for (int64_t c = 0; c < config_.channel_count; ++c) {
          const double draw = torch::rand({1}).item<double>();
          if (draw > config_.mask_ratio_channel) {
            continue;
          }
          for (const auto idx : valid_indices) {
            if (channel_acc[idx] == c) {
              target_set.insert(idx);
            }
          }
        }
      }

      if (target_set.empty() && max_target > 0 && valid_count > 1) {
        target_set.insert(valid_indices.back());
      }

      if (!target_set.empty() && max_target > 0) {
        std::vector<int64_t> seed_targets(target_set.begin(), target_set.end());
        for (const auto target_idx : seed_targets) {
          for (const auto candidate_idx : valid_indices) {
            if (candidate_idx == target_idx ||
                target_set.find(candidate_idx) != target_set.end()) {
              continue;
            }
            const bool same_window =
                channel_acc[candidate_idx] == channel_acc[target_idx] &&
                scale_acc[candidate_idx] == scale_acc[target_idx] &&
                start_acc[candidate_idx] == start_acc[target_idx] &&
                width_acc[candidate_idx] == width_acc[target_idx];
            const bool cross_domain =
                domain_acc[candidate_idx] != domain_acc[target_idx];
            const bool same_channel =
                channel_acc[candidate_idx] == channel_acc[target_idx];
            if (config_.mask_same_channel_block && same_channel &&
                static_cast<int64_t>(target_set.size()) < max_target) {
              target_set.insert(candidate_idx);
              continue;
            }
            if (config_.couple_time_frequency_masks &&
                config_.mask_same_window_across_domains && same_window &&
                cross_domain &&
                static_cast<int64_t>(target_set.size()) < max_target) {
              target_set.insert(candidate_idx);
            }
          }
        }
      }

      if (static_cast<int64_t>(target_set.size()) > max_target) {
        std::vector<int64_t> capped(target_set.begin(), target_set.end());
        target_set.clear();
        add_random_subset(capped, max_target, target_set);
      }

      std::set<int64_t> hard_forbidden_context;
      std::set<int64_t> soft_forbidden_context;
      auto recompute_forbids = [&](const std::set<int64_t> &targets,
                                   std::set<int64_t> &hard,
                                   std::set<int64_t> &soft) {
        hard.clear();
        soft.clear();
        for (const auto target_idx : targets) {
          for (const auto candidate_idx : valid_indices) {
            if (candidate_idx == target_idx ||
                targets.find(candidate_idx) != targets.end()) {
              continue;
            }
            const bool same_window =
                channel_acc[candidate_idx] == channel_acc[target_idx] &&
                scale_acc[candidate_idx] == scale_acc[target_idx] &&
                start_acc[candidate_idx] == start_acc[target_idx] &&
                width_acc[candidate_idx] == width_acc[target_idx];
            const bool cross_domain =
                domain_acc[candidate_idx] != domain_acc[target_idx];
            const bool same_channel =
                channel_acc[candidate_idx] == channel_acc[target_idx];
            if ((config_.couple_time_frequency_masks &&
                 config_.mask_same_window_across_domains && same_window &&
                 cross_domain) ||
                (config_.mask_same_channel_block && same_channel)) {
              hard.insert(candidate_idx);
              soft.erase(candidate_idx);
              continue;
            }
            const double overlap = interval_overlap_ratio(
                start_acc[target_idx], width_acc[target_idx],
                start_acc[candidate_idx], width_acc[candidate_idx]);
            if (overlap > config_.max_context_target_time_overlap &&
                hard.find(candidate_idx) == hard.end()) {
              soft.insert(candidate_idx);
            }
          }
        }
      };

      auto context_count_with = [&](const std::set<int64_t> &targets,
                                    const std::set<int64_t> &hard,
                                    const std::set<int64_t> &soft) {
        int64_t count = 0;
        for (const auto idx : valid_indices) {
          if (targets.find(idx) == targets.end() &&
              hard.find(idx) == hard.end() && soft.find(idx) == soft.end()) {
            ++count;
          }
        }
        return count;
      };

      recompute_forbids(target_set, hard_forbidden_context,
                        soft_forbidden_context);

      std::set<int64_t> empty_forbidden_context;
      while (!target_set.empty() &&
             context_count_with(target_set, hard_forbidden_context,
                                empty_forbidden_context) < min_context) {
        int64_t best_target = *target_set.begin();
        int64_t best_context_count = -1;
        int64_t best_hard_reduction = -1;
        for (const auto candidate_target : target_set) {
          auto reduced_targets = target_set;
          reduced_targets.erase(candidate_target);
          std::set<int64_t> reduced_hard;
          std::set<int64_t> reduced_soft;
          recompute_forbids(reduced_targets, reduced_hard, reduced_soft);
          const int64_t reduced_context = context_count_with(
              reduced_targets, reduced_hard, empty_forbidden_context);
          const int64_t hard_reduction =
              static_cast<int64_t>(hard_forbidden_context.size()) -
              static_cast<int64_t>(reduced_hard.size());
          if (reduced_context > best_context_count ||
              (reduced_context == best_context_count &&
               hard_reduction > best_hard_reduction)) {
            best_target = candidate_target;
            best_context_count = reduced_context;
            best_hard_reduction = hard_reduction;
          }
        }
        target_set.erase(best_target);
        recompute_forbids(target_set, hard_forbidden_context,
                          soft_forbidden_context);
      }

      auto context_count_after_forbid = [&]() {
        return context_count_with(target_set, hard_forbidden_context,
                                  soft_forbidden_context);
      };
      while (context_count_after_forbid() < min_context &&
             !soft_forbidden_context.empty()) {
        auto it = soft_forbidden_context.end();
        --it;
        soft_forbidden_context.erase(it);
      }

      for (const auto idx : target_set) {
        target_acc[b][idx] = true;
      }
      for (const auto idx : valid_indices) {
        const bool is_target = target_set.find(idx) != target_set.end();
        const bool is_forbidden =
            hard_forbidden_context.find(idx) != hard_forbidden_context.end() ||
            soft_forbidden_context.find(idx) != soft_forbidden_context.end();
        context_acc[b][idx] = !is_target && !is_forbidden;
      }
    }

    jepa_context_target_mask_t out{};
    out.context_mask = context_mask_cpu.to(batch.token_mask.device());
    out.target_mask = target_mask_cpu.to(batch.token_mask.device());
    out.valid_mask = batch.token_mask;
    return out;
  }

private:
  static void add_random_subset(const std::vector<int64_t> &indices,
                                int64_t desired,
                                std::set<int64_t> &target_set) {
    if (desired <= 0 || indices.empty()) {
      return;
    }
    const int64_t take =
        std::min<int64_t>(desired, static_cast<int64_t>(indices.size()));
    auto perm = torch::randperm(static_cast<int64_t>(indices.size()),
                                torch::TensorOptions().dtype(torch::kInt64));
    auto acc = perm.accessor<int64_t, 1>();
    for (int64_t i = 0; i < take; ++i) {
      target_set.insert(indices[static_cast<std::size_t>(acc[i])]);
    }
  }

  static double interval_overlap_ratio(int64_t start_a, int64_t width_a,
                                       int64_t start_b, int64_t width_b) {
    const int64_t end_a = start_a + width_a;
    const int64_t end_b = start_b + width_b;
    const int64_t overlap = std::max<int64_t>(
        0, std::min(end_a, end_b) - std::max(start_a, start_b));
    const int64_t denom = std::max<int64_t>(1, std::min(width_a, width_b));
    return static_cast<double>(overlap) / static_cast<double>(denom);
  }

  mtf_jepa_mae_vicreg_config_t config_{};
};

} // namespace embedding
