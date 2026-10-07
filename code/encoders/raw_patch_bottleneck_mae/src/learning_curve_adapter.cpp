// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include <torch/cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[rpb learning curve adapter] " + message);
}

Input protocol_input(const embedding::Batch &batch, const Config &config,
                     const ev::ProviderFitInput &fit) {
  Input out{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(out, config);
  return out;
}

std::string source_manifest(const std::vector<std::string> &ids) {
  std::string text;
  for (const auto &id : ids) text += std::to_string(id.size()) + ":" + id;
  return text;
}

std::string manifest_identity(const std::string &text) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << "rpb-fit-source-manifest-fnv1a-v1-" << std::hex << std::setw(16)
      << std::setfill('0') << hash;
  return out.str();
}

std::string path_key(const std::string &path) {
  require(!path.empty(), "checkpoint path must be nonempty");
  return fs::weakly_canonical(fs::absolute(path)).string();
}

struct CurveState {
  Settings settings;
  ContextDeletionOptions context_options;
  ev::ProviderFitInput fit;
  Dataset training;
  FrozenScaler scaler;
  Model model{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::vector<torch::Tensor> initial_parameters;
  ev::CurveProgress progress;
  ev::CurveLossPoint last_loss;
  std::map<std::string, std::pair<int64_t, int64_t>> saved;
  std::array<int64_t, 3> context_counts{0, 0, 0}; // Requested, actual, restored.
  std::map<std::string, std::array<int64_t, 3>> saved_context_counts;
  std::map<std::string, std::string> audit;
  uint64_t initialization_seed{0};
};

ev::CurveProgress progress_report(const std::shared_ptr<CurveState> &state) {
  // Queried milestone endpoints are permanent entries, so subsequent reports
  // retain every previously delivered loss point as an exact positional prefix.
  if (state->progress.completed > 0 && (state->progress.losses.empty() ||
      state->progress.losses.back().completed != state->last_loss.completed))
    state->progress.losses.push_back(state->last_loss);
  auto out = state->progress;
  const auto parameters = state->model->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    require(torch::isfinite(parameters[i]).all().item<bool>(), "nonfinite CUDA model parameters");
    if (!torch::equal(parameters[i].detach(), state->initial_parameters[i]))
      out.weights_changed = true;
  }
  return out;
}

void write_text(torch::serialize::OutputArchive &archive, const std::string &key,
                const std::string &text) {
  archive.write(key, embedding::archive::text_tensor(text), true);
}

void save_point(const std::shared_ptr<CurveState> &state, const std::string &path) {
  const auto key = path_key(path);
  const std::vector<std::string> outputs{path, path + ".training-raw.pt",
      path + ".scaler.pt", path + ".audit.pt"};
  for (const auto &output : outputs) {
    require(!fs::exists(output) && !fs::is_symlink(fs::symlink_status(output)),
        "refusing to replace an existing point artifact: " + output);
  }
  Checkpoint checkpoint;
  checkpoint.settings = state->settings;
  checkpoint.model = state->model;
  checkpoint.scaler = state->scaler;
  checkpoint.attempted_steps = state->progress.attempted;
  checkpoint.completed_steps = state->progress.completed;
  checkpoint.schema_id = state->training.schema_id;
  checkpoint.dataset_id = state->training.dataset_id;
  checkpoint.scaler_fit_dataset_id = state->training.dataset_id;
  checkpoint.source_fingerprint = workflow_source_fingerprint();
  if (state->context_options.enabled)
    checkpoint.training_policy_id = context_deletion::descriptor(state->context_options.recipe).policy_id;
  save_checkpoint(path, checkpoint, *state->optimizer);
  save_dataset(path + ".training-raw.pt", state->training);
  save_scaler(path + ".scaler.pt", state->scaler, state->settings.model,
              state->training.schema_id, state->training.dataset_id);
  torch::serialize::OutputArchive audit;
  write_text(audit, "artifact_kind", "rpb_learning_curve_training_audit_v1");
  for (const auto &[name, value] : state->audit) write_text(audit, name, value);
  write_text(audit, "fit_source_manifest", source_manifest(state->fit.training_source_ids));
  audit.write("attempted_steps", torch::tensor(state->progress.attempted), true);
  audit.write("completed_steps", torch::tensor(state->progress.completed), true);
  audit.write("sampled_rows", torch::tensor(state->progress.sampled_rows), true);
  audit.write("channel_order", torch::tensor(state->fit.channel_ids, torch::kInt64), true);
  audit.write("sampling_interval", torch::tensor(state->fit.sampling_interval, torch::kFloat64), true);
  audit.write("endpoint", torch::tensor(state->fit.endpoint, torch::kFloat64), true);
  audit.write("training_seconds", torch::tensor(state->progress.training_seconds, torch::kFloat64), true);
  write_text(audit, "model_weight_update_budget", std::to_string(state->progress.completed));
  const auto report = progress_report(state);
  audit.write("weights_changed", torch::tensor(report.weights_changed, torch::kBool), true);
  audit.write("finite_gradients", torch::tensor(report.finite_gradients, torch::kBool), true);
  if (state->context_options.enabled) {
    audit.write("context_deletion_ratio_value",
        torch::tensor(context_deletion::descriptor(state->context_options.recipe).ratio, torch::kFloat64), true);
    audit.write("context_deletion_stream_value", torch::tensor(static_cast<int64_t>(context_deletion::stream)), true);
    audit.write("context_requested_deleted_coordinates", torch::tensor(state->context_counts[0]), true);
    audit.write("context_actual_deleted_coordinates", torch::tensor(state->context_counts[1]), true);
    audit.write("context_restored_coordinates", torch::tensor(state->context_counts[2]), true);
  }
  embedding::archive::save_archive(path + ".audit.pt", audit);
  state->saved.emplace(key, std::make_pair(checkpoint.attempted_steps, checkpoint.completed_steps));
  if (state->context_options.enabled) state->saved_context_counts.emplace(key, state->context_counts);
}

ev::CurveSnapshot snapshot_point(const std::shared_ptr<CurveState> &state,
                                const std::string &path) {
  const auto saved = state->saved.find(path_key(path));
  require(saved != state->saved.end(), "snapshot requires a point saved by this trainer");
  // Separate model construction is essential: core keeps its construction device
  // in Config, and held-out callbacks must never mutate the live training model.
  auto checkpoint = std::make_shared<Checkpoint>(load_checkpoint(path, state->settings.model.device));
  require(checkpoint->dataset_id == state->training.dataset_id &&
      checkpoint->schema_id == state->training.schema_id &&
      checkpoint->scaler_fit_dataset_id == state->training.dataset_id &&
      checkpoint->scaler.identity() == state->scaler.identity() &&
      settings_text(checkpoint->settings) == settings_text(state->settings) &&
      checkpoint->attempted_steps == saved->second.first &&
      checkpoint->completed_steps == saved->second.second,
      "saved point settings, scaler, training identity or counters differ from the trainer");
  require(checkpoint->training_policy_id == (state->context_options.enabled ?
      context_deletion::descriptor(state->context_options.recipe).policy_id : ""),
      "saved training policy differs from the trainer");
  checkpoint->model->eval();
  for (auto &parameter : checkpoint->model->parameters()) parameter.set_requires_grad(false);
  EvaluationOptions options;
  options.checkpoint_path = path;
  options.surface_prefix = "curve_checkpoint";
  const auto frozen = make_evaluation_provider(options)(state->fit);
  const auto base = std::string("curve_checkpoint_trained") +
      (state->settings.model.channel_mixer_layers > 0 ? "_contextual" : "");
  ev::CurveSnapshot snapshot;
  snapshot.features.provenance = frozen.provenance +
      "; exact saved continuous CUDA training point; no matched random provider surface";
  if (state->context_options.enabled) {
    const auto &selected = context_deletion::descriptor(state->context_options.recipe);
    snapshot.features.provenance += "; model_tag=" + std::string(selected.model_tag) +
        "; training_policy=" + selected.policy_id + "; inference remains ordinary mode2/mixer1/native32";
  }
  snapshot.features.audit_fields = frozen.audit_fields;
  for (auto field = snapshot.features.audit_fields.begin(); field != snapshot.features.audit_fields.end();) {
    if (field->first.rfind("curve_checkpoint_untrained_", 0) == 0)
      field = snapshot.features.audit_fields.erase(field);
    else
      ++field;
  }
  for (const auto &[name, value] : state->audit) snapshot.features.audit_fields[name] = value;
  snapshot.features.audit_fields["checkpoint_path"] = path;
  snapshot.features.audit_fields["attempted_steps"] = std::to_string(saved->second.first);
  snapshot.features.audit_fields["completed_steps"] = std::to_string(saved->second.second);
  snapshot.features.audit_fields["snapshot_policy"] =
      "independent_CPU_feature_model_and_CUDA_reconstruction_model;frozen_train_scaler;no_refit";
  if (state->context_options.enabled) {
    const auto &counts = state->saved_context_counts.at(path_key(path));
    snapshot.features.audit_fields["context_requested_deleted_coordinates"] = std::to_string(counts[0]);
    snapshot.features.audit_fields["context_actual_deleted_coordinates"] = std::to_string(counts[1]);
    snapshot.features.audit_fields["context_restored_coordinates"] = std::to_string(counts[2]);
  }
  snapshot.features.surfaces.emplace("curve_global", frozen.surfaces.at(base + "_global"));
  snapshot.features.surfaces.emplace("curve_channel_concatenation",
      frozen.surfaces.at(base + "_channel_concatenation"));
  snapshot.features.extract = [frozen, base, enabled = state->context_options.enabled,
      recipe = state->context_options.recipe](const embedding::Batch &batch) {
    const auto all = frozen.extract(batch);
    ev::FeatureMap result;
    result.emplace("curve_global", all.at(base + "_global"));
    result.emplace("curve_channel_concatenation", all.at(base + "_channel_concatenation"));
    if (enabled) {
      const auto &selected = context_deletion::descriptor(recipe);
      for (auto &[name, surface] : result) {
        (void)name;
        surface.provenance += "; model_tag=" + std::string(selected.model_tag) + "; training_policy=" + selected.policy_id;
      }
    }
    return result;
  };
  // Ordinary checkpoint, raw/scaler companions and producer audit were saved
  // already. Do not re-export the frozen provider's unrelated random control.
  snapshot.features.save_assets = [audit_fields = snapshot.features.audit_fields](
      const std::string &directory) {
    const auto destination = (fs::path(directory) / "curve-snapshot-audit.pt").string();
    require(!fs::exists(destination), "refusing to replace snapshot provenance");
    torch::serialize::OutputArchive audit;
    write_text(audit, "artifact_kind", "rpb_learning_curve_snapshot_audit_v1");
    for (const auto &[name, value] : audit_fields) write_text(audit, name, value);
    embedding::archive::save_archive(destination, audit);
  };
  // Only metadata are captured; reconstruction cannot access training rows.
  auto metadata = state->fit;
  metadata.training_observations = {};
  metadata.training_source_ids.clear();
  snapshot.reconstruct = [checkpoint, metadata](const embedding::Batch &batch,
                                                const torch::Tensor &hidden) {
    torch::NoGradGuard no_grad;
    const auto raw = protocol_input(batch, checkpoint->settings.model, metadata);
    const auto normalized = checkpoint->scaler.transform(raw, checkpoint->settings.model);
    const auto output = checkpoint->model->forward(normalized, hidden);
    require(output.reconstruction.is_cuda() &&
        torch::isfinite(output.reconstruction).all().item<bool>(),
        "snapshot reconstruction must be finite CUDA model output");
    return ev::CurveReconstruction{output.reconstruction.detach().to(torch::kCPU),
        normalized.data.detach().to(torch::kCPU), output.eligible_channels.to(torch::kCPU)};
  };
  return snapshot;
}
} // namespace

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings) {
  return make_learning_curve_trainer(settings, ContextDeletionOptions{});
}

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options) {
  context_deletion::validate_options(options);
  validate_settings(settings);
  require(settings.model.device.is_cuda(), "learning-curve training requires an explicit CUDA device");
  require(torch::cuda::is_available(), "CUDA is unavailable; CPU fallback is not permitted");
  require(!options.enabled || (settings.model.global_bottleneck_mode == 2 &&
      settings.model.channel_mixer_layers == 1 && settings.model.export_width == 32),
      "context deletion requires the unchanged mode2/mixer1/native32 architecture");
  return [settings, options](const ev::ProviderFitInput &fit) {
    require(fit.shape.channel_count == settings.model.channel_count &&
        fit.shape.history_length == settings.model.history_length &&
        fit.shape.input_width == settings.model.input_width,
        "resolved configuration dimensions differ from the declared fit card");
    require(static_cast<int64_t>(fit.channel_ids.size()) == settings.model.channel_count &&
        std::set<int64_t>(fit.channel_ids.begin(), fit.channel_ids.end()).size() == fit.channel_ids.size(),
        "fit must declare unique semantic channel IDs");
    const auto ids = resolved_channel_ids(settings.model);
    require(std::set<int64_t>(ids.begin(), ids.end()) ==
        std::set<int64_t>(fit.channel_ids.begin(), fit.channel_ids.end()),
        "resolved configuration semantic IDs differ from the fit card");
    require(!fit.feature_units.empty() && std::isfinite(fit.endpoint) &&
        std::isfinite(fit.sampling_interval) && fit.sampling_interval > 0 &&
        std::abs(settings.model.sampling_interval - fit.sampling_interval) <=
            1e-12 * std::max(1.0, std::abs(settings.model.sampling_interval)),
        "fit units, endpoint or sampling interval are missing/inconsistent");
    require(fit.training_observations.data.defined() &&
        fit.training_observations.data.dim() == 4 &&
        fit.shape.device.is_cpu() &&
        fit.training_observations.data.device().is_cpu() &&
        fit.training_observations.data.scalar_type() == fit.shape.dtype &&
        fit.training_observations.feature_mask.defined() &&
        fit.training_observations.feature_mask.device().is_cpu() &&
        fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)),
        "fit requires CPU training observations matching declared precision and source IDs");
    for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty permitted training source ID");

    auto state = std::make_shared<CurveState>();
    state->settings = settings;
    state->context_options = options;
    state->settings.seed = static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL);
    state->fit = fit;
    state->fit.training_observations.data = fit.training_observations.data.detach().clone();
    state->fit.training_observations.feature_mask = fit.training_observations.feature_mask.detach().clone();
    const auto raw = protocol_input(state->fit.training_observations, settings.model, state->fit);
    state->training = describe_dataset(raw, settings.model, fit.feature_units);
    state->scaler = fit_scaler(raw, settings.model);
    state->initialization_seed = training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL);
    torch::manual_seed(state->initialization_seed);
    // Construct with CUDA Config from the outset: Model::to alone cannot update
    // its private allocation/device contract. Module initialization remains CPU.
    state->model = Model(state->settings.model);
    state->model->train();
    state->optimizer = std::make_unique<torch::optim::AdamW>(state->model->parameters(),
        torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
    for (const auto &parameter : state->model->parameters()) {
      state->initial_parameters.push_back(parameter.detach().clone());
      state->progress.parameter_count += parameter.numel();
      if (parameter.is_cuda()) state->progress.cuda_parameter_count += parameter.numel();
    }
    require(state->progress.parameter_count > 0 &&
        state->progress.parameter_count == state->progress.cuda_parameter_count,
        "every model parameter must reside on CUDA");
    state->progress.training_device = settings.model.device.str();
    state->progress.preprocessing_id = state->scaler.identity();
    state->progress.training_dataset_id = state->training.dataset_id;
    state->audit = {{"provider_id", kEncoderId}, {"adapter_id", "rpb_continuous_cuda_learning_curve_v1"},
        {"protocol_id", fit.protocol_id}, {"fit_policy", "permitted_training_observations_only;scaler_fit_once"},
        {"resolved_settings", settings_text(state->settings)}, {"feature_units", fit.feature_units},
        {"initialization_seed", std::to_string(state->initialization_seed)},
        {"actual_training_seed", std::to_string(state->settings.seed)},
        {"rng_policy", "splitmix64-counter-rows-masks-torch-attempt-v1"},
        {"optimizer_policy", "one_continuous_AdamW_state;absolute_completed_update_budgets"},
        {"training_device", state->progress.training_device},
        {"parameter_count", std::to_string(state->progress.parameter_count)},
        {"cuda_parameter_count", std::to_string(state->progress.cuda_parameter_count)},
        {"output_semantics", output_semantics(settings.model)},
        {"reconstruction_export_semantics", reconstruction_output_semantics(settings.model)},
        {"training_dataset_id", state->training.dataset_id},
        {"scaler_fit_dataset_id", state->training.dataset_id}, {"preprocessing_id", state->scaler.identity()},
        {"fit_source_manifest_id", manifest_identity(source_manifest(fit.training_source_ids))},
        {"training_observation_rows", std::to_string(fit.training_source_ids.size())},
        {"training_source_groups", std::to_string(std::set<std::string>(
            fit.training_source_ids.begin(), fit.training_source_ids.end()).size())},
        {"core_writer_source_fingerprint", workflow_source_fingerprint()},
        {"training_producer_source_fingerprint", EVALUATION_SOURCE_ID},
        {"source_fingerprint_scope", "ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source"},
        {"trace_policy", "first_completed_update;every_log_every_updates;queried_milestone_endpoints_persist_as_cumulative_prefix"},
        {"sampling_policy", "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts"}};
    if (options.enabled) {
      const auto &selected = context_deletion::descriptor(options.recipe);
      state->audit.emplace("model_tag", selected.model_tag);
      state->audit.emplace("training_policy_id", selected.policy_id);
      state->audit.emplace("context_deletion_ratio", selected.ratio_text);
      state->audit.emplace("context_deletion_stream", "0x6374782d64726f70");
      state->audit.emplace("context_deletion_rng_policy", context_deletion::rng_policy);
      state->audit.emplace("context_deletion_repair_policy", context_deletion::repair_policy);
      state->audit.emplace("context_deletion_visibility_policy", context_deletion::visibility_policy);
      state->audit.emplace("context_deletion_count_policy", "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only");
      state->audit.emplace("context_deletion_resume_policy", "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API");
    }

    ev::CurveTrainer trainer;
    trainer.audit_fields = state->audit;
    trainer.train_to = [state](int64_t budget) {
      require(budget >= state->progress.completed, "completed-update budgets must be monotonic");
      require(budget <= state->settings.steps, "requested completed updates exceed the frozen session budget");
      if (budget == state->progress.completed) return progress_report(state);
      state->model->train();
      torch::cuda::synchronize(state->settings.model.device.index());
      const auto began = std::chrono::steady_clock::now();
      const auto finish_timing = [&] {
        torch::cuda::synchronize(state->settings.model.device.index());
        state->progress.training_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - began).count();
      };
      try {
        while (state->progress.completed < budget &&
               state->progress.attempted < state->settings.attempt_limit) {
          const auto attempt = state->progress.attempted;
          require(state->progress.sampled_rows <= std::numeric_limits<int64_t>::max() -
              state->settings.batch_size, "sampled row counter overflow");
          const auto indices = training_detail::sampled_indices(state->training.input.data.size(0),
              state->settings.batch_size, state->settings.seed, attempt);
          const auto batch = state->scaler.transform(
              training_detail::selected(state->training.input, indices), state->settings.model);
          state->progress.last_input_cuda = batch.data.is_cuda();
          require(state->progress.last_input_cuda, "normalized training input is not on CUDA");
          const auto mask = make_training_mask(batch.observed, state->settings.model,
              training_detail::counter_seed(state->settings.seed, attempt, 0x6d61736bULL));
          torch::manual_seed(training_detail::counter_seed(state->settings.seed, attempt, 0x746f726368ULL));
          ++state->progress.attempted;
          state->progress.sampled_rows += state->settings.batch_size;
          if (!mask.eligible_channels.any().item<bool>()) continue;
          state->optimizer->zero_grad();
          ForwardOutput output;
          if (state->context_options.enabled) {
            const auto context = context_deletion::make_plan(mask, batch.channel_ids,
                state->settings.model, state->settings.seed, attempt, state->context_options.recipe);
            const std::array<int64_t, 3> counts{context.requested_count, context.actual_count, context.restored_count};
            for (size_t i = 0; i < counts.size(); ++i) {
              require(state->context_counts[i] <= std::numeric_limits<int64_t>::max() - counts[i],
                  "cumulative context-deletion counter overflow");
              state->context_counts[i] += counts[i];
            }
            output = context_deletion::training_forward(*state->model, batch, mask, context);
          } else {
            output = state->model->forward(batch, mask.hidden);
          }
          state->progress.last_loss_cuda = output.loss.is_cuda();
          require(state->progress.last_loss_cuda && output.eligible_example_count > 0 &&
              torch::isfinite(output.loss).all().item<bool>(), "invalid/non-CUDA training loss");
          output.loss.backward();
          for (const auto &parameter : state->model->parameters())
            require(!parameter.grad().defined() || parameter.grad().is_cuda(),
                "a model gradient is not on CUDA");
          const auto gradient_norm = torch::nn::utils::clip_grad_norm_(state->model->parameters(),
              state->settings.gradient_clip_norm > 0 ? state->settings.gradient_clip_norm :
                  std::numeric_limits<double>::infinity(), 2.0, true);
          state->progress.finite_gradients = std::isfinite(gradient_norm);
          require(state->progress.finite_gradients, "nonfinite training gradients");
          state->optimizer->step();
          ++state->progress.completed;
          state->last_loss = {state->progress.attempted, state->progress.completed,
              output.target_cell_count, output.loss.item<double>(), gradient_norm};
          if (state->progress.completed == 1 ||
              state->progress.completed % state->settings.log_every == 0)
            state->progress.losses.push_back(state->last_loss);
        }
      } catch (...) {
        finish_timing();
        throw;
      }
      finish_timing();
      require(state->progress.completed == budget, "continuous training exhausted its cumulative attempt budget");
      return progress_report(state);
    };
    trainer.save_checkpoint = [state](const std::string &path) { save_point(state, path); };
    trainer.snapshot = [state](const std::string &path) { return snapshot_point(state, path); };
    return trainer;
  };
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
