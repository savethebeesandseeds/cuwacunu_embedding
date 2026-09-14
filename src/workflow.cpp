// SPDX-License-Identifier: MIT
#include "embedding/workflow.h"

#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace embedding {
namespace {

// One field list drives both strict parsing and checkpoint serialization.
#define MODEL_INTS(X)                                                          \
  X(channel_count) X(history_length) X(input_width) X(d_model) X(latent_dim)    \
      X(projector_dim) X(predictor_hidden_dim) X(num_encoder_layers)            \
          X(num_predictor_layers) X(num_decoder_layers) X(num_heads)            \
              X(frequency_num_bins)
#define MODEL_DOUBLES(X)                                                       \
  X(dropout) X(mask_ratio_time) X(mask_ratio_frequency) X(mask_ratio_channel)   \
      X(min_context_ratio) X(lambda_jepa) X(lambda_mae) X(lambda_tf_align)      \
          X(lambda_vicreg) X(vicreg_sim_weight) X(vicreg_var_weight)            \
              X(vicreg_cov_weight) X(vicreg_variance_floor)                    \
                  X(vicreg_variance_epsilon) X(target_ema_tau)                \
                      X(lambda_global_vicreg) X(lambda_channel_vicreg)        \
                          X(max_context_target_time_overlap)                  \
                              X(vicreg_view_gaussian_jitter_std)              \
                                  X(vicreg_view_time_dropout_scale)
#define MODEL_BOOLS(X)                                                         \
  X(use_frequency_tokens) X(frequency_log_magnitude) X(use_target_ema)          \
      X(stop_gradient_target) X(use_mae_decoder) X(use_jepa_loss)              \
          X(use_tf_align_loss) X(use_vicreg_loss) X(use_global_vicreg)          \
              X(use_channel_vicreg) X(use_raw_reconstruction_targets)         \
                  X(strict_finite_loss) X(couple_time_frequency_masks)        \
                      X(mask_same_window_across_domains)                      \
                          X(mask_same_channel_block)
#define RUN_INTS(X) X(steps) X(batch_size) X(seed) X(threads) X(log_every)
#define RUN_DOUBLES(X) X(learning_rate) X(weight_decay) X(gradient_clip_norm)

void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}

std::string trim(const std::string &text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos)
    return {};
  return text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

int64_t integer(const std::string &text) {
  int64_t result{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  require(parsed.ec == std::errc() && parsed.ptr == text.data() + text.size(),
          "expected integer, got '" + text + "'");
  return result;
}

double real(const std::string &text) {
  size_t used = 0;
  const double result = std::stod(text, &used);
  require(used == text.size() && std::isfinite(result),
          "expected finite number, got '" + text + "'");
  return result;
}

bool boolean(const std::string &text) {
  require(text == "true" || text == "false", "expected true or false, got '" + text + "'");
  return text == "true";
}

std::vector<int64_t> integer_list(const std::string &text) {
  require(!text.empty() && text.back() != ',', "expected comma-separated integers");
  std::istringstream stream(text);
  std::string item;
  std::vector<int64_t> result;
  while (std::getline(stream, item, ','))
    result.push_back(integer(trim(item)));
  return result;
}

torch::Device device_from(const std::string &text) {
  require(text == "cpu" || text == "cuda", "device must be cpu or cuda");
  return torch::Device(text);
}

void validate_settings(const Settings &settings) {
  validate_config(settings.model);
  require(settings.model.dtype == torch::kFloat32, "workflow supports float32 models");
  require(settings.steps > 0 && settings.batch_size > 0 && settings.threads > 0 &&
              settings.threads <= std::numeric_limits<int>::max() && settings.log_every > 0,
          "steps, batch_size, threads and log_every must be positive");
  require(settings.seed >= 0, "seed must be nonnegative");
  require(std::isfinite(settings.learning_rate) && settings.learning_rate > 0 &&
              std::isfinite(settings.weight_decay) && settings.weight_decay >= 0 &&
              std::isfinite(settings.gradient_clip_norm) && settings.gradient_clip_norm >= 0,
          "learning_rate must be positive; weight_decay and gradient_clip_norm nonnegative");
}

void activate(const Settings &settings) {
  validate_settings(settings);
  require(!settings.model.device.is_cuda() || torch::cuda::is_available(),
          "CUDA is unavailable; use --device cpu or enable the container's GPU");
  torch::set_num_threads(static_cast<int>(settings.threads));
  torch::manual_seed(static_cast<uint64_t>(settings.seed));
}

void validate_batch(const Batch &batch, const Config &config) {
  require(batch.data.defined() && batch.data.scalar_type() == torch::kFloat32 &&
              batch.data.dim() == 4 && batch.data.size(0) > 0,
          "data must be nonempty float32 [B,C,H,F]");
  require(batch.data.size(1) == config.channel_count &&
              batch.data.size(2) == config.history_length && batch.data.size(3) == config.input_width,
          "data C/H/F dimensions do not match model configuration");
  require(batch.feature_mask.defined() && batch.feature_mask.scalar_type() == torch::kBool &&
              batch.feature_mask.sizes() == batch.data.sizes(),
          "feature_mask must be bool with the same [B,C,H,F] shape as data");
  require(torch::isfinite(batch.data).logical_or(batch.feature_mask.logical_not()).all().item<bool>(),
          "observed data contains NaN or infinity");
}

// Save in the destination directory so the final rename is atomic on Debian.
// A failed save leaves the previous checkpoint intact.
void save_archive(const std::string &path, torch::serialize::OutputArchive &archive) {
  require(!path.empty(), "output path is empty");
  const std::filesystem::path destination(path);
  if (!destination.parent_path().empty())
    std::filesystem::create_directories(destination.parent_path());
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto temporary = path + ".tmp-" + std::to_string(nonce);
  require(!std::filesystem::exists(temporary), "temporary output already exists: " + temporary);
  archive.save_to(temporary);
  std::filesystem::rename(temporary, destination);
}

void distinct_paths(const std::string &input, const std::string &output) {
  if (!input.empty())
    require(std::filesystem::weakly_canonical(input) != std::filesystem::weakly_canonical(output),
            "input and output paths must differ");
}

torch::Tensor text_tensor(const std::string &text) {
  return torch::tensor(std::vector<uint8_t>(text.begin(), text.end()), torch::kUInt8);
}

std::string tensor_text(const torch::Tensor &tensor) {
  auto bytes = tensor.to(torch::kCPU).contiguous();
  require(bytes.scalar_type() == torch::kUInt8 && bytes.dim() == 1, "invalid checkpoint config");
  return std::string(reinterpret_cast<const char *>(bytes.data_ptr<uint8_t>()), bytes.numel());
}

void check_checkpoint_version(torch::serialize::InputArchive &archive) {
  torch::Tensor version;
  archive.read("format_version", version, true);
  require(version.numel() == 1 && version.item<int64_t>() == 1,
          "unsupported checkpoint format_version");
}

using Arguments = std::map<std::string, std::string>;

Arguments arguments(int argc, char **argv, const std::set<std::string> &allowed) {
  Arguments result;
  for (int i = 2; i < argc; i += 2) {
    const std::string name(argv[i]);
    require(allowed.count(name), "unknown option: " + name);
    require(i + 1 < argc, "missing value for " + name);
    require(result.emplace(name, argv[i + 1]).second, "duplicate option: " + name);
  }
  return result;
}

std::string required(const Arguments &args, const std::string &name) {
  const auto it = args.find(name);
  require(it != args.end() && !it->second.empty(), "required option: " + name);
  return it->second;
}

std::string optional(const Arguments &args, const std::string &name, const std::string &fallback = {}) {
  const auto it = args.find(name);
  return it == args.end() ? fallback : it->second;
}

void usage() {
  std::cout << R"(Usage:
  embedding synthetic --output data.pt [--config settings.conf] [--samples 32] [--seed 101]
  embedding train --checkpoint model.pt [--config settings.conf] [--input data.pt]
                  [--steps 8] [--batch-size 4] [--device cpu|cuda] [--seed 101]
  embedding train --resume model.pt --checkpoint continued.pt [--input data.pt]
                  [--steps 8] [--batch-size 4] [--device cpu|cuda]
  embedding embed --checkpoint model.pt --input data.pt --output embeddings.pt
                  [--batch-size 4] [--device cpu|cuda]

Train without --input uses synthetic batches. --steps is the number of additional
updates in this invocation. Resume restores the saved architecture and AdamW state;
use the same --input archive to continue the same dataset. Embed defaults to CPU.
Input archives have data=float32 [B,C,H,F], feature_mask=bool [B,C,H,F].
Config is strict key=value text; omitted keys use the small baseline defaults.
)";
}

} // namespace

Settings default_settings() {
  Settings settings;
  auto &c = settings.model;
  c.channel_count = 3;
  c.history_length = 32;
  c.input_width = 3;
  c.d_model = 16;
  c.latent_dim = 12;
  c.projector_dim = 20;
  c.predictor_hidden_dim = 24;
  c.num_encoder_layers = 1;
  c.num_predictor_layers = 1;
  c.num_decoder_layers = 1;
  c.num_heads = 3;
  c.dropout = 0.05;
  c.time_scales = {4, 8};
  c.scale_strides = {2, 4};
  c.frequency_num_bins = 6;
  c.mask_ratio_time = 0.35;
  c.mask_ratio_frequency = 0.20;
  c.mask_ratio_channel = 0.02;
  c.min_context_ratio = 0.35;
  c.lambda_jepa = 1.0;
  c.lambda_mae = 0.20;
  c.lambda_tf_align = 0.05;
  c.lambda_vicreg = 0.02;
  c.target_ema_tau = 0.98;
  c.use_channel_vicreg = true;
  c.lambda_global_vicreg = 0.5;
  c.lambda_channel_vicreg = 0.5;
  return settings;
}

Settings parse_settings(const std::string &text) {
  auto settings = default_settings();
  std::istringstream stream(text);
  std::set<std::string> seen;
  std::string line;
  int line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    line = trim(line.substr(0, line.find('#')));
    if (line.empty())
      continue;
    try {
      const auto separator = line.find('=');
      require(separator != std::string::npos, "expected key=value");
      const auto key = trim(line.substr(0, separator));
      const auto value = trim(line.substr(separator + 1));
      require(seen.insert(key).second, "duplicate key: " + key);
#define PARSE_INT(name) if (key == #name) { settings.model.name = integer(value); continue; }
      MODEL_INTS(PARSE_INT)
#undef PARSE_INT
#define PARSE_DOUBLE(name) if (key == #name) { settings.model.name = real(value); continue; }
      MODEL_DOUBLES(PARSE_DOUBLE)
#undef PARSE_DOUBLE
#define PARSE_BOOL(name) if (key == #name) { settings.model.name = boolean(value); continue; }
      MODEL_BOOLS(PARSE_BOOL)
#undef PARSE_BOOL
#define PARSE_INT(name) if (key == #name) { settings.name = integer(value); continue; }
      RUN_INTS(PARSE_INT)
#undef PARSE_INT
#define PARSE_DOUBLE(name) if (key == #name) { settings.name = real(value); continue; }
      RUN_DOUBLES(PARSE_DOUBLE)
#undef PARSE_DOUBLE
      if (key == "time_scales") settings.model.time_scales = integer_list(value);
      else if (key == "scale_strides") settings.model.scale_strides = value.empty() ? std::vector<int64_t>{} : integer_list(value);
      else if (key == "device") settings.model.device = device_from(value);
      else throw std::runtime_error("unknown key: " + key);
    } catch (const std::exception &error) {
      throw std::runtime_error("config line " + std::to_string(line_number) + ": " + error.what());
    }
  }
  validate_settings(settings);
  return settings;
}

Settings read_settings(const std::string &path) {
  std::ifstream file(path);
  require(file.good(), "cannot open config: " + path);
  std::ostringstream text;
  text << file.rdbuf();
  require(!file.bad(), "failed reading config: " + path);
  return parse_settings(text.str());
}

std::string settings_text(const Settings &settings) {
  validate_settings(settings);
  std::ostringstream out;
  out << std::setprecision(17) << std::boolalpha;
#define WRITE_MODEL(name) out << #name "=" << settings.model.name << '\n';
  MODEL_INTS(WRITE_MODEL)
  MODEL_DOUBLES(WRITE_MODEL)
  MODEL_BOOLS(WRITE_MODEL)
#undef WRITE_MODEL
#define WRITE_RUN(name) out << #name "=" << settings.name << '\n';
  RUN_INTS(WRITE_RUN)
  RUN_DOUBLES(WRITE_RUN)
#undef WRITE_RUN
  const auto write_list = [&](const char *key, const std::vector<int64_t> &values) {
    out << key << '=';
    for (size_t i = 0; i < values.size(); ++i)
      out << (i == 0 ? "" : ",") << values[i];
    out << '\n';
  };
  write_list("time_scales", settings.model.time_scales);
  write_list("scale_strides", settings.model.scale_strides);
  out << "device=" << (settings.model.device.is_cuda() ? "cuda" : "cpu") << '\n';
  return out.str();
}

Batch synthetic_batch(const Config &config, int64_t samples, int64_t step) {
  using torch::indexing::Slice;
  require(config.input_width == 3, "synthetic data requires input_width=3; supply --input for other widths");
  require(samples > 0 && step >= 0, "samples must be positive and step nonnegative");
  constexpr double pi = 3.14159265358979323846;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto t = torch::arange(config.history_length, options);
  auto normalized = t / static_cast<double>(std::max<int64_t>(1, config.history_length - 1));
  auto data = torch::zeros({samples, config.channel_count, config.history_length, config.input_width}, options);
  for (int64_t b = 0; b < samples; ++b) {
    const int64_t family = (step + b) % 4;
    for (int64_t c = 0; c < config.channel_count; ++c) {
      const double phase = 0.13 * static_cast<double>(step + b + c);
      const double gain = 1.0 + 0.10 * static_cast<double>(c);
      auto wave = gain * torch::sin(2.0 * pi * normalized * 4.0 + phase);
      auto trend = normalized * (0.25 + 0.05 * static_cast<double>(c)) + static_cast<double>(b) * 0.01;
      if (family == 1)
        wave = torch::where(t.lt(config.history_length / 2),
                            torch::sin(2.0 * pi * normalized * 3.0 + phase),
                            torch::sin(2.0 * pi * normalized * 7.0 + phase));
      else if (family == 2)
        wave = gain * torch::sin(2.0 * pi * normalized * 5.0 + phase) +
               0.20 * torch::cos(2.0 * pi * normalized * (2.0 + c) + phase);
      else if (family == 3)
        trend = trend + 0.15 * torch::sin(2.0 * pi * normalized);
      auto noise = torch::randn({config.history_length}, options) * (family == 3 ? 0.06 : 0.03);
      data.index_put_({b, c, Slice(), Slice()}, torch::stack({wave, trend, noise}, 1));
    }
  }
  auto mask = torch::ones(data.sizes(), options.dtype(torch::kBool));
  for (int64_t b = 0; b < samples; ++b) {
    const double missing = (step + b) % 4 == 3 ? 0.30 : (step % 5 == 0 ? 0.08 : 0.0);
    if (missing > 0)
      mask[b].copy_(torch::rand(data[b].sizes(), options).gt(missing));
  }
  return {torch::where(mask, data, torch::zeros_like(data)), mask};
}

void save_batch(const std::string &path, const Batch &batch) {
  torch::serialize::OutputArchive archive;
  archive.write("data", batch.data.detach().to(torch::kCPU), true);
  archive.write("feature_mask", batch.feature_mask.detach().to(torch::kCPU), true);
  save_archive(path, archive);
}

Batch load_batch(const std::string &path, const Config &config) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, torch::kCPU);
  Batch batch;
  archive.read("data", batch.data, true);
  archive.read("feature_mask", batch.feature_mask, true);
  validate_batch(batch, config);
  batch.data = torch::where(batch.feature_mask, batch.data, torch::zeros_like(batch.data));
  return batch;
}

void save_checkpoint(const std::string &path, const Settings &settings,
                     Model &model, torch::optim::AdamW &optimizer, int64_t completed_steps) {
  require(completed_steps >= 0, "completed_steps must be nonnegative");
  Settings saved = settings;
  saved.model = model->config();
  torch::serialize::OutputArchive archive, weights, optimizer_state;
  archive.write("format_version", torch::tensor(int64_t{1}), true);
  archive.write("settings", text_tensor(settings_text(saved)), true);
  archive.write("completed_steps", torch::tensor(completed_steps), true);
  model->save(weights);
  optimizer.save(optimizer_state);
  archive.write("model", weights);
  archive.write("optimizer", optimizer_state);
  save_archive(path, archive);
}

Checkpoint load_checkpoint(const std::string &path, const torch::Device &device) {
  torch::serialize::InputArchive archive, weights;
  archive.load_from(path, device);
  check_checkpoint_version(archive);
  torch::Tensor text, step;
  archive.read("settings", text, true);
  archive.read("completed_steps", step, true);
  Checkpoint checkpoint;
  checkpoint.settings = parse_settings(tensor_text(text));
  checkpoint.settings.model.device = device;
  require(step.numel() == 1 && step.item<int64_t>() >= 0, "invalid checkpoint step");
  checkpoint.completed_steps = step.item<int64_t>();
  checkpoint.model = Model(checkpoint.settings.model);
  archive.read("model", weights);
  checkpoint.model->load(weights);
  return checkpoint;
}

void load_optimizer(const std::string &path, torch::optim::AdamW &optimizer,
                    const torch::Device &device) {
  torch::serialize::InputArchive archive, state;
  archive.load_from(path, device);
  check_checkpoint_version(archive);
  archive.read("optimizer", state);
  optimizer.load(state);
}

int run_cli(int argc, char **argv) {
  if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
    usage();
    return argc < 2 ? 1 : 0;
  }
  const std::string command(argv[1]);
  if (argc == 3 && std::string(argv[2]) == "--help") {
    usage();
    return 0;
  }
  if (command == "synthetic") {
    const auto args = arguments(argc, argv, {"--config", "--output", "--samples", "--seed"});
    auto settings = args.count("--config") ? read_settings(args.at("--config")) : default_settings();
    if (args.count("--seed")) settings.seed = integer(args.at("--seed"));
    settings.model.device = torch::kCPU;
    activate(settings);
    const auto samples = integer(optional(args, "--samples", "32"));
    const auto output = required(args, "--output");
    save_batch(output, synthetic_batch(settings.model, samples));
    std::cout << "saved " << samples << " samples to " << output << '\n';
    return 0;
  }
  if (command == "train") {
    const auto args = arguments(argc, argv, {"--config", "--checkpoint", "--input", "--resume",
                                            "--steps", "--batch-size", "--device", "--seed"});
    const auto output = required(args, "--checkpoint");
    const auto input = optional(args, "--input");
    const auto resume = optional(args, "--resume");
    distinct_paths(input, output);
    require(resume.empty() || (!args.count("--config") && !args.count("--seed")),
            "resume restores saved config/seed; omit --config and --seed");
    Checkpoint checkpoint;
    if (!resume.empty()) {
      // CPU is also the portable default for resumed checkpoints.
      checkpoint = load_checkpoint(resume, device_from(optional(args, "--device", "cpu")));
    } else {
      checkpoint.settings = args.count("--config") ? read_settings(args.at("--config")) : default_settings();
    }
    auto &settings = checkpoint.settings;
    if (args.count("--device")) settings.model.device = device_from(args.at("--device"));
    if (args.count("--steps")) settings.steps = integer(args.at("--steps"));
    if (args.count("--batch-size")) settings.batch_size = integer(args.at("--batch-size"));
    if (args.count("--seed")) settings.seed = integer(args.at("--seed"));
    activate(settings);
    Batch dataset;
    if (!input.empty()) {
      dataset = load_batch(input, settings.model);
      require(dataset.feature_mask.flatten(1).any(1).all().item<bool>(),
              "training samples must each contain at least one observed feature");
    } else {
      require(settings.model.input_width == 3, "synthetic training requires input_width=3; supply --input");
    }
    if (!checkpoint.model) checkpoint.model = Model(settings.model);
    auto &model = checkpoint.model;
    std::vector<torch::Tensor> parameters;
    for (const auto &parameter : model->parameters())
      if (parameter.requires_grad()) parameters.push_back(parameter);
    torch::optim::AdamW optimizer(parameters, torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
    if (!resume.empty()) load_optimizer(resume, optimizer, settings.model.device);
    model->train();
    require(checkpoint.completed_steps <= std::numeric_limits<int64_t>::max() - settings.steps,
            "step counter would overflow");
    const int64_t end_step = checkpoint.completed_steps + settings.steps;
    for (int64_t step = checkpoint.completed_steps; step < end_step; ++step) {
      // Derive randomness from the absolute step so restart needs no opaque RNG snapshot.
      torch::manual_seed(static_cast<uint64_t>(settings.seed) + static_cast<uint64_t>(step));
      Batch batch;
      if (input.empty()) {
        batch = synthetic_batch(settings.model, settings.batch_size, step);
      } else {
        auto indices = torch::randint(dataset.data.size(0), {settings.batch_size}, torch::kInt64);
        batch = {dataset.data.index_select(0, indices), dataset.feature_mask.index_select(0, indices)};
      }
      optimizer.zero_grad();
      auto result = model->forward(batch.data, batch.feature_mask);
      require(torch::isfinite(result.loss).all().item<bool>(), "non-finite training loss");
      result.loss.backward();
      torch::nn::utils::clip_grad_norm_(
          parameters, settings.gradient_clip_norm > 0 ? settings.gradient_clip_norm
                                                     : std::numeric_limits<double>::infinity(),
          2.0, /*error_if_nonfinite=*/true);
      optimizer.step();
      model->update_target_network();
      if ((step + 1) % settings.log_every == 0 || step + 1 == end_step)
        std::cout << "step=" << step + 1 << " loss=" << result.loss.item<double>()
                  << " jepa=" << result.loss_jepa.item<double>()
                  << " mae=" << result.loss_mae.item<double>()
                  << " tf_align=" << result.loss_tf_align.item<double>()
                  << " vicreg=" << result.loss_vicreg.item<double>() << '\n';
    }
    save_checkpoint(output, settings, model, optimizer, end_step);
    std::cout << "saved checkpoint at step " << end_step << " to " << output << '\n';
    return 0;
  }
  if (command == "embed") {
    const auto args = arguments(argc, argv, {"--checkpoint", "--input", "--output", "--device", "--batch-size"});
    const auto input = required(args, "--input");
    const auto output = required(args, "--output");
    const auto checkpoint_path = required(args, "--checkpoint");
    distinct_paths(input, output);
    distinct_paths(checkpoint_path, output);
    auto checkpoint = load_checkpoint(checkpoint_path, device_from(optional(args, "--device", "cpu")));
    if (args.count("--batch-size")) checkpoint.settings.batch_size = integer(args.at("--batch-size"));
    activate(checkpoint.settings);
    auto batch = load_batch(input, checkpoint.settings.model);
    torch::NoGradGuard no_grad;
    checkpoint.model->eval();
    std::vector<torch::Tensor> global, channels, sample_valid, channel_valid;
    const auto count = batch.data.size(0);
    for (int64_t start = 0; start < count; start += checkpoint.settings.batch_size) {
      const auto length = std::min(checkpoint.settings.batch_size, count - start);
      auto encoded = checkpoint.model->encode(batch.data.narrow(0, start, length),
                                              batch.feature_mask.narrow(0, start, length));
      require(torch::isfinite(encoded.pooled_embedding).all().item<bool>() &&
                  torch::isfinite(encoded.pooled_by_channel).all().item<bool>(), "non-finite embedding");
      global.push_back(encoded.pooled_embedding.to(torch::kCPU));
      channels.push_back(encoded.pooled_by_channel.to(torch::kCPU));
      sample_valid.push_back(encoded.sample_valid_mask.to(torch::kCPU));
      channel_valid.push_back(encoded.channel_valid_mask.to(torch::kCPU));
    }
    torch::serialize::OutputArchive archive;
    archive.write("pooled_embedding", torch::cat(global), true);
    archive.write("pooled_by_channel", torch::cat(channels), true);
    archive.write("sample_valid_mask", torch::cat(sample_valid), true);
    archive.write("channel_valid_mask", torch::cat(channel_valid), true);
    save_archive(output, archive);
    std::cout << "saved " << count << " global and per-channel embeddings to " << output << '\n';
    return 0;
  }
  throw std::runtime_error("unknown command: " + command + "; use --help");
}

#undef MODEL_INTS
#undef MODEL_DOUBLES
#undef MODEL_BOOLS
#undef RUN_INTS
#undef RUN_DOUBLES

} // namespace embedding
