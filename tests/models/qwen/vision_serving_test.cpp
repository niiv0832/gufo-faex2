// Image prompts through the serving stack on real weights: autoregression
// against an explicitly selected speculative backend, sampling, concurrency,
// cancellation, live-prefix and disk restoration.
//
// qwen_vision_serving_test MODEL SPECULATION IMAGES [--mmproj PATH]
//                          [--disk-only|--append-only]
// SPECULATION is `-` (autoregression only), `mtp-embedded` (the prediction
// head inside MODEL), `mtp=PATH` or `dflash=PATH`. The selected backend must
// load and must have proposals accepted; nothing falls back to another one.
// IMAGES is a directory with red.png, blue.png and blue-wide.png, or
// `synthetic` to generate those solid-colour images.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/image.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {
using namespace gufo;
using Backend = server::InferenceBackend;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

// A solid-colour 8-bit RGB PNG with stored (uncompressed) deflate blocks.
void WriteSolidPng(const std::filesystem::path& path, std::uint32_t width,
                   std::uint32_t height, std::array<std::uint8_t, 3> rgb) {
  const auto put32 = [](std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      out.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  std::vector<std::uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(height) * (1 + (3 * width)));
  for (std::uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);  // filter: none
    for (std::uint32_t x = 0; x < width; ++x)
      raw.insert(raw.end(), rgb.begin(), rgb.end());
  }
  std::vector<std::uint8_t> zlib = {0x78, 0x01};
  std::uint32_t a = 1, b = 0;  // Adler-32
  for (const auto byte : raw) {
    a = (a + byte) % 65521U;
    b = (b + a) % 65521U;
  }
  for (std::size_t offset = 0; offset < raw.size();) {
    const std::size_t size = std::min<std::size_t>(raw.size() - offset, 65535);
    zlib.push_back(offset + size == raw.size() ? 1 : 0);
    zlib.push_back(static_cast<std::uint8_t>(size));
    zlib.push_back(static_cast<std::uint8_t>(size >> 8U));
    zlib.push_back(static_cast<std::uint8_t>(~size));
    zlib.push_back(static_cast<std::uint8_t>(~size >> 8U));
    zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                raw.begin() + static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
  }
  put32(zlib, (b << 16U) | a);
  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  const auto chunk = [&](std::string_view type,
                         std::span<const std::uint8_t> data) {
    put32(png, static_cast<std::uint32_t>(data.size()));
    const std::size_t begin = png.size();
    png.insert(png.end(), type.begin(), type.end());
    png.insert(png.end(), data.begin(), data.end());
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t i = begin; i < png.size(); ++i) {
      crc ^= png[i];
      for (int bit = 0; bit < 8; ++bit)
        crc = (crc >> 1U) ^ (0xEDB88320U & (0U - (crc & 1U)));
    }
    put32(png, ~crc);
  };
  std::vector<std::uint8_t> header;
  put32(header, width);
  put32(header, height);
  header.insert(header.end(), {8, 2, 0, 0, 0});  // 8-bit RGB
  chunk("IHDR", header);
  chunk("IDAT", zlib);
  chunk("IEND", {});
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
  Require(output.good(), "cannot write a synthetic image");
}

std::filesystem::path TemporaryDirectory(std::string_view name) {
  std::string pattern =
      (std::filesystem::temp_directory_path() / (std::string(name) + "-XXXXXX"))
          .string();
  Require(::mkdtemp(pattern.data()) != nullptr,
          "cannot create a temporary directory");
  return pattern;
}

struct RemoveDirectory {
  std::filesystem::path path;
  ~RemoveDirectory() {
    if (path.empty())
      return;
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

// Speculation that proposes nothing, or whose proposals are never accepted,
// is autoregression under another name.
void RequireAcceptedDrafts(std::span<const Backend::Result> results,
                           std::string_view label) {
  std::size_t proposed = 0;
  std::size_t accepted = 0;
  for (const auto& result : results) {
    Require(result.draft_tokens > 0,
            "image request bypassed speculative decoding");
    Require(result.draft_accepted_tokens <= result.draft_tokens,
            "image request accepted more tokens than it drafted");
    proposed += result.draft_tokens;
    accepted += result.draft_accepted_tokens;
  }
  std::cout << label << ": requests=" << results.size()
            << " draft_tokens=" << proposed
            << " draft_accepted_tokens=" << accepted << '\n';
  Require(accepted > 0, "no speculative proposal was accepted");
}

std::string Lower(std::string value) {
  for (auto& character : value) {
    character =
        static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

server::ChatRequest ImageRequest(const std::filesystem::path& image) {
  server::ChatRequest request;
  // This 16-token recognition check evaluates the answer, not a thinking trace.
  request.reasoning.enabled = false;
  request.messages.emplace_back(tokenization::ChatRole::kUser,
                                "Describe the dominant color of this image, "
                                "then explain it in a full sentence.");
  request.messages.back().images.push_back(
      {0, std::make_shared<const std::vector<std::uint8_t>>(
              core::ReadImageFile(image))});
  return request;
}

void CheckConcurrent(Backend& backend,
                     std::span<const server::ChatRequest> requests,
                     std::span<const Backend::Result> expected,
                     const sampling::SamplingConfig& sampling,
                     bool timed_mtp = false) {
  std::vector<std::shared_ptr<Backend::GenerationRequest>> pending;
  for (const auto& request : requests)
    pending.push_back(backend.start_chat(request, 16, sampling));
  std::size_t width = 1;
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const auto result = pending[i]->Wait();
    Require(result.tokens == expected[i].tokens,
            "concurrent image request changed token IDs");
    Require(result.draft_accepted_tokens <= result.draft_tokens,
            "image request accepted more tokens than it drafted");
    // Flash-Next's greedy controller measures batch cost and may select a
    // different depth at C4. Seeded sampling must replay both draws and depth.
    if (!timed_mtp || sampling.uses_random_sampling())
      Require(
          result.draft_tokens == expected[i].draft_tokens &&
              result.draft_accepted_tokens == expected[i].draft_accepted_tokens,
          "concurrent image request changed proposal/acceptance accounting");
    width = std::max(width, result.physical_execution_width);
  }
  Require(width > 1, "image test did not exercise a shared decode batch");
  std::cout << "image concurrency=" << requests.size()
            << " physical_width=" << width << '\n';
}

void CheckFlashIncrementalOracle(
    const std::shared_ptr<models::qwen38_flash_next::Model>& model,
    const server::ChatRequest& request, const Backend::Result& cached) {
  const auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          model->tokenizer(), request.messages, {},
          tokenization::ResolveQwenChatOptions(request.reasoning,
                                               request.add_vision_id),
          model->VisionEncoder()->identity(), model->MaxContext()));
  const std::vector<std::int32_t> tokens(prompt->tokens.begin(),
                                         prompt->tokens.end());
  std::string error;
  auto incremental = model->CreateSession(
      gufo::core::SessionMode::kAutoregressive, model->MaxContext(), &error);
  auto bulk = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                   model->MaxContext(), &error);
  Require(incremental && bulk, error);
  incremental->ConfigureVision(prompt);
  bulk->ConfigureVision(prompt);
  Require(incremental->Sync(
              std::span(tokens).first(cached.cached_prompt_tokens), &error),
          error);
  Require(incremental->Sync(tokens, &error) && bulk->Sync(tokens, &error),
          error);
  std::vector<tokenization::TokenId> generated;
  double max_kl = 0;
  float max_logit_error = 0;
  for (std::size_t i = 0; i < 16; ++i) {
    const auto left = incremental->Logits();
    const auto right = bulk->Logits();
    const float left_max = *std::max_element(left.begin(), left.end());
    const float right_max = *std::max_element(right.begin(), right.end());
    double left_sum = 0, right_sum = 0;
    for (std::size_t j = 0; j < left.size(); ++j) {
      Require(std::isfinite(left[j]) && std::isfinite(right[j]),
              "nonfinite Flash image logits");
      left_sum += std::exp(static_cast<double>(left[j] - left_max));
      right_sum += std::exp(static_cast<double>(right[j] - right_max));
      max_logit_error = std::max(max_logit_error, std::abs(left[j] - right[j]));
    }
    double kl = 0;
    for (std::size_t j = 0; j < left.size(); ++j) {
      const double log_p = left[j] - left_max - std::log(left_sum);
      const double log_q = right[j] - right_max - std::log(right_sum);
      kl += std::exp(log_p) * (log_p - log_q);
    }
    max_kl = std::max(max_kl, kl);
    const auto token = static_cast<std::int32_t>(
        std::max_element(left.begin(), left.end()) - left.begin());
    if (model->IsStopToken(token))
      break;
    generated.push_back(static_cast<tokenization::TokenId>(token));
    Require(
        incremental->Evaluate(token, &error) && bulk->Evaluate(token, &error),
        error);
  }
  std::cout << "Flash fresh incremental oracle exact="
            << (generated == cached.tokens)
            << " bulk_incremental_max_logit_error=" << max_logit_error
            << " max_KL_temperature1=" << max_kl << '\n';
  Require(generated == cached.tokens,
          "Flash snapshot changed incremental inference");
  Require(max_logit_error == 0,
          "Flash image continuation changed logits with prefill chunking");
}

void CheckQwenIncrementalOracle(
    const std::shared_ptr<const hip::QwenGpuModel>& model,
    const server::ChatRequest& request, const Backend::Result& previous,
    const Backend::Result& cached) {
  const auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          model->GetTokenizer(), request.messages, {},
          tokenization::ResolveQwenChatOptions(request.reasoning,
                                               request.add_vision_id),
          model->VisionEncoder()->identity(), 1024));
  hip::QwenGpuExecutor incremental(model, 1024);
  incremental.ConfigureVision(prompt, model->VisionEncoder());
  const auto tokens = std::span(prompt->tokens);
  // Replay the same execution history: the retained reply used decode
  // arithmetic, not matrix prefill. The separate cold comparison below
  // deliberately continues to check that stronger guarantee.
  (void)incremental.ForwardPromptBatch(tokens.first(previous.prompt_tokens));
  for (std::size_t i = previous.prompt_tokens; i < cached.cached_prompt_tokens;
       ++i)
    (void)incremental.ForwardToken(tokens[i], i);
  auto next = incremental.ForwardPromptBatch(
      tokens.subspan(cached.cached_prompt_tokens), cached.cached_prompt_tokens);
  std::vector<tokenization::TokenId> generated;
  for (std::size_t i = 0; i < cached.tokens.size(); ++i) {
    generated.push_back(next);
    next = incremental.ForwardToken(next, tokens.size() + i);
  }
  std::cout << "Qwen fresh live-history oracle exact="
            << (generated == cached.tokens)
            << " prefix=" << cached.cached_prompt_tokens
            << " total=" << tokens.size() << '\n';
  Require(generated == cached.tokens,
          "Qwen snapshot changed incremental inference");

  // Separate chunk invariance from replaying a generation history: BF16 BLAS
  // selected different reductions for the two batch shapes despite identical
  // normed rows, which amplified into different continuation tokens.
  incremental.Reset();
  (void)incremental.ForwardPromptBatch(tokens.first(previous.prompt_tokens));
  (void)incremental.ForwardPromptBatch(tokens.subspan(previous.prompt_tokens),
                                       previous.prompt_tokens);
  const auto split_logits = incremental.CopyLastLogits();
  const std::vector<float> expected(split_logits.begin(), split_logits.end());
  incremental.Reset();
  (void)incremental.ForwardPromptBatch(tokens);
  const auto cold_logits = incremental.CopyLastLogits();
  Require(std::ranges::equal(expected, cold_logits),
          "Qwen image prefill logits changed with chunk boundaries");
}

// Stage 1 (B1) diagnostic, enabled by GUFO_VISION_SERVING_DIAGNOSTIC=1:
// quantifies the greedy near-tie between the decode-written cache history and
// the matrix prefill of the same tokens. Read-only: prints per-step margins,
// logit deltas and KL (plus common-prefix aggregates up to first divergence);
// it adds no requirements and changes no assertions.
void DumpQwenVisionContinuationDiagnostic(
    const std::shared_ptr<const hip::QwenGpuModel>& model,
    const server::ChatRequest& request, const Backend::Result& previous,
    const Backend::Result& cached,
    std::span<const tokenization::TokenId> cold_serving_tokens) {
  const char* enabled = ::getenv("GUFO_VISION_SERVING_DIAGNOSTIC");
  if (enabled == nullptr || std::string_view(enabled) != "1")
    return;
  const auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          model->GetTokenizer(), request.messages, {},
          tokenization::ResolveQwenChatOptions(request.reasoning,
                                               request.add_vision_id),
          model->VisionEncoder()->identity(), 1024));
  const auto tokens = std::span(prompt->tokens);
  struct Step {
    tokenization::TokenId token;
    float margin;
  };
  auto trace = [&](bool decode_history) {
    hip::QwenGpuExecutor executor(model, 1024);
    executor.ConfigureVision(prompt, model->VisionEncoder());
    if (decode_history) {
      (void)executor.ForwardPromptBatch(tokens.first(previous.prompt_tokens));
      for (std::size_t i = previous.prompt_tokens;
           i < cached.cached_prompt_tokens; ++i)
        (void)executor.ForwardToken(tokens[i], i, false);
      (void)executor.ForwardPromptBatch(
          tokens.subspan(cached.cached_prompt_tokens),
          cached.cached_prompt_tokens);
    } else {
      (void)executor.ForwardPromptBatch(tokens);
    }
    std::vector<Step> steps;
    std::vector<std::vector<float>> rows;
    steps.reserve(cached.tokens.size());
    rows.reserve(cached.tokens.size());
    for (std::size_t i = 0; i < cached.tokens.size(); ++i) {
      const auto logits = executor.CopyLastLogits();
      const auto ranked = std::ranges::max_element(logits) - logits.begin();
      float best = logits[ranked], second = -std::numeric_limits<float>::max();
      for (std::size_t j = 0; j < logits.size(); ++j) {
        if (static_cast<std::ptrdiff_t>(j) != ranked && logits[j] > second)
          second = logits[j];
      }
      steps.push_back(
          {static_cast<tokenization::TokenId>(ranked), best - second});
      rows.emplace_back(logits.begin(), logits.end());
      (void)executor.ForwardToken(
          static_cast<tokenization::TokenId>(ranked),
          static_cast<std::uint32_t>(tokens.size() + i));
    }
    return std::pair{steps, rows};
  };
  const auto [cached_steps, cached_rows] = trace(true);
  const auto [cold_steps, cold_rows] = trace(false);
  const auto matches = [](const std::vector<Step>& steps,
                          std::span<const tokenization::TokenId> expected) {
    return steps.size() == expected.size() &&
           std::equal(steps.begin(), steps.end(), expected.begin(),
                      [](const Step& step, tokenization::TokenId id) {
                        return step.token == id;
                      });
  };
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "vision-diag: cached_trajectory_matches_serving="
      << matches(cached_steps, cached.tokens)
      << " cold_trajectory_matches_serving="
      << matches(cold_steps, cold_serving_tokens) << " prompt=" << tokens.size()
      << " cache_prefix=" << cached.cached_prompt_tokens << '\n';
  double all_max_kl = 0, common_max_kl = 0;
  float all_max_delta = 0, common_max_delta = 0;
  std::size_t divergence = cached.tokens.size();
  std::vector<float> step_deltas(cached.tokens.size(), 0.0F);
  std::vector<double> step_kls(cached.tokens.size(), 0.0);
  for (std::size_t i = 0; i < cached.tokens.size(); ++i) {
    const auto& left = cached_rows[i];
    const auto& right = cold_rows[i];
    const float left_max = *std::ranges::max_element(left);
    const float right_max = *std::ranges::max_element(right);
    double left_sum = 0, right_sum = 0;
    float step_delta = 0;
    for (std::size_t j = 0; j < left.size(); ++j) {
      step_delta = std::max(step_delta, std::abs(left[j] - right[j]));
      left_sum += std::exp(static_cast<double>(left[j] - left_max));
      right_sum += std::exp(static_cast<double>(right[j] - right_max));
    }
    double kl = 0;
    for (std::size_t j = 0; j < left.size(); ++j) {
      const double log_p = left[j] - left_max - std::log(left_sum);
      const double log_q = right[j] - right_max - std::log(right_sum);
      kl += std::exp(log_p) * (log_p - log_q);
    }
    step_deltas[i] = step_delta;
    step_kls[i] = kl;
    all_max_delta = std::max(all_max_delta, step_delta);
    all_max_kl = std::max(all_max_kl, kl);
    if (divergence == cached.tokens.size()) {
      common_max_delta = std::max(common_max_delta, step_delta);
      common_max_kl = std::max(common_max_kl, kl);
      if (cached_steps[i].token != cold_steps[i].token)
        divergence = i;
    }
  }
  const std::size_t common_steps =
      divergence < cached.tokens.size() ? divergence + 1 : cached.tokens.size();
  out << "vision-diag: common_prefix_steps=" << common_steps
      << " common_max_logit_delta=" << common_max_delta
      << " common_max_KL_temperature1=" << common_max_kl
      << " all_steps_max_logit_delta=" << all_max_delta
      << " all_steps_max_KL_temperature1=" << all_max_kl
      << " first_divergence_step=" << divergence << '\n';
  for (std::size_t i = 0; i < cached.tokens.size(); ++i) {
    out << "vision-diag step=" << i << " cached_token=" << cached_steps[i].token
        << " cold_token=" << cold_steps[i].token
        << " cached_margin=" << cached_steps[i].margin
        << " cold_margin=" << cold_steps[i].margin
        << " max_delta=" << step_deltas[i] << " kl=" << step_kls[i] << '\n';
    if (i == divergence) {
      const auto a = cached_steps[i].token;
      const auto b = cold_steps[i].token;
      out << "vision-diag divergence_logits step=" << i << " token_a=" << a
          << " (L_cached=" << cached_rows[i][a]
          << ", L_cold=" << cold_rows[i][a] << ") token_b=" << b
          << " (L_cached=" << cached_rows[i][b]
          << ", L_cold=" << cold_rows[i][b] << ")\n";
    }
  }
  std::cout << out.str();
}

void CheckAppendSnapshotOracle(
    const std::shared_ptr<const hip::QwenGpuModel>& qwen,
    const std::shared_ptr<models::qwen38_flash_next::Model>& flash,
    const std::filesystem::path& images) {
  auto request = ImageRequest(images / "red.png");
  request.messages.emplace_back(tokenization::ChatRole::kAssistant, "Red.");
  request.messages.emplace_back(tokenization::ChatRole::kUser,
                                "Name only the newest image's color.");
  request.messages.back().images.push_back(
      {0, std::make_shared<const std::vector<std::uint8_t>>(
              core::ReadImageFile(images / "blue.png"))});
  const auto& tokenizer = flash ? flash->tokenizer() : qwen->GetTokenizer();
  const auto encoder = flash ? flash->VisionEncoder() : qwen->VisionEncoder();
  const auto full = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages, {},
          tokenization::ResolveQwenChatOptions(request.reasoning),
          encoder->identity(), 1024));
  const auto equal_logits = [](std::span<const float> a,
                               std::span<const float> b) {
    Require(!a.empty() && a.size() == b.size(), "missing append oracle logits");
    float maximum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      Require(std::isfinite(a[i]) && std::isfinite(b[i]),
              "nonfinite append logits");
      maximum = std::max(maximum, std::abs(a[i] - b[i]));
    }
    std::cout << " append_max_logit_error=" << maximum << '\n';
    Require(maximum == 0, "attaching a future image changed computed state");
  };
  for (std::size_t image = 0; image < full->images.size(); ++image) {
    const auto count = full->images[image].grid.offset;
    auto before = std::make_shared<models::qwen::vision::Prompt>(*full);
    before->images.resize(image);
    before->rope = full->rope.Prefix(count);
    const auto identity = full->IdentityForPrefix(count);
    before->cache_identity.assign(identity.begin(), identity.end());
    if (flash) {
      const std::vector<std::int32_t> tokens(full->tokens.begin(),
                                             full->tokens.end());
      for (const auto mode : {core::SessionMode::kAutoregressive,
                              core::SessionMode::kSpeculative}) {
        if (mode == core::SessionMode::kSpeculative && !flash->HasMtp())
          continue;
        std::string error;
        auto delayed = flash->CreateSession(mode, 1024, &error);
        auto attached = flash->CreateSession(mode, 1024, &error);
        auto restored = flash->CreateSession(mode, 1024, &error);
        Require(delayed && attached && restored, error);
        delayed->ConfigureVision(before);
        attached->ConfigureVision(full);
        Require(delayed->Sync(std::span(tokens).first(count), &error), error);
        Require(attached->Sync(std::span(tokens).first(count), &error), error);
        equal_logits(delayed->Logits(), attached->Logits());
        delayed->ConfigureVision(full);
        Require(delayed->Position() == count, "image append reset the prefix");
        auto snapshot = delayed->SaveSnapshot(&error);
        Require(snapshot != nullptr, error);
        restored->ConfigureVision(full);
        Require(restored->RestoreSnapshot(*snapshot, &error), error);
        restored->ConfigureVision(full);
        Require(delayed->Sync(tokens, &error) &&
                    attached->Sync(tokens, &error) &&
                    restored->Sync(tokens, &error),
                error);
        equal_logits(delayed->Logits(), attached->Logits());
        equal_logits(delayed->Logits(), restored->Logits());
        sampling::SamplingConfig config;
        config.temperature = 0.8F;
        config.top_k = 20;
        config.seed = 1234;
        sampling::SamplerState a(config, full->tokens), b(config, full->tokens);
        models::qwen38_flash_next::Session::DecodeResult left, right;
        Require(delayed->DecodeStep(4, a, &left, &error) &&
                    restored->DecodeStep(4, b, &right, &error),
                error);
        Require(left.tokens == right.tokens && left.stop == right.stop,
                "restored appended image changed sampled AR/MTP decoding");
      }
    } else {
      hip::QwenGpuExecutor delayed(qwen, 1024), attached(qwen, 1024),
          restored(qwen, 1024);
      const auto tokens = std::span(full->tokens);
      delayed.ConfigureVision(before, encoder);
      attached.ConfigureVision(full, encoder);
      (void)delayed.ForwardPromptBatch(tokens.first(count));
      (void)attached.ForwardPromptBatch(tokens.first(count));
      equal_logits(delayed.CopyLastLogits(), attached.CopyLastLogits());
      delayed.ConfigureVision(full, encoder);
      auto snapshot = delayed.SaveSnapshot(count);
      Require(snapshot->PayloadBytes() ==
                  delayed.SnapshotPayloadBytes(count) +
                      image * sizeof(models::qwen::vision::ImageGrid),
              "snapshot accounts only consumed image metadata");
      restored.ConfigureVision(full, encoder);
      restored.RestoreSnapshot(*snapshot);
      restored.ConfigureVision(full, encoder);
      int cancellation_checks = 0;
      restored.SetCancellationCheck([&] { return ++cancellation_checks >= 5; });
      bool vision_cancelled = false;
      try {
        (void)restored.ForwardPromptBatch(tokens.subspan(count), count);
      } catch (const std::runtime_error& error) {
        vision_cancelled =
            std::string_view(error.what()) == "vision encoding cancelled";
      }
      Require(vision_cancelled && cancellation_checks == 5,
              "prefill must observe cancellation inside image encoding");
      restored.SetCancellationCheck({});
      restored.RestoreSnapshot(*snapshot);
      restored.ConfigureVision(full, encoder);
      (void)restored.ForwardPromptBatch(tokens.subspan(count), count);
      (void)delayed.ForwardPromptBatch(tokens.subspan(count), count);
      (void)attached.ForwardPromptBatch(tokens.subspan(count), count);
      equal_logits(delayed.CopyLastLogits(), restored.CopyLastLogits());

      // The completed image embedding is reusable after restoration, so the
      // next cancellation reaches the text layers rather than the encoder.
      restored.RestoreSnapshot(*snapshot);
      restored.ConfigureVision(full, encoder);
      cancellation_checks = 0;
      restored.SetCancellationCheck([&] { return ++cancellation_checks >= 5; });
      bool cancelled = false;
      try {
        (void)restored.ForwardPromptBatch(tokens.subspan(count), count);
      } catch (const std::runtime_error& error) {
        cancelled = std::string_view(error.what()) == "Qwen prefill cancelled";
      }
      Require(cancelled && cancellation_checks == 5,
              "prefill must observe cancellation within the layer stack");
      // Exercise bounded submission too, against the uncancellable executor.
      restored.SetCancellationCheck([] { return false; });
      restored.RestoreSnapshot(*snapshot);
      restored.ConfigureVision(full, encoder);
      (void)restored.ForwardPromptBatch(tokens.subspan(count), count);
      equal_logits(delayed.CopyLastLogits(), attached.CopyLastLogits());
      equal_logits(delayed.CopyLastLogits(), restored.CopyLastLogits());
    }
  }
  std::cout << "text/image and image/image snapshot oracle: exact\n";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 1)
    return 77;
  try {
    constexpr std::string_view usage =
        "usage: qwen_vision_serving_test MODEL "
        "-|mtp-embedded|mtp=PATH|dflash=PATH IMAGE_DIRECTORY|synthetic "
        "[--mmproj PATH] [--disk-only|--append-only]";
    Require(argc >= 4, usage);
    bool disk_only = false;
    bool append_only = false;
    std::filesystem::path mmproj;
    for (int i = 4; i < argc; ++i) {
      const std::string_view option(argv[i]);
      if (option == "--disk-only") {
        disk_only = true;
      } else if (option == "--append-only") {
        append_only = true;
      } else if (option == "--mmproj" && i + 1 < argc) {
        mmproj = argv[++i];
      } else {
        Require(false, usage);
      }
    }
    Require(!(disk_only && append_only), usage);
    const std::string model_path(argv[1]);
    // The speculative backend is named, never inferred from the artifacts.
    const std::string_view speculation(argv[2]);
    server::TextSpeculativeConfig speculative;
    std::string draft;
    if (speculation == "mtp-embedded") {
      speculative.backend = server::TextSpeculativeBackend::kMtp;
    } else if (speculation.starts_with("mtp=")) {
      speculative.backend = server::TextSpeculativeBackend::kMtp;
      draft = speculation.substr(4);
      Require(!draft.empty(), usage);
    } else if (speculation.starts_with("dflash=")) {
      speculative.backend = server::TextSpeculativeBackend::kDFlash;
      draft = speculation.substr(7);
      Require(!draft.empty(), usage);
    } else {
      Require(speculation == "-", usage);
    }
    const bool speculation_requested =
        speculative.backend != server::TextSpeculativeBackend::kDisabled;
    const bool embedded_mtp = speculation == "mtp-embedded";
    RemoveDirectory synthetic_images;
    std::filesystem::path images(argv[3]);
    if (images == "synthetic") {
      images = TemporaryDirectory("gufo-vision-images");
      synthetic_images.path = images;
      WriteSolidPng(images / "red.png", 256, 256, {255, 0, 0});
      WriteSolidPng(images / "blue.png", 256, 256, {0, 0, 255});
      WriteSolidPng(images / "blue-wide.png", 512, 128, {0, 0, 255});
    }
    std::string error;
    auto reader = std::shared_ptr<const core::GgufReader>(
        core::GgufReader::OpenFile(model_path, &error));
    Require(reader != nullptr, error);
    constexpr std::uint32_t context = 1024;
    const bool flash =
        reader->GetMetadataString("general.architecture") == "qwen4exp";
    std::shared_ptr<const hip::QwenGpuModel> qwen;
    std::shared_ptr<models::qwen38_flash_next::Model> qfn;
    if (flash) {
      Require(speculative.backend != server::TextSpeculativeBackend::kDFlash,
              "Flash-Next has no DFlash backend");
      speculative.draft_model_path = draft;
      qfn = models::qwen38_flash_next::Model::Load(
          model_path, {.max_context = context, .mtp_model_path = draft},
          &error);
      Require(qfn != nullptr && qfn->VisionEncoder() != nullptr, error);
      Require(!speculation_requested || qfn->HasMtp(),
              "MTP was requested but the Flash-Next model has no MTP head");
    } else {
      // The backend refuses to load when the checkpoint has no embedded head.
      speculative.draft_model_path = embedded_mtp ? "embedded" : draft;
      // Qwen3.6-35B-A3B projects image rows to 2048, Qwen3.8-27B to 5120.
      const std::uint32_t width =
          reader->GetMetadataUint64("qwen35moe.embedding_length") == 2048
              ? 2048
              : 5120;
      auto encoder =
          models::qwen::vision::Encoder::Open(model_path, mmproj, width);
      Require(encoder != nullptr, "missing vision sidecar");
      qwen =
          hip::QwenGpuModel::CreateFromGguf(reader, &error, std::move(encoder));
      Require(qwen != nullptr, error);
    }
    std::cout << "speculation=" << speculation << " architecture="
              << reader->GetMetadataString("general.architecture").value_or("")
              << '\n';
    if (!disk_only)
      CheckAppendSnapshotOracle(qwen, qfn, images);
    if (append_only)
      return 0;
    const auto load = [&](bool use_spec,
                          const server::TextDiskCacheConfig& cache = {}) {
      auto backend = std::make_unique<Backend>();
      const auto mode =
          use_spec ? speculative : server::TextSpeculativeConfig{};
      auto mode_cache = cache;
      if (!use_spec)
        mode_cache.draft_model_artifact_fingerprint.clear();
      const bool loaded = flash ? backend->load(qfn, &error, context, 4, {}, {},
                                                mode, mode_cache)
                                : backend->load(qwen, &error, context, 4, {},
                                                {}, mode, mode_cache);
      Require(loaded, error);
      return backend;
    };
    std::array requests{
        ImageRequest(images / "red.png"), ImageRequest(images / "blue.png"),
        ImageRequest(images / "blue-wide.png"), server::ChatRequest{}};
    requests[3].messages.emplace_back(
        tokenization::ChatRole::kUser,
        "Continue counting from one, with commas between the numbers.");
    requests[3].reasoning.enabled = false;
    auto ar = load(false);
    sampling::SamplingConfig greedy;
    greedy.temperature = 0;
    sampling::SamplingConfig sampled;
    sampled.temperature = 0.8F;
    sampled.top_k = 30;
    sampled.top_p = 0.9F;
    sampled.min_p = 0.05F;
    sampled.seed = 47;
    std::array<Backend::Result, 4> references;
    std::unique_ptr<Backend> spec;
    if (!disk_only) {
      for (std::size_t i = 0; i < requests.size(); ++i) {
        references[i] = ar->chat(requests[i], 16, greedy);
        Require(!references[i].cache_hit,
                "different images shared a token-only cache entry");
        const auto replay = ar->chat(requests[i], 16, greedy);
        Require(replay.cache_hit && replay.prefill_tokens == 0 &&
                    replay.tokens == references[i].tokens,
                "image prefix replay is not exact");
        std::cout << "AR " << i << ": " << references[i].text << '\n';
      }
      Require(references[0].tokens != references[1].tokens &&
                  references[0].prompt_tokens == references[1].prompt_tokens,
              "same-size different images did not influence generation "
              "independently");
      Require(Lower(references[0].text).find("red") != std::string::npos &&
                  Lower(references[1].text).find("blue") != std::string::npos,
              "solid-color image recognition failed");
      ar.reset();
      ar = load(false);
      CheckConcurrent(*ar, requests, references, greedy);
      if (speculation_requested) {
        spec = load(true);
        std::array<Backend::Result, 4> speculative_references;
        std::array<Backend::Result, 4> speculative_replays;
        for (std::size_t i = 0; i < requests.size(); ++i) {
          speculative_references[i] = spec->chat(requests[i], 16, greedy);
          Require(speculative_references[i].tokens == references[i].tokens,
                  "greedy image speculation differs from autoregression");
          // The retained prefix serves the retry without any prefill.
          speculative_replays[i] = spec->chat(requests[i], 16, greedy);
          const auto& replay = speculative_replays[i];
          Require(replay.cache_hit && !replay.cache_disk_hit &&
                      replay.prefill_tokens == 0 &&
                      replay.cached_prompt_tokens == replay.prompt_tokens &&
                      replay.tokens == references[i].tokens,
                  "speculative image prefix replay is not exact");
        }
        RequireAcceptedDrafts(speculative_references, "speculative cold");
        RequireAcceptedDrafts(speculative_replays, "speculative RAM replay");
        spec.reset();
        spec = load(true);
        CheckConcurrent(*spec, requests, speculative_references, greedy, flash);
      }
      // Sampled replay and per-request RNG independence. Existing sampler
      // suites cover the distribution itself; this checks image-path
      // integration.
      for (auto* backend : {ar.get(), spec.get()}) {
        if (!backend)
          continue;
        std::array<Backend::Result, 4> expected;
        for (std::size_t i = 0; i < requests.size(); ++i) {
          expected[i] = backend->chat(requests[i], 16, sampled);
          Require(expected[i].tokens ==
                      backend->chat(requests[i], 16, sampled).tokens,
                  "seeded image generation is not reproducible");
        }
        CheckConcurrent(*backend, requests, expected, sampled,
                        flash && backend == spec.get());
        auto interrupted = requests[0];
        interrupted.messages.back().content =
            "Name the image color, then count from one to one hundred.";
        std::vector<Backend::Result> resumed_results;
        for (const auto& sampling : {greedy, sampled}) {
          const auto expected = backend->chat(interrupted, 8, sampling);
          std::size_t pieces = 0;
          const auto cancelled =
              backend->chat(interrupted, 128, sampling, {},
                            [&](std::string_view) { return ++pieces < 3; });
          Require(cancelled.cancelled,
                  "image cancellation fixture did not interrupt generation");
          const auto resumed = backend->chat(interrupted, 8, sampling);
          Require(resumed.cache_hit && resumed.prefill_tokens == 0 &&
                      resumed.cached_prompt_tokens == resumed.prompt_tokens &&
                      resumed.tokens == expected.tokens,
                  "cancelled image request lost its exact reusable frontier");
          resumed_results.push_back(resumed);
        }
        if (backend == spec.get())
          RequireAcceptedDrafts(resumed_results, "speculative after cancel");
        // The slot that was cancelled and reused answers like a backend that
        // never saw the request.
        {
          const auto cold = load(backend == spec.get());
          std::size_t i = 0;
          for (const auto& sampling : {greedy, sampled}) {
            const auto fresh = cold->chat(interrupted, 8, sampling);
            Require(
                i > 0 || (!fresh.cache_hit && fresh.cached_prompt_tokens == 0),
                "the cold cancellation reference reused a prefix");
            Require(fresh.tokens == resumed_results[i].tokens,
                    "a cancelled and reused slot differs from a cold backend");
            ++i;
          }
        }
        std::cout << "image cancellation/resume exact and equal to cold, "
                     "speculative="
                  << (backend == spec.get()) << '\n';
      }
    } else {
      references[0] = ar->chat(requests[0], 16, greedy);
    }
    // Cancellation above introduces a fifth independent prompt into four
    // slots. Re-establish the conversation before requiring live-prefix reuse.
    const auto root_replay = ar->chat(requests[0], 16, greedy);
    Require(root_replay.tokens == references[0].tokens,
            "image root changed after cancellation or cache eviction");
    references[0] = root_replay;
    auto continuation = requests[0];
    continuation.messages.emplace_back(tokenization::ChatRole::kAssistant,
                                       references[0].text);
    continuation.messages.emplace_back(
        tokenization::ChatRole::kUser,
        "What color did you see? Explain in one sentence.");
    auto multi_image = requests[0];
    multi_image.messages.back().content =
        "Reply with two color names only: first image, then second image.";
    multi_image.messages.back().images.push_back(
        requests[1].messages.back().images.front());
    const auto continued = ar->chat(continuation, 16, greedy);
    Require(continued.cache_hit && continued.cached_prompt_tokens > 0 &&
                continued.prefill_tokens > 0,
            "image continuation did not reuse its prefix");
    if (!disk_only) {
      const auto multiple = ar->chat(multi_image, 32, greedy);
      std::cout << "multiple images: " << multiple.text << '\n';
      const auto colors = Lower(multiple.text);
      Require(colors.find("red") != std::string::npos &&
                  colors.find("blue") != std::string::npos &&
                  colors.find("red") < colors.find("blue"),
              "multiple image order/content was lost");
    }
    ar.reset();
    spec.reset();
    // Fresh executors are independent cold oracles for a continued image chat.
    ar = load(false);
    const auto cold_continuation = ar->chat(continuation, 16, greedy);
    const bool continuation_exact =
        cold_continuation.tokens == continued.tokens;
    if (!continuation_exact) {
      std::cout << "continued cache: " << continued.text
                << "\ncontinued cold: " << cold_continuation.text << '\n';
      std::cout << "cached IDs:";
      for (auto token : continued.tokens)
        std::cout << ' ' << token;
      std::cout << "\ncold IDs:";
      for (auto token : cold_continuation.tokens)
        std::cout << ' ' << token;
      std::cout << '\n';
    }
    ar.reset();
    if (!flash)
      CheckQwenIncrementalOracle(qwen, continuation, references[0], continued);
    if (!flash)
      DumpQwenVisionContinuationDiagnostic(qwen, continuation, references[0],
                                           continued, cold_continuation.tokens);
    if (flash)
      CheckFlashIncrementalOracle(qfn, continuation, continued);

    const RemoveDirectory cleanup{TemporaryDirectory("gufo-vision-cache")};
    server::TextDiskCacheConfig cache{
        .directory = cleanup.path,
        .capacity_bytes = std::size_t{2} << 30,
        .staging_capacity_bytes = std::size_t{512} << 20,
        .model_artifact_fingerprint = core::GgufIdentityHex(*reader)};
    if (!draft.empty()) {
      auto draft_reader = core::GgufReader::OpenFile(draft, &error);
      Require(draft_reader != nullptr, error);
      cache.draft_model_artifact_fingerprint =
          core::GgufIdentityHex(*draft_reader);
    }
    // Disk spacing may retain the stable user frontier rather than another
    // snapshot just a few assistant-prefix tokens later. Bound re-prefill by
    // that actual template suffix, while keeping token equality strict.
    // The in-memory retries above still require zero prefill.
    std::array<std::size_t, 2> disk_frontiers{};
    for (std::size_t i = 0; i < disk_frontiers.size(); ++i) {
      const auto prompt = models::qwen::vision::Prepare(
          flash ? qfn->tokenizer() : qwen->GetTokenizer(), requests[i].messages,
          {},
          tokenization::ResolveQwenChatOptions(requests[i].reasoning,
                                               requests[i].add_vision_id),
          (flash ? qfn->VisionEncoder() : qwen->VisionEncoder())->identity(),
          context);
      disk_frontiers[i] = prompt.stable_prefix_tokens;
      Require(
          disk_frontiers[i] > 0 && disk_frontiers[i] <= prompt.tokens.size(),
          "image prompt has no stable disk frontier");
    }
    const auto restored_prompt = [&](const Backend::Result& result,
                                     std::size_t i) {
      return result.cache_disk_hit &&
             result.cached_prompt_tokens >= disk_frontiers[i] &&
             result.cached_prompt_tokens + result.prefill_tokens ==
                 result.prompt_tokens;
    };
    for (const bool use_spec : {false, true}) {
      if (use_spec && !speculation_requested)
        continue;
      std::vector<Backend::Result> disk_results;
      cache.directory = cleanup.path / (use_spec ? "spec" : "ar");
      auto first = load(use_spec, cache);
      const auto red = first->chat(requests[0], 16, greedy);
      const auto sampled_red = first->chat(requests[0], 16, sampled);
      // Shutdown drains bounded persistence. A GPU-backed Qwen snapshot can
      // occupy most of the staging budget; a second concurrent enqueue is
      // allowed to miss. Qualify restoration independently of disk speed.
      first.reset();
      first = load(use_spec, cache);
      const auto blue = first->chat(requests[1], 16, greedy);
      const auto sampled_blue = first->chat(requests[1], 16, sampled);
      Require(
          red.cache_disk_queued_bytes > 0 && blue.cache_disk_queued_bytes > 0,
          "image snapshots did not reach disk");
      first.reset();
      {
        auto replay = load(use_spec, cache);
        for (const auto i : {0U, 1U}) {
          const auto result = replay->chat(requests[i], 16, sampled);
          std::cout << "sampled disk image=" << i << " speculative=" << use_spec
                    << " hit=" << result.cache_disk_hit
                    << " prefill=" << result.prefill_tokens
                    << " cached=" << result.cached_prompt_tokens
                    << " prompt=" << result.prompt_tokens << " exact="
                    << (result.tokens ==
                        (i == 0 ? sampled_red.tokens : sampled_blue.tokens))
                    << '\n';
          Require(restored_prompt(result, i) &&
                      result.tokens ==
                          (i == 0 ? sampled_red.tokens : sampled_blue.tokens),
                  "disk-restored sampled image replay differs");
          disk_results.push_back(result);
        }
      }
      auto restored = load(use_spec, cache);
      for (const auto i : {0U, 1U}) {
        const auto result = restored->chat(requests[i], 16, greedy);
        std::cout << "disk image=" << i << " speculative=" << use_spec
                  << " hit=" << result.cache_disk_hit
                  << " prefill=" << result.prefill_tokens << " exact="
                  << (result.tokens == (i == 0 ? red.tokens : blue.tokens))
                  << '\n';
        Require(restored_prompt(result, i) &&
                    result.tokens == (i == 0 ? red.tokens : blue.tokens),
                "disk-restored image state or identity differs");
        disk_results.push_back(result);
      }
      if (use_spec)
        RequireAcceptedDrafts(disk_results, "speculative disk restore");
      const auto result = restored->chat(continuation, 16, greedy);
      Require(result.cached_prompt_tokens > red.prompt_tokens,
              "continued image check must reuse generated history");
      if (!flash)
        CheckQwenIncrementalOracle(qwen, continuation, red, result);
      if (flash || !disk_only)
        Require(result.tokens == continued.tokens,
                "disk-restored image continuation differs");
      std::cout << "disk image identity/layout replay exact, speculative="
                << use_spec << '\n';
    }
    if (!disk_only)
      Require(continuation_exact,
              "continued image cache differs from cold prefill");
    std::cout << (disk_only ? "image disk restoration: passed\n"
                            : "image AR/speculation, sampling, concurrency and "
                              "disk restoration: passed\n");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
