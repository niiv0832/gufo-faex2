// Memory accounting of one target session on real weights. The session runs
// what makes it allocate lazily: a prompt (grouped MoE prefill, the router
// split weights, re-encoded experts, quantized activations), a saved state
// (second recurrent buffer, replay log) and a verification chunk
// (verification logits).
//
// After every step GetMemoryUsage names what the step added and stays within
// EstimateMemoryUsage, and the copies the model derives from its weights stay
// within EstimateDerivedWeightBytes. With sessions > 1 the same holds for
// each session and the derived copies are made once.
//
// qwen_session_memory_test [MODEL.gguf [CONTEXT [SESSIONS]]]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it. The defaults are a 4096-token context and one
// session.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/detail/memory_layout.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::hip::QwenGpuExecutor;
using gufo::hip::QwenGpuMemoryUsage;
using gufo::tokenization::TokenId;

int failures{0};

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

std::size_t DeviceFreeBytes() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  return hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess ? free_bytes
                                                                : 0;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 4) {
    std::cerr << "Usage: qwen_session_memory_test [MODEL.gguf [CONTEXT "
                 "[SESSIONS]]]\n";
    return 2;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
  }
  const auto context =
      static_cast<std::uint32_t>(argc > 2 ? std::atoi(argv[2]) : 4096);
  const auto sessions =
      static_cast<std::size_t>(argc > 3 ? std::atoi(argv[3]) : 1);
  if (context < 1024 || sessions == 0 || sessions > 8) {
    std::cerr << "CONTEXT must be at least 1024 and SESSIONS 1 to 8\n";
    return 2;
  }
  try {
    gufo::models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
    gufo::models::qwen38_flash_next::rocm::RegisterQwenNpuFfn();
    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(path, &error);
    if (!reader_owner)
      throw std::runtime_error("open: " + error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(reader_owner));
    const auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (!model)
      throw std::runtime_error("model: " + error);
    const auto& config = model->GetConfig();
    const auto policy = gufo::hip::QwenExecutionPolicy::Production();
    const auto layout = gufo::hip::detail::ComputeQwenArenaMemoryLayout(
        config, context, policy);
    const auto reserved =
        QwenGpuExecutor::EstimateMemoryUsage(*model, context, policy);
    const std::size_t derived_reserved = model->EstimateDerivedWeightBytes();
    const std::size_t stream_scratch =
        gufo::hip::EstimateQwenMoePrefillStreamScratchBytes(
            config, std::min(context, gufo::hip::detail::kArenaMaxPromptBatch));
    std::cout << "config,hidden=" << config.hidden_size
              << ",layers=" << config.num_layers
              << ",experts=" << config.num_experts << ",context=" << context
              << ",sessions=" << sessions << '\n'
              << "layout,kv=" << layout.kv_bytes
              << ",recurrent=" << layout.recurrent_bytes
              << ",bank=" << layout.bank_bytes
              << ",fixed=" << layout.fixed_scratch_bytes
              << ",moe_decode=" << layout.moe_decode_bytes
              << ",moe_prefill=" << layout.moe_prefill_bytes
              << ",replay_log=" << layout.replay_log_bytes << '\n'
              << "reserved,state=" << reserved.request_state_bytes
              << ",scratch=" << reserved.temporary_scratch_bytes
              << ",moe_stream_scratch=" << stream_scratch
              << ",derived_weights=" << derived_reserved << '\n';
    // The serving profile's reserve, for the record; nothing is allocated.
    constexpr std::uint32_t kServingContext = 131072;
    if (config.context_length >= kServingContext) {
      const auto serving =
          QwenGpuExecutor::EstimateMemoryUsage(*model, kServingContext, policy);
      const auto arena = gufo::hip::QwenGpuArena::EstimateMemoryUsage(
          config, kServingContext, policy);
      std::cout << "reserved_at_" << kServingContext
                << ",state=" << serving.request_state_bytes
                << ",scratch=" << serving.temporary_scratch_bytes
                << ",arena_state=" << arena.request_state_bytes
                << ",arena_scratch=" << arena.temporary_scratch_bytes << '\n';
    }
    Check(model->GetDerivedWeightBytes() == 0,
          "a model nothing ran on holds no derived copies");
    Check(reserved.request_state_bytes == layout.ReservedStateBytes(),
          "the reserved state is the K/V cache and both recurrent buffers");
    Check(reserved.temporary_scratch_bytes >=
              layout.ReservedScratchBytes(gufo::hip::HasQwenNpuFfnFactory()) +
                  stream_scratch,
          "the reserved scratch leaves out arena or stream buffers");
    Check(!config.IsMoe() || (stream_scratch > 0 && derived_reserved > 0),
          "a MoE model reserves nothing for its grouped prefill");

    const auto& tokenizer = model->GetTokenizer();
    std::string text;
    for (int entry = 0; entry < 120; ++entry) {
      text += "Entry " + std::to_string(entry) + " holds " +
              std::to_string((entry * 37) % 91) + " units. ";
    }
    const auto tokens = tokenizer.Encode(text);
    if (tokens.size() < 700)
      throw std::runtime_error("the prompt is shorter than the test needs");
    const auto prompt = std::span(tokens).first(600);
    const auto verified = std::span(tokens).subspan(600, 4);

    const std::size_t free_before = DeviceFreeBytes();
    std::vector<std::unique_ptr<QwenGpuExecutor>> owned;
    for (std::size_t session = 0; session < sessions; ++session) {
      auto executor = QwenGpuExecutor::Create(model, &error, context, policy);
      if (!executor)
        throw std::runtime_error("executor: " + error);
      owned.push_back(std::move(executor));
    }
    const auto within = [&](const QwenGpuMemoryUsage& usage) {
      return usage.request_state_bytes <= reserved.request_state_bytes &&
             usage.temporary_scratch_bytes <= reserved.temporary_scratch_bytes;
    };
    std::size_t held_total = 0;
    for (std::size_t session = 0; session < sessions; ++session) {
      auto& executor = *owned[session];
      const auto report = [&](const char* step, const QwenGpuMemoryUsage& now) {
        std::cout << "session=" << session << ',' << step
                  << ",state=" << now.request_state_bytes
                  << ",scratch=" << now.temporary_scratch_bytes
                  << ",derived_weights=" << model->GetDerivedWeightBytes()
                  << ',' << (within(now) ? "within" : "OVER") << '\n';
        Check(within(now), std::string(step) + " holds more than the reserve");
      };
      const auto created = executor.GetMemoryUsage();
      report("created", created);
      Check(created.request_state_bytes == layout.LiveStateBytes(),
            "a new session holds one recurrent buffer and its K/V cache");
      Check(
          created.temporary_scratch_bytes >= layout.ConstructedScratchBytes() &&
              created.temporary_scratch_bytes <
                  reserved.temporary_scratch_bytes,
          "a new session reports its reserve instead of what it holds");

      (void)executor.ForwardPromptBatch(prompt, 0, true);
      const auto prefilled = executor.GetMemoryUsage();
      report("prompt", prefilled);
      Check(
          prefilled.temporary_scratch_bytes - created.temporary_scratch_bytes >=
              stream_scratch,
          "the grouped prefill scratch of the stream is not counted");
      const std::size_t derived = model->GetDerivedWeightBytes();
      Check(derived <= derived_reserved,
            "the derived copies exceed their reserve");
      Check(!config.IsMoe() || derived > 0,
            "the derived copies a prompt made are not counted");

      const auto position = static_cast<std::uint32_t>(prompt.size());
      executor.SaveState(position);
      const auto saved = executor.GetMemoryUsage();
      report("saved_state", saved);
      Check(saved.request_state_bytes - prefilled.request_state_bytes ==
                layout.recurrent_bytes,
            "a saved state adds exactly the second recurrent buffer");
      Check(saved.request_state_bytes == reserved.request_state_bytes,
            "a session with a saved state holds its whole reserved state");
      Check(saved.temporary_scratch_bytes - prefilled.temporary_scratch_bytes ==
                layout.replay_log_bytes,
            "a saved state adds exactly the replay log to the scratch");

      (void)executor.ForwardVerificationChunk(verified, position);
      executor.FinishVerification();
      executor.RestoreState();
      const auto verified_usage = executor.GetMemoryUsage();
      report("verification", verified_usage);
      Check(verified_usage.temporary_scratch_bytes -
                    saved.temporary_scratch_bytes ==
                verified.size() * config.vocab_size * sizeof(float),
            "a verification chunk adds exactly its rows of logits");
      Check(model->GetDerivedWeightBytes() == derived,
            "the derived copies grew after the first prompt of a session");
      held_total += verified_usage.TotalBytes();
    }
    const std::size_t free_after = DeviceFreeBytes();
    const std::size_t derived_held = model->GetDerivedWeightBytes();
    Check(derived_held == derived_reserved || !config.IsMoe(),
          "a prompt through every layer leaves derived copies unmade");
    // Other processes share the device, so the measured change is reported
    // for a matched run and is not a pass criterion.
    std::cout << "total,sessions_held=" << held_total
              << ",sessions_reserved=" << sessions * reserved.TotalBytes()
              << ",derived_held=" << derived_held
              << ",derived_reserved=" << derived_reserved
              << ",accounted=" << held_total + derived_held
              << ",device_free_drop="
              << (free_before > free_after ? free_before - free_after : 0)
              << '\n';
    std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_session_memory_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen session memory test.\n";
  return 77;
#endif
}
