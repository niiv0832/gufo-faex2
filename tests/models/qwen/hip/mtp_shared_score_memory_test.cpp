// Lazily allocated workspaces of the MTP head on real weights: the rows of a
// shared score and the batched priming scratch.
//
// Each workspace is several device buffers. A failed allocation in the middle
// leaves none of them behind, the next call allocates the whole set again,
// and every buffer the head allocated is freed with it. GetMemoryUsage grows
// by exactly the workspace when it appears, a head that led a shared score
// keeps its rows after another head leads, and no head ever holds more than
// EstimateMemoryUsage reserves.
//
// qwen_mtp_shared_score_memory_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it or without an embedded MTP head.

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
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::hip::QwenMtpGpuExecutor;
using gufo::tokenization::TokenId;

constexpr std::uint32_t kContext = 4096;
constexpr std::size_t kHeads = 4;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Accounting mismatches are reported together; Expect stops the run where
// going on would use a half-built workspace.
int failures{0};

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

// Counts the head's device allocations and fails a chosen one.
struct Probe {
  std::size_t allocated{0};
  std::size_t allocated_bytes{0};
  std::size_t freed{0};
  std::size_t calls{0};
  std::size_t fail_at{0};  // 1-based call to refuse; 0 refuses none
} probe;

hipError_t ProbeAllocate(void** pointer, std::size_t bytes) {
  if (++probe.calls == probe.fail_at)
    return hipErrorOutOfMemory;
  const hipError_t status = hipMalloc(pointer, bytes);
  if (status == hipSuccess) {
    ++probe.allocated;
    probe.allocated_bytes += bytes;
  }
  return status;
}

hipError_t ProbeFree(void* pointer) {
  ++probe.freed;
  return hipFree(pointer);
}

struct Window {
  std::size_t allocated{probe.allocated};
  std::size_t allocated_bytes{probe.allocated_bytes};
  std::size_t freed{probe.freed};
  explicit Window(std::size_t fail_at = 0) {
    probe.calls = 0;
    probe.fail_at = fail_at;
  }
  ~Window() { probe.fail_at = 0; }
  [[nodiscard]] std::size_t Allocated() const {
    return probe.allocated - allocated;
  }
  [[nodiscard]] std::size_t Bytes() const {
    return probe.allocated_bytes - allocated_bytes;
  }
  [[nodiscard]] std::size_t Freed() const { return probe.freed - freed; }
};

std::vector<float> HiddenRows(std::size_t rows, std::size_t width) {
  std::vector<float> values(rows * width);
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (auto& value : values) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    value = (static_cast<float>(state >> 40) / 8388608.0F) - 1.0F;
  }
  return values;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_mtp_shared_score_memory_test [MODEL.gguf]\n";
    return 2;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
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
    const auto target = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (!target)
      throw std::runtime_error("model: " + error);
    const auto mtp = gufo::hip::QwenMtpGpuModel::CreateEmbedded(target, &error);
    if (!mtp) {
      std::cerr << "no embedded MTP head (" << error << "); skipping\n";
      return 77;
    }
    const auto& config = mtp->GetConfig();
    const std::size_t hidden = config.hidden_size;
    const auto layout = gufo::hip::detail::ComputeQwenMtpHeadMemoryLayout(
        config, kContext, QwenMtpGpuExecutor::KeepsHalfKvOnly(config));
    const auto reserved =
        QwenMtpGpuExecutor::EstimateMemoryUsage(*mtp, kContext);
    std::cout << "config,hidden=" << hidden << ",vocab=" << config.vocab_size
              << ",layers=" << config.num_layers
              << ",intermediate=" << config.intermediate_size
              << ",kv_bytes=" << layout.kv_bytes
              << ",fixed_scratch=" << layout.fixed_scratch_bytes
              << ",prime_scratch=" << layout.prime_scratch_bytes
              << ",shared_score=" << layout.shared_score_bytes
              << ",reserved_state=" << reserved.request_state_bytes
              << ",reserved_scratch=" << reserved.temporary_scratch_bytes
              << '\n';
    if (hidden == 2048 && config.vocab_size == 248320) {
      Check(layout.shared_score_bytes == 8011808,
            "the shared score of Qwen3.6-35B-A3B takes 8011808 bytes");
    }
    Check(reserved.request_state_bytes == layout.kv_bytes,
          "the reserved state is the head's K/V cache");
    Check(reserved.temporary_scratch_bytes == layout.ReservedScratchBytes(),
          "the reserve covers the fixed, priming and shared-score buffers");

    auto& allocator = gufo::hip::detail::MtpDeviceAllocator();
    const auto production_allocator = allocator;
    allocator = {.allocate = &ProbeAllocate, .free = &ProbeFree};
    const Window lifetime;

    std::vector<std::unique_ptr<QwenMtpGpuExecutor>> owned;
    std::vector<QwenMtpGpuExecutor*> heads;
    for (std::size_t index = 0; index < kHeads; ++index) {
      const Window window;
      auto head = QwenMtpGpuExecutor::Create(mtp, kContext, &error);
      if (!head)
        throw std::runtime_error("MTP executor: " + error);
      Check(window.Bytes() == layout.kv_bytes + layout.fixed_scratch_bytes,
            "a new head allocates another size than its layout: " +
                std::to_string(window.Bytes()));
      Expect(head->SupportsSharedScore(), "this LM head has no shared score");
      heads.push_back(head.get());
      owned.push_back(std::move(head));
    }
    const auto scratch = [&](std::size_t index) {
      return heads[index]->GetMemoryUsage().temporary_scratch_bytes;
    };
    for (std::size_t index = 0; index < kHeads; ++index) {
      const auto usage = heads[index]->GetMemoryUsage();
      Check(usage.request_state_bytes == layout.kv_bytes &&
                usage.temporary_scratch_bytes == layout.fixed_scratch_bytes,
            "a new head holds its cache and fixed scratch only");
    }

    const auto rows = HiddenRows(kHeads, hidden);
    const auto step = [&] {
      for (std::size_t index = 0; index < kHeads; ++index) {
        heads[index]->Rewind(0);
        (void)heads[index]->ForwardTargetHidden(
            static_cast<TokenId>(1000 + index),
            std::span(rows).subspan(index * hidden, hidden), 0, false);
        heads[index]->FinishStep();
      }
    };
    step();
    std::vector<TokenId> single(kHeads);
    for (std::size_t index = 0; index < kHeads; ++index)
      single[index] = heads[index]->ScoreLast();

    // A refused allocation leaves no part of the shared-score rows behind.
    for (const std::size_t fail_at : {1U, 2U, 3U}) {
      bool refused = false;
      const Window window(fail_at);
      try {
        (void)QwenMtpGpuExecutor::ScoreTogether(heads, false);
      } catch (const std::exception&) {
        refused = true;
      }
      const bool ok = refused && window.Allocated() == fail_at - 1 &&
                      window.Freed() == window.Allocated() &&
                      !heads[0]->HoldsSharedScoreWorkspace();
      std::cout << "shared_score,fail_at=" << fail_at << ",refused=" << refused
                << ",allocated=" << window.Allocated()
                << ",freed=" << window.Freed()
                << ",held=" << heads[0]->HoldsSharedScoreWorkspace()
                << ",scratch=" << scratch(0) << ',' << (ok ? "PASS" : "FAIL")
                << '\n';
      Expect(ok, "a refused shared-score allocation leaves a partial set");
      Check(scratch(0) == layout.fixed_scratch_bytes,
            "a refused shared-score allocation changes the held memory");
    }

    // The retry allocates the whole set and scores.
    {
      const Window window;
      const auto tokens = QwenMtpGpuExecutor::ScoreTogether(heads, false);
      bool ok =
          window.Allocated() == 3 && window.Freed() == 0 &&
          window.Bytes() == layout.shared_score_bytes &&
          heads[0]->HoldsSharedScoreWorkspace() &&
          scratch(0) - layout.fixed_scratch_bytes == layout.shared_score_bytes;
      for (std::size_t index = 0; index < kHeads; ++index) {
        ok = ok && tokens[index] == single[index];
        ok = ok &&
             (index == 0 || (!heads[index]->HoldsSharedScoreWorkspace() &&
                             scratch(index) == layout.fixed_scratch_bytes));
      }
      std::cout << "shared_score,retry,allocated=" << window.Allocated()
                << ",lead_scratch_growth="
                << scratch(0) - layout.fixed_scratch_bytes
                << ",expected=" << layout.shared_score_bytes << ','
                << (ok ? "PASS" : "FAIL") << '\n';
      Check(ok, "the first shared score adds exactly its rows to the lead");
    }

    // Another head leads: it allocates its own rows, the former lead keeps
    // its set, and a head that led before allocates nothing.
    std::size_t held = 1;
    for (const std::size_t lead : {2U, 0U, 3U, 1U, 2U}) {
      std::vector<QwenMtpGpuExecutor*> order{heads[lead]};
      for (std::size_t index = 0; index < kHeads; ++index) {
        if (index != lead)
          order.push_back(heads[index]);
      }
      const bool led_before = heads[lead]->HoldsSharedScoreWorkspace();
      step();
      const Window window;
      const auto tokens = QwenMtpGpuExecutor::ScoreTogether(order, false);
      held += led_before ? 0 : 1;
      std::size_t holding = 0;
      std::size_t total = 0;
      bool ok = window.Allocated() == (led_before ? 0U : 3U) &&
                window.Freed() == 0 && tokens[0] == single[lead];
      for (std::size_t index = 0; index < kHeads; ++index) {
        const bool holds = heads[index]->HoldsSharedScoreWorkspace();
        holding += holds;
        total += scratch(index);
        ok = ok &&
             scratch(index) == layout.fixed_scratch_bytes +
                                   (holds ? layout.shared_score_bytes : 0) &&
             scratch(index) <= reserved.temporary_scratch_bytes;
      }
      ok = ok && holding == held;
      std::cout << "lead=" << lead << ",allocated=" << window.Allocated()
                << ",heads_holding_rows=" << holding
                << ",pool_scratch=" << total << ',' << (ok ? "PASS" : "FAIL")
                << '\n';
      Check(ok, "a change of lead is not reflected in the held memory");
    }
    Check(held == kHeads, "every head led once");

    // The priming scratch follows the same rule.
    const std::vector<TokenId> prime_tokens{11, 12, 13, 14};
    const auto prime_rows = HiddenRows(prime_tokens.size(), hidden);
    for (const std::size_t fail_at : {1U, 2U, 3U}) {
      bool refused = false;
      heads[0]->Rewind(0);
      const Window window(fail_at);
      try {
        heads[0]->PrimeBatch(prime_tokens, prime_rows, 0);
      } catch (const std::exception&) {
        refused = true;
      }
      const bool ok = refused && window.Allocated() == fail_at - 1 &&
                      window.Freed() == window.Allocated();
      std::cout << "prime,fail_at=" << fail_at << ",refused=" << refused
                << ",allocated=" << window.Allocated()
                << ",freed=" << window.Freed() << ',' << (ok ? "PASS" : "FAIL")
                << '\n';
      Expect(ok, "a refused priming allocation leaves a partial set");
      Check(
          scratch(0) == layout.fixed_scratch_bytes + layout.shared_score_bytes,
          "a refused priming allocation changes the held memory");
    }
    {
      const Window window;
      heads[0]->PrimeBatch(prime_tokens, prime_rows, 0);
      const auto usage = heads[0]->GetMemoryUsage();
      const bool ok =
          window.Allocated() == 3 &&
          window.Bytes() == layout.prime_scratch_bytes &&
          usage.temporary_scratch_bytes == reserved.temporary_scratch_bytes &&
          usage.request_state_bytes == reserved.request_state_bytes;
      std::cout << "prime,retry,allocated=" << window.Allocated()
                << ",scratch=" << usage.temporary_scratch_bytes
                << ",reserved=" << reserved.temporary_scratch_bytes << ','
                << (ok ? "PASS" : "FAIL") << '\n';
      Check(ok, "a head holding every workspace equals its reserve");
    }

    heads.clear();
    owned.clear();
    const bool balanced = lifetime.Allocated() == lifetime.Freed();
    std::cout << "lifetime,allocated=" << lifetime.Allocated()
              << ",freed=" << lifetime.Freed() << ','
              << (balanced ? "PASS" : "FAIL") << '\n';
    allocator = production_allocator;
    Check(balanced, "the heads did not free every buffer they allocated");
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_mtp_shared_score_memory_test: " << failure.what()
              << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen MTP shared score memory test.\n";
  return 77;
#endif
}
