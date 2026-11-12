// MTP head with an FP16-only K/V cache (eight query heads per KV head, as in
// Qwen3.6-35B-A3B) on real weights. A snapshot must carry every cached
// position: a head restored from it has to reproduce the draft logits of the
// head that wrote it, bit for bit, at every attention route (online, split-K
// per head, and the shared-KV kernel from 512 positions on).
//
// qwen_mtp_half_kv_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it, without an embedded MTP head, or when the head
// keeps both cache precisions.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::tokenization::TokenId;

constexpr std::uint32_t kContext = 4096;
constexpr std::size_t kSteps = 4;

std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text;
}

// Deterministic rows in [-1, 1): stand-ins for the target's hidden states.
std::vector<float> HiddenRows(std::size_t rows, std::size_t width) {
  std::vector<float> values(rows * width);
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (auto& value : values) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    value = (static_cast<float>(state >> 40) / 8388608.0F) - 1.0F;
  }
  return values;
}

// Draft logits of `kSteps` steps that follow `primed` cached positions.
std::vector<float> Draft(gufo::hip::QwenMtpGpuExecutor& head,
                         std::span<const TokenId> tokens,
                         std::span<const float> hidden, std::size_t primed) {
  const std::size_t width = head.GetHiddenSize();
  std::vector<float> all;
  for (std::size_t i = 0; i < kSteps; ++i) {
    const std::size_t position = primed + i;
    (void)head.ForwardTargetHidden(tokens[position],
                                   hidden.subspan(position * width, width),
                                   static_cast<std::uint32_t>(position));
    const auto logits = head.CopyLastLogits();
    all.insert(all.end(), logits.begin(), logits.end());
  }
  return all;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::unique_ptr<gufo::hip::QwenMtpGpuExecutor> Head(
    const std::shared_ptr<const gufo::hip::QwenMtpGpuModel>& model) {
  std::string error;
  auto head = gufo::hip::QwenMtpGpuExecutor::Create(model, kContext, &error);
  if (!head)
    throw std::runtime_error("MTP executor: " + error);
  return head;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_mtp_half_kv_test [MODEL.gguf]\n";
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
    if (!gufo::hip::QwenMtpGpuExecutor::KeepsHalfKvOnly(config)) {
      std::cerr << "this head keeps both cache precisions; skipping\n";
      return 77;
    }
    const auto tokens = target->GetTokenizer().Encode(Ledger(260));
    const auto hidden = HiddenRows(tokens.size(), config.hidden_size);
    auto writer = Head(mtp);
    auto reader_head = Head(mtp);
    auto damaged = Head(mtp);
    bool passed = true;
    // Online attention, split-K per head, and the shared-KV kernel.
    for (const std::size_t primed :
         {std::size_t{100}, std::size_t{300}, std::size_t{1500}}) {
      if (tokens.size() < primed + kSteps)
        throw std::runtime_error("ledger is shorter than the test needs");
      const auto prefix = std::span(tokens).first(primed);
      const auto prefix_hidden =
          std::span(hidden).first(primed * config.hidden_size);
      writer->Reset();
      writer->PrimeBatch(prefix, prefix_hidden, 0);
      std::vector<std::uint8_t> payload(writer->StateBytes());
      const std::size_t expected =
          sizeof(std::uint32_t) + primed * config.num_key_value_heads *
                                      config.head_dim * 2 *
                                      sizeof(std::uint16_t);
      writer->SaveState(payload);
      const auto reference = Draft(*writer, tokens, hidden, primed);

      reader_head->Reset();
      // Stale cache rows from another prompt must not survive the restore.
      reader_head->PrimeBatch(
          std::span(tokens).subspan(4, primed),
          std::span(hidden).subspan(4 * config.hidden_size,
                                    primed * config.hidden_size),
          0);
      reader_head->LoadState(payload);
      const auto restored = Draft(*reader_head, tokens, hidden, primed);

      // The check must notice a snapshot that loses the later positions:
      // blank the second half of the cached keys and expect other logits.
      auto broken = payload;
      const std::size_t plane_bytes =
          (payload.size() - sizeof(std::uint32_t)) / 2;
      std::fill_n(broken.begin() + sizeof(std::uint32_t) + (plane_bytes / 2),
                  plane_bytes / 2, std::uint8_t{0});
      damaged->Reset();
      damaged->LoadState(broken);
      const auto corrupted = Draft(*damaged, tokens, hidden, primed);

      const bool ok = payload.size() == expected && !reference.empty() &&
                      Same(reference, restored) && !Same(reference, corrupted);
      std::cout << "primed=" << primed << ",state_bytes=" << payload.size()
                << ",restored_equal=" << Same(reference, restored)
                << ",damage_detected=" << !Same(reference, corrupted) << ','
                << (ok ? "PASS" : "FAIL") << '\n';
      passed = passed && ok;
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_mtp_half_kv_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen MTP half-KV test.\n";
  return 77;
#endif
}
