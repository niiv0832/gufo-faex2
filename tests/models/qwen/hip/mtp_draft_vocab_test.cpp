// The MTP draft vocabulary on real weights: the head scores only the rows of
// speculative::MtpDraftScoresToken, and each of those logits is, bit for bit,
// the logit the target's whole LM head gives the same token. Switched to the
// whole vocabulary, the head returns that head's logits.
//
// qwen_mtp_draft_vocab_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it, without an embedded MTP head, or when the head
// scores the whole vocabulary.

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
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/mtp_draft_vocab.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::tokenization::TokenId;

constexpr std::uint32_t kContext = 1024;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

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
    std::cerr << "Usage: qwen_mtp_draft_vocab_test [MODEL.gguf]\n";
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
    const auto& ids = mtp->GetDraftVocab().ids;
    if (ids.empty()) {
      std::cerr << "this head scores the whole vocabulary; skipping\n";
      return 77;
    }
    const auto& tokenizer = target->GetTokenizer();
    const std::size_t vocab = config.vocab_size;
    const std::size_t hidden = config.hidden_size;

    // The list is exactly the tokens the rule keeps, in ascending order.
    std::size_t kept = 0;
    for (std::size_t id = 0; id < vocab; ++id) {
      if (!gufo::speculative::MtpDraftScoresToken(
              tokenizer.DecodeToken(static_cast<TokenId>(id))))
        continue;
      Expect(
          kept < ids.size() && ids[kept] == static_cast<TokenId>(id),
          "draft vocabulary differs from the rule at id " + std::to_string(id));
      ++kept;
    }
    Expect(kept == ids.size() && kept < vocab,
           "draft vocabulary has ids the rule drops");
    // What a Russian and English deployment writes stays draftable.
    for (const char* text :
         {"Привет! Как дела? Сегодня хорошая погода, ёж съел щи.",
          "The quick brown fox — naïve café, 3.14 ≈ π, x → y.",
          "def main():\n    return {\"key\": [1, 2, 3]}\n",
          "<|im_end|>\n<|im_start|>assistant\n<tool_call>"}) {
      for (const TokenId token : tokenizer.Encode(text)) {
        Expect(gufo::speculative::MtpDraftScoresToken(
                   tokenizer.DecodeToken(token)),
               std::string("a token of this text is not draftable: ") + text);
      }
    }
    std::size_t dropped_cjk = 0;
    for (const TokenId token : tokenizer.Encode("今天天气很好，我们去公园。"))
      dropped_cjk +=
          gufo::speculative::MtpDraftScoresToken(tokenizer.DecodeToken(token))
              ? 0
              : 1;
    Expect(dropped_cjk > 0, "the rule keeps Chinese text");

    std::string head_error;
    auto head =
        gufo::hip::QwenMtpGpuExecutor::Create(mtp, kContext, &head_error);
    if (!head)
      throw std::runtime_error("MTP executor: " + head_error);
    Expect(head->ScoredVocab() == ids.size(), "the head scores another count");

    const auto tokens = tokenizer.Encode(
        "Inventory ledger. Entry 1 holds 12 units. Запись 2: семь штук.");
    const auto rows = HiddenRows(tokens.size(), hidden);
    float* d_hidden = nullptr;
    float* d_logits = nullptr;
    HIP_CHECK(hipMalloc(&d_hidden, hidden * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_logits, vocab * sizeof(float)));
    std::vector<float> full(vocab);
    bool passed = true;
    for (std::size_t position = 0; position < tokens.size(); ++position) {
      const TokenId drafted = head->ForwardTargetHidden(
          tokens[position], std::span(rows).subspan(position * hidden, hidden),
          static_cast<std::uint32_t>(position));
      const auto last_hidden = head->CopyLastHidden();
      const auto logits = head->CopyLastLogits();
      Expect(logits.size() == ids.size(), "the head returns another count");
      // The same head input through the target's whole LM head.
      HIP_CHECK(hipMemcpy(d_hidden, last_hidden.data(), hidden * sizeof(float),
                          hipMemcpyHostToDevice));
      const auto& output = mtp->GetWeights().output;
      gufo::hip::LaunchGEMV(output.data, output.type, d_hidden, d_logits, vocab,
                            hidden, nullptr,
                            gufo::models::qwen::QwenGemmMode::kHipMtp);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(full.data(), d_logits, vocab * sizeof(float),
                          hipMemcpyDeviceToHost));
      std::size_t differing = 0;
      std::size_t best = 0;
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const float reference = full[static_cast<std::size_t>(ids[i])];
        differing += std::memcmp(&reference, &logits[i], sizeof(float)) != 0;
        if (logits[i] > logits[best])
          best = i;
      }
      // Every other position repeats the step on the whole vocabulary.
      std::size_t whole_differing = 0;
      if (position % 2 == 1) {
        head->Rewind(static_cast<std::uint32_t>(position));
        head->ScoreWholeVocabulary(true);
        const TokenId whole_drafted = head->ForwardTargetHidden(
            tokens[position],
            std::span(rows).subspan(position * hidden, hidden),
            static_cast<std::uint32_t>(position));
        const auto whole = head->CopyLastLogits();
        Expect(whole.size() == vocab && head->ScoredVocab() == vocab,
               "the whole vocabulary has another count");
        std::size_t whole_best = 0;
        for (std::size_t id = 0; id < vocab; ++id) {
          whole_differing +=
              std::memcmp(&full[id], &whole[id], sizeof(float)) != 0;
          if (whole[id] > whole[whole_best])
            whole_best = id;
        }
        whole_differing += whole_drafted != static_cast<TokenId>(whole_best);
        head->ScoreWholeVocabulary(false);
      }
      differing += whole_differing;
      const bool ok = differing == 0 && drafted == ids[best];
      if (!ok || position == 0 || position + 1 == tokens.size()) {
        std::cout << "position=" << position << ",scored=" << ids.size()
                  << ",vocab=" << vocab << ",differing=" << differing
                  << ",drafted=" << drafted << ",expected=" << ids[best] << ','
                  << (ok ? "PASS" : "FAIL") << '\n';
      }
      passed = passed && ok;
    }
    HIP_CHECK(hipFree(d_hidden));
    HIP_CHECK(hipFree(d_logits));
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_mtp_draft_vocab_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen MTP draft vocabulary test.\n";
  return 77;
#endif
}
