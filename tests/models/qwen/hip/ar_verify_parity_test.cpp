// Target contract of speculative verification on real weights: the logits a
// verification chunk computes for a row must equal, bit for bit, the logits
// ordinary single-token decoding computes for the same token at the same
// position after the same history. Sampled speculation accepts and resamples
// against the verification distribution, so any difference changes the
// distribution the user is served.
//
// The test keeps the production order: prefill the history, SaveState at the
// current token, then either decode token by token or verify the chunk. It
// also reports the first layer whose output differs, which localizes a
// regression to one operation.
//
// qwen_ar_verify_parity_test [MODEL.gguf [IDS.txt PREFIX ROWS]]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it or for a model without experts.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

constexpr std::size_t kTapsPerPass = 9;

struct Case {
  std::string name;
  std::vector<Token> ids;
  std::uint32_t prefix;
  std::uint32_t rows;
};

struct Delta {
  std::size_t differing = 0;
  double max_abs = 0.0;
};

Delta Compare(std::span<const float> reference,
              std::span<const float> candidate) {
  if (reference.size() != candidate.size())
    throw std::runtime_error("compared spans differ in size");
  Delta delta;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (std::memcmp(&reference[index], &candidate[index], sizeof(float)) != 0)
      ++delta.differing;
    delta.max_abs = std::max(delta.max_abs,
                             std::abs(static_cast<double>(candidate[index]) -
                                      static_cast<double>(reference[index])));
  }
  return delta;
}

// KL(softmax(reference) || softmax(candidate)) in nats.
double Divergence(std::span<const float> reference,
                  std::span<const float> candidate) {
  const auto log_partition = [](std::span<const float> logits) {
    const double top = *std::ranges::max_element(logits);
    double sum = 0.0;
    for (const float value : logits)
      sum += std::exp(static_cast<double>(value) - top);
    return top + std::log(sum);
  };
  const double reference_z = log_partition(reference);
  const double candidate_z = log_partition(candidate);
  double divergence = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double log_p = reference[index] - reference_z;
    divergence += std::exp(log_p) * (log_p - (candidate[index] - candidate_z));
  }
  return divergence;
}

void Prefill(Executor& executor, const Case& test,
             std::span<const std::uint32_t> taps) {
  executor.Reset();
  executor.SetPromptHiddenCapture(true, taps);
  (void)executor.ForwardPromptBatch(std::span(test.ids).first(test.prefix), 0,
                                    true);
  // The history is processed; the current token ids[prefix] is not.
  executor.SaveState(test.prefix);
}

bool RunCase(Executor& executor, const Case& test, std::size_t layers,
             std::size_t hidden, std::size_t vocab) {
  bool equal = true;
  std::vector<float> ar_logits;
  std::vector<float> verify_logits;
  std::ptrdiff_t first_layer = -1;
  for (std::size_t first = 0; first < layers; first += kTapsPerPass) {
    std::vector<std::uint32_t> taps;
    for (std::size_t layer = first;
         layer < std::min(layers, first + kTapsPerPass); ++layer)
      taps.push_back(static_cast<std::uint32_t>(layer));
    const std::size_t width = taps.size() * hidden;

    Prefill(executor, test, taps);
    std::vector<float> ar_hidden;
    ar_logits.clear();
    for (std::uint32_t row = 0; row < test.rows; ++row) {
      (void)executor.ForwardToken(test.ids[test.prefix + row],
                                  test.prefix + row, true);
      const auto logits = executor.CopyLastLogits();
      ar_logits.insert(ar_logits.end(), logits.begin(), logits.end());
      const auto features = executor.CopyLastHidden();
      if (features.size() != width)
        throw std::runtime_error("decode layer capture has a wrong shape");
      ar_hidden.insert(ar_hidden.end(), features.begin(), features.end());
    }
    executor.FinishVerification();

    Prefill(executor, test, taps);
    const gufo::hip::QwenGpuVerificationItem item{
        &executor, std::span(test.ids).subspan(test.prefix, test.rows),
        test.prefix, true};
    (void)Executor::ForwardVerificationBatch(std::span(&item, 1));
    const auto verify_hidden = executor.GetVerificationHiddenStates();
    const auto logits = executor.GetVerificationLogits();
    if (verify_hidden.size() != test.rows * width ||
        logits.size() != test.rows * vocab)
      throw std::runtime_error("verification capture has a wrong shape");
    verify_logits.assign(logits.begin(), logits.end());
    executor.FinishVerification();

    for (std::size_t tap = 0; tap < taps.size(); ++tap) {
      for (std::uint32_t row = 0; row < test.rows; ++row) {
        const std::size_t offset = (row * width) + (tap * hidden);
        const auto delta = Compare(std::span(ar_hidden).subspan(offset, hidden),
                                   verify_hidden.subspan(offset, hidden));
        if (delta.differing == 0)
          continue;
        equal = false;
        if (first_layer < 0)
          first_layer = static_cast<std::ptrdiff_t>(taps[tap]);
        std::cout << test.name << ",layer=" << taps[tap] << ",row=" << row
                  << ",differing=" << delta.differing
                  << ",max_abs=" << delta.max_abs << '\n';
      }
    }
  }
  for (std::uint32_t row = 0; row < test.rows; ++row) {
    const auto reference = std::span(ar_logits).subspan(row * vocab, vocab);
    const auto candidate = std::span(verify_logits).subspan(row * vocab, vocab);
    const auto delta = Compare(reference, candidate);
    equal = equal && delta.differing == 0;
    std::cout << test.name << ",logits,row=" << row
              << ",position=" << test.prefix + row
              << ",differing=" << delta.differing
              << ",max_abs=" << delta.max_abs
              << ",kl=" << Divergence(reference, candidate) << '\n';
  }
  std::cout << test.name << ",prefix=" << test.prefix << ",rows=" << test.rows
            << ",first_differing_layer=" << first_layer << ','
            << (equal ? "PASS" : "FAIL") << '\n';
  return equal;
}

std::vector<Token> ReadIds(const std::string& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot read " + path);
  std::vector<Token> ids;
  std::uint64_t value = 0;
  while (input >> value)
    ids.push_back(static_cast<Token>(value));
  if (!input.eof())
    throw std::runtime_error("malformed token ids in " + path);
  return ids;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc != 1 && argc != 2 && argc != 5) {
    std::cerr << "Usage: qwen_ar_verify_parity_test [MODEL.gguf [IDS.txt "
                 "PREFIX ROWS]]\n";
    return 2;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
  }
  try {
    // A short Russian sentence; position 15 with two verification rows is the
    // input on which the audit of 2026-11-13 found the two paths to differ.
    const std::vector<Token> russian{
        52337, 12666,  22078,  1739,   52033, 172007, 8688,   318,
        12666, 94498,  157,    123,    114,   44122,  1892,   153366,
        7347,  205495, 155277, 151194, 44122, 1892,   232021, 11};
    std::vector<Case> cases;
    if (argc == 5) {
      cases.push_back({"file", ReadIds(argv[2]),
                       static_cast<std::uint32_t>(std::stoul(argv[3])),
                       static_cast<std::uint32_t>(std::stoul(argv[4]))});
    } else {
      cases.push_back({"ru-p15-r2", russian, 15, 2});
      cases.push_back({"ru-p15-r1", russian, 15, 1});
      cases.push_back({"ru-p1-r2", russian, 1, 2});
      cases.push_back({"ru-p16-r8", russian, 16, 8});
    }
    gufo::models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
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
    if (!config.IsMoe()) {
      std::cerr << "the model has no experts; skipping\n";
      return 77;
    }
    const auto executor = Executor::Create(model, &error, 4096);
    if (!executor)
      throw std::runtime_error("executor: " + error);
    bool passed = true;
    for (const auto& test : cases) {
      if (test.rows == 0 || test.prefix == 0 ||
          test.ids.size() < test.prefix + test.rows)
        throw std::runtime_error("case " + test.name + " is out of range");
      for (const Token id : test.ids)
        if (id >= config.vocab_size)
          throw std::runtime_error("token id outside the vocabulary");
      passed = RunCase(*executor, test, config.num_layers, config.hidden_size,
                       config.vocab_size) &&
               passed;
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_ar_verify_parity_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  return 77;
#endif
}
