#include "src/models/qwen/mtp_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/mtp_draft_vocab.hpp"

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void TestTopKDistribution() {
  using gufo::speculative::MtpTopKDistribution;
  // ids 0..5; the best three at temperature 1 are 4, 1, 3.
  const std::vector<float> logits{0.0F, 3.0F, -1.0F, 2.0F, 4.0F, 1.0F};
  std::vector<gufo::tokenization::TokenId> ids;
  std::vector<float> probs;
  MtpTopKDistribution(logits, 1.0F, 3, &ids, &probs);
  Expect(ids == std::vector<gufo::tokenization::TokenId>{4, 1, 3},
         "top-k keeps the highest logits in descending order");
  const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
  Expect(std::abs(sum - 1.0) < 1e-6, "top-k probabilities are normalized");
  const double e = std::exp(1.0);
  Expect(std::abs(probs[0] - e * e / (e * e + e + 1.0)) < 1e-6,
         "probabilities are the renormalized softmax over the kept ids");
  MtpTopKDistribution(logits, 0.5F, 2, &ids, &probs);
  Expect(std::abs(probs[0] - 1.0 / (1.0 + std::exp(-2.0))) < 1e-6,
         "temperature sharpens the draft distribution");
  MtpTopKDistribution(logits, 1.0F, 99, &ids, &probs);
  Expect(ids.size() == logits.size(), "k never exceeds the scored vocabulary");
}

// The one-pass selection must return exactly what sorting every id did:
// the same ids in the same order and bit-identical probabilities, with ties
// and rows shorter than k.
void TestTopKMatchesFullSort() {
  using gufo::speculative::MtpTopKDistribution;
  using gufo::tokenization::TokenId;
  std::uint64_t state = 0x243F6A8885A308D3ULL;
  const auto next = [&state] {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(state >> 33);
  };
  for (const std::size_t size :
       {std::size_t{1}, std::size_t{31}, std::size_t{32}, std::size_t{33},
        std::size_t{5000}, std::size_t{248320}}) {
    for (int round = 0; round < 4; ++round) {
      std::vector<float> logits(size);
      // Coarse values on odd rounds force many exact ties around the cut.
      for (auto& value : logits)
        value = round % 2 == 0
                    ? (static_cast<float>(next() % 2000001) / 50000.0F) - 20.0F
                    : static_cast<float>(next() % 9) - 4.0F;
      for (const float temperature : {1.0F, 0.7F}) {
        std::vector<TokenId> ids, reference_ids(size);
        std::vector<float> probs, reference_probs;
        MtpTopKDistribution(logits, temperature, 32, &ids, &probs);
        std::iota(reference_ids.begin(), reference_ids.end(), TokenId{0});
        const std::size_t k = std::min<std::size_t>(32, size);
        std::partial_sort(
            reference_ids.begin(),
            reference_ids.begin() + static_cast<std::ptrdiff_t>(k),
            reference_ids.end(), [&](TokenId a, TokenId b) {
              return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
            });
        reference_ids.resize(k);
        const double top = logits[reference_ids[0]];
        std::vector<double> weights(k);
        double sum = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
          weights[i] = std::exp((logits[reference_ids[i]] - top) / temperature);
          sum += weights[i];
        }
        for (std::size_t i = 0; i < k; ++i)
          reference_probs.push_back(static_cast<float>(weights[i] / sum));
        Expect(ids == reference_ids, "one-pass top-k keeps the sorted ids");
        Expect(probs == reference_probs,
               "one-pass top-k keeps the sorted probabilities bit for bit");
      }
    }
  }
}

void TestSampleRow() {
  using gufo::speculative::MtpSampleRow;
  const std::vector<gufo::tokenization::TokenId> ids{7, 8, 9};
  const std::vector<float> probs{0.5F, 0.3F, 0.2F};
  Expect(MtpSampleRow(ids, probs, 0.0) == 7 && MtpSampleRow(ids, probs, 0.49) == 7,
         "low uniforms select the first candidate");
  Expect(MtpSampleRow(ids, probs, 0.5) == 8 && MtpSampleRow(ids, probs, 0.79) == 8,
         "inverse CDF selects the middle candidate");
  Expect(MtpSampleRow(ids, probs, 0.99999) == 9,
         "high uniforms select the last candidate");
  const std::vector<float> tail{1.0F, 0.0F, 0.0F};
  Expect(MtpSampleRow(ids, tail, 0.9999999) == 7,
         "zero-probability candidates are never sampled");
}

void TestMoeCohortGate() {
  using gufo::speculative::MtpMoeSkipsDraft;
  Expect(!MtpMoeSkipsDraft(true, 1, 2) && !MtpMoeSkipsDraft(true, 2, 2),
         "small MoE cohorts keep the draft");
  Expect(MtpMoeSkipsDraft(true, 3, 2) && MtpMoeSkipsDraft(true, 4, 2),
         "wide MoE cohorts verify their experts without a draft");
  Expect(!MtpMoeSkipsDraft(false, 4, 2), "dense targets are never gated");
  Expect(!MtpMoeSkipsDraft(true, 4, 0), "a zero limit disables the gate");
}

}  // namespace

void TestDraftVocabularyRule() {
  using gufo::speculative::MtpDraftScoresToken;
  for (const char* kept :
       {"", " the", "\n\n", "<|im_end|>", "naïve", "Привет", " ёж", "π", "—",
        "→", "≈", "\xF0\x9F\x98\x80", "\xD0", "\x82", "\xF0"}) {
    Expect(MtpDraftScoresToken(kept), std::string("draft drops: ") + kept);
  }
  // Chinese, Japanese, Korean, Arabic, Hebrew, Hindi, Thai, fullwidth forms,
  // a CJK extension character, and a lead byte of a CJK sequence.
  for (const char* dropped :
       {"今天", "こんにちは", "한국", "مرحبا", "שלום", "नमस्ते", "สวัสดี", "，",
        "\xF0\xA0\x80\x80", "\xE4", " a\xE4\xBB"}) {
    Expect(!MtpDraftScoresToken(dropped),
           std::string("draft keeps: ") + dropped);
  }
}

void TestDraftScriptWindow() {
  using gufo::speculative::kMtpDraftScriptWindow;
  using gufo::speculative::MtpDraftNeedsWholeVocabulary;
  using gufo::tokenization::TokenId;
  // ids 0..9 are listed except 7.
  std::vector<bool> listed(10, true);
  listed[7] = false;
  std::vector<TokenId> history(200, 3);
  Expect(!MtpDraftNeedsWholeVocabulary(history, listed),
         "a listed history drafts from the draft vocabulary");
  Expect(!MtpDraftNeedsWholeVocabulary({}, listed), "an empty history too");
  history.back() = 7;
  Expect(MtpDraftNeedsWholeVocabulary(history, listed),
         "an unlisted last token switches to the whole vocabulary");
  history.back() = 3;
  history[history.size() - kMtpDraftScriptWindow] = 7;
  Expect(MtpDraftNeedsWholeVocabulary(history, listed),
         "the window covers its oldest token");
  history[history.size() - kMtpDraftScriptWindow] = 3;
  history[history.size() - kMtpDraftScriptWindow - 1] = 7;
  Expect(!MtpDraftNeedsWholeVocabulary(history, listed),
         "a token older than the window is forgotten");
  const std::vector<TokenId> beyond{3, 42};
  Expect(MtpDraftNeedsWholeVocabulary(beyond, listed),
         "an id beyond the vocabulary counts as unlisted");
}

int main() {
  try {
    TestDraftVocabularyRule();
    TestDraftScriptWindow();
    TestTopKDistribution();
    TestTopKMatchesFullSort();
    TestSampleRow();
    TestMoeCohortGate();
  } catch (const std::exception& e) {
    std::cerr << "MTP policy test failed: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen_mtp_policy_test: ok\n";
  return 0;
}
