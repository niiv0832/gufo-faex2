#include "src/models/qwen/mtp_policy.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

int main() {
  try {
    TestTopKDistribution();
    TestSampleRow();
    TestMoeCohortGate();
  } catch (const std::exception& e) {
    std::cerr << "MTP policy test failed: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen_mtp_policy_test: ok\n";
  return 0;
}
