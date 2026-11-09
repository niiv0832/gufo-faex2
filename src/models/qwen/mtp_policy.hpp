#ifndef GUFO_MODELS_QWEN_MTP_POLICY_HPP_
#define GUFO_MODELS_QWEN_MTP_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::speculative {

// Candidates reported per sampled MTP row. Rejection sampling is lossless for
// any truncation because the verifier uses exactly these probabilities; a
// wider row only brings the draft closer to the target's top-k distribution.
inline constexpr std::size_t kMtpSampledTopK = 32;

// The k highest logits (descending) and their softmax at `temperature`,
// renormalized over the kept ids in double precision.
inline void MtpTopKDistribution(std::span<const float> logits,
                                float temperature, std::size_t k,
                                std::vector<tokenization::TokenId>* ids,
                                std::vector<float>* probs) {
  k = std::min(k, logits.size());
  ids->resize(logits.size());
  std::iota(ids->begin(), ids->end(), tokenization::TokenId{0});
  const auto by_logit = [&](tokenization::TokenId a, tokenization::TokenId b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  };
  std::partial_sort(ids->begin(), ids->begin() + static_cast<std::ptrdiff_t>(k),
                    ids->end(), by_logit);
  ids->resize(k);
  probs->resize(k);
  if (k == 0)
    return;
  const double top = logits[(*ids)[0]];
  std::vector<double> weights(k);
  double sum = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    weights[i] = std::exp((logits[(*ids)[i]] - top) / temperature);
    sum += weights[i];
  }
  for (std::size_t i = 0; i < k; ++i)
    (*probs)[i] = static_cast<float>(weights[i] / sum);
}

// Inverse-CDF draw from one candidate row; never returns a zero-probability id.
inline tokenization::TokenId MtpSampleRow(
    std::span<const tokenization::TokenId> ids, std::span<const float> probs,
    double u) {
  double cumulative = 0.0;
  std::size_t last = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (!(probs[i] > 0.0F))
      continue;
    last = i;
    cumulative += probs[i];
    if (u < cumulative)
      return ids[i];
  }
  return ids[last];
}

// MoE targets verify each draft row against its own routed experts: with
// more than `max_cohort` sessions decoding, no draft is cheaper (Qwen3.6-35B-A3B
// MTP-1 at four sessions: 33.5 vs 36.8 t/s per session, gfx1151 2026-11-09).
// A zero limit disables the gate.
inline bool MtpMoeSkipsDraft(bool moe, std::size_t cohort,
                             std::size_t max_cohort) {
  return moe && max_cohort > 0 && cohort > max_cohort;
}

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_MTP_POLICY_HPP_
