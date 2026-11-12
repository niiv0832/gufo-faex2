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
  // One pass with the k best so far, kept in descending order (ties: the
  // lower id first). Almost every logit fails the single comparison against
  // the current k-th value, so a row of the full vocabulary costs one scan
  // rather than a sort of its ids. Equal logits keep the earlier id, as ids
  // are visited in ascending order.
  ids->resize(k);
  std::vector<float>& best = *probs;
  best.resize(k);
  std::size_t kept = 0;
  for (std::size_t id = 0; id < logits.size() && k != 0; ++id) {
    const float value = logits[id];
    if (kept == k && !(value > best[k - 1]))
      continue;
    std::size_t slot = kept < k ? kept++ : k - 1;
    for (; slot != 0 && value > best[slot - 1]; --slot) {
      best[slot] = best[slot - 1];
      (*ids)[slot] = (*ids)[slot - 1];
    }
    best[slot] = value;
    (*ids)[slot] = static_cast<tokenization::TokenId>(id);
  }
  probs->resize(k);
  if (k == 0)
    return;
  const double top = best[0];
  std::vector<double> weights(k);
  double sum = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    weights[i] = std::exp((best[i] - top) / temperature);
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

// MoE targets verify each draft row against its own routed experts, and each
// session drafts with a head pass of its own: with more than `max_cohort`
// sessions decoding, no draft is cheaper. Qwen3.6-35B-A3B MTP-1 on gfx1151,
// tokens per second summed over the sessions, with and without drafts
// (2026-11-12): three sessions 152 and 132, four 163 and 153, six 167 and
// 187, eight 178 and 208. A zero limit disables the gate.
inline constexpr std::size_t kMtpMoeMaxCohort = 4;

inline bool MtpMoeSkipsDraft(bool moe, std::size_t cohort,
                             std::size_t max_cohort) {
  return moe && max_cohort > 0 && cohort > max_cohort;
}

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_MTP_POLICY_HPP_
