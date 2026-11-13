#ifndef GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_SAMPLED_FLOAT_BOUND_HPP_
#define GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_SAMPLED_FLOAT_BOUND_HPP_

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "tests/core/sampled_speculation_oracle.hpp"

namespace gufo::test {

// How far single precision may move the device samplers from the
// double-precision reference.
//
// With m the largest kept logit, a kept token has x = (l - m) / T <= 0 and
// the weight w = exp(x). In float, with e = 2^-24: the subtraction, the
// reciprocal of T and the product each round once, so x carries a relative
// error of 3e, which moves w by 3|x|e; expf is taken to be within 4 ulp. A
// penalized logit is the result of up to three more float operations, each
// within e of the magnitudes involved. A weight is thus within
//   rel = (3|x| + 4 + [penalties] 4(|l| + |m|)/T) e
// of w. Sums of non-negative floats add at most e per addition: a segment of
// the kept tokens and the 256 segment sums give (L + 256) e, relative.
// Hence for the total W:       dW = sum(w rel) + (L + 256) e W.
// The accepted mass of a draft token d, q min(1, p/q) = min(p, q), follows
// p = w_d / W:                 |d min(p, q)| <= p (rel_d + dW/W + 2e).
// A residual weight max(0, w - q W) moves by w rel + q dW + e q W + e v, and
// their sum V = Z W by         dV = sum(w rel) + (dW + e W) Q + (L + 257) e V
// with Q the draft mass on the kept tokens. An interval of the residual CDF
// is a quotient of such sums, so its length moves by at most 2 dV / V; a
// rejection has probability Z, which makes the contribution to the emitted
// distribution 2 dV / W. Bounds are doubled for the second-order terms.
// Values below FLT_MIN may be flushed to zero on the device; that moves a
// probability by less than 4 FLT_MIN. The last terms are the resolution of
// the bisection that measures the device.
struct FloatBound {
  /// dW / W.
  double total{0.0};
  /// rel of each kept token, in target order.
  std::vector<double> weight;
  /// Allowed error of an interval of the ordinary sampler.
  double sampled{0.0};
  /// Allowed contribution of the residual to the emitted distribution.
  double emitted_residual{0.0};

  /// Allowed error of the accepted mass min(p, q) of the kept token `index`
  /// (target.size() for a token outside the target).
  [[nodiscard]] double Accepted(std::size_t index, double p, double q) const {
    const double relative =
        total + (2 * 0x1.0p-24) + (index < weight.size() ? weight[index] : 0);
    return (2.0 * p * relative) + (4.0 * FLT_MIN) + (q * 0x1.0p-40);
  }
};

/// `draft` may be null for the ordinary sampler. `vocabulary` is the length
/// of the logit row.
inline FloatBound BoundFloat(std::span<const sampled_oracle::Candidate> target,
                             float temperature, bool penalized,
                             std::size_t vocabulary,
                             const sampled_oracle::DraftRow* draft,
                             double residual_mass) {
  constexpr double e = 0x1.0p-24;
  const double segment =
      std::min<double>(static_cast<double>(target.size()),
                       std::ceil(static_cast<double>(vocabulary) / 256.0));
  FloatBound bound;
  double total = 0.0;
  double moved = 0.0;
  double draft_mass = 0.0;
  for (const auto& candidate : target) {
    const double x = (candidate.logit - target.front().logit) / temperature;
    const double weight = std::exp(x);
    double relative = (3.0 * std::abs(x)) + 4.0;
    if (penalized)
      relative += 4.0 *
                  (std::abs(candidate.logit) + std::abs(target.front().logit)) /
                  temperature;
    bound.weight.push_back(relative * e);
    total += weight;
    moved += weight * relative * e;
    if (draft != nullptr)
      draft_mass += draft->Probability(candidate.token);
  }
  const double total_moved = moved + ((segment + 256.0) * e * total);
  bound.total = total_moved / total;
  bound.sampled = (4.0 * bound.total) + (4.0 * FLT_MIN) + 0x1.0p-28;
  const double residual = residual_mass * total;
  const double residual_moved = moved +
                                ((total_moved + (e * total)) * draft_mass) +
                                ((segment + 257.0) * e * residual);
  bound.emitted_residual =
      (4.0 * residual_moved / total) + (4.0 * FLT_MIN) + 0x1.0p-28;
  return bound;
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_SAMPLED_FLOAT_BOUND_HPP_
