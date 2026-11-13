#ifndef GUFO_TESTS_CORE_SAMPLED_SPECULATION_ORACLE_HPP_
#define GUFO_TESTS_CORE_SAMPLED_SPECULATION_ORACLE_HPP_

// Double-precision reference for sampled speculative decoding. It shares no
// code with src/: the target distribution, the acceptance probability, the
// residual distribution and the draft row are written here from their
// definitions, so a test can hold the production CPU and GPU samplers to
// them.
//
// A draft row reports probabilities q for its candidates. The verifier
// accepts a draft token x with probability a(x) = min(1, p(x) / q(x));
// otherwise it emits a token of the residual r(x) = max(0, p(x) - q(x)) / Z,
// or of p itself when no residual remains. The reported row is a row of
// rounded floats and need not sum to one, so the candidates are not drawn
// with exactly the probabilities reported: an inverse-CDF draw over [0, 1)
// gives every candidate the interval its reported probability spans, cuts
// the row at one, and leaves what the row is short of one to its last
// positive candidate. With q' those probabilities, the emitted token has
//   P(x) = q'(x) a(x) + (sum_y q'(y) (1 - a(y))) r(x).
// P equals p when the row sums to one. In general
//   sum_x |P(x) - p(x)| <= 2 |sum(q) - 1|:
// with the reported masses in place of q' the law is p + (sum(q) - 1) r,
// the law is linear in the masses with coefficients of total weight one, and
// q' differs from q by |sum(q) - 1| in total.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

namespace sampled_oracle {

using Token = std::uint32_t;

/// Request sampling controls, applied in the documented order: penalties,
/// temperature, top-k, top-p, min-p.
struct Filters {
  float temperature{1.0F};
  int top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  float repeat_penalty{1.0F};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};
};

struct Penalty {
  Token token{0};
  std::uint32_t generated{0};
  bool repeated{false};
};

struct Candidate {
  Token token{0};
  double logit{0.0};
  double probability{0.0};
};

/// The target distribution p, most likely first (ties: the lower id).
/// `allowed`, when given, is a constraint mask indexed by token id.
inline std::vector<Candidate> TargetDistribution(
    std::span<const float> logits, const Filters& filters,
    std::span<const Penalty> penalties = {},
    std::span<const std::uint8_t> allowed = {}) {
  std::vector<Candidate> kept;
  for (std::size_t id = 0; id < logits.size(); ++id) {
    if (!std::isfinite(logits[id]) || (!allowed.empty() && allowed[id] == 0))
      continue;
    double logit = logits[id];
    for (const Penalty& penalty : penalties) {
      if (penalty.token != id)
        continue;
      if (penalty.repeated && filters.repeat_penalty != 1.0F) {
        const double scale = filters.repeat_penalty;
        logit = logit <= 0.0 ? logit * scale : logit / scale;
      }
      logit -= static_cast<double>(filters.frequency_penalty) *
               static_cast<double>(penalty.generated);
      if (penalty.generated != 0)
        logit -= static_cast<double>(filters.presence_penalty);
    }
    kept.push_back({static_cast<Token>(id), logit, 0.0});
  }
  if (kept.empty())
    throw std::runtime_error("oracle: no finite allowed logit");
  std::stable_sort(
      kept.begin(), kept.end(),
      [](const Candidate& a, const Candidate& b) { return a.logit > b.logit; });
  if (filters.temperature == 0.0F) {
    kept.resize(1);
    kept.front().probability = 1.0;
    return kept;
  }
  const double temperature = filters.temperature;
  const auto floor_of = [&](std::size_t size) {
    return std::min(size, std::max<std::size_t>(filters.min_keep, 1));
  };
  if (filters.top_k > 0) {
    kept.resize(
        std::min(kept.size(), std::max(static_cast<std::size_t>(filters.top_k),
                                       floor_of(kept.size()))));
  }
  const auto weight = [&](const Candidate& candidate) {
    return std::exp((candidate.logit - kept.front().logit) / temperature);
  };
  if (filters.top_p < 1.0F && kept.size() > 1) {
    double total = 0.0;
    for (const Candidate& candidate : kept)
      total += weight(candidate);
    const double target = static_cast<double>(filters.top_p) * total;
    const std::size_t floor = floor_of(kept.size());
    double cumulative = 0.0;
    std::size_t keep = 0;
    while (keep < kept.size()) {
      cumulative += weight(kept[keep]);
      ++keep;
      if (keep >= floor && cumulative >= target)
        break;
    }
    kept.resize(keep);
  }
  if (filters.min_p > 0.0F && kept.size() > 1) {
    const double threshold =
        kept.front().logit +
        temperature * std::log(static_cast<double>(filters.min_p));
    std::size_t keep = 0;
    while (keep < kept.size() && kept[keep].logit >= threshold)
      ++keep;
    kept.resize(std::min(kept.size(), std::max(keep, floor_of(kept.size()))));
  }
  double total = 0.0;
  for (Candidate& candidate : kept) {
    candidate.probability = weight(candidate);
    total += candidate.probability;
  }
  for (Candidate& candidate : kept)
    candidate.probability /= total;
  return kept;
}

inline double ProbabilityOf(std::span<const Candidate> distribution,
                            Token token) {
  for (const Candidate& candidate : distribution)
    if (candidate.token == token)
      return candidate.probability;
  return 0.0;
}

/// One sparse draft row: unique ids with the probabilities reported for them.
struct DraftRow {
  std::vector<Token> ids;
  std::vector<float> probabilities;

  [[nodiscard]] double Probability(Token token) const {
    for (std::size_t i = 0; i < ids.size(); ++i)
      if (ids[i] == token)
        return probabilities[i];
    return 0.0;
  }
  [[nodiscard]] double Sum() const {
    double sum = 0.0;
    for (const float value : probabilities)
      sum += value;
    return sum;
  }
};

struct Verification {
  /// min(1, p(draft) / q(draft)).
  double acceptance{0.0};
  /// sum_x max(0, p(x) - q(x)).
  double residual_mass{0.0};
  /// 1 - sum_y min(p(y), q(y)): the rejected share of the reported masses.
  /// Equals residual_mass when q sums to one.
  double rejection_mass{0.0};
  /// r, in the order of p; empty when residual_mass is zero.
  std::vector<Candidate> residual;
};

inline Verification Verify(std::span<const Candidate> target,
                           const DraftRow& draft, Token draft_token) {
  Verification result;
  const double p = ProbabilityOf(target, draft_token);
  const double q = draft.Probability(draft_token);
  if (!(q > 0.0))
    throw std::runtime_error("oracle: the draft token has no draft mass");
  result.acceptance = std::min(1.0, p / q);
  double overlap = 0.0;
  for (const Candidate& candidate : target) {
    const double mass = draft.Probability(candidate.token);
    overlap += std::min(candidate.probability, mass);
    const double excess = std::max(0.0, candidate.probability - mass);
    if (excess > 0.0) {
      result.residual.push_back({candidate.token, candidate.logit, excess});
      result.residual_mass += excess;
    }
  }
  result.rejection_mass = 1.0 - overlap;
  for (Candidate& candidate : result.residual)
    candidate.probability /= result.residual_mass;
  return result;
}

/// The probability with which an inverse-CDF draw over [0, 1) proposes each
/// slot of `draft`: slots own consecutive intervals of their reported
/// probabilities, the row is cut at one, and the last positive slot also
/// owns what the row is short of one.
inline std::vector<double> ProposedProbabilities(const DraftRow& draft) {
  std::vector<double> proposed(draft.ids.size(), 0.0);
  double begin = 0.0;
  std::size_t last = draft.ids.size();
  for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
    if (!(draft.probabilities[slot] > 0.0F))
      continue;
    const double end = begin + static_cast<double>(draft.probabilities[slot]);
    proposed[slot] = std::min(end, 1.0) - std::min(begin, 1.0);
    begin = end;
    last = slot;
  }
  if (last == draft.ids.size())
    throw std::runtime_error("oracle: the draft row has no mass");
  if (begin < 1.0)
    proposed[last] += 1.0 - begin;
  return proposed;
}

/// The law of the token a propose-verify step emits.
struct Emitted {
  /// P, in the order of p.
  std::vector<Candidate> law;
  /// The chance that the draft is accepted.
  double acceptance_rate{0.0};
  /// max_x |P(x) - p(x)| and sum_x |P(x) - p(x)|.
  double max_error{0.0};
  double total_error{0.0};
  /// 2 |sum(q) - 1|, the bound of total_error.
  double bound{0.0};
};

inline Emitted EmittedLaw(std::span<const Candidate> target,
                          const DraftRow& draft) {
  const auto proposed = ProposedProbabilities(draft);
  std::vector<Candidate> residual;
  double residual_mass = 0.0;
  for (const Candidate& candidate : target) {
    const double excess = std::max(
        0.0, candidate.probability - draft.Probability(candidate.token));
    if (excess > 0.0) {
      residual.push_back({candidate.token, candidate.logit, excess});
      residual_mass += excess;
    }
  }
  for (Candidate& candidate : residual)
    candidate.probability /= residual_mass;
  // Without a residual a rejected draft is replaced from p.
  const std::span<const Candidate> correction =
      residual.empty() ? target : std::span<const Candidate>(residual);

  Emitted result;
  for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
    if (proposed[slot] > 0.0)
      result.acceptance_rate +=
          proposed[slot] *
          std::min(1.0, ProbabilityOf(target, draft.ids[slot]) /
                            static_cast<double>(draft.probabilities[slot]));
  }
  for (const Candidate& candidate : target) {
    double accepted = 0.0;
    for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
      if (draft.ids[slot] == candidate.token && proposed[slot] > 0.0)
        accepted =
            proposed[slot] *
            std::min(1.0, candidate.probability /
                              static_cast<double>(draft.probabilities[slot]));
    }
    const double value =
        accepted + ((1.0 - result.acceptance_rate) *
                    ProbabilityOf(correction, candidate.token));
    result.law.push_back({candidate.token, candidate.logit, value});
    const double error = std::abs(value - candidate.probability);
    result.max_error = std::max(result.max_error, error);
    result.total_error += error;
  }
  result.bound = 2.0 * std::abs(draft.Sum() - 1.0);
  return result;
}

/// The draft row of an MTP head: the k highest logits (ties: the lower
/// index) and their softmax at `temperature`. Indices are positions in
/// `scored`, the head's own numbering.
struct HeadRow {
  std::vector<std::size_t> indices;
  std::vector<double> probabilities;
};

inline HeadRow HeadTopK(std::span<const float> scored, float temperature,
                        std::size_t k) {
  std::vector<std::size_t> order(scored.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  k = std::min(k, order.size());
  std::partial_sort(
      order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k),
      order.end(), [&](std::size_t a, std::size_t b) {
        return scored[a] != scored[b] ? scored[a] > scored[b] : a < b;
      });
  HeadRow row;
  row.indices.assign(order.begin(),
                     order.begin() + static_cast<std::ptrdiff_t>(k));
  double total = 0.0;
  for (const std::size_t index : row.indices) {
    row.probabilities.push_back(
        std::exp((static_cast<double>(scored[index]) -
                  static_cast<double>(scored[row.indices.front()])) /
                 static_cast<double>(temperature)));
    total += row.probabilities.back();
  }
  for (double& value : row.probabilities)
    value /= total;
  return row;
}

// --- The sampler's random numbers --------------------------------------------
// The sampler draws xorshift64* numbers: the state is updated, multiplied by a
// constant, and its upper 53 bits make a uniform in [0, 1). These helpers
// build the state that yields a chosen uniform, so a test can place draws on
// and around the boundaries of a distribution.

inline constexpr std::uint64_t kStarMultiplier = UINT64_C(0x2545F4914F6CDD1D);

constexpr std::uint64_t NextState(std::uint64_t state) {
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state;
}

constexpr double UniformOfState(std::uint64_t next_state) {
  return static_cast<double>((next_state * kStarMultiplier) >> 11U) * 0x1.0p-53;
}

constexpr std::uint64_t PreviousState(std::uint64_t state) {
  // Undo each xorshift by repeating it until the shifted bits run out.
  std::uint64_t value = state;
  for (int i = 0; i < 3; ++i)
    value = state ^ (value >> 27U);
  state = value;
  for (int i = 0; i < 3; ++i)
    value = state ^ (value << 25U);
  state = value;
  for (int i = 0; i < 6; ++i)
    value = state ^ (value >> 12U);
  return value;
}

constexpr std::uint64_t MultiplierInverse() {
  std::uint64_t inverse = kStarMultiplier;
  for (int i = 0; i < 6; ++i)
    inverse *= 2U - kStarMultiplier * inverse;
  return inverse;
}

/// The largest uniform the sampler can draw.
inline constexpr double kLastUniform = 1.0 - 0x1.0p-53;

/// A state whose next draw is `uniform`, rounded down to 53 bits. The 11
/// bits a draw discards select among 2048 such states.
constexpr std::uint64_t StateDrawing(double uniform, std::uint64_t spare = 1) {
  const auto bits = static_cast<std::uint64_t>(uniform * 0x1.0p53);
  const std::uint64_t output = (bits << 11U) | (spare & 0x7FFU);
  return PreviousState(output * MultiplierInverse());
}

// --- Recovering a sampler's step function
// -------------------------------------

/// A maximal run of uniforms [begin, end) a sampler maps to one outcome.
struct Run {
  std::int64_t outcome{0};
  double begin{0.0};
  double end{0.0};
};

/// The step function u -> outcome of an inverse-CDF sampler over
/// [0, kLastUniform], found by bisection down to adjacent doubles or `depth`
/// halvings. `grid` initial points guard against outcomes that are not
/// contiguous.
inline std::vector<Run> RecoverRuns(
    const std::function<std::int64_t(double)>& sample, int depth = 64,
    int grid = 16) {
  std::map<double, std::int64_t> starts;
  const std::function<void(double, std::int64_t, double, std::int64_t, int)>
      explore = [&](double low, std::int64_t at_low, double high,
                    std::int64_t at_high, int level) {
        if (at_low == at_high)
          return;
        const double middle = low + ((high - low) / 2.0);
        if (level >= depth || middle <= low || middle >= high) {
          starts[high] = at_high;
          return;
        }
        const std::int64_t at_middle = sample(middle);
        explore(low, at_low, middle, at_middle, level + 1);
        explore(middle, at_middle, high, at_high, level + 1);
      };
  double previous = 0.0;
  std::int64_t at_previous = sample(previous);
  starts[0.0] = at_previous;
  for (int step = 1; step <= grid; ++step) {
    const double point =
        step == grid ? kLastUniform : static_cast<double>(step) / grid;
    const std::int64_t at_point = sample(point);
    explore(previous, at_previous, point, at_point, 0);
    previous = point;
    at_previous = at_point;
  }
  std::vector<Run> runs;
  for (const auto& [begin, outcome] : starts) {
    if (!runs.empty())
      runs.back().end = begin;
    runs.push_back({outcome, begin, 1.0});
  }
  return runs;
}

/// Total length of the runs of each outcome, and whether every outcome has a
/// single run.
struct Measure {
  std::map<std::int64_t, double> length;
  bool contiguous{true};
};

inline Measure MeasureRuns(std::span<const Run> runs) {
  Measure measure;
  for (const Run& run : runs) {
    if (measure.length.contains(run.outcome))
      measure.contiguous = false;
    measure.length[run.outcome] += run.end - run.begin;
  }
  return measure;
}

/// max over outcomes of |measured length - expected probability|; outcomes
/// absent from one side count with zero.
inline double MeasureError(const Measure& measure,
                           std::span<const Candidate> expected) {
  double worst = 0.0;
  for (const Candidate& candidate : expected) {
    const auto found = measure.length.find(candidate.token);
    worst = std::max(
        worst, std::abs((found == measure.length.end() ? 0.0 : found->second) -
                        candidate.probability));
  }
  for (const auto& [outcome, length] : measure.length) {
    if (outcome < 0 ||
        ProbabilityOf(expected, static_cast<Token>(outcome)) == 0)
      worst = std::max(worst, length);
  }
  return worst;
}

/// The uniform at which a decision that holds for small draws stops holding:
/// 1 when it holds for every draw, 0 when it never does.
inline double DecisionThreshold(const std::function<bool(double)>& holds) {
  if (!holds(0.0))
    return 0.0;
  if (holds(kLastUniform))
    return 1.0;
  double low = 0.0;
  double high = kLastUniform;
  for (;;) {
    const double middle = low + ((high - low) / 2.0);
    if (middle <= low || middle >= high)
      return high;
    (holds(middle) ? low : high) = middle;
  }
}

}  // namespace sampled_oracle

#endif  // GUFO_TESTS_CORE_SAMPLED_SPECULATION_ORACLE_HPP_
