// The CPU sampler and the MTP draft policy against the independent reference
// of tests/core/sampled_speculation_oracle.hpp.
//
// For each case the reference first proves the scheme itself: with the case's
// target p and draft row q, accepting with min(1, p/q) and resampling from
// max(0, p - q) emits exactly p. The production code is then held to the
// reference:
//   - SamplerState::Distribution is p;
//   - SamplingDistribution::SampleResidual maps the uniforms [0, 1) onto the
//     residual: each token owns one interval whose length is its residual
//     probability;
//   - SamplingDistribution::Sample maps them onto p;
//   - MtpTopKDistribution is the softmax of the head's k best logits and
//     MtpSampleRow draws from the row it reports.
// Both sides compute in double precision; the tolerances are the rounding of
// the sums involved and are derived where they are set.

#include "tests/core/sampled_speculation_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/mtp_policy.hpp"
#include "tests/core/sampled_speculation_cases.hpp"

namespace {

using sampled_cases::Case;
using sampled_oracle::Candidate;
using sampled_oracle::DraftRow;
using sampled_oracle::Token;

bool g_passed = true;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    g_passed = false;
    std::cout << "FAIL: " << message << '\n';
  }
}

gufo::sampling::SamplerState MakeSampler(const Case& test) {
  gufo::sampling::SamplingConfig config;
  config.temperature = test.filters.temperature;
  config.top_k = test.filters.top_k;
  config.top_p = test.filters.top_p;
  config.min_p = test.filters.min_p;
  config.min_keep = test.filters.min_keep;
  config.repeat_penalty = test.filters.repeat_penalty;
  config.frequency_penalty = test.filters.frequency_penalty;
  config.presence_penalty = test.filters.presence_penalty;
  config.repeat_last_n = test.repeat_last_n;
  config.seed = 0;
  gufo::sampling::SamplerState sampler(config, test.prompt);
  sampler.Accept(test.generated);
  return sampler;
}

// Each probability is a quotient of an exponential and a sum of at most
// `terms` exponentials. Two correct double evaluations differ by the rounding
// of the sum, (terms - 1) * 2^-53 relative, and of a few single operations.
double RelativeTolerance(std::size_t terms) {
  return static_cast<double>(terms + 16) * 0x1.0p-52;
}

// The residual excess p - q cancels: an error of `relative * p` in p moves
// the excess by as much unless the excess is zero for every admissible p.
// p never exceeds one, which keeps a row with q = 1 at exactly zero.
double ResidualTolerance(std::span<const Candidate> target,
                         const DraftRow& draft, double relative,
                         double residual_mass) {
  if (!(residual_mass > 0.0))
    return 4.0 * relative;
  double moved = 0.0;
  for (const Candidate& candidate : target) {
    const double highest =
        std::min(1.0, candidate.probability * (1.0 + relative));
    if (highest - draft.Probability(candidate.token) > 0.0)
      moved += relative * candidate.probability;
  }
  return std::min(1.0, (4.0 * moved / residual_mass) + 0x1.0p-50);
}

void CheckCase(const Case& test) {
  const auto target = sampled_oracle::TargetDistribution(
      test.logits, test.filters, test.Penalties(), test.allowed);
  auto sampler = MakeSampler(test);
  const auto produced = sampler.Distribution(test.logits);
  const double relative = RelativeTolerance(test.logits.size());

  // p: the same support and the same probabilities.
  double worst_p = 0.0;
  Expect(produced.entries().size() == target.size(),
         test.name + ": the target support has " +
             std::to_string(produced.entries().size()) + " tokens, expected " +
             std::to_string(target.size()));
  for (const Candidate& candidate : target) {
    const double value = produced.probability(candidate.token);
    const double error = std::abs(value - candidate.probability);
    worst_p = std::max(worst_p, error / candidate.probability);
    Expect(error <= relative * candidate.probability,
           test.name + ": p(" + std::to_string(candidate.token) +
               ") = " + std::to_string(value) + ", expected " +
               std::to_string(candidate.probability));
  }

  // Ordinary sampling draws p.
  {
    const auto runs = sampled_oracle::RecoverRuns([&](double uniform) {
      std::uint64_t state = sampled_oracle::StateDrawing(uniform);
      return static_cast<std::int64_t>(produced.Sample(&state));
    });
    const auto measure = sampled_oracle::MeasureRuns(runs);
    Expect(measure.contiguous, test.name + ": Sample is not an inverse CDF");
    Expect(sampled_oracle::MeasureError(measure, target) <= 2.0 * relative,
           test.name + ": Sample does not draw p");
  }

  double worst_identity = 0.0;
  double worst_residual = 0.0;
  for (const DraftRow& draft : test.drafts) {
    const double draft_sum = draft.Sum();
    for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
      if (!(draft.probabilities[slot] > 0.0F))
        continue;
      const Token draft_token = draft.ids[slot];
      const auto expected = sampled_oracle::Verify(target, draft, draft_token);
      // The scheme emits p up to the mass q is short of one.
      worst_identity = std::max(worst_identity, expected.identity_error);
      Expect(expected.identity_error <= std::abs(1.0 - draft_sum) + 0x1.0p-50,
             test.name + ": the reference scheme does not emit p");
      // The production acceptance test reads this p; it is checked above.
      Expect(
          expected.acceptance ==
              std::min(1.0, sampled_oracle::ProbabilityOf(target, draft_token) /
                                draft.probabilities[slot]),
          test.name + ": acceptance");
    }

    // The residual does not depend on the draft token.
    const auto expected =
        sampled_oracle::Verify(target, draft, draft.ids[test.DraftSlot(draft)]);
    // An empty residual means p <= q everywhere. A row that sums to one is
    // then never rejected; a row above one (within the verifier's 1e-5) can
    // be, and production then falls back to p.
    const auto& reference =
        expected.residual.empty() ? target : expected.residual;
    std::size_t draws = 0;
    const auto runs = sampled_oracle::RecoverRuns([&](double uniform) {
      const std::uint64_t before = sampled_oracle::StateDrawing(uniform);
      std::uint64_t state = before;
      const auto token =
          produced.SampleResidual(draft.ids, draft.probabilities, &state);
      // One draw, or none when a single token remains.
      if (state == sampled_oracle::NextState(before))
        draws = std::max<std::size_t>(draws, 1);
      else
        Expect(state == before && reference.size() == 1,
               test.name + ": SampleResidual consumed an unexpected draw");
      return static_cast<std::int64_t>(token);
    });
    Expect((draws == 1) == (reference.size() > 1),
           test.name + ": SampleResidual draw count");
    const auto measure = sampled_oracle::MeasureRuns(runs);
    Expect(measure.contiguous,
           test.name + ": SampleResidual is not an inverse CDF");
    const double tolerance =
        ResidualTolerance(target, draft, relative, expected.residual_mass);
    const double error = sampled_oracle::MeasureError(measure, reference);
    worst_residual = std::max(worst_residual, error);
    Expect(error <= tolerance, test.name + ": residual differs by " +
                                   std::to_string(error) + ", tolerance " +
                                   std::to_string(tolerance));
    // Mass the draft could not propose is never accepted and stays in the
    // residual in full.
    for (const Candidate& candidate : target) {
      if (draft.Probability(candidate.token) != 0.0)
        continue;
      Expect(expected.residual_mass > 0.0 &&
                 std::abs(sampled_oracle::ProbabilityOf(expected.residual,
                                                        candidate.token) *
                              expected.residual_mass -
                          candidate.probability) <= 0x1.0p-50,
             test.name + ": target mass outside the draft row is lost");
    }

    // The sampler-level entry point is the same residual.
    for (const double uniform : {0.0, 0.25, 0.5, 0.75, 0.999}) {
      auto direct = MakeSampler(test);
      direct.SetRngState(sampled_oracle::StateDrawing(uniform));
      std::uint64_t state = sampled_oracle::StateDrawing(uniform);
      Expect(
          direct.SampleResidual(test.logits, draft.ids, draft.probabilities) ==
                  produced.SampleResidual(draft.ids, draft.probabilities,
                                          &state) &&
              direct.rng_state() == state,
          test.name + ": SamplerState::SampleResidual differs");
    }
  }
  std::cout << test.name << ",support=" << target.size()
            << ",drafts=" << test.drafts.size() << ",p_rel_err=" << worst_p
            << ",identity_err=" << worst_identity
            << ",residual_err=" << worst_residual << '\n';
}

// The mask of a real grammar: after "{" the schema admits only its two
// values. The expected mask is written by hand from the schema.
void CheckConstraintMask() {
  const std::vector<std::string> pieces{"{", "\"v\":1", "\"v\":2", "}", "x"};
  auto constraint = std::make_shared<gufo::sampling::TokenConstraint>();
  constraint->grammar = gufo::sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{"v":{"type":"integer","enum":[1,2]}},
              "required":["v"],"additionalProperties":false})"),
      false);
  constraint->vocabulary =
      std::make_shared<gufo::sampling::ConstraintVocabulary>(
          pieces.size(), [&](std::uint32_t i) {
            return gufo::sampling::ConstraintVocabulary::Piece{pieces[i],
                                                               false};
          });
  Case test = sampled_cases::Named("mask");
  gufo::sampling::SamplingConfig config;
  config.temperature = test.filters.temperature;
  config.constraint = constraint;
  config.seed = 0;
  gufo::sampling::SamplerState sampler(config);
  sampler.Accept(0);
  Expect(sampler.NeedsConstraintMask(), "mask: the grammar restricts tokens");
  const auto produced = sampler.Distribution(test.logits);
  const auto target = sampled_oracle::TargetDistribution(
      test.logits, test.filters, {}, test.allowed);
  Expect(produced.entries().size() == target.size(), "mask: support");
  for (const Candidate& candidate : target)
    Expect(std::abs(produced.probability(candidate.token) -
                    candidate.probability) <= RelativeTolerance(5),
           "mask: p of an allowed token");
  for (const DraftRow& draft : test.drafts) {
    const auto expected = sampled_oracle::Verify(target, draft, draft.ids[0]);
    Expect(expected.identity_error <= 0x1.0p-50, "mask: identity");
    // The draft row is unconstrained: its forbidden tokens have p = 0, are
    // never accepted, and the residual stays inside the mask.
    Expect(sampled_oracle::ProbabilityOf(target, 0) == 0.0 &&
               sampled_oracle::ProbabilityOf(target, 4) == 0.0,
           "mask: forbidden tokens keep no target mass");
    const auto runs = sampled_oracle::RecoverRuns([&](double uniform) {
      std::uint64_t state = sampled_oracle::StateDrawing(uniform);
      return static_cast<std::int64_t>(
          produced.SampleResidual(draft.ids, draft.probabilities, &state));
    });
    Expect(sampled_oracle::MeasureError(sampled_oracle::MeasureRuns(runs),
                                        expected.residual) <=
               ResidualTolerance(target, draft, RelativeTolerance(5),
                                 expected.residual_mass),
           "mask: residual");
  }
  std::cout << "mask,support=" << target.size() << '\n';
}

// The draft row of the MTP head and the draw from it.
void CheckDraftPolicy() {
  for (const auto& head : sampled_cases::HeadRows()) {
    double worst_drawn = 0.0;
    for (const float temperature : {1.0F, 0.7F, 0.05F, 3.0F}) {
      const auto expected = sampled_oracle::HeadTopK(
          head.scored, temperature, gufo::speculative::kMtpSampledTopK);
      std::vector<gufo::tokenization::TokenId> ids;
      std::vector<float> probabilities;
      gufo::speculative::MtpTopKDistribution(head.scored, temperature,
                                             gufo::speculative::kMtpSampledTopK,
                                             &ids, &probabilities);
      const std::string label = head.name + ",T=" + std::to_string(temperature);
      Expect(ids.size() == expected.indices.size(), label + ": row width");
      double sum = 0.0;
      for (std::size_t i = 0; i < ids.size(); ++i) {
        Expect(ids[i] == expected.indices[i], label + ": candidate order");
        // Both sides round one double quotient to float; their doubles differ
        // by the rounding of a 32-term sum, far below half a float ulp, so
        // the floats differ by at most one ulp.
        const float reference = static_cast<float>(expected.probabilities[i]);
        Expect(probabilities[i] == reference ||
                   probabilities[i] == std::nextafter(reference, 0.0F) ||
                   probabilities[i] == std::nextafter(reference, 1.0F),
               label + ": q differs from the softmax of the best logits");
        sum += probabilities[i];
      }
      // The verifier accepts a row within 1e-5 of one; 32 rounded floats are
      // within 32 * 2^-25.
      Expect(std::abs(sum - 1.0) <= 32 * 0x1.0p-25, label + ": q sum");

      // The draw: candidates own consecutive intervals of their reported
      // probabilities over [0, 1). A rounded row does not sum to one exactly:
      // the last positive candidate also takes what the row is short of one,
      // or the trailing candidates lose what it exceeds one by. The
      // distribution drawn therefore differs from the reported one by
      // |sum - 1| in total, and by nothing else.
      const auto runs = sampled_oracle::RecoverRuns([&](double uniform) {
        return static_cast<std::int64_t>(
            gufo::speculative::MtpSampleRow(ids, probabilities, uniform));
      });
      const auto measure = sampled_oracle::MeasureRuns(runs);
      Expect(measure.contiguous,
             label + ": MtpSampleRow is not an inverse CDF");
      double drawn_error = 0.0;
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto found = measure.length.find(ids[i]);
        drawn_error +=
            std::abs((found == measure.length.end() ? 0.0 : found->second) -
                     static_cast<double>(probabilities[i]));
      }
      worst_drawn = std::max(worst_drawn, drawn_error);
      Expect(drawn_error <= std::abs(sum - 1.0) + (64 * 0x1.0p-52),
             label + ": the row drawn differs from the row reported by " +
                 std::to_string(drawn_error));
      for (const double uniform : {0.0, sampled_oracle::kLastUniform})
        Expect(probabilities[static_cast<std::size_t>(
                   std::ranges::find(ids, gufo::speculative::MtpSampleRow(
                                              ids, probabilities, uniform)) -
                   ids.begin())] > 0.0F,
               label + ": a zero-probability candidate was drawn");
    }
    std::cout << head.name << ",scored=" << head.scored.size()
              << ",drawn_minus_reported=" << worst_drawn << '\n';
  }
}

// The generator the reference inverts is the one production draws from.
void CheckRandomNumbers() {
  for (std::uint64_t seed :
       {UINT64_C(1), UINT64_C(73), UINT64_C(0x9E3779B97F4A7C15),
        UINT64_C(0xFFFFFFFFFFFFFFFF), UINT64_C(0x0123456789ABCDEF)}) {
    std::uint64_t state = seed;
    const double uniform = gufo::sampling::Uniform(&state);
    Expect(state == sampled_oracle::NextState(seed) &&
               uniform == sampled_oracle::UniformOfState(state),
           "the reference generator differs from the sampler's");
    Expect(sampled_oracle::PreviousState(state) == seed,
           "the generator step is not inverted");
  }
  for (const double wanted :
       {0.0, 0x1.0p-53, 0.1, 0.5, 0.75, sampled_oracle::kLastUniform}) {
    for (const std::uint64_t spare : {UINT64_C(1), UINT64_C(0x7FF)}) {
      std::uint64_t state = sampled_oracle::StateDrawing(wanted, spare);
      Expect(gufo::sampling::Uniform(&state) ==
                 std::floor(wanted * 0x1.0p53) * 0x1.0p-53,
             "a chosen uniform is not drawn");
    }
  }
}

// Malformed sparse rows are refused before anything is drawn.
void CheckRowValidation() {
  const Case test = sampled_cases::Named("outside-support");
  const auto produced = MakeSampler(test).Distribution(test.logits);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<Token> ids{3, 5};
  for (const std::vector<float>& bad :
       {std::vector<float>{nan, 0.5F}, std::vector<float>{inf, 0.0F},
        std::vector<float>{-0.25F, 1.25F}, std::vector<float>{1.0F}}) {
    std::uint64_t state = 99;
    bool refused = false;
    try {
      (void)produced.SampleResidual(ids, bad, &state);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Expect(refused && state == 99,
           "a malformed sparse row must be refused without a draw");
  }
}

}  // namespace

int main() {
  try {
    CheckRandomNumbers();
    for (const Case& test : sampled_cases::All())
      CheckCase(test);
    CheckConstraintMask();
    CheckDraftPolicy();
    CheckRowValidation();
  } catch (const std::exception& failure) {
    std::cerr << "sampled_speculation_oracle_test: " << failure.what() << '\n';
    return 1;
  }
  std::cout << (g_passed ? "PASS" : "FAIL") << '\n';
  return g_passed ? 0 : 1;
}
