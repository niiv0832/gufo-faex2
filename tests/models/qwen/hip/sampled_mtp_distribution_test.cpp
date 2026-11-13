// Statistical check of the distribution sampled MTP speculation serves, on
// real weights. At a fixed context the first token a propose-verify step
// emits must be distributed as the target distribution p of ordinary
// decoding under the same request.
//
// The design below was fixed before the first run and is not tuned to
// results.
//   Contexts   4: Russian prose, English prose, Python (windows of the audit
//              corpus) and a Chinese paragraph, which drafts from the whole
//              vocabulary. Position: the first one, from token 64 (or half of
//              a shorter text) on, whose target distribution under the
//              request has at least three tokens and none above 0.7.
//   Request    the deployment profile: temperature 1, top_k 20, top_p 0.95,
//              min_p 0.05.
//   Samples    N = 10000 production steps (propose, verify, accept or
//              resample) and N = 10000 ordinary samples per context. The
//              random state of sample i is a fixed hash of (context, arm, i).
//   Checks     four per context, each at significance 0.001:
//              1. goodness of fit of the MTP tokens to p;
//              2. goodness of fit of the ordinary tokens to p (a control of
//                 the reference and of the harness);
//              3. homogeneity of the MTP and ordinary tokens;
//              4. the number of accepted drafts against Binomial(N, a), a
//                 being the acceptance probability the reference derives from
//                 the reported draft row and the probabilities with which its
//                 candidates are actually proposed.
//              1-3 are Pearson chi-square tests; tokens whose expected count
//              N p is below 5 share one cell. 4 is an exact two-sided
//              binomial test.
//   Family     16 checks: under the null hypothesis at least one rejects
//              with probability at most 16 * 0.001 = 0.016.
//   Verdict    any rejection fails the run and calls for an investigation;
//              the criteria are not to be changed. Without a rejection the
//              statement is "no difference detected at this power", with the
//              smallest difference each check detects with probability 0.9.
// Not statistical, and exact: no emitted token lies outside the support of p.
//
// qwen_sampled_mtp_distribution_test [MODEL.gguf [CONTEXT [N]]]
// CONTEXT is ru, en, code, zh or all (the default). Another N is for trying
// the tool out; only N = 10000 is the registered check. Without a model
// argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the test is
// skipped without it or without an embedded MTP head. Exit status: 0 no
// rejection, 1 a rejection or a failed exact check, 2 an invalid run.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "tests/models/qwen/hip/support/sampled_mtp_harness.hpp"

namespace {

using gufo::test::Candidate;
using gufo::test::Token;

constexpr std::uint32_t kContext = 1024;
constexpr std::size_t kSamples = 10000;
constexpr double kAlpha = 0.001;
constexpr double kPower = 0.9;
constexpr double kMinimumExpected = 5.0;
constexpr std::size_t kReplayed = 50;

std::uint64_t Mix(std::uint64_t value) {
  value += UINT64_C(0x9E3779B97F4A7C15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94D049BB133111EB);
  return value ^ (value >> 31U);
}

std::uint64_t SampleState(std::size_t context, std::size_t arm,
                          std::size_t sample) {
  return Mix(UINT64_C(0x5A3D1E0000000000) +
             (static_cast<std::uint64_t>(context) << 40U) +
             (static_cast<std::uint64_t>(arm) << 36U) + sample);
}

// --- Chi-square and binomial tails
// ----------------------------------------------

// Regularized upper incomplete gamma function Q(a, x).
double UpperGamma(double a, double x) {
  if (x <= 0.0)
    return 1.0;
  if (x < a + 1.0) {
    double term = 1.0 / a;
    double sum = term;
    for (int n = 1; n < 100000 && std::abs(term) > std::abs(sum) * 1e-17; ++n) {
      term *= x / (a + n);
      sum += term;
    }
    return 1.0 - (sum * std::exp((-x) + (a * std::log(x)) - std::lgamma(a)));
  }
  // Lentz's continued fraction.
  const double tiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / tiny;
  double d = 1.0 / b;
  double h = d;
  for (int n = 1; n < 100000; ++n) {
    const double an = -n * (n - a);
    b += 2.0;
    d = (an * d) + b;
    if (std::abs(d) < tiny)
      d = tiny;
    c = b + (an / c);
    if (std::abs(c) < tiny)
      c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < 1e-16)
      break;
  }
  return h * std::exp((-x) + (a * std::log(x)) - std::lgamma(a));
}

double ChiSquareTail(double statistic, double df) {
  return UpperGamma(df / 2.0, statistic / 2.0);
}

double ChiSquareCritical(double df, double alpha) {
  double low = 0.0;
  double high = 1000.0;
  for (int step = 0; step < 200; ++step) {
    const double middle = (low + high) / 2.0;
    (ChiSquareTail(middle, df) > alpha ? low : high) = middle;
  }
  return high;
}

// P(X > x) for a noncentral chi-square: a Poisson mixture of central ones.
double NoncentralTail(double x, double df, double noncentrality) {
  const double mean = noncentrality / 2.0;
  double tail = 0.0;
  for (int j = 0; j < 2000; ++j) {
    const double weight =
        std::exp((-mean) + (j * std::log(std::max(mean, 1e-300))) -
                 std::lgamma(j + 1.0));
    tail += weight * ChiSquareTail(x, df + (2.0 * j));
    if (j > mean && weight < 1e-18)
      break;
  }
  return tail;
}

// The noncentrality a chi-square test of `df` degrees detects with
// probability kPower at significance kAlpha.
double DetectableNoncentrality(double df) {
  const double critical = ChiSquareCritical(df, kAlpha);
  double low = 0.0;
  double high = 500.0;
  for (int step = 0; step < 100; ++step) {
    const double middle = (low + high) / 2.0;
    (NoncentralTail(critical, df, middle) < kPower ? low : high) = middle;
  }
  return high;
}

double BinomialLogMass(std::size_t n, std::size_t k, double p) {
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) -
         std::lgamma(static_cast<double>(n - k) + 1.0) +
         (static_cast<double>(k) * std::log(p)) +
         (static_cast<double>(n - k) * std::log1p(-p));
}

// Exact two-sided binomial test: the total mass of the outcomes that are no
// more likely than the observed one.
double BinomialTwoSided(std::size_t n, std::size_t observed, double p) {
  if (p <= 0.0)
    return observed == 0 ? 1.0 : 0.0;
  if (p >= 1.0)
    return observed == n ? 1.0 : 0.0;
  const double reference = BinomialLogMass(n, observed, p);
  double total = 0.0;
  for (std::size_t k = 0; k <= n; ++k) {
    const double mass = BinomialLogMass(n, k, p);
    if (mass <= reference + 1e-9)
      total += std::exp(mass);
  }
  return std::min(1.0, total);
}

// --- Cells
// ----------------------------------------------------------------------

// Tokens expected at least kMinimumExpected times have a cell each; the rest
// share the last cell, which joins the smallest other cell while it is
// expected fewer times itself.
struct Cells {
  std::map<Token, std::size_t> index;
  std::vector<double> probability;
  std::size_t pooled{0};  // the cell of every unlisted token

  [[nodiscard]] std::size_t Of(Token token) const {
    const auto found = index.find(token);
    return found == index.end() ? pooled : found->second;
  }
};

Cells MakeCells(std::span<const Candidate> target, std::size_t samples) {
  Cells cells;
  double rest = 0.0;
  for (const Candidate& candidate : target) {
    if (candidate.probability * static_cast<double>(samples) >=
        kMinimumExpected) {
      cells.index[candidate.token] = cells.probability.size();
      cells.probability.push_back(candidate.probability);
    } else {
      rest += candidate.probability;
    }
  }
  if (cells.probability.empty())
    throw std::runtime_error("no token is expected often enough");
  if (rest * static_cast<double>(samples) >= kMinimumExpected) {
    cells.pooled = cells.probability.size();
    cells.probability.push_back(rest);
  } else {
    cells.pooled =
        static_cast<std::size_t>(std::ranges::min_element(cells.probability) -
                                 cells.probability.begin());
    cells.probability[cells.pooled] += rest;
  }
  return cells;
}

struct Test {
  double statistic{0.0};
  double df{0.0};
  double p_value{1.0};
};

Test GoodnessOfFit(const Cells& cells, std::span<const std::size_t> counts,
                   std::size_t samples) {
  Test test;
  for (std::size_t cell = 0; cell < cells.probability.size(); ++cell) {
    const double expected =
        cells.probability[cell] * static_cast<double>(samples);
    const double difference = static_cast<double>(counts[cell]) - expected;
    test.statistic += difference * difference / expected;
  }
  test.df = static_cast<double>(cells.probability.size() - 1);
  test.p_value = ChiSquareTail(test.statistic, test.df);
  return test;
}

Test Homogeneity(std::span<const std::size_t> first,
                 std::span<const std::size_t> second) {
  Test test;
  double total_first = 0.0;
  double total_second = 0.0;
  for (std::size_t cell = 0; cell < first.size(); ++cell) {
    total_first += static_cast<double>(first[cell]);
    total_second += static_cast<double>(second[cell]);
  }
  std::size_t used = 0;
  for (std::size_t cell = 0; cell < first.size(); ++cell) {
    const double row = static_cast<double>(first[cell] + second[cell]);
    if (row == 0.0)
      continue;
    ++used;
    const double expected_first =
        row * total_first / (total_first + total_second);
    const double expected_second = row - expected_first;
    const double d1 = static_cast<double>(first[cell]) - expected_first;
    const double d2 = static_cast<double>(second[cell]) - expected_second;
    test.statistic += (d1 * d1 / expected_first) + (d2 * d2 / expected_second);
  }
  test.df = static_cast<double>(used - 1);
  test.p_value = ChiSquareTail(test.statistic, test.df);
  return test;
}

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// --- One context
// ------------------------------------------------------------------

struct Verdict {
  bool invalid{false};
  bool exact_failure{false};
  std::size_t rejections{0};
};

Verdict RunContext(const gufo::test::Stand& stand, std::size_t context,
                   std::size_t samples) {
  Verdict verdict;
  const gufo::test::Text& text = stand.texts[context];
  const auto filters = sampled_cases::kProfile;
  const auto& tokenizer = stand.target->GetTokenizer();

  gufo::test::Reference reference(stand, kContext);
  const std::size_t position = gufo::test::ChoosePosition(
      reference, text, std::min<std::size_t>(64, text.ids.size() / 2), 4);
  const std::vector<Token> prompt(
      text.ids.begin(),
      text.ids.begin() + static_cast<std::ptrdiff_t>(position));
  const Token current = text.ids[position];
  auto sequence = prompt;
  sequence.push_back(current);
  const std::vector<Token> published{current};

  reference.Start(prompt);
  const auto target_logits = reference.Decode(current);
  const auto target =
      gufo::test::TargetOf(target_logits, filters, prompt, published);
  // The kept set must not hinge on single-precision rounding.
  for (const float top_p : {0.95F * (1.0F - 1e-5F), 0.95F * (1.0F + 1e-5F)}) {
    for (const float min_p : {0.05F * (1.0F - 4e-6F), 0.05F * (1.0F + 4e-6F)}) {
      auto moved = filters;
      moved.top_p = top_p;
      moved.min_p = min_p;
      if (gufo::test::TargetOf(target_logits, moved, prompt, published)
              .size() != target.size()) {
        std::cout << text.name << ",INVALID: the kept set hinges on rounding\n";
        verdict.invalid = true;
        return verdict;
      }
    }
  }
  const Cells cells = MakeCells(target, samples);

  // The production session, as the server configures MTP.
  std::string error;
  auto executor =
      gufo::hip::QwenGpuExecutor::Create(stand.target, &error, kContext);
  if (!executor)
    throw std::runtime_error("executor: " + error);
  auto owned_backend = gufo::hip::QwenMtpGpuDraftBackend::Create(
      stand.mtp,
      {.max_context = kContext,
       .max_draft_tokens = 1,
       .vision_input = &executor->VisionInput()},
      &error);
  if (!owned_backend)
    throw std::runtime_error("MTP backend: " + error);
  auto* backend = owned_backend.get();
  gufo::speculative::SpeculativeOptions options;
  options.max_draft_tokens = 1;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 1;
  options.use_batched_verification = true;
  options.retain_frontier_logits = true;
  options.enable_adaptive_draft_length = false;
  gufo::speculative::SpeculativeVerifier verifier(
      *executor, std::move(owned_backend), options);
  const auto base = gufo::test::SamplerOf(filters, prompt, published);
  const auto position32 = static_cast<std::uint32_t>(position);
  const Token stop = tokenizer.GetEosTokenId();

  // The reported draft row of this context and the acceptance probability
  // the reference derives from it.
  (void)verifier.Prime(prompt);
  gufo::test::DraftRow draft;
  {
    std::uint64_t rng = 1;
    const auto proposal = backend->ProposeSampled(sequence, position32, 1,
                                                  filters.temperature, &rng);
    draft = {{proposal.candidate_ids.begin(), proposal.candidate_ids.end()},
             proposal.candidate_probabilities};
  }
  const auto law = sampled_oracle::EmittedLaw(target, draft);

  struct Step {
    Token first{0};
    bool accepted{false};
    std::uint64_t state{0};
  };
  const auto step = [&](std::size_t sample) {
    auto sampler = base;
    sampler.SetRngState(SampleState(context, 0, sample));
    const auto result =
        verifier.VerifyStep(sequence, position32, current, stop, 16, sampler);
    if (result.draft_count != 1 || result.emitted_tokens.empty())
      throw std::runtime_error("a step ran without a draft");
    // An accepted stop token ends the step after the draft draw and the
    // decision; it is reported as the step's only token, not as an accepted
    // draft.
    const bool accepted_stop =
        result.accepted_count == 0 && result.hit_eos &&
        sampler.rng_state() ==
            sampled_oracle::NextState(
                sampled_oracle::NextState(SampleState(context, 0, sample)));
    // Without an accepted draft the frontier is the row after the current
    // token: it must stay the row of ordinary decoding.
    if (result.accepted_count == 0 && sample % 500 == 0 &&
        !SameBits(result.next_token_logits, target_logits))
      throw std::runtime_error("the verification row left ordinary decoding");
    return Step{result.emitted_tokens.front(),
                result.accepted_count == 1 || accepted_stop,
                sampler.rng_state()};
  };

  // Each of the first steps from a freshly prefilled session.
  std::vector<Step> fresh;
  for (std::size_t sample = 0; sample < std::min(kReplayed, samples);
       ++sample) {
    (void)verifier.Prime(prompt);
    fresh.push_back(step(sample));
  }
  // All steps from one prefill: the draft and the target return to the
  // frontier after each step, as a cancelled step does.
  (void)verifier.Prime(prompt);
  const auto frontier = verifier.Snapshot();
  std::vector<std::size_t> mtp(cells.probability.size(), 0);
  std::size_t accepted = 0;
  std::size_t outside = 0;
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const Step taken = step(sample);
    if (sample < fresh.size() && (taken.first != fresh[sample].first ||
                                  taken.accepted != fresh[sample].accepted ||
                                  taken.state != fresh[sample].state)) {
      std::cout << text.name << ",INVALID: step " << sample
                << " differs between a fresh and a rewound session\n";
      verdict.invalid = true;
      return verdict;
    }
    ++mtp[cells.Of(taken.first)];
    accepted += taken.accepted;
    outside += sampled_oracle::ProbabilityOf(target, taken.first) == 0.0;
    verifier.RestoreSnapshot(*frontier);
    executor->RestoreState();
  }

  // Ordinary sampling of the same position. The forward pass is
  // deterministic; every sample runs the device sampler on its row.
  std::vector<std::size_t> ordinary(cells.probability.size(), 0);
  {
    auto plain =
        gufo::hip::QwenGpuExecutor::Create(stand.target, &error, kContext);
    if (!plain)
      throw std::runtime_error("executor: " + error);
    (void)plain->ForwardPromptBatch(prompt, 0, true);
    (void)plain->ForwardToken(current, position32, true);
    if (!SameBits(plain->CopyLastLogits(), target_logits)) {
      std::cout << text.name << ",INVALID: decoding is not repeatable\n";
      verdict.invalid = true;
      return verdict;
    }
    for (std::size_t sample = 0; sample < samples; ++sample) {
      auto sampler = base;
      sampler.SetRngState(SampleState(context, 1, sample));
      const Token token = plain->SampleLastLogits(sampler);
      ++ordinary[cells.Of(token)];
      outside += sampled_oracle::ProbabilityOf(target, token) == 0.0;
    }
  }

  const Test fit_mtp = GoodnessOfFit(cells, mtp, samples);
  const Test fit_ordinary = GoodnessOfFit(cells, ordinary, samples);
  const Test homogeneity = Homogeneity(mtp, ordinary);
  const double acceptance_p =
      BinomialTwoSided(samples, accepted, law.acceptance_rate);

  // The smallest differences the checks detect with probability kPower.
  // Goodness of fit: Cohen's w = sqrt(sum (p' - p)^2 / p) with noncentrality
  // N w^2; two samples of N each: N w^2 / 2. As a shift of probability d
  // between the two likeliest cells, w^2 = d^2 (1/p1 + 1/p2).
  const double df = static_cast<double>(cells.probability.size() - 1);
  const double detectable = DetectableNoncentrality(df);
  auto sorted = cells.probability;
  std::ranges::sort(sorted, std::greater<>());
  const double per_shift = std::sqrt((1.0 / sorted[0]) + (1.0 / sorted[1]));
  const double n = static_cast<double>(samples);
  const double fit_w = std::sqrt(detectable / n);
  const double two_w = std::sqrt(2.0 * detectable / n);
  // Acceptance: (z_{1-alpha/2} + z_{power}) standard errors.
  const double acceptance_shift =
      (3.2905 + 1.2816) *
      std::sqrt(law.acceptance_rate * (1.0 - law.acceptance_rate) / n);

  std::cout << text.name << ",position=" << position << ",current=" << current
            << ",samples=" << samples << ",support=" << target.size()
            << ",cells=" << cells.probability.size()
            << ",p_max=" << target.front().probability
            << ",reference_law_err=" << law.total_error
            << ",reference_law_bound=" << law.bound << '\n';
  std::cout << text.name << ",cell_probability";
  for (const double value : cells.probability)
    std::cout << ' ' << value;
  std::cout << '\n' << text.name << ",mtp_counts";
  for (const std::size_t count : mtp)
    std::cout << ' ' << count;
  std::cout << '\n' << text.name << ",ordinary_counts";
  for (const std::size_t count : ordinary)
    std::cout << ' ' << count;
  std::cout << '\n';
  const auto report = [&](const char* name, const Test& test, double w) {
    const bool rejected = test.p_value < kAlpha;
    verdict.rejections += rejected;
    std::cout << text.name << ',' << name << ",chi2=" << test.statistic
              << ",df=" << test.df << ",p_value=" << test.p_value
              << ",detectable_w=" << w
              << ",detectable_shift_between_top_cells=" << w / per_shift
              << (rejected ? ",REJECTED" : ",not rejected") << '\n';
  };
  report("mtp_vs_p", fit_mtp, fit_w);
  report("ordinary_vs_p", fit_ordinary, fit_w);
  report("mtp_vs_ordinary", homogeneity, two_w);
  const bool acceptance_rejected = acceptance_p < kAlpha;
  verdict.rejections += acceptance_rejected;
  std::cout << text.name << ",acceptance,accepted=" << accepted
            << ",expected_rate=" << law.acceptance_rate
            << ",observed_rate=" << static_cast<double>(accepted) / n
            << ",p_value=" << acceptance_p
            << ",detectable_shift=" << acceptance_shift
            << (acceptance_rejected ? ",REJECTED" : ",not rejected") << '\n';
  std::cout << text.name << ",tokens_outside_support=" << outside
            << (outside == 0 ? ",PASS" : ",FAIL") << '\n';
  verdict.exact_failure = outside != 0;
  return verdict;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 4) {
    std::cerr << "Usage: qwen_sampled_mtp_distribution_test [MODEL.gguf "
                 "[ru|en|code|zh|all [N]]]\n";
    return 2;
  }
  try {
    const auto stand = gufo::test::LoadStand(argc > 1 ? argv[1] : "");
    if (!stand)
      return 77;
    std::cout << std::defaultfloat << std::setprecision(6);
    const std::string wanted = argc > 2 ? argv[2] : "all";
    const std::size_t samples =
        argc > 3 ? static_cast<std::size_t>(std::stoul(argv[3])) : kSamples;
    if (samples != kSamples)
      std::cout << "NOTE: N=" << samples
                << " is not the registered N=" << kSamples
                << "; this run is not the acceptance check\n";
    std::size_t ran = 0;
    std::size_t rejections = 0;
    bool invalid = false;
    bool exact_failure = false;
    for (std::size_t context = 0; context < stand->texts.size(); ++context) {
      if (wanted != "all" && wanted != stand->texts[context].name)
        continue;
      ++ran;
      const Verdict verdict = RunContext(*stand, context, samples);
      rejections += verdict.rejections;
      invalid = invalid || verdict.invalid;
      exact_failure = exact_failure || verdict.exact_failure;
    }
    if (ran == 0) {
      std::cerr << "unknown context " << wanted << '\n';
      return 2;
    }
    std::cout << "contexts=" << ran << ",checks=" << 4 * ran
              << ",alpha_per_check=" << kAlpha
              << ",family_bound=" << 16 * kAlpha << ",rejections=" << rejections
              << ',';
    if (invalid) {
      std::cout << "INVALID RUN\n";
      return 2;
    }
    if (rejections != 0 || exact_failure) {
      std::cout << "REJECTED: investigate; do not change the criteria\n";
      return 1;
    }
    std::cout << "no difference detected at this power\n";
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_sampled_mtp_distribution_test: " << failure.what()
              << '\n';
    return 2;
  }
#else
  (void)argc;
  (void)argv;
  return 77;
#endif
}
