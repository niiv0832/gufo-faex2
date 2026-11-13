// Sampled speculative verification against the independent reference of
// tests/core/sampled_speculation_oracle.hpp, on synthetic logits.
//
// Device kernels (LaunchGPUSpeculativeSampling, every route: one block,
// segmented, ordered by a sort and ordered by block selection):
//   - the draws for which a draft token is accepted are [0, min(1, p/q));
//   - after a rejection the draws [0, 1) map onto the residual
//     max(0, p - q) / Z, one interval per token;
//   - the ordinary sampler (LaunchGPUSampling) maps them onto p.
// The kernels compute in single precision. Their results are compared with
// the double-precision reference as contributions to the distribution of the
// emitted token, within the bound derived at FloatBound.
//
// Verifier (SpeculativeVerifier with a host target, the path of executors
// without device sampling): the decision at a chosen draw, the emitted
// tokens, the committed target state, the random numbers consumed, and the
// refusal of malformed draft rows before anything advances.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/sampling.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "tests/core/sampled_speculation_cases.hpp"
#include "tests/core/sampled_speculation_oracle.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/sampled_float_bound.hpp"

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

void CheckHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

gufo::hip::GpuSamplingParameters Parameters(const Case& test) {
  return {.temperature = test.filters.temperature,
          .top_k = test.filters.top_k,
          .top_p = test.filters.top_p,
          .min_p = test.filters.min_p,
          .min_keep = test.filters.min_keep,
          .repeat_penalty = test.filters.repeat_penalty,
          .frequency_penalty = test.filters.frequency_penalty,
          .presence_penalty = test.filters.presence_penalty};
}

std::vector<gufo::sampling::TokenPenalty> DevicePenalties(const Case& test) {
  std::vector<gufo::sampling::TokenPenalty> penalties;
  for (const auto& penalty : test.Penalties())
    penalties.push_back({.token = penalty.token,
                         .generated_count = penalty.generated,
                         .repeated = penalty.repeated ? 1U : 0U});
  return penalties;
}

// One logit row on the device with the workspace the samplers need.
class DeviceRow {
public:
  explicit DeviceRow(const Case& test)
      : vocab_(test.logits.size()),
        parameters_(Parameters(test)),
        penalties_(DevicePenalties(test)) {
    CheckHip(hipMalloc(&logits_, vocab_ * sizeof(float)), "allocate logits");
    CheckHip(hipMalloc(&token_, sizeof(std::uint32_t)), "allocate token");
    CheckHip(hipMalloc(&accepted_, sizeof(std::uint32_t)), "allocate flag");
    CheckHip(hipMemcpy(logits_, test.logits.data(), vocab_ * sizeof(float),
                       hipMemcpyHostToDevice),
             "upload logits");
    gufo::hip::AllocateGpuSamplingWorkspace(&workspace_, vocab_, 64);
  }
  DeviceRow(const DeviceRow&) = delete;
  DeviceRow& operator=(const DeviceRow&) = delete;
  ~DeviceRow() {
    gufo::hip::FreeGpuSamplingWorkspace(&workspace_);
    (void)hipFree(accepted_);
    (void)hipFree(token_);
    (void)hipFree(logits_);
  }

  struct Outcome {
    std::uint32_t token{0};
    bool accepted{false};
  };

  Outcome Verify(const DraftRow& draft, std::size_t slot,
                 double acceptance_uniform, double residual_uniform) {
    gufo::hip::LaunchGPUSpeculativeSampling(
        logits_, token_, accepted_, vocab_, parameters_, draft.ids[slot],
        draft.probabilities[slot], draft.ids.data(), draft.probabilities.data(),
        draft.ids.size(), acceptance_uniform, residual_uniform,
        penalties_.data(), penalties_.size(), &workspace_);
    Outcome outcome;
    std::uint32_t accepted = 0;
    CheckHip(hipMemcpy(&outcome.token, token_, sizeof(outcome.token),
                       hipMemcpyDeviceToHost),
             "download token");
    CheckHip(hipMemcpy(&accepted, accepted_, sizeof(accepted),
                       hipMemcpyDeviceToHost),
             "download flag");
    if (accepted > 1 || outcome.token >= vocab_)
      throw std::runtime_error("the device sampler reported a failure");
    outcome.accepted = accepted == 1;
    ++launches_;
    return outcome;
  }

  std::uint32_t Sample(double uniform) {
    auto parameters = parameters_;
    parameters.uniform = uniform;
    gufo::hip::LaunchGPUSampling(logits_, token_, vocab_, parameters,
                                 penalties_.data(), penalties_.size(),
                                 &workspace_);
    std::uint32_t token = 0;
    CheckHip(hipMemcpy(&token, token_, sizeof(token), hipMemcpyDeviceToHost),
             "download token");
    ++launches_;
    return token;
  }

  [[nodiscard]] std::size_t Launches() const noexcept { return launches_; }

private:
  std::size_t vocab_;
  gufo::hip::GpuSamplingParameters parameters_;
  std::vector<gufo::sampling::TokenPenalty> penalties_;
  float* logits_{nullptr};
  std::uint32_t* token_{nullptr};
  std::uint32_t* accepted_{nullptr};
  gufo::hip::GpuSamplingWorkspace workspace_;
  std::size_t launches_{0};
};

gufo::test::FloatBound BoundOf(const Case& test,
                               std::span<const Candidate> target,
                               const DraftRow* draft, double residual_mass) {
  return gufo::test::BoundFloat(target, test.filters.temperature,
                                !test.Penalties().empty(), test.logits.size(),
                                draft, residual_mass);
}

// The kept set must not hinge on rounding: the reference keeps the same
// tokens when top-p and min-p move by more than single precision can.
bool FiltersAreStable(const Case& test, std::size_t support) {
  for (const float top_p_scale : {1.0F - 1e-5F, 1.0F + 1e-5F}) {
    for (const float min_p_scale : {1.0F - 4e-6F, 1.0F + 4e-6F}) {
      auto filters = test.filters;
      if (filters.top_p < 1.0F)
        filters.top_p =
            std::min(filters.top_p * top_p_scale, std::nextafter(1.0F, 0.0F));
      filters.min_p *= min_p_scale;
      if (sampled_oracle::TargetDistribution(test.logits, filters,
                                             test.Penalties())
              .size() != support)
        return false;
    }
  }
  return true;
}

void CheckDeviceCase(const Case& test) {
  const auto target = sampled_oracle::TargetDistribution(
      test.logits, test.filters, test.Penalties());
  if (!FiltersAreStable(test, target.size()))
    throw std::logic_error(test.name + ": the kept set hinges on rounding");
  DeviceRow row(test);

  // The ordinary sampler draws p.
  const auto plain = BoundOf(test, target, nullptr, 0.0);
  double worst_plain = 0.0;
  {
    const auto measure =
        sampled_oracle::MeasureRuns(sampled_oracle::RecoverRuns(
            [&](double uniform) {
              return static_cast<std::int64_t>(row.Sample(uniform));
            },
            30));
    worst_plain = sampled_oracle::MeasureError(measure, target);
    Expect(measure.contiguous,
           test.name + ": the sampler is not an inverse CDF");
    Expect(worst_plain <= plain.sampled,
           test.name + ": the sampler draws p with error " +
               std::to_string(worst_plain));
  }

  double worst_accepted = 0.0;
  double worst_residual = 0.0;
  double worst_conditional = 0.0;
  std::size_t residual_rows = 0;
  for (std::size_t index = 0; index < test.drafts.size(); ++index) {
    const DraftRow& draft = test.drafts[index];
    const std::string label = test.name + ",row=" + std::to_string(index);
    // The slots to draft: the ends of the row, its likeliest candidate, the
    // one accepted least often and one accepted only some of the time.
    std::vector<std::size_t> slots;
    std::size_t likeliest = test.DraftSlot(draft);
    std::size_t least = likeliest;
    std::size_t last = likeliest;
    std::size_t partial = likeliest;
    double least_acceptance = 2.0;
    for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
      if (!(draft.probabilities[slot] > 0.0F))
        continue;
      last = slot;
      if (draft.probabilities[slot] > draft.probabilities[likeliest])
        likeliest = slot;
      const double acceptance =
          sampled_oracle::Verify(target, draft, draft.ids[slot]).acceptance;
      if (acceptance < least_acceptance) {
        least_acceptance = acceptance;
        least = slot;
      }
      if (acceptance > 0.0 && acceptance < 1.0)
        partial = slot;
    }
    slots = {test.DraftSlot(draft), last, likeliest, least, partial};
    std::ranges::sort(slots);
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());

    const auto reference =
        sampled_oracle::Verify(target, draft, draft.ids[slots.front()]);
    const auto bound = BoundOf(test, target, &draft, reference.residual_mass);
    for (const std::size_t slot : slots) {
      const Token draft_token = draft.ids[slot];
      const double q = draft.probabilities[slot];
      const double p = sampled_oracle::ProbabilityOf(target, draft_token);
      const double expected = std::min(1.0, p / q);
      double low = 0.0;
      double high = 1.0;
      const auto accepts = [&](double uniform) {
        const auto outcome = row.Verify(draft, slot, uniform, 0.5);
        Expect(!outcome.accepted || outcome.token == draft_token,
               label + ": an accepted draft is not the emitted token");
        return outcome.accepted;
      };
      if (!accepts(0.0)) {
        high = 0.0;
      } else if (accepts(sampled_oracle::kLastUniform)) {
        low = 1.0;
      } else {
        high = sampled_oracle::kLastUniform;
        for (int step = 0; step < 44; ++step) {
          const double middle = low + ((high - low) / 2.0);
          (accepts(middle) ? low : high) = middle;
        }
      }
      // min(p, q) as accepted mass.
      const double accepted_error = std::abs(high - expected) * q;
      std::size_t kept = 0;
      while (kept < target.size() && target[kept].token != draft_token)
        ++kept;
      worst_accepted = std::max(worst_accepted, accepted_error);
      Expect(accepted_error <= bound.Accepted(kept, p, q),
             label + ",draft=" + std::to_string(draft_token) +
                 ": accepted up to " + std::to_string(high) + ", expected " +
                 std::to_string(expected));
    }

    // The residual, reached through the candidate rejected most often.
    if (!(least_acceptance < 1.0) ||
        row.Verify(draft, least, sampled_oracle::kLastUniform, 0.5).accepted)
      continue;
    ++residual_rows;
    const auto measure =
        sampled_oracle::MeasureRuns(sampled_oracle::RecoverRuns(
            [&](double uniform) {
              const auto outcome = row.Verify(
                  draft, least, sampled_oracle::kLastUniform, uniform);
              Expect(!outcome.accepted, label + ": the residual draw decided");
              return static_cast<std::int64_t>(outcome.token);
            },
            30));
    Expect(measure.contiguous,
           label + ": the residual sampler is not an inverse CDF");
    // An empty residual: production falls back to p (see the CPU test).
    const double conditional = sampled_oracle::MeasureError(
        measure, reference.residual.empty() ? target : reference.residual);
    const double emitted = conditional * reference.rejection_mass;
    worst_conditional = std::max(worst_conditional, conditional);
    worst_residual = std::max(worst_residual, emitted);
    if (reference.residual.empty())
      Expect(conditional <= plain.sampled,
             label +
                 ": without a residual the correction is not drawn "
                 "from p");
    Expect(emitted <= bound.emitted_residual,
           label + ": the residual moves the emitted distribution by " +
               std::to_string(emitted) + ", bound " +
               std::to_string(bound.emitted_residual));
    for (const auto& [token, length] : measure.length)
      Expect(length * reference.rejection_mass <= bound.emitted_residual ||
                 sampled_oracle::ProbabilityOf(target,
                                               static_cast<Token>(token)) > 0.0,
             label + ": the residual emits a token outside the target");
  }
  std::cout << test.name << ",vocab=" << test.logits.size()
            << ",support=" << target.size() << ",p_err=" << worst_plain
            << ",accepted_mass_err=" << worst_accepted
            << ",residual_rows=" << residual_rows
            << ",residual_err=" << worst_conditional
            << ",emitted_err=" << worst_residual
            << ",launches=" << row.Launches() << '\n';
}

// The same launch gives the same answer every time, on every route.
void CheckRepeatability() {
  for (const char* name : {"ties-unfiltered", "zeros", "outside-support",
                           "profile", "control-tokens"}) {
    const Case test = sampled_cases::Named(name);
    DeviceRow row(test);
    const DraftRow& draft = test.drafts.back();
    const std::size_t slot = test.DraftSlot(draft);
    std::size_t differing = 0;
    constexpr std::size_t kRepeats = 4000;
    for (const auto& [acceptance, residual] :
         {std::pair{0.3, 0.37},
          std::pair{sampled_oracle::kLastUniform, 0.61}}) {
      const auto first = row.Verify(draft, slot, acceptance, residual);
      for (std::size_t repeat = 1; repeat < kRepeats; ++repeat) {
        const auto next = row.Verify(draft, slot, acceptance, residual);
        differing +=
            next.accepted != first.accepted || next.token != first.token;
      }
    }
    Expect(differing == 0, std::string(name) + ": " +
                               std::to_string(differing) + " of " +
                               std::to_string(2 * (kRepeats - 1)) +
                               " repeated launches gave another answer");
    std::cout << name << ",vocab=" << test.logits.size()
              << ",repeats=" << 2 * kRepeats << ",differing=" << differing
              << '\n';
  }
}

// Requests that draw no random numbers never reach the speculative kernel;
// malformed launches are refused on the host.
void CheckLaunchValidation() {
  for (const char* name : {"greedy", "top-k-one"}) {
    const Case test = sampled_cases::Named(name);
    DeviceRow row(test);
    bool refused = false;
    try {
      (void)row.Verify(test.drafts[0], 0, 0.5, 0.5);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Expect(refused, std::string(name) +
                        ": a deterministic request reached "
                        "the speculative sampler");
  }
  const Case test = sampled_cases::Named("ties");
  DeviceRow row(test);
  for (const auto& [acceptance, residual] :
       {std::pair{1.0, 0.5}, std::pair{0.5, 1.0}, std::pair{-0.25, 0.5},
        std::pair{std::numeric_limits<double>::quiet_NaN(), 0.5}}) {
    bool refused = false;
    try {
      (void)row.Verify(test.drafts[0], 0, acceptance, residual);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Expect(refused, "a draw outside [0, 1) was launched");
  }
  DraftRow zero{{0, 1}, {0.0F, 1.0F}};
  bool refused = false;
  try {
    (void)row.Verify(zero, 0, 0.5, 0.5);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  Expect(refused, "a draft token without draft mass was launched");
}

// --- The verifier on a host target
// ---------------------------------------------

constexpr Token kCurrent = 3;
constexpr Token kPromptToken = 7;

// Verification row r holds logits `rows[r]`: the row after the current token
// is the target of the draft, the row after the draft that of the bonus.
class HostTarget final : public gufo::speculative::ISpeculativeTargetExecutor {
public:
  HostTarget(std::vector<std::vector<float>> rows, Token stop)
      : rows_(std::move(rows)), stop_(stop) {}

  void Reset() noexcept override { state_.clear(); }
  Token ForwardPromptBatch(std::span<const Token> prompt) override {
    state_.assign(prompt.begin(), prompt.end());
    prompt_size_ = state_.size();
    return kCurrent;
  }
  Token ForwardToken(Token token, std::uint32_t position, bool) override {
    if (position != state_.size())
      throw std::logic_error("the target was advanced out of order");
    state_.push_back(token);
    ++forwards_;
    return 0;
  }
  void SaveState(std::uint32_t valid_context) override {
    if (valid_context != state_.size())
      throw std::logic_error("the target was saved at another length");
    saved_ = state_;
    ++saves_;
  }
  void RestoreState() override {
    state_ = saved_;
    ++restores_;
  }
  std::span<const float> CopyLastLogits() override {
    return rows_[std::min(state_.size() - prompt_size_ - 1, rows_.size() - 1)];
  }
  std::size_t VocabularySize() const noexcept override {
    return rows_.front().size();
  }
  Token GetEosTokenId() const noexcept override { return stop_; }
  std::string_view DecodeToken(Token) const noexcept override { return ""; }

  std::vector<Token> state_;
  std::vector<Token> saved_;
  std::size_t forwards_{0};
  std::size_t saves_{0};
  std::size_t restores_{0};

private:
  std::vector<std::vector<float>> rows_;
  Token stop_;
  std::size_t prompt_size_{0};
};

class FixedDraft final : public gufo::speculative::IDraftBackend {
public:
  explicit FixedDraft(gufo::speculative::DraftProposal proposal)
      : proposal_(std::move(proposal)) {}
  std::string_view Name() const noexcept override { return "FixedDraft"; }
  gufo::speculative::DraftProposal Propose(std::span<const Token>,
                                           std::uint32_t,
                                           std::uint32_t) override {
    return proposal_;
  }
  gufo::speculative::DraftProposal ProposeSampled(std::span<const Token>,
                                                  std::uint32_t, std::uint32_t,
                                                  float,
                                                  std::uint64_t*) override {
    return proposal_;
  }
  bool SupportsSampledProposals() const noexcept override { return true; }

private:
  gufo::speculative::DraftProposal proposal_;
};

gufo::sampling::SamplerState HostSampler(const Case& test) {
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

gufo::speculative::SpeculativeOptions OneDraft() {
  gufo::speculative::SpeculativeOptions options;
  options.max_draft_tokens = 1;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 1;
  options.enable_adaptive_draft_length = false;
  return options;
}

void CheckVerifierCase(const Case& test, Token stop) {
  const auto target = sampled_oracle::TargetDistribution(
      test.logits, test.filters, test.Penalties());
  // The row after an accepted draft: the reversed logits, another target.
  std::vector<float> bonus_row(test.logits.rbegin(), test.logits.rend());
  std::size_t decisions = 0;
  for (std::size_t index = 0; index < test.drafts.size(); ++index) {
    const DraftRow& draft = test.drafts[index];
    for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
      if (!(draft.probabilities[slot] > 0.0F))
        continue;
      const Token draft_token = draft.ids[slot];
      const auto expected = sampled_oracle::Verify(target, draft, draft_token);
      const double threshold = expected.acceptance;
      // Draws on both sides of the threshold and next to it. Production and
      // the reference compute p in double precision: they agree to 1e-12.
      for (const double wanted :
           {0.0, threshold * 0.5, threshold * (1.0 - 1e-9),
            threshold * (1.0 + 1e-9), (threshold + 1.0) * 0.5,
            sampled_oracle::kLastUniform}) {
        if (!(wanted < 1.0))
          continue;
        // The draw the generator can produce at or below this value.
        const double uniform = std::floor(wanted * 0x1.0p53) * 0x1.0p-53;
        const bool accept = uniform < threshold;
        if (threshold > 0.0 && threshold < 1.0 &&
            std::abs(uniform - threshold) <= 1e-12 * threshold)
          continue;
        const std::string label = test.name + ",row=" + std::to_string(index) +
                                  ",draft=" + std::to_string(draft_token) +
                                  ",u=" + std::to_string(uniform);
        HostTarget host({test.logits, bonus_row}, stop);
        gufo::speculative::DraftProposal proposal;
        proposal.tokens = {draft_token};
        proposal.candidate_ids = draft.ids;
        proposal.candidate_probabilities = draft.probabilities;
        proposal.candidates_per_token = draft.ids.size();
        proposal.start_pos = 1;
        gufo::speculative::SpeculativeVerifier verifier(
            host, std::make_unique<FixedDraft>(proposal), OneDraft());
        const std::vector<Token> prompt{kPromptToken};
        (void)verifier.Prime(prompt);
        const std::vector<Token> sequence{kPromptToken, kCurrent};
        auto sampler = HostSampler(test);
        const std::vector<Token> history(sampler.history().begin(),
                                         sampler.history().end());
        const std::uint64_t first = sampled_oracle::StateDrawing(uniform);
        sampler.SetRngState(first);
        const auto result =
            verifier.VerifyStep(sequence, 1, kCurrent, stop, 2, sampler);
        ++decisions;
        const std::uint64_t after_decision = sampled_oracle::NextState(first);
        Expect(std::ranges::equal(sampler.history(), history),
               label + ": the request's sampler history changed");
        Expect(host.saves_ == 1, label + ": the target state was not saved");
        if (accept && draft_token == stop) {
          // An accepted stop token ends the step; nothing else is drawn.
          Expect(result.emitted_tokens == std::vector<Token>{stop} &&
                     result.hit_eos && sampler.rng_state() == after_decision &&
                     host.state_ == sequence,
                 label + ": an accepted stop token");
          continue;
        }
        if (accept) {
          auto after = HostSampler(test);
          after.Accept(draft_token);
          const auto bonus_distribution = after.Distribution(bonus_row);
          std::uint64_t state = after_decision;
          const Token bonus = bonus_distribution.Sample(&state);
          Expect(result.accepted_count == 1 &&
                     result.emitted_tokens ==
                         std::vector<Token>{draft_token, bonus},
                 label + ": an accepted draft and its bonus token");
          Expect(sampler.rng_state() == state,
                 label + ": random numbers consumed after acceptance");
          Expect(host.restores_ == 0 &&
                     host.state_ == std::vector<Token>{kPromptToken, kCurrent,
                                                       draft_token},
                 label + ": the accepted draft is not the committed history");
          continue;
        }
        const auto& residual =
            expected.residual.empty() ? target : expected.residual;
        std::uint64_t state = after_decision;
        const Token correction =
            HostSampler(test)
                .Distribution(test.logits)
                .SampleResidual(draft.ids, draft.probabilities, &state);
        Expect(result.accepted_count == 0 &&
                   result.emitted_tokens == std::vector<Token>{correction} &&
                   sampled_oracle::ProbabilityOf(residual, correction) > 0.0,
               label + ": a rejected draft is not replaced from the residual");
        Expect(sampler.rng_state() == state &&
                   state == (residual.size() > 1
                                 ? sampled_oracle::NextState(after_decision)
                                 : after_decision),
               label + ": random numbers consumed after rejection");
        Expect(host.restores_ == 1 && host.state_ == sequence,
               label + ": a rejected draft stayed in the target state");
        Expect(result.hit_eos == (correction == stop),
               label + ": stop token as the residual choice");
      }
    }
  }
  std::cout << test.name << ",verifier_decisions=" << decisions << '\n';
}

// Draft rows the verifier must refuse before the target, the draft feedback
// or the request's random numbers move.
void CheckVerifierValidation() {
  const Case test = sampled_cases::Named("outside-support");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  struct Malformed {
    const char* what;
    std::vector<Token> ids;
    std::vector<float> probabilities;
    Token draft;
  };
  const std::vector<Malformed> rows{
      {"repeated id", {0, 0, 1}, {0.25F, 0.25F, 0.5F}, 0},
      {"non-finite", {0, 1}, {nan, 0.5F}, 1},
      {"infinite", {0, 1}, {std::numeric_limits<float>::infinity(), 0.0F}, 0},
      {"negative", {0, 1}, {-0.25F, 1.25F}, 1},
      {"above one", {0, 1}, {1.5F, 0.0F}, 0},
      {"sum below one", {0, 1}, {0.5F, 0.4999F}, 0},
      {"sum above one", {0, 1}, {0.5F, 0.5001F}, 0},
      {"draft outside the row", {0, 1}, {0.5F, 0.5F}, 2},
      {"draft without mass", {0, 1}, {0.0F, 1.0F}, 0},
      {"id outside the vocabulary", {0, 16}, {0.5F, 0.5F}, 0},
  };
  for (const auto& row : rows) {
    HostTarget host({test.logits, test.logits}, 15);
    gufo::speculative::DraftProposal proposal;
    proposal.tokens = {row.draft};
    proposal.candidate_ids = row.ids;
    proposal.candidate_probabilities = row.probabilities;
    proposal.candidates_per_token = row.ids.size();
    proposal.start_pos = 1;
    gufo::speculative::SpeculativeVerifier verifier(
        host, std::make_unique<FixedDraft>(proposal), OneDraft());
    const std::vector<Token> prompt{kPromptToken};
    (void)verifier.Prime(prompt);
    const std::vector<Token> sequence{kPromptToken, kCurrent};
    auto sampler = HostSampler(test);
    sampler.SetRngState(4242);
    bool refused = false;
    try {
      (void)verifier.VerifyStep(sequence, 1, kCurrent, 15, 2, sampler);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    Expect(refused && host.saves_ == 0 && host.forwards_ == 0 &&
               host.state_ == prompt && sampler.rng_state() == 4242,
           std::string("malformed draft row (") + row.what +
               ") was not refused before the state advanced");
  }
  std::cout << "verifier_validation,rows=" << rows.size() << '\n';
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (const int gate =
          gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                    "qwen_sampled_verify_ops_test");
      gate != gufo::test::kHipTestSuccess)
    return gate;
  try {
    for (const Case& test : sampled_cases::All()) {
      if (test.Random())
        CheckVerifierCase(test, test.logits.size() == sampled_cases::kQwenVocab
                                    ? sampled_cases::kImEnd
                                    : static_cast<Token>(test.logits.size()));
    }
    CheckVerifierValidation();
    CheckLaunchValidation();
    CheckRepeatability();
    for (const Case& test : sampled_cases::All(384)) {
      if (!test.Random())
        continue;
      CheckDeviceCase(test);
      // Past 16384 tokens the unordered sampler runs in segments and top-k
      // selects its candidates by blocks.
      if (test.logits.size() < 16384)
        CheckDeviceCase(sampled_cases::Spread(test, 24576));
    }
  } catch (const std::exception& failure) {
    std::cerr << "qwen_sampled_verify_ops_test: " << failure.what() << '\n';
    return 1;
  }
  std::cout << (g_passed ? "PASS" : "FAIL") << '\n';
  return g_passed ? 0 : 1;
#else
  return 77;
#endif
}
