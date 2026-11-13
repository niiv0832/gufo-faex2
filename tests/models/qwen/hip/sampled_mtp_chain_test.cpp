// Sampled MTP speculation on real weights, as one chain: the row the draft
// backend reports -> its token ids -> the device verifier -> the history the
// target and the draft are left with. Every link is held to a prediction
// made without the code under test: ordinary decoding gives the target
// logits and hidden states, an MTP head run apart from the backend gives the
// head logits over the whole vocabulary, and the double-precision reference
// of tests/core/sampled_speculation_oracle.hpp turns them into the draft row,
// the acceptance probability, the residual and the emitted token.
//
// Per text and sampling request:
//   - the reported row is the softmax of the 32 best head logits among the
//     tokens the step may draft (the draft vocabulary, or every token after a
//     text in another script), with global ids, and the draft token is the
//     inverse-CDF draw from it;
//   - VerifySampledToken on the verification row accepts over [0, min(1,
//     p/q)) and maps the remaining draws onto the residual; its random
//     numbers and sampler history follow the decision;
//   - a whole verifier step from a chosen random state emits the predicted
//     tokens, on acceptance and on rejection, including an end of turn and a
//     tool call, and leaves the target and the draft with exactly the history
//     that ordinary decoding of the emitted tokens gives, bit for bit.
// Across sessions: proposals and verifier steps of two and four sessions in
// one batch, whichever session leads it, and a session moving from a lone
// step to a batch of four and back, equal the lone steps bit for bit.
//
// qwen_sampled_mtp_chain_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it or without an embedded MTP head.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "tests/models/qwen/hip/support/sampled_float_bound.hpp"
#include "tests/models/qwen/hip/support/sampled_mtp_harness.hpp"

namespace {

using gufo::test::Candidate;
using gufo::test::DraftRow;
using gufo::test::Filters;
using gufo::test::Reference;
using gufo::test::Request;
using gufo::test::Stand;
using gufo::test::Text;
using gufo::test::Token;

constexpr std::uint32_t kContext = 1024;

bool g_passed = true;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    g_passed = false;
    std::cout << "FAIL: " << message << '\n';
  }
}

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::uint64_t Mix(std::uint64_t value) {
  value += UINT64_C(0x9E3779B97F4A7C15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94D049BB133111EB);
  return value ^ (value >> 31U);
}

// One production session, configured as the server configures MTP.
struct Session {
  std::unique_ptr<gufo::hip::QwenGpuExecutor> executor;
  gufo::hip::QwenMtpGpuDraftBackend* backend{nullptr};
  std::unique_ptr<gufo::speculative::SpeculativeVerifier> verifier;
};

Session MakeSession(const Stand& stand) {
  Session session;
  std::string error;
  session.executor =
      gufo::hip::QwenGpuExecutor::Create(stand.target, &error, kContext);
  if (!session.executor)
    throw std::runtime_error("executor: " + error);
  auto backend = gufo::hip::QwenMtpGpuDraftBackend::Create(
      stand.mtp,
      {.max_context = kContext,
       .max_draft_tokens = 1,
       .vision_input = &session.executor->VisionInput()},
      &error);
  if (!backend)
    throw std::runtime_error("MTP backend: " + error);
  session.backend = backend.get();
  gufo::speculative::SpeculativeOptions options;
  options.max_draft_tokens = 1;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 1;
  options.use_batched_verification = true;
  options.retain_frontier_logits = true;
  options.enable_adaptive_draft_length = false;
  session.verifier = std::make_unique<gufo::speculative::SpeculativeVerifier>(
      *session.executor, std::move(backend), options);
  return session;
}

// A text cut at a position: the prompt, the current token, what follows.
struct Cut {
  std::string name;
  std::vector<Token> prompt;
  Token current{0};
  std::vector<Token> next;

  [[nodiscard]] std::vector<Token> Sequence() const {
    auto sequence = prompt;
    sequence.push_back(current);
    return sequence;
  }
  [[nodiscard]] std::uint32_t Position() const {
    return static_cast<std::uint32_t>(prompt.size());
  }
};

Cut CutAt(const Text& text, std::size_t position) {
  Cut cut;
  cut.name = text.name;
  cut.prompt.assign(text.ids.begin(),
                    text.ids.begin() + static_cast<std::ptrdiff_t>(position));
  cut.current = text.ids[position];
  cut.next.assign(text.ids.begin() + static_cast<std::ptrdiff_t>(position) + 1,
                  text.ids.end());
  return cut;
}

// What the reference predicts for a step of `cut` under `request`.
struct Prediction {
  std::vector<float> target_logits;  // after the current token
  std::vector<float> head_logits;    // whole vocabulary
  bool whole{false};
  DraftRow draft;
  std::vector<double> proposed;
  std::vector<Candidate> target;
};

struct World {
  const Stand& stand;
  Reference reference;
  std::unique_ptr<gufo::hip::QwenMtpGpuExecutor> head;
  std::span<const Token> draft_ids;
  std::size_t vocab;
};

Prediction Predict(World& world, const Cut& cut, const Filters& filters,
                   std::span<const std::uint8_t> allowed = {}) {
  Prediction prediction;
  world.reference.Start(cut.prompt);
  prediction.target_logits = world.reference.Decode(cut.current);
  const auto sequence = cut.Sequence();
  prediction.head_logits =
      gufo::test::HeadLogits(*world.head, sequence, world.reference, true);
  prediction.whole =
      world.draft_ids.empty() ||
      gufo::test::NeedsWholeVocabulary(sequence, world.draft_ids);
  prediction.draft =
      gufo::test::ExpectedDraft(prediction.head_logits, world.draft_ids,
                                prediction.whole, filters.temperature);
  prediction.proposed = sampled_oracle::ProposedProbabilities(prediction.draft);
  sampled_cases::Case history;
  history.filters = filters;
  history.prompt = cut.prompt;
  history.generated = {cut.current};
  prediction.target = sampled_oracle::TargetDistribution(
      prediction.target_logits, filters, history.Penalties(), allowed);
  return prediction;
}

// The slot an inverse-CDF draw of `uniform` proposes, with the distance to
// the nearest end of its interval.
struct DraftDraw {
  std::size_t slot{0};
  double margin{0.0};
};

DraftDraw DrawDraft(std::span<const double> proposed, double uniform) {
  double begin = 0.0;
  std::size_t last = 0;
  for (std::size_t slot = 0; slot < proposed.size(); ++slot) {
    if (!(proposed[slot] > 0.0))
      continue;
    last = slot;
    const double end = begin + proposed[slot];
    if (uniform < end)
      return {slot, std::min(uniform - begin, end - uniform)};
    begin = end;
  }
  return {last, 0.0};
}

// The proposal against the reference row.
void CheckProposal(const std::string& label,
                   const gufo::speculative::DraftProposal& proposal,
                   const Prediction& prediction, double uniform) {
  const DraftRow& expected = prediction.draft;
  Expect(proposal.tokens.size() == 1 &&
             proposal.candidates_per_token == expected.ids.size() &&
             proposal.candidate_ids.size() == expected.ids.size() &&
             proposal.candidate_probabilities.size() == expected.ids.size(),
         label + ": the proposal has another shape");
  if (proposal.candidate_ids.size() != expected.ids.size() ||
      proposal.tokens.empty())
    return;
  std::size_t wrong_ids = 0;
  std::size_t wrong_mass = 0;
  for (std::size_t slot = 0; slot < expected.ids.size(); ++slot) {
    wrong_ids += proposal.candidate_ids[slot] != expected.ids[slot];
    // One rounded double quotient on each side: at most one float ulp apart.
    const float reference = expected.probabilities[slot];
    const float value = proposal.candidate_probabilities[slot];
    wrong_mass +=
        !(value == reference || value == std::nextafter(reference, 0.0F) ||
          value == std::nextafter(reference, 1.0F));
  }
  Expect(wrong_ids == 0, label + ": " + std::to_string(wrong_ids) +
                             " candidate ids differ from the head's best "
                             "tokens");
  Expect(wrong_mass == 0, label + ": " + std::to_string(wrong_mass) +
                              " probabilities differ from the softmax of the "
                              "head logits");
  // The draw, from the row production reported.
  DraftRow reported{
      {proposal.candidate_ids.begin(), proposal.candidate_ids.end()},
      proposal.candidate_probabilities};
  const auto draw =
      DrawDraft(sampled_oracle::ProposedProbabilities(reported), uniform);
  Expect(draw.margin <= 1e-9 || proposal.tokens[0] == reported.ids[draw.slot],
         label + ": the draft token is not the draw from the reported row");
}

DraftRow RowOf(const gufo::speculative::DraftProposal& proposal) {
  return {{proposal.candidate_ids.begin(), proposal.candidate_ids.end()},
          proposal.candidate_probabilities};
}

// --- The device verifier on a real verification row
// ----------------------------

struct Constraint {
  std::shared_ptr<gufo::sampling::TokenConstraint> constraint;
  std::vector<std::uint8_t> allowed;
};

void Dissect(World& world, Session& session, const Cut& cut,
             const Request& request, const Constraint* constraint = nullptr) {
  const std::string label =
      cut.name + "," + request.name + (constraint != nullptr ? "+mask" : "");
  const auto prediction =
      Predict(world, cut, request.filters,
              constraint != nullptr ? std::span(constraint->allowed)
                                    : std::span<const std::uint8_t>{});
  const auto sequence = cut.Sequence();
  const std::uint32_t position = cut.Position();
  (void)session.verifier->Prime(cut.prompt);
  // Production hidden states of the prompt are those of ordinary decoding.
  Expect(SameBits(session.executor->GetPromptHiddenStates(),
                  std::span(world.reference.Hidden())
                      .first(cut.prompt.size() * world.head->GetHiddenSize())),
         label + ": prompt hidden states differ from ordinary decoding");

  auto config = gufo::test::ConfigOf(request.filters);
  if (constraint != nullptr)
    config.constraint = constraint->constraint;
  gufo::sampling::SamplerState base(config, cut.prompt);
  if (constraint == nullptr)
    base.Accept(cut.current);

  // The proposal.
  const std::uint64_t seed = Mix(0x5EED0000 + position);
  auto proposing = base;
  proposing.SetRngState(seed);
  const auto proposal = session.backend->ProposeSampled(
      sequence, position, 1, request.filters.temperature,
      proposing.mutable_rng_state());
  Expect(proposing.rng_state() == sampled_oracle::NextState(seed),
         label + ": the proposal did not consume exactly one draw");
  CheckProposal(
      label, proposal, prediction,
      sampled_oracle::UniformOfState(sampled_oracle::NextState(seed)));
  if (proposal.tokens.empty() || proposal.candidate_ids.empty())
    return;
  const DraftRow draft = RowOf(proposal);

  // The draft vocabulary gives each of its tokens the logit of the whole
  // head.
  if (!prediction.whole) {
    const auto shortlist =
        gufo::test::HeadLogits(*world.head, sequence, world.reference, false);
    std::size_t differing = shortlist.size() != world.draft_ids.size();
    for (std::size_t i = 0; i < shortlist.size() && differing == 0; ++i)
      differing += std::memcmp(&shortlist[i],
                               &prediction.head_logits[world.draft_ids[i]],
                               sizeof(float)) != 0;
    Expect(differing == 0,
           label +
               ": a draft-vocabulary logit is not the whole head's logit "
               "of the same token");
  }

  // The verification row.
  session.executor->SaveState(position);
  const std::vector<Token> inputs{cut.current, proposal.tokens[0]};
  (void)session.executor->ForwardVerificationChunk(inputs, position, false);
  {
    const auto row = session.executor->CopyVerificationLogits(0);
    Expect(SameBits(row, prediction.target_logits),
           label + ": the verification row differs from ordinary decoding");
  }
  const auto& target = prediction.target;
  const bool host = constraint != nullptr;
  const auto reference =
      sampled_oracle::Verify(target, draft, proposal.tokens[0]);
  const auto bound =
      gufo::test::BoundFloat(target, request.filters.temperature,
                             request.filters.repeat_penalty != 1.0F ||
                                 request.filters.presence_penalty != 0.0F ||
                                 request.filters.frequency_penalty != 0.0F,
                             world.vocab, &draft, reference.residual_mass);
  const auto law = sampled_oracle::EmittedLaw(target, draft);

  // Slots: the drafted one, the ends of the row, the likeliest, the least
  // accepted and one accepted some of the time.
  std::vector<std::size_t> slots;
  std::size_t least = 0;
  double least_acceptance = 2.0;
  std::size_t partial = 0;
  std::size_t last = 0;
  for (std::size_t slot = 0; slot < draft.ids.size(); ++slot) {
    if (!(draft.probabilities[slot] > 0.0F))
      continue;
    last = slot;
    if (draft.ids[slot] == proposal.tokens[0])
      slots.push_back(slot);
    const double acceptance =
        std::min(1.0, sampled_oracle::ProbabilityOf(target, draft.ids[slot]) /
                          static_cast<double>(draft.probabilities[slot]));
    if (acceptance < least_acceptance) {
      least_acceptance = acceptance;
      least = slot;
    }
    if (acceptance > 0.0 && acceptance < 1.0)
      partial = slot;
  }
  slots.insert(slots.end(), {std::size_t{0}, last, least, partial});
  std::ranges::sort(slots);
  slots.erase(std::unique(slots.begin(), slots.end()), slots.end());

  double worst_accepted = 0.0;
  for (const std::size_t slot : slots) {
    const Token token = draft.ids[slot];
    const double q = draft.probabilities[slot];
    if (!(q > 0.0))
      continue;
    const double p = sampled_oracle::ProbabilityOf(target, token);
    const auto accepts = [&](double uniform) {
      auto sampler = base;
      const std::uint64_t first = sampled_oracle::StateDrawing(uniform);
      sampler.SetRngState(first);
      const auto outcome = session.executor->VerifySampledToken(
          0, token, draft.ids, draft.probabilities, q, sampler);
      const std::uint64_t after = sampled_oracle::NextState(first);
      if (outcome.accepted) {
        // Acceptance draws once and commits the token to the sampler.
        Expect(outcome.token == token && sampler.rng_state() == after &&
                   !sampler.history().empty() &&
                   sampler.history().back() == token,
               label + ": an accepted draft left another sampler state");
      } else {
        // A rejection draws the residual as well (the host path skips the
        // draw when a single token remains) and commits nothing.
        const bool two =
            sampler.rng_state() == sampled_oracle::NextState(after);
        Expect((two || (host && sampler.rng_state() == after)) &&
                   std::ranges::equal(sampler.history(), base.history()),
               label + ": a rejected draft left another sampler state");
      }
      return outcome.accepted;
    };
    const double threshold = sampled_oracle::DecisionThreshold(accepts);
    const double error = std::abs(threshold - std::min(1.0, p / q)) * q;
    worst_accepted = std::max(worst_accepted, error);
    std::size_t kept = 0;
    while (kept < target.size() && target[kept].token != token)
      ++kept;
    Expect(error <= (host ? 1e-12 : bound.Accepted(kept, p, q)),
           label + ",draft=" + std::to_string(token) + ": accepted up to " +
               std::to_string(threshold) + ", expected " +
               std::to_string(std::min(1.0, p / q)));
  }

  // The residual, through the candidate rejected most often. A draw pair
  // (acceptance, residual) comes from one state: the 11 spare bits of the
  // second draw choose a first draw that rejects.
  const int depth = target.size() > 4096 ? 14 : 30;
  double conditional = -1.0;
  if (least_acceptance < 0.95) {
    const Token token = draft.ids[least];
    const double q = draft.probabilities[least];
    const auto measure =
        sampled_oracle::MeasureRuns(sampled_oracle::RecoverRuns(
            [&](double uniform) {
              for (std::uint64_t spare = 1; spare < 2048; ++spare) {
                const std::uint64_t second =
                    sampled_oracle::StateDrawing(uniform, spare);
                const std::uint64_t first =
                    sampled_oracle::PreviousState(second);
                if (sampled_oracle::UniformOfState(second) <
                    least_acceptance + 0.02)
                  continue;
                auto sampler = base;
                sampler.SetRngState(first);
                const auto outcome = session.executor->VerifySampledToken(
                    0, token, draft.ids, draft.probabilities, q, sampler);
                Expect(!outcome.accepted,
                       label + ": a draw above the threshold was accepted");
                return static_cast<std::int64_t>(outcome.token);
              }
              throw std::runtime_error("no rejecting state for a draw");
            },
            depth));
    Expect(measure.contiguous,
           label + ": the residual sampler is not an inverse CDF");
    conditional = sampled_oracle::MeasureError(
        measure, reference.residual.empty() ? target : reference.residual);
    const double resolution = std::ldexp(1.0, -depth - 2);
    Expect(conditional * reference.rejection_mass <=
               (host ? 1e-9 : bound.emitted_residual) + resolution,
           label + ": the residual moves the emitted distribution by " +
               std::to_string(conditional * reference.rejection_mass));
  }

  // The ordinary sampler on the same row.
  double plain = -1.0;
  if (!host) {
    const auto measure =
        sampled_oracle::MeasureRuns(sampled_oracle::RecoverRuns(
            [&](double uniform) {
              auto sampler = base;
              sampler.SetRngState(sampled_oracle::StateDrawing(uniform));
              return static_cast<std::int64_t>(
                  session.executor->SampleVerificationLogits(0, sampler));
            },
            depth));
    plain = sampled_oracle::MeasureError(measure, target);
    Expect(plain <= bound.sampled + std::ldexp(1.0, -depth - 2),
           label + ": the ordinary sampler draws p with error " +
               std::to_string(plain));
  }
  session.executor->FinishVerification();

  double outside = 0.0;
  for (const Candidate& candidate : target)
    if (draft.Probability(candidate.token) == 0.0)
      outside += candidate.probability;
  std::cout << label << ",position=" << position
            << ",vocabulary=" << (prediction.whole ? "whole" : "draft")
            << ",support=" << target.size()
            << ",target_mass_outside_row=" << outside
            << ",accept_rate=" << law.acceptance_rate
            << ",law_err=" << law.total_error << ",law_bound=" << law.bound
            << ",accepted_mass_err=" << worst_accepted
            << ",residual_err=" << conditional << ",p_err=" << plain << '\n';
}

// --- A whole verifier step from a chosen random state
// ---------------------------

struct Plan {
  std::uint64_t seed{0};
  Token draft{0};
  bool accept{false};
  /// The bonus token after an acceptance, the correction after a rejection.
  Token second{0};
};

// The first random state, by the reference alone, whose step has the wanted
// outcome with every draw well inside its interval.
std::optional<Plan> FindPlan(World& world, const Cut& cut,
                             const Filters& filters,
                             const Prediction& prediction, bool accept,
                             std::optional<Token> first_token) {
  const Token stop = world.stand.target->GetTokenizer().GetEosTokenId();
  for (std::uint64_t attempt = 1; attempt < 200000; ++attempt) {
    Plan plan;
    plan.seed =
        Mix((static_cast<std::uint64_t>(cut.Position()) << 32U) + attempt);
    const std::uint64_t s1 = sampled_oracle::NextState(plan.seed);
    const std::uint64_t s2 = sampled_oracle::NextState(s1);
    const std::uint64_t s3 = sampled_oracle::NextState(s2);
    const auto draw =
        DrawDraft(prediction.proposed, sampled_oracle::UniformOfState(s1));
    if (draw.margin < 1e-5)
      continue;
    plan.draft = prediction.draft.ids[draw.slot];
    const double q = prediction.draft.probabilities[draw.slot];
    const double threshold = std::min(
        1.0, sampled_oracle::ProbabilityOf(prediction.target, plan.draft) / q);
    const double decision = sampled_oracle::UniformOfState(s2);
    if (std::abs(decision - threshold) < 1e-4 && threshold < 1.0)
      continue;
    plan.accept = decision < threshold;
    if (plan.accept != accept)
      continue;
    if (accept && first_token && plan.draft != *first_token)
      continue;
    const double third = sampled_oracle::UniformOfState(s3);
    if (plan.accept) {
      if (plan.draft == stop ||
          world.stand.target->GetTokenizer().IsStopToken(plan.draft))
        return plan;
      // The bonus token: the target after the draft, with it in the history.
      world.reference.Start(cut.prompt);
      (void)world.reference.Decode(cut.current);
      const auto logits = world.reference.Decode(plan.draft);
      const std::vector<Token> generated{cut.current, plan.draft};
      const auto drawn = gufo::test::InverseCdf(
          gufo::test::DeviceOrder(
              gufo::test::TargetOf(logits, filters, cut.prompt, generated),
              filters),
          third);
      if (drawn.margin < 1e-4)
        continue;
      plan.second = drawn.token;
      return plan;
    }
    const auto verification =
        sampled_oracle::Verify(prediction.target, prediction.draft, plan.draft);
    if (verification.residual.empty())
      continue;
    const auto drawn = gufo::test::InverseCdf(
        gufo::test::DeviceOrder(verification.residual, filters), third);
    if (drawn.margin < 1e-4)
      continue;
    plan.second = drawn.token;
    if (first_token && plan.second != *first_token)
      continue;
    return plan;
  }
  return std::nullopt;
}

void EndToEnd(World& world, Session& session, const Cut& cut,
              const Request& request, bool accept,
              std::optional<Token> first_token = std::nullopt) {
  const std::string label =
      cut.name + "," + request.name + (accept ? ",accept" : ",reject") +
      (first_token ? ",token=" + std::to_string(*first_token) : "");
  const auto prediction = Predict(world, cut, request.filters);
  const auto plan =
      FindPlan(world, cut, request.filters, prediction, accept, first_token);
  if (!plan) {
    Expect(false, label +
                      ": the reference finds no random state with this "
                      "outcome");
    return;
  }
  const auto& tokenizer = world.stand.target->GetTokenizer();
  const bool stops = accept && tokenizer.IsStopToken(plan->draft);
  const std::vector<Token> expected =
      stops    ? std::vector<Token>{plan->draft}
      : accept ? std::vector<Token>{plan->draft, plan->second}
               : std::vector<Token>{plan->second};

  const auto sequence = cut.Sequence();
  const std::uint32_t position = cut.Position();
  (void)session.verifier->Prime(cut.prompt);
  const std::vector<Token> published{cut.current};
  auto sampler = gufo::test::SamplerOf(request.filters, cut.prompt, published);
  sampler.SetRngState(plan->seed);
  const auto result = session.verifier->VerifyStep(
      sequence, position, cut.current, tokenizer.GetEosTokenId(), 16, sampler);

  Expect(result.draft_count == 1 && result.emitted_tokens == expected,
         label + ": the step emitted another token sequence (first " +
             std::to_string(result.emitted_tokens.empty()
                                ? 0
                                : result.emitted_tokens.front()) +
             ", expected " + std::to_string(expected.front()) + ")");
  // The draft draw, the decision, and one more draw for the bonus or the
  // correction; an accepted stop token ends the step after two.
  std::uint64_t state = plan->seed;
  for (int draw = 0; draw < (stops ? 2 : 3); ++draw)
    state = sampled_oracle::NextState(state);
  Expect(sampler.rng_state() == state,
         label + ": the step consumed another number of draws");
  if (result.emitted_tokens != expected) {
    session.verifier->Reset();
    return;
  }

  // The history ordinary decoding gives the same tokens.
  world.reference.Start(cut.prompt);
  auto frontier = world.reference.Decode(cut.current);
  const bool committed_draft = accept && !stops;
  if (committed_draft)
    frontier = world.reference.Decode(plan->draft);
  Expect(SameBits(result.next_token_logits, frontier),
         label + ": the frontier logits differ from ordinary decoding");
  auto history = sequence;
  history.insert(history.end(), expected.begin(), expected.end());
  const auto next_position = static_cast<std::uint32_t>(history.size() - 1);

  // The draft side: its next proposal is the reference row of this history.
  {
    const auto head_logits =
        gufo::test::HeadLogits(*world.head, history, world.reference, true);
    const bool whole =
        world.draft_ids.empty() ||
        gufo::test::NeedsWholeVocabulary(history, world.draft_ids);
    Prediction next;
    next.draft = gufo::test::ExpectedDraft(head_logits, world.draft_ids, whole,
                                           request.filters.temperature);
    std::uint64_t rng = Mix(plan->seed);
    const auto proposal = session.backend->ProposeSampled(
        history, next_position, 1, request.filters.temperature, &rng);
    CheckProposal(label + ",next-proposal", proposal, next,
                  sampled_oracle::UniformOfState(rng));
  }
  // The target side: the token after the emitted ones.
  const auto reference_next = world.reference.Decode(history.back());
  (void)session.executor->ForwardToken(history.back(), next_position, true);
  Expect(SameBits(session.executor->CopyLastLogits(), reference_next),
         label +
             ": the committed target history differs from ordinary "
             "decoding of the emitted tokens");
  std::cout << label << ",draft=" << plan->draft << ",emitted=";
  for (const Token token : expected)
    std::cout << token << ' ';
  std::cout << "\n";
  session.verifier->Reset();
}

// --- Several sessions
// -----------------------------------------------------------

bool SameProposal(const gufo::speculative::DraftProposal& a,
                  const gufo::speculative::DraftProposal& b) {
  return a.tokens == b.tokens && a.candidate_ids == b.candidate_ids &&
         a.candidates_per_token == b.candidates_per_token &&
         a.start_pos == b.start_pos &&
         SameBits(a.candidate_probabilities, b.candidate_probabilities);
}

// Proposals of the draft backends, alone and in batches. A backend advances
// as after a rejected draft: the hidden state of its current token, then the
// text's next token as the correction.
void CheckSharedProposals(World& world, std::vector<Session>& sessions,
                          std::span<const Cut> cuts, float temperature) {
  struct Member {
    std::vector<Token> sequence;
    std::vector<std::vector<float>> hidden;  // of the current token, per step
    std::size_t step{0};
  };
  constexpr std::size_t kSteps = 3;
  std::vector<Member> members(cuts.size());
  for (std::size_t index = 0; index < cuts.size(); ++index) {
    world.reference.Start(cuts[index].prompt);
    Token current = cuts[index].current;
    for (std::size_t step = 0; step < kSteps; ++step) {
      (void)world.reference.Decode(current);
      const auto row =
          world.reference.HiddenRow(world.reference.Tokens().size() - 1);
      members[index].hidden.emplace_back(row.begin(), row.end());
      current = cuts[index].next[step];
    }
  }
  const auto prime = [&]() {
    for (std::size_t index = 0; index < cuts.size(); ++index) {
      (void)sessions[index].verifier->Prime(cuts[index].prompt);
      members[index].sequence = cuts[index].Sequence();
      members[index].step = 0;
    }
  };
  const auto seed_of = [](std::size_t index, std::size_t step) {
    return Mix(0xC0FFEE00 + (index * 16) + step);
  };
  const auto advance = [&](std::size_t index) {
    auto& member = members[index];
    sessions[index].backend->UpdateTargetHidden(member.hidden[member.step]);
    sessions[index].backend->AcceptFeedback({}, cuts[index].next[member.step]);
    member.sequence.push_back(cuts[index].next[member.step]);
    ++member.step;
  };
  // A round proposes for `group` in one batch, in that order, then advances
  // its members.
  using Round = std::vector<std::size_t>;
  const std::vector<Round> rounds{
      {0},           // one session
      {0, 1, 2, 3},  // joined by three: a batch of four
      {0},           // alone again
      {2, 1},        // two sessions, led by another one
      {3, 2, 1},     // and three, led by the whole-vocabulary session
  };
  // Every proposal alone first.
  std::vector<std::vector<gufo::speculative::DraftProposal>> alone(cuts.size());
  prime();
  for (std::size_t index = 0; index < cuts.size(); ++index) {
    for (std::size_t step = 0; step < kSteps; ++step) {
      std::uint64_t rng = seed_of(index, step);
      const auto& member = members[index];
      alone[index].push_back(sessions[index].backend->ProposeSampled(
          member.sequence,
          static_cast<std::uint32_t>(member.sequence.size() - 1), 1,
          temperature, &rng));
      advance(index);
    }
  }
  prime();
  for (std::size_t round = 0; round < rounds.size(); ++round) {
    const Round& group = rounds[round];
    std::vector<std::uint64_t> rng(group.size());
    std::vector<gufo::speculative::DraftProposalRequest> requests;
    for (std::size_t slot = 0; slot < group.size(); ++slot) {
      const auto& member = members[group[slot]];
      rng[slot] = seed_of(group[slot], member.step);
      requests.push_back(
          {sessions[group[slot]].backend, member.sequence,
           static_cast<std::uint32_t>(member.sequence.size() - 1), 1,
           temperature, &rng[slot]});
    }
    const auto proposals = requests.front().backend->ProposeBatch(requests);
    std::size_t differing = proposals.size() != group.size();
    for (std::size_t slot = 0; slot < group.size() && differing == 0; ++slot) {
      const std::size_t index = group[slot];
      const auto& expected = alone[index][members[index].step];
      differing += !SameProposal(proposals[slot], expected) ||
                   rng[slot] != sampled_oracle::NextState(
                                    seed_of(index, members[index].step));
      Expect(!proposals[slot].tokens.empty(),
             "a batch within the cohort limit returned no draft");
    }
    Expect(differing == 0, "proposal round " + std::to_string(round) + " of " +
                               std::to_string(group.size()) +
                               " sessions differs from the lone proposals");
    std::cout << "shared-proposals,round=" << round
              << ",sessions=" << group.size() << ",leader=" << group.front()
              << ",differing=" << differing << '\n';
    for (const std::size_t index : group)
      advance(index);
  }
  for (auto& session : sessions)
    session.verifier->Reset();
}

// Whole verifier steps of several sessions, alone and in batches.
void CheckSharedSteps(World& world, std::vector<Session>& sessions,
                      std::span<const Cut> cuts, const Request& request) {
  const Token stop = world.stand.target->GetTokenizer().GetEosTokenId();
  struct Member {
    std::vector<Token> sequence;
    Token current{0};
    std::optional<gufo::sampling::SamplerState> sampler;
    std::vector<gufo::speculative::SpeculativeVerifier::StepResult> results;
    std::vector<std::uint64_t> states;
  };
  std::vector<Member> members(cuts.size());
  const auto prime = [&]() {
    for (std::size_t index = 0; index < cuts.size(); ++index) {
      (void)sessions[index].verifier->Prime(cuts[index].prompt);
      auto& member = members[index];
      member.sequence = cuts[index].Sequence();
      member.current = cuts[index].current;
      const std::vector<Token> published{member.current};
      member.sampler.emplace(gufo::test::SamplerOf(
          request.filters, cuts[index].prompt, published));
      member.sampler->SetRngState(Mix(0xBA7C4000 + index));
      member.results.clear();
      member.states.clear();
    }
  };
  // The runner's part of a step: publish the emitted tokens.
  const auto publish =
      [&](std::size_t index,
          gufo::speculative::SpeculativeVerifier::StepResult result) {
        auto& member = members[index];
        for (const Token token : result.emitted_tokens) {
          member.sequence.push_back(token);
          member.sampler->Accept(token);
        }
        member.current = result.next_token;
        member.states.push_back(member.sampler->rng_state());
        member.results.push_back(std::move(result));
      };
  const auto step = [&](std::span<const std::size_t> group) {
    std::vector<gufo::speculative::SpeculativeVerifier::StepRequest> requests;
    for (const std::size_t index : group) {
      auto& member = members[index];
      requests.push_back(
          {*sessions[index].verifier, member.sequence,
           static_cast<std::uint32_t>(member.sequence.size() - 1),
           member.current, stop, 16, *member.sampler});
    }
    auto results =
        gufo::speculative::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t slot = 0; slot < group.size(); ++slot)
      publish(group[slot], std::move(results[slot]));
  };
  using Round = std::vector<std::size_t>;
  const auto run = [&](const std::vector<Round>& rounds) {
    prime();
    for (const Round& round : rounds)
      step(round);
    auto record = std::move(members);
    members.assign(cuts.size(), {});
    return record;
  };
  // Three steps of session 0 and two of the others, each alone.
  const auto alone = run({{0}, {1}, {2}, {3}, {0}, {1}, {2}, {3}, {0}});
  const std::vector<std::pair<std::string, std::vector<Round>>> schedules{
      // Session 0 alone, then with three others, then alone.
      {"c1-c4-c1", {{0}, {1, 2, 3, 0}, {0}, {3, 2, 1}}},
      // Pairs, each led by another session.
      {"c2", {{0, 1}, {3, 2}, {1, 0}, {2, 3}, {0}}},
      // Four at once, led by each kind of session.
      {"c4", {{0, 1, 2, 3}, {3, 0, 1, 2}, {0}}},
  };
  for (const auto& [name, rounds] : schedules) {
    const auto together = run(rounds);
    std::size_t differing = 0;
    std::size_t accepted = 0;
    std::size_t steps = 0;
    for (std::size_t index = 0; index < cuts.size(); ++index) {
      const auto& a = alone[index];
      const auto& b = together[index];
      differing += a.results.size() != b.results.size();
      for (std::size_t s = 0; s < std::min(a.results.size(), b.results.size());
           ++s) {
        differing +=
            a.results[s].emitted_tokens != b.results[s].emitted_tokens ||
            a.results[s].accepted_count != b.results[s].accepted_count ||
            a.results[s].draft_count != b.results[s].draft_count ||
            a.states[s] != b.states[s] ||
            !SameBits(a.results[s].next_token_logits,
                      b.results[s].next_token_logits);
        accepted += b.results[s].accepted_count;
        steps += b.results[s].draft_count;
      }
    }
    Expect(differing == 0 && steps > 0,
           request.name + "," + name +
               ": batched verifier steps differ from the lone steps");
    std::cout << "shared-steps," << request.name << ',' << name
              << ",drafts=" << steps << ",accepted=" << accepted
              << ",differing=" << differing << '\n';
  }
  for (auto& session : sessions)
    session.verifier->Reset();
}

Constraint MakeConstraint(const Stand& stand) {
  const auto& tokenizer = stand.target->GetTokenizer();
  Constraint result;
  result.constraint = std::make_shared<gufo::sampling::TokenConstraint>();
  result.constraint->grammar = gufo::sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{"v":{"type":"integer"}},
              "required":["v"],"additionalProperties":false})"),
      false);
  const auto vocab =
      static_cast<std::uint32_t>(stand.target->GetConfig().vocab_size);
  result.constraint->vocabulary =
      std::make_shared<gufo::sampling::ConstraintVocabulary>(
          vocab, [&](std::uint32_t id) {
            return gufo::sampling::ConstraintVocabulary::Piece{
                tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
          });
  const auto allowed =
      result.constraint->Allowed(result.constraint->grammar->Start());
  result.allowed.assign(allowed->begin(), allowed->end());
  // The mask is an input here; its shape is still checked by hand: a JSON
  // object can only open with a brace (after optional white space).
  std::size_t open = 0;
  for (std::uint32_t id = 0; id < vocab; ++id) {
    if (result.allowed[id] == 0)
      continue;
    ++open;
    const auto piece = tokenizer.DecodeTokenCopy(id);
    const auto first = piece.find_first_not_of(" \t\r\n");
    Expect(first == std::string::npos || piece[first] == '{',
           "the JSON mask admits a token that does not open an object: " +
               std::to_string(id));
  }
  const auto brace = tokenizer.Encode("{");
  Expect(open > 0 && open < vocab / 100 && brace.size() == 1 &&
             result.allowed[brace.front()] != 0,
         "the JSON mask has an unexpected shape");
  return result;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_sampled_mtp_chain_test [MODEL.gguf]\n";
    return 2;
  }
  try {
    const auto stand = gufo::test::LoadStand(argc > 1 ? argv[1] : "");
    if (!stand)
      return 77;
    const auto& tokenizer = stand->target->GetTokenizer();
    std::cout << std::defaultfloat << std::setprecision(6);
    // The ids the synthetic checks use are those of this vocabulary.
    const std::vector<std::pair<const char*, Token>> control{
        {"<|endoftext|>", sampled_cases::kEndOfText},
        {"<|im_end|>", sampled_cases::kImEnd},
        {"<tool_call>", sampled_cases::kToolCall},
        {"</tool_call>", sampled_cases::kToolCallEnd},
        {"</think>", sampled_cases::kThinkEnd},
    };
    Expect(stand->target->GetConfig().vocab_size == sampled_cases::kQwenVocab,
           "the vocabulary has another size");
    for (const auto& [text, id] : control) {
      const auto encoded = tokenizer.Encode(text);
      Expect(encoded.size() == 1 && encoded.front() == id &&
                 tokenizer.DecodeToken(id) == text,
             std::string("control token id differs: ") + text);
    }
    Expect(tokenizer.GetEosTokenId() == sampled_cases::kImEnd &&
               tokenizer.IsStopToken(sampled_cases::kImEnd),
           "the end of turn is not the stop token");

    World world{*stand, Reference(*stand, kContext), nullptr,
                stand->mtp->GetDraftVocab().ids,
                stand->target->GetConfig().vocab_size};
    std::string error;
    world.head =
        gufo::hip::QwenMtpGpuExecutor::Create(stand->mtp, kContext, &error);
    if (!world.head)
      throw std::runtime_error("MTP head: " + error);
    Expect(world.head->SupportsSharedScore(),
           "this head does not score sessions together");

    std::vector<Session> sessions;
    for (int index = 0; index < 4; ++index)
      sessions.push_back(MakeSession(*stand));

    // The corpus texts and the Chinese text, each at its chosen position.
    std::vector<Cut> cuts;
    for (const Text& text : stand->texts) {
      const std::size_t first = std::min<std::size_t>(64, text.ids.size() / 2);
      cuts.push_back(CutAt(
          text, gufo::test::ChoosePosition(world.reference, text, first, 4)));
    }
    Expect(cuts.size() == 4, "four texts are expected");
    const auto requests = gufo::test::Requests();
    bool whole_seen = false;
    bool draft_seen = false;
    for (const Cut& cut : cuts) {
      const bool whole =
          gufo::test::NeedsWholeVocabulary(cut.Sequence(), world.draft_ids);
      whole_seen = whole_seen || whole;
      draft_seen = draft_seen || !whole;
      for (const Request& request : requests) {
        Dissect(world, sessions[0], cut, request);
        EndToEnd(world, sessions[0], cut, request, true);
        EndToEnd(world, sessions[0], cut, request, false);
      }
    }
    Expect(whole_seen && draft_seen,
           "both the draft vocabulary and the whole vocabulary must be "
           "exercised");

    // An end of turn and a tool call, as the accepted draft and as the
    // correction.
    const Text reply{
        "chat-end",
        tokenizer.Encode("<|im_start|>user\nReply with exactly one word: "
                         "hello<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                         "</think>\n\nhello")};
    const Text tool{
        "tool-call",
        tokenizer.Encode(
            "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user "
            "query.\n\nYou are provided with function signatures within "
            "<tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", "
            "\"function\": {\"name\": \"get_weather\", \"description\": \"Get "
            "the current weather for a city\", \"parameters\": {\"type\": "
            "\"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, "
            "\"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, "
            "return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": "
            "<function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call><|im_end|>\n<|im_start|>user\nWhat is the weather in "
            "Paris right now?<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
            "</think>\n\n")};
    const Request plain = requests[1];
    for (const auto& [text, token] :
         {std::pair{&reply, sampled_cases::kImEnd},
          std::pair{&tool, sampled_cases::kToolCall}}) {
      const Cut cut = CutAt(*text, text->ids.size() - 1);
      Dissect(world, sessions[0], cut, requests[0]);
      Dissect(world, sessions[0], cut, plain);
      EndToEnd(world, sessions[0], cut, plain, true, token);
      EndToEnd(world, sessions[0], cut, plain, false, token);
    }

    // A grammar mask on the target; the draft row stays unconstrained.
    {
      const auto constraint = MakeConstraint(*stand);
      Dissect(world, sessions[0], cuts[1], requests[0], &constraint);
      Dissect(world, sessions[0], cuts[2], plain, &constraint);
    }

    CheckSharedProposals(world, sessions, cuts, 1.0F);
    for (const Request& request : requests)
      CheckSharedSteps(world, sessions, cuts, request);
  } catch (const std::exception& failure) {
    std::cerr << "qwen_sampled_mtp_chain_test: " << failure.what() << '\n';
    return 1;
  }
  std::cout << (g_passed ? "PASS" : "FAIL") << '\n';
  return g_passed ? 0 : 1;
#else
  (void)argc;
  (void)argv;
  return 77;
#endif
}
