#ifndef GUFO_TESTS_CORE_SAMPLED_SPECULATION_CASES_HPP_
#define GUFO_TESTS_CORE_SAMPLED_SPECULATION_CASES_HPP_

// Synthetic targets and draft rows shared by the CPU and GPU checks of
// sampled speculative verification.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tests/core/sampled_speculation_oracle.hpp"

namespace sampled_cases {

using sampled_oracle::DraftRow;
using sampled_oracle::Filters;
using sampled_oracle::Penalty;
using sampled_oracle::Token;

// Ids of the pinned Qwen3.6-35B-A3B vocabulary (tokenizer.ggml.tokens of the
// GGUF); the model-gated test checks them against the loaded tokenizer.
inline constexpr std::size_t kQwenVocab = 248320;
inline constexpr Token kEndOfText = 248044;
inline constexpr Token kImEnd = 248046;
inline constexpr Token kToolCall = 248058;
inline constexpr Token kToolCallEnd = 248059;
inline constexpr Token kThinkEnd = 248069;

// The deployment's request profile.
inline constexpr Filters kProfile{
    .temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .min_p = 0.05F};

struct Case {
  std::string name;
  std::vector<float> logits;
  Filters filters;
  std::size_t repeat_last_n{64};
  /// Tokens before the request, then the tokens it generated.
  std::vector<Token> prompt;
  std::vector<Token> generated;
  /// Constraint mask by token id; empty without a constraint.
  std::vector<std::uint8_t> allowed;
  std::vector<DraftRow> drafts;

  /// The penalties of this history: a token is repeated when it is among the
  /// last repeat_last_n tokens of prompt + generated, and counted once per
  /// generated occurrence.
  [[nodiscard]] std::vector<Penalty> Penalties() const {
    std::map<Token, Penalty> found;
    if (filters.frequency_penalty != 0.0F || filters.presence_penalty != 0.0F) {
      for (const Token token : generated) {
        found[token].token = token;
        ++found[token].generated;
      }
    }
    if (filters.repeat_penalty != 1.0F) {
      std::vector<Token> history(prompt);
      history.insert(history.end(), generated.begin(), generated.end());
      const std::size_t window = std::min(history.size(), repeat_last_n);
      for (std::size_t i = history.size() - window; i < history.size(); ++i) {
        found[history[i]].token = history[i];
        found[history[i]].repeated = true;
      }
    }
    std::vector<Penalty> penalties;
    for (const auto& [token, penalty] : found)
      penalties.push_back(penalty);
    return penalties;
  }

  /// Whether a request with these filters draws random numbers.
  [[nodiscard]] bool Random() const {
    return filters.temperature > 0.0F &&
           (filters.top_k != 1 || filters.min_keep > 1);
  }

  /// A slot of `draft` that can be drawn.
  [[nodiscard]] std::size_t DraftSlot(const DraftRow& draft) const {
    for (std::size_t slot = 0; slot < draft.ids.size(); ++slot)
      if (draft.probabilities[slot] > 0.0F)
        return slot;
    throw std::logic_error(name + ": a draft row has no mass");
  }
};

/// Reproducible values in [0, 1).
class Stream {
public:
  explicit Stream(std::uint64_t seed) : state_(seed) {}
  double Next() {
    state_ = (state_ * UINT64_C(6364136223846793005)) +
             UINT64_C(1442695040888963407);
    return static_cast<double>(state_ >> 11U) * 0x1.0p-53;
  }

private:
  std::uint64_t state_;
};

/// The draft row an MTP head with logits `head` over the tokens `ids` gives:
/// its 32 best logits at `temperature`, with their global ids.
inline DraftRow HeadDraft(std::span<const float> head,
                          std::span<const Token> ids, float temperature) {
  const auto row = sampled_oracle::HeadTopK(head, temperature, 32);
  DraftRow draft;
  for (std::size_t i = 0; i < row.indices.size(); ++i) {
    draft.ids.push_back(ids[row.indices[i]]);
    draft.probabilities.push_back(static_cast<float>(row.probabilities[i]));
  }
  return draft;
}

/// A dense target of `size` logits with a handful of likely tokens, and the
/// rows of a head that sees it through noise: one over a shortlist of the odd
/// ids, one over the whole vocabulary.
inline Case Dense(std::string name, std::size_t size, Filters filters,
                  std::uint64_t seed) {
  Case test;
  test.name = std::move(name);
  test.filters = filters;
  Stream stream(seed);
  test.logits.resize(size);
  for (float& logit : test.logits)
    logit = static_cast<float>((stream.Next() * 9.0) - 12.0);
  // Likely tokens on both sides of the shortlist.
  const std::size_t step = size / 13;
  for (std::size_t peak = 0; peak < 12; ++peak) {
    test.logits[(peak * step) + 2] =
        static_cast<float>(4.0 - (0.37 * static_cast<double>(peak)));
  }
  std::vector<float> head(size);
  for (std::size_t id = 0; id < size; ++id)
    head[id] = test.logits[id] + static_cast<float>(stream.Next() - 0.5);
  std::vector<Token> whole(size);
  std::vector<Token> shortlist;
  std::vector<float> shortlist_head;
  for (std::size_t id = 0; id < size; ++id) {
    whole[id] = static_cast<Token>(id);
    if (id % 2 == 1) {
      shortlist.push_back(static_cast<Token>(id));
      shortlist_head.push_back(head[id]);
    }
  }
  const float temperature =
      filters.temperature > 0.0F ? filters.temperature : 1.0F;
  test.drafts.push_back(HeadDraft(shortlist_head, shortlist, temperature));
  test.drafts.push_back(HeadDraft(head, whole, temperature));
  return test;
}

/// Every case; the dense targets without top-k have `dense` tokens.
inline std::vector<Case> All(std::size_t dense = 4096) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<Case> cases;

  {
    // Target tokens the draft row lacks, and draft tokens the target lacks.
    Case test;
    test.name = "outside-support";
    test.logits = {2.0F, 1.75F, 1.5F,  1.25F, 1.0F,  0.75F, 0.5F,  0.25F,
                   0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -2.5F, -3.0F, -3.5F};
    test.filters.top_k = 4;
    test.drafts = {
        {{0, 10, 11}, {0.5F, 0.25F, 0.25F}},
        {{12, 13}, {0.75F, 0.25F}},
        {{3, 2, 1, 0}, {0.125F, 0.125F, 0.25F, 0.5F}},
    };
    cases.push_back(test);
    test.name = "outside-support-unfiltered";
    test.filters.top_k = 0;
    cases.push_back(test);
  }
  {
    // Equal logits across the top-k cut, and p equal to q exactly.
    Case test;
    test.name = "ties";
    test.logits = {1.0F, 1.0F, 1.0F, 0.0F, 0.0F, -1.0F, -1.0F, -1.0F};
    test.filters.top_k = 2;
    test.drafts = {
        {{0, 1}, {0.5F, 0.5F}},
        {{0, 2}, {0.5F, 0.5F}},
        {{1, 0}, {0.25F, 0.75F}},
    };
    cases.push_back(test);
    test.name = "ties-unfiltered";
    test.filters.top_k = 0;
    test.drafts.push_back(
        {{0, 1, 2, 3, 4, 5, 6, 7},
         {0.125F, 0.125F, 0.125F, 0.125F, 0.125F, 0.125F, 0.125F, 0.125F}});
    cases.push_back(test);
    test.name = "ties-nucleus";
    test.filters.top_p = 0.5F;
    cases.push_back(test);
  }
  {
    // Zero draft probabilities, and tokens without a finite target logit.
    Case test;
    test.name = "zeros";
    test.logits = {0.0F, -1.0F, -inf, -2.0F, nan, 0.5F};
    test.drafts = {
        {{5, 0}, {1.0F, 0.0F}},
        {{2, 4, 1}, {0.5F, 0.25F, 0.25F}},
        {{0, 1, 3, 5}, {0.0F, 0.5F, 0.0F, 0.5F}},
    };
    cases.push_back(test);
  }
  {
    // Probabilities of 1e-30 .. 1e-45 on both sides.
    Case test;
    test.name = "near-underflow";
    test.logits = {0.0F, -70.0F, -100.0F, -inf};
    test.drafts = {
        {{0, 1, 2}, {1.0F, 1e-30F, std::numeric_limits<float>::denorm_min()}},
        {{1, 0}, {1.0F, 0.0F}},
        {{2}, {1.0F}},
        {{0, 1}, {0.5F, 0.5F}},
    };
    cases.push_back(test);
  }
  {
    // p <= q on the whole support: no residual. The row sums to 1 + 8e-6,
    // inside the verifier's tolerance, so a rejection is possible.
    Case test;
    test.name = "empty-residual";
    test.logits = {0.0F, 0.0F};
    test.drafts = {{{0, 1}, {0.500004F, 0.500004F}}};
    cases.push_back(test);
  }
  cases.push_back(Dense("plain", dense, {}, 11));
  cases.push_back(Dense("profile", 4096, kProfile, 12));
  {
    Case test = Dense("profile-penalties", 4096, kProfile, 13);
    test.filters.repeat_penalty = 1.1F;
    test.filters.presence_penalty = 0.5F;
    test.filters.frequency_penalty = 0.25F;
    test.repeat_last_n = 4;
    // The likeliest tokens are in the history; the first prompt token has
    // left the repetition window.
    test.prompt = {317, 2, 632, 7};
    test.generated = {2, 947, 2};
    cases.push_back(test);
  }
  {
    // Sharpened and flattened targets.
    Filters filters;
    filters.temperature = 0.05F;
    cases.push_back(Dense("temperature-low", dense, filters, 14));
    filters.temperature = 2.5F;
    filters.min_p = 0.02F;
    filters.min_keep = 3;
    cases.push_back(Dense("temperature-high-min-keep", dense, filters, 15));
  }
  {
    // Deterministic requests: the distribution is one token.
    Case test = Dense("greedy", 4096, {.temperature = 0.0F}, 16);
    test.drafts = {{{2, 317}, {0.625F, 0.375F}}, {{317, 5}, {0.5F, 0.5F}}};
    cases.push_back(test);
    test.name = "top-k-one";
    test.filters = {.temperature = 1.0F, .top_k = 1};
    cases.push_back(test);
  }
  for (const bool profile : {false, true}) {
    // End-of-turn and tool tokens of the real vocabulary as the draft and as
    // the residual choice.
    Case test;
    test.name = profile ? "control-tokens-profile" : "control-tokens";
    test.filters = profile ? kProfile : Filters{};
    test.logits.assign(kQwenVocab, -inf);
    Stream stream(17);
    for (std::size_t id = 500; id < kQwenVocab; id += 971)
      test.logits[id] = static_cast<float>((stream.Next() * 4.0) - 5.0);
    test.logits[1000] = 2.75F;
    test.logits[kImEnd] = 3.0F;
    test.logits[kToolCall] = 2.5F;
    test.logits[kToolCallEnd] = 1.0F;
    test.logits[kThinkEnd] = 1.5F;
    test.logits[kEndOfText] = 0.5F;
    test.drafts = {
        // The draft over-proposes the end of turn; the tool call is short.
        {{kImEnd, 1000, kToolCall}, {0.75F, 0.125F, 0.125F}},
        // The row misses the control tokens altogether.
        {{1000, 1471, 2442}, {0.5F, 0.25F, 0.25F}},
        // A tool call the target does not want as much.
        {{kToolCall, kToolCallEnd, kEndOfText}, {0.5F, 0.25F, 0.25F}},
    };
    cases.push_back(test);
  }
  return cases;
}

/// The same case with its tokens `stride` ids apart in a vocabulary of
/// `size`; the ids between them have no finite logit.
inline Case Spread(const Case& source, std::size_t size) {
  const std::size_t stride = size / source.logits.size();
  if (stride < 2)
    throw std::logic_error(source.name + ": nothing to spread");
  Case test = source;
  test.name += "@" + std::to_string(size);
  test.logits.assign(size, -std::numeric_limits<float>::infinity());
  for (std::size_t id = 0; id < source.logits.size(); ++id)
    test.logits[id * stride] = source.logits[id];
  const auto move = [&](std::vector<Token>& ids) {
    for (Token& id : ids)
      id = static_cast<Token>(id * stride);
  };
  move(test.prompt);
  move(test.generated);
  for (DraftRow& draft : test.drafts)
    move(draft.ids);
  return test;
}

inline Case Named(const std::string& name) {
  if (name == "mask") {
    // Tokens 1 and 2 are allowed; the draft row ignores the mask.
    Case test;
    test.name = name;
    test.logits = {3.0F, 1.0F, 0.5F, 2.0F, 4.0F};
    test.allowed = {0, 1, 1, 0, 0};
    test.drafts = {
        {{4, 0, 1}, {0.5F, 0.25F, 0.25F}},
        {{1, 2}, {0.875F, 0.125F}},
    };
    return test;
  }
  for (const Case& test : All())
    if (test.name == name)
      return test;
  throw std::logic_error("unknown case " + name);
}

struct HeadCase {
  std::string name;
  std::vector<float> scored;
};

inline std::vector<HeadCase> HeadRows() {
  std::vector<HeadCase> rows;
  Stream stream(31);
  HeadCase dense{"head-dense", std::vector<float>(4096)};
  for (float& value : dense.scored)
    value = static_cast<float>((stream.Next() * 20.0) - 10.0);
  rows.push_back(dense);
  HeadCase ties{"head-ties", std::vector<float>(161)};
  for (std::size_t i = 0; i < ties.scored.size(); ++i)
    ties.scored[i] = static_cast<float>(i % 7);
  rows.push_back(ties);
  rows.push_back({"head-narrow", {0.5F, -1.0F, 3.0F, 3.0F, 0.0F, 2.0F, -4.0F}});
  // One dominant logit: the tail of the row rounds to zero probability.
  HeadCase peaked{"head-peaked", std::vector<float>(100, -60.0F)};
  peaked.scored[41] = 55.0F;
  peaked.scored[7] = 54.0F;
  rows.push_back(peaked);
  rows.push_back({"head-flat", std::vector<float>(64, 1.25F)});
  return rows;
}

}  // namespace sampled_cases

#endif  // GUFO_TESTS_CORE_SAMPLED_SPECULATION_CASES_HPP_
