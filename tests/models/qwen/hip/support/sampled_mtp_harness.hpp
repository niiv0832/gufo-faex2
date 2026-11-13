#ifndef GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_SAMPLED_MTP_HARNESS_HPP_
#define GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_SAMPLED_MTP_HARNESS_HPP_

// The model, the texts and the reference computations shared by the checks
// of sampled MTP speculation on real weights.

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#include "tests/core/sampled_speculation_cases.hpp"
#include "tests/core/sampled_speculation_oracle.hpp"

namespace gufo::test {

using sampled_oracle::Candidate;
using sampled_oracle::DraftRow;
using sampled_oracle::Filters;
using sampled_oracle::Token;

/// The opening tokens of a human-written text.
struct Text {
  std::string name;
  std::vector<Token> ids;
};

// Windows of the frozen audit corpus (Russian prose, English prose, Python),
// tokenized by the reference tokenizer of the pinned GGUF.
inline std::vector<Text> CorpusTexts() {
  return {
      {"ru",
       {15729,  58503,  52033,  149537, 159046, 52033,  5301,   152689, 318,
        148551, 220,    16,     24,     24,     24,     159067, 207613, 45480,
        179224, 1892,   159046, 52033,  5301,   152689, 8,      1892,   151718,
        12696,  150530, 187942, 12,     148397, 148414, 95525,  170312, 13,
        155782, 150704, 32884,  77779,  26125,  32401,  161008, 210844, 149193,
        170509, 7347,   210844, 149280, 160023, 11,     5624,   94378,  18096,
        182030, 11,     178147, 34226,  194736, 148288, 159424, 230977, 84496,
        7347,   161614, 4096,   192444, 163731, 152807, 149537, 210844, 272,
        241180, 161420, 42565,  174192, 151822, 149127, 5624,   69829,  179045,
        13,     195474, 234710, 149193, 153413, 148705, 13,     271,    15729,
        58503,  149537, 210844, 1892,   170634, 22346,  157746, 148820, 150047,
        7347,   150749, 148820, 150047, 200998}},
      {"en",
       {32,    17933, 3379,  369,   264,   21461, 3545,  364,   18377, 7035,
        310,   449,   25570, 290,   7099,  38787, 5028,  264,   1133,  10728,
        318,   20171, 430,   5392,  466,   26410, 681,   26600, 48270, 279,
        25570, 13,    1049,  12495, 264,   21100, 15214, 383,   6575,  17933,
        5245,  303,   864,   279,   26410, 11,    5392,  466,   975,   10728,
        557,   59462, 466,   64499, 18253, 310,   10752, 279,   8059,  314,
        25570, 11,    321,   47126, 279,   1817,  13,    43792, 1429,  364,
        21306, 11,    279,   26257, 321,   3521,  8597,  314,   279,   17933,
        3379,  557,   799,   314,   279,   1379,  30415, 4216,  303,   279,
        2018,  85631, 13,    271,   623,   9564,  11,    2094,  220,   16,
        19,    19,    15,    11}},
      {"code",
       {1005, 25875, 4253,  1590,  198,   262,  4071,  82693, 1406,  523,
        364,  11778, 33633, 71483, 271,   262,  1259,  48093, 548,   283,
        4178, 62,    4431,  1780,  62,    390,  1780,  62,    7506,  504,
        8962, 284,   40253, 837,   198,   262,  653,   41959, 11,    279,
        869,  314,   279,   25875, 2774,  369,  2574,  539,   198,   262,
        653,  220,   9987,  16,    31230, 62,   7506,  348,   696,   390,
        348,  220,   16,    15,    332,   62,   4431,  198,   262,   653,
        9493, 2663,  513,   1777,  1806,  539,  696,   284,   40253, 606,
        2912, 271,   262,   653,   1165,  2224, 49835, 11,    748,   958,
        1259, 902,   548,   524,   1259,  2251, 548,   198,   262,   687,
        1259, 902,   3676,  17535}},
  };
}

/// A text in a script the MTP draft vocabulary lacks: its drafts come from
/// the whole vocabulary.
inline constexpr const char* kChineseText =
    "活字印刷术的发明改变了知识传播的方式。在此之前，书籍只能靠手工抄写，"
    "一部书往往需要数月才能完成，价格昂贵，只有少数人能够拥有。"
    "印刷术出现以后，书籍的成本迅速下降，学者们可以更方便地交流思想，"
    "普通人也有机会读到经典著作。许多历史学家认为，这项技术";

struct Stand {
  std::shared_ptr<const hip::QwenGpuModel> target;
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp;
  std::vector<Text> texts;
};

/// Loads the weights named by `path` or GUFO_QWEN_LAYERED_PREFILL_MODEL.
/// Returns null, with the reason printed, when the test must be skipped.
inline std::unique_ptr<Stand> LoadStand(const std::string& argument) {
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = !argument.empty() ? argument
                           : env != nullptr  ? env
                                             : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return nullptr;
  }
  models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
  models::qwen38_flash_next::rocm::RegisterQwenNpuFfn();
  std::string error;
  auto reader_owner = core::GgufReader::OpenFile(path, &error);
  if (!reader_owner)
    throw std::runtime_error("open: " + error);
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto stand = std::make_unique<Stand>();
  stand->target = hip::QwenGpuModel::CreateFromGguf(reader, &error);
  if (!stand->target)
    throw std::runtime_error("model: " + error);
  stand->mtp = hip::QwenMtpGpuModel::CreateEmbedded(stand->target, &error);
  if (!stand->mtp) {
    std::cerr << "no embedded MTP head (" << error << "); skipping\n";
    return nullptr;
  }
  stand->texts = CorpusTexts();
  const auto& tokenizer = stand->target->GetTokenizer();
  stand->texts.push_back({"zh", tokenizer.Encode(kChineseText)});
  const std::size_t vocab = stand->target->GetConfig().vocab_size;
  for (const Text& text : stand->texts)
    for (const Token id : text.ids)
      if (id >= vocab)
        throw std::runtime_error(text.name + ": token outside the vocabulary");
  return stand;
}

/// Sampling requests the checks run.
struct Request {
  std::string name;
  Filters filters;
};

inline std::vector<Request> Requests() {
  return {
      {"profile", sampled_cases::kProfile},
      {"plain", {}},
      {"penalties",
       {.temperature = 0.7F,
        .top_k = 40,
        .top_p = 0.9F,
        .repeat_penalty = 1.1F,
        .frequency_penalty = 0.1F,
        .presence_penalty = 0.2F}},
  };
}

inline sampling::SamplingConfig ConfigOf(const Filters& filters) {
  sampling::SamplingConfig config;
  config.temperature = filters.temperature;
  config.top_k = filters.top_k;
  config.top_p = filters.top_p;
  config.min_p = filters.min_p;
  config.min_keep = filters.min_keep;
  config.repeat_penalty = filters.repeat_penalty;
  config.frequency_penalty = filters.frequency_penalty;
  config.presence_penalty = filters.presence_penalty;
  config.seed = 0;
  return config;
}

/// The request's sampler once it has published `generated` after `prompt`.
inline sampling::SamplerState SamplerOf(const Filters& filters,
                                        std::span<const Token> prompt,
                                        std::span<const Token> generated) {
  sampling::SamplerState sampler(ConfigOf(filters), prompt);
  sampler.Accept(generated);
  return sampler;
}

/// The reference target distribution for the same request and history.
inline std::vector<Candidate> TargetOf(std::span<const float> logits,
                                       const Filters& filters,
                                       std::span<const Token> prompt,
                                       std::span<const Token> generated) {
  sampled_cases::Case history;
  history.filters = filters;
  history.prompt.assign(prompt.begin(), prompt.end());
  history.generated.assign(generated.begin(), generated.end());
  return sampled_oracle::TargetDistribution(logits, filters,
                                            history.Penalties());
}

/// The order in which the device sampler lays the kept tokens over [0, 1):
/// by token id without top-k and top-p, most likely first with them.
inline std::vector<Candidate> DeviceOrder(std::vector<Candidate> distribution,
                                          const Filters& filters) {
  const bool ordered = filters.top_k > 0 || filters.top_p < 1.0F ||
                       (filters.min_p > 0.0F && filters.min_keep > 1);
  if (!ordered)
    std::ranges::sort(distribution, {}, &Candidate::token);
  return distribution;
}

struct Drawn {
  Token token{0};
  /// Distance from the draw to the nearest interval end.
  double margin{0.0};
};

/// The token whose interval of `distribution` holds `uniform`.
inline Drawn InverseCdf(std::span<const Candidate> distribution,
                        double uniform) {
  double begin = 0.0;
  for (std::size_t i = 0; i < distribution.size(); ++i) {
    const double end = begin + distribution[i].probability;
    if (uniform < end || i + 1 == distribution.size())
      return {distribution[i].token,
              std::min(uniform - begin,
                       i + 1 == distribution.size() ? 1.0 : end - uniform)};
    begin = end;
  }
  throw std::logic_error("empty distribution");
}

/// Ordinary decoding of `text`: the prompt, then tokens one at a time. It
/// defines the target logits and supplies the hidden states an MTP head
/// reads.
class Reference {
public:
  Reference(const Stand& stand, std::uint32_t context) {
    std::string error;
    executor_ = hip::QwenGpuExecutor::Create(stand.target, &error, context);
    if (!executor_)
      throw std::runtime_error("executor: " + error);
    hidden_size_ = stand.target->GetConfig().hidden_size;
  }

  /// Prefills `prompt`.
  void Start(std::span<const Token> prompt) {
    executor_->Reset();
    executor_->SetPromptHiddenCapture(true, {});
    (void)executor_->ForwardPromptBatch(prompt, 0, true);
    const auto hidden = executor_->GetPromptHiddenStates();
    if (hidden.size() != prompt.size() * hidden_size_)
      throw std::runtime_error("prompt hidden states have another shape");
    tokens_.assign(prompt.begin(), prompt.end());
    hidden_.assign(hidden.begin(), hidden.end());
  }

  /// Decodes one more token and returns the logits that follow it.
  std::vector<float> Decode(Token token) {
    (void)executor_->ForwardToken(
        token, static_cast<std::uint32_t>(tokens_.size()), true);
    const auto logits = executor_->CopyLastLogits();
    std::vector<float> row(logits.begin(), logits.end());
    const auto hidden = executor_->CopyLastHidden();
    if (hidden.size() != hidden_size_)
      throw std::runtime_error("decode hidden state has another shape");
    tokens_.push_back(token);
    hidden_.insert(hidden_.end(), hidden.begin(), hidden.end());
    return row;
  }

  [[nodiscard]] const std::vector<Token>& Tokens() const { return tokens_; }
  /// One row per processed token.
  [[nodiscard]] const std::vector<float>& Hidden() const { return hidden_; }
  [[nodiscard]] std::span<const float> HiddenRow(std::size_t row) const {
    return std::span(hidden_).subspan(row * hidden_size_, hidden_size_);
  }

private:
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
  std::size_t hidden_size_{0};
  std::vector<Token> tokens_;
  std::vector<float> hidden_;
};

/// An MTP head run apart from the production draft backend: the logits it
/// gives the token after `tokens` over the whole vocabulary, from the hidden
/// states of ordinary decoding.
inline std::vector<float> HeadLogits(hip::QwenMtpGpuExecutor& head,
                                     std::span<const Token> tokens,
                                     const Reference& reference, bool whole) {
  const std::size_t count = tokens.size();
  if (count < 2 || reference.Tokens().size() < count - 1)
    throw std::logic_error("the head needs a processed history");
  const std::size_t hidden = head.GetHiddenSize();
  head.Reset();
  // Position i pairs token i + 1 with the hidden state of token i.
  if (count > 2)
    head.PrimeBatch(tokens.subspan(1, count - 2),
                    std::span(reference.Hidden()).first((count - 2) * hidden),
                    0);
  head.ScoreWholeVocabulary(whole);
  (void)head.ForwardTargetHidden(tokens.back(), reference.HiddenRow(count - 2),
                                 static_cast<std::uint32_t>(count - 2));
  const auto logits = head.CopyLastLogits();
  return {logits.begin(), logits.end()};
}

/// Whether one of the last 64 tokens is outside the draft vocabulary `ids`
/// (ascending): such a step drafts from the whole vocabulary.
inline bool NeedsWholeVocabulary(std::span<const Token> history,
                                 std::span<const Token> ids) {
  const std::size_t window = std::min<std::size_t>(history.size(), 64);
  for (const Token token : history.last(window))
    if (!std::ranges::binary_search(ids, token))
      return true;
  return false;
}

/// The draft row the reference expects from whole-vocabulary head logits:
/// the 32 best of the tokens the step may draft, at `temperature`.
inline DraftRow ExpectedDraft(std::span<const float> whole_logits,
                              std::span<const Token> draft_ids, bool whole,
                              float temperature) {
  std::vector<float> scored;
  std::vector<Token> ids;
  if (whole) {
    scored.assign(whole_logits.begin(), whole_logits.end());
    ids.resize(scored.size());
    for (std::size_t id = 0; id < ids.size(); ++id)
      ids[id] = static_cast<Token>(id);
  } else {
    ids.assign(draft_ids.begin(), draft_ids.end());
    for (const Token id : ids)
      scored.push_back(whole_logits[id]);
  }
  return sampled_cases::HeadDraft(scored, ids, temperature);
}

/// The first position of `text`, from `first` on, whose target distribution
/// under the deployment profile has at least three tokens and no token above
/// 0.7; the last admissible position when there is none. The choice reads
/// ordinary decoding only.
inline std::size_t ChoosePosition(Reference& reference, const Text& text,
                                  std::size_t first, std::size_t reserve) {
  const std::size_t last = text.ids.size() - reserve - 1;
  first = std::min(first, last);
  reference.Start(std::span(text.ids).first(first));
  for (std::size_t position = first; position <= last; ++position) {
    const auto logits = reference.Decode(text.ids[position]);
    const auto target =
        sampled_oracle::TargetDistribution(logits, sampled_cases::kProfile);
    if (target.size() >= 3 && target.front().probability <= 0.7)
      return position;
  }
  return last;
}

}  // namespace gufo::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_SAMPLED_MTP_HARNESS_HPP_
