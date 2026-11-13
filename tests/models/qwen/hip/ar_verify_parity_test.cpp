// Target contract of speculative decoding on real weights. Ordinary
// single-token decoding defines the target distribution. Everything that
// stands in for it must produce the same logits bit for bit, for the same
// token at the same position after the same history:
//   - a speculative verification row, alone or sharing a launch with the rows
//     of other sessions, whichever session leads the launch;
//   - the token decoded after a fully accepted chunk;
//   - the token decoded after a rejected draft has been rolled back and the
//     accepted prefix committed;
//   - a decode step batched across sessions;
//   - a session reused after an abandoned verification.
// Sampled speculation accepts and resamples against the verification rows, so
// a difference there changes the distribution the user is served even when
// the most likely token is unchanged.
//
// The test keeps the production order: prefill the history, SaveState at the
// current token, verify, FinishVerification, then RestoreState and
// CommitVerificationChunk when a draft was rejected.
//
// qwen_ar_verify_parity_test [MODEL.gguf [PREFIX ROWS IDS.txt...]]
// Without a model argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights;
// the test is skipped without it or for a model without experts. One to four
// files of whitespace-separated token ids replace the built-in histories.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

constexpr std::uint32_t kContext = 4096;
constexpr std::size_t kTapsPerPass = 9;

// A short Russian sentence. Position 15 with two verification rows is the
// input on which verification was first found to differ from decoding.
const std::vector<Token> kRussian{52337,  12666,  22078, 1739,   52033,  172007,
                                  8688,   318,    12666, 94498,  157,    123,
                                  114,    44122,  1892,  153366, 7347,   205495,
                                  155277, 151194, 44122, 1892,   232021, 11};

struct Session {
  std::vector<Token> ids;
  std::unique_ptr<Executor> executor;
  /// Decode logits for the tokens at prefix .. prefix + rows, inclusive.
  std::vector<float> reference;
};

struct Scenario {
  std::string name;
  std::uint32_t prefix;
  std::uint32_t rows;
  std::size_t sessions;
};

struct Context {
  std::size_t layers;
  std::size_t hidden;
  std::size_t vocab;
  bool passed = true;
};

// A history of `length` tokens drawn from the sentence; distinct per seed.
std::vector<Token> History(std::size_t length, std::uint64_t seed) {
  std::vector<Token> ids;
  ids.reserve(length);
  std::uint64_t state = (seed * 0x9E3779B97F4A7C15ULL) + 1;
  for (std::size_t index = 0; index < length; ++index) {
    if (seed == 0 && index < kRussian.size()) {
      ids.push_back(kRussian[index]);
      continue;
    }
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    ids.push_back(kRussian[(state >> 33) % kRussian.size()]);
  }
  return ids;
}

bool SameBits(std::span<const float> reference,
              std::span<const float> candidate) {
  return reference.size() == candidate.size() &&
         std::memcmp(reference.data(), candidate.data(),
                     reference.size_bytes()) == 0;
}

// KL(softmax(reference) || softmax(candidate)) in nats.
double Divergence(std::span<const float> reference,
                  std::span<const float> candidate) {
  const auto log_partition = [](std::span<const float> logits) {
    const double top = *std::ranges::max_element(logits);
    double sum = 0.0;
    for (const float value : logits)
      sum += std::exp(static_cast<double>(value) - top);
    return top + std::log(sum);
  };
  const double reference_z = log_partition(reference);
  const double candidate_z = log_partition(candidate);
  double divergence = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double log_p = reference[index] - reference_z;
    divergence += std::exp(log_p) * (log_p - (candidate[index] - candidate_z));
  }
  return divergence;
}

void Check(Context& context, const std::string& label,
           std::span<const float> reference, std::span<const float> candidate) {
  if (SameBits(reference, candidate))
    return;
  context.passed = false;
  std::size_t differing = 0;
  double max_abs = 0.0;
  const std::size_t count = std::min(reference.size(), candidate.size());
  for (std::size_t index = 0; index < count; ++index) {
    differing +=
        std::memcmp(&reference[index], &candidate[index], sizeof(float)) != 0;
    max_abs =
        std::max(max_abs, std::abs(static_cast<double>(candidate[index]) -
                                   static_cast<double>(reference[index])));
  }
  std::cout << label << ",differing=" << differing << ",max_abs=" << max_abs;
  if (reference.size() == candidate.size() && reference.size() == context.vocab)
    std::cout << ",kl=" << Divergence(reference, candidate);
  std::cout << ",FAIL\n";
}

std::span<const float> Row(const Session& session, const Context& context,
                           std::size_t row) {
  return std::span(session.reference)
      .subspan(row * context.vocab, context.vocab);
}

void Prefill(Session& session, std::uint32_t prefix, bool save_state,
             std::span<const std::uint32_t> taps = {}) {
  auto& executor = *session.executor;
  executor.Reset();
  executor.SetPromptHiddenCapture(true, taps);
  (void)executor.ForwardPromptBatch(std::span(session.ids).first(prefix), 0,
                                    true);
  // The history is processed; the current token ids[prefix] is not.
  if (save_state)
    executor.SaveState(prefix);
}

std::vector<float> Decode(Session& session, std::uint32_t prefix,
                          std::uint32_t count, bool save_state) {
  Prefill(session, prefix, save_state);
  std::vector<float> rows;
  for (std::uint32_t row = 0; row < count; ++row) {
    (void)session.executor->ForwardToken(session.ids[prefix + row],
                                         prefix + row, true);
    const auto logits = session.executor->CopyLastLogits();
    rows.insert(rows.end(), logits.begin(), logits.end());
  }
  if (save_state)
    session.executor->FinishVerification();
  return rows;
}

std::span<const float> DecodeNext(Session& session, std::uint32_t position) {
  (void)session.executor->ForwardToken(session.ids[position], position, true);
  return session.executor->CopyLastLogits();
}

// The first layer whose output differs between decode and verification.
void Localize(Session& session, const Scenario& scenario,
              const Context& context) {
  for (std::size_t first = 0; first < context.layers; first += kTapsPerPass) {
    std::vector<std::uint32_t> taps;
    for (std::size_t layer = first;
         layer < std::min(context.layers, first + kTapsPerPass); ++layer)
      taps.push_back(static_cast<std::uint32_t>(layer));
    const std::size_t width = taps.size() * context.hidden;
    Prefill(session, scenario.prefix, true, taps);
    std::vector<float> decoded;
    for (std::uint32_t row = 0; row < scenario.rows; ++row) {
      (void)session.executor->ForwardToken(session.ids[scenario.prefix + row],
                                           scenario.prefix + row, true);
      const auto features = session.executor->CopyLastHidden();
      decoded.insert(decoded.end(), features.begin(), features.end());
    }
    session.executor->FinishVerification();
    Prefill(session, scenario.prefix, true, taps);
    const gufo::hip::QwenGpuVerificationItem item{
        session.executor.get(),
        std::span(session.ids).subspan(scenario.prefix, scenario.rows),
        scenario.prefix, true};
    (void)Executor::ForwardVerificationBatch(std::span(&item, 1));
    const auto verified = session.executor->GetVerificationHiddenStates();
    session.executor->FinishVerification();
    if (decoded.size() != scenario.rows * width ||
        verified.size() != decoded.size())
      throw std::runtime_error("layer capture has a wrong shape");
    for (std::size_t tap = 0; tap < taps.size(); ++tap) {
      for (std::uint32_t row = 0; row < scenario.rows; ++row) {
        const std::size_t offset = (row * width) + (tap * context.hidden);
        if (SameBits(std::span(decoded).subspan(offset, context.hidden),
                     verified.subspan(offset, context.hidden)))
          continue;
        std::cout << scenario.name << ",first_differing_layer=" << taps[tap]
                  << ",row=" << row << '\n';
        return;
      }
    }
  }
}

// Verifies every session's chunk in one launch, in the given order, and
// leaves the sessions with their verification pending. `accepted[session]`
// rows carry the session's own tokens; the rest carry a wrong draft.
std::vector<std::vector<Token>> Verify(std::vector<Session>& sessions,
                                       std::span<const std::size_t> order,
                                       const Scenario& scenario,
                                       std::span<const std::uint32_t> accepted,
                                       std::size_t vocab) {
  std::vector<std::vector<Token>> inputs(sessions.size());
  std::vector<gufo::hip::QwenGpuVerificationItem> items;
  for (const std::size_t index : order) {
    auto& session = sessions[index];
    Prefill(session, scenario.prefix, true);
    for (std::uint32_t row = 0; row < scenario.rows; ++row) {
      const Token token = session.ids[scenario.prefix + row];
      inputs[index].push_back(
          row < accepted[index]
              ? token
              : static_cast<Token>((static_cast<std::uint64_t>(token) + 1) %
                                   vocab));
    }
  }
  for (const std::size_t index : order)
    items.push_back(
        {sessions[index].executor.get(), inputs[index], scenario.prefix, true});
  (void)Executor::ForwardVerificationBatch(items);
  return inputs;
}

void RunScenario(std::vector<Session>& all, const Scenario& scenario,
                 Context& context) {
  std::span<Session> used(all.data(), scenario.sessions);
  std::vector<Session>& sessions = all;
  const std::uint32_t prefix = scenario.prefix;
  const std::uint32_t rows = scenario.rows;
  const std::size_t vocab = context.vocab;
  const auto label = [&](const std::string& what, std::size_t session,
                         std::size_t row) {
    return scenario.name + ',' + what + ",session=" + std::to_string(session) +
           ",row=" + std::to_string(row);
  };

  // Decoding is repeatable, and SaveState does not change what it computes.
  for (std::size_t index = 0; index < used.size(); ++index) {
    auto& session = used[index];
    session.reference = Decode(session, prefix, rows + 1, true);
    Check(context, label("decode_repeat", index, 0), session.reference,
          Decode(session, prefix, rows + 1, true));
    Check(context, label("decode_without_save", index, 0), session.reference,
          Decode(session, prefix, rows + 1, false));
  }

  // Each rotation gives another session the leading (coordinator) role.
  std::vector<std::size_t> order(used.size());
  const std::vector<std::uint32_t> all_accepted(used.size(), rows);
  for (std::size_t rotation = 0; rotation < used.size(); ++rotation) {
    for (std::size_t index = 0; index < order.size(); ++index)
      order[index] = (index + rotation) % order.size();
    (void)Verify(sessions, order, scenario, all_accepted, vocab);
    for (std::size_t index = 0; index < used.size(); ++index) {
      auto& session = used[index];
      const auto logits = session.executor->GetVerificationLogits();
      Check(context,
            label("verify_leader" + std::to_string(order.front()), index, 0),
            std::span(session.reference).first(rows * vocab), logits);
      // The device row a sampler reads is the same row.
      const auto device = session.executor->CopyVerificationLogits(rows - 1);
      Check(context, label("verify_device_row", index, rows - 1),
            Row(session, context, rows - 1), device);
    }
    // Full accept: the next token decodes from the verified state.
    for (std::size_t index = 0; index < used.size(); ++index) {
      auto& session = used[index];
      session.executor->FinishVerification();
      Check(context, label("after_full_accept", index, rows),
            Row(session, context, rows), DecodeNext(session, prefix + rows));
    }
  }

  // Reject after `kept` committed inputs (the current token is always kept);
  // sessions alternate between rejecting and accepting everything.
  for (std::uint32_t kept = 1; kept < rows; ++kept) {
    std::vector<std::uint32_t> accepted(used.size(), rows);
    for (std::size_t index = 0; index < used.size(); index += 2)
      accepted[index] = kept;
    for (std::size_t index = 0; index < order.size(); ++index)
      order[index] = index;
    const auto inputs = Verify(sessions, order, scenario, accepted, vocab);
    for (std::size_t index = 0; index < used.size(); ++index) {
      auto& session = used[index];
      const std::uint32_t committed = accepted[index];
      Check(context, label("verify_before_reject", index, 0),
            std::span(session.reference).first(committed * vocab),
            session.executor->GetVerificationLogits().first(committed * vocab));
      session.executor->FinishVerification();
      if (committed < rows) {
        session.executor->RestoreState();
        session.executor->CommitVerificationChunk(
            std::span(inputs[index]).first(committed), prefix);
      }
      // The continuation, and the token after it, match plain decoding.
      for (std::uint32_t row = committed; row <= rows; ++row)
        Check(context,
              label("after_reject_kept" + std::to_string(kept), index, row),
              Row(session, context, row), DecodeNext(session, prefix + row));
    }
  }

  // A verification abandoned without a commit leaves a reusable session.
  {
    const std::vector<std::uint32_t> none(used.size(), 1);
    for (std::size_t index = 0; index < order.size(); ++index)
      order[index] = index;
    (void)Verify(sessions, order, scenario, none, vocab);
    for (std::size_t index = 0; index < used.size(); ++index)
      Check(context, label("reuse_after_cancel", index, 0),
            used[index].reference, Decode(used[index], prefix, rows + 1, true));
  }

  // A decode step batched across sessions.
  if (used.size() > 1) {
    for (auto& session : used)
      Prefill(session, prefix, false);
    for (std::uint32_t row = 0; row <= rows; ++row) {
      std::vector<gufo::hip::QwenGpuBatchItem> items;
      for (auto& session : used)
        items.push_back(
            {session.executor.get(), session.ids[prefix + row], prefix + row});
      (void)Executor::ForwardTokenBatch(items);
      for (std::size_t index = 0; index < used.size(); ++index)
        Check(context, label("decode_batch", index, row),
              Row(used[index], context, row),
              used[index].executor->CopyLastLogits());
    }
  }

  const bool ok = context.passed;
  if (!ok && used.size() == 1)
    Localize(used.front(), scenario, context);
  std::cout << scenario.name << ",prefix=" << prefix << ",rows=" << rows
            << ",sessions=" << used.size() << ',' << (ok ? "PASS" : "FAIL")
            << '\n';
}

std::vector<Token> ReadIds(const std::string& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot read " + path);
  std::vector<Token> ids;
  std::uint64_t value = 0;
  while (input >> value)
    ids.push_back(static_cast<Token>(value));
  if (!input.eof())
    throw std::runtime_error("malformed token ids in " + path);
  return ids;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc == 3 || argc > 8) {
    std::cerr << "Usage: qwen_ar_verify_parity_test [MODEL.gguf [PREFIX ROWS "
                 "IDS.txt...]]\n";
    return 2;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
  }
  try {
    gufo::models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(path, &error);
    if (!reader_owner)
      throw std::runtime_error("open: " + error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(reader_owner));
    const auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (!model)
      throw std::runtime_error("model: " + error);
    const auto& config = model->GetConfig();
    if (!config.IsMoe()) {
      std::cerr << "the model has no experts; skipping\n";
      return 77;
    }
    Context context{config.num_layers, config.hidden_size, config.vocab_size};

    std::vector<Session> sessions;
    std::vector<Scenario> scenarios;
    if (argc > 3) {
      const auto prefix = static_cast<std::uint32_t>(std::stoul(argv[2]));
      const auto rows = static_cast<std::uint32_t>(std::stoul(argv[3]));
      for (int index = 4; index < argc; ++index)
        sessions.push_back({ReadIds(argv[index]), nullptr, {}});
      for (std::size_t count = 1; count <= sessions.size(); count *= 2)
        scenarios.push_back(
            {"file-c" + std::to_string(count), prefix, rows, count});
    } else {
      for (std::uint64_t seed = 0; seed < 4; ++seed)
        sessions.push_back({History(2200, seed), nullptr, {}});
      scenarios = {
          // The original counterexample, then the same position shared with
          // other sessions.
          {"p15-r2-c1", 15, 2, 1},
          {"p15-r2-c2", 15, 2, 2},
          {"p15-r2-c4", 15, 2, 4},
          {"p1-r2-c1", 1, 2, 1},
          // The widest chunk one session may verify.
          {"p16-r8-c1", 16, 8, 1},
          {"p16-r4-c4", 16, 4, 4},
          // Past the split-K decode attention threshold (128 positions).
          {"p200-r2-c1", 200, 2, 1},
          {"p200-r3-c4", 200, 3, 4},
          // Past the optimized attention threshold (1024 positions).
          {"p1100-r2-c1", 1100, 2, 1},
          {"p1100-r2-c4", 1100, 2, 4},
          // Past one full prefill chunk.
          {"p2100-r2-c2", 2100, 2, 2},
      };
    }
    for (auto& session : sessions) {
      for (const Token id : session.ids)
        if (id >= config.vocab_size)
          throw std::runtime_error("token id outside the vocabulary");
      session.executor = Executor::Create(model, &error, kContext);
      if (!session.executor)
        throw std::runtime_error("executor: " + error);
    }
    for (const auto& scenario : scenarios) {
      if (scenario.rows == 0 || scenario.rows > 8 || scenario.prefix == 0 ||
          scenario.sessions > sessions.size())
        throw std::runtime_error(scenario.name + " is out of range");
      for (std::size_t index = 0; index < scenario.sessions; ++index)
        if (sessions[index].ids.size() < scenario.prefix + scenario.rows + 1 ||
            scenario.prefix + scenario.rows + 1 > kContext)
          throw std::runtime_error(scenario.name + " needs a longer history");
      Context local = context;
      local.passed = true;
      RunScenario(sessions, scenario, local);
      context.passed = context.passed && local.passed;
    }
    return context.passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_ar_verify_parity_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  return 77;
#endif
}
