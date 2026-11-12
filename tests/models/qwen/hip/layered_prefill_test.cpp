// Layered prefill on real weights. A prompt's passes split into layer groups
// (QwenGpuExecutor::SetPrefillYield), with other sessions decoding between
// them, must leave every token and logit of all sessions bitwise unchanged
// against the same sessions run alone with unsplit passes.
//
// qwen_layered_prefill_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it. A model with an embedded MTP head also runs the
// speculative mode, with one and with two batched decoders. One more case
// lets a further session prefill a short prompt inside a yield, as serving
// does for a short request that arrives during a pass.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::tokenization::TokenId;
using Verifier = gufo::speculative::SpeculativeVerifier;

constexpr std::uint32_t kContext = 16384;
constexpr std::size_t kPassTokens = 4096;
constexpr std::size_t kPromptSteps = 8;

// Tokens and logit bytes of one session, in order.
struct Trace {
  std::vector<std::uint64_t> items;

  void Add(TokenId token, std::span<const float> logits) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](const void* data, std::size_t bytes) {
      const auto* p = static_cast<const unsigned char*>(data);
      for (std::size_t i = 0; i < bytes; ++i)
        hash = (hash ^ p[i]) * 1099511628211ULL;
    };
    mix(&token, sizeof(token));
    mix(logits.data(), logits.size_bytes());
    items.push_back(hash);
  }
};

struct Session {
  std::unique_ptr<gufo::hip::QwenGpuExecutor> executor;
  std::unique_ptr<Verifier> verifier;  // speculative mode only
  gufo::sampling::SamplerState sampler = gufo::sampling::SamplerState();
  std::vector<TokenId> sequence;
  std::uint32_t position{0};
  TokenId frontier{0};
  bool stopped{false};
};

std::string Ledger(std::size_t entries) {
  std::string text = "Inventory ledger.\n";
  for (std::size_t i = 0; i < entries; ++i)
    text += "Entry " + std::to_string(i) + " holds " +
            std::to_string(i * 7919 % 1009) + " units of part " +
            std::to_string(i * 31 % 97) + ".\n";
  return text + "Which part does entry 42 hold? Answer briefly.";
}

Session Create(const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
               const std::shared_ptr<const gufo::hip::QwenMtpGpuModel>& mtp) {
  std::string error;
  Session session;
  session.executor =
      gufo::hip::QwenGpuExecutor::Create(model, &error, kContext);
  if (!session.executor)
    throw std::runtime_error("executor: " + error);
  if (mtp) {
    auto draft = gufo::hip::QwenMtpGpuDraftBackend::Create(
        mtp,
        gufo::hip::QwenMtpGpuDraftConfig{
            .max_context = kContext,
            .max_draft_tokens = 1,
            .vision_input = &session.executor->VisionInput(),
        },
        &error);
    if (!draft)
      throw std::runtime_error("MTP session: " + error);
    // The options serving uses for an MTP head of depth one.
    gufo::speculative::SpeculativeOptions options;
    options.max_draft_tokens = 1;
    options.min_draft_tokens = 1;
    options.initial_draft_tokens = 1;
    options.use_batched_verification = true;
    options.retain_frontier_logits = true;
    options.enable_adaptive_draft_length = false;
    session.verifier = std::make_unique<Verifier>(*session.executor,
                                                  std::move(draft), options);
  }
  return session;
}

// Prefills `prompt` from an empty state in full-width passes, each split into
// `groups` layer groups with `yield` between them, and records the frontier.
void Prefill(Session& session, std::span<const TokenId> prompt,
             std::size_t groups, const std::function<void()>& yield,
             Trace& trace) {
  if (!session.verifier)
    session.executor->Reset();
  session.executor->SetPrefillYield(groups, yield);
  for (std::size_t offset = 0; offset < prompt.size(); offset += kPassTokens) {
    const auto pass =
        prompt.subspan(offset, std::min(kPassTokens, prompt.size() - offset));
    const bool last = offset + pass.size() == prompt.size();
    const auto position = static_cast<std::uint32_t>(offset);
    if (!session.verifier)
      session.frontier =
          session.executor->ForwardPromptBatch(pass, position, last);
    else if (offset == 0)
      session.frontier = session.verifier->Prime(pass, last);
    else
      session.frontier = session.verifier->ExtendPrompt(pass, position, last);
  }
  session.executor->SetPrefillYield(1, {});
  session.sequence.assign(prompt.begin(), prompt.end());
  session.position = static_cast<std::uint32_t>(prompt.size());
  session.sampler = gufo::sampling::SamplerState({}, session.sequence);
  session.stopped = false;
  trace.Add(session.frontier, session.executor->CopyLastLogits());
  if (session.verifier)
    session.sequence.push_back(session.frontier);
}

void Finish(Session& session, const Verifier::StepResult& step, Trace& trace) {
  for (const TokenId token : step.emitted_tokens) {
    session.sequence.push_back(token);
    session.sampler.Accept(token);
    ++session.position;
    trace.Add(token, {});
  }
  trace.Add(step.next_token, step.next_token_logits);
  session.frontier = step.next_token;
  session.stopped = step.hit_eos || session.position + 8 >= kContext;
}

Verifier::StepRequest Request(Session& session) {
  return {*session.verifier,
          session.sequence,
          session.position,
          session.frontier,
          session.executor->GetTokenizer().GetEosTokenId(),
          8,
          session.sampler};
}

// One greedy decode step: a token, or one speculative verification.
void Step(Session& session, Trace& trace) {
  if (session.stopped)
    return;
  if (!session.verifier) {
    const TokenId token = session.frontier;
    session.frontier = session.executor->ForwardToken(token, session.position);
    ++session.position;
    trace.Add(token, session.executor->CopyLastLogits());
    return;
  }
  const auto request = Request(session);
  Finish(session,
         session.verifier->VerifyStep(
             request.sequence, request.position, request.current_token,
             request.eos_id, request.max_emitted_tokens, session.sampler),
         trace);
}

// One batched verification of two speculative sessions, as serving runs them.
void StepPair(Session& first, Session& second, Trace& first_trace,
              Trace& second_trace) {
  if (first.stopped || second.stopped)
    return;
  const Verifier::StepRequest requests[] = {Request(first), Request(second)};
  auto steps = Verifier::VerifyBatch(requests);
  Finish(first, steps[0], first_trace);
  Finish(second, steps[1], second_trace);
}

bool Same(const Trace& got, const Trace& want, std::size_t count,
          const std::string& what) {
  if (got.items.size() < count || want.items.size() < count) {
    std::cerr << what << ": too few items (" << got.items.size() << " and "
              << want.items.size() << ", need " << count << ")\n";
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (got.items[i] != want.items[i]) {
      std::cerr << what << ": differs at item " << i << " of " << count << '\n';
      return false;
    }
  }
  return true;
}

struct Lone {
  Trace prompt;
  Trace decoder;
  Trace second_decoder;
};

// The prompt and the decoders, each alone and unsplit. `pair` steps both
// decoders in one batch.
Lone RunLone(Session& prompt_session, Session& decoder, Session& second,
             std::span<const TokenId> prompt, std::span<const TokenId> first,
             std::span<const TokenId> other, bool pair, std::size_t steps) {
  Lone lone;
  Prefill(prompt_session, prompt, 1, {}, lone.prompt);
  for (std::size_t i = 0; i < kPromptSteps; ++i)
    Step(prompt_session, lone.prompt);
  Prefill(decoder, first, 1, {}, lone.decoder);
  if (pair)
    Prefill(second, other, 1, {}, lone.second_decoder);
  for (std::size_t i = 0; i < steps; ++i) {
    if (pair)
      StepPair(decoder, second, lone.decoder, lone.second_decoder);
    else
      Step(decoder, lone.decoder);
  }
  return lone;
}

bool RunMode(const std::string& mode,
             const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
             const std::shared_ptr<const gufo::hip::QwenMtpGpuModel>& mtp,
             bool pair) {
  auto prompt_session = Create(model, mtp);
  auto decoder = Create(model, mtp);
  auto second = pair ? Create(model, mtp) : Session{};
  const auto& tokenizer = model->GetTokenizer();
  // Three passes: two full ones and a tail.
  const auto prompt = tokenizer.Encode(Ledger(520));
  if (prompt.size() <= 2 * kPassTokens || prompt.size() + 64 >= kContext)
    throw std::runtime_error("prompt has " + std::to_string(prompt.size()) +
                             " tokens; expected just over two passes");
  const auto first = tokenizer.Encode(
      "Write a C function that reverses a singly linked list in place, then "
      "explain its complexity.");
  const auto other = tokenizer.Encode(
      "List the planets of the solar system with one fact about each.");
  const std::size_t passes = (prompt.size() + kPassTokens - 1) / kPassTokens;
  const std::size_t layers = model->GetConfig().num_layers;
  // Enough decode steps for a yield before every layer of every pass.
  const std::size_t steps = passes * layers;

  const auto lone = RunLone(prompt_session, decoder, second, prompt, first,
                            other, pair, steps);
  const auto again = RunLone(prompt_session, decoder, second, prompt, first,
                             other, pair, steps);
  if (!Same(again.prompt, lone.prompt, lone.prompt.items.size(),
            mode + " lone prompt repeat") ||
      !Same(again.decoder, lone.decoder, lone.decoder.items.size(),
            mode + " lone decoder repeat") ||
      !Same(again.second_decoder, lone.second_decoder,
            lone.second_decoder.items.size(),
            mode + " lone second decoder repeat")) {
    // Not a layered-prefill result: the reference itself is not repeatable.
    std::cout << mode << ",UNSTABLE_REFERENCE\n";
    return false;
  }

  bool passed = true;
  for (const std::size_t groups :
       {std::size_t{8}, std::size_t{16}, std::size_t{3}, layers}) {
    Trace decoder_trace, second_trace, prompt_trace;
    Prefill(decoder, first, 1, {}, decoder_trace);
    if (pair)
      Prefill(second, other, 1, {}, second_trace);
    std::size_t yields = 0;
    Prefill(
        prompt_session, prompt, groups,
        [&] {
          ++yields;
          if (pair)
            StepPair(decoder, second, decoder_trace, second_trace);
          else
            Step(decoder, decoder_trace);
        },
        prompt_trace);
    for (std::size_t i = 0; i < kPromptSteps; ++i)
      Step(prompt_session, prompt_trace);
    const std::string name = mode + "," + std::to_string(groups) + "_groups";
    const std::size_t expected = passes * (groups - 1);
    bool ok = yields == expected;
    if (!ok)
      std::cerr << name << ": " << yields << " yields, expected " << expected
                << '\n';
    ok = Same(prompt_trace, lone.prompt, lone.prompt.items.size(),
              name + " prompt") &&
         ok;
    ok = Same(decoder_trace, lone.decoder, decoder_trace.items.size(),
              name + " decoder") &&
         ok;
    if (pair)
      ok = Same(second_trace, lone.second_decoder, second_trace.items.size(),
                name + " second decoder") &&
           ok;
    ok = decoder_trace.items.size() > yields / 2 && ok;
    std::cout << name << ",yields=" << yields
              << ",prompt_items=" << prompt_trace.items.size()
              << ",decoder_items=" << decoder_trace.items.size() << ','
              << (ok ? "PASS" : "FAIL") << '\n';
    passed = passed && ok;
  }
  // A guest prefills a short prompt in the third yield and decodes in the
  // following ones; the first decoder keeps stepping.
  {
    auto guest = Create(model, mtp);
    const auto guest_prompt = tokenizer.Encode(Ledger(18));
    Trace guest_lone;
    Prefill(guest, guest_prompt, 1, {}, guest_lone);
    for (std::size_t i = 0; i < steps; ++i)
      Step(guest, guest_lone);
    Trace decoder_trace, second_trace, guest_trace, prompt_trace;
    Prefill(decoder, first, 1, {}, decoder_trace);
    if (pair)
      Prefill(second, other, 1, {}, second_trace);
    std::size_t yields = 0;
    Prefill(
        prompt_session, prompt, 8,
        [&] {
          ++yields;
          if (yields == 3)
            Prefill(guest, guest_prompt, 1, {}, guest_trace);
          else if (yields > 3)
            Step(guest, guest_trace);
          if (pair)
            StepPair(decoder, second, decoder_trace, second_trace);
          else
            Step(decoder, decoder_trace);
        },
        prompt_trace);
    for (std::size_t i = 0; i < kPromptSteps; ++i)
      Step(prompt_session, prompt_trace);
    const std::string name = mode + ",8_groups_guest_prefill";
    bool ok = Same(prompt_trace, lone.prompt, lone.prompt.items.size(),
                   name + " prompt");
    ok = Same(decoder_trace, lone.decoder, decoder_trace.items.size(),
              name + " decoder") &&
         ok;
    ok = Same(guest_trace, guest_lone, guest_trace.items.size(),
              name + " guest") &&
         ok;
    ok = guest_trace.items.size() > 8 && ok;
    std::cout << name << ",yields=" << yields
              << ",guest_tokens=" << guest_prompt.size()
              << ",guest_items=" << guest_trace.items.size() << ','
              << (ok ? "PASS" : "FAIL") << '\n';
    passed = passed && ok;
  }
  return passed;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_layered_prefill_test [MODEL.gguf]\n";
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
    gufo::models::qwen38_flash_next::rocm::RegisterQwenNpuFfn();
    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(path, &error);
    if (!reader_owner)
      throw std::runtime_error("open: " + error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(reader_owner));
    const auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (!model)
      throw std::runtime_error("model: " + error);
    bool passed = RunMode("AR", model, nullptr, false);
    std::string mtp_error;
    const auto mtp =
        gufo::hip::QwenMtpGpuModel::CreateEmbedded(model, &mtp_error);
    if (mtp) {
      passed = RunMode("MTP1", model, mtp, false) && passed;
      passed = RunMode("MTP1_batched_2", model, mtp, true) && passed;
    } else {
      std::cout << "MTP1,SKIPPED (" << mtp_error << ")\n";
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_layered_prefill_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen layered prefill test.\n";
  return 77;
#endif
}
