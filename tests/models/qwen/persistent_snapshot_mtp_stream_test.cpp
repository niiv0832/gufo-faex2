// Real MTP snapshots must not copy their full owned host payload while
// streaming. A separate tiny speculative request exercises pending-token
// persistence and continuation. No external model or mock draft is used.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/qwen_text_model_runner.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/sampling.hpp"
#include "src/core/snapshot_stream.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#define GUFO_TEST_DEFINE_SNAPSHOT_ALLOCATION_OPERATORS
#include "tests/models/qwen/support/snapshot_allocation_guard.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

namespace {
using namespace gufo;
using Token = server::TextRunnerToken;
using test::snapshot_alloc::AllocationLedger;
using test::snapshot_alloc::AllocationScope;
using test::snapshot_alloc::kChunkLimit;
using test::snapshot_alloc::kLiveLimit;

constexpr std::uint32_t kLargePosition = 11264;
constexpr std::uint32_t kModelContext = 11280;
// 4-byte position + FP32/FP16 K/V + 128 floats + primed byte.
constexpr std::size_t kLargeMtpBytes = 17302021;
// SHA-256 of the names in these comments, without a newline. These identify
// fixed synthetic fixtures; neither value claims to hash an external GGUF.
// gufo-qwen-mtp-persistent-stream-target-fixture-v1
constexpr char kTargetFingerprint[] =
    "8d21fe26f4a6ca3886b2f975662adc26dc119b466614cd2bf0145cfdb1becf9c";
// gufo-qwen-mtp-persistent-stream-head-fixture-v1
constexpr char kMtpFingerprint[] =
    "fa22e3aa6f6152dc9c696db702972fcc31773beada4c7baae924e4eb3487c665";

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

void Stage(std::string_view stage) noexcept {
  (void)::write(STDOUT_FILENO, "setup_stage=", 12);
  (void)::write(STDOUT_FILENO, stage.data(), stage.size());
  (void)::write(STDOUT_FILENO, "\n", 1);
}

template<typename T>
T ReadLe(std::span<const std::uint8_t> bytes, std::size_t offset) {
  Require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset,
          "independent persistent reader exceeds the payload");
  T value = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i)
    value |= static_cast<T>(bytes[offset + i]) << (8 * i);
  return value;
}

template<typename T>
void AppendLe(std::vector<std::uint8_t>& bytes, T value) {
  for (std::size_t i = 0; i < sizeof(T); ++i)
    bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

// Reuse immutable target tensors for the synthetic head. The fusion matrix
// and gated Q projection have equal element counts only for this fixture.
std::shared_ptr<const hip::QwenGpuModel> MakeSyntheticTarget() {
  auto config = models::qwen::make_small_qwen_config();
  config.context_length = kModelContext;
  config.mtp_num_layers = 1;
  Require(config.IsValidQwen() && config.hidden_size == 128 &&
              config.AttentionSize() == 128 &&
              config.num_key_value_heads == 2 && config.head_dim == 64 &&
              config.num_layers == 4 && !config.IsMoe(),
          "synthetic MTP configuration differs from the independent fixture");
  Stage("mtp_synthetic_weights_begin");
  auto synthetic = models::qwen::build_synthetic_qwen_weights(config);
  const auto bytes = synthetic.pool.size() * sizeof(float);
  void* device = nullptr;
  HIP_CHECK(hipMalloc(&device, bytes));
  struct DeviceCleanup {
    void* pointer;
    ~DeviceCleanup() {
      if (pointer != nullptr)
        (void)hipFree(pointer);
    }
  } cleanup{device};
  HIP_CHECK(
      hipMemcpy(device, synthetic.pool.data(), bytes, hipMemcpyHostToDevice));
  Stage("mtp_device_weights_copied");
  auto weights = std::move(synthetic.weights);
  const auto remap = [&](models::QwenTensorRef& tensor) {
    if (tensor.data == nullptr)
      return;
    const auto offset =
        static_cast<const float*>(tensor.data) - synthetic.pool.data();
    Require(offset >= 0 &&
                static_cast<std::size_t>(offset) < synthetic.pool.size() &&
                tensor.num_elements <=
                    synthetic.pool.size() - static_cast<std::size_t>(offset),
            "synthetic MTP tensor exceeds its owned storage");
    tensor.data = static_cast<const float*>(device) + offset;
  };
  remap(weights.token_embd);
  remap(weights.output_norm);
  remap(weights.output);
  for (auto& layer : weights.layers) {
    for (auto member : {&models::QwenLayerWeights::attn_norm,
                        &models::QwenLayerWeights::ffn_norm,
                        &models::QwenLayerWeights::attn_q,
                        &models::QwenLayerWeights::attn_k,
                        &models::QwenLayerWeights::attn_v,
                        &models::QwenLayerWeights::attn_output,
                        &models::QwenLayerWeights::attn_q_norm,
                        &models::QwenLayerWeights::attn_k_norm,
                        &models::QwenLayerWeights::attn_qkv,
                        &models::QwenLayerWeights::attn_gate,
                        &models::QwenLayerWeights::ssm_a,
                        &models::QwenLayerWeights::ssm_conv1d,
                        &models::QwenLayerWeights::ssm_dt,
                        &models::QwenLayerWeights::ssm_alpha,
                        &models::QwenLayerWeights::ssm_beta,
                        &models::QwenLayerWeights::ssm_norm,
                        &models::QwenLayerWeights::ssm_out,
                        &models::QwenLayerWeights::ffn_gate,
                        &models::QwenLayerWeights::ffn_up,
                        &models::QwenLayerWeights::ffn_down}) {
      remap(layer.*member);
    }
  }
  const auto& attention = weights.layers[3];
  Require(attention.is_full_attention &&
              attention.attn_q.type == core::GgmlType::kF32 &&
              attention.attn_q.num_elements == 128U * 256U &&
              weights.output_norm.num_elements == 128,
          "synthetic embedded MTP tensor shapes are invalid");
  weights.mtp = models::QwenEmbeddedMtp{
      .embedding_norm = weights.output_norm,
      .hidden_norm = weights.output_norm,
      .fusion_projection = attention.attn_q,
      .shared_head_norm = weights.output_norm,
      .layer = attention,
  };
  std::vector<std::string> vocabulary;
  for (std::uint32_t i = 0; i < config.vocab_size; ++i)
    vocabulary.push_back("token_" + std::to_string(i));
  std::string error;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromVocabulary(
      vocabulary, {}, {}, &error);
  Require(tokenizer != nullptr, error);
  std::vector<hip::QwenGpuWeightRegion> regions{{
      .device_data = device,
      .size = bytes,
      .owns_device_allocation = true,
  }};
  auto target = std::make_shared<hip::QwenGpuModel>(
      nullptr, std::move(weights), std::move(tokenizer), std::move(regions));
  cleanup.pointer = nullptr;
  Stage("mtp_target_model_ready");
  return target;
}

std::vector<std::uint8_t> SeededMtpPayload() {
  static_assert(std::endian::native == std::endian::little);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(kLargeMtpBytes);
  AppendLe(bytes, kLargePosition);
  // FP32: plane, head, token, channel. Every boundary has distinct values.
  for (std::uint32_t plane = 0; plane < 2; ++plane) {
    for (std::uint32_t head = 0; head < 2; ++head) {
      for (std::uint32_t token = 0; token < kLargePosition; ++token) {
        for (std::uint32_t channel = 0; channel < 64; ++channel) {
          const float value =
              static_cast<float>(1 + plane * 4096 + head * 2048 +
                                 (token % 31) * 64 + channel) /
              16384.0F;
          AppendLe(bytes, std::bit_cast<std::uint32_t>(value));
        }
      }
    }
  }
  // FP16: plane, token, head, channel. All patterns are finite binary16.
  for (std::uint32_t plane = 0; plane < 2; ++plane) {
    for (std::uint32_t token = 0; token < kLargePosition; ++token) {
      for (std::uint32_t head = 0; head < 2; ++head) {
        for (std::uint32_t channel = 0; channel < 64; ++channel) {
          const auto bits =
              static_cast<std::uint16_t>(0x3000 + plane * 2048 + head * 1024 +
                                         (token % 13) * 64 + channel);
          AppendLe(bytes, bits);
        }
      }
    }
  }
  for (std::uint32_t i = 0; i < 128; ++i)
    AppendLe(bytes, std::bit_cast<std::uint32_t>((i + 1) / 256.0F));
  bytes.push_back(1);
  Require(bytes.size() == kLargeMtpBytes && bytes.size() > kLiveLimit,
          "independent MTP payload must exceed the complete heap bound");
  return bytes;
}

void CheckLedger(const AllocationLedger& ledger) {
  Require(ledger.rejection == nullptr && ledger.live_bytes == 0 &&
              ledger.peak_bytes <= kLiveLimit &&
              ledger.largest_bytes <= kChunkLimit,
          "stream leaked, suppressed a rejected allocation, or exceeded cap");
}

template<typename Stream>
std::vector<std::uint8_t> CheckBoundedStream(
    const Stream& stream, std::span<const std::uint8_t> expected,
    std::string_view label) {
  std::vector<std::uint8_t> actual(expected.size(), 0xA5);
  std::size_t written = 0;
  std::size_t callbacks = 0;
  std::size_t largest = 0;
  const core::PersistentSnapshotSink sink = [&](auto bytes) {
    Require(!bytes.empty() && bytes.size() <= kChunkLimit &&
                bytes.size() <= actual.size() - written,
            "stream callback is empty, oversized, or exceeds payload");
    std::memcpy(actual.data() + written, bytes.data(), bytes.size());
    written += bytes.size();
    largest = std::max(largest, bytes.size());
    ++callbacks;
  };
  AllocationLedger ledger;
  try {
    const AllocationScope measured(ledger);
    stream(sink);
  } catch (const std::bad_alloc&) {
    std::cerr << "stream_allocation_rejected case=" << label << " reason="
              << (ledger.rejection != nullptr ? ledger.rejection : "unknown")
              << " bytes=" << ledger.rejected_bytes
              << " payload=" << expected.size() << '\n';
    throw;
  }
  CheckLedger(ledger);
  Require(written == expected.size() &&
              std::equal(actual.begin(), actual.end(), expected.begin()),
          "stream bytes differ from the independent/reference payload");
  Require(callbacks > 1, "fixture did not exercise multiple stream chunks");
  std::cout << "bounded_stream case=" << label << " bytes=" << written
            << " callbacks=" << callbacks << " largest_chunk=" << largest
            << " largest_allocation=" << ledger.largest_bytes
            << " peak_live_bytes=" << ledger.peak_bytes << '\n';
  return actual;
}

struct SinkFailure {};

void CheckActualMtpLegacyFallback(
    const speculative::IDraftBackendSnapshot& snapshot) {
  std::size_t callbacks = 0;
  const core::PersistentSnapshotSink sink = [&](auto) { ++callbacks; };
  AllocationLedger ledger;
  bool rejected = false;
  try {
    const AllocationScope measured(ledger);
    // Invoke the real compatibility implementation on the actual snapshot,
    // bypassing only its new override. Callback chunking after a full copy
    // cannot satisfy the allocation contract.
    using DraftSnapshot = speculative::IDraftBackendSnapshot;
    snapshot.DraftSnapshot::StreamPersistent(sink);
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  Require(rejected && callbacks == 0 && ledger.rejection != nullptr &&
              std::string_view(ledger.rejection) == "single-allocation" &&
              ledger.rejected_bytes == kLargeMtpBytes && ledger.live_bytes == 0,
          "actual MTP legacy full-copy negative control did not fail closed");
  std::cout << "negative_control=actual_mtp_legacy_fallback rejected_bytes="
            << ledger.rejected_bytes << '\n';
}

template<typename Stream>
void CheckThrowingSink(const Stream& stream,
                       std::span<const std::uint8_t> expected) {
  std::size_t callbacks = 0;
  std::size_t observed = 0;
  const core::PersistentSnapshotSink sink = [&](auto bytes) {
    Require(!bytes.empty() && bytes.size() <= kChunkLimit &&
                bytes.size() <= expected.size() - observed,
            "throwing sink received an invalid chunk");
    if (++callbacks == 2)
      throw SinkFailure{};
    Require(std::equal(bytes.begin(), bytes.end(), expected.begin() + observed),
            "bytes emitted before failure differ from the original");
    observed += bytes.size();
  };
  AllocationLedger ledger;
  bool caught = false;
  try {
    const AllocationScope measured(ledger);
    stream(sink);
  } catch (const SinkFailure&) {
    caught = true;
  }
  Require(
      caught && callbacks == 2 && observed > 0 && observed < expected.size(),
      "stream did not propagate a partial sink failure");
  CheckLedger(ledger);
}

// A legacy fallback on this actual snapshot allocates 17,302,021 bytes.
// Chunking callbacks after that allocation cannot pass the heap observer.
void TestLargeActualMtpSnapshot(
    const std::shared_ptr<const hip::QwenMtpGpuModel>& mtp) {
  Stage("large_mtp_backend_begin");
  Require(!hip::QwenMtpGpuExecutor::KeepsHalfKvOnly(mtp->GetConfig()),
          "fixture must exercise both FP32 and FP16 MTP planes");
  std::string error;
  auto backend = hip::QwenMtpGpuDraftBackend::Create(
      mtp, {.max_context = kModelContext, .max_draft_tokens = 1}, &error);
  Require(backend != nullptr, error);
  Require(backend->Name() == "QwenMtpGpuDraftBackend",
          "fixture did not create the actual MTP backend");
  auto expected = SeededMtpPayload();
  Require(
      hip::QwenMtpGpuExecutor::StateBytesAt(mtp->GetConfig(), kLargePosition) +
              513 ==
          kLargeMtpBytes,
      "production MTP state sizing disagrees with independent fixture");
  backend->RestorePersistentSnapshot(expected);
  Require(backend->Primed() && backend->NextPosition() == kLargePosition &&
              backend->SnapshotPayloadBytes() == kLargeMtpBytes,
          "actual MTP backend did not restore the seeded GPU state");
  Stage("large_mtp_gpu_state_restored");
  auto snapshot = backend->Snapshot();
  Require(snapshot->PersistentPayloadBytes() == kLargeMtpBytes,
          "actual MTP snapshot does not exceed the allocation cap");
  std::vector<std::uint8_t> reference(kLargeMtpBytes);
  Require(snapshot->SerializePersistent(reference) == reference.size() &&
              reference == expected,
          "actual MTP snapshot changed the seeded planes/hidden/primed byte");
  backend.reset();
  Stage("large_mtp_snapshot_reference_ready");
  CheckActualMtpLegacyFallback(*snapshot);
  const auto stream = [&](const core::PersistentSnapshotSink& sink) {
    snapshot->StreamPersistent(sink);
  };
  const auto streamed = CheckBoundedStream(stream, expected, "large-mtp");
  CheckThrowingSink(stream, expected);
  (void)CheckBoundedStream(stream, expected, "large-mtp-after-throw");

  auto reloaded = hip::QwenMtpGpuDraftBackend::Create(
      mtp, {.max_context = kModelContext, .max_draft_tokens = 1}, &error);
  Require(reloaded != nullptr, error);
  reloaded->RestorePersistentSnapshot(streamed);
  Require(reloaded->NextPosition() == kLargePosition && reloaded->Primed(),
          "streamed MTP state lost its position or primed flag");
  auto round_trip = reloaded->Snapshot();
  std::fill(reference.begin(), reference.end(), 0x5A);
  Require(round_trip->SerializePersistent(reference) == reference.size() &&
              reference == expected,
          "streamed MTP bytes do not round-trip through the actual executor");
  std::cout << "case_pass=large_actual_mtp payload_bytes=" << kLargeMtpBytes
            << " state_source=seeded_restore inference_quality=not_exercised\n";
}

std::vector<std::uint8_t> SerializeRunnerSnapshot(
    const server::TextModelRunner& runner,
    const server::TextRunnerSnapshot& snapshot) {
  std::vector<std::uint8_t> bytes(
      runner.PersistentSnapshotPayloadBytes(snapshot));
  Require(runner.SerializePersistentSnapshot(snapshot, bytes) == bytes.size(),
          "direct runner serializer returned an invalid byte count");
  return bytes;
}

void CheckPendingHeader(std::span<const std::uint8_t> bytes,
                        std::span<const Token> history, std::size_t position) {
  constexpr std::array<std::uint8_t, 8> magic{'G', 'Q', 'W', 'R',
                                              'U', 'N', '0', '1'};
  Require(bytes.size() >= 112 &&
              std::equal(magic.begin(), magic.end(), bytes.begin()) &&
              ReadLe<std::uint32_t>(bytes, 8) == 5 &&
              ReadLe<std::uint32_t>(bytes, 12) == 112 &&
              ReadLe<std::uint64_t>(bytes, 16) == position &&
              ReadLe<std::uint64_t>(bytes, 48) == bytes.size(),
          "independent outer persistent header validation failed");
  const auto count = ReadLe<std::uint32_t>(bytes, 64);
  Require(count > 0 && count <= 9 && position <= history.size() &&
              history.size() - position == count &&
              ReadLe<std::uint32_t>(bytes, 28) == 3 &&
              ReadLe<std::uint32_t>(bytes, 68) == 0 &&
              ReadLe<std::uint32_t>(bytes, 108) == 0,
          "actual speculative decode did not retain a valid pending frontier");
  for (std::size_t i = 0; i < 9; ++i) {
    Require(ReadLe<std::uint32_t>(bytes, 72 + 4 * i) ==
                (i < count ? history[position + i] : 0),
            "pending token or unused outer-header slot changed");
  }
  const auto gpu_bytes = ReadLe<std::uint64_t>(bytes, 40);
  const auto logits = ReadLe<std::uint64_t>(bytes, 32);
  const auto verifier_bytes = ReadLe<std::uint64_t>(bytes, 56);
  Require(gpu_bytes <= bytes.size() - 112 && logits == 256,
          "unexpected target or logits section size");
  const auto verifier_offset = 112 + gpu_bytes + logits * 4;
  Require(verifier_offset <= bytes.size() && verifier_bytes >= 96 &&
              verifier_bytes == bytes.size() - verifier_offset,
          "verifier section exceeds the declared persistent payload");
  const auto verifier = bytes.subspan(verifier_offset, verifier_bytes);
  Require(ReadLe<std::uint32_t>(verifier, 8) == 2 &&
              ReadLe<std::uint32_t>(verifier, 12) == 96 &&
              ReadLe<std::uint64_t>(verifier, 16) > 0 &&
              ReadLe<std::uint64_t>(verifier, 32) > 0,
          "pending snapshot does not contain an actual draft payload/round");
  std::cout << "pending_header count=" << count << " position=" << position
            << " verifier_bytes=" << verifier_bytes << '\n';
}

void CheckDecode(const server::TextDecodeStep& step) {
  if (step.failure)
    std::rethrow_exception(step.failure);
  Require(!step.stop && !step.selections.empty() && step.draft_tokens > 0 &&
              step.draft_rounds > 0,
          "actual MTP decode did not execute a speculative round");
}

void TestPendingActualMtpContinuation(
    const std::shared_ptr<const hip::QwenGpuModel>& target,
    const std::shared_ptr<const hip::QwenMtpGpuModel>& mtp) {
  Stage("pending_runner_begin");
  auto runner = server::CreateQwenTextRunner(
      target, {.max_context = 8,
               .mtp = mtp,
               .speculative_options = {.max_draft_tokens = 1,
                                       .min_draft_tokens = 1,
                                       .initial_draft_tokens = 1,
                                       .enable_adaptive_draft_length = false,
                                       .retain_frontier_logits = true},
               .model_artifact_fingerprint = kTargetFingerprint,
               .draft_model_artifact_fingerprint = kMtpFingerprint});
  auto state = runner->CreateState();
  state->SetStopAtEos(false);
  const std::array<Token, 2> prompt{1, 2};
  std::size_t offset = 0;
  while (offset < prompt.size()) {
    const auto step = runner->Prefill(*state, prompt, offset, prompt.size());
    Require(step.consumed_tokens > 0 &&
                step.consumed_tokens <= prompt.size() - offset,
            "actual MTP prefill made invalid progress");
    offset += step.consumed_tokens;
    Require(step.decode_ready == (offset == prompt.size()),
            "MTP prefill decode-ready boundary differs from prompt end");
  }
  Require(runner->CheckpointPosition(*state) == prompt.size(),
          "actual MTP prefill did not reach its checkpoint");
  Stage("pending_prefill_ready");
  sampling::SamplerState sampler({.temperature = 0.0F}, prompt);
  const auto first = runner->DecodeStep(*state, 4, sampler);
  CheckDecode(first);
  std::vector<Token> history(prompt.begin(), prompt.end());
  for (const auto& selection : first.selections)
    history.push_back(selection.token);
  const auto position = runner->CheckpointPosition(*state);
  auto snapshot = runner->Snapshot(*state);
  const auto reference = SerializeRunnerSnapshot(*runner, *snapshot);
  CheckPendingHeader(reference, history, position);
  Stage("pending_snapshot_reference_ready");
  const auto stream = [&](const core::PersistentSnapshotSink& sink) {
    runner->StreamPersistentSnapshot(*snapshot, sink);
  };
  const auto streamed = CheckBoundedStream(stream, reference, "pending-runner");
  CheckThrowingSink(stream, reference);
  (void)CheckBoundedStream(stream, reference, "pending-runner-after-throw");

  auto restored = runner->CreateState();
  restored->SetStopAtEos(false);
  runner->RestorePersistentSnapshot(*restored, streamed);
  auto restored_snapshot = runner->Snapshot(*restored);
  Require(SerializeRunnerSnapshot(*runner, *restored_snapshot) == reference,
          "persistent restore lost pending/MTP/target snapshot bytes");
  CheckPendingHeader(streamed, history, runner->CheckpointPosition(*restored));
  // Persistent restore does not reconstruct the request's logical sequence.
  // Exercise the real continuation entry point before consuming pending rows.
  const auto prefix = std::span<const Token>(history).first(position);
  Require(history.size() < 8 && history.size() > position,
          "fixture has no pending row or no room for continuation");
  for (auto* value : {state.get(), restored.get()}) {
    runner->PreparePrefixReuse(*value, prefix);
    auto consumed = position;
    while (consumed < history.size()) {
      const auto step =
          runner->Prefill(*value, history, consumed, history.size() - consumed);
      Require(step.consumed_tokens == 1,
              "pending resume must consume one published token at a time");
      consumed += step.consumed_tokens;
    }
    Require(runner->CheckpointPosition(*value) == history.size(),
            "pending continuation advanced to the wrong checkpoint");
  }
  Stage("pending_tokens_consumed");
  sampling::SamplerState original_sampler({.temperature = 0.0F}, history);
  sampling::SamplerState restored_sampler({.temperature = 0.0F}, history);
  const auto original_step = runner->DecodeStep(*state, 3, original_sampler);
  const auto restored_step = runner->DecodeStep(*restored, 3, restored_sampler);
  CheckDecode(original_step);
  CheckDecode(restored_step);
  Require(original_step.selections.size() == restored_step.selections.size() &&
              original_step.draft_tokens == restored_step.draft_tokens &&
              original_step.draft_accepted_tokens ==
                  restored_step.draft_accepted_tokens &&
              original_step.draft_rounds == restored_step.draft_rounds &&
              runner->CheckpointPosition(*state) ==
                  runner->CheckpointPosition(*restored),
          "reloaded pending continuation changed draft work or position");
  for (std::size_t i = 0; i < original_step.selections.size(); ++i) {
    Require(
        original_step.selections[i].token == restored_step.selections[i].token,
        "reloaded pending continuation changed its next tokens");
  }
  auto final_original = runner->Snapshot(*state);
  auto final_restored = runner->Snapshot(*restored);
  Require(SerializeRunnerSnapshot(*runner, *final_original) ==
              SerializeRunnerSnapshot(*runner, *final_restored),
          "pending continuation diverged in exact target/MTP state");
  std::cout << "case_pass=actual_mtp_pending first_drafts="
            << first.draft_tokens
            << " resumed_drafts=" << restored_step.draft_tokens << '\n';
}
}  // namespace

int main() {
  try {
    Stage("allocator_alignment_begin");
    test::snapshot_alloc::CheckAllocatorAlignment();
    Stage("allocator_alignment_ready");
    if (const auto gate =
            test::GateHipDevice(test::HipDeviceRequirement::kRequired,
                                "qwen_persistent_snapshot_mtp_stream_test");
        gate != test::kHipTestSuccess) {
      std::cerr << "MTP streaming and pending branches NOT_EXERCISED\n";
      return gate;
    }
    auto target = MakeSyntheticTarget();
    std::string error;
    auto mtp = hip::QwenMtpGpuModel::CreateEmbedded(target, &error);
    Require(mtp != nullptr, error);
    Stage("embedded_mtp_model_ready");
    TestLargeActualMtpSnapshot(mtp);
    TestPendingActualMtpContinuation(target, mtp);
    std::cout
        << "qwen_persistent_snapshot_mtp_stream_test: both cases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen_persistent_snapshot_mtp_stream_test: " << error.what()
              << '\n';
    return 1;
  }
}
