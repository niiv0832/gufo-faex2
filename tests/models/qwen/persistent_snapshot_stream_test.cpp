// A real Qwen runner must stream without materializing a complete host payload.
// Synthetic weights and a short prompt keep this independent of model files.
// The deliberately large recurrent state makes the old full-vector stream fail
// a thread-local C++ heap bound. This does not measure HIP allocations or RSS.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/qwen_text_model_runner.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#define GUFO_TEST_DEFINE_SNAPSHOT_ALLOCATION_OPERATORS
#include "tests/models/qwen/support/snapshot_allocation_guard.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

namespace {
using namespace gufo::test::snapshot_alloc;
// SHA-256 of "gufo-qwen-persistent-snapshot-stream-fixture-v1" (no newline).
// Names this fixed synthetic fixture; it is not an external GGUF content hash.
constexpr char kSyntheticArtifactFingerprint[] =
    "69016631b4ed99b9b570b31c65ff665c3e9894b6d276c38e81b21415092506a4";

// Measured peak of live C++ heap on the streaming thread (gfx1151 stand,
// 2026-11-14): 8,409,971 bytes. It is one 8 MiB scratch buffer that device
// rows are copied through (8,388,608) plus 21,363 bytes of headers, the
// frontier-logits span bookkeeping and std::function state. The bound keeps a
// 64 KiB margin for those small allocations and fails if a second scratch
// buffer or a payload-sized copy (19.5 MB here) ever becomes live. A bound one
// byte under the measured peak was seen to fail first.
constexpr std::size_t kPeakLiveLimit = 8U * 1024U * 1024U + 65536U;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

void Stage(std::string_view stage) noexcept {
  // Allocation-free markers also survive an abort during HIP/LLVM startup.
  (void)::write(STDOUT_FILENO, "setup_stage=", 12);
  (void)::write(STDOUT_FILENO, stage.data(), stage.size());
  (void)::write(STDOUT_FILENO, "\n", 1);
}

void ReportStreamFailure(const char* stage, const AllocationLedger& ledger,
                         std::size_t delivered, std::size_t callbacks,
                         const std::exception& error) {
  // Called only after AllocationScope has unwound. Preserve the exception;
  // HIP may translate a rejected C++ allocation into hipErrorOutOfMemory.
  std::cerr << "stream_failure stage=" << stage << " rejection="
            << (ledger.rejection != nullptr ? ledger.rejection : "none")
            << " rejected_bytes=" << ledger.rejected_bytes
            << " peak_live_bytes=" << ledger.peak_bytes
            << " remaining_live_bytes=" << ledger.live_bytes
            << " allocations=" << ledger.allocations
            << " largest_allocation=" << ledger.largest_bytes
            << " delivered_bytes=" << delivered << " callbacks=" << callbacks
            << " error=" << error.what() << '\n';
}
}  // namespace

namespace {
using namespace gufo;

// The model owns its device copy; synthetic host storage only supplies offsets
// during construction. All dense tensor references point into that one copy.
std::shared_ptr<const hip::QwenGpuModel> MakeSyntheticModel() {
  auto config = models::qwen::make_small_qwen_config();
  config.num_layers = 12;
  config.ssm_time_step_rank = 32;
  config.ssm_inner_size = 4096;
  config.context_length = 8;
  Require(config.IsValidQwen(), "synthetic Qwen configuration is invalid");
  Stage("synthetic_weights_begin");
  auto synthetic = models::qwen::build_synthetic_qwen_weights(config);
  Stage("synthetic_weights_ready");
  const auto bytes = synthetic.pool.size() * sizeof(float);
  void* device = nullptr;
  HIP_CHECK(hipMalloc(&device, bytes));
  Stage("device_weights_allocated");
  struct DeviceCleanup {
    void* pointer;
    ~DeviceCleanup() {
      if (pointer != nullptr)
        (void)hipFree(pointer);
    }
  } cleanup{device};
  HIP_CHECK(
      hipMemcpy(device, synthetic.pool.data(), bytes, hipMemcpyHostToDevice));
  Stage("device_weights_copied");
  auto weights = std::move(synthetic.weights);
  const auto remap = [&](models::QwenTensorRef& tensor) {
    if (tensor.data == nullptr)
      return;
    const auto offset =
        static_cast<const float*>(tensor.data) - synthetic.pool.data();
    Require(
        offset >= 0 && static_cast<std::size_t>(offset) < synthetic.pool.size(),
        "synthetic tensor lies outside its owned storage");
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
  auto model = std::make_shared<hip::QwenGpuModel>(
      nullptr, std::move(weights), std::move(tokenizer), std::move(regions));
  cleanup.pointer = nullptr;
  Stage("model_ready");
  return model;
}

void TestActualQwenStreamDoesNotAllocateFullPayload() {
  auto model = MakeSyntheticModel();
  auto runner = server::CreateQwenTextRunner(
      model, {.max_context = 8,
              .model_artifact_fingerprint = kSyntheticArtifactFingerprint});
  Stage("runner_ready");
  auto state = runner->CreateState();
  Stage("state_ready");
  const std::array<server::TextRunnerToken, 2> prompt{1, 2};
  std::size_t offset = 0;
  Stage("prefill_begin");
  while (offset != prompt.size()) {
    const auto step = runner->Prefill(*state, prompt, offset, prompt.size());
    Require(step.consumed_tokens != 0 &&
                step.consumed_tokens <= prompt.size() - offset,
            "synthetic Qwen prefill made invalid progress");
    offset += step.consumed_tokens;
  }
  Require(runner->CheckpointPosition(*state) == prompt.size(),
          "synthetic Qwen checkpoint position");
  Stage("prefill_complete");
  auto snapshot = runner->Snapshot(*state);
  Stage("snapshot_ready");
  const auto payload_bytes = runner->PersistentSnapshotPayloadBytes(*snapshot);
  // Outer112 + compact80 + K/V3072 + conv663552 + DeltaNet18874368 +
  // logits1024. The plain fixture has neither pending tokens nor a verifier.
  Require(payload_bytes == 19542208,
          "synthetic Qwen persistent layout differs from the fixture");
  Require(payload_bytes > kLiveLimit,
          "fixture must exceed the complete measured heap budget");
  std::vector<std::uint8_t> expected(payload_bytes, 0xA5);
  Stage("reference_serialize_begin");
  Require(runner->SerializePersistentSnapshot(*snapshot, expected) ==
              expected.size(),
          "legacy Qwen serializer byte count");
  // Independent fixture fields: a shared production header builder must not
  // make the direct-vs-stream comparison its own format oracle.
  std::array<std::uint8_t, 112> expected_header{};
  const std::array<std::uint8_t, 8> magic{'G', 'Q', 'W', 'R',
                                          'U', 'N', '0', '1'};
  std::copy(magic.begin(), magic.end(), expected_header.begin());
  const auto put = [&](std::size_t offset, std::uint64_t value,
                       std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
      expected_header[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
  };
  std::uint32_t frontier = 0;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (std::uint32_t token = 0; token < 256; ++token) {
    float logit = 0;
    // Independent fixture offset; do not read the frontier/header sizes.
    std::memcpy(&logit, expected.data() + 19541184 + token * sizeof(float),
                sizeof(logit));
    Require(std::isfinite(logit), "synthetic frontier logits must be finite");
    if (logit > best_logit) {
      best_logit = logit;
      frontier = token;
    }
  }
  put(8, 5, 4);
  put(12, 112, 4);
  put(16, 2, 8);
  put(24, frontier, 4);
  put(32, 256, 8);
  put(40, 19541072, 8);
  put(48, 19542208, 8);
  Require(std::equal(expected_header.begin(), expected_header.end(),
                     expected.begin()),
          "Qwen persistent header differs from independent plain fixture");
  Stage("reference_serialize_complete");
  std::vector<std::uint8_t> actual(payload_bytes, 0x5A);
  std::size_t written = 0;
  std::size_t callbacks = 0;
  std::size_t largest_chunk = 0;
  const server::TextModelRunner::SnapshotSink sink = [&](auto bytes) {
    Require(bytes.size() <= actual.size() - written,
            "Qwen stream exceeded its declared payload");
    std::memcpy(actual.data() + written, bytes.data(), bytes.size());
    written += bytes.size();
    largest_chunk = std::max(largest_chunk, bytes.size());
    ++callbacks;
  };
  AllocationLedger ledger;
  std::cout << "fixture_ready payload_bytes=" << payload_bytes
            << " chunk_limit=" << kChunkLimit << " live_limit=" << kLiveLimit
            << " scope=plain-qwen mtp=not-exercised\n"
            << std::flush;
  Stage("stream_first_begin");
  try {
    const AllocationScope measured(ledger);
    runner->StreamPersistentSnapshot(*snapshot, sink);
  } catch (const std::bad_alloc& error) {
    ReportStreamFailure("first", ledger, written, callbacks, error);
    if (ledger.rejection != nullptr) {
      const bool expected_full_payload =
          std::strcmp(ledger.rejection, "single-allocation") == 0 &&
          ledger.rejected_bytes == payload_bytes;
      std::cerr << "stream_allocation_rejected reason=" << ledger.rejection
                << " requested_bytes=" << ledger.rejected_bytes
                << " payload_bytes=" << payload_bytes
                << " peak_live_bytes=" << ledger.peak_bytes
                << " expected_full_payload=" << expected_full_payload << '\n';
      Require(expected_full_payload,
              "stream allocation failure was not the expected full payload");
    }
    throw;
  } catch (const std::exception& error) {
    ReportStreamFailure("first", ledger, written, callbacks, error);
    throw;
  }
  Require(ledger.rejection == nullptr,
          "Qwen stream caught and concealed an allocation bound failure");
  Require(ledger.live_bytes == 0,
          "Qwen stream retained measured allocations after return");
  Require(written == expected.size() && actual == expected,
          "streamed Qwen bytes differ from the direct serializer");
  Require(callbacks > 1 && largest_chunk <= kChunkLimit,
          "Qwen stream callback is not bounded");
  if (ledger.peak_bytes > kPeakLiveLimit) {
    throw std::runtime_error(
        "Qwen stream peak of live heap " + std::to_string(ledger.peak_bytes) +
        " exceeds the measured bound " + std::to_string(kPeakLiveLimit));
  }
  // A snapshot of another runner type is refused before any byte is emitted.
  struct ForeignSnapshot final : server::TextRunnerSnapshot {
    [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
      return 16;
    }
  };
  {
    const ForeignSnapshot foreign;
    std::size_t foreign_callbacks = 0;
    bool refused = false;
    try {
      runner->StreamPersistentSnapshot(foreign,
                                       [&](auto) { ++foreign_callbacks; });
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Require(refused && foreign_callbacks == 0,
            "Qwen stream accepted a snapshot of another runner type");
  }
  // Throw after copied K, and after 16 MiB of DeltaNet bytes (two 8 MiB
  // transfers). The frozen snapshot must survive both partial writes and
  // remain reusable. These byte boundaries do not depend on production sizing.
  struct SinkFailure {};
  for (const auto stop_after :
       {std::size_t{1728}, std::size_t{666816} + kChunkLimit}) {
    std::size_t delivered = 0;
    std::size_t throwing_callbacks = 0;
    const server::TextModelRunner::SnapshotSink throwing = [&](auto bytes) {
      Require(bytes.size() <= kChunkLimit,
              "throwing stream exceeded chunk cap");
      delivered += bytes.size();
      ++throwing_callbacks;
      if (delivered >= stop_after)
        throw SinkFailure{};
    };
    AllocationLedger failure_ledger;
    bool threw = false;
    try {
      const AllocationScope measured(failure_ledger);
      runner->StreamPersistentSnapshot(*snapshot, throwing);
    } catch (const SinkFailure&) {
      threw = true;
    } catch (const std::exception& error) {
      ReportStreamFailure("throwing_sink", failure_ledger, delivered,
                          throwing_callbacks, error);
      throw;
    }
    Require(threw && delivered >= stop_after && delivered < payload_bytes,
            "Qwen stream did not propagate the partial sink failure");
    Require(
        failure_ledger.rejection == nullptr && failure_ledger.live_bytes == 0,
        "failed Qwen stream leaked or concealed allocation failure");
    std::fill(actual.begin(), actual.end(), 0x5A);
    Require(runner->SerializePersistentSnapshot(*snapshot, actual) ==
                    actual.size() &&
                actual == expected,
            "sink failure mutated the frozen Qwen snapshot");
    std::fill(actual.begin(), actual.end(), 0x5A);
    written = 0;
    callbacks = 0;
    largest_chunk = 0;
    AllocationLedger retry_ledger;
    try {
      const AllocationScope measured(retry_ledger);
      runner->StreamPersistentSnapshot(*snapshot, sink);
    } catch (const std::exception& error) {
      ReportStreamFailure("retry", retry_ledger, written, callbacks, error);
      throw;
    }
    Require(
        retry_ledger.rejection == nullptr && retry_ledger.live_bytes == 0 &&
            written == expected.size() && actual == expected && callbacks > 1 &&
            largest_chunk <= kChunkLimit,
        "Qwen stream retry after sink failure was not byte-exact and bounded");
  }
  std::cout << "stream_pass bytes=" << written << " callbacks=" << callbacks
            << " largest_chunk=" << largest_chunk
            << " allocations=" << ledger.allocations
            << " largest_allocation=" << ledger.largest_bytes
            << " peak_live_bytes=" << ledger.peak_bytes
            << " remaining_live_bytes=" << ledger.live_bytes << '\n';
}
}  // namespace

int main() {
  Stage("main_enter");
  try {
    CheckAllocatorAlignment();
  } catch (const std::exception& error) {
    std::cerr << "allocator setup: " << error.what() << '\n';
    return 1;
  }
  Stage("allocator_alignment_pass");
  Stage("hip_device_gate_begin");
  if (const auto gate =
          gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kRequired,
                                    "qwen_persistent_snapshot_stream_test");
      gate != gufo::test::kHipTestSuccess)
    return gate;
  Stage("hip_device_gate_pass");
  try {
    TestActualQwenStreamDoesNotAllocateFullPayload();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen_persistent_snapshot_stream_test: " << error.what()
              << '\n';
    return 1;
  }
}
