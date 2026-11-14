// A real Qwen runner must stream without materializing a complete host payload.
// Synthetic weights and a short prompt keep this independent of model files.
// The deliberately large recurrent state makes the old full-vector stream fail
// a thread-local C++ heap bound. HIP/driver allocations are outside that bound.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
#include "tests/models/qwen/support/synthetic_weights.hpp"

namespace {
constexpr std::size_t kChunkLimit = 16U * 1024U * 1024U;
constexpr std::size_t kLiveLimit = kChunkLimit + 4096U;
// SHA-256 of "gufo-qwen-persistent-snapshot-stream-fixture-v1" (no newline).
// Names this fixed synthetic fixture; it is not an external GGUF content hash.
constexpr char kSyntheticArtifactFingerprint[] =
    "69016631b4ed99b9b570b31c65ff665c3e9894b6d276c38e81b21415092506a4";

struct AllocationLedger {
  struct Entry {
    void* pointer{nullptr};
    std::size_t bytes{0};
  };
  std::array<Entry, 4096> entries{};
  std::size_t live_bytes{0};
  std::size_t peak_bytes{0};
  std::size_t largest_bytes{0};
  std::size_t allocations{0};
  std::size_t rejected_bytes{0};
  const char* rejection{nullptr};

  void Check(std::size_t bytes) {
    if (bytes > kChunkLimit) {
      Reject(bytes, "single-allocation");
    }
    if (bytes > kLiveLimit - live_bytes) {
      Reject(bytes, "aggregate-live-bytes");
    }
  }

  void Add(void* pointer, std::size_t bytes) {
    for (auto& entry : entries) {
      if (entry.pointer == nullptr) {
        entry = {pointer, bytes};
        live_bytes += bytes;
        peak_bytes = std::max(peak_bytes, live_bytes);
        largest_bytes = std::max(largest_bytes, bytes);
        ++allocations;
        return;
      }
    }
    Reject(bytes, "observer-capacity");
  }

  void Remove(void* pointer) noexcept {
    for (auto& entry : entries) {
      if (entry.pointer == pointer) {
        live_bytes -= entry.bytes;
        entry = {};
        return;
      }
    }
  }

  [[noreturn]] void Reject(std::size_t bytes, const char* reason) {
    rejected_bytes = bytes;
    rejection = reason;
    throw std::bad_alloc();
  }
};

thread_local AllocationLedger* active_ledger = nullptr;

class AllocationScope {
public:
  explicit AllocationScope(AllocationLedger& ledger) {
    if (active_ledger != nullptr)
      throw std::logic_error("allocation observer is already armed");
    active_ledger = &ledger;
  }
  ~AllocationScope() { active_ledger = nullptr; }
  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;
};

void* Allocate(std::size_t bytes, std::size_t alignment = 0) {
  if (active_ledger != nullptr)
    active_ledger->Check(bytes);
  void* pointer = nullptr;
  if (alignment == 0) {
    pointer = std::malloc(std::max(bytes, std::size_t{1}));
  } else {
    if ((alignment & (alignment - 1)) != 0)
      throw std::bad_alloc();
    // C++ callers (including LLVM during HIP startup) may request alignment
    // 1, 2 or 4. POSIX additionally requires a multiple of sizeof(void*).
    const auto posix_alignment = std::max(alignment, sizeof(void*));
    if (::posix_memalign(&pointer, posix_alignment,
                         std::max(bytes, std::size_t{1})) != 0)
      pointer = nullptr;
  }
  if (pointer == nullptr)
    throw std::bad_alloc();
  try {
    if (active_ledger != nullptr)
      active_ledger->Add(pointer, bytes);
  } catch (...) {
    std::free(pointer);
    throw;
  }
  return pointer;
}

void Deallocate(void* pointer) noexcept {
  if (pointer != nullptr && active_ledger != nullptr)
    active_ledger->Remove(pointer);
  std::free(pointer);
}

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
}  // namespace

void* operator new(std::size_t bytes) {
  return Allocate(bytes);
}
void* operator new[](std::size_t bytes) {
  return Allocate(bytes);
}
void* operator new(std::size_t bytes, std::align_val_t alignment) {
  return Allocate(bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
  return Allocate(bytes, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer) noexcept {
  Deallocate(pointer);
}
void operator delete[](void* pointer) noexcept {
  Deallocate(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  Deallocate(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  Deallocate(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  Deallocate(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  Deallocate(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  Deallocate(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  Deallocate(pointer);
}

namespace {
using namespace gufo;

void CheckAllocatorAlignment() {
  for (const std::size_t alignment : {1U, 2U, 4U, 8U, 16U, 64U, 128U}) {
    void* pointer = ::operator new(32, std::align_val_t(alignment));
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
    ::operator delete(pointer, std::align_val_t(alignment));
    Require(aligned, "test allocator violated a valid C++ alignment");
  }
}

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
  try {
    const AllocationScope measured(ledger);
    runner->StreamPersistentSnapshot(*snapshot, sink);
  } catch (const std::bad_alloc&) {
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
  }
  Require(ledger.rejection == nullptr,
          "Qwen stream caught and concealed an allocation bound failure");
  Require(ledger.live_bytes == 0,
          "Qwen stream retained measured allocations after return");
  Require(written == expected.size() && actual == expected,
          "streamed Qwen bytes differ from the direct serializer");
  Require(callbacks > 1 && largest_chunk <= kChunkLimit,
          "Qwen stream callback is not bounded");
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
