#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/snapshot_stream.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

namespace {
using gufo::hip::QwenGpuArena;
using gufo::hip::QwenGpuSnapshot;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

void Put(std::span<std::uint8_t> bytes, std::size_t offset, std::uint64_t value,
         unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    bytes[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
}

std::vector<std::uint8_t> SeedBytes(std::size_t size, unsigned salt) {
  std::vector<std::uint8_t> result(size);
  for (std::size_t i = 0; i < size; ++i)
    result[i] = static_cast<std::uint8_t>(salt + i * 37U + (i >> 7U) * 19U +
                                          (i >> 19U) * 11U);
  return result;
}

struct Fixture {
  gufo::core::ModelConfig config = gufo::models::qwen::make_small_qwen_config();
  gufo::hip::QwenExecutionPolicy policy;
  std::uint32_t context;
  std::uint32_t valid;
  std::size_t element_bytes;
  std::size_t plane_elements;
  std::size_t live_plane_bytes;
  std::vector<std::uint8_t> kv, conv, delta, golden;

  Fixture(bool fp16, bool bf16, std::uint32_t valid_rows, bool large)
      : context(large ? 4103 : 11),
        valid(valid_rows),
        element_bytes(fp16 ? 2 : 4) {
    config.num_layers = 8;  // Two attention and six recurrent layers.
    if (large) {
      config.num_key_value_heads = 4;
      config.num_attention_heads = 4;
      config.head_dim = 256;
    }
    config.context_length = context;
    policy.kv_cache_storage = fp16 ? gufo::hip::QwenKvCacheStorage::kFp16
                                   : gufo::hip::QwenKvCacheStorage::kFp32;
    policy.recurrent_state_storage =
        bf16 ? gufo::hip::QwenRecurrentStateStorage::kBf16
             : gufo::hip::QwenRecurrentStateStorage::kFp32;
    const std::size_t layers = config.FullAttentionLayerCount();
    const std::size_t width = config.num_key_value_heads * config.head_dim;
    plane_elements = layers * context * width;
    live_plane_bytes = layers * valid * width * element_bytes;
    const auto conv_elements =
        static_cast<std::size_t>(config.SsmLayerCount()) * config.SsmQkvSize() *
        config.ssm_conv_kernel;
    const auto delta_elements =
        static_cast<std::size_t>(config.SsmLayerCount()) *
        config.ssm_time_step_rank * config.ssm_state_size *
        config.SsmValueSize();
    kv = SeedBytes(2 * plane_elements * element_bytes, 13);
    conv = SeedBytes(conv_elements * 4, 71);
    delta = SeedBytes(delta_elements * (bf16 ? 2 : 4), 193);
    golden.assign(80, 0);
    std::memcpy(golden.data(), "GQKVSNP1", 8);
    Put(golden, 8, 1, 4);
    Put(golden, 12, 80, 4);
    Put(golden, 16, layers, 4);
    Put(golden, 20, width, 4);
    Put(golden, 24, context, 4);
    Put(golden, 28, valid, 4);
    Put(golden, 32, fp16 ? 1 : 0, 4);
    Put(golden, 36, bf16 ? 1 : 0, 4);
    Put(golden, 40, layers * valid * width, 8);
    Put(golden, 48, conv_elements, 8);
    Put(golden, 56, delta_elements, 8);
    Put(golden, 64, 80 + 2 * live_plane_bytes + conv.size() + delta.size(), 8);
    // Independent byte oracle. FP16 is [layer,token,head,dim], FP32 is
    // [layer,head,token,dim]. Do not call production compact-layout builders.
    for (std::size_t plane = 0; plane < 2; ++plane) {
      for (std::size_t layer = 0; layer < layers; ++layer) {
        const auto outer = fp16 ? valid : config.num_key_value_heads;
        const auto inner = fp16 ? config.num_key_value_heads : valid;
        for (std::size_t a = 0; a < outer; ++a) {
          for (std::size_t b = 0; b < inner; ++b) {
            const auto token = fp16 ? a : b;
            const auto head = fp16 ? b : a;
            const auto index = KvIndex(plane, layer, head, token, 0);
            const auto offset = index * element_bytes;
            golden.insert(
                golden.end(), kv.begin() + offset,
                kv.begin() + offset + config.head_dim * element_bytes);
          }
        }
      }
    }
    golden.insert(golden.end(), conv.begin(), conv.end());
    golden.insert(golden.end(), delta.begin(), delta.end());
  }

  std::size_t KvIndex(std::size_t plane, std::size_t layer, std::size_t head,
                      std::size_t token, std::size_t dimension) const {
    const auto heads = config.num_key_value_heads;
    const auto dim = config.head_dim;
    const auto local = policy.UsesFp16AttentionKv()
                           ? ((layer * context + token) * heads + head) * dim
                           : ((layer * heads + head) * context + token) * dim;
    return plane * plane_elements + local + dimension;
  }

  void* Kv(QwenGpuArena& arena) const {
    return policy.UsesFp16AttentionKv() ? arena.d_attention_kv_f16
                                        : static_cast<void*>(arena.d_kv_cache);
  }

  void Seed(QwenGpuArena& arena) const {
    HIP_CHECK(hipStreamSynchronize(arena.stream));
    HIP_CHECK(
        hipMemcpy(Kv(arena), kv.data(), kv.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(arena.d_ssm_conv_state, conv.data(), conv.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(arena.d_ssm_deltanet_state, delta.data(), delta.size(),
                        hipMemcpyHostToDevice));
  }

  void VerifyRestored(QwenGpuArena& arena) const {
    std::vector<std::uint8_t> actual(kv.size());
    HIP_CHECK(hipMemcpy(actual.data(), Kv(arena), actual.size(),
                        hipMemcpyDeviceToHost));
    for (std::size_t plane = 0; plane < 2; ++plane)
      for (std::size_t layer = 0; layer < config.FullAttentionLayerCount();
           ++layer)
        for (std::size_t head = 0; head < config.num_key_value_heads; ++head)
          for (std::size_t token = 0; token < valid; ++token) {
            const auto offset =
                KvIndex(plane, layer, head, token, 0) * element_bytes;
            Require(std::equal(actual.begin() + offset,
                               actual.begin() + offset +
                                   config.head_dim * element_bytes,
                               kv.begin() + offset),
                    "restored live KV differs from independent seeded bytes");
          }
    actual.resize(conv.size());
    HIP_CHECK(hipMemcpy(actual.data(), arena.d_ssm_conv_state, actual.size(),
                        hipMemcpyDeviceToHost));
    Require(actual == conv, "restored convolution bytes changed");
    actual.resize(delta.size());
    HIP_CHECK(hipMemcpy(actual.data(), arena.d_ssm_deltanet_state,
                        actual.size(), hipMemcpyDeviceToHost));
    Require(actual == delta, "restored recurrent bytes changed");
  }
};

std::vector<std::uint8_t> Collect(const QwenGpuSnapshot& snapshot) {
  std::vector<std::uint8_t> result;
  result.reserve(snapshot.CompactPayloadBytes());
  snapshot.StreamCompact([&](auto bytes) {
    Require(
        !bytes.empty() && bytes.size() <= gufo::core::kSnapshotStreamChunkBytes,
        "compact stream callback must be nonempty and bounded");
    result.insert(result.end(), bytes.begin(), bytes.end());
  });
  return result;
}

void TestArena(bool fp16, bool bf16, std::uint32_t valid, bool large) {
  Fixture f(fp16, bf16, valid, large);
  if (large)
    Require(f.live_plane_bytes > gufo::core::kSnapshotStreamChunkBytes &&
                f.live_plane_bytes % gufo::core::kSnapshotStreamChunkBytes != 0,
            "large fixture must cross a chunk boundary and retain a tail");
  QwenGpuArena source(f.config, f.context, f.policy);
  f.Seed(source);
  const auto snapshot = source.SaveSnapshot(valid);
  Require(snapshot->CompactPayloadBytes() == f.golden.size(),
          "compact payload size matches independent layout");
  // A snapshot must remain frozen when its original arena is reused.
  HIP_CHECK(hipMemset(f.Kv(source), 0xD3, f.kv.size()));
  HIP_CHECK(hipMemset(source.d_ssm_conv_state, 0xD3, f.conv.size()));
  HIP_CHECK(hipMemset(source.d_ssm_deltanet_state, 0xD3, f.delta.size()));
  std::vector<std::uint8_t> direct(f.golden.size(), 0xA5);
  Require(
      snapshot->SerializeCompact(direct) == direct.size() && direct == f.golden,
      "legacy compact bytes match independent golden");
  Require(Collect(*snapshot) == f.golden,
          "streamed compact bytes match independent golden");
  for (const auto threshold :
       {std::size_t{81}, 80 + 2 * f.live_plane_bytes + f.conv.size() + 1}) {
    std::size_t received = 0;
    bool threw = false;
    try {
      snapshot->StreamCompact([&](auto bytes) {
        received += bytes.size();
        if (received >= threshold)
          throw std::runtime_error("injected GPU chunk sink failure");
      });
    } catch (const std::runtime_error& error) {
      threw =
          std::string_view(error.what()) == "injected GPU chunk sink failure";
    }
    Require(threw, "sink must throw after GPU copy and in a later region");
    Require(Collect(*snapshot) == f.golden,
            "throwing sink leaves a reusable immutable GPU snapshot");
  }
  QwenGpuArena restored(f.config, f.context, f.policy);
  HIP_CHECK(hipMemset(f.Kv(restored), 0xE7, f.kv.size()));
  restored.RestoreCompactSnapshot(Collect(*snapshot), valid);
  f.VerifyRestored(restored);
}

void TestRopeV2() {
  Fixture f(true, false, 7, false);
  auto synthetic = gufo::models::qwen::build_synthetic_qwen_weights(f.config);
  std::vector<std::string> vocabulary(f.config.vocab_size, "token");
  for (std::size_t i = 0; i < vocabulary.size(); ++i)
    vocabulary[i] += std::to_string(i);
  std::string error;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocabulary, {}, {}, &error);
  Require(tokenizer != nullptr, error);
  // No forward executes: host weights only supply configuration and route
  // metadata. Their owning synthetic pool outlives model and executor.
  auto model = std::make_shared<gufo::hip::QwenGpuModel>(
      nullptr, std::move(synthetic.weights), std::move(tokenizer),
      std::vector<gufo::hip::QwenGpuWeightRegion>{});
  auto golden = f.golden;
  golden.resize(golden.size() + 12, 0);
  Put(golden, 8, 2, 4);
  Put(golden, 64, golden.size(), 8);
  Put(golden, 72, 1, 4);
  Put(golden, golden.size() - 12, 1, 4);
  Put(golden, golden.size() - 8, 2, 4);
  Put(golden, golden.size() - 4, 2, 4);
  gufo::hip::QwenGpuExecutor executor(model, f.context, f.policy);
  executor.Reset();
  executor.RestoreCompactSnapshot(golden, f.valid);
  const auto snapshot = executor.SaveSnapshot(f.valid);
  Require(Collect(*snapshot) == golden,
          "version2 image grid survives restore and streaming byte exactly");
  const auto layout = QwenGpuSnapshot::ReadRopeLayout(golden);
  Require(layout.images.size() == 1 && layout.images[0].offset == 1 &&
              layout.images[0].height == 2 && layout.images[0].width == 2,
          "independent rope fixture preserves offset and grid dimensions");
}

// Two images of different grids: the second entry of the rope table must keep
// its own offset and dimensions through restore, save and streaming.
void TestRopeV2TwoImages() {
  Fixture f(true, false, 7, false);
  auto synthetic = gufo::models::qwen::build_synthetic_qwen_weights(f.config);
  std::vector<std::string> vocabulary(f.config.vocab_size, "token");
  for (std::size_t i = 0; i < vocabulary.size(); ++i)
    vocabulary[i] += std::to_string(i);
  std::string error;
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocabulary, {}, {}, &error);
  Require(tokenizer != nullptr, error);
  auto model = std::make_shared<gufo::hip::QwenGpuModel>(
      nullptr, std::move(synthetic.weights), std::move(tokenizer),
      std::vector<gufo::hip::QwenGpuWeightRegion>{});
  // Independent fixture: tokens 0..3 are a 2x2 image, token 4 is text,
  // tokens 5..6 are a 1x2 image. Each entry is offset, height, width.
  auto golden = f.golden;
  const auto table = golden.size();
  golden.resize(table + 24, 0);
  Put(golden, 8, 2, 4);
  Put(golden, 64, golden.size(), 8);
  Put(golden, 72, 2, 4);
  Put(golden, table, 0, 4);
  Put(golden, table + 4, 2, 4);
  Put(golden, table + 8, 2, 4);
  Put(golden, table + 12, 5, 4);
  Put(golden, table + 16, 1, 4);
  Put(golden, table + 20, 2, 4);
  gufo::hip::QwenGpuExecutor executor(model, f.context, f.policy);
  executor.Reset();
  executor.RestoreCompactSnapshot(golden, f.valid);
  const auto snapshot = executor.SaveSnapshot(f.valid);
  Require(snapshot->CompactPayloadBytes() == golden.size(),
          "two-image snapshot is sized with both rope entries");
  std::vector<std::uint8_t> direct(golden.size(), 0xA5);
  Require(
      snapshot->SerializeCompact(direct) == direct.size() && direct == golden,
      "two-image snapshot serializes both rope entries byte exactly");
  Require(Collect(*snapshot) == golden,
          "two-image snapshot streams both rope entries byte exactly");
  const auto layout = QwenGpuSnapshot::ReadRopeLayout(Collect(*snapshot));
  Require(layout.images.size() == 2 && layout.images[0].offset == 0 &&
              layout.images[0].height == 2 && layout.images[0].width == 2 &&
              layout.images[1].offset == 5 && layout.images[1].height == 1 &&
              layout.images[1].width == 2,
          "streamed rope table keeps both images in order");
  // A second executor restored from the streamed bytes reproduces them.
  gufo::hip::QwenGpuExecutor second(model, f.context, f.policy);
  second.Reset();
  second.RestoreCompactSnapshot(Collect(*snapshot), f.valid);
  Require(Collect(*second.SaveSnapshot(f.valid)) == golden,
          "two-image snapshot survives a second restore and stream");
}
}  // namespace

int main() {
  if (const auto gate =
          gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kRequired,
                                    "qwen_snapshot_streaming_test");
      gate != gufo::test::kHipTestSuccess)
    return gate;
  try {
    for (const bool fp16 : {false, true})
      for (const bool bf16 : {false, true}) {
        TestArena(fp16, bf16, 0, false);
        TestArena(fp16, bf16, 7, false);
        TestArena(fp16, bf16, 4099, true);
      }
    TestRopeV2();
    TestRopeV2TwoImages();
    std::cout << "Qwen compact streaming contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen_snapshot_streaming_test: " << error.what() << '\n';
    return 1;
  }
}
