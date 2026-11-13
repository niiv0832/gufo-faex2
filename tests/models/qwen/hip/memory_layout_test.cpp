// Memory layout arithmetic for the Qwen3.6-35B-A3B serving profile: four
// sessions of 131072 tokens with the MTP head and vision. The reserve covers
// every combination of lazily allocated buffers, and the shared-score rows
// enter it once per head at exactly 8011808 bytes.

#include "src/models/qwen/hip/detail/memory_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

using gufo::hip::detail::ComputeQwenArenaMemoryLayout;
using gufo::hip::detail::ComputeQwenMtpHeadMemoryLayout;

constexpr std::uint32_t kContext = 131072;
constexpr std::size_t kSessions = 4;

int failures{0};

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

gufo::core::ModelConfig Qwen36A3bConfig() {
  gufo::core::ModelConfig config;
  config.architecture = "qwen35moe";
  config.num_layers = 40;
  config.hidden_size = 2048;
  config.intermediate_size = 9728;  // the GGUF names none: the default stays
  config.num_attention_heads = 16;
  config.num_key_value_heads = 2;
  config.head_dim = 256;
  config.vocab_size = 248320;
  config.context_length = 262144;
  config.full_attention_interval = 4;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 128;
  config.ssm_group_count = 16;
  config.ssm_time_step_rank = 32;
  config.ssm_inner_size = 4096;
  config.num_experts = 256;
  config.num_experts_used = 8;
  config.expert_ff = 512;
  config.shared_expert_ff = 512;
  return config;
}

void TestMtpHead() {
  const auto config = Qwen36A3bConfig();
  const auto layout = ComputeQwenMtpHeadMemoryLayout(config, kContext, true);
  Check(layout.shared_score_bytes == 8011808,
        "the shared score is 8 rows of hidden and vocabulary floats and ids");
  Check(layout.shared_score_bytes ==
            8 * (2048 + 248320) * sizeof(float) + 8 * sizeof(std::uint32_t),
        "the shared score formula");
  Check(layout.kv_bytes == std::size_t{2} * kContext * 2 * 256 * 2,
        "the head keeps K and V in FP16 alone");
  // Measured on the device by qwen_mtp_shared_score_memory_test.
  Check(layout.fixed_scratch_bytes == 1817708, "the head's fixed scratch");
  Check(layout.prime_scratch_bytes == 115347456, "the priming scratch");
  const std::size_t reserve = layout.ReservedScratchBytes();
  for (const bool primed : {false, true}) {
    for (const bool led : {false, true}) {
      const std::size_t held = layout.fixed_scratch_bytes +
                               (primed ? layout.prime_scratch_bytes : 0) +
                               (led ? layout.shared_score_bytes : 0);
      Check(held <= reserve, "a head holds more than its reserve");
      Check((held == reserve) == (primed && led),
            "only a head holding every workspace reaches its reserve");
    }
  }
  // The lead of a cohort changes, and a former lead keeps its rows: the pool
  // reserves them for every head, and one head's rows are counted once.
  Check(kSessions * reserve - kSessions * (layout.fixed_scratch_bytes +
                                           layout.prime_scratch_bytes) ==
            kSessions * 8011808,
        "four heads reserve four shared-score workspaces");
  const auto dual = ComputeQwenMtpHeadMemoryLayout(config, kContext, false);
  Check(dual.kv_bytes == 3 * layout.kv_bytes,
        "a head with both caches holds FP32 and FP16 planes");
  std::cout << "mtp_head,context=" << kContext << ",kv=" << layout.kv_bytes
            << ",fixed=" << layout.fixed_scratch_bytes
            << ",prime=" << layout.prime_scratch_bytes
            << ",shared_score=" << layout.shared_score_bytes
            << ",reserved_scratch=" << reserve << '\n';
}

void TestArena() {
  const auto config = Qwen36A3bConfig();
  const auto policy = gufo::hip::QwenExecutionPolicy::Production();
  const auto layout = ComputeQwenArenaMemoryLayout(config, kContext, policy);
  const std::size_t element =
      gufo::hip::QwenRecurrentStateElementBytes(policy.recurrent_state_storage);
  Check(layout.kv_bytes == std::size_t{10} * 2 * 256 * kContext * 2 *
                               (policy.UsesFp16AttentionKv() ? 2 : 4),
        "ten attention layers keep K and V planes of the whole context");
  Check(
      layout.recurrent_bytes == std::size_t{30} * 8192 * 4 * sizeof(float) +
                                    std::size_t{30} * 32 * 128 * 128 * element,
      "thirty recurrent layers keep a convolution and a DeltaNet state");
  // 64 decode rows of router, top-k, expert and shared-expert scratch.
  const std::size_t moe_row =
      256 + 1 + 8 + 8 + 1 + 8 * 512 + 8 * 2048 + 3 * 512 + 2048;
  Check(layout.moe_decode_bytes == (64 * moe_row + 8) * sizeof(float),
        "the MoE decode scratch");
  // A prompt batch of 4096 rows: FP16 inputs, router and shared-expert
  // floats, split router inputs and outputs.
  const std::size_t batch = 4096;
  Check(layout.moe_prefill_bytes ==
            batch * 2048 * 2 +
                batch * (256 + 1 + 3 * 512 + 2048) * sizeof(float) +
                batch * 3 * 2048 * 2 + batch * 264 * sizeof(float),
        "the MoE prompt scratch");
  Check(layout.replay_log_bytes ==
            std::size_t{30} * 16 * (8192 + 2 * 32) * sizeof(float) +
                sizeof(std::uint32_t),
        "the replay log keeps sixteen positions of every recurrent layer");
  Check(layout.ReservedStateBytes() ==
            layout.kv_bytes + 2 * layout.recurrent_bytes,
        "the reserved state holds both recurrent buffers");

  for (const bool npu : {false, true}) {
    const std::size_t reserve = layout.ReservedScratchBytes(npu);
    for (const bool second_bank : {false, true}) {
      for (const bool replay_log : {false, true}) {
        if (second_bank && !npu)
          continue;  // paired prompt chunks need the NPU runtime
        const std::size_t held = layout.ConstructedScratchBytes() +
                                 (second_bank ? layout.bank_bytes : 0) +
                                 (replay_log ? layout.replay_log_bytes : 0);
        Check(held <= reserve, "an arena holds more scratch than its reserve");
      }
    }
    Check(reserve == layout.bank_bytes * (npu ? 2 : 1) +
                         layout.fixed_scratch_bytes + layout.moe_decode_bytes +
                         layout.moe_prefill_bytes + layout.replay_log_bytes,
          "the scratch reserve is the sum of its named buffers");
  }
  for (const bool rollback : {false, true}) {
    Check(layout.LiveStateBytes() + (rollback ? layout.recurrent_bytes : 0) <=
              layout.ReservedStateBytes(),
          "an arena holds more state than its reserve");
  }

  // The context changes the K/V planes alone: the prompt batch is capped.
  const auto small = ComputeQwenArenaMemoryLayout(config, 4096, policy);
  Check(small.bank_bytes == layout.bank_bytes &&
            small.moe_prefill_bytes == layout.moe_prefill_bytes &&
            small.recurrent_bytes == layout.recurrent_bytes &&
            small.kv_bytes * 32 == layout.kv_bytes,
        "only the K/V planes scale with the context");

  auto dense = config;
  dense.num_experts = 0;
  const auto dense_layout =
      ComputeQwenArenaMemoryLayout(dense, kContext, policy);
  Check(
      dense_layout.moe_decode_bytes == 0 && dense_layout.moe_prefill_bytes == 0,
      "a dense model reserves no MoE scratch");

  std::cout << "arena,context=" << kContext << ",kv=" << layout.kv_bytes
            << ",recurrent=" << layout.recurrent_bytes
            << ",bank=" << layout.bank_bytes
            << ",fixed=" << layout.fixed_scratch_bytes
            << ",moe_decode=" << layout.moe_decode_bytes
            << ",moe_prefill=" << layout.moe_prefill_bytes
            << ",replay_log=" << layout.replay_log_bytes
            << ",reserved_state=" << layout.ReservedStateBytes()
            << ",reserved_scratch=" << layout.ReservedScratchBytes(false)
            << ",reserved_scratch_with_npu="
            << layout.ReservedScratchBytes(true) << '\n';
}

void TestSharedAndVision() {
  const auto config = Qwen36A3bConfig();
  Check(gufo::hip::detail::QwenRouterSplitLayerBytes(config) ==
            264 * 3 * 2048 * sizeof(std::uint16_t),
        "one layer's router split weights");
  constexpr std::size_t kDescriptor = 24;
  const std::size_t vision =
      gufo::hip::detail::MaxQwenVisionInputBytes(config, kContext, kDescriptor);
  Check(vision == kContext * (3 * sizeof(std::int32_t) + 2048 * sizeof(float)) +
                      kDescriptor,
        "a context of image tokens: positions and embedding rows");
  std::cout << "vision_input,context=" << kContext << ",max_bytes=" << vision
            << ",router_split_layer="
            << gufo::hip::detail::QwenRouterSplitLayerBytes(config) << '\n';
}

}  // namespace

int main() {
  TestMtpHead();
  TestArena();
  TestSharedAndVision();
  if (failures != 0) {
    std::cerr << failures << " memory layout check(s) failed\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
