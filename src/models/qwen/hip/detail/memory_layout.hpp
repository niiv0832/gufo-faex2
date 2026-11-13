#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_MEMORY_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_MEMORY_LAYOUT_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

// Byte sizes of the device buffers a Qwen session holds, as arithmetic over
// the model configuration. The allocation sites and the memory accounting
// (EstimateMemoryUsage, GetMemoryUsage, the serving resource claim) read the
// same numbers, and none of this needs a device.
//
// Accounting keeps four things apart:
//   1. memory of the model, shared by every session: loaded weights, the
//      vision encoder, the MTP head, and the copies derived from the weights
//      on first use (QwenGpuModel::GetResidentBytes and
//      EstimateResidentBytes);
//   2. memory of one session: its executor, its draft head and what their
//      streams keep (QwenGpuMemoryUsage);
//   3. what is allocated now (the Get* functions);
//   4. the reserved maximum: every buffer a session or the model can come to
//      hold, lazily allocated ones included (the Estimate* functions).
//      Admission counts this before any state exists, so an allocated total
//      never exceeds it.
namespace gufo::hip::detail {

inline constexpr std::uint32_t kArenaMaxPromptBatch = 4096;
inline constexpr std::size_t kArenaMaxTargetLayerTaps = 9;
inline constexpr std::size_t kArenaMoeDecodeRows = 64;
inline constexpr std::size_t kArenaRouterSplitRows = 264;
inline constexpr std::size_t kArenaSsmReplayCapacity = 16;
inline constexpr std::size_t kMtpMaxSharedScoreRows = 8;
inline constexpr std::size_t kMtpPrimeChunk = 1024;

[[nodiscard]] constexpr std::size_t CheckedBytesProduct(std::size_t left,
                                                        std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Qwen GPU allocation size overflows");
  }
  return left * right;
}

[[nodiscard]] constexpr std::size_t CheckedBytesSum(std::size_t left,
                                                    std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen GPU allocation total overflows");
  }
  return left + right;
}

/// Device bytes of one QwenGpuArena, by owner and lifetime.
struct QwenArenaMemoryLayout {
  /// Attention K and V planes of every full-attention layer.
  std::size_t kv_bytes{0};
  /// Convolution and DeltaNet state of every recurrent layer: one buffer of
  /// the rollback pair.
  std::size_t recurrent_bytes{0};
  /// Per-batch prompt scratch (QwenPrefillBank). The constructor allocates
  /// one; paired prompt chunks, which only an NPU offload runs, add a second
  /// on first use.
  std::size_t bank_bytes{0};
  /// Scratch outside the bank: logits, target layer taps, split-K attention
  /// and the BF16 weight staging buffer.
  std::size_t fixed_scratch_bytes{0};
  /// MoE feed-forward scratch of decode rows and of a prompt batch; zero for
  /// dense models.
  std::size_t moe_decode_bytes{0};
  std::size_t moe_prefill_bytes{0};
  /// Recurrent replay log of a speculative session, allocated by its first
  /// capture.
  std::size_t replay_log_bytes{0};

  /// Serialized session state: what a snapshot of the whole context copies.
  [[nodiscard]] constexpr std::size_t LiveStateBytes() const {
    return CheckedBytesSum(kv_bytes, recurrent_bytes);
  }
  /// What the constructor allocates.
  [[nodiscard]] constexpr std::size_t ConstructedScratchBytes() const {
    return CheckedBytesSum(
        CheckedBytesSum(bank_bytes, fixed_scratch_bytes),
        CheckedBytesSum(moe_decode_bytes, moe_prefill_bytes));
  }
  /// Everything the arena can hold: both recurrent buffers, the replay log
  /// and, where paired prompt chunks can run, both banks.
  [[nodiscard]] constexpr std::size_t ReservedStateBytes() const {
    return CheckedBytesSum(LiveStateBytes(), recurrent_bytes);
  }
  [[nodiscard]] constexpr std::size_t ReservedScratchBytes(
      bool paired_prompt_chunks) const {
    return CheckedBytesSum(
        CheckedBytesSum(ConstructedScratchBytes(),
                        paired_prompt_chunks ? bank_bytes : 0),
        replay_log_bytes);
  }
};

[[nodiscard]] constexpr QwenArenaMemoryLayout ComputeQwenArenaMemoryLayout(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy) {
  const auto mul = [](std::size_t left, std::size_t right) {
    return CheckedBytesProduct(left, right);
  };
  const std::size_t context = std::max<std::uint32_t>(max_context, 1U);
  const std::size_t batch =
      std::min<std::size_t>(context, kArenaMaxPromptBatch);
  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv = mul(config.num_key_value_heads, config.head_dim);
  const std::size_t q_projection = mul(2, attention);
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t inner = config.ssm_inner_size;
  const std::size_t recurrent = std::max(attention, inner);
  const std::size_t projection = std::max(q_projection, ssm_qkv);
  const std::size_t time_step = config.ssm_time_step_rank;
  constexpr std::size_t kFloat = sizeof(float);
  constexpr std::size_t kHalf = sizeof(std::uint16_t);

  QwenArenaMemoryLayout layout;
  const auto add = [&mul](std::size_t elements, std::size_t element_bytes,
                          std::size_t& total) {
    total = CheckedBytesSum(total, mul(elements, element_bytes));
  };

  auto& bank = layout.bank_bytes;
  add(mul(batch, hidden), kFloat, bank);        // hidden
  add(mul(batch, hidden), kFloat, bank);        // normed
  add(mul(batch, attention), kFloat, bank);     // q
  add(mul(batch, kv), kFloat, bank);            // k
  add(mul(batch, kv), kFloat, bank);            // v
  add(mul(batch, hidden), kFloat, bank);        // attn_out
  add(mul(batch, intermediate), kFloat, bank);  // ffn_gate
  add(mul(batch, intermediate), kFloat, bank);  // ffn_up
  add(mul(batch, intermediate), kFloat, bank);  // ffn_act
  add(mul(batch, hidden), kFloat, bank);        // ffn_out
  add(mul(batch, projection), kFloat, bank);    // ssm_qkv
  add(mul(batch, ssm_qkv), kFloat, bank);       // conv_out
  add(mul(batch, recurrent), kFloat, bank);     // ssm_gate
  add(mul(batch, recurrent), kFloat, bank);     // ssm_out
  add(mul(batch, time_step), kFloat, bank);     // alpha_buf
  add(mul(batch, time_step), kFloat, bank);     // beta_buf
  add(mul(mul(batch, config.ssm_group_count), 3), kFloat, bank);
  add(mul(mul(batch, time_step), 2), kFloat, bank);
  add(std::max<std::size_t>(batch, 2), sizeof(std::uint32_t), bank);
  const std::size_t scratch_width = std::max<std::size_t>(
      {intermediate, hidden, projection,
       CheckedBytesSum(ssm_qkv, CheckedBytesSum(inner, mul(2, time_step)))});
  const std::size_t scratch_elements = mul(batch, scratch_width);
  add(scratch_elements, kHalf, bank);  // scratch_bf16
  // Tiled Q8_1 activations: sixteen tokens per tile, 36 bytes of payload and
  // a 4-byte sidecar per 32 elements.
  const std::size_t q8_rows = mul((batch + 15) / 16, 16);
  const std::size_t q8_blocks = (mul(q8_rows, scratch_width) + 31) / 32;
  bank = CheckedBytesSum(
      bank, CheckedBytesSum(mul(mul(q8_blocks, kFloat), 10), 4096));

  auto& fixed = layout.fixed_scratch_bytes;
  add(config.vocab_size, kFloat, fixed);
  add(mul(kArenaMaxTargetLayerTaps, hidden), kFloat, fixed);
  add(DecodeAttentionScratchElements(config.num_attention_heads,
                                     config.head_dim),
      kFloat, fixed);
  const std::size_t weight_width = std::max<std::size_t>(
      {q_projection, kv, attention, intermediate, ssm_qkv, inner, time_step});
  add(mul(hidden, weight_width), kHalf, fixed);

  if (config.IsMoe()) {
    const std::size_t experts = config.num_experts;
    const std::size_t used = config.num_experts_used;
    const std::size_t shared_ff = config.shared_expert_ff;
    const std::size_t row_floats =
        experts + 1 + used + used + 1 + mul(used, config.expert_ff) +
        mul(used, hidden) + mul(3, shared_ff) + hidden;
    add(CheckedBytesSum(mul(kArenaMoeDecodeRows, row_floats), 8), kFloat,
        layout.moe_decode_bytes);
    auto& prefill = layout.moe_prefill_bytes;
    add(mul(batch, hidden), kHalf, prefill);  // x_half
    add(CheckedBytesSum(
            CheckedBytesSum(mul(batch, experts), batch),
            CheckedBytesSum(mul(mul(3, batch), shared_ff), mul(batch, hidden))),
        kFloat, prefill);
    add(mul(mul(batch, 3), hidden), kHalf, prefill);          // router_x
    add(mul(batch, kArenaRouterSplitRows), kFloat, prefill);  // router_y
  }

  add(mul(mul(mul(config.FullAttentionLayerCount(), kv), context), 2),
      policy.UsesFp16AttentionKv() ? kHalf : kFloat, layout.kv_bytes);

  const std::size_t ssm_layers = config.SsmLayerCount();
  add(mul(mul(ssm_layers, ssm_qkv), config.ssm_conv_kernel), kFloat,
      layout.recurrent_bytes);
  add(mul(mul(mul(ssm_layers, time_step), config.ssm_state_size),
          config.SsmValueSize()),
      QwenRecurrentStateElementBytes(policy.recurrent_state_storage),
      layout.recurrent_bytes);

  const std::size_t replay_slots = mul(ssm_layers, kArenaSsmReplayCapacity);
  add(mul(replay_slots, CheckedBytesSum(ssm_qkv, mul(2, time_step))), kFloat,
      layout.replay_log_bytes);
  layout.replay_log_bytes =
      CheckedBytesSum(layout.replay_log_bytes, sizeof(std::uint32_t));
  return layout;
}

/// Rows and widest row of the BF16 activations a prompt batch stages in the
/// arena; scratch derived from them elsewhere is bounded by this shape.
struct QwenPromptScratchShape {
  std::size_t batch{0};
  std::size_t width{0};
};

[[nodiscard]] constexpr QwenPromptScratchShape ComputeQwenPromptScratchShape(
    const core::ModelConfig& config, std::uint32_t max_context) {
  const std::size_t projection = std::max<std::size_t>(
      CheckedBytesProduct(2, config.AttentionSize()), config.SsmQkvSize());
  return {
      .batch = std::min<std::size_t>(std::max<std::uint32_t>(max_context, 1U),
                                     kArenaMaxPromptBatch),
      .width = std::max<std::size_t>(
          {config.intermediate_size, config.hidden_size, projection,
           CheckedBytesSum(
               config.SsmQkvSize(),
               CheckedBytesSum(
                   config.ssm_inner_size,
                   CheckedBytesProduct(2, config.ssm_time_step_rank)))}),
  };
}

/// Device bytes of one QwenMtpGpuExecutor, by lifetime.
struct QwenMtpHeadMemoryLayout {
  /// K and V cache of the head's attention.
  std::size_t kv_bytes{0};
  /// Single-row scratch the constructor allocates.
  std::size_t fixed_scratch_bytes{0};
  /// Batched cache priming scratch, allocated by the first PrimeBatch.
  std::size_t prime_scratch_bytes{0};
  /// Rows of a shared score, allocated on the head that first leads one and
  /// kept until the head is destroyed. Another head leading a later pass
  /// allocates its own, so every head of a pool can come to hold one.
  std::size_t shared_score_bytes{0};

  [[nodiscard]] constexpr std::size_t ReservedScratchBytes() const {
    return CheckedBytesSum(
        CheckedBytesSum(fixed_scratch_bytes, prime_scratch_bytes),
        shared_score_bytes);
  }
};

/// `half_kv_only` is QwenMtpGpuExecutor::KeepsHalfKvOnly.
[[nodiscard]] constexpr QwenMtpHeadMemoryLayout ComputeQwenMtpHeadMemoryLayout(
    const core::ModelConfig& config, std::uint32_t max_context,
    bool half_kv_only) {
  const auto mul = [](std::size_t left, std::size_t right) {
    return CheckedBytesProduct(left, right);
  };
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv = mul(config.num_key_value_heads, config.head_dim);
  constexpr std::size_t kFloat = sizeof(float);
  constexpr std::size_t kHalf = sizeof(std::uint16_t);

  QwenMtpHeadMemoryLayout layout;
  layout.kv_bytes =
      mul(mul(mul(2, max_context), kv), half_kv_only ? kHalf : kFloat + kHalf);

  // target_hidden, embedding, hidden, normed, attn_out, ffn_out and
  // feedback_hidden; fusion holds two rows.
  std::size_t floats = mul(9, hidden);
  // qg holds two rows; q, gate and context one each.
  floats = CheckedBytesSum(floats, mul(5, attention));
  floats = CheckedBytesSum(floats, mul(2, kv));
  floats = CheckedBytesSum(floats,
                           std::max<std::size_t>(config.intermediate_size, 1));
  if (config.IsMoe()) {
    const std::size_t used = config.num_experts_used;
    floats = CheckedBytesSum(
        floats, config.num_experts + 1 + used + used + 1 +
                    mul(used, config.expert_ff) + mul(used, hidden) +
                    mul(3, config.shared_expert_ff) + hidden + 8);
  }
  floats = CheckedBytesSum(floats, config.vocab_size);
  floats =
      CheckedBytesSum(floats, DecodeAttentionScratchElements(
                                  config.num_attention_heads, config.head_dim));
  layout.fixed_scratch_bytes =
      CheckedBytesSum(mul(floats, kFloat), sizeof(std::uint32_t));

  const std::size_t prime_floats =
      mul(kMtpPrimeChunk,
          CheckedBytesSum(CheckedBytesSum(mul(8, hidden), mul(3, kv)),
                          mul(2, attention)));
  layout.prime_scratch_bytes = CheckedBytesSum(
      CheckedBytesSum(mul(prime_floats, kFloat),
                      mul(kMtpPrimeChunk, sizeof(std::uint32_t))),
      mul(mul(mul(kMtpPrimeChunk, 2), hidden), kHalf));

  layout.shared_score_bytes =
      CheckedBytesSum(mul(mul(kMtpMaxSharedScoreRows,
                              CheckedBytesSum(hidden, config.vocab_size)),
                          kFloat),
                      mul(kMtpMaxSharedScoreRows, sizeof(std::uint32_t)));
  return layout;
}

/// Device bytes of the vision input a session can hold at `max_context`:
/// three position ids per token and, were every token an image patch, one
/// embedding row per token. `descriptor_bytes` is sizeof(vision::DeviceRope).
[[nodiscard]] constexpr std::size_t MaxQwenVisionInputBytes(
    const core::ModelConfig& config, std::uint32_t max_context,
    std::size_t descriptor_bytes) {
  const std::size_t per_token =
      CheckedBytesSum(3 * sizeof(std::int32_t),
                      CheckedBytesProduct(config.hidden_size, sizeof(float)));
  return CheckedBytesSum(CheckedBytesProduct(max_context, per_token),
                         descriptor_bytes);
}

/// Bytes of the bf16 router copy RouterSplitWeights builds for one layer.
[[nodiscard]] constexpr std::size_t QwenRouterSplitLayerBytes(
    const core::ModelConfig& config) {
  return CheckedBytesProduct(
      CheckedBytesProduct(kArenaRouterSplitRows * 3, config.hidden_size),
      sizeof(std::uint16_t));
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_MEMORY_LAYOUT_HPP_
