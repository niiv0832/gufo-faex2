#if defined(ENGINE_ENABLE_HIP)
#include <cstddef>
#include <stdexcept>
#include <string>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/moe_ffn.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"

namespace gufo::hip {
namespace {
// Below this many rows the per-expert buckets are too thin for grouped GEMMs:
// the row-wise device path (MoeFfnRows) is faster. Qwen3.6-35B-A3B on gfx1151,
// whole prompt of N tokens: 8 tokens 36 ms row-wise against 43 ms grouped, 16
// tokens 53 against 48, 63 tokens 133 against 65 (2026-11-12).
constexpr std::size_t kMinGroupedRows = 16;
}  // namespace

void MoePrefillRouterLogitsFallback(hipblasHandle_t handle,
                                    const models::QwenTensorRef& router,
                                    const models::QwenTensorRef& shared_gate,
                                    const float* x, float* router_logits,
                                    float* shared_logit, std::size_t rows,
                                    std::size_t experts, std::size_t hidden,
                                    void* bf16_scratch, hipStream_t stream) {
  const auto is_f32 = [](const models::QwenTensorRef& tensor) {
    return tensor.type == core::GgmlType::kF32;
  };
  if (!models::SupportsMoeRouterType(router.type) ||
      !models::SupportsMoeRouterType(shared_gate.type)) {
    throw std::invalid_argument(
        "Qwen MoE router and shared-expert gate must be F32 or BF16, found " +
        std::string(core::ToString(router.type)) + " and " +
        std::string(core::ToString(shared_gate.type)));
  }
  if (is_f32(router) && is_f32(shared_gate)) {
    LaunchHipblasGEMM(handle, router.data, false, x, router_logits, rows,
                      experts, hidden, bf16_scratch, stream);
    LaunchHipblasGEMM(handle, shared_gate.data, false, x, shared_logit, rows, 1,
                      hidden, bf16_scratch, stream);
    return;
  }
  // SGEMM would read BF16 words as floats. The row kernel reads each tensor
  // by its type and keeps the activations in fp32.
  LaunchMoeRouterLogits(router.data, router.type, shared_gate.data,
                        shared_gate.type, x, router_logits, shared_logit,
                        experts, hidden, stream, rows);
}

bool QwenGpuExecutor::MoePrefillFfn(const models::QwenLayerWeights& layer,
                                    std::uint32_t layer_index,
                                    std::size_t batch_size) {
  if (!arena_.numerical_overrides.grouped_moe_prefill) {
    return false;
  }
  const auto& config = weights_.config;
  QwenMoePrefill* const backend = model_->MoePrefill(arena_.GetMaxBatch());
  if (backend == nullptr || batch_size < kMinGroupedRows ||
      batch_size > backend->MaxTokens() || !backend->Supports(layer_index)) {
    return false;
  }
  const std::size_t hidden = config.hidden_size;
  const std::size_t shared_ff = config.shared_expert_ff;
  auto& scratch = arena_.moe_prefill;
  hipStream_t stream = arena_.stream;
  // FFN RMSNorm left the FP32 rows in d_normed and their BF16 copy in
  // d_scratch_bf16.
  LaunchFloatToFp16(arena_.d_normed, scratch.x_half, batch_size * hidden,
                    stream);
  const std::uint16_t* split_weights = model_->RouterSplitWeights(layer_index);
  if (split_weights != nullptr && arena_.hipblaslt_gemm != nullptr) {
    LaunchRouterSplitActivations(arena_.d_normed, scratch.router_x, batch_size,
                                 hidden, stream);
    if (!arena_.hipblaslt_gemm->RunBf16(split_weights, scratch.router_x,
                                        scratch.router_y, batch_size,
                                        kRouterSplitRows, 3 * hidden, stream)) {
      split_weights = nullptr;
    } else {
      LaunchRouterSplitLogits(scratch.router_y, scratch.router_logits,
                              scratch.shared_logit, batch_size,
                              config.num_experts, kRouterSplitRows, stream);
    }
  } else {
    split_weights = nullptr;
  }
  if (split_weights == nullptr) {
    MoePrefillRouterLogitsFallback(
        arena_.hipblas_handle, layer.ffn_gate_inp, layer.ffn_gate_inp_shexp,
        arena_.d_normed, scratch.router_logits, scratch.shared_logit,
        batch_size, config.num_experts, hidden, arena_.d_scratch_bf16, stream);
  }
  // Dense shared expert: gate/up projections, SwiGLU, down projection.
  LaunchBatchedQuantGEMM(layer.ffn_gate_shexp.type, layer.ffn_gate_shexp.data,
                         arena_.d_scratch_bf16, scratch.shared_gate, batch_size,
                         shared_ff, hidden, stream);
  LaunchBatchedQuantGEMM(layer.ffn_up_shexp.type, layer.ffn_up_shexp.data,
                         arena_.d_scratch_bf16, scratch.shared_up, batch_size,
                         shared_ff, hidden, stream);
  LaunchBatchedSwiGLUActivation(scratch.shared_gate, scratch.shared_up,
                                scratch.shared_act, arena_.d_scratch_bf16,
                                batch_size * shared_ff, stream);
  LaunchBatchedQuantGEMM(layer.ffn_down_shexp.type, layer.ffn_down_shexp.data,
                         arena_.d_scratch_bf16, scratch.shared_out, batch_size,
                         hidden, shared_ff, stream);
  return backend->Run(layer_index, static_cast<std::uint32_t>(batch_size),
                      scratch.router_logits, scratch.x_half, scratch.shared_out,
                      scratch.shared_logit, arena_.d_ffn_out, stream);
}

}  // namespace gufo::hip
#endif
