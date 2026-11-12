#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_ffn.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/moe_vec.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"

namespace gufo::hip {

void MoeFfnDecodeToken(const models::QwenLayerWeights& layer,
                       const core::ModelConfig& config,
                       const QwenMoeScratch& scratch, const float* x,
                       float* out, hipStream_t stream, bool x_quantized) {
  const int experts = static_cast<int>(config.num_experts);
  const int used = static_cast<int>(config.num_experts_used);
  const int ff = static_cast<int>(config.expert_ff);
  const int hidden = static_cast<int>(config.hidden_size);
  const int shared_ff = static_cast<int>(config.shared_expert_ff);

  LaunchMoeRouterLogits(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type,
                        layer.ffn_gate_inp_shexp.data,
                        layer.ffn_gate_inp_shexp.type, x, scratch.router_logits,
                        scratch.shared_logit, experts, hidden, stream);
  LaunchMoeRouterTopK(scratch.router_logits, scratch.shared_logit, scratch.ids,
                      scratch.weights, scratch.shared_weight,
                      static_cast<std::uint32_t>(experts),
                      static_cast<std::uint32_t>(used), 1, stream);
  // activation[s] = SiLU(gate_e x) * (up_e x) for the selected experts e.
  if ((x_quantized
           ? qfn_mmq_moe_gated_vec_preq(
                 static_cast<int>(layer.ffn_gate_exps.type),
                 layer.ffn_gate_exps.data, layer.ffn_up_exps.data, scratch.x_q8,
                 scratch.ids, scratch.act, ff, hidden, 1, experts, used, stream)
           : qfn_mmq_moe_gated_vec(static_cast<int>(layer.ffn_gate_exps.type),
                                   layer.ffn_gate_exps.data,
                                   layer.ffn_up_exps.data, x, scratch.ids,
                                   scratch.act, ff, hidden, 1, experts, used,
                                   stream)) != 0) {
    throw std::runtime_error("qwen35moe expert gate/up projection failed");
  }
  // Each slot row multiplies its own expert's down matrix: used rows, one
  // expert each.
  if (qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                      layer.ffn_down_exps.data, scratch.act, scratch.ids,
                      scratch.down_out, hidden, ff, used, experts, 1, stream,
                      nullptr, nullptr) != 0) {
    throw std::runtime_error("qwen35moe expert down projection failed");
  }
  LaunchFusedSwiGLUGEMV(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
                        layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, x,
                        scratch.shared_act, shared_ff, hidden, stream);
  LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
             scratch.shared_act, scratch.shared_out, hidden, shared_ff, stream);
  if (out != nullptr) {
    LaunchMoeCombine(out, scratch.down_out, scratch.weights, scratch.shared_out,
                     scratch.shared_weight, static_cast<std::uint32_t>(used),
                     static_cast<std::uint32_t>(hidden), 1, stream);
  }
}

void MoeFfnFinishRows(const core::ModelConfig& config,
                      const QwenMoeScratch& scratch, float* hidden,
                      const float* norm_weight, float* normed, std::size_t rows,
                      hipStream_t stream) {
  LaunchMoeCombineResidualAddRMSNorm(
      hidden, scratch.down_out, scratch.weights, scratch.shared_out,
      scratch.shared_weight, norm_weight, normed,
      static_cast<std::uint32_t>(config.num_experts_used),
      static_cast<std::uint32_t>(config.hidden_size), 1e-6F,
      static_cast<std::uint32_t>(rows), stream);
}

void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream) {
  const int experts = static_cast<int>(config.num_experts);
  const int used = static_cast<int>(config.num_experts_used);
  const int ff = static_cast<int>(config.expert_ff);
  const int hidden = static_cast<int>(config.hidden_size);
  HIP_CHECK(
      hipMemsetAsync(scratch.ids, 0, used * sizeof(std::int32_t), stream));
  for (const auto& layer : weights.layers) {
    (void)qfn_mmq_moe_gated_vec(
        static_cast<int>(layer.ffn_gate_exps.type), layer.ffn_gate_exps.data,
        layer.ffn_up_exps.data, x, scratch.ids, scratch.act, ff, hidden, 1,
        experts, used, stream);
    (void)qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                          layer.ffn_down_exps.data, scratch.act, scratch.ids,
                          scratch.down_out, hidden, ff, used, experts, 1,
                          stream, nullptr, nullptr);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
