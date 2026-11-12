#ifndef GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
#define GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Rows the arena's MoE scratch holds: eight sessions each verifying eight
/// draft tokens.
inline constexpr std::size_t kMoeFfnScratchRows = 64;

/// MoE feed-forward for `rows` independent rows of `hidden` floats on the
/// arena's stream, with no host involvement: one router launch, device top-k,
/// the expert vector kernels in chunks of eight rows, the shared expert and
/// one combine pass, in pieces of kMoeFfnScratchRows rows. Each row gets the
/// arithmetic of MoeFfnDecodeToken. Batched verification and prompt chunks too
/// short for the grouped-expert prefill use it.
void MoeFfnRows(const models::QwenLayerWeights& layer,
                const core::ModelConfig& config, QwenGpuArena& arena,
                const float* x, float* out, std::size_t rows);

/// The last steps of a MoE layer for `rows` rows whose expert outputs are in
/// the arena's MoE scratch (MoeFfnDecodeToken or the batched rows with a null
/// `out`): hidden += combined experts, normed = RMSNorm(hidden) * norm_weight.
void MoeFfnFinishRows(const core::ModelConfig& config,
                      const QwenMoeScratch& scratch, float* hidden,
                      const float* norm_weight, float* normed, std::size_t rows,
                      hipStream_t stream);

/// MoE feed-forward for one token with no host involvement: device top-k, the
/// quantized expert vector kernels addressed by the device ids, a gated shared
/// expert and one combine pass. Safe to capture in a HIP graph. A null `out`
/// leaves the expert outputs in `scratch` for MoeFfnFinishRows. With
/// `x_quantized`, scratch.x_q8 already holds x as block_q8_1 and the expert
/// gate/up kernels skip their own quantization pass.
void MoeFfnDecodeToken(const models::QwenLayerWeights& layer,
                       const core::ModelConfig& config,
                       const QwenMoeScratch& scratch, const float* x,
                       float* out, hipStream_t stream,
                       bool x_quantized = false);

/// Runs every layer's expert kernels once outside graph capture so the matrix
/// library's workspace pool is sized before the decode graph is captured.
void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
