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

/// MoE feed-forward for one token with no host involvement: device top-k, the
/// quantized expert vector kernels addressed by the device ids, a gated shared
/// expert and one combine pass. Safe to capture in a HIP graph.
void MoeFfnDecodeToken(const models::QwenLayerWeights& layer,
                       const core::ModelConfig& config,
                       const QwenMoeScratch& scratch, const float* x,
                       float* out, hipStream_t stream);

/// Runs every layer's expert kernels once outside graph capture so the matrix
/// library's workspace pool is sized before the decode graph is captured.
void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
