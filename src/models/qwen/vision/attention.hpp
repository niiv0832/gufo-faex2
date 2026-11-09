#ifndef GUFO_MODELS_QWEN_VISION_ATTENTION_HPP_
#define GUFO_MODELS_QWEN_VISION_ATTENTION_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::models::qwen::vision {

inline constexpr unsigned kVisionHeads = 16;
inline constexpr unsigned kVisionHeadDim = 72;
inline constexpr unsigned kVisionHidden = kVisionHeads * kVisionHeadDim;

enum class AttentionKernel {
  // BLAS scores (BF16) -> softmax -> BLAS values, bounded row tiles.
  kTiled,
  // One fused kernel with an online softmax; scores never reach memory.
  kFused,
};

// GUFO_VISION_ATTENTION=tiled selects the BLAS path; anything else is fused.
[[nodiscard]] AttentionKernel DefaultAttentionKernel();

// Non-causal multi-head attention of the vision tower.
// q, k, v: [kVisionHeads][count][kVisionHeadDim] BF16 (RoPE applied).
// output: [count][kVisionHidden] BF16, heads concatenated per token.
void Attention(AttentionKernel kind, hipblasHandle_t blas,
               const hip_bfloat16* q, const hip_bfloat16* k,
               const hip_bfloat16* v, hip_bfloat16* output, unsigned count,
               hipStream_t stream);

}  // namespace gufo::models::qwen::vision
#endif  // GUFO_MODELS_QWEN_VISION_ATTENTION_HPP_
