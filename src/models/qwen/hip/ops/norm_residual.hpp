#ifndef GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Computes RMSNorm on GPU: out = (x / sqrt(mean(x^2) + eps)) * weight
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps = 1e-6F,
                   hipStream_t stream = nullptr);

/// Computes per-head RMSNorm across num_heads
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps = 1e-6F, hipStream_t stream = nullptr);

/// Whether LaunchResidualAddRMSNorm can also write the block_q8_1 form of
/// its rows of `dim` floats.
[[nodiscard]] constexpr bool ResidualAddRMSNormQuantizes(
    std::size_t dim) noexcept {
  return dim <= 2048 && dim % 512 == 0;
}

/// Bytes of one row of `dim` floats as block_q8_1: 32 quants and two halves
/// per block.
[[nodiscard]] constexpr std::size_t Q8_1RowBytes(std::size_t dim) noexcept {
  return (dim / 32) * 36;
}

/// a += b, then out = RMSNorm(a) * weight, in one launch (dim not 5120), for
/// `rows` consecutive rows of `dim` floats. A non-null `q8` (only where
/// ResidualAddRMSNormQuantizes(dim)) also receives the rows of `out` as
/// block_q8_1, the bytes the matrix library's quantize_q8_1 writes for them.
void LaunchResidualAddRMSNorm(float* a, const float* b, const float* weight,
                              float* out, std::size_t dim, float eps = 1e-6F,
                              hipStream_t stream = nullptr,
                              std::size_t rows = 1, void* q8 = nullptr);

/// Computes residual add: out = a + b
void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens (optional BF16 output in single pass)
void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps = 1e-6F,
                          hipStream_t stream = nullptr);

/// Batched Per-Head RMSNorm across B tokens
void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps = 1e-6F,
                                 hipStream_t stream = nullptr);

/// Batched Residual Add across B tokens
void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
