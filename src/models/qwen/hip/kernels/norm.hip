#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/kernels/staged_norm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"

namespace gufo::hip {

// block_q8_1 of the matrix library: the scale and the scaled sum of the
// quants as halves, then 32 quants.
struct NormQ8Block {
  __half d;
  __half sum;
  std::int8_t qs[32];
};
static_assert(sizeof(NormQ8Block) == 36);

/// a += b followed by out = RMSNorm(a), one block per row of `dim` floats;
/// the sums and their order match ResidualAddKernel followed by the generic
/// RMSNormKernel (and their batched forms).
__global__ void ResidualAddRMSNormKernel(float* __restrict__ rows_a,
                                         const float* __restrict__ rows_b,
                                         const float* __restrict__ weight,
                                         float* __restrict__ rows_out,
                                         std::size_t dim, float eps,
                                         NormQ8Block* __restrict__ rows_q8) {
  __shared__ float s_val[kStagedNormDim];
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;
  float* const a = rows_a + (blockIdx.x * dim);
  const float* const b = rows_b + (blockIdx.x * dim);
  float* const out = rows_out + (blockIdx.x * dim);
  // A row of up to kStagedNormDim floats is staged in LDS so 1024 threads
  // compute it; the squares are then summed by 256 threads in the stride-256
  // order of the 256-thread kernel, which keeps the result bit for bit.
  const bool staged = dim <= kStagedNormDim;
  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = a[i] + b[i];
    a[i] = val;
    if (staged) {
      s_val[i] = val;
    } else {
      sum_sq += val * val;
    }
  }
  const float inv_rms = StagedRowInvRms(s_val, s_sum, sum_sq, dim, eps, staged);
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    const float normed = ((staged ? s_val[i] : a[i]) * inv_rms) * w;
    out[i] = normed;
    if (rows_q8 != nullptr) {
      s_val[i] = normed;
    }
  }
  if (rows_q8 == nullptr) {
    return;
  }
  // The row as block_q8_1, each block of 32 values with quantize_q8_1's
  // expressions: d = max|x| / 127, q = round(x / d), sum = d * sum(q).
  __syncthreads();
  NormQ8Block* const q8 = rows_q8 + (blockIdx.x * (dim / 32));
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const std::size_t base = i & ~std::size_t{31};
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = fmaxf(amax, fabsf(s_val[base + j]));
    }
    const float d = amax / 127.0f;
    const auto quant = [amax, d](float value) -> std::int8_t {
      return amax == 0.0f ? 0 : roundf(value / d);
    };
    q8[i / 32].qs[i % 32] = quant(s_val[i]);
    if (i == base) {
      int sum = 0;
      for (std::size_t j = 0; j < 32; ++j) {
        sum += static_cast<int>(quant(s_val[base + j]));
      }
      q8[i / 32].d = __float2half(d);
      q8[i / 32].sum = __float2half(d * sum);
    }
  }
}

template<std::size_t CachedDim = 0>
__global__ void RMSNormKernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ out, std::size_t dim,
                              float eps) {
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;
  static_assert(CachedDim % 256 == 0);
  constexpr std::size_t cached_items = CachedDim == 0 ? 1 : CachedDim / 256;
  float values[cached_items];
  float weights[cached_items];
  float sum_sq = 0.0F;
  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      const std::size_t i = tid + item * 256;
      const float val = x[i];
      values[item] = val;
      weights[item] = weight != nullptr ? weight[i] : 1.0F;
      // Keep the generic loop's FMA when the compiler unrolls the row.
      sum_sq = __fmaf_rn(val, val, sum_sq);
    }
  } else {
    for (std::size_t i = tid; i < dim; i += blockDim.x) {
      const float val = x[i];
      sum_sq += val * val;
    }
  }
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      out[tid + item * 256] = (values[item] * inv_rms) * weights[item];
    }
  } else {
    for (std::size_t i = tid; i < dim; i += blockDim.x) {
      const float w = (weight != nullptr) ? weight[i] : 1.0F;
      out[i] = (x[i] * inv_rms) * w;
    }
  }
}
__global__ void PerHeadRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     std::uint32_t num_heads,
                                     std::uint32_t head_dim, float eps) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const float* x_head = x + (static_cast<std::size_t>(h) * head_dim);
  float* out_head = out + (static_cast<std::size_t>(h) * head_dim);

  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(head_dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out_head[i] = (x_head[i] * inv_rms) * w;
  }
}
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps, hipStream_t stream) {
  // Qwen27B decode reuses this complete row after the reduction. Retain it
  // in registers, including its weights, without changing the sum tree.
  if (dim == 5120) {
    RMSNormKernel<5120><<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
  } else {
    RMSNormKernel<><<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
  }
}
void LaunchResidualAddRMSNorm(float* a, const float* b, const float* weight,
                              float* out, std::size_t dim, float eps,
                              hipStream_t stream, std::size_t rows, void* q8) {
  ResidualAddRMSNormKernel<<<static_cast<unsigned>(rows),
                             StagedNormBlockThreads(dim), 0, stream>>>(
      a, b, weight, out, dim, eps,
      ResidualAddRMSNormQuantizes(dim) ? static_cast<NormQ8Block*>(q8)
                                       : nullptr);
}
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps, hipStream_t stream) {
  PerHeadRMSNormKernel<<<num_heads, 256, 0, stream>>>(x, weight, out, num_heads,
                                                      head_dim, eps);
}
}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
