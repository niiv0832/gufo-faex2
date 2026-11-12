#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_STAGED_NORM_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_STAGED_NORM_HPP_

// The shared tail of the one-block-per-row RMSNorm kernels that stage a short
// row in LDS. Included by kernel sources only.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>

namespace gufo::hip {

/// Longest row a block stages in LDS; longer rows keep 256 threads.
inline constexpr std::size_t kStagedNormDim = 2048;

inline unsigned StagedNormBlockThreads(std::size_t dim) {
  return dim <= kStagedNormDim ? 1024U : 256U;
}

/// 1 / RMS of one row. Staged: the row is in `s_val` and threads 0..255 sum
/// its squares with stride 256, the order a 256-thread block uses. Otherwise
/// the block has 256 threads and `sum_sq` is this thread's sum. Either way the
/// 256 partial sums meet in the same tree.
__device__ inline float StagedRowInvRms(const float* s_val, float* s_sum,
                                        float sum_sq, std::size_t dim,
                                        float eps, bool staged) {
  const std::size_t tid = threadIdx.x;
  if (staged) {
    __syncthreads();
    if (tid < 256) {
      sum_sq = 0.0F;
      for (std::size_t i = tid; i < dim; i += 256) {
        const float val = s_val[i];
        sum_sq += val * val;
      }
    }
  }
  if (tid < 256) {
    s_sum[tid] = sum_sq;
  }
  __syncthreads();
  for (std::size_t stride = 128; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  return rsqrtf(mean_sq + eps);
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_STAGED_NORM_HPP_
