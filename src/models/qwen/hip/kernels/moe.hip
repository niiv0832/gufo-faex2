#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/kernels/staged_norm.hpp"
#include "src/models/qwen/hip/moe_skew.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"

namespace gufo::hip {

namespace {

constexpr std::uint32_t kMoeRouterThreads = 256;

__global__ void MoeRouterTopKKernel(const float* __restrict__ router_logits,
                                    const float* __restrict__ shared_logit,
                                    std::int32_t* __restrict__ expert_ids,
                                    float* __restrict__ expert_weights,
                                    float* __restrict__ shared_weight,
                                    std::uint32_t num_experts,
                                    std::uint32_t num_used) {
  // One block per token row.
  router_logits += static_cast<std::size_t>(blockIdx.x) * num_experts;
  shared_logit += blockIdx.x;
  expert_ids += static_cast<std::size_t>(blockIdx.x) * num_used;
  expert_weights += static_cast<std::size_t>(blockIdx.x) * num_used;
  shared_weight += blockIdx.x;
  extern __shared__ float probs[];
  __shared__ float s_max[8];
  __shared__ float s_sum[8];

  const std::uint32_t tid = threadIdx.x;

  float local_max = -INFINITY;
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, router_logits[e]);
  }
  float warp_max = local_max;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_max = fmaxf(warp_max, __shfl_xor(warp_max, offset));
  }
  const std::uint32_t warp = tid >> 5;
  const std::uint32_t lane = tid & 31U;
  if (lane == 0) {
    s_max[warp] = warp_max;
  }
  __syncthreads();
  const std::uint32_t warp_count = blockDim.x >> 5;
  if (tid == 0) {
    float block_max = s_max[0];
    for (std::uint32_t w = 1; w < warp_count; ++w) {
      block_max = fmaxf(block_max, s_max[w]);
    }
    s_max[0] = block_max;
  }
  __syncthreads();
  const float peak = s_max[0];

  float local_sum = 0.0F;
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    const float v = expf(router_logits[e] - peak);
    probs[e] = v;
    local_sum += v;
  }
  float warp_sum = local_sum;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_sum += __shfl_xor(warp_sum, offset);
  }
  if (lane == 0) {
    s_sum[warp] = warp_sum;
  }
  __syncthreads();
  if (tid == 0) {
    float block_sum = 0.0F;
    for (std::uint32_t w = 0; w < warp_count; ++w) {
      block_sum += s_sum[w];
    }
    s_sum[0] = block_sum;
  }
  __syncthreads();
  const float total = s_sum[0];

  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    probs[e] /= total;
  }
  __syncthreads();

  // Top-k by repeated block-wide argmax: ties resolve toward the lower index,
  // as a stable sort would.
  __shared__ float s_best_value[8];
  __shared__ std::int32_t s_best_index[8];
  __shared__ float s_pick_value[64];
  __shared__ std::int32_t s_pick_index[64];
  const std::uint32_t capacity =
      num_used < 64 ? num_used : static_cast<std::uint32_t>(64);
  for (std::uint32_t k = 0; k < capacity; ++k) {
    float best = -1.0F;
    std::int32_t best_e = 0x7fffffff;
    for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
      const float v = probs[e];
      if (v > best || (v == best && static_cast<std::int32_t>(e) < best_e)) {
        best = v;
        best_e = static_cast<std::int32_t>(e);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_xor(best, offset);
      const std::int32_t other_e = __shfl_xor(best_e, offset);
      if (other > best || (other == best && other_e < best_e)) {
        best = other;
        best_e = other_e;
      }
    }
    if (lane == 0) {
      s_best_value[warp] = best;
      s_best_index[warp] = best_e;
    }
    __syncthreads();
    if (tid == 0) {
      float block_best = s_best_value[0];
      std::int32_t block_e = s_best_index[0];
      for (std::uint32_t w = 1; w < warp_count; ++w) {
        if (s_best_value[w] > block_best ||
            (s_best_value[w] == block_best && s_best_index[w] < block_e)) {
          block_best = s_best_value[w];
          block_e = s_best_index[w];
        }
      }
      s_pick_value[k] = block_best;
      s_pick_index[k] = block_e;
      probs[block_e] = -1.0F;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float selected_sum = 0.0F;
    for (std::uint32_t k = 0; k < capacity; ++k) {
      selected_sum += s_pick_value[k];
    }
    const float inv = selected_sum > 0.0F ? 1.0F / selected_sum : 0.0F;
    for (std::uint32_t k = 0; k < capacity; ++k) {
      expert_ids[k] = s_pick_index[k];
      expert_weights[k] = s_pick_value[k] * inv;
    }
    shared_weight[0] = 1.0F / (1.0F + expf(-shared_logit[0]));
  }
}

__global__ void MoeCombineKernel(float* __restrict__ out,
                                 const float* __restrict__ slot_out,
                                 const float* __restrict__ slot_weight,
                                 const float* __restrict__ shared_out,
                                 const float* __restrict__ shared_weight,
                                 std::uint32_t used, std::uint32_t hidden) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) {
    return;
  }
  const std::size_t row = blockIdx.y;
  out += row * hidden;
  slot_out += row * used * hidden;
  slot_weight += row * used;
  shared_out += row * hidden;
  shared_weight += row;
  float sum = 0.0F;
  for (std::uint32_t s = 0; s < used; ++s) {
    sum += slot_weight[s] * slot_out[static_cast<std::size_t>(s) * hidden + i];
  }
  out[i] = sum + shared_weight[0] * shared_out[i];
}

// MoeCombineKernel, then a += that sum, then out = RMSNorm(a) * weight, one
// block per row: the three launches the end of a MoE layer otherwise takes.
// Each expression and its order are those of MoeCombineKernel and
// ResidualAddRMSNormKernel.
__global__ void MoeCombineResidualAddRMSNormKernel(
    float* __restrict__ rows_a, const float* __restrict__ rows_slot_out,
    const float* __restrict__ rows_slot_weight,
    const float* __restrict__ rows_shared_out,
    const float* __restrict__ rows_shared_weight,
    const float* __restrict__ weight, float* __restrict__ rows_out,
    std::uint32_t used, std::uint32_t hidden, float eps) {
  __shared__ float s_val[kStagedNormDim];
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;
  const std::size_t row = blockIdx.x;
  float* const a = rows_a + (row * hidden);
  const float* const slot_out = rows_slot_out + (row * used * hidden);
  const float* const slot_weight = rows_slot_weight + (row * used);
  const float* const shared_out = rows_shared_out + (row * hidden);
  const float* const shared_weight = rows_shared_weight + row;
  float* const out = rows_out + (row * hidden);
  // Staged like ResidualAddRMSNormKernel: 1024 threads for a short row.
  const bool staged = hidden <= kStagedNormDim;
  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < hidden; i += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t s = 0; s < used; ++s) {
      sum += slot_weight[s] * slot_out[static_cast<std::size_t>(s) * hidden + i];
    }
    const float ffn = sum + shared_weight[0] * shared_out[i];
    const float val = a[i] + ffn;
    a[i] = val;
    if (staged) {
      s_val[i] = val;
    } else {
      sum_sq += val * val;
    }
  }
  const float inv_rms =
      StagedRowInvRms(s_val, s_sum, sum_sq, hidden, eps, staged);
  for (std::size_t i = tid; i < hidden; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out[i] = ((staged ? s_val[i] : a[i]) * inv_rms) * w;
  }
}

// Row `m` of A dotted with x, summed exactly as FastGEMVBlockKernel does so a
// fused launch matches the two GEMVs it replaces bit for bit.
__device__ float RouterRowDot(const void* A, core::GgmlType a_type,
                              std::size_t m, const float* x, std::size_t K,
                              float* s_warp_sums) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;
  float dot = 0.0F;
  if (a_type == core::GgmlType::kBF16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);
    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);
    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x[k];
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();
  float block_sum = 0.0F;
  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
  }
  return block_sum;
}

// One block per expert row plus a last block for the shared-expert gate, for
// every input row (grid y). A row computes exactly what it would alone.
__global__ void MoeRouterLogitsKernel(
    const void* __restrict__ router, core::GgmlType router_type,
    const void* __restrict__ shared, core::GgmlType shared_type,
    const float* __restrict__ rows_x, float* __restrict__ rows_router_logits,
    float* __restrict__ rows_shared_logit, std::size_t experts, std::size_t K) {
  __shared__ float s_warp_sums[16];
  const std::size_t row = blockIdx.y;
  const float* const x = rows_x + (row * K);
  float* const router_logits = rows_router_logits + (row * experts);
  float* const shared_logit = rows_shared_logit + row;
  const std::size_t m = blockIdx.x;
  const bool is_shared = m == experts;
  const float sum = RouterRowDot(is_shared ? shared : router,
                                 is_shared ? shared_type : router_type,
                                 is_shared ? 0 : m, x, K, s_warp_sums);
  if (threadIdx.x == 0) {
    const float r = 0.0F;
    (is_shared ? shared_logit[0] : router_logits[m]) = sum + r;
  }
}

__device__ inline std::uint16_t FloatToBf16Bits(float f) {
  std::uint32_t u = __float_as_uint(f);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<std::uint16_t>(u >> 16);
}

__device__ inline float Bf16BitsToFloat(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

__global__ void RouterSplitActivationsKernel(const float* __restrict__ x,
                                             std::uint16_t* __restrict__ dst,
                                             std::size_t k) {
  const float* row = x + (static_cast<std::size_t>(blockIdx.x) * k);
  std::uint16_t* out = dst + (static_cast<std::size_t>(blockIdx.x) * 3 * k);
  for (std::size_t i = threadIdx.x; i < k; i += blockDim.x) {
    const std::uint16_t hi = FloatToBf16Bits(row[i]);
    const std::uint16_t lo = FloatToBf16Bits(row[i] - Bf16BitsToFloat(hi));
    out[i] = hi;
    out[k + i] = lo;
    out[(2 * k) + i] = hi;
  }
}

__global__ void RouterSplitWeightsKernel(const void* __restrict__ src,
                                         bool src_bf16,
                                         std::uint16_t* __restrict__ dst,
                                         std::size_t first_row, std::size_t k) {
  const std::size_t r = blockIdx.x;
  std::uint16_t* out = dst + ((first_row + r) * 3 * k);
  for (std::size_t i = threadIdx.x; i < k; i += blockDim.x) {
    std::uint16_t hi;
    std::uint16_t lo = 0;
    if (src_bf16) {
      hi = static_cast<const std::uint16_t*>(src)[(r * k) + i];
    } else {
      const float w = static_cast<const float*>(src)[(r * k) + i];
      hi = FloatToBf16Bits(w);
      lo = FloatToBf16Bits(w - Bf16BitsToFloat(hi));
    }
    out[i] = hi;
    out[k + i] = hi;
    out[(2 * k) + i] = lo;
  }
}

__global__ void RouterSplitLogitsKernel(const float* __restrict__ y,
                                        float* __restrict__ router,
                                        float* __restrict__ shared,
                                        std::size_t experts, std::size_t ld) {
  const float* row = y + (static_cast<std::size_t>(blockIdx.x) * ld);
  for (std::size_t i = threadIdx.x; i < experts; i += blockDim.x) {
    router[(static_cast<std::size_t>(blockIdx.x) * experts) + i] = row[i];
  }
  if (threadIdx.x == 0) {
    shared[blockIdx.x] = row[experts];
  }
}

// One block records the router decisions of one layer call: the selection
// histogram and slot weights per row-count bucket, distinct experts per call
// (W2.2), 7th/8th slot <1% tail counts (W5.3), experts shared between the rows
// of a batched call, and -- for single-row calls -- the overlap with the
// layer's previous token. Buffers have fixed addresses (W0.4), so the recorded
// launches are legal inside captured HIP graphs.
__global__ void MoeSkewRecordKernel(const QwenMoeSkewDevice skew,
                                    const std::int32_t* __restrict__ ids,
                                    const float* __restrict__ weights,
                                    std::uint32_t layer_index,
                                    std::uint32_t rows) {
  __shared__ std::uint32_t s_reduce[128];
  __shared__ std::uint32_t s_seen[kMoeSkewMaxExperts];
  if (rows == 0 || layer_index >= skew.layers ||
      skew.experts > kMoeSkewMaxExperts) {
    return;
  }
  const std::uint32_t bucket = rows <= 4 ? rows - 1 : (rows <= 8 ? 4 : 5);
  const std::uint32_t used = skew.used;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t block_threads = blockDim.x;
  const std::size_t layer_base =
      static_cast<std::size_t>(layer_index) * kMoeSkewBuckets;
  // Block-wide sum of a per-thread partial; every thread in the 128-thread
  // block must execute this lambda unconditionally so all barriers match.
  auto block_sum = [&](std::uint32_t value) -> std::uint32_t {
    s_reduce[tid] = value;
    __syncthreads();
    for (std::uint32_t stride = block_threads >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce[tid] += s_reduce[tid + stride];
      }
      __syncthreads();
    }
    const std::uint32_t total = s_reduce[0];
    __syncthreads();
    return total;
  };
  for (std::uint32_t e = tid; e < skew.experts; e += block_threads) {
    s_seen[e] = 0U;
  }
  __syncthreads();
  std::uint32_t local_tail7 = 0;
  std::uint32_t local_tail8 = 0;
  for (std::uint32_t index = tid; index < rows * used; index += block_threads) {
    const std::uint32_t slot = index % used;
    const std::int32_t id = ids[index];
    const float w = weights[index];
    if (id >= 0 && static_cast<std::uint32_t>(id) < skew.experts) {
      atomicAdd(&skew.hist[(layer_base + bucket) * skew.experts + id], 1);
      atomicAdd(&skew.slot_weights[(layer_base + bucket) * used + slot], w);
      s_seen[id] = 1U;
    }
    if (used >= 2) {
      if (slot == used - 2 && w < 0.01F) {
        ++local_tail7;
      } else if (slot == used - 1 && w < 0.01F) {
        ++local_tail8;
      }
    }
  }
  __syncthreads();
  std::uint32_t local_unique = 0;
  for (std::uint32_t e = tid; e < skew.experts; e += block_threads) {
    local_unique += s_seen[e];
  }
  const std::uint32_t total_unique = block_sum(local_unique);
  const std::uint32_t total_tail7 = block_sum(local_tail7);
  const std::uint32_t total_tail8 = block_sum(local_tail8);
  if (tid == 0) {
    atomicAdd(&skew.unique_experts[layer_base + bucket],
              static_cast<unsigned long long>(total_unique));
    if (total_tail7 != 0) {
      atomicAdd(&skew.tail_lt1pct[(layer_base + bucket) * 2 + 0],
                static_cast<unsigned long long>(total_tail7));
    }
    if (total_tail8 != 0) {
      atomicAdd(&skew.tail_lt1pct[(layer_base + bucket) * 2 + 1],
                static_cast<unsigned long long>(total_tail8));
    }
  }
  // Experts shared between two rows of one call: every pair contributes the
  // number of ids present in both rows, out of `used` each. `pair` enumerates
  // (first, second) with first < second row by row.
  if (rows > 1) {
    const std::uint32_t pair_count = rows * (rows - 1) / 2;
    std::uint32_t shared = 0;
    for (std::uint32_t pair = tid; pair < pair_count; pair += block_threads) {
      std::uint32_t first = 0;
      std::uint32_t rest = pair;
      while (rest >= rows - 1 - first) {
        rest -= rows - 1 - first;
        ++first;
      }
      const std::uint32_t second = first + 1 + rest;
      for (std::uint32_t a = 0; a < used; ++a) {
        const std::int32_t id = ids[first * used + a];
        if (id < 0) {
          continue;
        }
        for (std::uint32_t b = 0; b < used; ++b) {
          if (ids[second * used + b] == id) {
            ++shared;
            break;
          }
        }
      }
    }
    const std::uint32_t total_shared = block_sum(shared);
    if (tid == 0) {
      atomicAdd(&skew.dup[layer_base + bucket],
                static_cast<unsigned long long>(total_shared));
      atomicAdd(&skew.dup_pairs[layer_base + bucket],
                static_cast<unsigned long long>(pair_count));
    }
  }
  // Neighbor-token overlap: single-row calls compare against the ids of the
  // layer's previous single-row call (skipping the first token when prev_ids
  // is -1), then become the new previous ids. The block reduction syncs
  // separate the reads of the old ids from the writes.
  if (rows == 1) {
    const bool has_prev = skew.prev_ids[layer_index * used] >= 0;
    std::uint32_t shared = 0;
    if (has_prev) {
      for (std::uint32_t a = tid; a < used; a += block_threads) {
        const std::int32_t id = ids[a];
        if (id < 0) {
          continue;
        }
        for (std::uint32_t b = 0; b < used; ++b) {
          if (skew.prev_ids[layer_index * used + b] == id) {
            ++shared;
            break;
          }
        }
      }
    }
    const std::uint32_t total_overlap = block_sum(shared);
    if (tid == 0 && has_prev) {
      atomicAdd(&skew.overlap[layer_index],
                static_cast<unsigned long long>(total_overlap));
      atomicAdd(&skew.overlap_rows[layer_index], 1ULL);
    }
    for (std::uint32_t a = tid; a < used; a += block_threads) {
      skew.prev_ids[layer_index * used + a] = ids[a];
    }
  }
  if (tid == 0) {
    atomicAdd(&skew.rows[layer_base + bucket], 1ULL);
    atomicAdd(&skew.tokens[layer_base + bucket],
              static_cast<unsigned long long>(rows));
  }
}

}  // namespace

void LaunchMoeRouterLogits(const void* router, core::GgmlType router_type,
                           const void* shared, core::GgmlType shared_type,
                           const float* x, float* router_logits,
                           float* shared_logit, std::size_t experts,
                           std::size_t hidden, hipStream_t stream,
                           std::size_t rows) {
  MoeRouterLogitsKernel<<<dim3(static_cast<unsigned>(experts + 1),
                               static_cast<unsigned>(rows)),
                          256, 0, stream>>>(
      router, router_type, shared, shared_type, x, router_logits, shared_logit,
      experts, hidden);
}

__global__ void ConcatRowsKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ out, std::size_t cols) {
  const float* ra = a + (static_cast<std::size_t>(blockIdx.x) * cols);
  const float* rb = b + (static_cast<std::size_t>(blockIdx.x) * cols);
  float* ro = out + (static_cast<std::size_t>(blockIdx.x) * 2 * cols);
  for (std::size_t i = threadIdx.x; i < cols; i += blockDim.x) {
    ro[i] = ra[i];
    ro[cols + i] = rb[i];
  }
}

void LaunchConcatRows(const float* a, const float* b, float* out,
                      std::size_t rows, std::size_t cols, hipStream_t stream) {
  ConcatRowsKernel<<<dim3(rows), 256, 0, stream>>>(a, b, out, cols);
}

void LaunchRouterSplitActivations(const float* x, std::uint16_t* dst,
                                  std::size_t rows, std::size_t k,
                                  hipStream_t stream) {
  RouterSplitActivationsKernel<<<dim3(rows), 256, 0, stream>>>(x, dst, k);
}

void LaunchRouterSplitWeights(const void* src, bool src_bf16,
                              std::uint16_t* dst, std::size_t first_row,
                              std::size_t rows, std::size_t k,
                              hipStream_t stream) {
  RouterSplitWeightsKernel<<<dim3(rows), 256, 0, stream>>>(src, src_bf16, dst,
                                                           first_row, k);
}

void LaunchRouterSplitLogits(const float* y, float* router, float* shared,
                             std::size_t rows, std::size_t experts,
                             std::size_t ld, hipStream_t stream) {
  RouterSplitLogitsKernel<<<dim3(rows), 256, 0, stream>>>(y, router, shared,
                                                          experts, ld);
}

void LaunchMoeRouterTopK(const float* router_logits, const float* shared_logit,
                         std::int32_t* expert_ids, float* expert_weights,
                         float* shared_weight, std::uint32_t num_experts,
                         std::uint32_t num_used, std::uint32_t rows,
                         hipStream_t stream) {
  const std::size_t shared_bytes =
      static_cast<std::size_t>(num_experts) * sizeof(float);
  MoeRouterTopKKernel<<<rows, kMoeRouterThreads, shared_bytes, stream>>>(
      router_logits, shared_logit, expert_ids, expert_weights, shared_weight,
      num_experts, num_used);
}

void LaunchMoeSkewRecord(const QwenMoeSkewDevice& skew,
                         std::uint32_t layer_index, std::size_t rows,
                         const std::int32_t* ids, const float* weights,
                         hipStream_t stream) {
  MoeSkewRecordKernel<<<1, 128, 0, stream>>>(
      skew, ids, weights, layer_index,
      static_cast<std::uint32_t>(
          rows < kMoeSkewMaxRows ? rows : kMoeSkewMaxRows));
}

void LaunchMoeCombineResidualAddRMSNorm(
    float* hidden_rows, const float* slot_out, const float* slot_weight,
    const float* shared_out, const float* shared_weight, const float* weight,
    float* out, std::uint32_t used, std::uint32_t hidden, float eps,
    std::uint32_t rows, hipStream_t stream) {
  MoeCombineResidualAddRMSNormKernel<<<rows, StagedNormBlockThreads(hidden),
                                       0, stream>>>(
      hidden_rows, slot_out, slot_weight, shared_out, shared_weight, weight,
      out, used, hidden, eps);
}

void LaunchMoeCombine(float* out, const float* slot_out,
                      const float* slot_weight, const float* shared_out,
                      const float* shared_weight, std::uint32_t used,
                      std::uint32_t hidden, std::uint32_t rows,
                      hipStream_t stream) {
  MoeCombineKernel<<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(
      out, slot_out, slot_weight, shared_out, shared_weight, used, hidden);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
