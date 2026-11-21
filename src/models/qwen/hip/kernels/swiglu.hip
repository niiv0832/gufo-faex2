#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSwiGLUGEMVKernel_1Row(
        const hip_bfloat16* __restrict__ gate_w,
        const hip_bfloat16* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* g_row = gate_w + (m * K);
  const auto* u_row = up_w + (m * K);
  const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
  const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
  const std::size_t K_vec = K / 8;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv = g_vec[vi];
    const uint4 uv = u_vec[vi];
    const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
    const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);

    dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                (static_cast<float>(g_vals[1]) * x0.y) +
                (static_cast<float>(g_vals[2]) * x0.z) +
                (static_cast<float>(g_vals[3]) * x0.w) +
                (static_cast<float>(g_vals[4]) * x1.x) +
                (static_cast<float>(g_vals[5]) * x1.y) +
                (static_cast<float>(g_vals[6]) * x1.z) +
                (static_cast<float>(g_vals[7]) * x1.w);

    dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
              (static_cast<float>(u_vals[1]) * x0.y) +
              (static_cast<float>(u_vals[2]) * x0.z) +
              (static_cast<float>(u_vals[3]) * x0.w) +
              (static_cast<float>(u_vals[4]) * x1.x) +
              (static_cast<float>(u_vals[5]) * x1.y) +
              (static_cast<float>(u_vals[6]) * x1.z) +
              (static_cast<float>(u_vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_gate += static_cast<float>(g_row[k]) * xv;
    dot_up += static_cast<float>(u_row[k]) * xv;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSwiGLUGEMVKernel_2Rows(
        const hip_bfloat16* __restrict__ gate_w,
        const hip_bfloat16* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* g_row0 = gate_w + (m0 * K);
  const auto* u_row0 = up_w + (m0 * K);
  const auto* g_row1 = (m1 < M) ? gate_w + (m1 * K) : nullptr;
  const auto* u_row1 = (m1 < M) ? up_w + (m1 * K) : nullptr;

  const auto* g_vec0 = reinterpret_cast<const uint4*>(g_row0);
  const auto* u_vec0 = reinterpret_cast<const uint4*>(u_row0);
  const auto* g_vec1 = reinterpret_cast<const uint4*>(g_row1);
  const auto* u_vec1 = reinterpret_cast<const uint4*>(u_row1);
  const std::size_t K_vec = K / 8;

  float dot_g0 = 0.0F, dot_u0 = 0.0F;
  float dot_g1 = 0.0F, dot_u1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv0 = g_vec0[vi];
    const uint4 uv0 = u_vec0[vi];
    const auto* g_vals0 = reinterpret_cast<const hip_bfloat16*>(&gv0);
    const auto* u_vals0 = reinterpret_cast<const hip_bfloat16*>(&uv0);

    dot_g0 += (static_cast<float>(g_vals0[0]) * x0.x) +
              (static_cast<float>(g_vals0[1]) * x0.y) +
              (static_cast<float>(g_vals0[2]) * x0.z) +
              (static_cast<float>(g_vals0[3]) * x0.w) +
              (static_cast<float>(g_vals0[4]) * x1.x) +
              (static_cast<float>(g_vals0[5]) * x1.y) +
              (static_cast<float>(g_vals0[6]) * x1.z) +
              (static_cast<float>(g_vals0[7]) * x1.w);

    dot_u0 += (static_cast<float>(u_vals0[0]) * x0.x) +
              (static_cast<float>(u_vals0[1]) * x0.y) +
              (static_cast<float>(u_vals0[2]) * x0.z) +
              (static_cast<float>(u_vals0[3]) * x0.w) +
              (static_cast<float>(u_vals0[4]) * x1.x) +
              (static_cast<float>(u_vals0[5]) * x1.y) +
              (static_cast<float>(u_vals0[6]) * x1.z) +
              (static_cast<float>(u_vals0[7]) * x1.w);

    if (g_row1 != nullptr) {
      const uint4 gv1 = g_vec1[vi];
      const uint4 uv1 = u_vec1[vi];
      const auto* g_vals1 = reinterpret_cast<const hip_bfloat16*>(&gv1);
      const auto* u_vals1 = reinterpret_cast<const hip_bfloat16*>(&uv1);

      dot_g1 += (static_cast<float>(g_vals1[0]) * x0.x) +
                (static_cast<float>(g_vals1[1]) * x0.y) +
                (static_cast<float>(g_vals1[2]) * x0.z) +
                (static_cast<float>(g_vals1[3]) * x0.w) +
                (static_cast<float>(g_vals1[4]) * x1.x) +
                (static_cast<float>(g_vals1[5]) * x1.y) +
                (static_cast<float>(g_vals1[6]) * x1.z) +
                (static_cast<float>(g_vals1[7]) * x1.w);

      dot_u1 += (static_cast<float>(u_vals1[0]) * x0.x) +
                (static_cast<float>(u_vals1[1]) * x0.y) +
                (static_cast<float>(u_vals1[2]) * x0.z) +
                (static_cast<float>(u_vals1[3]) * x0.w) +
                (static_cast<float>(u_vals1[4]) * x1.x) +
                (static_cast<float>(u_vals1[5]) * x1.y) +
                (static_cast<float>(u_vals1[6]) * x1.z) +
                (static_cast<float>(u_vals1[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_g0 += static_cast<float>(g_row0[k]) * xv;
    dot_u0 += static_cast<float>(u_row0[k]) * xv;
    if (g_row1 != nullptr) {
      dot_g1 += static_cast<float>(g_row1[k]) * xv;
      dot_u1 += static_cast<float>(u_row1[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_g0 += __shfl_xor(dot_g0, offset);
    dot_u0 += __shfl_xor(dot_u0, offset);
    dot_g1 += __shfl_xor(dot_g1, offset);
    dot_u1 += __shfl_xor(dot_u1, offset);
  }

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

__global__ void FastFusedSwiGLUGEMVBlockKernel(const void* __restrict__ gate_w,
                                               core::GgmlType gate_type,
                                               const void* __restrict__ up_w,
                                               core::GgmlType up_type,
                                               const float* __restrict__ x,
                                               float* __restrict__ out,
                                               std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_type == core::GgmlType::kBF16 && up_type == core::GgmlType::kBF16) {
    const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
    const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                  (static_cast<float>(g_vals[1]) * x0.y) +
                  (static_cast<float>(g_vals[2]) * x0.z) +
                  (static_cast<float>(g_vals[3]) * x0.w) +
                  (static_cast<float>(g_vals[4]) * x1.x) +
                  (static_cast<float>(g_vals[5]) * x1.y) +
                  (static_cast<float>(g_vals[6]) * x1.z) +
                  (static_cast<float>(g_vals[7]) * x1.w);

      dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                (static_cast<float>(u_vals[1]) * x0.y) +
                (static_cast<float>(u_vals[2]) * x0.z) +
                (static_cast<float>(u_vals[3]) * x0.w) +
                (static_cast<float>(u_vals[4]) * x1.x) +
                (static_cast<float>(u_vals[5]) * x1.y) +
                (static_cast<float>(u_vals[6]) * x1.z) +
                (static_cast<float>(u_vals[7]) * x1.w);
    }
  } else {
    // Per-weight fallback for the non-(both-bf16) route: each of gate/up is
    // computed independently — Q8 (integer dot) or bf16 — based on its own
    // flags. A weight that is neither (legacy fp32) keeps its initialized
    // 0.0 dot so the block reduction collapses to 0, matching the pre-fallback
    // (HEAD) semantics. For Q8 the full row dot is computed in thread 0 only;
    // every other lane contributes 0 so the block reduction below collapses to
    // exactly the row dot (not multiplied by block/thread count).
    if (gate_type == core::GgmlType::kBF16) {
      const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
      const std::size_t K_vec = K / 8;
      const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
      for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
        const uint4 gv = g_vec[vi];
        const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
        const std::size_t k = vi * 8;
        const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
        const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
        dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                    (static_cast<float>(g_vals[1]) * x0.y) +
                    (static_cast<float>(g_vals[2]) * x0.z) +
                    (static_cast<float>(g_vals[3]) * x0.w) +
                    (static_cast<float>(g_vals[4]) * x1.x) +
                    (static_cast<float>(g_vals[5]) * x1.y) +
                    (static_cast<float>(g_vals[6]) * x1.z) +
                    (static_cast<float>(g_vals[7]) * x1.w);
      }
    } else if (gate_type != core::GgmlType::kF32) {
      // Quantized weight (Q8_K now; Q8_0/Q5_K/Q6_K wired later via
      // QuantWarpBlockDot). All lanes cooperate; lane 0 returns this warp's
      // partial row-dot and the block reduction below combines the warps.
      dot_gate = QuantWarpBlockDot(gate_type, gate_w, m, x, K, lane_id, warp_id,
                                   block_threads >> 5u);
    }
    // else (kF32 / legacy fp32) keeps dot_gate == 0.0, matching HEAD semantics.

    if (up_type == core::GgmlType::kBF16) {
      const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
      const std::size_t K_vec = K / 8;
      const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
      for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
        const uint4 uv = u_vec[vi];
        const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
        const std::size_t k = vi * 8;
        const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
        const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
        dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                  (static_cast<float>(u_vals[1]) * x0.y) +
                  (static_cast<float>(u_vals[2]) * x0.z) +
                  (static_cast<float>(u_vals[3]) * x0.w) +
                  (static_cast<float>(u_vals[4]) * x1.x) +
                  (static_cast<float>(u_vals[5]) * x1.y) +
                  (static_cast<float>(u_vals[6]) * x1.z) +
                  (static_cast<float>(u_vals[7]) * x1.w);
      }
    } else if (up_type != core::GgmlType::kF32) {
      // Quantized weight (Q8_K now; Q8_0/Q5_K/Q6_K wired later via
      // QuantWarpBlockDot). All lanes cooperate; lane 0 returns this warp's
      // partial row-dot and the block reduction below combines the warps.
      dot_up = QuantWarpBlockDot(up_type, up_w, m, x, K, lane_id, warp_id,
                                 block_threads >> 5u);
    }
    // else (kF32 / legacy fp32) keeps dot_up == 0.0, matching HEAD semantics.
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[16];
  __shared__ float s_up_sums[16];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      out[m] = (block_gate * sig_g) * block_up;
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQuantSwiGLUGEMVKernel_1Row(const void* __restrict__ gate_w,
                                               core::GgmlType gate_type,
                                               const void* __restrict__ up_w,
                                               core::GgmlType up_type,
                                               const float* __restrict__ x,
                                               float* __restrict__ out,
                                               std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const float dot_gate =
      QuantWarpBlockDot(gate_type, gate_w, m, x, K, lane_id, 0, 1);
  const float dot_up = QuantWarpBlockDot(up_type, up_w, m, x, K, lane_id, 0, 1);

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

template<std::uint32_t WavesPerBlock,
         core::GgmlType StaticType = core::GgmlType::kF32,
         core::GgmlType StaticUpType = StaticType>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQuantSwiGLUGEMVKernel_2Rows(const void* __restrict__ gate_w,
                                                core::GgmlType gate_type,
                                                const void* __restrict__ up_w,
                                                core::GgmlType up_type,
                                                const float* __restrict__ x,
                                                float* __restrict__ out,
                                                std::size_t M, std::size_t K) {
  if constexpr (StaticType != core::GgmlType::kF32) {
    gate_type = StaticType;
  }
  if constexpr (StaticUpType != core::GgmlType::kF32) {
    up_type = StaticUpType;
  }
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t pair_idx = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = pair_idx * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  if (gate_type == core::GgmlType::kQ8_0 && up_type == core::GgmlType::kQ8_0) {
    const std::size_t num_blocks = K / kQ8_0BlockSize;
    const auto* g0_blocks =
        static_cast<const Q8_0Block*>(gate_w) + (m0 * num_blocks);
    const auto* u0_blocks =
        static_cast<const Q8_0Block*>(up_w) + (m0 * num_blocks);
    const auto* g1_blocks =
        (m1 < M) ? (static_cast<const Q8_0Block*>(gate_w) + (m1 * num_blocks))
                 : nullptr;
    const auto* u1_blocks =
        (m1 < M) ? (static_cast<const Q8_0Block*>(up_w) + (m1 * num_blocks))
                 : nullptr;

    float sum_g0 = 0.0F, sum_u0 = 0.0F;
    float sum_g1 = 0.0F, sum_u1 = 0.0F;

    for (std::size_t b = lane_id; b < num_blocks; b += 32) {
      const float* xb = x + (b * kQ8_0BlockSize);
      const Q8_0Block& gb0 = g0_blocks[b];
      const Q8_0Block& ub0 = u0_blocks[b];
      const float dg0 = __half2float(gb0.d);
      const float du0 = __half2float(ub0.d);

      float blk_g0 = 0.0F, blk_u0 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk_g0 += static_cast<float>(gb0.qs[i]) * xb[i];
        blk_u0 += static_cast<float>(ub0.qs[i]) * xb[i];
      }
      sum_g0 += dg0 * blk_g0;
      sum_u0 += du0 * blk_u0;

      if (g1_blocks != nullptr && u1_blocks != nullptr) {
        const Q8_0Block& gb1 = g1_blocks[b];
        const Q8_0Block& ub1 = u1_blocks[b];
        const float dg1 = __half2float(gb1.d);
        const float du1 = __half2float(ub1.d);
        float blk_g1 = 0.0F, blk_u1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          blk_g1 += static_cast<float>(gb1.qs[i]) * xb[i];
          blk_u1 += static_cast<float>(ub1.qs[i]) * xb[i];
        }
        sum_g1 += dg1 * blk_g1;
        sum_u1 += du1 * blk_u1;
      }
    }

    for (int off = 16; off > 0; off >>= 1) {
      sum_g0 += __shfl_xor(sum_g0, off);
      sum_u0 += __shfl_xor(sum_u0, off);
      sum_g1 += __shfl_xor(sum_g1, off);
      sum_u1 += __shfl_xor(sum_u1, off);
    }

    if (lane_id == 0) {
      const float sig_g0 = 1.0F / (1.0F + expf(-sum_g0));
      out[m0] = (sum_g0 * sig_g0) * sum_u0;
      if (m1 < M) {
        const float sig_g1 = 1.0F / (1.0F + expf(-sum_g1));
        out[m1] = (sum_g1 * sig_g1) * sum_u1;
      }
    }
    return;
  }

  const float dot_g0 =
      QuantWarpBlockDot(gate_type, gate_w, m0, x, K, lane_id, 0, 1);
  const float dot_u0 =
      QuantWarpBlockDot(up_type, up_w, m0, x, K, lane_id, 0, 1);
  const float dot_g1 =
      (m1 < M) ? QuantWarpBlockDot(gate_type, gate_w, m1, x, K, lane_id, 0, 1)
               : 0.0F;
  const float dot_u1 =
      (m1 < M) ? QuantWarpBlockDot(up_type, up_w, m1, x, K, lane_id, 0, 1)
               : 0.0F;

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

// Specialized single-token Q8_0 + Q8_0 fused SwiGLU GEMV: removes the generic
// QuantWarpBlockDot switch (cutting VGPR pressure from 95 to ~50) while
// preserving the exact left-to-right FP32 FMA and butterfly reduction order.
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQ8_0SwiGLUGEMVKernel_2Rows(
        const Q8_0Block* __restrict__ gate_w,
        const Q8_0Block* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::uint32_t lane_id = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::size_t pair_idx =
      (static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp_id;
  const std::size_t m0 = pair_idx * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const std::uint32_t num_blocks = static_cast<std::uint32_t>(K / kQ8_0BlockSize);
  const Q8_0Block* g0_blocks = gate_w + (m0 * num_blocks);
  const Q8_0Block* u0_blocks = up_w + (m0 * num_blocks);
  const Q8_0Block* g1_blocks = (m1 < M) ? gate_w + (m1 * num_blocks) : nullptr;
  const Q8_0Block* u1_blocks = (m1 < M) ? up_w + (m1 * num_blocks) : nullptr;

  float sum_g0 = 0.0F;
  float sum_u0 = 0.0F;
  float sum_g1 = 0.0F;
  float sum_u1 = 0.0F;

  for (std::uint32_t b = lane_id; b < num_blocks; b += 32u) {
    const float* xb = x + (b * kQ8_0BlockSize);
    const Q8_0Block& gb0 = g0_blocks[b];
    const Q8_0Block& ub0 = u0_blocks[b];
    sum_g0 += __half2float(gb0.d) * Q8_0BlockDotFp32(gb0, xb);
    sum_u0 += __half2float(ub0.d) * Q8_0BlockDotFp32(ub0, xb);
    if (g1_blocks != nullptr) {
      const Q8_0Block& gb1 = g1_blocks[b];
      const Q8_0Block& ub1 = u1_blocks[b];
      sum_g1 += __half2float(gb1.d) * Q8_0BlockDotFp32(gb1, xb);
      sum_u1 += __half2float(ub1.d) * Q8_0BlockDotFp32(ub1, xb);
    }
  }

  sum_g0 = ReduceWaveXor<16>(sum_g0);
  sum_u0 = ReduceWaveXor<16>(sum_u0);
  sum_g1 = ReduceWaveXor<16>(sum_g1);
  sum_u1 = ReduceWaveXor<16>(sum_u1);

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-sum_g0));
    out[m0] = (sum_g0 * sig_g0) * sum_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-sum_g1));
      out[m1] = (sum_g1 * sig_g1) * sum_u1;
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQ8_0SwiGLUGEMVKernel_1Row(
        const Q8_0Block* __restrict__ gate_w,
        const Q8_0Block* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::uint32_t lane_id = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::size_t m =
      (static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const std::uint32_t num_blocks = static_cast<std::uint32_t>(K / kQ8_0BlockSize);
  const Q8_0Block* g_blocks = gate_w + (m * num_blocks);
  const Q8_0Block* u_blocks = up_w + (m * num_blocks);

  float sum_g = 0.0F;
  float sum_u = 0.0F;
  for (std::uint32_t b = lane_id; b < num_blocks; b += 32u) {
    const float* xb = x + (b * kQ8_0BlockSize);
    const Q8_0Block& gb = g_blocks[b];
    const Q8_0Block& ub = u_blocks[b];
    sum_g += __half2float(gb.d) * Q8_0BlockDotFp32(gb, xb);
    sum_u += __half2float(ub.d) * Q8_0BlockDotFp32(ub, xb);
  }

  sum_g = ReduceWaveXor<16>(sum_g);
  sum_u = ReduceWaveXor<16>(sum_u);

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-sum_g));
    out[m] = (sum_g * sig_g) * sum_u;
  }
}

// Exact fused Q8_0 gate + up projection and SwiGLU activation for small
// verification batches. Each output keeps the scalar Q8_0 block dot, descending
// butterfly reduction, and BatchedSwiGLUActivationKernel formula bit for bit.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool XorStage,
         std::size_t TokensPerStep = Batch>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0FusedSwiGLUKernel(const Q8_0Block* __restrict__ gate_w,
                                         const Q8_0Block* __restrict__ up_w,
                                         const float* __restrict__ x,
                                         float* __restrict__ y, std::uint32_t m,
                                         std::uint32_t k) {
  static_assert(TokensPerStep > 0 && Batch % TokensPerStep == 0);
  constexpr std::uint32_t kBlocksPerTile = 32;
  constexpr std::uint32_t kVectorsPerBlock = kQ8_0BlockSize / 4;
  constexpr std::uint32_t kStride = kQ8_0BlockSize + (XorStage ? 0 : 4);
  constexpr std::uint32_t kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::uint32_t num_blocks = k / kQ8_0BlockSize;
  float sums_g[RowsPerWave][Batch] = {};
  float sums_u[RowsPerWave][Batch] = {};

  for (std::uint32_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    constexpr std::uint32_t kTileVectors =
        Batch * kBlocksPerTile * kVectorsPerBlock;
    for (std::uint32_t flat = threadIdx.x; flat < kTileVectors;
         flat += blockDim.x) {
      const std::uint32_t token = flat / (kBlocksPerTile * kVectorsPerBlock);
      const std::uint32_t within = flat % (kBlocksPerTile * kVectorsPerBlock);
      const std::uint32_t block = within / kVectorsPerBlock;
      const std::uint32_t vector = within % kVectorsPerBlock;
      const std::uint32_t source_block = tile_base + block;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_block < num_blocks) {
        value = *reinterpret_cast<const float4*>(
            x + (token * k) + (source_block * kQ8_0BlockSize) + (vector * 4u));
      }
      *reinterpret_cast<float4*>(
          staged_x + (token * kTileStride) +
          (XorStage ? (vector * 128u) + ((block ^ vector) * 4u)
                    : (block * kStride) + (vector * 4u))) = value;
    }
    SyncKQuantTile();

    const std::uint32_t block = tile_base + lane;
    if (block < num_blocks) {
      const Q8_0Block* g_rows[RowsPerWave];
      const Q8_0Block* u_rows[RowsPerWave];
      float dg[RowsPerWave];
      float du[RowsPerWave];
#pragma unroll
      for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
        const std::uint32_t row = row_base + r;
        const std::uint32_t safe_row = row < m ? row : m - 1u;
        g_rows[r] = &gate_w[(safe_row * num_blocks) + block];
        u_rows[r] = &up_w[(safe_row * num_blocks) + block];
        dg[r] = __half2float(g_rows[r]->d);
        du[r] = __half2float(u_rows[r]->d);
      }
      constexpr bool kPreloadWeights = TokensPerStep < Batch;
      constexpr std::uint32_t kPackedGroups =
          kPreloadWeights ? kVectorsPerBlock : 1;
      std::uint32_t packed_g[RowsPerWave][kPackedGroups];
      std::uint32_t packed_u[RowsPerWave][kPackedGroups];
      if constexpr (kPreloadWeights) {
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (std::uint32_t group = 0; group < kVectorsPerBlock; ++group) {
            __builtin_memcpy(&packed_g[r][group], g_rows[r]->qs + (group * 4),
                             sizeof(std::uint32_t));
            __builtin_memcpy(&packed_u[r][group], u_rows[r]->qs + (group * 4),
                             sizeof(std::uint32_t));
          }
        }
      }
#pragma unroll
      for (std::uint32_t token_base = 0; token_base < Batch;
           token_base += TokensPerStep) {
        if constexpr (kPreloadWeights) {
          __builtin_amdgcn_sched_barrier(0);
        }
        float dots_g[RowsPerWave][TokensPerStep] = {};
        float dots_u[RowsPerWave][TokensPerStep] = {};
#pragma unroll
        for (std::uint32_t group = 0; group < kVectorsPerBlock; ++group) {
          if constexpr (!kPreloadWeights) {
#pragma unroll
            for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
              __builtin_memcpy(&packed_g[r][0], g_rows[r]->qs + (group * 4),
                               sizeof(std::uint32_t));
              __builtin_memcpy(&packed_u[r][0], u_rows[r]->qs + (group * 4),
                               sizeof(std::uint32_t));
            }
          }
#pragma unroll
          for (std::uint32_t lt = 0; lt < TokensPerStep; ++lt) {
            const std::uint32_t token = token_base + lt;
            const float4 xv = *reinterpret_cast<const float4*>(
                staged_x + (token * kTileStride) +
                (XorStage ? (group * 128u) + ((lane ^ group) * 4u)
                          : (lane * kStride) + (group * 4u)));
#pragma unroll
            for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
              const std::uint32_t pg = packed_g[r][kPreloadWeights ? group : 0];
              const std::uint32_t pu = packed_u[r][kPreloadWeights ? group : 0];
              dots_g[r][lt] +=
                  static_cast<float>(static_cast<std::int8_t>(pg & 0xFFU)) *
                  xv.x;
              dots_u[r][lt] +=
                  static_cast<float>(static_cast<std::int8_t>(pu & 0xFFU)) *
                  xv.x;
              dots_g[r][lt] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((pg >> 8U) & 0xFFU)) *
                  xv.y;
              dots_u[r][lt] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((pu >> 8U) & 0xFFU)) *
                  xv.y;
              dots_g[r][lt] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((pg >> 16U) & 0xFFU)) *
                  xv.z;
              dots_u[r][lt] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((pu >> 16U) & 0xFFU)) *
                  xv.z;
              dots_g[r][lt] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((pg >> 24U) & 0xFFU)) *
                  xv.w;
              dots_u[r][lt] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((pu >> 24U) & 0xFFU)) *
                  xv.w;
            }
          }
        }
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (std::uint32_t lt = 0; lt < TokensPerStep; ++lt) {
            sums_g[r][token_base + lt] += dg[r] * dots_g[r][lt];
            sums_u[r][token_base + lt] += du[r] * dots_u[r][lt];
          }
        }
      }
    }
    SyncKQuantTile();
  }

#pragma unroll
  for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::uint32_t token = 0; token < Batch; ++token) {
      sums_g[r][token] = ReduceKQuantWave<16>(sums_g[r][token]);
      sums_u[r][token] = ReduceKQuantWave<16>(sums_u[r][token]);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
      const std::uint32_t row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::uint32_t token = 0; token < Batch; ++token) {
        const float g = sums_g[r][token];
        const float u = sums_u[r][token];
        const float sig_g = 1.0F / (1.0F + expf(-g));
        y[(token * m) + row] = (g * sig_g) * u;
      }
    }
  }
}

void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size, hipStream_t stream) {
  if (gate_type == core::GgmlType::kQ8_0 && up_type == core::GgmlType::kQ8_0 &&
      (hidden_size % kQ8_0BlockSize == 0) &&
      (hidden_size / kQ8_0BlockSize) <=
          std::numeric_limits<std::uint32_t>::max()) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const auto* gw = static_cast<const Q8_0Block*>(gate_w);
    const auto* uw = static_cast<const Q8_0Block*>(up_w);
    if (intermediate_size >= 2 && (intermediate_size % 2 == 0)) {
      const std::size_t num_pairs = intermediate_size / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1U) / kWavesPerBlock;
      Wave32FusedQ8_0SwiGLUGEMVKernel_2Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(gw, uw, x, out,
                                               intermediate_size, hidden_size);
      return;
    }
    const std::size_t grid =
        (intermediate_size + kWavesPerBlock - 1U) / kWavesPerBlock;
    Wave32FusedQ8_0SwiGLUGEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(gw, uw, x, out, intermediate_size,
                                             hidden_size);
    return;
  }
  if (gate_type == core::GgmlType::kBF16 && up_type == core::GgmlType::kBF16 &&
      (hidden_size % 8 == 0)) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

    if (intermediate_size >= 2 && (intermediate_size % 2 == 0)) {
      const std::size_t num_pairs = intermediate_size / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32FusedSwiGLUGEMVKernel_2Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(
              static_cast<const hip_bfloat16*>(gate_w),
              static_cast<const hip_bfloat16*>(up_w), x, out, intermediate_size,
              hidden_size);
      return;
    }

    const std::size_t grid =
        (intermediate_size + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedSwiGLUGEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(gate_w),
            static_cast<const hip_bfloat16*>(up_w), x, out, intermediate_size,
            hidden_size);
    return;
  }

  if (gate_type != core::GgmlType::kF32 && up_type != core::GgmlType::kF32) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

    if (intermediate_size >= 2 && (intermediate_size % 2 == 0)) {
      const std::size_t num_pairs = intermediate_size / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      const auto launch =
          [&]<core::GgmlType StaticType,
              core::GgmlType StaticUpType = StaticType>() {
        Wave32FusedQuantSwiGLUGEMVKernel_2Rows<kWavesPerBlock, StaticType,
                                               StaticUpType>
            <<<grid, kBlockThreads, 0, stream>>>(
                gate_w, gate_type, up_w, up_type, x, out, intermediate_size,
                hidden_size);
      };
      if (gate_type == up_type) {
        switch (gate_type) {
          case core::GgmlType::kQ4_K:
            launch.template operator()<core::GgmlType::kQ4_K>();
            return;
          case core::GgmlType::kQ5_K:
            launch.template operator()<core::GgmlType::kQ5_K>();
            return;
          case core::GgmlType::kIQ4_XS:
            launch.template operator()<core::GgmlType::kIQ4_XS>();
            return;
          default:
            break;
        }
      }
      // Mixed-format FFNs benefit from specializing each projection while
      // keeping the same dot products and SwiGLU arithmetic.
      if (gate_type == core::GgmlType::kQ4_K) {
        if (up_type == core::GgmlType::kQ5_K) {
          launch.template operator()<core::GgmlType::kQ4_K,
                                     core::GgmlType::kQ5_K>();
          return;
        }
        if (up_type == core::GgmlType::kIQ4_XS) {
          launch.template operator()<core::GgmlType::kQ4_K,
                                     core::GgmlType::kIQ4_XS>();
          return;
        }
      }
      if (gate_type == core::GgmlType::kIQ4_XS) {
        if (up_type == core::GgmlType::kQ4_K) {
          launch.template operator()<core::GgmlType::kIQ4_XS,
                                     core::GgmlType::kQ4_K>();
          return;
        }
        if (up_type == core::GgmlType::kQ5_K) {
          launch.template operator()<core::GgmlType::kIQ4_XS,
                                     core::GgmlType::kQ5_K>();
          return;
        }
      }
      launch.template operator()<core::GgmlType::kF32>();
      return;
    }

    const std::size_t grid =
        (intermediate_size + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedQuantSwiGLUGEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(gate_w, gate_type, up_w, up_type,
                                             x, out, intermediate_size,
                                             hidden_size);
    return;
  }

  FastFusedSwiGLUGEMVBlockKernel<<<intermediate_size, 512, 0, stream>>>(
      gate_w, gate_type, up_w, up_type, x, out, intermediate_size, hidden_size);
}

bool TryLaunchBatchedQ8_0FusedSwiGLU(const void* gate_w, const void* up_w,
                                     const float* input, float* output,
                                     std::size_t batch_size,
                                     std::size_t intermediate_size,
                                     std::size_t hidden_size,
                                     hipStream_t stream) {
  if (batch_size < 2 || batch_size > 8 || intermediate_size == 0 ||
      hidden_size == 0 || (hidden_size % kQ8_0BlockSize) != 0) {
    return false;
  }
  const std::size_t row_bytes =
      (hidden_size / kQ8_0BlockSize) * sizeof(Q8_0Block);
  if (!FitsSmallBatch32BitIndices(batch_size, intermediate_size, hidden_size,
                                  row_bytes)) {
    return false;
  }
  const auto* gw = static_cast<const Q8_0Block*>(gate_w);
  const auto* uw = static_cast<const Q8_0Block*>(up_w);
  const auto m = static_cast<std::uint32_t>(intermediate_size);
  const auto k = static_cast<std::uint32_t>(hidden_size);
  const auto launch = [&]<std::size_t Batch, bool XorStage,
                          std::size_t Step = Batch>() {
    constexpr std::uint32_t kWaves = 8;
    constexpr std::size_t kRows = 1;
    constexpr std::size_t kRowsPerBlock = kWaves * kRows;
    SmallBatchQ8_0FusedSwiGLUKernel<kWaves, Batch, kRows, XorStage, Step>
        <<<static_cast<unsigned int>((intermediate_size + kRowsPerBlock - 1) /
                                     kRowsPerBlock),
           kWaves * 32, 0, stream>>>(gw, uw, input, output, m, k);
  };
  switch (batch_size) {
    case 2:
      launch.template operator()<2, true, 2>();
      return true;
    case 3:
      launch.template operator()<3, false, 3>();
      return true;
    case 4:
      launch.template operator()<4, true, 2>();
      return true;
    case 5:
      launch.template operator()<5, true, 5>();
      return true;
    case 6:
      launch.template operator()<6, true, 2>();
      return true;
    case 7:
      launch.template operator()<7, true, 7>();
      return true;
    default:
      launch.template operator()<8, true, 2>();
      return true;
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
