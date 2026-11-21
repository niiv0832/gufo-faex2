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
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSSMInputProjectionsKernel_1Row(
        const void* __restrict__ qkv_w, core::GgmlType qkv_type,
        const void* __restrict__ gate_w, core::GgmlType gate_type,
        const void* __restrict__ alpha_w, core::GgmlType alpha_type,
        const void* __restrict__ beta_w, core::GgmlType beta_type,
        const float* __restrict__ x, float* __restrict__ qkv_out,
        float* __restrict__ gate_out, float* __restrict__ alpha_out,
        float* __restrict__ beta_out, std::size_t hidden_size,
        std::size_t qkv_size, std::size_t inner_size,
        std::size_t time_step_rank) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    weight_type = qkv_type;
    out_target = qkv_out;
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    weight_type = gate_type;
    out_target = gate_out;
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    weight_type = alpha_type;
    out_target = alpha_out;
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    weight_type = beta_type;
    out_target = beta_out;
    row_idx = m - alpha_end;
  }

  float dot = 0.0F;

  if (weight_type == core::GgmlType::kBF16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
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
    for (std::size_t k = (K_vec * 8) + lane_id; k < hidden_size; k += 32) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else if (weight_type == core::GgmlType::kQ8_0) {
    const std::size_t num_blocks = hidden_size / kQ8_0BlockSize;
    const auto* row_blocks =
        static_cast<const Q8_0Block*>(A) + (row_idx * num_blocks);
    for (std::size_t b = lane_id; b < num_blocks; b += 32) {
      const float* xb = x + (b * kQ8_0BlockSize);
      const Q8_0Block& wblk = row_blocks[b];
      const float d = __half2float(wblk.d);
      float blk_dot = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk_dot += static_cast<float>(wblk.qs[i]) * xb[i];
      }
      dot += d * blk_dot;
    }
  } else if (weight_type != core::GgmlType::kF32) {
    dot = QuantWarpBlockDot(weight_type, A, row_idx, x, hidden_size, lane_id, 0,
                            1);
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + lane_id; k < hidden_size; k += 32) {
      dot += row[k] * x[k];
    }
  }

  if (weight_type == core::GgmlType::kBF16 ||
      weight_type == core::GgmlType::kF32 ||
      weight_type == core::GgmlType::kQ8_0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_xor(dot, offset);
    }
  }

  if (lane_id == 0) {
    out_target[row_idx] = dot;
  }
}

__global__ void FusedSSMInputProjectionsKernel(
    const void* __restrict__ qkv_w, core::GgmlType qkv_type,
    const void* __restrict__ gate_w, core::GgmlType gate_type,
    const void* __restrict__ alpha_w, core::GgmlType alpha_type,
    const void* __restrict__ beta_w, core::GgmlType beta_type,
    const float* __restrict__ x, float* __restrict__ qkv_out,
    float* __restrict__ gate_out, float* __restrict__ alpha_out,
    float* __restrict__ beta_out, std::size_t hidden_size, std::size_t qkv_size,
    std::size_t inner_size, std::size_t time_step_rank) {
  const std::size_t m = blockIdx.x;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    weight_type = qkv_type;
    out_target = qkv_out;
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    weight_type = gate_type;
    out_target = gate_out;
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    weight_type = alpha_type;
    out_target = alpha_out;
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    weight_type = beta_type;
    out_target = beta_out;
    row_idx = m - alpha_end;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (weight_type == core::GgmlType::kBF16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
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
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else if (weight_type != core::GgmlType::kF32) {
    // Quantized weight: route via QuantWarpBlockDot (block_q8_K now; Q8_0/Q5_K/
    // Q6_K wired in later tasks). All lanes cooperate; lane 0 returns this
    // warp's partial row-dot and the warp + cross-warp block reduction below
    // combines the warps (num_warps = block_threads >> 5u).
    dot = QuantWarpBlockDot(weight_type, A, row_idx, x, hidden_size, lane_id,
                            warp_id, block_threads >> 5u);
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

// Specialized single-token Q8_0 (qkv, gate) + F32 (alpha, beta) SSM input
// projection: drops the generic QuantWarpBlockDot switch and splits the F32
// control tail at a block-uniform boundary while keeping bit-exact arithmetic.
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSSMInputQ8_0F32Kernel_1Row(
        const Q8_0Block* __restrict__ qkv_w,
        const Q8_0Block* __restrict__ gate_w, const float* __restrict__ alpha_w,
        const float* __restrict__ beta_w, const float* __restrict__ x,
        float* __restrict__ qkv_out, float* __restrict__ gate_out,
        float* __restrict__ alpha_out, float* __restrict__ beta_out,
        std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
        std::size_t time_step_rank) {
  const std::uint32_t lane_id = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::size_t block_m =
      static_cast<std::size_t>(blockIdx.x) * WavesPerBlock;
  const std::size_t m = block_m + warp_id;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows) {
    return;
  }

  if (block_m >= gate_end) {
    const bool is_alpha = m < alpha_end;
    const float* A = is_alpha ? alpha_w : beta_w;
    float* out_target = is_alpha ? alpha_out : beta_out;
    const std::size_t row_idx = is_alpha ? (m - gate_end) : (m - alpha_end);
    const auto* row_vec =
        reinterpret_cast<const float4*>(A + (row_idx * hidden_size));
    const auto* x_vec = reinterpret_cast<const float4*>(x);
    const std::uint32_t k_vec = static_cast<std::uint32_t>(hidden_size / 4);
    float dot = 0.0F;
    for (std::uint32_t vi = lane_id; vi < k_vec; vi += 32u) {
      const float4 v = row_vec[vi];
      const float4 xv = x_vec[vi];
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    dot = ReduceWaveXor<16>(dot);
    if (lane_id == 0) {
      out_target[row_idx] = dot;
    }
    return;
  }

  const bool is_qkv = m < qkv_size;
  const Q8_0Block* A = is_qkv ? qkv_w : gate_w;
  float* out_target = is_qkv ? qkv_out : gate_out;
  const std::size_t row_idx = is_qkv ? m : (m - qkv_size);
  const std::uint32_t num_blocks =
      static_cast<std::uint32_t>(hidden_size / kQ8_0BlockSize);
  const Q8_0Block* row_blocks = A + (row_idx * num_blocks);
  float dot = 0.0F;
  for (std::uint32_t b = lane_id; b < num_blocks; b += 32u) {
    const Q8_0Block& wblk = row_blocks[b];
    dot += __half2float(wblk.d) *
           Q8_0BlockDotFp32(wblk, x + (b * kQ8_0BlockSize));
  }
  dot = ReduceWaveXor<16>(dot);
  if (lane_id == 0) {
    out_target[row_idx] = dot;
  }
}

// Exact batched F32 SSM alpha/beta controls in a single launch, matching the
// per-row F32 reduction tree of Wave32FusedSSMInputProjectionsKernel_1Row.
template<std::uint32_t WavesPerBlock, std::size_t Batch>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void BatchedF32SsmControlsKernel(const float* __restrict__ alpha_w,
                                     const float* __restrict__ beta_w,
                                     const float* __restrict__ x,
                                     float* __restrict__ alpha_out,
                                     float* __restrict__ beta_out,
                                     std::uint32_t rank, std::uint32_t k) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= 2u * rank) {
    return;
  }

  const bool is_alpha = m < rank;
  const float* A = is_alpha ? alpha_w : beta_w;
  float* out_target = is_alpha ? alpha_out : beta_out;
  const std::uint32_t lr = is_alpha ? m : (m - rank);
  const auto* rv = reinterpret_cast<const float4*>(
      A + (static_cast<std::size_t>(lr) * k));
  const std::uint32_t kv = k / 4u;

  float dots[Batch] = {};
  for (std::uint32_t vi = lane; vi < kv; vi += 32u) {
    const float4 v = rv[vi];
#pragma unroll
    for (std::uint32_t token = 0; token < Batch; ++token) {
      const float4 xv =
          reinterpret_cast<const float4*>(x + (token * k))[vi];
      dots[token] += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
  }
#pragma unroll
  for (std::uint32_t token = 0; token < Batch; ++token) {
    dots[token] = ReduceWaveXor<16>(dots[token]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t token = 0; token < Batch; ++token) {
      out_target[(token * rank) + lr] = dots[token];
    }
  }
}

// Exact fused small-batch Q8_0 (qkv, gate) + F32 (alpha, beta) SSM input
// projection. Requires qkv_size and inner_size divisible by WavesPerBlock *
// RowsPerWave so workgroups split cleanly between Q8_0 and F32 paths.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool XorStage>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0FusedSSMInputKernel(
        const Q8_0Block* __restrict__ qkv_w,
        const Q8_0Block* __restrict__ gate_w, const float* __restrict__ alpha_w,
        const float* __restrict__ beta_w, const float* __restrict__ x,
        float* __restrict__ qkv_out, float* __restrict__ gate_out,
        float* __restrict__ alpha_out, float* __restrict__ beta_out,
        std::uint32_t qkv_sz, std::uint32_t gate_sz, std::uint32_t rank,
        std::uint32_t k) {
  constexpr std::uint32_t kRowsPerBlock = WavesPerBlock * RowsPerWave;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t block_row_base = blockIdx.x * kRowsPerBlock;
  const std::uint32_t gate_end = qkv_sz + gate_sz;
  const std::uint32_t alpha_end = gate_end + rank;
  const std::uint32_t total_m = alpha_end + rank;

  if (block_row_base >= gate_end) {
    const std::uint32_t row_base = block_row_base + (warp_id * RowsPerWave);
    const std::uint32_t kv = k / 4u;
#pragma unroll
    for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
      const std::uint32_t m = row_base + r;
      if (m >= total_m) {
        continue;
      }
      const bool is_alpha = m < alpha_end;
      const float* A = is_alpha ? alpha_w : beta_w;
      float* out_target = is_alpha ? alpha_out : beta_out;
      const std::uint32_t lr = is_alpha ? (m - gate_end) : (m - alpha_end);
      const auto* rv = reinterpret_cast<const float4*>(
          A + (static_cast<std::size_t>(lr) * k));
      float dots[Batch] = {};
      for (std::uint32_t vi = lane; vi < kv; vi += 32u) {
        const float4 v = rv[vi];
#pragma unroll
        for (std::uint32_t token = 0; token < Batch; ++token) {
          const float4 xv =
              reinterpret_cast<const float4*>(x + (token * k))[vi];
          dots[token] +=
              (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
        }
      }
#pragma unroll
      for (std::uint32_t token = 0; token < Batch; ++token) {
        dots[token] = ReduceKQuantWave<16>(dots[token]);
      }
      if (lane == 0) {
#pragma unroll
        for (std::uint32_t token = 0; token < Batch; ++token) {
          out_target[(token * rank) + lr] = dots[token];
        }
      }
    }
    return;
  }

  constexpr std::uint32_t kBlocksPerTile = 32;
  constexpr std::uint32_t kVectorsPerBlock = kQ8_0BlockSize / 4;
  constexpr std::uint32_t kStride = kQ8_0BlockSize + (XorStage ? 0 : 4);
  constexpr std::uint32_t kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::uint32_t row_base = block_row_base + (warp_id * RowsPerWave);
  const std::uint32_t num_blocks = k / kQ8_0BlockSize;
  const bool is_qkv = block_row_base < qkv_sz;
  const Q8_0Block* A = is_qkv ? qkv_w : gate_w;
  float* out_ptr = is_qkv ? qkv_out : gate_out;
  const std::uint32_t out_stride = is_qkv ? qkv_sz : gate_sz;
  const std::uint32_t local_row_base = is_qkv ? row_base : (row_base - qkv_sz);

  float sums[RowsPerWave][Batch] = {};
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
      const Q8_0Block* rows[RowsPerWave];
      float scale[RowsPerWave];
#pragma unroll
      for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
        const std::uint32_t lr = local_row_base + r;
        const std::uint32_t safe_lr = lr < out_stride ? lr : out_stride - 1u;
        rows[r] = &A[(safe_lr * num_blocks) + block];
        scale[r] = __half2float(rows[r]->d);
      }
      float block_dots[RowsPerWave][Batch] = {};
#pragma unroll
      for (std::uint32_t group = 0; group < kVectorsPerBlock; ++group) {
        std::uint32_t packed[RowsPerWave];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
          __builtin_memcpy(&packed[r], rows[r]->qs + (group * 4),
                           sizeof(std::uint32_t));
        }
#pragma unroll
        for (std::uint32_t token = 0; token < Batch; ++token) {
          const float4 xv = *reinterpret_cast<const float4*>(
              staged_x + (token * kTileStride) +
              (XorStage ? (group * 128u) + ((lane ^ group) * 4u)
                        : (lane * kStride) + (group * 4u)));
#pragma unroll
          for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
            const std::uint32_t p = packed[r];
            block_dots[r][token] +=
                static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
            block_dots[r][token] +=
                static_cast<float>(
                    static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
                xv.y;
            block_dots[r][token] +=
                static_cast<float>(
                    static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
                xv.z;
            block_dots[r][token] +=
                static_cast<float>(
                    static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
                xv.w;
          }
        }
      }
#pragma unroll
      for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
        for (std::uint32_t token = 0; token < Batch; ++token) {
          sums[r][token] += scale[r] * block_dots[r][token];
        }
      }
    }
    SyncKQuantTile();
  }

#pragma unroll
  for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::uint32_t token = 0; token < Batch; ++token) {
      sums[r][token] = ReduceKQuantWave<16>(sums[r][token]);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
      const std::uint32_t lr = local_row_base + r;
      if (lr >= out_stride) {
        continue;
      }
#pragma unroll
      for (std::uint32_t token = 0; token < Batch; ++token) {
        out_ptr[(token * out_stride) + lr] = sums[r][token];
      }
    }
  }
}

void LaunchFusedSSMInputProjections(
    const void* qkv_w, core::GgmlType qkv_type, const void* gate_w,
    core::GgmlType gate_type, const void* alpha_w, core::GgmlType alpha_type,
    const void* beta_w, core::GgmlType beta_type, const float* x,
    float* qkv_out, float* gate_out, float* alpha_out, float* beta_out,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream) {
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  if (alpha_type == core::GgmlType::kF32 && beta_type == core::GgmlType::kF32 &&
      (qkv_size == 0 || qkv_type == core::GgmlType::kQ8_0) &&
      (inner_size == 0 || gate_type == core::GgmlType::kQ8_0) &&
      (hidden_size % kQ8_0BlockSize == 0) &&
      ((qkv_size + inner_size) % 4 == 0) &&
      (hidden_size / 4) <= std::numeric_limits<std::uint32_t>::max()) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const std::size_t grid = (total_rows + kWavesPerBlock - 1U) / kWavesPerBlock;
    Wave32FusedSSMInputQ8_0F32Kernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const Q8_0Block*>(qkv_w),
            static_cast<const Q8_0Block*>(gate_w),
            static_cast<const float*>(alpha_w),
            static_cast<const float*>(beta_w), x, qkv_out, gate_out, alpha_out,
            beta_out, hidden_size, qkv_size, inner_size, time_step_rank);
    return;
  }
  if (hidden_size % 8 == 0) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const std::size_t grid = (total_rows + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedSSMInputProjectionsKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            qkv_w, qkv_type, gate_w, gate_type, alpha_w, alpha_type, beta_w,
            beta_type, x, qkv_out, gate_out, alpha_out, beta_out, hidden_size,
            qkv_size, inner_size, time_step_rank);
    return;
  }

  FusedSSMInputProjectionsKernel<<<total_rows, 128, 0, stream>>>(
      qkv_w, qkv_type, gate_w, gate_type, alpha_w, alpha_type, beta_w,
      beta_type, x, qkv_out, gate_out, alpha_out, beta_out, hidden_size,
      qkv_size, inner_size, time_step_rank);
}

bool TryLaunchBatchedQ8_0FusedSSMInput(
    const void* qkv_w, const void* gate_w, const void* alpha_w,
    const void* beta_w, const float* x, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream) {
  constexpr std::uint32_t kWaves = 16;
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kRowsPerBlock = kWaves * kRows;
  if (batch_size < 2 || batch_size > 8 || hidden_size == 0 ||
      (hidden_size % kQ8_0BlockSize) != 0 || qkv_size == 0 || inner_size == 0 ||
      time_step_rank == 0 || (qkv_size % kRowsPerBlock) != 0 ||
      (inner_size % kRowsPerBlock) != 0) {
    return false;
  }
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  const std::size_t row_bytes =
      (hidden_size / kQ8_0BlockSize) * sizeof(Q8_0Block);
  if (!FitsSmallBatch32BitIndices(batch_size, total_rows, hidden_size,
                                  row_bytes)) {
    return false;
  }
  const auto* qw = static_cast<const Q8_0Block*>(qkv_w);
  const auto* gw = static_cast<const Q8_0Block*>(gate_w);
  const auto* aw = static_cast<const float*>(alpha_w);
  const auto* bw = static_cast<const float*>(beta_w);
  const auto qs = static_cast<std::uint32_t>(qkv_size);
  const auto gs = static_cast<std::uint32_t>(inner_size);
  const auto rk = static_cast<std::uint32_t>(time_step_rank);
  const auto k = static_cast<std::uint32_t>(hidden_size);
  const unsigned int grid =
      static_cast<unsigned int>((total_rows + kRowsPerBlock - 1) / kRowsPerBlock);
  const auto launch = [&]<std::size_t Batch, bool XorStage>() {
    SmallBatchQ8_0FusedSSMInputKernel<kWaves, Batch, kRows, XorStage>
        <<<grid, kWaves * 32, 0, stream>>>(qw, gw, aw, bw, x, qkv_out, gate_out,
                                           alpha_out, beta_out, qs, gs, rk, k);
  };
  switch (batch_size) {
    case 2:
      launch.template operator()<2, false>();
      return true;
    case 3:
      launch.template operator()<3, false>();
      return true;
    case 4:
      launch.template operator()<4, false>();
      return true;
    case 5:
      launch.template operator()<5, true>();
      return true;
    case 6:
      launch.template operator()<6, true>();
      return true;
    case 7:
      launch.template operator()<7, true>();
      return true;
    default:
      launch.template operator()<8, true>();
      return true;
  }
}

bool TryLaunchBatchedF32SsmControls(const void* alpha_w, const void* beta_w,
                                    const float* x, float* alpha_out,
                                    float* beta_out, std::size_t batch_size,
                                    std::size_t time_step_rank,
                                    std::size_t hidden_size,
                                    hipStream_t stream) {
  if (batch_size < 2 || batch_size > 8 || time_step_rank == 0 ||
      hidden_size == 0 || (hidden_size % 4) != 0) {
    return false;
  }
  if (!FitsSmallBatch32BitIndices(batch_size, 2 * time_step_rank, hidden_size,
                                  hidden_size * sizeof(float))) {
    return false;
  }
  constexpr std::uint32_t kWaves = 4;
  const unsigned int grid = static_cast<unsigned int>(
      ((2 * time_step_rank) + kWaves - 1) / kWaves);
  const auto* aw = static_cast<const float*>(alpha_w);
  const auto* bw = static_cast<const float*>(beta_w);
  const auto rk = static_cast<std::uint32_t>(time_step_rank);
  const auto k = static_cast<std::uint32_t>(hidden_size);
  const auto launch = [&]<std::size_t Batch>() {
    BatchedF32SsmControlsKernel<kWaves, Batch><<<grid, kWaves * 32, 0, stream>>>(
        aw, bw, x, alpha_out, beta_out, rk, k);
  };
  switch (batch_size) {
    case 2:
      launch.template operator()<2>();
      return true;
    case 3:
      launch.template operator()<3>();
      return true;
    case 4:
      launch.template operator()<4>();
      return true;
    case 5:
      launch.template operator()<5>();
      return true;
    case 6:
      launch.template operator()<6>();
      return true;
    case 7:
      launch.template operator()<7>();
      return true;
    default:
      launch.template operator()<8>();
      return true;
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
