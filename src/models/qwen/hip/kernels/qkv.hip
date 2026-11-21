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
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQKVProjectionsKernel_1Row(
        const void* __restrict__ q_w, core::GgmlType q_type,
        const void* __restrict__ k_w, core::GgmlType k_type,
        const void* __restrict__ v_w, core::GgmlType v_type,
        const float* __restrict__ x, float* __restrict__ q_out,
        float* __restrict__ k_out, float* __restrict__ v_out, std::size_t q_dim,
        std::size_t kv_dim, std::size_t hidden_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    weight_type = q_type;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    weight_type = k_type;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    weight_type = v_type;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
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

__global__ void FusedQKVProjectionsKernel(
    const void* __restrict__ q_w, core::GgmlType q_type,
    const void* __restrict__ k_w, core::GgmlType k_type,
    const void* __restrict__ v_w, core::GgmlType v_type,
    const float* __restrict__ x, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out, std::size_t q_dim,
    std::size_t kv_dim, std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    weight_type = q_type;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    weight_type = k_type;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    weight_type = v_type;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
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

// Specialized all-Q8_0 single-token fused QKV projection: drops the generic
// QuantWarpBlockDot switch while preserving exact Q8_0 block-dot and butterfly
// reduction order.
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQKVQ8_0Kernel_1Row(
        const Q8_0Block* __restrict__ q_w, const Q8_0Block* __restrict__ k_w,
        const Q8_0Block* __restrict__ v_w, const float* __restrict__ x,
        float* __restrict__ q_out, float* __restrict__ k_out,
        float* __restrict__ v_out, std::size_t q_dim, std::size_t kv_dim,
        std::size_t hidden_size) {
  const std::uint32_t lane_id = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::size_t m =
      (static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp_id;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows) {
    return;
  }

  const Q8_0Block* A = nullptr;
  float* out_target = nullptr;
  std::size_t row_idx = 0;
  if (m < q_dim) {
    A = q_w;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
  }

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

// Exact fused Q8_0 Q/K/V small-batch projection sharing LDS activation staging
// across all three matrices in a single launch.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool XorStage>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0FusedQKVKernel(
        const Q8_0Block* __restrict__ q_w, const Q8_0Block* __restrict__ k_w,
        const Q8_0Block* __restrict__ v_w, const float* __restrict__ x,
        float* __restrict__ q_out, float* __restrict__ k_out,
        float* __restrict__ v_out, std::uint32_t q_dim, std::uint32_t kv_dim,
        std::uint32_t k) {
  constexpr std::uint32_t kBlocksPerTile = 32;
  constexpr std::uint32_t kVectorsPerBlock = kQ8_0BlockSize / 4;
  constexpr std::uint32_t kStride = kQ8_0BlockSize + (XorStage ? 0 : 4);
  constexpr std::uint32_t kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t total_m = q_dim + (2u * kv_dim);
  const std::uint32_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::uint32_t num_blocks = k / kQ8_0BlockSize;
  float sums[RowsPerWave][Batch] = {};

  const Q8_0Block* row_base_ptr[RowsPerWave];
  float* out_ptr[RowsPerWave];
  std::uint32_t out_stride[RowsPerWave];
  std::uint32_t local_row[RowsPerWave];
  bool row_valid[RowsPerWave];
#pragma unroll
  for (std::uint32_t r = 0; r < RowsPerWave; ++r) {
    const std::uint32_t row = row_base + r;
    row_valid[r] = row < total_m;
    const std::uint32_t safe_row = row < total_m ? row : total_m - 1u;
    if (safe_row < q_dim) {
      row_base_ptr[r] = q_w + (safe_row * num_blocks);
      out_ptr[r] = q_out;
      out_stride[r] = q_dim;
      local_row[r] = safe_row;
    } else if (safe_row < q_dim + kv_dim) {
      const std::uint32_t lr = safe_row - q_dim;
      row_base_ptr[r] = k_w + (lr * num_blocks);
      out_ptr[r] = k_out;
      out_stride[r] = kv_dim;
      local_row[r] = lr;
    } else {
      const std::uint32_t lr = safe_row - q_dim - kv_dim;
      row_base_ptr[r] = v_w + (lr * num_blocks);
      out_ptr[r] = v_out;
      out_stride[r] = kv_dim;
      local_row[r] = lr;
    }
  }

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
        rows[r] = row_base_ptr[r] + block;
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
      if (!row_valid[r]) {
        continue;
      }
#pragma unroll
      for (std::uint32_t token = 0; token < Batch; ++token) {
        out_ptr[r][(token * out_stride[r]) + local_row[r]] = sums[r][token];
      }
    }
  }
}

void LaunchFusedQKVProjections(const void* q_w, core::GgmlType q_type,
                               const void* k_w, core::GgmlType k_type,
                               const void* v_w, core::GgmlType v_type,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream) {
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (q_type == core::GgmlType::kQ8_0 && k_type == core::GgmlType::kQ8_0 &&
      v_type == core::GgmlType::kQ8_0 &&
      (hidden_size % kQ8_0BlockSize == 0) &&
      (hidden_size / kQ8_0BlockSize) <=
          std::numeric_limits<std::uint32_t>::max()) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const std::size_t grid = (total_rows + kWavesPerBlock - 1U) / kWavesPerBlock;
    Wave32FusedQKVQ8_0Kernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const Q8_0Block*>(q_w),
            static_cast<const Q8_0Block*>(k_w),
            static_cast<const Q8_0Block*>(v_w), x, q_out, k_out, v_out, q_dim,
            kv_dim, hidden_size);
    return;
  }
  if (hidden_size % 8 == 0) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const std::size_t grid = (total_rows + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedQKVProjectionsKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(q_w, q_type, k_w, k_type, v_w,
                                             v_type, x, q_out, k_out, v_out,
                                             q_dim, kv_dim, hidden_size);
    return;
  }

  FusedQKVProjectionsKernel<<<total_rows, 128, 0, stream>>>(
      q_w, q_type, k_w, k_type, v_w, v_type, x, q_out, k_out, v_out, q_dim,
      kv_dim, hidden_size);
}

bool TryLaunchBatchedQ8_0FusedQKV(const void* q_w, const void* k_w,
                                  const void* v_w, const float* x, float* q_out,
                                  float* k_out, float* v_out,
                                  std::size_t batch_size, std::size_t q_dim,
                                  std::size_t kv_dim, std::size_t hidden_size,
                                  hipStream_t stream) {
  if (batch_size < 2 || batch_size > 8 || q_dim == 0 || kv_dim == 0 ||
      hidden_size == 0 || (hidden_size % kQ8_0BlockSize) != 0) {
    return false;
  }
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  const std::size_t row_bytes =
      (hidden_size / kQ8_0BlockSize) * sizeof(Q8_0Block);
  if (!FitsSmallBatch32BitIndices(batch_size, total_rows, hidden_size,
                                  row_bytes)) {
    return false;
  }
  const auto* qw = static_cast<const Q8_0Block*>(q_w);
  const auto* kw = static_cast<const Q8_0Block*>(k_w);
  const auto* vw = static_cast<const Q8_0Block*>(v_w);
  const auto qd = static_cast<std::uint32_t>(q_dim);
  const auto kvd = static_cast<std::uint32_t>(kv_dim);
  const auto k = static_cast<std::uint32_t>(hidden_size);
  const auto launch = [&]<std::uint32_t Waves, std::size_t Batch,
                          bool XorStage>() {
    constexpr std::size_t kRows = 2;
    constexpr std::size_t kRowsPerBlock = Waves * kRows;
    SmallBatchQ8_0FusedQKVKernel<Waves, Batch, kRows, XorStage>
        <<<static_cast<unsigned int>((total_rows + kRowsPerBlock - 1) /
                                     kRowsPerBlock),
           Waves * 32, 0, stream>>>(qw, kw, vw, x, q_out, k_out, v_out, qd, kvd,
                                    k);
  };
  switch (batch_size) {
    case 2:
      launch.template operator()<8, 2, true>();
      return true;
    case 3:
      launch.template operator()<8, 3, true>();
      return true;
    case 4:
      launch.template operator()<8, 4, false>();
      return true;
    case 5:
      launch.template operator()<16, 5, true>();
      return true;
    case 6:
      launch.template operator()<16, 6, true>();
      return true;
    case 7:
      launch.template operator()<16, 7, true>();
      return true;
    default:
      launch.template operator()<16, 8, true>();
      return true;
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
