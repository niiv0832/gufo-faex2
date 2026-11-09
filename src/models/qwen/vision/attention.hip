// Vision-tower attention (Qwen3.8-27B, Qwen3.6-35B-A3B, Flash-Next towers).
#include "src/models/qwen/vision/attention.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace gufo::models::qwen::vision {
namespace {
using Bf16 = hip_bfloat16;
constexpr unsigned kHidden = kVisionHidden;
constexpr unsigned kHeads = kVisionHeads;
constexpr unsigned kHeadDim = kVisionHeadDim;
constexpr unsigned kThreads = 256;

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
void CheckBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("vision attention hipBLAS failure: " +
                             std::to_string(status));
}
__device__ __forceinline__ float Round(float x) {
  return static_cast<float>(Bf16(x));
}

struct Scratch {
  void* data{nullptr};
  explicit Scratch(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Scratch() {
    if (data != nullptr)
      (void)hipFree(data);
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  template<class T>
  T* As() const {
    return static_cast<T*>(data);
  }
};

// Keep bounded rows in registers through normalization. Larger rows retain
// the streaming path; caching them would spill and reread private memory.
template<unsigned kValues>
__global__ void Softmax(const Bf16* scores, Bf16* probabilities,
                        unsigned length) {
  const unsigned row = blockIdx.x;
  __shared__ float reduce[kThreads];
  float values[kValues != 0 ? kValues : 1];
  float maximum = -INFINITY;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      values[j] =
          i < length
              ? Round(static_cast<float>(scores[row * length + i]) * 0.1178511301977579F)
              : -INFINITY;
      maximum = fmaxf(maximum, values[j]);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      maximum = fmaxf(maximum, Round(Round(scores[row * length + i]) *
                                     0.1178511301977579F));
    }
  }
  reduce[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] =
          fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float sum = 0;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      if (threadIdx.x + j * kThreads < length) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
        // Limit overlapping exponential temporaries without changing their
        // order. Wider shapes otherwise spill registers on gfx1151.
        if constexpr (kValues > 16)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads)
      sum += expf(Round(static_cast<float>(scores[row * length + i]) * 0.1178511301977579F) -
                  maximum);
  }
  reduce[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0F / reduce[0];
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      if (i < length)
        probabilities[row * length + i] = Bf16(values[j] * inverse);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      probabilities[row * length + i] = Bf16(
          expf(Round(static_cast<float>(scores[row * length + i]) * 0.1178511301977579F) -
               maximum) *
          inverse);
    }
  }
}

__global__ void AttentionRows(const float* input, Bf16* output, unsigned begin,
                              unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  output[(begin + token) * kHidden + head * kHeadDim + d] =
      Bf16(input[(head * rows + token) * kHeadDim + d]);
}

void TiledAttention(hipblasHandle_t blas, const Bf16* q, const Bf16* k,
                    const Bf16* v, Bf16* output, unsigned count,
                    hipStream_t stream) {
  // Smaller attention tiles reduce scratch traffic for the 4096-patch
  // shape while preserving its BLAS reductions. Ragged tails require
  // their existing GEMM shapes to preserve rounding.
  const unsigned tile = std::min(count, count == 4096 ? 64U : 128U);
  // Scores leave the BLAS reduction already rounded to BF16, as the softmax
  // consumed them; this halves the dominant scratch traffic.
  Scratch scores(std::size_t{kHeads} * tile * count * sizeof(Bf16));
  Scratch probabilities(std::size_t{kHeads} * tile * count * sizeof(Bf16));
  Scratch output_tile(std::size_t{tile} * kHidden * sizeof(float));
  CheckBlas(hipblasSetStream(blas, stream));
  for (unsigned begin = 0; begin < count; begin += tile) {
    const unsigned rows = std::min(tile, count - begin);
    constexpr float zero = 0, one = 1;
    CheckBlas(hipblasGemmStridedBatchedEx(
        blas, HIPBLAS_OP_T, HIPBLAS_OP_N, count, rows, kHeadDim, &one, k,
        HIP_R_16BF, kHeadDim, count * kHeadDim, q + begin * kHeadDim,
        HIP_R_16BF, kHeadDim, count * kHeadDim, &zero, scores.data, HIP_R_16BF,
        count, rows * count, kHeads, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    if (count <= 1024)
      Softmax<4><<<kHeads * rows, kThreads, 0, stream>>>(
          scores.As<Bf16>(), probabilities.As<Bf16>(), count);
    else if (count <= 2048)
      Softmax<8><<<kHeads * rows, kThreads, 0, stream>>>(
          scores.As<Bf16>(), probabilities.As<Bf16>(), count);
    else if (count <= 4096)
      Softmax<16><<<kHeads * rows, kThreads, 0, stream>>>(
          scores.As<Bf16>(), probabilities.As<Bf16>(), count);
    else if (count <= 8192)
      Softmax<32><<<kHeads * rows, kThreads, 0, stream>>>(
          scores.As<Bf16>(), probabilities.As<Bf16>(), count);
    else
      Softmax<0><<<kHeads * rows, kThreads, 0, stream>>>(
          scores.As<Bf16>(), probabilities.As<Bf16>(), count);
    CheckBlas(hipblasGemmStridedBatchedEx(
        blas, HIPBLAS_OP_N, HIPBLAS_OP_N, kHeadDim, rows, count, &one, v,
        HIP_R_16BF, kHeadDim, count * kHeadDim, probabilities.data, HIP_R_16BF,
        count, rows * count, &zero, output_tile.data, HIP_R_32F, kHeadDim,
        rows * kHeadDim, kHeads, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    AttentionRows<<<(rows * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                    stream>>>(output_tile.As<float>(), output, begin, rows);
  }
}

// Fused attention (flash-style, online softmax) on 16x16x16 BF16 WMMA tiles.
// Each wave owns 16 query rows; kWaves waves share K/V blocks of kKeys keys
// staged in LDS (head dimension 72 zero-padded to 80). Scores and the output
// accumulator stay in registers: on RDNA3 wave32 an f32 16x16 accumulator
// holds element x[i] of lane l at row 2i + l/16, column l%16, so per-row
// reductions are shuffles within each 16-lane half. Only the probabilities
// pass through LDS, to become the A operand of the value product. Scores keep
// the encoder's rounding (BF16 dot, BF16 scaled score); probabilities enter
// the value product as BF16 before normalization.
constexpr unsigned kWaves = 8;
constexpr unsigned kWave = 32;
constexpr unsigned kRows = 16 * kWaves;
constexpr unsigned kKeys = 64;
constexpr unsigned kPad = 80;
constexpr unsigned kChunks = kPad / 16;        // 5 head-dimension chunks
constexpr unsigned kKeyTiles = kKeys / 16;     // 4 key tiles per block
constexpr unsigned kRowWords = kHeadDim * 2 / 16;  // 9 x 16 B per row
constexpr unsigned kPadWords = kPad * 2 / 16;      // 10 x 16 B per LDS row
constexpr float kScale = 0.1178511301977579F;

__device__ __forceinline__ float HalfMax(float x) {
  for (unsigned m = 1; m < 16; m <<= 1)
    x = fmaxf(x, __shfl_xor(x, m));
  return x;
}
__device__ __forceinline__ float HalfSum(float x) {
  for (unsigned m = 1; m < 16; m <<= 1)
    x += __shfl_xor(x, m);
  return x;
}
__device__ __forceinline__ void WaveSync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Rows of `count` x 72 BF16 into LDS rows of 80 (zero tail), 16 B at a time.
__device__ __forceinline__ void Stage(uint4 (*tile)[kPadWords],
                                      const Bf16* source, unsigned first,
                                      unsigned rows, unsigned count) {
  const auto* words = reinterpret_cast<const uint4*>(source);
  for (unsigned i = threadIdx.x; i < rows * kPadWords; i += blockDim.x) {
    const unsigned r = i / kPadWords, w = i % kPadWords, row = first + r;
    tile[r][w] = row < count && w < kRowWords
                     ? words[std::size_t{row} * kRowWords + w]
                     : make_uint4(0, 0, 0, 0);
  }
}

__global__ __launch_bounds__(kWaves * kWave) void FusedAttentionKernel(
    const Bf16* q, const Bf16* k, const Bf16* v, Bf16* output,
    unsigned count) {
  using namespace rocwmma;
  using Element = rocwmma::bfloat16_t;
  using Acc = fragment<accumulator, 16, 16, 16, float>;
  __shared__ uint4 tile_k[kKeys][kPadWords];
  __shared__ uint4 tile_v[kKeys][kPadWords];
  __shared__ Element tile_p[kWaves][16][kKeys];

  const unsigned head = blockIdx.y;
  const unsigned row0 = blockIdx.x * kRows;
  const unsigned wave = threadIdx.x / kWave;
  const unsigned lane = threadIdx.x % kWave;
  const unsigned half = lane / 16, column = lane % 16;
  const std::size_t plane = std::size_t{head} * count * kHeadDim;

  // Stage this block's queries through the K buffer (kRows >= kKeys rows).
  static_assert(kRows == 2 * kKeys);
  fragment<matrix_a, 16, 16, 16, Element, row_major> query[kChunks];
  for (unsigned part = 0; part < 2; ++part) {
    Stage(tile_k, q + plane, row0 + part * kKeys, kKeys, count);
    __syncthreads();
    if (wave / (kWaves / 2) == part) {
      const auto* base = reinterpret_cast<const Element*>(
          tile_k[(wave % (kWaves / 2)) * 16]);
      for (unsigned c = 0; c < kChunks; ++c)
        load_matrix_sync(query[c], base + c * 16, kPad);
    }
    __syncthreads();
  }

  Acc out[kChunks];
  for (auto& o : out)
    fill_fragment(o, 0.0F);
  float row_max[8], row_sum[8];
  for (unsigned i = 0; i < 8; ++i) {
    row_max[i] = -INFINITY;
    row_sum[i] = 0.0F;
  }

  for (unsigned key0 = 0; key0 < count; key0 += kKeys) {
    Stage(tile_k, k + plane, key0, kKeys, count);
    Stage(tile_v, v + plane, key0, kKeys, count);
    __syncthreads();
    const auto* keys_base = reinterpret_cast<const Element*>(tile_k[0]);
    const auto* values_base = reinterpret_cast<const Element*>(tile_v[0]);

    Acc scores[kKeyTiles];
    for (unsigned t = 0; t < kKeyTiles; ++t) {
      fill_fragment(scores[t], 0.0F);
      for (unsigned c = 0; c < kChunks; ++c) {
        fragment<matrix_b, 16, 16, 16, Element, col_major> keys;
        load_matrix_sync(keys, keys_base + (t * 16) * kPad + c * 16, kPad);
        mma_sync(scores[t], query[c], keys, scores[t]);
      }
    }
    // Online softmax per row (2i + half), entirely in registers.
    for (unsigned i = 0; i < 8; ++i) {
      float tile_max = -INFINITY;
      for (unsigned t = 0; t < kKeyTiles; ++t) {
        const bool valid = key0 + t * 16 + column < count;
        const float s =
            valid ? Round(Round(scores[t].x[i]) * kScale) : -INFINITY;
        scores[t].x[i] = s;
        tile_max = fmaxf(tile_max, s);
      }
      tile_max = HalfMax(tile_max);
      const float maximum = fmaxf(row_max[i], tile_max);
      const float alpha = expf(row_max[i] - maximum);  // -inf -> 0
      row_max[i] = maximum;
      float sum = 0.0F;
      for (unsigned t = 0; t < kKeyTiles; ++t) {
        const float s = scores[t].x[i];
        const float p = s == -INFINITY ? 0.0F : expf(s - maximum);
        sum += p;
        tile_p[wave][2 * i + half][t * 16 + column] = static_cast<Element>(p);
      }
      row_sum[i] = row_sum[i] * alpha + HalfSum(sum);
      for (unsigned c = 0; c < kChunks; ++c)
        out[c].x[i] *= alpha;
    }
    WaveSync();
    for (unsigned t = 0; t < kKeyTiles; ++t) {
      fragment<matrix_a, 16, 16, 16, Element, row_major> probs;
      load_matrix_sync(probs, &tile_p[wave][0][t * 16], kKeys);
      for (unsigned c = 0; c < kChunks; ++c) {
        fragment<matrix_b, 16, 16, 16, Element, row_major> values;
        load_matrix_sync(values, values_base + (t * 16) * kPad + c * 16, kPad);
        mma_sync(out[c], probs, values, out[c]);
      }
    }
    __syncthreads();  // K/V and P are rewritten by the next block
  }

  for (unsigned c = 0; c < kChunks; ++c) {
    const unsigned d = c * 16 + column;
    if (d >= kHeadDim)
      continue;
    for (unsigned i = 0; i < 8; ++i) {
      const unsigned row = row0 + wave * 16 + 2 * i + half;
      if (row < count)
        output[std::size_t{row} * kHidden + head * kHeadDim + d] =
            Bf16(out[c].x[i] / row_sum[i]);
    }
  }
}

void FusedAttention(const Bf16* q, const Bf16* k, const Bf16* v, Bf16* output,
                    unsigned count, hipStream_t stream) {
  const dim3 grid((count + kRows - 1) / kRows, kHeads);
  FusedAttentionKernel<<<grid, kWaves * kWave, 0, stream>>>(q, k, v, output,
                                                           count);
  Check(hipGetLastError());
}

}  // namespace

AttentionKernel DefaultAttentionKernel() {
  static const AttentionKernel kind = [] {
    const char* value = std::getenv("GUFO_VISION_ATTENTION");
    return value != nullptr && std::string_view(value) == "tiled"
               ? AttentionKernel::kTiled
               : AttentionKernel::kFused;
  }();
  return kind;
}

void Attention(AttentionKernel kind, hipblasHandle_t blas, const Bf16* q,
               const Bf16* k, const Bf16* v, Bf16* output, unsigned count,
               hipStream_t stream) {
  if (count == 0)
    return;
  if (kind == AttentionKernel::kTiled)
    TiledAttention(blas, q, k, v, output, count, stream);
  else
    FusedAttention(q, k, v, output, count, stream);
}

}  // namespace gufo::models::qwen::vision
