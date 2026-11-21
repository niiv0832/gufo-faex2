#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

void TestQ8KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 512;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8KBlockTest = gufo::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  // Deterministic pseudo-random generator (same values on every run).
  std::uint32_t seed = 12345U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };

  // Synthetic Q8_K weights: qs in [-127,127], d in ~[-2,2]; x is fp32 in
  // [-1,1].
  std::vector<Q8KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = rnd_float(-2.0F, 2.0F);
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
    for (std::size_t i = 0; i < 16; ++i) {
      blk.bsums[i] = 0;
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = rnd_float(-1.0F, 1.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_A, gufo::core::GgmlType::kQ8_K, d_x, d_y, M,
                                K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(
      hipMemcpy(y_gpu.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Primary check: CPU DotProductQ8_K (exact fp dequant dot of weights against
  // x).
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = gufo::quant::DotProductQ8_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_diff = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_diff =
        abs_diff / std::max(1e-4F, std::abs(y_dequant_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_diff);
    primary_max_abs = std::max(primary_max_abs, abs_diff);
  }

  std::cout << "Q8K GEMV vs CPU oracle: max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8K GEMV mismatch against CPU oracle\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestQ8KSmallBatchFp32GEMMEquivalence() {
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kBlockSize = 256;
  constexpr std::size_t kColumns = 512;
  constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
  constexpr std::size_t kMaximumBatch = 8;

  using Q8KBlockTest = gufo::quant::block_q8_K;
  std::uint32_t seed = 98431U;
  auto random = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto random_float = [&random](float low, float high) -> float {
    const float unit = static_cast<float>(random() & 0xFFFFU) / 65535.0F;
    return low + unit * (high - low);
  };

  std::vector<Q8KBlockTest> weights(kRows * kBlocksPerRow);
  std::vector<float> inputs(kMaximumBatch * kColumns);
  for (auto& block : weights) {
    block.d = random_float(-0.25F, 0.25F);
    for (auto& value : block.qs) {
      value = static_cast<std::int8_t>(static_cast<int>(random() % 255U) - 127);
    }
    std::fill(std::begin(block.bsums), std::end(block.bsums), 0);
  }
  for (auto& input : inputs) {
    input = random_float(-1.0F, 1.0F);
  }

  void* device_weights = nullptr;
  float* device_inputs = nullptr;
  float* device_reference = nullptr;
  float* device_batched = nullptr;
  HIP_CHECK(hipMalloc(&device_weights, weights.size() * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(&device_inputs, inputs.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&device_reference, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_batched, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_weights, weights.data(),
                      weights.size() * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_inputs, inputs.data(),
                      inputs.size() * sizeof(float), hipMemcpyHostToDevice));

  for (const std::size_t batch :
       {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5},
        std::size_t{6}, std::size_t{7}, std::size_t{8}}) {
    for (std::size_t token = 0; token < batch; ++token) {
      gufo::hip::LaunchQ8KBlockGEMV(device_weights, gufo::core::GgmlType::kQ8_K,
                                    device_inputs + (token * kColumns),
                                    device_reference + (token * kRows), kRows,
                                    kColumns, nullptr);
    }
    gufo::hip::LaunchBatchedQuantGEMMFp32(
        gufo::core::GgmlType::kQ8_K, device_weights, device_inputs,
        device_batched, batch, kRows, kColumns, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> reference(batch * kRows);
    std::vector<float> batched(batch * kRows);
    HIP_CHECK(hipMemcpy(reference.data(), device_reference,
                        reference.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batched.data(), device_batched,
                        batched.size() * sizeof(float), hipMemcpyDeviceToHost));

    for (std::size_t index = 0; index < reference.size(); ++index) {
      if (reference[index] != batched[index]) {
        std::cerr << "shared Q8_K small-batch mismatch at batch " << batch
                  << " index " << index << ": expected " << reference[index]
                  << ", got " << batched[index] << "\n";
        std::abort();
      }
    }
  }

  HIP_CHECK(hipFree(device_weights));
  HIP_CHECK(hipFree(device_inputs));
  HIP_CHECK(hipFree(device_reference));
  HIP_CHECK(hipFree(device_batched));
}

// `opt-c206-q8-small-batch` selects among physical kernel widths by shape, so a
// 16x512 toy matrix does not exercise the routes a real projection reaches. The
// shapes below are the ones the Q8 shard actually dispatches: attention output
// (5120x5120), FFN gate/up (17408x5120) and the tied LM head (248320x5120),
// truncated in rows where the full tensor would not fit but keeping each
// distinct row-to-K ratio that the selector branches on.
void TestQ8_0SmallBatchFp32GEMMEquivalence(std::size_t kRows,
                                           std::size_t kColumns) {
  constexpr std::size_t kBlockSize = 32;
  const std::size_t kBlocksPerRow = kColumns / kBlockSize;
  constexpr std::size_t kMaximumBatch = 8;

  using Q8_0BlockTest = gufo::quant::block_q8_0;
  std::uint32_t seed = 31789U;
  auto random = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto random_float = [&random](float low, float high) -> float {
    const float unit = static_cast<float>(random() & 0xFFFFU) / 65535.0F;
    return low + unit * (high - low);
  };
  auto float_to_half_bits = [](float value) -> std::uint16_t {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    std::int32_t exponent =
        static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    std::uint32_t mantissa = (bits >> 13) & 0x3FFU;
    if (exponent <= 0) {
      if (exponent < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mantissa |= 0x400U;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
      return static_cast<std::uint16_t>(sign | (mantissa >> shift));
    }
    if (exponent >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00U |
                                        (mantissa != 0 ? 0x200U : 0U));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) | mantissa);
  };

  std::vector<Q8_0BlockTest> weights(kRows * kBlocksPerRow);
  std::vector<float> inputs(kMaximumBatch * kColumns);
  for (auto& block : weights) {
    block.d = float_to_half_bits(random_float(-0.25F, 0.25F));
    for (auto& value : block.qs) {
      value = static_cast<std::int8_t>(static_cast<int>(random() % 255U) - 127);
    }
  }
  for (auto& input : inputs) {
    input = random_float(-1.0F, 1.0F);
  }

  void* device_weights = nullptr;
  float* device_inputs = nullptr;
  float* device_reference = nullptr;
  float* device_batched = nullptr;
  HIP_CHECK(hipMalloc(&device_weights, weights.size() * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(&device_inputs, inputs.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&device_reference, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_batched, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_weights, weights.data(),
                      weights.size() * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_inputs, inputs.data(),
                      inputs.size() * sizeof(float), hipMemcpyHostToDevice));

  // Cover every public small-batch width, including a one-row tail and >8
  // multi-pass verification widths.
  constexpr std::size_t kMaxTestBatch = 16;
  std::vector<float> extra_inputs(kMaxTestBatch * kColumns);
  for (auto& input : extra_inputs) {
    input = random_float(-1.0F, 1.0F);
  }
  HIP_CHECK(hipFree(device_inputs));
  HIP_CHECK(hipFree(device_reference));
  HIP_CHECK(hipFree(device_batched));
  HIP_CHECK(hipMalloc(&device_inputs, extra_inputs.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&device_reference, kMaxTestBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_batched, kMaxTestBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_inputs, extra_inputs.data(),
                      extra_inputs.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  for (const std::size_t batch :
       {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
        std::size_t{5}, std::size_t{6}, std::size_t{7}, std::size_t{8},
        std::size_t{9}, std::size_t{12}, std::size_t{16}}) {
    for (std::size_t token = 0; token < batch; ++token) {
      gufo::hip::LaunchQ8KBlockGEMV(device_weights, gufo::core::GgmlType::kQ8_0,
                                    device_inputs + (token * kColumns),
                                    device_reference + (token * kRows), kRows,
                                    kColumns, nullptr);
    }
    gufo::hip::LaunchBatchedQuantGEMMFp32(
        gufo::core::GgmlType::kQ8_0, device_weights, device_inputs,
        device_batched, batch, kRows, kColumns, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> reference(batch * kRows);
    std::vector<float> batched(batch * kRows);
    HIP_CHECK(hipMemcpy(reference.data(), device_reference,
                        reference.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batched.data(), device_batched,
                        batched.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t index = 0; index < reference.size(); ++index) {
      if (std::memcmp(&reference[index], &batched[index], sizeof(float)) != 0) {
        std::cerr << "exact Q8_0 small-batch mismatch at " << kRows << "x"
                  << kColumns << " batch " << batch << " index " << index
                  << ": expected " << reference[index] << ", got "
                  << batched[index] << "\n";
        std::abort();
      }
    }
  }

  HIP_CHECK(hipFree(device_weights));
  HIP_CHECK(hipFree(device_inputs));
  HIP_CHECK(hipFree(device_reference));
  HIP_CHECK(hipFree(device_batched));
}

void TestQ8_0BlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 32;
  constexpr std::size_t K = 512;  // K % 32 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8_0BlockTest = gufo::quant::block_q8_0;
  static_assert(sizeof(Q8_0BlockTest) == 34, "Q8_0 block must be 34 bytes");

  // Deterministic pseudo-random generator (same values on every run).
  std::uint32_t seed = 12345U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };
  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
  };

  // Synthetic Q8_0 weights: qs in [-127,127], d (as fp16) in ~[-2,2]; x is fp32
  // in [-1,1].
  std::vector<Q8_0BlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = float_to_half_bits(rnd_float(-2.0F, 2.0F));
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = rnd_float(-1.0F, 1.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_A, gufo::core::GgmlType::kQ8_0, d_x, d_y, M,
                                K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(
      hipMemcpy(y_gpu.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8_0 GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Primary check: CPU DotProductQ8_0 (exact fp dequant dot of weights against
  // x).
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = gufo::quant::DotProductQ8_0(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_diff = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_diff =
        abs_diff / std::max(1e-4F, std::abs(y_dequant_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_diff);
    primary_max_abs = std::max(primary_max_abs, abs_diff);
  }

  std::cout << "Q8_0 GEMV vs CPU oracle: max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8_0 GEMV mismatch against CPU oracle\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestQ8_0FusedAndExtremeBitwiseParity() {
  using Q8_0BlockTest = gufo::quant::block_q8_0;
  std::uint32_t seed = 424242U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };
  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10)
        return static_cast<std::uint16_t>(sign);
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
  };
  auto fill_q8 = [&](std::vector<Q8_0BlockTest>& blocks) {
    for (std::size_t idx = 0; idx < blocks.size(); ++idx) {
      auto& b = blocks[idx];
      if (idx % 37 == 0) {
        // Explicit -0.0f16 scale to verify signed-zero parity on K=512 L16
        // path.
        b.d = 0x8000U;
      } else if (idx % 41 == 0) {
        // Subnormal scale.
        b.d = 0x0001U;
      } else {
        b.d = float_to_half_bits(rnd_float(-0.25F, 0.25F));
      }
      for (int i = 0; i < 32; ++i) {
        b.qs[i] =
            static_cast<std::int8_t>(static_cast<int>(rnd() % 256U) - 128);
      }
    }
  };
  auto assert_exact = [](const char* label, const std::vector<float>& ref,
                         const std::vector<float>& got) {
    if (ref.size() != got.size() ||
        std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
      for (std::size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
          std::uint32_t rb = 0, gb = 0;
          std::memcpy(&rb, &ref[i], 4);
          std::memcpy(&gb, &got[i], 4);
          std::cerr << label << " bitwise mismatch at " << i
                    << ": ref=" << ref[i] << " (0x" << std::hex << rb
                    << ") got=" << got[i] << " (0x" << gb << ")" << std::dec
                    << "\n";
          std::abort();
        }
      }
    }
  };

  // 1) Extreme values & signed-zero parity on K=512 (L16 vs 32-lane
  // single/batched).
  {
    constexpr std::size_t kM = 37;
    constexpr std::size_t kK = 512;
    constexpr std::size_t kMaxB = 8;
    std::vector<Q8_0BlockTest> h_w(kM * (kK / 32));
    fill_q8(h_w);
    std::vector<float> h_x(kMaxB * kK);
    for (std::size_t i = 0; i < h_x.size(); ++i) {
      h_x[i] =
          (i % 19 == 0) ? -0.0F : (i % 23 == 0 ? 0.0F : rnd_float(-1.0F, 1.0F));
    }
    void* d_w = nullptr;
    float *d_x = nullptr, *d_ref = nullptr, *d_got = nullptr;
    HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ref, kMaxB * kM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_got, kMaxB * kM * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    for (std::size_t b = 2; b <= kMaxB; ++b) {
      for (std::size_t t = 0; t < b; ++t) {
        gufo::hip::LaunchQ8KBlockGEMV(d_w, gufo::core::GgmlType::kQ8_0,
                                      d_x + (t * kK), d_ref + (t * kM), kM, kK,
                                      nullptr);
      }
      gufo::hip::LaunchBatchedQuantGEMMFp32(gufo::core::GgmlType::kQ8_0, d_w,
                                            d_x, d_got, b, kM, kK, nullptr);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> ref(b * kM), got(b * kM);
      HIP_CHECK(
          hipMemcpy(ref.data(), d_ref, ref.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(got.data(), d_got, got.size() * 4, hipMemcpyDeviceToHost));
      assert_exact("Q8_0 K=512 extreme L16", ref, got);
    }
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_ref));
    HIP_CHECK(hipFree(d_got));
  }

  // 2) Fused single-token and batched Q8_0 SwiGLU (even 512 and odd 513).
  for (const std::size_t kM : {std::size_t{512}, std::size_t{513}}) {
    constexpr std::size_t kK = 2048;
    constexpr std::size_t kMaxB = 8;
    std::vector<Q8_0BlockTest> h_gw(kM * (kK / 32)), h_uw(kM * (kK / 32));
    fill_q8(h_gw);
    fill_q8(h_uw);
    std::vector<float> h_x(kMaxB * kK);
    for (auto& v : h_x)
      v = rnd_float(-1.0F, 1.0F);
    void *d_gw = nullptr, *d_uw = nullptr;
    float *d_x = nullptr, *d_g = nullptr, *d_u = nullptr, *d_ref = nullptr,
          *d_got = nullptr;
    HIP_CHECK(hipMalloc(&d_gw, h_gw.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_uw, h_uw.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_g, kM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_u, kM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ref, kMaxB * kM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_got, kMaxB * kM * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_gw, h_gw.data(), h_gw.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_uw, h_uw.data(), h_uw.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                        hipMemcpyHostToDevice));

    // Single-token LaunchFusedSwiGLUGEMV vs separate GEMV + SwiGLU activation.
    gufo::hip::LaunchQ8KBlockGEMV(d_gw, gufo::core::GgmlType::kQ8_0, d_x, d_g,
                                  kM, kK, nullptr);
    gufo::hip::LaunchQ8KBlockGEMV(d_uw, gufo::core::GgmlType::kQ8_0, d_x, d_u,
                                  kM, kK, nullptr);
    gufo::hip::LaunchBatchedSwiGLUActivation(d_g, d_u, d_ref, nullptr, kM,
                                             nullptr);
    gufo::hip::LaunchFusedSwiGLUGEMV(d_gw, gufo::core::GgmlType::kQ8_0, d_uw,
                                     gufo::core::GgmlType::kQ8_0, d_x, d_got,
                                     kM, kK, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> ref1(kM), got1(kM);
    HIP_CHECK(hipMemcpy(ref1.data(), d_ref, kM * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got1.data(), d_got, kM * 4, hipMemcpyDeviceToHost));
    assert_exact("Q8_0 single FusedSwiGLU", ref1, got1);

    for (std::size_t b = 2; b <= kMaxB; ++b) {
      for (std::size_t t = 0; t < b; ++t) {
        gufo::hip::LaunchFusedSwiGLUGEMV(d_gw, gufo::core::GgmlType::kQ8_0,
                                         d_uw, gufo::core::GgmlType::kQ8_0,
                                         d_x + (t * kK), d_ref + (t * kM), kM,
                                         kK, nullptr);
      }
      if (!gufo::hip::TryLaunchBatchedQ8_0FusedSwiGLU(d_gw, d_uw, d_x, d_got, b,
                                                      kM, kK, nullptr)) {
        std::cerr << "TryLaunchBatchedQ8_0FusedSwiGLU returned false\n";
        std::abort();
      }
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> ref(b * kM), got(b * kM);
      HIP_CHECK(
          hipMemcpy(ref.data(), d_ref, ref.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(got.data(), d_got, got.size() * 4, hipMemcpyDeviceToHost));
      assert_exact("Q8_0 batched FusedSwiGLU", ref, got);
    }
    HIP_CHECK(hipFree(d_gw));
    HIP_CHECK(hipFree(d_uw));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_g));
    HIP_CHECK(hipFree(d_u));
    HIP_CHECK(hipFree(d_ref));
    HIP_CHECK(hipFree(d_got));
  }

  // 3) Fused Q8_0 QKV (production shape + odd dims).
  for (auto [q_dim, kv_dim, kK] :
       std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>{
           {8192, 512, 2048}, {67, 19, 512}}) {
    constexpr std::size_t kMaxB = 8;
    std::vector<Q8_0BlockTest> h_q(q_dim * (kK / 32)), h_k(kv_dim * (kK / 32)),
        h_v(kv_dim * (kK / 32));
    fill_q8(h_q);
    fill_q8(h_k);
    fill_q8(h_v);
    std::vector<float> h_x(kMaxB * kK);
    for (auto& v : h_x)
      v = rnd_float(-1.0F, 1.0F);
    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_x = nullptr, *d_rq = nullptr, *d_rk = nullptr, *d_rv = nullptr,
          *d_gq = nullptr, *d_gk = nullptr, *d_gv = nullptr;
    HIP_CHECK(hipMalloc(&d_q, h_q.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_k, h_k.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_v, h_v.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rq, kMaxB * q_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rk, kMaxB * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rv, kMaxB * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gq, kMaxB * q_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gk, kMaxB * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gv, kMaxB * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, h_v.data(), h_v.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    for (std::size_t b = 2; b <= kMaxB; ++b) {
      for (std::size_t t = 0; t < b; ++t) {
        gufo::hip::LaunchFusedQKVProjections(
            d_q, gufo::core::GgmlType::kQ8_0, d_k, gufo::core::GgmlType::kQ8_0,
            d_v, gufo::core::GgmlType::kQ8_0, d_x + (t * kK),
            d_rq + (t * q_dim), d_rk + (t * kv_dim), d_rv + (t * kv_dim), q_dim,
            kv_dim, kK, nullptr);
      }
      if (!gufo::hip::TryLaunchBatchedQ8_0FusedQKV(d_q, d_k, d_v, d_x, d_gq,
                                                   d_gk, d_gv, b, q_dim, kv_dim,
                                                   kK, nullptr)) {
        std::cerr << "TryLaunchBatchedQ8_0FusedQKV returned false\n";
        std::abort();
      }
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> rq(b * q_dim), gq(b * q_dim), rk(b * kv_dim),
          gk(b * kv_dim), rv(b * kv_dim), gv(b * kv_dim);
      HIP_CHECK(
          hipMemcpy(rq.data(), d_rq, rq.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gq.data(), d_gq, gq.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(rk.data(), d_rk, rk.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gk.data(), d_gk, gk.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(rv.data(), d_rv, rv.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gv.data(), d_gv, gv.size() * 4, hipMemcpyDeviceToHost));
      assert_exact("Q8_0 batched FusedQKV Q", rq, gq);
      assert_exact("Q8_0 batched FusedQKV K", rk, gk);
      assert_exact("Q8_0 batched FusedQKV V", rv, gv);
    }
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_rq));
    HIP_CHECK(hipFree(d_rk));
    HIP_CHECK(hipFree(d_rv));
    HIP_CHECK(hipFree(d_gq));
    HIP_CHECK(hipFree(d_gk));
    HIP_CHECK(hipFree(d_gv));
  }

  // 4) Fused Q8_0+F32 SSM input & standalone batched F32 controls.
  for (auto [qkv_sz, gate_sz, rank, kK] : std::vector<
           std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>{
           {8192, 4096, 32, 2048}, {64, 32, 17, 512}}) {
    constexpr std::size_t kMaxB = 8;
    std::vector<Q8_0BlockTest> h_qkv(qkv_sz * (kK / 32)),
        h_gate(gate_sz * (kK / 32));
    fill_q8(h_qkv);
    fill_q8(h_gate);
    std::vector<float> h_a(rank * kK), h_b(rank * kK), h_x(kMaxB * kK);
    for (auto& v : h_a)
      v = rnd_float(-0.5F, 0.5F);
    for (auto& v : h_b)
      v = rnd_float(-0.5F, 0.5F);
    for (auto& v : h_x)
      v = rnd_float(-1.0F, 1.0F);
    void *d_qkv = nullptr, *d_gate = nullptr;
    float *d_a = nullptr, *d_b = nullptr, *d_x = nullptr;
    float *d_rq = nullptr, *d_rg = nullptr, *d_ra = nullptr, *d_rb = nullptr;
    float *d_gq = nullptr, *d_gg = nullptr, *d_ga = nullptr, *d_gb = nullptr;
    HIP_CHECK(hipMalloc(&d_qkv, h_qkv.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_gate, h_gate.size() * sizeof(Q8_0BlockTest)));
    HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, h_b.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rq, kMaxB * qkv_sz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rg, kMaxB * gate_sz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ra, kMaxB * rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rb, kMaxB * rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gq, kMaxB * qkv_sz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gg, kMaxB * gate_sz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ga, kMaxB * rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gb, kMaxB * rank * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(),
                        h_qkv.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, h_gate.data(),
                        h_gate.size() * sizeof(Q8_0BlockTest),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    for (std::size_t b = 2; b <= kMaxB; ++b) {
      for (std::size_t t = 0; t < b; ++t) {
        gufo::hip::LaunchFusedSSMInputProjections(
            d_qkv, gufo::core::GgmlType::kQ8_0, d_gate,
            gufo::core::GgmlType::kQ8_0, d_a, gufo::core::GgmlType::kF32, d_b,
            gufo::core::GgmlType::kF32, d_x + (t * kK), d_rq + (t * qkv_sz),
            d_rg + (t * gate_sz), d_ra + (t * rank), d_rb + (t * rank), kK,
            qkv_sz, gate_sz, rank, nullptr);
      }
      if (!gufo::hip::TryLaunchBatchedQ8_0FusedSSMInput(
              d_qkv, d_gate, d_a, d_b, d_x, d_gq, d_gg, d_ga, d_gb, b, kK,
              qkv_sz, gate_sz, rank, nullptr)) {
        std::cerr << "TryLaunchBatchedQ8_0FusedSSMInput returned false\n";
        std::abort();
      }
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> rq(b * qkv_sz), gq(b * qkv_sz), rg(b * gate_sz),
          gg(b * gate_sz), ra(b * rank), ga(b * rank), rb(b * rank),
          gb(b * rank);
      HIP_CHECK(
          hipMemcpy(rq.data(), d_rq, rq.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gq.data(), d_gq, gq.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(rg.data(), d_rg, rg.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gg.data(), d_gg, gg.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(ra.data(), d_ra, ra.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(ga.data(), d_ga, ga.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(rb.data(), d_rb, rb.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gb.data(), d_gb, gb.size() * 4, hipMemcpyDeviceToHost));
      assert_exact("Q8_0 batched FusedSSM qkv", rq, gq);
      assert_exact("Q8_0 batched FusedSSM gate", rg, gg);
      assert_exact("Q8_0 batched FusedSSM alpha", ra, ga);
      assert_exact("Q8_0 batched FusedSSM beta", rb, gb);

      // Also verify standalone TryLaunchBatchedF32SsmControls.
      HIP_CHECK(hipMemset(d_ga, 0, ga.size() * 4));
      HIP_CHECK(hipMemset(d_gb, 0, gb.size() * 4));
      if (!gufo::hip::TryLaunchBatchedF32SsmControls(d_a, d_b, d_x, d_ga, d_gb,
                                                     b, rank, kK, nullptr)) {
        std::cerr << "TryLaunchBatchedF32SsmControls returned false\n";
        std::abort();
      }
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(
          hipMemcpy(ga.data(), d_ga, ga.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(gb.data(), d_gb, gb.size() * 4, hipMemcpyDeviceToHost));
      assert_exact("F32 batched SsmControls alpha", ra, ga);
      assert_exact("F32 batched SsmControls beta", rb, gb);
    }
    HIP_CHECK(hipFree(d_qkv));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_rq));
    HIP_CHECK(hipFree(d_rg));
    HIP_CHECK(hipFree(d_ra));
    HIP_CHECK(hipFree(d_rb));
    HIP_CHECK(hipFree(d_gq));
    HIP_CHECK(hipFree(d_gg));
    HIP_CHECK(hipFree(d_ga));
    HIP_CHECK(hipFree(d_gb));
  }
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen Q8 GEMV ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestQ8KBlockGEMVEquivalence();
  TestQ8KSmallBatchFp32GEMMEquivalence();
  TestQ8_0SmallBatchFp32GEMMEquivalence(16, 512);
  TestQ8_0SmallBatchFp32GEMMEquivalence(17, 512);
  TestQ8_0SmallBatchFp32GEMMEquivalence(2047, 512);
  TestQ8_0SmallBatchFp32GEMMEquivalence(2048, 512);
  TestQ8_0SmallBatchFp32GEMMEquivalence(512, 2048);
  TestQ8_0SmallBatchFp32GEMMEquivalence(513, 2048);
  TestQ8_0SmallBatchFp32GEMMEquivalence(2048, 4096);
  TestQ8_0SmallBatchFp32GEMMEquivalence(4096, 2048);
  TestQ8_0SmallBatchFp32GEMMEquivalence(8192, 2048);
  TestQ8_0SmallBatchFp32GEMMEquivalence(100001, 2048);
  TestQ8_0SmallBatchFp32GEMMEquivalence(48, 5120);
  TestQ8_0SmallBatchFp32GEMMEquivalence(96, 5120);
  TestQ8_0SmallBatchFp32GEMMEquivalence(5120, 5120);
  TestQ8_0SmallBatchFp32GEMMEquivalence(17408, 5120);
  TestQ8_0SmallBatchFp32GEMMEquivalence(5120, 17408);
  TestQ8_0BlockGEMVEquivalence();
  TestQ8_0FusedAndExtremeBitwiseParity();
  std::cout << "Qwen Q8 GEMV ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen Q8 GEMV ops test.\n";
  return 77;
#endif
}
