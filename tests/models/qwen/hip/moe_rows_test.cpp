// MoE feed-forward for several rows (MoeFfnRows) on real weights: every row
// must get, bit for bit, the output MoeFfnDecodeToken computes for it alone.
// Prompt chunks shorter than the grouped-expert prefill and batched
// verification both rely on this.
//
// qwen_moe_rows_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it or for a model without experts.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/moe_ffn.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/moe_vec.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

// Rows shaped like normalized hidden states: unit RMS, varied direction.
std::vector<float> Rows(std::size_t rows, std::size_t width,
                        std::uint64_t seed) {
  std::vector<float> values(rows * width);
  std::uint64_t state = seed;
  for (auto& value : values) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u = (static_cast<float>(state >> 40) / 8388608.0F) - 1.0F;
    value = u * u * u * 2.6F;
  }
  return values;
}

// Synthetic packed weight buffer for K = 512, M = 2048 across Q4_K (12),
// Q5_K (13), and Q6_K (14), including an infinite scale on expert 0 row 0 to
// verify non-finite zeroing alongside normal and inactive (-1) expert IDs.
std::vector<std::uint8_t> MakeDownWeights(gufo::core::GgmlType type,
                                          std::size_t experts, std::size_t rows,
                                          std::size_t cols,
                                          std::uint64_t seed) {
  const std::size_t block_bytes = type == gufo::core::GgmlType::kQ4_K   ? 144
                                  : type == gufo::core::GgmlType::kQ5_K ? 176
                                                                        : 210;
  const std::size_t blocks_per_row = cols / 256;
  const std::size_t total_blocks = experts * rows * blocks_per_row;
  std::vector<std::uint8_t> data(total_blocks * block_bytes);
  std::uint64_t state = seed;
  for (auto& byte : data) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    byte = static_cast<std::uint8_t>(state >> 56);
  }
  for (std::size_t b = 0; b < total_blocks; ++b) {
    std::uint8_t* ptr = data.data() + b * block_bytes;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const float d =
        b == 0 ? INFINITY
               : (static_cast<float>((state >> 40) % 200 + 1) * 0.001F);
    const float m = static_cast<float>((state >> 48) % 200 + 1) * 0.001F;
    const __half dh = __float2half(d);
    const __half mh = __float2half(m);
    if (type == gufo::core::GgmlType::kQ6_K) {
      std::memcpy(ptr + block_bytes - sizeof(__half), &dh, sizeof(__half));
    } else {
      std::memcpy(ptr, &dh, sizeof(__half));
      std::memcpy(ptr + sizeof(__half), &mh, sizeof(__half));
    }
  }
  return data;
}

bool TestDownExpsDeduplication() {
  constexpr int kExperts = 16;
  constexpr int kUsed = 8;
  constexpr int kCols = 512;
  constexpr int kMaxRows = 16;
  constexpr int kMaxSlots = kMaxRows * kUsed;
  const auto x = Rows(kMaxSlots, kCols, 0x9E3779B97F4A7C15ULL);
  float* d_x = nullptr;
  std::int32_t* d_ids = nullptr;
  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ids, kMaxSlots * sizeof(std::int32_t)));
  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  bool all_ok = true;
  for (const auto type :
       {gufo::core::GgmlType::kQ4_K, gufo::core::GgmlType::kQ5_K,
        gufo::core::GgmlType::kQ6_K}) {
    for (const int rows : {2048, 2047}) {
      const auto weights =
          MakeDownWeights(type, kExperts, static_cast<std::size_t>(rows), kCols,
                          0xBF58476D1CE4E5B9ULL + static_cast<int>(type));
      void* d_w = nullptr;
      float* d_single = nullptr;
      float* d_batched = nullptr;
      HIP_CHECK(hipMalloc(&d_w, weights.size()));
      HIP_CHECK(hipMalloc(&d_single, static_cast<std::size_t>(kMaxSlots) *
                                         rows * sizeof(float)));
      HIP_CHECK(hipMalloc(&d_batched, static_cast<std::size_t>(kMaxSlots) *
                                          rows * sizeof(float)));
      HIP_CHECK(hipMemcpy(d_w, weights.data(), weights.size(),
                          hipMemcpyHostToDevice));

      // Patterns: 0 = all distinct, 1 = all identical expert, 2 = mixed 2..8
      // collisions + inactive (-1) + non-finite expert 0, 3 = realistic pair
      // overlap across token rows.
      for (int pattern = 0; pattern < 4; ++pattern) {
        std::vector<std::int32_t> ids(kMaxSlots);
        for (int s = 0; s < kMaxSlots; ++s) {
          if (pattern == 0) {
            ids[s] = s % kExperts;
          } else if (pattern == 1) {
            ids[s] = 5;
          } else if (pattern == 2) {
            ids[s] = (s % 7 == 0) ? -1 : (s % 4);
          } else {
            const int t = s / kUsed;
            const int u = s % kUsed;
            ids[s] = (t > 0 && u < 4) ? ids[(t - 1) * kUsed + u]
                                      : ((t * 5 + u * 3 + 1) % kExperts);
          }
        }
        HIP_CHECK(hipMemcpy(d_ids, ids.data(),
                            ids.size() * sizeof(std::int32_t),
                            hipMemcpyHostToDevice));
        for (int s = 0; s < kMaxSlots; ++s) {
          if (qfn_mmq_moe_vec(
                  static_cast<int>(type), d_w, d_x + s * kCols, d_ids + s,
                  d_single + static_cast<std::size_t>(s) * rows, rows, kCols, 1,
                  kExperts, 1, nullptr, nullptr, nullptr) != 0) {
            throw std::runtime_error("single-slot qfn_mmq_moe_vec failed");
          }
        }
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> single(static_cast<std::size_t>(kMaxSlots) * rows);
        HIP_CHECK(hipMemcpy(single.data(), d_single,
                            single.size() * sizeof(float),
                            hipMemcpyDeviceToHost));

        for (const int tok_rows : {1, 2, 3, 4, 8, 9, 12, 16}) {
          const int n_slots = tok_rows * kUsed;
          HIP_CHECK(hipMemset(
              d_batched, 0x7F,
              static_cast<std::size_t>(n_slots) * rows * sizeof(float)));
          if (qfn_mmq_moe_vec(static_cast<int>(type), d_w, d_x, d_ids,
                              d_batched, rows, kCols, n_slots, kExperts, 1,
                              nullptr, nullptr, nullptr) != 0) {
            throw std::runtime_error("batched qfn_mmq_moe_vec failed");
          }
          HIP_CHECK(hipDeviceSynchronize());
          std::vector<float> batched(static_cast<std::size_t>(n_slots) * rows);
          HIP_CHECK(hipMemcpy(batched.data(), d_batched,
                              batched.size() * sizeof(float),
                              hipMemcpyDeviceToHost));
          const bool ok = std::memcmp(single.data(), batched.data(),
                                      batched.size() * sizeof(float)) == 0;
          if (!ok) {
            std::cerr << "FAIL down_exps dedup: type=" << static_cast<int>(type)
                      << " M=" << rows << " pattern=" << pattern
                      << " tok_rows=" << tok_rows << '\n';
            all_ok = false;
          }
        }
      }
      HIP_CHECK(hipFree(d_w));
      HIP_CHECK(hipFree(d_single));
      HIP_CHECK(hipFree(d_batched));
    }
  }
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_ids));
  std::cout << "down_exps_dedup_synthetic=" << (all_ok ? "PASS" : "FAIL")
            << '\n';
  return all_ok;
}

std::vector<hip_bfloat16> ToBf16(const std::vector<float>& values) {
  std::vector<hip_bfloat16> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = hip_bfloat16(values[i]);
  }
  return out;
}

bool TestFusedRouterParity() {
  constexpr std::size_t kMaxExperts = 256;
  constexpr std::size_t kMaxUsed = 8;
  constexpr std::size_t kMaxHidden = 2056;
  constexpr std::size_t kMaxRows = 70;
  constexpr std::size_t kSyncCap = 8;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  void* d_router = nullptr;
  void* d_shared = nullptr;
  float* d_x = nullptr;
  float* d_ref_logits = nullptr;
  float* d_ref_slogit = nullptr;
  std::int32_t* d_ref_ids = nullptr;
  float* d_ref_weights = nullptr;
  float* d_ref_sweight = nullptr;
  float* d_fused_logits = nullptr;
  float* d_fused_slogit = nullptr;
  std::int32_t* d_fused_ids = nullptr;
  float* d_fused_weights = nullptr;
  float* d_fused_sweight = nullptr;
  std::uint32_t* d_sync = nullptr;

  HIP_CHECK(hipMalloc(&d_router, kMaxExperts * kMaxHidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_shared, kMaxHidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x, kMaxRows * kMaxHidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ref_logits, kMaxRows * kMaxExperts * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ref_slogit, kMaxRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ref_ids, kMaxRows * kMaxUsed * sizeof(std::int32_t)));
  HIP_CHECK(hipMalloc(&d_ref_weights, kMaxRows * kMaxUsed * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ref_sweight, kMaxRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fused_logits, kMaxRows * kMaxExperts * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fused_slogit, kMaxRows * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_fused_ids, kMaxRows * kMaxUsed * sizeof(std::int32_t)));
  HIP_CHECK(hipMalloc(&d_fused_weights, kMaxRows * kMaxUsed * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fused_sweight, kMaxRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sync, kSyncCap * sizeof(std::uint32_t)));
  HIP_CHECK(hipMemset(d_sync, 0, kSyncCap * sizeof(std::uint32_t)));

  struct RouterCase {
    const char* name;
    gufo::core::GgmlType router_type;
    gufo::core::GgmlType shared_type;
    std::size_t experts;
    std::size_t used;
    std::size_t hidden;
    int pattern;  // 0=normal, 1=ties, 2=all_equal, 3=extremes_neg_zero
  };
  const std::vector<RouterCase> cases = {
      {"f32_prod", gufo::core::GgmlType::kF32, gufo::core::GgmlType::kF32, 256,
       8, 2048, 0},
      {"f32_ties", gufo::core::GgmlType::kF32, gufo::core::GgmlType::kF32, 256,
       8, 2048, 1},
      {"f32_all_equal", gufo::core::GgmlType::kF32, gufo::core::GgmlType::kF32,
       256, 8, 2048, 2},
      {"f32_extremes", gufo::core::GgmlType::kF32, gufo::core::GgmlType::kF32,
       256, 8, 2048, 3},
      {"f32_tail_k2051", gufo::core::GgmlType::kF32, gufo::core::GgmlType::kF32,
       256, 8, 2051, 0},
      {"f32_small_e16", gufo::core::GgmlType::kF32, gufo::core::GgmlType::kF32,
       16, 2, 512, 1},
      {"bf16_prod", gufo::core::GgmlType::kBF16, gufo::core::GgmlType::kBF16,
       256, 8, 2048, 0},
      {"bf16_ties", gufo::core::GgmlType::kBF16, gufo::core::GgmlType::kBF16,
       64, 4, 1024, 1},
      {"mixed_f32_bf16", gufo::core::GgmlType::kF32,
       gufo::core::GgmlType::kBF16, 256, 8, 2048, 0},
      {"mixed_bf16_f32", gufo::core::GgmlType::kBF16,
       gufo::core::GgmlType::kF32, 256, 8, 2048, 0},
  };

  bool all_ok = true;
  std::size_t checked = 0;
  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const auto& c = cases[ci];
    auto h_router =
        Rows(c.experts, c.hidden, 0x3C6EF372FE94F82BULL + ci * 131U);
    auto h_shared = Rows(1, c.hidden, 0xA54FF53A5F1D36F1ULL + ci * 97U);
    auto h_x = Rows(kMaxRows, c.hidden, 0x510E527FADE682D1ULL + ci * 73U);
    if (c.pattern == 1) {
      // Force exact multi-way ties across groups of experts.
      for (std::size_t e = 1; e < c.experts; ++e) {
        if ((e % 4) != 0) {
          std::memcpy(&h_router[e * c.hidden], &h_router[(e & ~3U) * c.hidden],
                      c.hidden * sizeof(float));
        }
      }
    } else if (c.pattern == 2) {
      for (std::size_t e = 1; e < c.experts; ++e) {
        std::memcpy(&h_router[e * c.hidden], h_router.data(),
                    c.hidden * sizeof(float));
      }
    } else if (c.pattern == 3) {
      for (std::size_t i = 0; i < h_x.size(); ++i) {
        h_x[i] = (i % 17 == 0) ? -0.0F : (h_x[i] * 25.0F);
      }
    }

    if (c.router_type == gufo::core::GgmlType::kBF16) {
      const auto bf = ToBf16(h_router);
      HIP_CHECK(hipMemcpyAsync(d_router, bf.data(),
                               bf.size() * sizeof(hip_bfloat16),
                               hipMemcpyHostToDevice, stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(d_router, h_router.data(),
                               h_router.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream));
    }
    if (c.shared_type == gufo::core::GgmlType::kBF16) {
      const auto bf = ToBf16(h_shared);
      HIP_CHECK(hipMemcpyAsync(d_shared, bf.data(),
                               bf.size() * sizeof(hip_bfloat16),
                               hipMemcpyHostToDevice, stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(d_shared, h_shared.data(),
                               h_shared.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream));
    }
    HIP_CHECK(hipMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));

    const std::vector<std::size_t> row_counts =
        ci == 0 ? std::vector<std::size_t>{1, 2, 3,  4,  5,  6,  7,
                                           8, 9, 12, 16, 26, 63, 70}
                : std::vector<std::size_t>{1, 2, 4, 8, 16};
    for (const std::size_t rows : row_counts) {
      ++checked;
      HIP_CHECK(hipMemsetAsync(d_fused_logits, 0x5A,
                               rows * c.experts * sizeof(float), stream));
      HIP_CHECK(
          hipMemsetAsync(d_fused_slogit, 0x5A, rows * sizeof(float), stream));
      HIP_CHECK(hipMemsetAsync(d_fused_ids, 0x5A,
                               rows * c.used * sizeof(std::int32_t), stream));
      HIP_CHECK(hipMemsetAsync(d_fused_weights, 0x5A,
                               rows * c.used * sizeof(float), stream));
      HIP_CHECK(
          hipMemsetAsync(d_fused_sweight, 0x5A, rows * sizeof(float), stream));

      gufo::hip::LaunchMoeRouterLogits(
          d_router, c.router_type, d_shared, c.shared_type, d_x, d_ref_logits,
          d_ref_slogit, c.experts, c.hidden, stream, rows);
      gufo::hip::LaunchMoeRouterTopK(d_ref_logits, d_ref_slogit, d_ref_ids,
                                     d_ref_weights, d_ref_sweight,
                                     static_cast<std::uint32_t>(c.experts),
                                     static_cast<std::uint32_t>(c.used),
                                     static_cast<std::uint32_t>(rows), stream);

      gufo::hip::LaunchMoeFusedRouterTopK(
          d_router, c.router_type, d_shared, c.shared_type, d_x, d_fused_logits,
          d_fused_slogit, d_fused_ids, d_fused_weights, d_fused_sweight, d_sync,
          kSyncCap, c.experts, c.used, c.hidden, rows, stream);
      HIP_CHECK(hipStreamSynchronize(stream));

      std::vector<float> ref_logits(rows * c.experts);
      std::vector<float> fused_logits(rows * c.experts);
      std::vector<float> ref_slogit(rows);
      std::vector<float> fused_slogit(rows);
      std::vector<std::int32_t> ref_ids(rows * c.used);
      std::vector<std::int32_t> fused_ids(rows * c.used);
      std::vector<float> ref_w(rows * c.used);
      std::vector<float> fused_w(rows * c.used);
      std::vector<float> ref_sw(rows);
      std::vector<float> fused_sw(rows);
      std::vector<std::uint32_t> sync_after(kSyncCap);

      HIP_CHECK(hipMemcpy(ref_logits.data(), d_ref_logits,
                          ref_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_logits.data(), d_fused_logits,
                          fused_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(ref_slogit.data(), d_ref_slogit,
                          ref_slogit.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_slogit.data(), d_fused_slogit,
                          fused_slogit.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(ref_ids.data(), d_ref_ids,
                          ref_ids.size() * sizeof(std::int32_t),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_ids.data(), d_fused_ids,
                          fused_ids.size() * sizeof(std::int32_t),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(ref_w.data(), d_ref_weights,
                          ref_w.size() * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_w.data(), d_fused_weights,
                          fused_w.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(ref_sw.data(), d_ref_sweight,
                          ref_sw.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_sw.data(), d_fused_sweight,
                          fused_sw.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(sync_after.data(), d_sync,
                          sync_after.size() * sizeof(std::uint32_t),
                          hipMemcpyDeviceToHost));

      bool sync_zero = true;
      for (const auto v : sync_after) {
        sync_zero = sync_zero && (v == 0U);
      }
      const bool ok = sync_zero &&
                      std::memcmp(ref_logits.data(), fused_logits.data(),
                                  ref_logits.size() * sizeof(float)) == 0 &&
                      std::memcmp(ref_slogit.data(), fused_slogit.data(),
                                  ref_slogit.size() * sizeof(float)) == 0 &&
                      std::memcmp(ref_ids.data(), fused_ids.data(),
                                  ref_ids.size() * sizeof(std::int32_t)) == 0 &&
                      std::memcmp(ref_w.data(), fused_w.data(),
                                  ref_w.size() * sizeof(float)) == 0 &&
                      std::memcmp(ref_sw.data(), fused_sw.data(),
                                  ref_sw.size() * sizeof(float)) == 0;
      if (!ok) {
        std::cerr << "FAIL fused_router: case=" << c.name << " rows=" << rows
                  << " sync_zero=" << sync_zero << '\n';
        all_ok = false;
      }
    }
  }

  // Verify HIP graph capture + multiple replays self-reset sync counters and
  // match the two-kernel reference bit for bit on changed activations.
  {
    constexpr std::size_t kRows = 4;
    constexpr std::size_t kExperts = 256;
    constexpr std::size_t kUsed = 8;
    constexpr std::size_t kHidden = 2048;
    const auto h_router = Rows(kExperts, kHidden, 0x9B05688C2B3E6C1FULL);
    const auto h_shared = Rows(1, kHidden, 0x1F83D9ABFB41BD6BULL);
    HIP_CHECK(hipMemcpy(d_router, h_router.data(),
                        h_router.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_shared, h_shared.data(),
                        h_shared.size() * sizeof(float),
                        hipMemcpyHostToDevice));

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    gufo::hip::LaunchMoeFusedRouterTopK(
        d_router, gufo::core::GgmlType::kF32, d_shared,
        gufo::core::GgmlType::kF32, d_x, d_fused_logits, d_fused_slogit,
        d_fused_ids, d_fused_weights, d_fused_sweight, d_sync, kSyncCap,
        kExperts, kUsed, kHidden, kRows, stream);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (int replay = 0; replay < 3; ++replay) {
      ++checked;
      const auto h_x = Rows(kRows, kHidden, 0x5BE0CD19137E2179ULL + replay);
      HIP_CHECK(hipMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream));
      gufo::hip::LaunchMoeRouterLogits(d_router, gufo::core::GgmlType::kF32,
                                       d_shared, gufo::core::GgmlType::kF32,
                                       d_x, d_ref_logits, d_ref_slogit,
                                       kExperts, kHidden, stream, kRows);
      gufo::hip::LaunchMoeRouterTopK(d_ref_logits, d_ref_slogit, d_ref_ids,
                                     d_ref_weights, d_ref_sweight, kExperts,
                                     kUsed, kRows, stream);
      HIP_CHECK(hipGraphLaunch(exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      std::vector<std::int32_t> ref_ids(kRows * kUsed);
      std::vector<std::int32_t> fused_ids(kRows * kUsed);
      std::vector<float> ref_w(kRows * kUsed);
      std::vector<float> fused_w(kRows * kUsed);
      std::vector<float> ref_sw(kRows);
      std::vector<float> fused_sw(kRows);
      HIP_CHECK(hipMemcpy(ref_ids.data(), d_ref_ids,
                          ref_ids.size() * sizeof(std::int32_t),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_ids.data(), d_fused_ids,
                          fused_ids.size() * sizeof(std::int32_t),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(ref_w.data(), d_ref_weights,
                          ref_w.size() * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_w.data(), d_fused_weights,
                          fused_w.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(ref_sw.data(), d_ref_sweight,
                          ref_sw.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(fused_sw.data(), d_fused_sweight,
                          fused_sw.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const bool ok = std::memcmp(ref_ids.data(), fused_ids.data(),
                                  ref_ids.size() * sizeof(std::int32_t)) == 0 &&
                      std::memcmp(ref_w.data(), fused_w.data(),
                                  ref_w.size() * sizeof(float)) == 0 &&
                      std::memcmp(ref_sw.data(), fused_sw.data(),
                                  ref_sw.size() * sizeof(float)) == 0;
      if (!ok) {
        std::cerr << "FAIL fused_router graph replay=" << replay << '\n';
        all_ok = false;
      }
    }
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  HIP_CHECK(hipFree(d_sync));
  HIP_CHECK(hipFree(d_fused_sweight));
  HIP_CHECK(hipFree(d_fused_weights));
  HIP_CHECK(hipFree(d_fused_ids));
  HIP_CHECK(hipFree(d_fused_slogit));
  HIP_CHECK(hipFree(d_fused_logits));
  HIP_CHECK(hipFree(d_ref_sweight));
  HIP_CHECK(hipFree(d_ref_weights));
  HIP_CHECK(hipFree(d_ref_ids));
  HIP_CHECK(hipFree(d_ref_slogit));
  HIP_CHECK(hipFree(d_ref_logits));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_shared));
  HIP_CHECK(hipFree(d_router));
  HIP_CHECK(hipStreamDestroy(stream));
  std::cout << "fused_router_synthetic=" << (all_ok ? "PASS" : "FAIL")
            << " checked=" << checked << '\n';
  return all_ok;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_moe_rows_test [MODEL.gguf]\n";
    return 2;
  }
  try {
    if (!TestDownExpsDeduplication() || !TestFusedRouterParity()) {
      return 1;
    }
  } catch (const std::exception& failure) {
    std::cerr << "qwen_moe_rows_test (synthetic): " << failure.what() << '\n';
    return 1;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
  }
  try {
    gufo::models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
    gufo::models::qwen38_flash_next::rocm::RegisterQwenNpuFfn();
    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(path, &error);
    if (!reader_owner)
      throw std::runtime_error("open: " + error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(reader_owner));
    const auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (!model)
      throw std::runtime_error("model: " + error);
    const auto& config = model->GetConfig();
    if (!config.IsMoe()) {
      std::cerr << "the model has no experts; skipping\n";
      return 77;
    }
    const auto& weights = model->GetWeights();
    gufo::hip::QwenGpuArena arena(config, 4096);
    const std::size_t hidden = config.hidden_size;
    // More rows than the scratch holds, so the chunking is exercised too.
    constexpr std::size_t kMaxRows = gufo::hip::kMoeFfnScratchRows + 6;
    float* device_x = nullptr;
    float* device_rows = nullptr;
    float* device_single = nullptr;
    HIP_CHECK(hipMalloc(&device_x, kMaxRows * hidden * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_rows, kMaxRows * hidden * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_single, kMaxRows * hidden * sizeof(float)));
    gufo::hip::WarmMoeDecode(weights, config, arena.moe, device_x,
                             arena.stream);
    bool passed = true;
    for (const std::size_t layer_index :
         {std::size_t{0}, std::size_t{7}, weights.layers.size() / 2,
          weights.layers.size() - 1}) {
      const auto& layer = weights.layers[layer_index];
      for (const std::size_t rows :
           {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
            std::size_t{8}, std::size_t{9}, std::size_t{12}, std::size_t{16},
            std::size_t{26}, std::size_t{63}, kMaxRows}) {
        const auto x = Rows(rows, hidden, 0xD1B54A32D192ED03ULL + layer_index);
        HIP_CHECK(hipMemcpy(device_x, x.data(), x.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        gufo::hip::MoeFfnRows(layer, config, arena, device_x, device_rows,
                              rows);
        HIP_CHECK(hipStreamSynchronize(arena.stream));
        std::vector<float> batched(rows * hidden);
        HIP_CHECK(hipMemcpy(batched.data(), device_rows,
                            batched.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        for (std::size_t row = 0; row < rows; ++row) {
          gufo::hip::MoeFfnDecodeToken(
              layer, config, arena.moe, device_x + (row * hidden),
              device_single + (row * hidden), arena.stream);
          HIP_CHECK(hipStreamSynchronize(arena.stream));
        }
        std::vector<float> single(rows * hidden);
        HIP_CHECK(hipMemcpy(single.data(), device_single,
                            single.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        bool nonzero = false;
        for (const float value : batched)
          nonzero = nonzero || value != 0.0F;
        const bool equal = std::memcmp(batched.data(), single.data(),
                                       batched.size() * sizeof(float)) == 0;
        std::size_t differing = 0;
        for (std::size_t row = 0; row < rows; ++row)
          differing +=
              std::memcmp(&batched[row * hidden], &single[row * hidden],
                          hidden * sizeof(float)) != 0;
        const bool ok = nonzero && equal;
        std::cout << "layer=" << layer_index << ",rows=" << rows
                  << ",differing_rows=" << differing << ','
                  << (ok ? "PASS" : "FAIL") << '\n';
        passed = passed && ok;
      }
    }
    HIP_CHECK(hipFree(device_x));
    HIP_CHECK(hipFree(device_rows));
    HIP_CHECK(hipFree(device_single));
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_moe_rows_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen MoE rows test.\n";
  return 77;
#endif
}
