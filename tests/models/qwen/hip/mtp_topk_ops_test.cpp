// GPU top-k draft distribution helper (LaunchMtpRowTopK + MtpTopKSoftmax)
// against the CPU reference (MtpTopKDistribution), bit for bit on ids and
// probabilities across realistic logits, ties, signed zeros, infinities, NaNs,
// scratch fallbacks, and HIP graph replay.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/mtp_policy.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

using gufo::hip::kMaxMtpRowTopK;
using gufo::hip::LaunchMtpRowTopK;
using gufo::hip::MtpTopKCandidate;
using gufo::speculative::MtpTopKDistribution;
using gufo::speculative::MtpTopKSoftmax;
using gufo::tokenization::TokenId;

std::uint64_t NextRng(std::uint64_t* state) {
  std::uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

float UniformFloat(std::uint64_t* state) {
  return static_cast<float>(NextRng(state) >> 40) * (1.0F / 16777216.0F);
}

enum class Pattern {
  kRealistic,
  kHeavyTies,
  kSignedZeroTies,
  kAllEqual,
  kNegInf,
  kPosInfMix,
  kMixedNan,
  kAllNan,
  kExtremeRange,
  kAscending,
};

const char* PatternName(Pattern pattern) {
  switch (pattern) {
    case Pattern::kRealistic:
      return "realistic";
    case Pattern::kHeavyTies:
      return "heavy_ties";
    case Pattern::kSignedZeroTies:
      return "signed_zero_ties";
    case Pattern::kAllEqual:
      return "all_equal";
    case Pattern::kNegInf:
      return "neg_inf";
    case Pattern::kPosInfMix:
      return "pos_inf_mix";
    case Pattern::kMixedNan:
      return "mixed_nan";
    case Pattern::kAllNan:
      return "all_nan";
    case Pattern::kExtremeRange:
      return "extreme_range";
    case Pattern::kAscending:
      return "ascending";
  }
  return "unknown";
}

void FillRow(std::span<float> row, Pattern pattern, std::uint64_t seed) {
  std::uint64_t state = seed;
  const std::size_t n = row.size();
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  switch (pattern) {
    case Pattern::kRealistic:
      for (std::size_t i = 0; i < n; ++i) {
        row[i] = -6.0F + 8.0F * UniformFloat(&state);
      }
      for (std::size_t p = 0; p < std::min<std::size_t>(64, n); ++p) {
        const std::size_t idx = NextRng(&state) % n;
        row[idx] = 6.0F + 14.0F * UniformFloat(&state);
      }
      break;
    case Pattern::kHeavyTies:
      for (std::size_t i = 0; i < n; ++i) {
        row[i] = static_cast<float>(static_cast<int>(NextRng(&state) % 7) - 3);
      }
      break;
    case Pattern::kSignedZeroTies:
      for (std::size_t i = 0; i < n; ++i) {
        if ((i & 3U) == 0U) {
          row[i] = -0.0F;
        } else if ((i & 3U) == 1U) {
          row[i] = +0.0F;
        } else {
          row[i] = -1.0F;
        }
      }
      break;
    case Pattern::kAllEqual:
      std::fill(row.begin(), row.end(), 1.25F);
      break;
    case Pattern::kNegInf:
      std::fill(row.begin(), row.end(), -inf);
      break;
    case Pattern::kPosInfMix:
      for (std::size_t i = 0; i < n; ++i) {
        row[i] = -10.0F + 20.0F * UniformFloat(&state);
      }
      if (n > 0) {
        row[n / 2] = inf;
      }
      if (n > 5) {
        row[n - 1] = inf;
        row[3] = -inf;
      }
      break;
    case Pattern::kMixedNan:
      for (std::size_t i = 0; i < n; ++i) {
        row[i] = (i % 5 == 0) ? qnan : (-4.0F + 8.0F * UniformFloat(&state));
      }
      if (n > 0) {
        row[0] = qnan;
        row[n - 1] = qnan;
      }
      break;
    case Pattern::kAllNan:
      std::fill(row.begin(), row.end(), qnan);
      break;
    case Pattern::kExtremeRange:
      for (std::size_t i = 0; i < n; ++i) {
        row[i] = ((i & 1U) == 0U ? -1.0e4F : 1.0e4F) *
                 (0.5F + 0.5F * UniformFloat(&state));
      }
      break;
    case Pattern::kAscending:
      for (std::size_t i = 0; i < n; ++i) {
        row[i] = static_cast<float>(i) * 1.0e-4F;
      }
      break;
  }
}

struct DeviceWorkspace {
  float* d_logits{nullptr};
  MtpTopKCandidate* d_out{nullptr};
  float* d_scratch{nullptr};
  MtpTopKCandidate* h_pinned{nullptr};
  std::size_t max_logits{0};
  std::size_t max_out{0};
  std::size_t scratch_floats{0};
  hipStream_t stream{nullptr};

  DeviceWorkspace(std::size_t logits_cap, std::size_t out_cap,
                  std::size_t scratch_cap)
      : max_logits(logits_cap), max_out(out_cap), scratch_floats(scratch_cap) {
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMalloc(&d_logits, max_logits * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, max_out * sizeof(MtpTopKCandidate)));
    HIP_CHECK(hipMalloc(&d_scratch, scratch_floats * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&h_pinned, max_out * sizeof(MtpTopKCandidate)));
  }

  ~DeviceWorkspace() {
    if (h_pinned != nullptr)
      (void)hipHostFree(h_pinned);
    if (d_scratch != nullptr)
      (void)hipFree(d_scratch);
    if (d_out != nullptr)
      (void)hipFree(d_out);
    if (d_logits != nullptr)
      (void)hipFree(d_logits);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
};

bool VerifyCase(DeviceWorkspace& ws, std::size_t batch, std::size_t vocab,
                std::size_t k, Pattern pattern, std::size_t scratch_offset,
                std::size_t scratch_len, std::uint64_t seed) {
  std::vector<float> host_logits(batch * vocab);
  for (std::size_t row = 0; row < batch; ++row) {
    FillRow(std::span<float>(host_logits.data() + row * vocab, vocab), pattern,
            seed + row * 0x9E3779B97F4A7C15ULL);
  }
  HIP_CHECK(hipMemcpyAsync(ws.d_logits, host_logits.data(),
                           host_logits.size() * sizeof(float),
                           hipMemcpyHostToDevice, ws.stream));
  const std::size_t out_k = std::min({k, vocab, kMaxMtpRowTopK});
  HIP_CHECK(hipMemsetAsync(
      ws.d_out, 0xA5, batch * out_k * sizeof(MtpTopKCandidate), ws.stream));
  const std::span<float> scratch =
      scratch_len == 0
          ? std::span<float>{}
          : std::span<float>(ws.d_scratch + scratch_offset, scratch_len);
  LaunchMtpRowTopK(ws.d_logits, ws.d_out, batch, vocab, k, scratch, ws.stream);
  HIP_CHECK(hipMemcpyAsync(ws.h_pinned, ws.d_out,
                           batch * out_k * sizeof(MtpTopKCandidate),
                           hipMemcpyDeviceToHost, ws.stream));
  HIP_CHECK(hipStreamSynchronize(ws.stream));

  for (std::size_t row = 0; row < batch; ++row) {
    const std::span<const float> row_logits(host_logits.data() + row * vocab,
                                            vocab);
    std::vector<TokenId> ref_ids;
    std::vector<float> ref_probs;
    MtpTopKDistribution(row_logits, 1.0F, k, &ref_ids, &ref_probs);

    std::vector<TokenId> gpu_ids(out_k);
    std::vector<float> gpu_top_logits(out_k);
    for (std::size_t i = 0; i < out_k; ++i) {
      gpu_ids[i] = ws.h_pinned[row * out_k + i].id;
      gpu_top_logits[i] = ws.h_pinned[row * out_k + i].logit;
    }
    std::vector<float> gpu_probs;
    MtpTopKSoftmax(gpu_top_logits, 1.0F, &gpu_probs);

    if (gpu_ids != ref_ids || gpu_probs.size() != ref_probs.size() ||
        std::memcmp(gpu_probs.data(), ref_probs.data(),
                    ref_probs.size() * sizeof(float)) != 0) {
      std::cerr << "FAIL mtp_topk: batch=" << batch << " vocab=" << vocab
                << " k=" << k << " pattern=" << PatternName(pattern)
                << " scratch_offset=" << scratch_offset
                << " scratch_len=" << scratch_len << " row=" << row << '\n';
      for (std::size_t i = 0; i < std::min<std::size_t>(out_k, 8); ++i) {
        std::cerr << "  [" << i << "] gpu=(" << gpu_ids[i] << ", "
                  << gpu_top_logits[i] << ", p=" << gpu_probs[i] << ") ref=("
                  << ref_ids[i] << ", p=" << ref_probs[i] << ")\n";
      }
      return false;
    }
  }
  return true;
}

bool TestHipGraphReplay(DeviceWorkspace& ws) {
  constexpr std::size_t kBatch = 4;
  constexpr std::size_t kVocab = 161074;
  constexpr std::size_t kTopK = 32;
  const std::span<float> scratch(ws.d_scratch, ws.scratch_floats);

  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipStreamBeginCapture(ws.stream, hipStreamCaptureModeGlobal));
  LaunchMtpRowTopK(ws.d_logits, ws.d_out, kBatch, kVocab, kTopK, scratch,
                   ws.stream);
  HIP_CHECK(hipStreamEndCapture(ws.stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  bool ok = true;
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<float> host_logits(kBatch * kVocab);
    for (std::size_t row = 0; row < kBatch; ++row) {
      FillRow(std::span<float>(host_logits.data() + row * kVocab, kVocab),
              replay == 0 ? Pattern::kRealistic : Pattern::kHeavyTies,
              0x123456789ABCDEF0ULL + static_cast<std::uint64_t>(replay) * 97U +
                  row * 31U);
    }
    HIP_CHECK(hipMemcpyAsync(ws.d_logits, host_logits.data(),
                             host_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, ws.stream));
    HIP_CHECK(hipGraphLaunch(exec, ws.stream));
    HIP_CHECK(hipMemcpyAsync(ws.h_pinned, ws.d_out,
                             kBatch * kTopK * sizeof(MtpTopKCandidate),
                             hipMemcpyDeviceToHost, ws.stream));
    HIP_CHECK(hipStreamSynchronize(ws.stream));

    for (std::size_t row = 0; row < kBatch; ++row) {
      std::vector<TokenId> ref_ids;
      std::vector<float> ref_probs;
      MtpTopKDistribution(
          std::span<const float>(host_logits.data() + row * kVocab, kVocab),
          1.0F, kTopK, &ref_ids, &ref_probs);
      std::vector<TokenId> gpu_ids(kTopK);
      std::vector<float> gpu_top_logits(kTopK);
      for (std::size_t i = 0; i < kTopK; ++i) {
        gpu_ids[i] = ws.h_pinned[row * kTopK + i].id;
        gpu_top_logits[i] = ws.h_pinned[row * kTopK + i].logit;
      }
      std::vector<float> gpu_probs;
      MtpTopKSoftmax(gpu_top_logits, 1.0F, &gpu_probs);
      if (gpu_ids != ref_ids ||
          std::memcmp(gpu_probs.data(), ref_probs.data(),
                      ref_probs.size() * sizeof(float)) != 0) {
        std::cerr << "FAIL mtp_topk graph replay=" << replay << " row=" << row
                  << '\n';
        ok = false;
      }
    }
  }
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  return ok;
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "qwen_mtp_topk_ops_test");
  if (gate != gufo::test::kHipTestSuccess) {
    return gate;
  }
  try {
    constexpr std::size_t kMaxBatch = 8;
    constexpr std::size_t kMaxVocab = 248320;
    // Matches QwenMtpGpuExecutor's split-k scratch capacity (~132K floats).
    constexpr std::size_t kScratchFloats = 132096;
    DeviceWorkspace ws(4 * kMaxVocab, kMaxBatch * kMaxMtpRowTopK,
                       kScratchFloats);

    const std::vector<Pattern> all_patterns = {
        Pattern::kRealistic, Pattern::kHeavyTies, Pattern::kSignedZeroTies,
        Pattern::kAllEqual,  Pattern::kNegInf,    Pattern::kPosInfMix,
        Pattern::kMixedNan,  Pattern::kAllNan,    Pattern::kExtremeRange,
        Pattern::kAscending,
    };

    std::size_t checked = 0;
    bool all_ok = true;
    for (const std::size_t vocab :
         {std::size_t{1}, std::size_t{16}, std::size_t{31}, std::size_t{32},
          std::size_t{33}, std::size_t{1024}, std::size_t{2049},
          std::size_t{4096}}) {
      for (const std::size_t batch :
           {std::size_t{1}, std::size_t{4}, kMaxBatch}) {
        for (const Pattern pattern : all_patterns) {
          ++checked;
          if (!VerifyCase(ws, batch, vocab, 32, pattern, 0, kScratchFloats,
                          0xA0761D6478BD642FULL + checked * 17U)) {
            all_ok = false;
          }
        }
      }
    }

    // Production draft (161,074) and full (248,320) vocabularies across C1..C4
    // and different top-k sizes.
    for (const std::size_t vocab : {std::size_t{161074}, std::size_t{248320}}) {
      for (const std::size_t batch :
           {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
        for (const Pattern pattern : all_patterns) {
          ++checked;
          if (!VerifyCase(ws, batch, vocab, 32, pattern, 0, kScratchFloats,
                          0xE7037ED1A0B428DBULL + checked * 31U)) {
            all_ok = false;
          }
        }
      }
      for (const std::size_t k : {std::size_t{1}, std::size_t{16}}) {
        ++checked;
        if (!VerifyCase(ws, 2, vocab, k, Pattern::kRealistic, 0, kScratchFloats,
                        0x8EBC6AF09C88C6E3ULL + checked * 13U)) {
          all_ok = false;
        }
      }
    }

    // Scratch alignment (odd float offset) and small/empty scratch fallbacks.
    for (const auto [offset, len] :
         std::vector<std::pair<std::size_t, std::size_t>>{
             {1, kScratchFloats - 1}, {1, 512}, {0, 256}, {0, 0}}) {
      ++checked;
      if (!VerifyCase(ws, 2, 161074, 32, Pattern::kRealistic, offset, len,
                      0x589965CC75374CC3ULL + checked * 23U)) {
        all_ok = false;
      }
    }

    ++checked;
    if (!TestHipGraphReplay(ws)) {
      all_ok = false;
    }

    std::cout << "qwen_mtp_topk_ops_test: checked=" << checked
              << " status=" << (all_ok ? "PASS" : "FAIL") << '\n';
    return all_ok ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_mtp_topk_ops_test: " << failure.what() << '\n';
    return 1;
  }
#else
  std::cout << "HIP disabled, skipping Qwen MTP top-k ops test.\n";
  return 77;
#endif
}
