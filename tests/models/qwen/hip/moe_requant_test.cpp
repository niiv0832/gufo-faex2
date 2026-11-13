// The Q8_0 copy of Q6_K expert down weights that the grouped-expert prefill
// reads (RequantToQ8_0), on the real weights of every Q6_K layer.
//
// Exact contracts, checked for every block:
//   - the device conversion is repeatable bit for bit;
//   - every converted weight is within half a Q8_0 step of the original
//     value, the bound of the format (see BlockBound).
// Measured and printed, with no threshold: how the device conversion compares
// with the same algorithm in IEEE single precision on the host (the kernel is
// built with -ffast-math, so its division need not be correctly rounded),
// and how much the conversion loses against the original Q6_K values and
// against the Q6_K quantization step.
//
// qwen_moe_requant_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it or for a model without Q6_K expert down weights.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen/hip/quant_reference.hpp"

namespace {

using gufo::test_reference::QuantCode;

struct LayerStats {
  std::size_t blocks = 0;
  std::size_t weights = 0;
  std::size_t repeat_mismatch_bytes = 0;
  std::size_t host_scale_mismatch_blocks = 0;
  std::size_t host_code_mismatch_weights = 0;
  std::size_t bound_violations = 0;
  std::size_t changed_weights = 0;
  std::size_t beyond_half_q6_step = 0;
  double sum_error2 = 0.0;
  double sum_value2 = 0.0;
  double max_error_over_block_max = 0.0;
  double sum_error_over_block_max = 0.0;
  double max_block_rms_ratio = 0.0;
  double max_error_over_q6_step = 0.0;
  double sum_error_over_q6_step2 = 0.0;
  std::size_t q6_step_weights = 0;
};

// Half a step of the stored scale s bounds the rounding of v / s to an
// integer, provided no code is clamped. The stored scale is max|v| / 127
// rounded to half precision, within 2^-11 of it, so the largest quotient is at
// most 127 * (1 + 2^-10) < 127.5 and rounds to 127 without clamping. Division
// and product are single-precision operations of relative error at most
// 2^-22 each under fast-math reciprocal approximations, which moves a
// quotient of at most 127.2 by less than 2^-13; the bound keeps that margin.
double BlockBound(double scale) {
  return scale * (0.5 + std::ldexp(1.0, -12));
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_moe_requant_test [MODEL.gguf]\n";
    return 2;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
  }
  try {
    namespace rocm = gufo::models::qwen38_flash_next::rocm;
    using gufo::core::GgmlType;
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
    const auto& weights = model->GetWeights();
    if (!config.IsMoe() || config.expert_ff % 256 != 0) {
      std::cerr << "the model has no supported experts; skipping\n";
      return 77;
    }
    const std::size_t k = config.expert_ff;
    const std::size_t rows =
        std::size_t{config.num_experts} * config.hidden_size;
    const std::size_t blocks_per_row = k / 32;
    const std::size_t q8_bytes = rows * blocks_per_row * 34;
    const std::size_t q6_row_bytes =
        gufo::test_reference::RowBytes(GgmlType::kQ6_K, k);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    void* device = nullptr;
    HIP_CHECK(hipMalloc(&device, q8_bytes));
    std::vector<std::uint8_t> converted(q8_bytes);
    std::vector<std::uint8_t> repeated(q8_bytes);
    std::vector<QuantCode> row(k);
    bool passed = true;
    std::size_t layers = 0;
    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
      const auto& down = weights.layers[l].ffn_down_exps;
      if (down.type != GgmlType::kQ6_K)
        continue;
      // The original blocks come from the mapped file, not from the device.
      const auto* info = reader->FindTensor("blk." + std::to_string(l) +
                                            ".ffn_down_exps.weight");
      if (info == nullptr || info->type != GgmlType::kQ6_K ||
          info->size_bytes != rows * q6_row_bytes)
        throw std::runtime_error("unexpected ffn_down_exps tensor");
      const auto* source = static_cast<const std::uint8_t*>(info->data);
      ++layers;
      for (auto* host : {&converted, &repeated}) {
        HIP_CHECK(hipMemset(device, 0xA5, q8_bytes));
        rocm::RequantToQ8_0(down.data, static_cast<std::uint32_t>(down.type),
                            static_cast<std::uint32_t>(k), rows, device,
                            stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(
            hipMemcpy(host->data(), device, q8_bytes, hipMemcpyDeviceToHost));
      }
      LayerStats s;
      for (std::size_t i = 0; i < q8_bytes; ++i)
        s.repeat_mismatch_bytes += converted[i] != repeated[i];
      for (std::size_t r = 0; r < rows; ++r) {
        gufo::test_reference::DecodeRow(
            GgmlType::kQ6_K, source + (r * q6_row_bytes), k, row.data());
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
          const QuantCode* original = row.data() + (b * 32);
          const std::uint8_t* q8 =
              converted.data() + (((r * blocks_per_row) + b) * 34);
          std::uint8_t host[34];
          gufo::test_reference::RequantBlockQ8_0(original, host);
          s.host_scale_mismatch_blocks += std::memcmp(host, q8, 2) != 0;
          const double scale = gufo::test_reference::ReadHalf(q8);
          double block_max = 0.0;
          double block_error2 = 0.0;
          double block_value2 = 0.0;
          double block_error_max = 0.0;
          for (int i = 0; i < 32; ++i) {
            const double value = original[i].Value();
            const double stored = scale * static_cast<std::int8_t>(q8[2 + i]);
            const double err = std::fabs(stored - value);
            s.host_code_mismatch_weights += host[2 + i] != q8[2 + i];
            s.bound_violations += !(err <= BlockBound(scale));
            s.changed_weights += err != 0.0;
            block_max = std::max(block_max, std::fabs(value));
            block_error_max = std::max(block_error_max, err);
            block_error2 += err * err;
            block_value2 += value * value;
            // The step between adjacent Q6_K codes of this weight's own
            // 16-element sub-block.
            const double step = std::fabs(original[i].scale);
            if (step > 0.0) {
              ++s.q6_step_weights;
              const double ratio = err / step;
              s.max_error_over_q6_step =
                  std::max(s.max_error_over_q6_step, ratio);
              s.sum_error_over_q6_step2 += ratio * ratio;
              s.beyond_half_q6_step += ratio > 0.5;
            }
          }
          ++s.blocks;
          s.weights += 32;
          s.sum_error2 += block_error2;
          s.sum_value2 += block_value2;
          if (block_max > 0.0) {
            s.max_error_over_block_max = std::max(s.max_error_over_block_max,
                                                  block_error_max / block_max);
            s.sum_error_over_block_max += block_error_max / block_max;
            s.max_block_rms_ratio = std::max(
                s.max_block_rms_ratio, std::sqrt(block_error2 / block_value2));
          }
        }
      }
      const bool ok = s.repeat_mismatch_bytes == 0 && s.bound_violations == 0;
      passed = passed && ok;
      const auto weights_count = static_cast<double>(s.weights);
      std::cout << "layer=" << l << ",blocks=" << s.blocks
                << ",repeat_mismatch_bytes=" << s.repeat_mismatch_bytes
                << ",beyond_half_q8_step=" << s.bound_violations << ','
                << (ok ? "PASS" : "FAIL") << '\n'
                << "  device against IEEE host conversion: scale differs in "
                << s.host_scale_mismatch_blocks << " blocks, code differs in "
                << s.host_code_mismatch_weights << " of " << s.weights
                << " weights\n"
                << "  weights changed by the conversion: "
                << (100.0 * static_cast<double>(s.changed_weights) /
                    weights_count)
                << " %\n"
                << "  error / block max|w|: max " << s.max_error_over_block_max
                << ", mean of block maxima "
                << (s.sum_error_over_block_max / static_cast<double>(s.blocks))
                << " (format bound 1/254 = " << (1.0 / 254.0) << ")\n"
                << "  RMS error / RMS weight: whole tensor "
                << std::sqrt(s.sum_error2 / s.sum_value2) << ", worst block "
                << s.max_block_rms_ratio << '\n'
                << "  error / own Q6_K step: max " << s.max_error_over_q6_step
                << ", RMS "
                << std::sqrt(s.sum_error_over_q6_step2 /
                             static_cast<double>(s.q6_step_weights))
                << " (Q6_K rounding itself: at most 0.5, RMS about 0.289); "
                << "beyond half a step "
                << (100.0 * static_cast<double>(s.beyond_half_q6_step) /
                    static_cast<double>(s.q6_step_weights))
                << " %\n";
    }
    HIP_CHECK(hipFree(device));
    HIP_CHECK(hipStreamDestroy(stream));
    if (layers == 0) {
      std::cerr << "no Q6_K expert down weights; skipping\n";
      return 77;
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_moe_requant_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "HIP disabled, skipping Qwen MoE requant test.\n";
  return 77;
#endif
}
