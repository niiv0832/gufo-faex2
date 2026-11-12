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
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/moe_ffn.hpp"
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

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_moe_rows_test [MODEL.gguf]\n";
    return 2;
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
           {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{8},
            std::size_t{9}, std::size_t{26}, std::size_t{63}, kMaxRows}) {
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
