// The fused end of a MoE layer against the launches it replaces, bit for bit:
// LaunchMoeCombineResidualAddRMSNorm against LaunchMoeCombine, a residual add
// and an RMSNorm; and the row form of LaunchResidualAddRMSNorm against the
// batched residual add and RMSNorm.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kUsed = 8;
constexpr std::size_t kMaxRows = 9;

struct Generator {
  std::uint64_t state;
  float Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (static_cast<float>(state >> 40) / 8388608.0F) - 1.0F;
  }
};

float* Upload(const std::vector<float>& host) {
  float* device = nullptr;
  HIP_CHECK(hipMalloc(&device, host.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  return device;
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> host(count);
  HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> Fill(std::size_t count, Generator& generator, float scale) {
  std::vector<float> values(count);
  for (auto& value : values) {
    const float u = generator.Next();
    value = u * u * u * scale;
  }
  return values;
}

void Expect(bool condition, const char* message, std::size_t hidden,
            std::size_t rows) {
  if (!condition) {
    std::cerr << message << " (hidden " << hidden << ", rows " << rows << ")\n";
    std::exit(1);
  }
}

void TestHidden(std::size_t hidden) {
  Generator generator{0xA0761D6478BD642FULL + hidden};
  const auto host_hidden = Fill(kMaxRows * hidden, generator, 6.0F);
  const auto host_slots = Fill(kMaxRows * kUsed * hidden, generator, 3.0F);
  const auto host_shared = Fill(kMaxRows * hidden, generator, 3.0F);
  const auto host_attention = Fill(kMaxRows * hidden, generator, 4.0F);
  auto host_weights = Fill(kMaxRows * kUsed, generator, 1.0F);
  for (auto& weight : host_weights)
    weight = weight * weight + 0.01F;
  auto host_shared_weight = Fill(kMaxRows, generator, 1.0F);
  auto host_norm = Fill(hidden, generator, 2.0F);
  for (auto& weight : host_norm)
    weight += 1.0F;

  float* const slots = Upload(host_slots);
  float* const shared = Upload(host_shared);
  float* const attention = Upload(host_attention);
  float* const weights = Upload(host_weights);
  float* const shared_weight = Upload(host_shared_weight);
  float* const norm = Upload(host_norm);
  float* const hidden_a = Upload(host_hidden);
  float* const hidden_b = Upload(host_hidden);
  float* const combined = Upload(host_hidden);
  float* const normed_a = Upload(host_hidden);
  float* const normed_b = Upload(host_hidden);

  for (const std::size_t rows :
       {std::size_t{1}, std::size_t{2}, std::size_t{5}, kMaxRows}) {
    const std::size_t count = rows * hidden;
    // 1. The end of a MoE layer.
    for (float* const buffer : {hidden_a, hidden_b}) {
      HIP_CHECK(hipMemcpy(buffer, host_hidden.data(), count * sizeof(float),
                          hipMemcpyHostToDevice));
    }
    gufo::hip::LaunchMoeCombine(combined, slots, weights, shared, shared_weight,
                                kUsed, static_cast<std::uint32_t>(hidden),
                                static_cast<std::uint32_t>(rows));
    gufo::hip::LaunchBatchedResidualAdd(hidden_a, combined, hidden_a, rows,
                                        hidden);
    gufo::hip::LaunchBatchedRMSNorm(hidden_a, norm, normed_a, nullptr, rows,
                                    hidden);
    gufo::hip::LaunchMoeCombineResidualAddRMSNorm(
        hidden_b, slots, weights, shared, shared_weight, norm, normed_b, kUsed,
        static_cast<std::uint32_t>(hidden), 1e-6F,
        static_cast<std::uint32_t>(rows));
    HIP_CHECK(hipDeviceSynchronize());
    Expect(Same(Download(hidden_a, count), Download(hidden_b, count)),
           "fused MoE tail changes the hidden state", hidden, rows);
    Expect(Same(Download(normed_a, count), Download(normed_b, count)),
           "fused MoE tail changes the normalized rows", hidden, rows);

    // One row against the single-row launches the decode step replaces.
    if (rows == 1) {
      HIP_CHECK(hipMemcpy(hidden_a, host_hidden.data(), count * sizeof(float),
                          hipMemcpyHostToDevice));
      gufo::hip::LaunchResidualAdd(hidden_a, combined, hidden_a, hidden);
      gufo::hip::LaunchRMSNorm(hidden_a, norm, normed_a, hidden);
      HIP_CHECK(hipDeviceSynchronize());
      Expect(Same(Download(hidden_a, count), Download(hidden_b, count)) &&
                 Same(Download(normed_a, count), Download(normed_b, count)),
             "fused MoE tail differs from the single-row launches", hidden,
             rows);
    }

    // 2. Residual add and norm for several rows.
    for (float* const buffer : {hidden_a, hidden_b}) {
      HIP_CHECK(hipMemcpy(buffer, host_hidden.data(), count * sizeof(float),
                          hipMemcpyHostToDevice));
    }
    gufo::hip::LaunchBatchedResidualAdd(hidden_a, attention, hidden_a, rows,
                                        hidden);
    gufo::hip::LaunchBatchedRMSNorm(hidden_a, norm, normed_a, nullptr, rows,
                                    hidden);
    gufo::hip::LaunchResidualAddRMSNorm(hidden_b, attention, norm, normed_b,
                                        hidden, 1e-6F, nullptr, rows);
    HIP_CHECK(hipDeviceSynchronize());
    Expect(Same(Download(hidden_a, count), Download(hidden_b, count)) &&
               Same(Download(normed_a, count), Download(normed_b, count)),
           "row residual add and norm differ from the batched launches", hidden,
           rows);
    bool nonzero = false;
    for (const float value : Download(normed_b, count))
      nonzero = nonzero || value != 0.0F;
    Expect(nonzero, "normalized rows are all zero", hidden, rows);
  }
  for (float* const buffer :
       {slots, shared, attention, weights, shared_weight, norm, hidden_a,
        hidden_b, combined, normed_a, normed_b}) {
    HIP_CHECK(hipFree(buffer));
  }
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen MoE tail ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  // 2048 is Qwen3.6-35B-A3B; the others cover a ragged last stride.
  for (const std::size_t hidden :
       {std::size_t{2048}, std::size_t{1536}, std::size_t{1000}}) {
    TestHidden(hidden);
  }
  std::cout << "Qwen MoE tail ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen MoE tail ops test.\n";
  return 77;
#endif
}
