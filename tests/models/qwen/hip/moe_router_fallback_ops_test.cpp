// Router and shared-gate logits of the grouped-expert prefill without the
// split BF16 GEMM (MoePrefillRouterLogitsFallback) against a host reference,
// for every combination of F32 and BF16 tensors. A BF16 tensor must be read as
// BF16, and a type the path has no reader for must be refused before a launch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/moe_ffn.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

using gufo::core::GgmlType;

constexpr std::size_t kExperts = 12;
constexpr std::size_t kHidden = 256;
// More rows than the smallest prompt chunk the grouped prefill accepts.
constexpr std::size_t kRows = 19;

struct Generator {
  std::uint64_t state;
  float Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (static_cast<float>(state >> 40) / 8388608.0F) - 1.0F;
  }
};

std::vector<float> Fill(std::size_t count, Generator& generator) {
  std::vector<float> values(count);
  for (auto& value : values)
    value = generator.Next();
  return values;
}

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float FromBf16(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// One weight tensor in the storage type under test and the values a reader
// of that type must see.
struct Tensor {
  std::vector<float> values;
  void* device{nullptr};
  gufo::models::QwenTensorRef ref;

  Tensor(std::vector<float> source, GgmlType type) : values(std::move(source)) {
    std::vector<std::uint16_t> packed;
    const void* host = values.data();
    std::size_t bytes = values.size() * sizeof(float);
    if (type == GgmlType::kBF16) {
      packed.resize(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        packed[i] = ToBf16(values[i]);
        values[i] = FromBf16(packed[i]);
      }
      host = packed.data();
      bytes = packed.size() * sizeof(std::uint16_t);
    }
    HIP_CHECK(hipMalloc(&device, bytes));
    HIP_CHECK(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
    ref.data = device;
    ref.type = type;
    ref.num_elements = values.size();
    ref.available_bytes = bytes;
  }
  ~Tensor() { (void)hipFree(device); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
};

const char* Name(GgmlType type) {
  return type == GgmlType::kBF16 ? "BF16" : "F32";
}

bool TestTypes(hipblasHandle_t handle, GgmlType router_type,
               GgmlType shared_type) {
  Generator generator{0x9E3779B97F4A7C15ULL};
  const Tensor router(Fill(kExperts * kHidden, generator), router_type);
  const Tensor shared(Fill(kHidden, generator), shared_type);
  const auto x = Fill(kRows * kHidden, generator);
  float* device_x = nullptr;
  float* router_logits = nullptr;
  float* shared_logit = nullptr;
  void* scratch = nullptr;
  HIP_CHECK(hipMalloc(&device_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&router_logits, kRows * kExperts * sizeof(float)));
  HIP_CHECK(hipMalloc(&shared_logit, kRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&scratch, x.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(device_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  gufo::hip::MoePrefillRouterLogitsFallback(
      handle, router.ref, shared.ref, device_x, router_logits, shared_logit,
      kRows, kExperts, kHidden, scratch, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got_router(kRows * kExperts);
  std::vector<float> got_shared(kRows);
  HIP_CHECK(hipMemcpy(got_router.data(), router_logits,
                      got_router.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_shared.data(), shared_logit,
                      got_shared.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (void* const buffer :
       {static_cast<void*>(device_x), static_cast<void*>(router_logits),
        static_cast<void*>(shared_logit), scratch}) {
    HIP_CHECK(hipFree(buffer));
  }

  const auto dot = [&](const std::vector<float>& weights, std::size_t weight,
                       std::size_t row) {
    double sum = 0;
    for (std::size_t k = 0; k < kHidden; ++k) {
      sum += static_cast<double>(weights[(weight * kHidden) + k]) *
             static_cast<double>(x[(row * kHidden) + k]);
    }
    return sum;
  };
  // 256 products of magnitude below one summed in fp32.
  constexpr double kTolerance = 1e-4;
  double max_router = 0;
  double max_shared = 0;
  bool finite = true;
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t expert = 0; expert < kExperts; ++expert) {
      const float value = got_router[(row * kExperts) + expert];
      finite = finite && std::isfinite(value);
      max_router = std::max(max_router,
                            std::abs(value - dot(router.values, expert, row)));
    }
    finite = finite && std::isfinite(got_shared[row]);
    max_shared = std::max(
        max_shared, std::abs(got_shared[row] - dot(shared.values, 0, row)));
  }
  const bool ok =
      finite && max_router <= kTolerance && max_shared <= kTolerance;
  std::cout << "router=" << Name(router_type) << ",shared=" << Name(shared_type)
            << ",max_router_error=" << max_router
            << ",max_shared_error=" << max_shared << ','
            << (ok ? "PASS" : "FAIL") << '\n';
  return ok;
}

// A packed type has no reader here: the call must fail before any launch.
bool TestUnsupported(hipblasHandle_t handle) {
  Generator generator{7};
  const Tensor router(Fill(kExperts * kHidden, generator), GgmlType::kF32);
  const Tensor shared(Fill(kHidden, generator), GgmlType::kF32);
  bool ok = true;
  for (const bool router_packed : {true, false}) {
    auto router_ref = router.ref;
    auto shared_ref = shared.ref;
    (router_packed ? router_ref : shared_ref).type = GgmlType::kQ8_0;
    float* router_logits = nullptr;
    float* shared_logit = nullptr;
    float* device_x = nullptr;
    HIP_CHECK(hipMalloc(&device_x, kRows * kHidden * sizeof(float)));
    HIP_CHECK(hipMalloc(&router_logits, kRows * kExperts * sizeof(float)));
    HIP_CHECK(hipMalloc(&shared_logit, kRows * sizeof(float)));
    HIP_CHECK(hipMemset(device_x, 0, kRows * kHidden * sizeof(float)));
    bool refused = false;
    try {
      gufo::hip::MoePrefillRouterLogitsFallback(
          handle, router_ref, shared_ref, device_x, router_logits, shared_logit,
          kRows, kExperts, kHidden, nullptr, nullptr);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    HIP_CHECK(hipDeviceSynchronize());
    for (float* const buffer : {device_x, router_logits, shared_logit})
      HIP_CHECK(hipFree(buffer));
    std::cout << (router_packed ? "router" : "shared") << "=Q8_0,"
              << (refused ? "refused,PASS" : "accepted,FAIL") << '\n';
    ok = ok && refused;
  }
  return ok;
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen MoE router fallback ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  hipblasHandle_t handle = nullptr;
  HIPBLAS_CHECK(hipblasCreate(&handle));
  bool passed = true;
  for (const auto router : {GgmlType::kF32, GgmlType::kBF16}) {
    for (const auto shared : {GgmlType::kF32, GgmlType::kBF16}) {
      passed = TestTypes(handle, router, shared) && passed;
    }
  }
  passed = TestUnsupported(handle) && passed;
  HIPBLAS_CHECK(hipblasDestroy(handle));
  if (!passed) {
    std::cerr << "Qwen MoE router fallback ops test failed.\n";
    return 1;
  }
  std::cout << "Qwen MoE router fallback ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen MoE router fallback ops test.\n";
  return 77;
#endif
}
