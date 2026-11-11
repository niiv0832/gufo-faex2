#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kSharedKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::size_t kWidth = static_cast<std::size_t>(kHeads) * kHeadDim;

// Deterministic values in [-1, 1) with a wide spread of magnitudes, so the
// running maximum of the online softmax moves throughout a partition.
struct Generator {
  std::uint64_t state;
  float Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto bits = static_cast<std::uint32_t>(state >> 40);
    return (static_cast<float>(bits) / 8388608.0F) - 1.0F;
  }
};

template<typename T>
T* DeviceAlloc(std::size_t count) {
  T* pointer = nullptr;
  HIP_CHECK(hipMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

// One layer of token-major FP16 KV for `kv_heads` heads. Query head h of the
// shared geometry reads KV head h / 8; the unshared geometry stores that same
// head once per query head, which routes each query head through the
// independent one-head-per-workgroup kernel on identical operands.
std::vector<__half> BuildCache(std::uint32_t kv_heads, std::uint32_t context,
                               std::uint64_t seed) {
  std::vector<__half> shared(static_cast<std::size_t>(context) *
                             kSharedKvHeads * kHeadDim);
  Generator generator{seed};
  for (auto& value : shared) {
    const float magnitude = generator.Next();
    value = __float2half_rn(magnitude * magnitude * generator.Next() * 2.0F);
  }
  if (kv_heads == kSharedKvHeads) {
    return shared;
  }
  const std::uint32_t group = kHeads / kSharedKvHeads;
  std::vector<__half> expanded(static_cast<std::size_t>(context) * kv_heads *
                               kHeadDim);
  for (std::size_t token = 0; token < context; ++token) {
    for (std::uint32_t head = 0; head < kv_heads; ++head) {
      std::memcpy(&expanded[(token * kv_heads + head) * kHeadDim],
                  &shared[(token * kSharedKvHeads + head / group) * kHeadDim],
                  kHeadDim * sizeof(__half));
    }
  }
  return expanded;
}

struct Device {
  std::uint32_t kv_heads;
  std::uint32_t context;
  __half* keys;
  __half* values;
};

Device Upload(std::uint32_t kv_heads, std::uint32_t context) {
  const auto keys = BuildCache(kv_heads, context, 0x1234);
  const auto values = BuildCache(kv_heads, context, 0x9876);
  Device device{kv_heads, context, DeviceAlloc<__half>(keys.size()),
                DeviceAlloc<__half>(values.size())};
  HIP_CHECK(hipMemcpy(device.keys, keys.data(), keys.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device.values, values.data(),
                      values.size() * sizeof(__half), hipMemcpyHostToDevice));
  return device;
}

std::vector<float> Run(const Device& device, const float* query,
                       const float* gate, float* output, float* scratch,
                       std::size_t scratch_elements, std::uint32_t start,
                       std::size_t rows) {
  gufo::hip::LaunchCausalDecodeAttention(
      query, gate, nullptr, nullptr, device.keys, device.values, output, 0,
      start, rows, device.context, kHeads, device.kv_heads, kHeadDim, nullptr,
      std::span<float>(scratch, scratch_elements));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> host(rows * kWidth);
  HIP_CHECK(hipMemcpy(host.data(), output, host.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host;
}

// Sixteen query heads over two KV heads must reproduce, bit for bit, the
// kernel that gives every query head its own workgroup and cache stream.
void TestSharedGroupMatchesIndependentHeads() {
  constexpr std::uint32_t max_start = 33001;
  constexpr std::size_t max_rows = 3;
  constexpr std::uint32_t context = max_start + max_rows + 8;
  constexpr std::uint32_t starts[] = {127,  510,  511,   512,   513,
                                      1000, 4097, 16383, 20011, max_start};

  const Device shared = Upload(kSharedKvHeads, context);
  const Device independent = Upload(kHeads, context);

  std::vector<float> host_query(max_rows * kWidth),
      host_gate(max_rows * kWidth);
  Generator generator{0x5555};
  for (auto& value : host_query) {
    value = generator.Next() * 3.0F;
  }
  for (auto& value : host_gate) {
    value = generator.Next() * 2.0F;
  }
  auto* query = DeviceAlloc<float>(host_query.size());
  auto* gate = DeviceAlloc<float>(host_gate.size());
  auto* output = DeviceAlloc<float>(max_rows * kWidth);
  const std::size_t scratch_elements =
      max_rows *
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  auto* scratch = DeviceAlloc<float>(scratch_elements);
  HIP_CHECK(hipMemcpy(query, host_query.data(),
                      host_query.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(gate, host_gate.data(), host_gate.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  for (const std::uint32_t start : starts) {
    for (std::size_t rows = 1; rows <= max_rows; ++rows) {
      const auto candidate = Run(shared, query, gate, output, scratch,
                                 scratch_elements, start, rows);
      const auto reference = Run(independent, query, gate, output, scratch,
                                 scratch_elements, start, rows);
      bool nonzero = false;
      for (const float value : candidate) {
        nonzero = nonzero || value != 0.0F;
      }
      if (!nonzero || std::memcmp(candidate.data(), reference.data(),
                                  candidate.size() * sizeof(float)) != 0) {
        std::size_t first = 0;
        while (first < candidate.size() &&
               std::memcmp(&candidate[first], &reference[first],
                           sizeof(float)) == 0) {
          ++first;
        }
        std::cerr << "Shared eight-head attention differs at start " << start
                  << " rows " << rows << " element " << first << "\n";
        std::abort();
      }
    }
  }
  std::cout << "Sixteen-over-two decode attention is bit-identical to "
               "independent heads for "
            << std::size(starts) << " positions x " << max_rows << " rows\n";
  for (void* pointer :
       {static_cast<void*>(shared.keys), static_cast<void*>(shared.values),
        static_cast<void*>(independent.keys),
        static_cast<void*>(independent.values), static_cast<void*>(query),
        static_cast<void*>(gate), static_cast<void*>(output),
        static_cast<void*>(scratch)}) {
    HIP_CHECK(hipFree(pointer));
  }
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen 16-over-2 decode attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  TestSharedGroupMatchesIndependentHeads();
  std::cout << "Qwen 16-over-2 decode attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen 16-over-2 decode attention ops "
               "test.\n";
  return 77;
#endif
}
