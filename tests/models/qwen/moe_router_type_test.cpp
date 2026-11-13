// The MoE router and shared-expert gate are read by kernels that decode F32
// and BF16 rows only. The loader must refuse every other storage type: a
// quantized router accepted as an ordinary projection would be read as raw
// floats by single-token decode, short prefill and the grouped fallback.

#include <array>
#include <cstdlib>
#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"

int main() {
  using gufo::core::GgmlType;
  using gufo::models::SupportsMoeRouterType;
  bool passed = true;
  const auto expect = [&passed](bool condition, const char* message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      passed = false;
    }
  };
  expect(SupportsMoeRouterType(GgmlType::kF32), "an F32 router is supported");
  expect(SupportsMoeRouterType(GgmlType::kBF16), "a BF16 router is supported");
  for (const GgmlType type : std::array{
           GgmlType::kF16, GgmlType::kQ8_0, GgmlType::kQ8_K, GgmlType::kQ6_K,
           GgmlType::kQ5_K, GgmlType::kQ4_K, GgmlType::kQ3_K, GgmlType::kIQ4_NL,
           GgmlType::kIQ4_XS, GgmlType::kIQ3_S}) {
    if (SupportsMoeRouterType(type)) {
      std::cerr << "FAILED: router type " << gufo::core::ToString(type)
                << " has no router decoder and must be refused\n";
      passed = false;
    }
  }
  if (passed)
    std::cout << "Qwen MoE router type checks passed.\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
