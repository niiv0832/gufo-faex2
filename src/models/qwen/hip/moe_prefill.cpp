#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_prefill.hpp"

#include <utility>

namespace gufo::hip {
namespace {
QwenMoePrefillFactory& Factory() {
  static QwenMoePrefillFactory factory;
  return factory;
}

QwenMoePrefillSizing& Sizing() {
  static QwenMoePrefillSizing sizing;
  return sizing;
}
}  // namespace

void SetQwenMoePrefillFactory(QwenMoePrefillFactory factory,
                              QwenMoePrefillSizing sizing) {
  Factory() = std::move(factory);
  Sizing() = std::move(sizing);
}

std::size_t EstimateQwenMoePrefillResidentBytes(
    const models::QwenModelWeights& weights) {
  const auto& sizing = Sizing();
  return sizing.resident_bytes ? sizing.resident_bytes(weights) : 0;
}

std::size_t EstimateQwenMoePrefillStreamScratchBytes(
    const core::ModelConfig& config, std::uint32_t max_tokens) {
  const auto& sizing = Sizing();
  return sizing.stream_scratch_bytes
             ? sizing.stream_scratch_bytes(config, max_tokens)
             : 0;
}

std::unique_ptr<QwenMoePrefill> CreateQwenMoePrefill(
    const models::QwenModelWeights& weights, std::uint32_t max_tokens,
    std::string* error) {
  const auto& factory = Factory();
  return factory ? factory(weights, max_tokens, error) : nullptr;
}

}  // namespace gufo::hip
#endif
