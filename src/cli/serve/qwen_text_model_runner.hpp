#ifndef GUFO_SERVER_QWEN_TEXT_MODEL_RUNNER_HPP_
#define GUFO_SERVER_QWEN_TEXT_MODEL_RUNNER_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <cstdint>
#include <memory>
#include <string>

#include "src/core/speculative/speculative_verifier.hpp"

namespace gufo::hip {
class QwenDFlashGpuModel;
class QwenGpuModel;
class QwenMtpGpuModel;
}  // namespace gufo::hip

namespace gufo::server {
class TextModelRunner;

// Internal serving factory: the loader and snapshot contract tests construct
// the same runner from already loaded models. This does not load any weights.
struct QwenTextRunnerOptions {
  std::uint32_t max_context{0};
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash;
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp;
  speculative::SpeculativeOptions speculative_options;
  std::string model_artifact_fingerprint;
  std::string draft_model_artifact_fingerprint;
};

[[nodiscard]] std::shared_ptr<TextModelRunner> CreateQwenTextRunner(
    std::shared_ptr<const hip::QwenGpuModel> model,
    QwenTextRunnerOptions options);

}  // namespace gufo::server
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_SERVER_QWEN_TEXT_MODEL_RUNNER_HPP_
