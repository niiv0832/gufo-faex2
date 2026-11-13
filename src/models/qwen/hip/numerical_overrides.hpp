#ifndef GUFO_MODELS_QWEN_HIP_NUMERICAL_OVERRIDES_HPP_
#define GUFO_MODELS_QWEN_HIP_NUMERICAL_OVERRIDES_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip {

/// Opt-in environment overrides that change the arithmetic of a Qwen session:
/// block-sparse attention (GUFO_QWEN_SPARSE_ATTN and its parameters) and the
/// grouped-expert prefill switch (GUFO_QWEN_MOE_PREFILL). The arena resolves
/// them once and every consumer reads the arena's copy, so the state a session
/// computes and the identity it is stored under cannot disagree.
struct QwenNumericalOverrides {
  bool sparse_attention{false};
  std::uint32_t sparse_sink_blocks{2};
  std::uint32_t sparse_local_blocks{32};
  std::uint32_t sparse_top_blocks{256};
  std::uint32_t sparse_min_context{8192};
  bool grouped_moe_prefill{true};

  [[nodiscard]] bool operator==(const QwenNumericalOverrides&) const = default;

  /// Lines for a continuation-cache compatibility identity. The defaults add
  /// nothing, so states stored without an override stay addressable; the
  /// parameters of a disabled sparse attention do not participate.
  [[nodiscard]] std::string CacheIdentity() const {
    std::string identity;
    if (sparse_attention) {
      identity +=
          "sparse_attention=block-v1\nsparse_sink_blocks=" +
          std::to_string(sparse_sink_blocks) +
          "\nsparse_local_blocks=" + std::to_string(sparse_local_blocks) +
          "\nsparse_top_blocks=" + std::to_string(sparse_top_blocks) +
          "\nsparse_min_context=" + std::to_string(sparse_min_context) + '\n';
    }
    if (!grouped_moe_prefill) {
      identity += "moe_prefill=row-wise-v1\n";
    }
    return identity;
  }
};

/// Value of an environment variable, or null when it is unset.
using QwenEnvironmentLookup = std::function<const char*(const char*)>;

/// Sparse attention needs a supported head geometry, the FP16 KV cache and a
/// context its block lists can index; a request outside that stays dense.
[[nodiscard]] inline QwenNumericalOverrides ResolveQwenNumericalOverrides(
    const core::ModelConfig& config, const QwenExecutionPolicy& policy,
    std::uint32_t max_context, const QwenEnvironmentLookup& lookup) {
  QwenNumericalOverrides overrides;
  const char* const grouped = lookup("GUFO_QWEN_MOE_PREFILL");
  overrides.grouped_moe_prefill = grouped == nullptr || grouped[0] != '0';
  const char* const sparse = lookup("GUFO_QWEN_SPARSE_ATTN");
  const bool supported_heads =
      (config.num_attention_heads == 24 && config.num_key_value_heads == 4) ||
      (config.num_attention_heads == 16 && config.num_key_value_heads == 2);
  if (sparse == nullptr || sparse[0] == '0' || !supported_heads ||
      !policy.UsesFp16AttentionKv() || max_context > 262144) {
    return overrides;
  }
  const auto read = [&lookup](const char* name, std::uint32_t fallback) {
    const char* const value = lookup(name);
    return value != nullptr
               ? static_cast<std::uint32_t>(std::max(0, std::atoi(value)))
               : fallback;
  };
  overrides.sparse_attention = true;
  overrides.sparse_sink_blocks =
      read("GUFO_QWEN_SPARSE_SINK", overrides.sparse_sink_blocks);
  overrides.sparse_local_blocks =
      read("GUFO_QWEN_SPARSE_LOCAL", overrides.sparse_local_blocks);
  overrides.sparse_top_blocks =
      read("GUFO_QWEN_SPARSE_TOPK", overrides.sparse_top_blocks);
  overrides.sparse_min_context =
      read("GUFO_QWEN_SPARSE_MIN_CONTEXT", overrides.sparse_min_context);
  return overrides;
}

/// The overrides of this process's environment.
[[nodiscard]] inline QwenNumericalOverrides ResolveQwenNumericalOverrides(
    const core::ModelConfig& config, const QwenExecutionPolicy& policy,
    std::uint32_t max_context) {
  return ResolveQwenNumericalOverrides(
      config, policy, max_context,
      [](const char* name) -> const char* { return std::getenv(name); });
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_NUMERICAL_OVERRIDES_HPP_
