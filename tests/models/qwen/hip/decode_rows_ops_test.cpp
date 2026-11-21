// Row operations of batched decode and speculative verification on real
// weights: each must give every row, bit for bit, the result the single-token
// decode operation gives that row alone. Sampled speculation serves the
// verification rows as its target distribution, and attention keys, values
// and recurrent state are stored rounded, so even a last-bit difference in
// one projection grows into a different distribution a few layers later.
//
// qwen_decode_rows_ops_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it.

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
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::models::QwenTensorRef;

constexpr std::size_t kRows = 8;

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

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t elements) {
    HIP_CHECK(hipMalloc(&data_, elements * sizeof(float)));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { (void)hipFree(data_); }
  [[nodiscard]] float* data() const noexcept { return data_; }
  void Upload(const std::vector<float>& host) const {
    HIP_CHECK(hipMemcpy(data_, host.data(), host.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  }
  [[nodiscard]] std::vector<float> Download(std::size_t elements) const {
    std::vector<float> host(elements);
    HIP_CHECK(hipMemcpy(host.data(), data_, elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    return host;
  }

private:
  float* data_ = nullptr;
};

struct Checker {
  hipStream_t stream;
  bool passed = true;
  std::size_t checks = 0;

  void Report(const std::string& label, const QwenTensorRef* weight,
              const DeviceBuffer& single, const DeviceBuffer& batched,
              std::size_t rows, std::size_t width) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto reference = single.Download(rows * width);
    const auto candidate = batched.Download(rows * width);
    std::size_t differing = 0;
    bool nonzero = false;
    for (std::size_t index = 0; index < reference.size(); ++index) {
      differing +=
          std::memcmp(&reference[index], &candidate[index], sizeof(float)) != 0;
      nonzero = nonzero || reference[index] != 0.0F;
    }
    const bool ok = nonzero && differing == 0;
    ++checks;
    std::cout << label;
    if (weight != nullptr)
      std::cout << ",type=" << gufo::core::ToString(weight->type);
    std::cout << ",rows=" << rows << ",differing=" << differing << ','
              << (ok ? "PASS" : "FAIL") << '\n';
    passed = passed && ok;
  }

  // One weight: single-token GEMV per row against the rows projection.
  void Projection(const std::string& label, const QwenTensorRef& weight,
                  std::size_t outputs, std::size_t inputs, std::uint64_t seed) {
    if (weight.empty())
      throw std::runtime_error(label + ": the weight is missing");
    const DeviceBuffer x(kRows * inputs);
    const DeviceBuffer single(kRows * outputs);
    const DeviceBuffer batched(kRows * outputs);
    x.Upload(Rows(kRows, inputs, seed));
    for (const std::size_t rows : {std::size_t{1}, std::size_t{2}, kRows}) {
      for (std::size_t row = 0; row < rows; ++row)
        gufo::hip::LaunchGEMV(
            weight.data, weight.type, x.data() + (row * inputs),
            single.data() + (row * outputs), outputs, inputs, stream);
      gufo::hip::LaunchDecodeRowsProjection(weight, x.data(), batched.data(),
                                            rows, outputs, inputs, stream);
      Report(label, &weight, single, batched, rows, outputs);
    }
  }
};

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_decode_rows_ops_test [MODEL.gguf]\n";
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
    gufo::hip::QwenGpuArena arena(config, 4096);
    Checker check{arena.stream};
    const std::size_t hidden = config.hidden_size;
    const std::size_t attention = config.AttentionSize();
    const std::size_t kv =
        static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
    const std::size_t qkv = config.SsmQkvSize();
    const std::size_t inner = config.ssm_inner_size;
    const std::size_t rank = config.ssm_time_step_rank;

    const DeviceBuffer x(kRows * hidden);
    const DeviceBuffer residual(kRows * hidden);
    x.Upload(Rows(kRows, hidden, 0x9E3779B97F4A7C15ULL));
    residual.Upload(Rows(kRows, hidden, 0xC2B2AE3D27D4EB4FULL));

    bool seen_attention = false;
    bool seen_recurrent = false;
    for (std::size_t index = 0; index < weights.layers.size(); ++index) {
      const auto& layer = weights.layers[index];
      // The first and the last layer of each kind.
      const bool last =
          index + config.full_attention_interval >= weights.layers.size();
      bool& seen = layer.is_full_attention ? seen_attention : seen_recurrent;
      if (seen && !last)
        continue;
      seen = true;
      const std::string name = "layer=" + std::to_string(index);
      const std::uint64_t seed = 0xD1B54A32D192ED03ULL + index;

      {
        const DeviceBuffer single(kRows * hidden);
        const DeviceBuffer batched(kRows * hidden);
        const auto* norm = static_cast<const float*>(layer.attn_norm.data);
        for (std::size_t row = 0; row < kRows; ++row)
          gufo::hip::LaunchRMSNorm(x.data() + (row * hidden), norm,
                                   single.data() + (row * hidden), hidden,
                                   1e-6F, arena.stream);
        gufo::hip::LaunchBatchedRMSNorm(x.data(), norm, batched.data(), nullptr,
                                        kRows, hidden, 1e-6F, arena.stream);
        check.Report(name + ",attn_norm", nullptr, single, batched, kRows,
                     hidden);
      }
      if (hidden != 5120) {
        // The fused residual add and FFN pre-norm, one row against many.
        const auto* norm = static_cast<const float*>(layer.ffn_norm.data);
        const DeviceBuffer sum_single(kRows * hidden);
        const DeviceBuffer sum_batched(kRows * hidden);
        const DeviceBuffer single(kRows * hidden);
        const DeviceBuffer batched(kRows * hidden);
        // Copies ordered on the stream of the kernels that update them.
        HIP_CHECK(hipMemcpyAsync(sum_single.data(), x.data(),
                                 kRows * hidden * sizeof(float),
                                 hipMemcpyDeviceToDevice, arena.stream));
        HIP_CHECK(hipMemcpyAsync(sum_batched.data(), x.data(),
                                 kRows * hidden * sizeof(float),
                                 hipMemcpyDeviceToDevice, arena.stream));
        for (std::size_t row = 0; row < kRows; ++row)
          gufo::hip::LaunchResidualAddRMSNorm(
              sum_single.data() + (row * hidden),
              residual.data() + (row * hidden), norm,
              single.data() + (row * hidden), hidden, 1e-6F, arena.stream);
        gufo::hip::LaunchResidualAddRMSNorm(sum_batched.data(), residual.data(),
                                            norm, batched.data(), hidden, 1e-6F,
                                            arena.stream, kRows);
        check.Report(name + ",residual_ffn_norm", nullptr, single, batched,
                     kRows, hidden);
        check.Report(name + ",residual_sum", nullptr, sum_single, sum_batched,
                     kRows, hidden);
      }

      if (layer.is_full_attention) {
        // Single-token decode projects Q, K and V in one fused kernel.
        const DeviceBuffer q(kRows * 2 * attention);
        const DeviceBuffer k(kRows * kv);
        const DeviceBuffer v(kRows * kv);
        const DeviceBuffer batched_q(kRows * 2 * attention);
        const DeviceBuffer batched_k(kRows * kv);
        const DeviceBuffer batched_v(kRows * kv);
        for (std::size_t row = 0; row < kRows; ++row)
          gufo::hip::LaunchFusedQKVProjections(
              layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
              layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
              x.data() + (row * hidden), q.data() + (row * 2 * attention),
              k.data() + (row * kv), v.data() + (row * kv), 2 * attention, kv,
              hidden, arena.stream);
        gufo::hip::LaunchDecodeRowsProjection(
            layer.attn_q, x.data(), batched_q.data(), kRows, 2 * attention,
            hidden, arena.stream);
        gufo::hip::LaunchDecodeRowsProjection(layer.attn_k, x.data(),
                                              batched_k.data(), kRows, kv,
                                              hidden, arena.stream);
        gufo::hip::LaunchDecodeRowsProjection(layer.attn_v, x.data(),
                                              batched_v.data(), kRows, kv,
                                              hidden, arena.stream);
        check.Report(name + ",attn_q", &layer.attn_q, q, batched_q, kRows,
                     2 * attention);
        check.Report(name + ",attn_k", &layer.attn_k, k, batched_k, kRows, kv);
        check.Report(name + ",attn_v", &layer.attn_v, v, batched_v, kRows, kv);
        if (layer.attn_q.type == gufo::core::GgmlType::kQ8_0 &&
            layer.attn_k.type == gufo::core::GgmlType::kQ8_0 &&
            layer.attn_v.type == gufo::core::GgmlType::kQ8_0) {
          for (const std::size_t b :
               {std::size_t{2}, std::size_t{3}, std::size_t{4}, kRows}) {
            if (!gufo::hip::TryLaunchBatchedQ8_0FusedQKV(
                    layer.attn_q.data, layer.attn_k.data, layer.attn_v.data,
                    x.data(), batched_q.data(), batched_k.data(),
                    batched_v.data(), b, 2 * attention, kv, hidden,
                    arena.stream)) {
              throw std::runtime_error(name + ": fused batched QKV refused");
            }
            check.Report(name + ",fused_attn_q", &layer.attn_q, q, batched_q, b,
                         2 * attention);
            check.Report(name + ",fused_attn_k", &layer.attn_k, k, batched_k, b,
                         kv);
            check.Report(name + ",fused_attn_v", &layer.attn_v, v, batched_v, b,
                         kv);
          }
        }
        check.Projection(name + ",attn_output", layer.attn_output, hidden,
                         attention, seed);
      } else {
        // Single-token decode projects the four recurrent inputs in one fused
        // kernel.
        const DeviceBuffer s_qkv(kRows * qkv);
        const DeviceBuffer s_gate(kRows * inner);
        const DeviceBuffer s_alpha(kRows * rank);
        const DeviceBuffer s_beta(kRows * rank);
        const DeviceBuffer b_qkv(kRows * qkv);
        const DeviceBuffer b_gate(kRows * inner);
        const DeviceBuffer b_alpha(kRows * rank);
        const DeviceBuffer b_beta(kRows * rank);
        for (std::size_t row = 0; row < kRows; ++row)
          gufo::hip::LaunchFusedSSMInputProjections(
              layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
              layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
              layer.ssm_beta.data, layer.ssm_beta.type,
              x.data() + (row * hidden), s_qkv.data() + (row * qkv),
              s_gate.data() + (row * inner), s_alpha.data() + (row * rank),
              s_beta.data() + (row * rank), hidden, qkv, inner, rank,
              arena.stream);
        gufo::hip::LaunchDecodeRowsProjection(layer.attn_qkv, x.data(),
                                              b_qkv.data(), kRows, qkv, hidden,
                                              arena.stream);
        gufo::hip::LaunchDecodeRowsProjection(layer.attn_gate, x.data(),
                                              b_gate.data(), kRows, inner,
                                              hidden, arena.stream);
        gufo::hip::LaunchDecodeRowsSsmControls(
            layer.ssm_alpha, layer.ssm_beta, x.data(), b_alpha.data(),
            b_beta.data(), kRows, rank, hidden, arena.stream);
        check.Report(name + ",attn_qkv", &layer.attn_qkv, s_qkv, b_qkv, kRows,
                     qkv);
        check.Report(name + ",attn_gate", &layer.attn_gate, s_gate, b_gate,
                     kRows, inner);
        check.Report(name + ",ssm_alpha", &layer.ssm_alpha, s_alpha, b_alpha,
                     kRows, rank);
        check.Report(name + ",ssm_beta", &layer.ssm_beta, s_beta, b_beta, kRows,
                     rank);
        if (layer.attn_qkv.type == gufo::core::GgmlType::kQ8_0 &&
            layer.attn_gate.type == gufo::core::GgmlType::kQ8_0 &&
            layer.ssm_alpha.type == gufo::core::GgmlType::kF32 &&
            layer.ssm_beta.type == gufo::core::GgmlType::kF32) {
          for (const std::size_t b :
               {std::size_t{2}, std::size_t{3}, std::size_t{4}, kRows}) {
            if (!gufo::hip::TryLaunchBatchedQ8_0FusedSSMInput(
                    layer.attn_qkv.data, layer.attn_gate.data,
                    layer.ssm_alpha.data, layer.ssm_beta.data, x.data(),
                    b_qkv.data(), b_gate.data(), b_alpha.data(), b_beta.data(),
                    b, hidden, qkv, inner, rank, arena.stream)) {
              throw std::runtime_error(name + ": fused batched SSM refused");
            }
            check.Report(name + ",fused_attn_qkv", &layer.attn_qkv, s_qkv,
                         b_qkv, b, qkv);
            check.Report(name + ",fused_attn_gate", &layer.attn_gate, s_gate,
                         b_gate, b, inner);
            check.Report(name + ",fused_ssm_alpha", &layer.ssm_alpha, s_alpha,
                         b_alpha, b, rank);
            check.Report(name + ",fused_ssm_beta", &layer.ssm_beta, s_beta,
                         b_beta, b, rank);
          }
        }
        check.Projection(name + ",ssm_out", layer.ssm_out, hidden, inner, seed);
      }
      if (config.IsMoe() &&
          layer.ffn_gate_shexp.type == gufo::core::GgmlType::kQ8_0 &&
          layer.ffn_up_shexp.type == gufo::core::GgmlType::kQ8_0) {
        const std::size_t shared_ff = config.shared_expert_ff;
        const DeviceBuffer s_act(kRows * shared_ff);
        const DeviceBuffer b_act(kRows * shared_ff);
        for (std::size_t row = 0; row < kRows; ++row) {
          gufo::hip::LaunchFusedSwiGLUGEMV(
              layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
              layer.ffn_up_shexp.data, layer.ffn_up_shexp.type,
              x.data() + (row * hidden), s_act.data() + (row * shared_ff),
              shared_ff, hidden, arena.stream);
        }
        for (const std::size_t b :
             {std::size_t{2}, std::size_t{3}, std::size_t{4}, kRows}) {
          if (!gufo::hip::TryLaunchBatchedQ8_0FusedSwiGLU(
                  layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data, x.data(),
                  b_act.data(), b, shared_ff, hidden, arena.stream)) {
            throw std::runtime_error(name + ": fused batched SwiGLU refused");
          }
          check.Report(name + ",fused_shared_swiglu", &layer.ffn_gate_shexp,
                       s_act, b_act, b, shared_ff);
        }
      }
    }
    check.Projection("output_head", weights.output, config.vocab_size, hidden,
                     0xA0761D6478BD642FULL);
    // Both kinds of layer and the head were compared.
    if (!seen_attention || !seen_recurrent || check.checks < 30)
      throw std::runtime_error("too few operations were compared");
    std::cout << "checks=" << check.checks << '\n';
    return check.passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_decode_rows_ops_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  return 77;
#endif
}
