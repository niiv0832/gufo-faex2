#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models {
struct QwenModelWeights;
struct QwenTensorRef;
}  // namespace gufo::models

namespace gufo::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

/// Emits pure decode route decisions before entering graph capture. This must
/// never be called from the capture callback.
void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy);

/// Runs one graph-capture-safe decode layer stack using stable arena storage.
/// `recurrent_state` is the pass's QwenGpuArena::BeginRecurrentWrite result;
/// a captured graph bakes its pointers, so the graph key must carry
/// QwenGpuArena::RecurrentIoIdentity.
void ExecuteDecodeStep(
    QwenGpuArena& arena, const models::QwenModelWeights& weights,
    const QwenExecutionPolicy& policy, tokenization::TokenId token_id,
    std::uint32_t pos, bool compute_logits,
    QwenRecurrentStateIo recurrent_state,
    models::qwen::vision::DeviceInput* image_input = nullptr);

/// The projection batched decode and speculative verification apply to `rows`
/// independent rows. Every row must equal, bit for bit, what single-token
/// decode computes for it: sampled speculation serves the verification
/// distribution as the target.
void LaunchDecodeRowsProjection(const models::QwenTensorRef& weight,
                                const float* input, float* output,
                                std::size_t rows, std::size_t output_size,
                                std::size_t input_size, hipStream_t stream);

/// The recurrent step controls (alpha, beta) for `rows` independent rows, with
/// the same bit-for-bit contract.
void LaunchDecodeRowsSsmControls(const models::QwenTensorRef& alpha,
                                 const models::QwenTensorRef& beta,
                                 const float* input, float* alpha_output,
                                 float* beta_output, std::size_t rows,
                                 std::size_t time_step_rank,
                                 std::size_t hidden_size, hipStream_t stream);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
