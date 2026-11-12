#ifndef GUFO_MODELS_QWEN_HIP_MTP_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Immutable GPU-visible MTP layer weights. Current GGUF K-quant matrices are
/// expanded once to BF16; the tied embedding and LM head remain shared with
/// the target model.
class QwenMtpGpuModel final {
public:
  ~QwenMtpGpuModel();

  QwenMtpGpuModel(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel& operator=(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel(QwenMtpGpuModel&&) = delete;
  QwenMtpGpuModel& operator=(QwenMtpGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenMtpGpuModel> Create(
      std::shared_ptr<const core::GgufReader> mtp_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  /// The prediction head embedded in the target's own GGUF (MoE checkpoints);
  /// it reads the target's device weights and packs nothing.
  [[nodiscard]] static std::shared_ptr<const QwenMtpGpuModel> CreateEmbedded(
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenMtpWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }
  [[nodiscard]] double GetPackTimeSeconds() const noexcept {
    return pack_time_seconds_;
  }

  /// The LM-head rows a draft scores (speculative::MtpDraftScoresToken),
  /// gathered into a matrix of their own, their token ids in ascending
  /// order, and the same set indexed by token id. Without ids a draft scores
  /// the whole vocabulary through the target's head.
  struct DraftVocab {
    models::QwenTensorRef output;
    std::vector<tokenization::TokenId> ids;
    std::vector<bool> listed;
  };
  [[nodiscard]] const DraftVocab& GetDraftVocab() const noexcept {
    return draft_vocab_;
  }

private:
  void GatherDraftVocab();

  QwenMtpGpuModel(std::shared_ptr<const core::GgufReader> mtp_reader,
                  std::shared_ptr<const QwenGpuModel> target_model,
                  speculative::QwenMtpWeights weights,
                  std::vector<void*> allocations,
                  std::size_t packed_weight_bytes, double pack_time_seconds);

  std::shared_ptr<const core::GgufReader> mtp_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenMtpWeights weights_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
  double pack_time_seconds_{0.0};
  DraftVocab draft_vocab_;
};

/// Mutable single-session executor for the Qwen3.8 layer-64 MTP graph.
class QwenMtpGpuExecutor final {
public:
  ~QwenMtpGpuExecutor();

  QwenMtpGpuExecutor(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor& operator=(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor(QwenMtpGpuExecutor&&) = delete;
  QwenMtpGpuExecutor& operator=(QwenMtpGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuExecutor> Create(
      std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  void SetVisionInput(models::qwen::vision::DeviceInput* input) noexcept {
    vision_input_ = input;
  }
  void Reset() noexcept;
  void Rewind(std::uint32_t position);

  [[nodiscard]] tokenization::TokenId ForwardTargetHidden(
      tokenization::TokenId input_token, std::span<const float> target_hidden,
      std::uint32_t position, bool compute_logits = true);

  /// Writes the K/V cache entries of `tokens.size()` consecutive positions in
  /// batches: row i pairs input token tokens[i] with target hidden state
  /// hidden[i] at start_position + i. Only the projections the cache needs run.
  void PrimeBatch(std::span<const tokenization::TokenId> tokens,
                  std::span<const float> hidden, std::uint32_t start_position);

  /// Serialized cache state for snapshots: positions, then the K/V planes of
  /// the valid prefix: both precisions, or FP16 alone when that is the only
  /// cache this head keeps (KeepsHalfKvOnly).
  [[nodiscard]] std::size_t StateBytes() const noexcept;
  [[nodiscard]] static std::size_t StateBytesAt(
      const core::ModelConfig& config, std::uint32_t position) noexcept;
  void SaveState(std::span<std::uint8_t> destination) const;
  void LoadState(std::span<const std::uint8_t> source);
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept;
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const QwenMtpGpuModel& model, std::uint32_t max_context);

  [[nodiscard]] tokenization::TokenId ForwardFeedback(
      tokenization::TokenId input_token, std::uint32_t position,
      bool compute_logits = true);

  /// Commits one position to the K/V cache from its input token and target
  /// hidden state, with the rows ForwardTargetHidden would write, and stops
  /// there: no attention read, feed-forward or logits. The head's last hidden
  /// state is left unspecified.
  void CommitTargetHidden(tokenization::TokenId input_token,
                          std::span<const float> target_hidden,
                          std::uint32_t position);

  [[nodiscard]] std::span<const float> CopyLastHidden();
  /// The ScoredVocab() logits of the last forward pass.
  [[nodiscard]] std::span<const float> CopyLastLogits();
  /// Number of logits a forward pass computes, and the token id of logit
  /// `index`: the draft vocabulary, or every id in order without one or
  /// after ScoreWholeVocabulary(true).
  [[nodiscard]] std::size_t ScoredVocab() const noexcept;
  [[nodiscard]] tokenization::TokenId ScoredTokenId(
      std::size_t index) const noexcept;
  [[nodiscard]] const QwenMtpGpuModel::DraftVocab& GetDraftVocab()
      const noexcept {
    return model_->GetDraftVocab();
  }
  /// Chooses the vocabulary of the following forward passes.
  void ScoreWholeVocabulary(bool whole) noexcept { whole_vocab_ = whole; }

  [[nodiscard]] std::uint32_t GetNextPosition() const noexcept {
    return next_position_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }
  [[nodiscard]] bool IsMoe() const noexcept {
    return model_->GetConfig().IsMoe();
  }
  /// Eight query heads per KV head (Qwen3.6-35B-A3B): the head keeps its K/V
  /// in FP16 alone, as the target does, and its attention stages each KV tile
  /// once for the whole head group instead of once per query head from FP32.
  [[nodiscard]] static bool KeepsHalfKvOnly(
      const core::ModelConfig& config) noexcept {
    return config.num_key_value_heads != 0 &&
           config.num_attention_heads == 8 * config.num_key_value_heads;
  }

private:
  QwenMtpGpuExecutor(std::shared_ptr<const QwenMtpGpuModel> model,
                     std::uint32_t max_context);

  [[nodiscard]] tokenization::TokenId Run(tokenization::TokenId input_token,
                                          const float* hidden_input,
                                          std::uint32_t position,
                                          bool compute_logits,
                                          bool cache_only = false);
  void Allocate();
  void Free() noexcept;

  std::shared_ptr<const QwenMtpGpuModel> model_;
  std::uint32_t max_context_;
  std::uint32_t next_position_{0};
  bool whole_vocab_{false};
  hipStream_t stream_{nullptr};

  float* d_target_hidden_{nullptr};
  float* d_embedding_{nullptr};
  float* d_fusion_{nullptr};
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_qg_{nullptr};
  float* d_q_{nullptr};
  float* d_k_{nullptr};
  float* d_v_{nullptr};
  float* d_gate_{nullptr};
  float* d_context_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_act_{nullptr};
  float* d_ffn_out_{nullptr};
  float* d_feedback_hidden_{nullptr};
  float* d_logits_{nullptr};
  float* d_moe_buffer_{nullptr};
  QwenMoeScratch moe_;
  float* d_kv_cache_{nullptr};
  void* d_kv_cache_f16_{nullptr};
  float* d_split_k_scratch_{nullptr};
  std::uint32_t* d_out_token_{nullptr};

  static constexpr std::size_t kPrimeChunk = 1024;
  void EnsurePrimeScratch();
  std::uint32_t* d_prime_tokens_{nullptr};
  float* d_prime_buffer_{nullptr};
  void* d_prime_bf16_{nullptr};
  std::size_t prime_buffer_elements_{0};

  models::qwen::vision::DeviceInput* vision_input_{
      nullptr};  // owned by the target session
  std::vector<float> h_last_hidden_;
  std::vector<float> h_logits_;
};

struct QwenMtpGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{4};
  models::qwen::vision::DeviceInput* vision_input{nullptr};
};

/// Stateful MTP draft provider backed by the production HIP executor.
class QwenMtpGpuDraftBackend final : public speculative::IDraftBackend {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> Create(
      std::shared_ptr<const QwenMtpGpuModel> model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> CreateFromGguf(
      std::string_view model_path,
      std::shared_ptr<const QwenGpuModel> target_model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenMtpGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  /// Samples each draft token from the MTP head at the request temperature
  /// and reports its top-k distribution for lossless rejection sampling.
  [[nodiscard]] speculative::DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override;
  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return true;
  }
  /// MoE targets verify every draft row against its own experts; wide
  /// cohorts decode without a draft (GUFO_MTP_MOE_MAX_COHORT, default 2).
  [[nodiscard]] std::vector<speculative::DraftProposal> ProposeBatch(
      std::span<const speculative::DraftProposalRequest> requests) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;
  void Reset() noexcept override;

  void DiscardPendingTargetContext(std::uint32_t position) override;
  /// Records the state a cancellation rewinds to, for steps another backend
  /// proposes. Call once per step before the committed rows arrive.
  void MarkStep() {
    step_hidden_ = target_hidden_;
    step_checkpoint_ = executor_->GetNextPosition();
    step_valid_ = true;
  }
  [[nodiscard]] std::uint32_t NextPosition() const noexcept {
    return executor_->GetNextPosition();
  }
  [[nodiscard]] bool Primed() const noexcept { return primed_; }
  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return target_hidden_.size();
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> RequestMemoryBytes()
      const override {
    const auto usage = executor_->GetMemoryUsage();
    return {usage.request_state_bytes, usage.temporary_scratch_bytes};
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;

  [[nodiscard]] const std::string& GetLastError() const noexcept {
    return last_error_;
  }

private:
  QwenMtpGpuDraftBackend(std::unique_ptr<QwenMtpGpuExecutor> executor,
                         QwenMtpGpuDraftConfig config);
  [[nodiscard]] std::vector<std::uint8_t> SerializeState() const;
  void LoadSerializedState(std::span<const std::uint8_t> payload);
  [[nodiscard]] speculative::DraftProposal ProposeImpl(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state);
  /// Returns an empty proposal for a target-only step (the runner then
  /// appends the committed token, advancing the MTP cache).
  [[nodiscard]] speculative::DraftProposal SkipDraft(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos);
  void CheckProposalFrontier(std::span<const tokenization::TokenId> prompt_tokens,
                             std::uint32_t current_pos) const;

  std::unique_ptr<QwenMtpGpuExecutor> executor_;
  QwenMtpGpuDraftConfig config_;
  std::vector<float> target_hidden_;
  std::vector<float> proposal_target_hidden_;
  std::vector<float> committed_target_hidden_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  std::vector<float> step_hidden_;
  std::uint32_t step_checkpoint_{0};
  bool step_valid_{false};
  tokenization::TokenId proposal_input_{0};
  std::uint32_t proposal_checkpoint_{0};
  bool primed_{false};
  bool proposal_active_{false};
  std::string last_error_;
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MTP_HPP_
