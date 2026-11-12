#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/mtp_draft_vocab.hpp"
#include "src/models/qwen/mtp_policy.hpp"

namespace gufo::hip {

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_(executor_->GetHiddenSize()) {}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, QwenMtpGpuDraftConfig config,
    std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor = QwenMtpGpuExecutor::Create(std::move(model),
                                             config.max_context, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  executor->SetVisionInput(config.vision_input);
  return std::unique_ptr<QwenMtpGpuDraftBackend>(
      new QwenMtpGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::CreateFromGguf(
    std::string_view model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenMtpGpuDraftConfig config, std::string* error_msg) {
  if (model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  std::shared_ptr<const QwenMtpGpuModel> model;
  if (model_path == "embedded") {
    model = QwenMtpGpuModel::CreateEmbedded(std::move(target_model), error_msg);
  } else {
    auto reader_owner =
        core::GgufReader::OpenFile(std::string(model_path), error_msg);
    if (reader_owner == nullptr) {
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
    model = QwenMtpGpuModel::Create(std::move(reader), std::move(target_model),
                                    error_msg);
  }
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenMtpGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size == 0 ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    last_error_ = "MTP prompt hidden-state shape is invalid";
    return false;
  }

  try {
    if (context.prompt_tokens.size() > 1) {
      executor_->PrimeBatch(
          context.prompt_tokens.subspan(1),
          context.prompt_hidden_states.first(
              (context.prompt_tokens.size() - 1) * context.hidden_size),
          0);
    }
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(final_hidden, target_hidden_.begin());
    primed_ = true;
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    Reset();
    return false;
  }
}

bool QwenMtpGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ ||
      executor_->GetNextPosition() + 1 != position ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  if (context.prompt_tokens.empty()) {
    return true;
  }
  // Row r pairs token r with the hidden state of the token before it: the
  // carried frontier hidden for r == 0, then the supplied rows.
  std::vector<float> paired(context.prompt_tokens.size() *
                            target_hidden_.size());
  std::ranges::copy(target_hidden_, paired.begin());
  std::copy_n(
      context.prompt_hidden_states.begin(),
      (context.prompt_tokens.size() - 1) * target_hidden_.size(),
      paired.begin() + static_cast<std::ptrdiff_t>(target_hidden_.size()));
  executor_->PrimeBatch(context.prompt_tokens, paired, position - 1);
  std::ranges::copy(context.prompt_hidden_states.last(target_hidden_.size()),
                    target_hidden_.begin());
  return true;
}

void QwenMtpGpuDraftBackend::CheckProposalFrontier(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos) const {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() + 1 != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }
}

speculative::DraftProposal QwenMtpGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  return ProposeImpl(prompt_tokens, current_pos, max_tokens, 0.0F, nullptr);
}

speculative::DraftProposal QwenMtpGpuDraftBackend::ProposeSampled(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument(
        "MTP sampled proposal temperature must be finite and positive");
  }
  if (rng_state == nullptr) {
    throw std::invalid_argument("MTP sampled proposal requires RNG state");
  }
  return ProposeImpl(prompt_tokens, current_pos, max_tokens, temperature,
                     rng_state);
}

std::uint32_t QwenMtpGpuDraftBackend::ProposalRows(
    std::uint32_t max_tokens) const noexcept {
  // Verifying a row reads its own experts, so MoE targets pay for every
  // extra draft row; three is the sweet spot on prose and code.
  constexpr std::uint32_t kMoeDraftCap = 3;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  return executor_->IsMoe() ? std::min(count, kMoeDraftCap) : count;
}

void QwenMtpGpuDraftBackend::BeginProposal(
    std::span<const tokenization::TokenId> prompt_tokens, std::uint32_t count) {
  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposal_target_hidden_ = target_hidden_;
  step_hidden_ = target_hidden_;
  step_checkpoint_ = proposal_checkpoint_;
  step_valid_ = true;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  // A text in a script the draft vocabulary lacks is drafted from the whole
  // vocabulary; the choice holds for every row of this proposal.
  const auto& draft_vocab = executor_->GetDraftVocab();
  executor_->ScoreWholeVocabulary(!draft_vocab.ids.empty() &&
                                  speculative::MtpDraftNeedsWholeVocabulary(
                                      prompt_tokens, draft_vocab.listed));
}

tokenization::TokenId QwenMtpGpuDraftBackend::ChooseDraftToken(
    tokenization::TokenId argmax, float temperature, std::uint64_t* rng_state,
    speculative::DraftProposal& proposal) {
  if (!(temperature > 0.0F)) {
    return argmax;
  }
  // Each sampled row is the distribution actually drawn from, conditional on
  // the earlier sampled tokens fed back into the MTP layer.
  proposal.candidates_per_token = std::min<std::size_t>(
      speculative::kMtpSampledTopK, executor_->ScoredVocab());
  std::vector<tokenization::TokenId> row_ids;
  std::vector<float> row_probs;
  const auto logits = executor_->CopyLastLogits();
  speculative::MtpTopKDistribution(
      logits, temperature, proposal.candidates_per_token, &row_ids, &row_probs);
  for (auto& id : row_ids) {
    id = executor_->ScoredTokenId(static_cast<std::size_t>(id));
  }
  proposal.candidate_ids.insert(proposal.candidate_ids.end(), row_ids.begin(),
                                row_ids.end());
  proposal.candidate_probabilities.insert(
      proposal.candidate_probabilities.end(), row_probs.begin(),
      row_probs.end());
  return speculative::MtpSampleRow(row_ids, row_probs,
                                   sampling::Uniform(rng_state));
}

speculative::DraftProposal QwenMtpGpuDraftBackend::ProposeImpl(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  CheckProposalFrontier(prompt_tokens, current_pos);

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = ProposalRows(max_tokens);
  if (count == 0) {
    return proposal;
  }
  BeginProposal(prompt_tokens, count);
  auto token = ChooseDraftToken(
      executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                     proposal_checkpoint_, true),
      temperature, rng_state, proposal);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    token = ChooseDraftToken(
        executor_->ForwardFeedback(token, proposal_checkpoint_ + index, true),
        temperature, rng_state, proposal);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

speculative::DraftProposal QwenMtpGpuDraftBackend::SkipDraft(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos) {
  CheckProposalFrontier(prompt_tokens, current_pos);
  // A target-only step: the runner appends the committed token through
  // AppendTargetContext, which advances the MTP cache.
  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  return proposal;
}

std::vector<speculative::DraftProposal> QwenMtpGpuDraftBackend::ProposeBatch(
    std::span<const speculative::DraftProposalRequest> requests) {
  static const std::size_t kMaxCohort = [] {
    const char* v = std::getenv("GUFO_MTP_MOE_MAX_COHORT");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v))
                        : speculative::kMtpMoeMaxCohort;
  }();
  if (!executor_->IsMoe() || requests.size() < 2)
    return IDraftBackend::ProposeBatch(requests);
  std::vector<QwenMtpGpuDraftBackend*> backends;
  backends.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* backend =
        dynamic_cast<QwenMtpGpuDraftBackend*>(requests[index].backend);
    if (backend == nullptr)
      return IDraftBackend::ProposeBatch(requests);
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (backends[previous] == backend)
        throw std::invalid_argument("draft batch repeats a session");
    }
    backends.push_back(backend);
  }
  std::vector<speculative::DraftProposal> proposals;
  proposals.reserve(requests.size());
  if (speculative::MtpMoeSkipsDraft(true, requests.size(), kMaxCohort)) {
    for (std::size_t index = 0; index < requests.size(); ++index)
      proposals.push_back(backends[index]->SkipDraft(requests[index].tokens,
                                                     requests[index].position));
    return proposals;
  }

  // One-row proposals share the pass over the LM head: every session runs
  // its own layer, then the sessions scoring the same vocabulary are scored
  // together. Other shapes keep the per-session proposals.
  bool shared = requests.size() <= QwenMtpGpuExecutor::kMaxSharedScoreRows;
  for (std::size_t index = 0; index < requests.size() && shared; ++index) {
    const auto& request = requests[index];
    if (request.temperature > 0.0F && request.rng_state == nullptr)
      throw std::invalid_argument("MTP sampled proposal requires RNG state");
    shared = backends[index]->executor_->SupportsSharedScore() &&
             backends[index]->ProposalRows(request.max_tokens) == 1;
  }
  if (!shared)
    return IDraftBackend::ProposeBatch(requests);
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto& backend = *backends[index];
    backend.CheckProposalFrontier(requests[index].tokens,
                                  requests[index].position);
    backend.BeginProposal(requests[index].tokens, 1);
    (void)backend.executor_->ForwardTargetHidden(
        backend.proposal_input_, backend.target_hidden_,
        backend.proposal_checkpoint_, false);
    backend.executor_->FinishStep();
    proposals.emplace_back().start_pos = requests[index].position;
  }
  std::vector<bool> scored(requests.size(), false);
  std::vector<tokenization::TokenId> argmax(requests.size());
  for (std::size_t first = 0; first < requests.size(); ++first) {
    if (scored[first])
      continue;
    // The sessions that score the vocabulary of `first`.
    std::vector<std::size_t> members;
    std::vector<QwenMtpGpuExecutor*> heads;
    bool keep_logits = false;
    const std::size_t vocabulary = backends[first]->executor_->ScoredVocab();
    for (std::size_t index = first; index < requests.size(); ++index) {
      if (scored[index] ||
          backends[index]->executor_->ScoredVocab() != vocabulary)
        continue;
      scored[index] = true;
      members.push_back(index);
      heads.push_back(backends[index]->executor_.get());
      keep_logits = keep_logits || requests[index].temperature > 0.0F;
    }
    if (members.size() == 1) {
      argmax[first] = heads.front()->ScoreLast();
      continue;
    }
    const auto tokens = QwenMtpGpuExecutor::ScoreTogether(heads, keep_logits);
    for (std::size_t member = 0; member < members.size(); ++member)
      argmax[members[member]] = tokens[member];
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto& backend = *backends[index];
    const auto token =
        backend.ChooseDraftToken(argmax[index], requests[index].temperature,
                                 requests[index].rng_state, proposals[index]);
    backend.proposed_tokens_.push_back(token);
    proposals[index].tokens = backend.proposed_tokens_;
    backend.proposal_active_ = true;
  }
  return proposals;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }
  if (committed_target_hidden_.size() !=
      (accepted.size() + 1) * target_hidden_.size()) {
    throw std::logic_error("MTP feedback requires each committed target row");
  }

  // Pair input t+1 with target hidden t, as during prompt priming. Verification
  // has already updated target_hidden_ to the final committed row; it cannot
  // be reused for the earlier anchor. Accepted rows also need target features,
  // rather than the tentative draft hidden states used while proposing.
  // Only the cache rows of these positions are used afterwards: the next
  // proposal starts from a target hidden state again.
  executor_->Rewind(proposal_checkpoint_);
  executor_->CommitTargetHidden(proposal_input_, proposal_target_hidden_,
                                proposal_checkpoint_);
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    executor_->CommitTargetHidden(
        accepted[index],
        std::span<const float>(committed_target_hidden_)
            .subspan(index * target_hidden_.size(), target_hidden_.size()),
        proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1);
  }
  proposal_active_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  if (proposal_active_) {
    committed_target_hidden_.insert(committed_target_hidden_.end(),
                                    hidden.begin(), hidden.end());
  }
  std::ranges::copy(hidden, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  step_valid_ = false;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

void QwenMtpGpuDraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (proposal_active_ || !step_valid_ || step_checkpoint_ + 1 != position) {
    throw std::logic_error("MTP cancellation frontier is not retained");
  }
  executor_->Rewind(step_checkpoint_);
  target_hidden_ = step_hidden_;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

std::vector<std::uint8_t> QwenMtpGpuDraftBackend::SerializeState() const {
  const std::size_t state_bytes = executor_->StateBytes();
  std::vector<std::uint8_t> payload(
      state_bytes + (target_hidden_.size() * sizeof(float)) + 1);
  executor_->SaveState(std::span(payload).first(state_bytes));
  std::memcpy(payload.data() + state_bytes, target_hidden_.data(),
              target_hidden_.size() * sizeof(float));
  payload.back() = primed_ ? 1 : 0;
  return payload;
}

void QwenMtpGpuDraftBackend::LoadSerializedState(
    std::span<const std::uint8_t> payload) {
  const std::size_t tail = (target_hidden_.size() * sizeof(float)) + 1;
  if (payload.size() < tail + sizeof(std::uint32_t)) {
    throw std::invalid_argument("MTP snapshot payload is truncated");
  }
  const std::size_t state_bytes = payload.size() - tail;
  executor_->LoadState(payload.first(state_bytes));
  std::memcpy(target_hidden_.data(), payload.data() + state_bytes,
              target_hidden_.size() * sizeof(float));
  primed_ = payload.back() != 0;
  proposal_active_ = false;
  step_valid_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

namespace {
class QwenMtpDraftSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  explicit QwenMtpDraftSnapshot(std::vector<std::uint8_t> payload)
      : payload_(std::move(payload)) {}
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return payload_.size();
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    return payload_.size();
  }
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    if (destination.size() != payload_.size()) {
      throw std::invalid_argument(
          "MTP snapshot destination has the wrong size");
    }
    std::memcpy(destination.data(), payload_.data(), payload_.size());
    return payload_.size();
  }
  [[nodiscard]] std::span<const std::uint8_t> Payload() const noexcept {
    return payload_;
  }

private:
  std::vector<std::uint8_t> payload_;
};
}  // namespace

std::size_t QwenMtpGpuDraftBackend::SnapshotPayloadBytes() const {
  return executor_->StateBytes() + (target_hidden_.size() * sizeof(float)) + 1;
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenMtpGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "MTP snapshot requires a committed proposal boundary");
  }
  return std::make_unique<QwenMtpDraftSnapshot>(SerializeState());
}

void QwenMtpGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* mtp = dynamic_cast<const QwenMtpDraftSnapshot*>(&snapshot);
  if (mtp == nullptr) {
    throw std::invalid_argument("snapshot does not belong to an MTP draft");
  }
  LoadSerializedState(mtp->Payload());
}

void QwenMtpGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  LoadSerializedState(payload);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
