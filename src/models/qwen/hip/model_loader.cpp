#if defined(ENGINE_ENABLE_HIP)
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_allocation && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    } else if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      (void)munmap(region.host_copy, region.size);
    }
    region = {};
  }
}

/// Calls visit(worker, offset, chunk, size) for every chunk of the mapping
/// from up to `workers` threads, populating each chunk first and dropping its
/// pages afterwards. A visitor returning false stops the walk.
template<typename Visit>
void VisitMappedChunks(const core::GgufMappedRegion& source,
                       std::size_t chunk_bytes, std::size_t workers,
                       Visit&& visit) {
  const auto chunks =
      source.size / chunk_bytes + (source.size % chunk_bytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  std::atomic<bool> stopped{false};
  auto visit_chunks = [&](std::size_t worker) {
    while (failure.load(std::memory_order_relaxed) == 0 &&
           !stopped.load(std::memory_order_relaxed)) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * chunk_bytes;
      const auto bytes = std::min(chunk_bytes, source.size - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
      if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      const bool keep_going = visit(worker, offset, input, bytes);
      // The visitor owns a copy of these bytes now. Keep the original
      // mapping valid for the reader without retaining its populated PTEs.
      (void)madvise(input, bytes, MADV_DONTNEED);
      if (!keep_going) {
        stopped.store(true, std::memory_order_relaxed);
        break;
      }
    }
  };
  {
    std::vector<std::jthread> threads;
    const auto count = std::clamp<std::size_t>(workers, 1, chunks);
    for (std::size_t worker = 1; worker < count; ++worker)
      threads.emplace_back(visit_chunks, worker);
    visit_chunks(0);
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read mapped Qwen weights");
}

void CopyMappedWeights(const core::GgufMappedRegion& source, void* copy) {
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  VisitMappedChunks(source, kChunkBytes, 16,
                    [&](std::size_t, std::size_t offset, const void* chunk,
                        std::size_t bytes) {
                      std::memcpy(static_cast<std::uint8_t*>(copy) + offset,
                                  chunk, bytes);
                      return true;
                    });
}

/// Device-resident placement: one hipMalloc copy. Device memory is not on the
/// kernel's page LRU, so memory compaction never migrates it.
[[nodiscard]] hipError_t AllocateDeviceRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* device_data = nullptr;
  const auto status = hipMalloc(&device_data, source.size);
  if (status != hipSuccess)
    return status;
  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = nullptr,
                 .size = source.size,
                 .owns_device_allocation = true};
  return hipSuccess;
}

/// Streams one region from the read-only mapping into device memory. Eight
/// workers each copy 64 MiB chunks through their own pinned staging buffer,
/// so a shard in the page cache uploads at memory speed; a hipMemcpy straight
/// from the pageable mapping pins it page by page.
[[nodiscard]] hipError_t CopyMappedRegion(const core::GgufMappedRegion& source,
                                          void* device_data) {
  constexpr std::size_t kChunkBytes = 64ULL << 20;
  constexpr std::size_t kWorkers = 8;
  const auto free_staging = [](void* pointer) { (void)hipHostFree(pointer); };
  using Staging = std::unique_ptr<void, decltype(free_staging)>;
  std::vector<Staging> staging;
  staging.reserve(kWorkers);
  for (std::size_t i = 0; i < kWorkers; ++i)
    staging.emplace_back(nullptr, free_staging);
  std::atomic<hipError_t> failure{hipSuccess};
  VisitMappedChunks(
      source, kChunkBytes, kWorkers,
      [&](std::size_t worker, std::size_t offset, const void* chunk,
          std::size_t bytes) {
        auto& buffer = staging[worker];
        if (!buffer) {
          void* pointer = nullptr;
          if (const auto status = hipHostMalloc(&pointer, kChunkBytes);
              status != hipSuccess) {
            failure.store(status, std::memory_order_relaxed);
            return false;
          }
          buffer.reset(pointer);
        }
        std::memcpy(buffer.get(), chunk, bytes);
        const auto status =
            hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                      buffer.get(), bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess) {
          failure.store(status, std::memory_order_relaxed);
          return false;
        }
        return true;
      });
  return failure.load();
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* host_copy = mmap(nullptr, source.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED)
    return hipErrorOutOfMemory;
  const auto unmap = [&](void* pointer) { (void)munmap(pointer, source.size); };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  (void)madvise(host_copy, source.size, MADV_HUGEPAGE);
  CopyMappedWeights(source, host_copy);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      // A registered anonymous copy is on the kernel's page LRU: every
      // compaction migration makes the driver evict and restore the GPU
      // mapping, and kernels reading it stall. Prefer device memory while the
      // heap keeps its dispatch reserve; otherwise keep the registered copy.
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
          HeapFitsAboveReserve(free_bytes, source.size) &&
          AllocateDeviceRegion(source, destination) == hipSuccess) {
        map_error = CopyMappedRegion(source, destination.device_data);
      } else {
        (void)hipGetLastError();
        destination = {};
        map_error = MapRegisteredRegion(source, destination);
      }
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (map_error != hipSuccess || destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible: " + hipGetErrorString(map_error);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  for (auto* buffer : router_split_) {
    if (buffer != nullptr)
      (void)hipFree(buffer);
  }
  ReleaseWeightRegions(weight_regions_);
}

const std::uint16_t* QwenGpuModel::RouterSplitWeights(
    std::uint32_t layer) const {
  std::lock_guard lock(router_split_mutex_);
  if (router_split_.empty())
    router_split_.assign(weights_.layers.size(), nullptr);
  if (layer >= router_split_.size())
    return nullptr;
  if (router_split_[layer] != nullptr)
    return router_split_[layer];
  const auto& config = weights_.config;
  const auto& gate = weights_.layers[layer].ffn_gate_inp;
  const auto& shared = weights_.layers[layer].ffn_gate_inp_shexp;
  const auto is_supported = [](const models::QwenTensorRef& tensor) {
    return tensor.type == core::GgmlType::kF32 ||
           tensor.type == core::GgmlType::kBF16;
  };
  const std::size_t hidden = config.hidden_size;
  if (config.num_experts + 1 > kRouterSplitRows || !is_supported(gate) ||
      !is_supported(shared) ||
      gate.num_elements != config.num_experts * hidden ||
      shared.num_elements != hidden)
    return nullptr;
  std::uint16_t* buffer = nullptr;
  const std::size_t bytes = kRouterSplitRows * 3 * hidden * sizeof(*buffer);
  if (hipMalloc(&buffer, bytes) != hipSuccess)
    return nullptr;
  HIP_CHECK(hipMemset(buffer, 0, bytes));
  LaunchRouterSplitWeights(gate.data, gate.type == core::GgmlType::kBF16,
                           buffer, 0, config.num_experts, hidden);
  LaunchRouterSplitWeights(shared.data, shared.type == core::GgmlType::kBF16,
                           buffer, config.num_experts, 1, hidden);
  HIP_CHECK(hipDeviceSynchronize());
  router_split_[layer] = buffer;
  return buffer;
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer)
    return nullptr;

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down) &&
               remap(layer.ffn_gate_inp) && remap(layer.ffn_gate_exps) &&
               remap(layer.ffn_up_exps) && remap(layer.ffn_down_exps) &&
               remap(layer.ffn_gate_shexp) && remap(layer.ffn_up_shexp) &&
               remap(layer.ffn_down_shexp) && remap(layer.ffn_gate_inp_shexp);
  }
  if (weights_opt->mtp.has_value()) {
    auto& mtp = *weights_opt->mtp;
    auto& l = mtp.layer;
    const bool mtp_remapped =
        remap(mtp.embedding_norm) && remap(mtp.hidden_norm) &&
        remap(mtp.fusion_projection) && remap(mtp.shared_head_norm) &&
        remap(l.attn_norm) && remap(l.ffn_norm) && remap(l.attn_q) &&
        remap(l.attn_k) && remap(l.attn_v) && remap(l.attn_output) &&
        remap(l.attn_q_norm) && remap(l.attn_k_norm) && remap(l.ffn_gate_inp) &&
        remap(l.ffn_gate_exps) && remap(l.ffn_up_exps) &&
        remap(l.ffn_down_exps) && remap(l.ffn_gate_shexp) &&
        remap(l.ffn_up_shexp) && remap(l.ffn_down_shexp) &&
        remap(l.ffn_gate_inp_shexp);
    if (!mtp_remapped) {
      weights_opt->mtp.reset();
    }
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  return std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision));
}

QwenNpuFfn* QwenGpuModel::NpuFfn() const {
  std::call_once(npu_once_, [this] {
    if (std::getenv("GUFO_NPU_DIR") == nullptr)
      return;
    std::string error;
    npu_ffn_ = CreateQwenNpuFfn(weights_, &error);
    if (npu_ffn_ == nullptr && !error.empty())
      std::fprintf(stderr, "NPU off: %s\n", error.c_str());
  });
  return npu_ffn_.get();
}

QwenMoePrefill* QwenGpuModel::MoePrefill(std::uint32_t max_tokens) const {
  std::call_once(moe_prefill_once_, [this, max_tokens] {
    if (!weights_.config.IsMoe())
      return;
    std::string error;
    moe_prefill_ = CreateQwenMoePrefill(weights_, max_tokens, &error);
    if (moe_prefill_ == nullptr && !error.empty())
      std::fprintf(stderr, "MoE prefill off: %s\n", error.c_str());
  });
  return moe_prefill_.get();
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
