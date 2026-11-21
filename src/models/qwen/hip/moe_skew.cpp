#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_skew.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string_view>

#include "src/core/hip/hip_utils.hpp"

namespace gufo::hip {
namespace {

std::mutex& SkewFileMutex() noexcept {
  static std::mutex mutex;
  return mutex;
}

bool IsDisabledFlag(const char* flag) noexcept {
  return flag == nullptr || flag[0] == '\0' || std::string_view(flag) == "0";
}

}  // namespace

bool MoeSkewEnabled() {
  return !IsDisabledFlag(std::getenv("GUFO_QWEN_MOE_SKEW"));
}

std::string MoeSkewPath() {
  const char* const flag = std::getenv("GUFO_QWEN_MOE_SKEW");
  if (IsDisabledFlag(flag)) {
    return {};
  }
  // "1" records next to the working directory; anything else is a path.
  if (std::string_view(flag) == "1") {
    return "moe_skew.jsonl";
  }
  return flag;
}

QwenMoeSkewHost::QwenMoeSkewHost(std::uint32_t layers, std::uint32_t experts,
                                 std::uint32_t used, std::string path,
                                 hipStream_t stream)
    : layers_(layers), experts_(experts), used_(used), path_(std::move(path)) {
  device_.layers = layers;
  device_.experts = experts;
  device_.used = used;
  if (!AllocBuffers(stream)) {
    FreeBuffers();
    throw std::runtime_error("GUFO_QWEN_MOE_SKEW device allocation failed");
  }
}

bool QwenMoeSkewHost::AllocBuffers(hipStream_t stream) {
  const std::size_t lb_count = static_cast<std::size_t>(layers_) *
                               static_cast<std::size_t>(kMoeSkewBuckets);
  const std::size_t hist_count = lb_count * experts_;
  const std::size_t slot_count = lb_count * used_;
  const std::size_t tail_count = lb_count * 2;
  if (hipMalloc(reinterpret_cast<void**>(&device_.hist),
                hist_count * sizeof(std::int32_t)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.slot_weights),
                slot_count * sizeof(float)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.rows),
                lb_count * sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.tokens),
                lb_count * sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.unique_experts),
                lb_count * sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.tail_lt1pct),
                tail_count * sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.dup),
                lb_count * sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.dup_pairs),
                lb_count * sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.prev_ids),
                static_cast<std::size_t>(layers_) * used_ *
                    sizeof(std::int32_t)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.overlap),
                static_cast<std::size_t>(layers_) *
                    sizeof(unsigned long long)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.overlap_rows),
                static_cast<std::size_t>(layers_) *
                    sizeof(unsigned long long)) != hipSuccess) {
    return false;
  }
  hist_.assign(hist_count, 0);
  slot_weights_.assign(slot_count, 0.0F);
  rows_.assign(lb_count, 0);
  tokens_.assign(lb_count, 0);
  unique_experts_.assign(lb_count, 0);
  tail_lt1pct_.assign(tail_count, 0);
  dup_.assign(lb_count, 0);
  dup_pairs_.assign(lb_count, 0);
  prev_ids_.assign(static_cast<std::size_t>(layers_) * used_, -1);
  overlap_.assign(layers_, 0);
  overlap_rows_.assign(layers_, 0);
  Rearm(stream);
  return true;
}

void QwenMoeSkewHost::FreeBuffers() noexcept {
  (void)hipFree(device_.hist);
  (void)hipFree(device_.slot_weights);
  (void)hipFree(device_.rows);
  (void)hipFree(device_.tokens);
  (void)hipFree(device_.unique_experts);
  (void)hipFree(device_.tail_lt1pct);
  (void)hipFree(device_.dup);
  (void)hipFree(device_.dup_pairs);
  (void)hipFree(device_.prev_ids);
  (void)hipFree(device_.overlap);
  (void)hipFree(device_.overlap_rows);
  device_ = {};
}

QwenMoeSkewHost::~QwenMoeSkewHost() {
  Flush();
  FreeBuffers();
}

void QwenMoeSkewHost::CopyOut(hipStream_t stream) noexcept {
  // Same-stream copies order after the recording kernels; the trailing sync
  // makes the host mirror valid before the record is built.
  const std::size_t lb_count = static_cast<std::size_t>(layers_) *
                               static_cast<std::size_t>(kMoeSkewBuckets);
  (void)hipMemcpyAsync(hist_.data(), device_.hist,
                       lb_count * experts_ * sizeof(std::int32_t),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(slot_weights_.data(), device_.slot_weights,
                       lb_count * used_ * sizeof(float), hipMemcpyDeviceToHost,
                       stream);
  (void)hipMemcpyAsync(rows_.data(), device_.rows,
                       lb_count * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(tokens_.data(), device_.tokens,
                       lb_count * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(unique_experts_.data(), device_.unique_experts,
                       lb_count * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(tail_lt1pct_.data(), device_.tail_lt1pct,
                       lb_count * 2 * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(dup_.data(), device_.dup,
                       lb_count * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(dup_pairs_.data(), device_.dup_pairs,
                       lb_count * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(
      prev_ids_.data(), device_.prev_ids,
      static_cast<std::size_t>(layers_) * used_ * sizeof(std::int32_t),
      hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(
      overlap_.data(), device_.overlap,
      static_cast<std::size_t>(layers_) * sizeof(unsigned long long),
      hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(
      overlap_rows_.data(), device_.overlap_rows,
      static_cast<std::size_t>(layers_) * sizeof(unsigned long long),
      hipMemcpyDeviceToHost, stream);
  (void)hipStreamSynchronize(stream);
}

void QwenMoeSkewHost::AppendRecord() {
  ++ordinal_;
  auto layer_join = [&](auto&& render) {
    std::string out;
    for (std::size_t layer = 0; layer < layers_; ++layer) {
      out += render(layer);
      out += layer + 1 < layers_ ? "," : "";
    }
    return out;
  };
  auto bucket_join = [&](std::size_t layer, auto&& render) {
    std::string out = "[";
    for (int bucket = 0; bucket < kMoeSkewBuckets; ++bucket) {
      out += render(layer, static_cast<std::size_t>(bucket));
      out += bucket + 1 < kMoeSkewBuckets ? "," : "";
    }
    out += "]";
    return out;
  };
  auto render_float = [](float value) {
    return std::to_string(std::isfinite(value) ? value : 0.0F);
  };
  std::string record =
      "{\"ordinal\":" + std::to_string(ordinal_) +
      ",\"layers\":" + std::to_string(layers_) +
      ",\"experts\":" + std::to_string(experts_) +
      ",\"used\":" + std::to_string(used_) + ",\"rows\":[" +
      layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          return std::to_string(rows_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"tokens\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          return std::to_string(tokens_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"unique_experts\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          return std::to_string(unique_experts_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"tail_lt1pct\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          const std::size_t base = (l * kMoeSkewBuckets + b) * 2;
          return "[" + std::to_string(tail_lt1pct_[base + 0]) + "," +
                 std::to_string(tail_lt1pct_[base + 1]) + "]";
        });
      }) +
      "],\"hist\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          const std::size_t base = (l * kMoeSkewBuckets + b) * experts_;
          std::string row = "[";
          for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            row += std::to_string(hist_[base + expert]);
            row += expert + 1 < experts_ ? "," : "";
          }
          return row + "]";
        });
      }) +
      "],\"slot_weights\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          const std::size_t base = (l * kMoeSkewBuckets + b) * used_;
          std::string row = "[";
          for (std::uint32_t slot = 0; slot < used_; ++slot) {
            row += render_float(slot_weights_[base + slot]);
            row += slot + 1 < used_ ? "," : "";
          }
          return row + "]";
        });
      }) +
      "],\"dup\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          return std::to_string(dup_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"dup_pairs\":[" + layer_join([&](std::size_t layer) {
        return bucket_join(layer, [&](std::size_t l, std::size_t b) {
          return std::to_string(dup_pairs_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"overlap\":[" + layer_join([&](std::size_t layer) {
        return std::to_string(overlap_[layer]);
      }) +
      "],\"overlap_rows\":[" + layer_join([&](std::size_t layer) {
        return std::to_string(overlap_rows_[layer]);
      }) +
      "]}\n";
  records_.push_back(std::move(record));
}

void QwenMoeSkewHost::Rearm(hipStream_t stream) noexcept {
  const std::size_t lb_count = static_cast<std::size_t>(layers_) *
                               static_cast<std::size_t>(kMoeSkewBuckets);
  (void)hipMemsetAsync(device_.hist, 0,
                       lb_count * experts_ * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(device_.slot_weights, 0,
                       lb_count * used_ * sizeof(float), stream);
  (void)hipMemsetAsync(device_.rows, 0, lb_count * sizeof(unsigned long long),
                       stream);
  (void)hipMemsetAsync(device_.tokens, 0, lb_count * sizeof(unsigned long long),
                       stream);
  (void)hipMemsetAsync(device_.unique_experts, 0,
                       lb_count * sizeof(unsigned long long), stream);
  (void)hipMemsetAsync(device_.tail_lt1pct, 0,
                       lb_count * 2 * sizeof(unsigned long long), stream);
  (void)hipMemsetAsync(device_.dup, 0, lb_count * sizeof(unsigned long long),
                       stream);
  (void)hipMemsetAsync(device_.dup_pairs, 0,
                       lb_count * sizeof(unsigned long long), stream);
  // prev_ids = -1: no previous token yet.
  (void)hipMemsetAsync(
      device_.prev_ids, 0xFF,
      static_cast<std::size_t>(layers_) * used_ * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(
      device_.overlap, 0,
      static_cast<std::size_t>(layers_) * sizeof(unsigned long long), stream);
  (void)hipMemsetAsync(
      device_.overlap_rows, 0,
      static_cast<std::size_t>(layers_) * sizeof(unsigned long long), stream);
}

void QwenMoeSkewHost::OnReset(hipStream_t stream) noexcept {
  if (device_.hist == nullptr) {
    return;
  }
  // Reset paths are noexcept; a failed dump must never take the server down.
  try {
    // Probe the call counters first: a reset without any recorded call (the
    // warm-up, a failed load) must not emit an empty record, but the
    // neighbor ids still have to be re-armed for the next request.
    rows_probe_.assign(rows_.size(), 0);
    (void)hipMemcpyAsync(rows_probe_.data(), device_.rows,
                         rows_.size() * sizeof(unsigned long long),
                         hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
    const bool any =
        std::any_of(rows_probe_.begin(), rows_probe_.end(),
                    [](unsigned long long rows) { return rows != 0; });
    if (any) {
      CopyOut(stream);
      AppendRecord();
    }
    Rearm(stream);
    if (!records_.empty()) {
      Flush();
    }
  } catch (...) {
  }
}

void QwenMoeSkewHost::Flush() noexcept {
  if (records_.empty()) {
    return;
  }
  try {
    // Serialize multi-session appends: each JSONL line is ~150 KB and must
    // never interleave across concurrent pooled sessions.
    const std::lock_guard lock(SkewFileMutex());
    if (FILE* file = std::fopen(path_.c_str(), "a")) {
      for (const auto& record : records_) {
        std::fwrite(record.data(), 1, record.size(), file);
      }
      std::fflush(file);
      std::fclose(file);
    }
    records_.clear();
  } catch (...) {
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
