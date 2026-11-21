#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_skew.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"

namespace gufo::hip {

bool MoeSkewEnabled() {
  const char* const flag = std::getenv("GUFO_QWEN_MOE_SKEW");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

std::string MoeSkewPath() {
  const char* const flag = std::getenv("GUFO_QWEN_MOE_SKEW");
  if (flag == nullptr || flag[0] == '\0' || flag[0] == '0') {
    return {};
  }
  // "1" records next to the working directory; anything else is a path.
  if (flag[0] == '1' && flag[1] == '\0') {
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
  const size_t hist_count = size_t(layers_) * kMoeSkewBuckets * experts_;
  const size_t slot_count = size_t(layers_) * kMoeSkewBuckets * used_;
  if (hipMalloc(reinterpret_cast<void**>(&device_.hist),
                hist_count * sizeof(std::int32_t)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.slot_weights),
                slot_count * sizeof(float)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.rows),
                size_t(layers_) * kMoeSkewBuckets * sizeof(std::uint64_t)) !=
          hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.dup),
                size_t(layers_) * kMoeSkewBuckets * sizeof(std::int64_t)) !=
          hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.dup_pairs),
                size_t(layers_) * kMoeSkewBuckets * sizeof(std::uint64_t)) !=
          hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.prev_ids),
                size_t(layers_) * used_ * sizeof(std::int32_t)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.overlap),
                size_t(layers_) * sizeof(std::int64_t)) != hipSuccess ||
      hipMalloc(reinterpret_cast<void**>(&device_.overlap_rows),
                size_t(layers_) * sizeof(std::uint64_t)) != hipSuccess) {
    return false;
  }
  hist_.assign(hist_count, 0);
  slot_weights_.assign(slot_count, 0.0F);
  rows_.assign(size_t(layers_) * kMoeSkewBuckets, 0);
  dup_.assign(size_t(layers_) * kMoeSkewBuckets, 0);
  dup_pairs_.assign(size_t(layers_) * kMoeSkewBuckets, 0);
  prev_ids_.assign(size_t(layers_) * used_, 0);
  overlap_.assign(size_t(layers_), 0);
  overlap_rows_.assign(size_t(layers_), 0);
  Rearm(stream);
  return true;
}

void QwenMoeSkewHost::FreeBuffers() noexcept {
  (void)hipFree(device_.hist);
  (void)hipFree(device_.slot_weights);
  (void)hipFree(device_.rows);
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
  const size_t hist_bytes =
      size_t(layers_) * kMoeSkewBuckets * experts_ * sizeof(std::int32_t);
  (void)hipMemcpyAsync(hist_.data(), device_.hist, hist_bytes,
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(
      slot_weights_.data(), device_.slot_weights,
      size_t(layers_) * kMoeSkewBuckets * used_ * sizeof(float),
      hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(
      rows_.data(), device_.rows,
      size_t(layers_) * kMoeSkewBuckets * sizeof(std::uint64_t),
      hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(dup_.data(), device_.dup,
                       size_t(layers_) * kMoeSkewBuckets * sizeof(std::int64_t),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(
      dup_pairs_.data(), device_.dup_pairs,
      size_t(layers_) * kMoeSkewBuckets * sizeof(std::uint64_t),
      hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(prev_ids_.data(), device_.prev_ids,
                       size_t(layers_) * used_ * sizeof(std::int32_t),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(overlap_.data(), device_.overlap,
                       size_t(layers_) * sizeof(std::int64_t),
                       hipMemcpyDeviceToHost, stream);
  (void)hipMemcpyAsync(overlap_rows_.data(), device_.overlap_rows,
                       size_t(layers_) * sizeof(std::uint64_t),
                       hipMemcpyDeviceToHost, stream);
  (void)hipStreamSynchronize(stream);
}

void QwenMoeSkewHost::AppendRecord() {
  ++ordinal_;
  auto layer_join = [&](auto&& render) {
    std::string out;
    for (size_t layer = 0; layer < layers_; ++layer) {
      out += render(layer);
      out += layer + 1 < layers_ ? "," : "";
    }
    return out;
  };
  auto bucket_join = [&](size_t layer, auto&& render) {
    std::string out;
    for (int bucket = 0; bucket < kMoeSkewBuckets; ++bucket) {
      out += render(layer, size_t(bucket));
      out += bucket + 1 < kMoeSkewBuckets ? "," : "";
    }
    return out;
  };
  std::string record =
      "{\"ordinal\":" + std::to_string(ordinal_) +
      ",\"layers\":" + std::to_string(layers_) +
      ",\"experts\":" + std::to_string(experts_) +
      ",\"used\":" + std::to_string(used_) + ",\"rows\":[" +
      layer_join([&](size_t layer) {
        return bucket_join(layer, [&](size_t l, size_t b) {
          return std::to_string(rows_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"hist\":[" + layer_join([&](size_t layer) {
        return bucket_join(layer, [&](size_t l, size_t b) {
          const size_t base = (l * kMoeSkewBuckets + b) * experts_;
          std::string row = "[";
          for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            row += std::to_string(hist_[base + expert]);
            row += expert + 1 < experts_ ? "," : "";
          }
          return row + "]";
        });
      }) +
      "],\"slot_weights\":[" + layer_join([&](size_t layer) {
        return bucket_join(layer, [&](size_t l, size_t b) {
          const size_t base = (l * kMoeSkewBuckets + b) * used_;
          std::string row = "[";
          for (std::uint32_t slot = 0; slot < used_; ++slot) {
            row += std::to_string(slot_weights_[base + slot]);
            row += slot + 1 < used_ ? "," : "";
          }
          return row + "]";
        });
      }) +
      "],\"dup\":[" + layer_join([&](size_t layer) {
        return bucket_join(layer, [&](size_t l, size_t b) {
          return std::to_string(dup_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"dup_pairs\":[" + layer_join([&](size_t layer) {
        return bucket_join(layer, [&](size_t l, size_t b) {
          return std::to_string(dup_pairs_[l * kMoeSkewBuckets + b]);
        });
      }) +
      "],\"overlap\":[" + layer_join([&](size_t layer) {
        return std::to_string(overlap_[layer]);
      }) +
      "],\"overlap_rows\":[" + layer_join([&](size_t layer) {
        return std::to_string(overlap_rows_[layer]);
      }) +
      "]}\n";
  records_.push_back(std::move(record));
}

void QwenMoeSkewHost::Rearm(hipStream_t stream) noexcept {
  const size_t hist_bytes =
      size_t(layers_) * kMoeSkewBuckets * experts_ * sizeof(std::int32_t);
  (void)hipMemsetAsync(device_.hist, 0, hist_bytes, stream);
  (void)hipMemsetAsync(
      device_.slot_weights, 0,
      size_t(layers_) * kMoeSkewBuckets * used_ * sizeof(float), stream);
  (void)hipMemsetAsync(
      device_.rows, 0,
      size_t(layers_) * kMoeSkewBuckets * sizeof(std::uint64_t), stream);
  (void)hipMemsetAsync(device_.dup, 0,
                       size_t(layers_) * kMoeSkewBuckets * sizeof(std::int64_t),
                       stream);
  (void)hipMemsetAsync(
      device_.dup_pairs, 0,
      size_t(layers_) * kMoeSkewBuckets * sizeof(std::uint64_t), stream);
  // prev_ids = -1: no previous token yet.
  (void)hipMemsetAsync(device_.prev_ids, 0xFF,
                       size_t(layers_) * used_ * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(device_.overlap, 0,
                       size_t(layers_) * sizeof(std::int64_t), stream);
  (void)hipMemsetAsync(device_.overlap_rows, 0,
                       size_t(layers_) * sizeof(std::uint64_t), stream);
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
                         rows_.size() * sizeof(std::uint64_t),
                         hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
    const bool any = std::any_of(rows_probe_.begin(), rows_probe_.end(),
                                 [](std::uint64_t rows) { return rows != 0; });
    if (any) {
      CopyOut(stream);
      AppendRecord();
    }
    Rearm(stream);
    if (records_.size() >= 32) {
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
    // Append; several executors (sessions) of one process share the file,
    // each record is self-contained.
    if (FILE* file = std::fopen(path_.c_str(), "a")) {
      for (const auto& record : records_) {
        std::fwrite(record.data(), 1, record.size(), file);
      }
      std::fclose(file);
    }
    records_.clear();
  } catch (...) {
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
