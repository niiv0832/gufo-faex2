#ifndef GUFO_MODELS_QWEN_HIP_MOE_SKEW_HPP_
#define GUFO_MODELS_QWEN_HIP_MOE_SKEW_HPP_

// Optional GUFO_QWEN_MOE_SKEW=1 router telemetry (W0.4). When the flag is
// unset nothing is allocated and no launch happens: the arithmetic and the
// launch stream of the production path are bit for bit unchanged. When it is
// set, a tiny recording kernel runs after every MoeRouterTopK and accumulates
// fixed-address device buffers -- fixed addresses keep the buffers legal
// inside captured HIP graphs (W2.1).

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gufo::hip {

/// Row-count buckets of the per-layer histograms: exactly 1-4 rows (the MTP
/// verify widths), then 5-8 and more than 8 (large batched decodes and
/// row-wise prefill chunks).
inline constexpr int kMoeSkewBuckets = 6;
inline constexpr int kMoeSkewMaxLayers = 64;
inline constexpr int kMoeSkewMaxExperts = 512;
inline constexpr int kMoeSkewMaxUsed = 8;
/// The row-wise MoE path chunks at kMoeFfnScratchRows; wider calls are
/// lumped into the last bucket.
inline constexpr int kMoeSkewMaxRows = 64;

static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t),
              "HIP 64-bit atomicAdd requires 64-bit unsigned long long");

/// The fixed-address device buffers of one executor. Pointers only: the
/// owner is QwenMoeSkewHost. 64-bit atomic counters use `unsigned long long*`
/// directly so ROCm HIP `atomicAdd` resolves cleanly on LP64 Linux.
struct QwenMoeSkewDevice {
  /// Selected-expert counts per layer, row bucket and expert.
  std::int32_t* hist;  ///< [layers][buckets][experts]
  /// Routing weight accumulated per selection slot (slot 0 = top-1).
  float* slot_weights;  ///< [layers][buckets][used]
  /// Calls recorded per layer and bucket.
  unsigned long long* rows;  ///< [layers][buckets]
  /// Token rows recorded per layer and bucket.
  unsigned long long* tokens;  ///< [layers][buckets]
  /// Sum of distinct experts activated per call across all rows in the call.
  unsigned long long* unique_experts;  ///< [layers][buckets]
  /// Token rows whose 7th / 8th selected expert weight is below 1% (W5.3).
  unsigned long long* tail_lt1pct;  ///< [layers][buckets][2]
  /// Expert instances shared between two rows of one call.
  unsigned long long* dup;  ///< [layers][buckets]
  /// Row pairs compared for the dup counter.
  unsigned long long* dup_pairs;  ///< [layers][buckets]
  /// Selected ids of the previous single-row call of the layer.
  std::int32_t* prev_ids;  ///< [layers][used]
  /// Ids of a single-row call shared with the previous single-row call.
  unsigned long long* overlap;  ///< [layers]
  /// Single-row calls that had a predecessor counted into overlap.
  unsigned long long* overlap_rows;  ///< [layers]
  std::uint32_t layers;
  std::uint32_t experts;
  std::uint32_t used;
};

/// Records one router call of `rows` rows of `layer_index` into the buffers.
void LaunchMoeSkewRecord(const QwenMoeSkewDevice& skew,
                         std::uint32_t layer_index, std::size_t rows,
                         const std::int32_t* ids, const float* weights,
                         hipStream_t stream);

/// Unset or "0" disables the telemetry; "1" records to moe_skew.jsonl in the
/// working directory; any other non-empty value names the JSONL file.
[[nodiscard]] bool MoeSkewEnabled();
[[nodiscard]] std::string MoeSkewPath();

/// Host-side owner of the device buffers: allocation, per-request dumps and
/// the JSONL file. One executor owns one instance; every dumped record covers
/// exactly one request (the span between two executor resets).
class QwenMoeSkewHost {
public:
  QwenMoeSkewHost(std::uint32_t layers, std::uint32_t experts,
                  std::uint32_t used, std::string path, hipStream_t stream);
  ~QwenMoeSkewHost();
  QwenMoeSkewHost(const QwenMoeSkewHost&) = delete;
  QwenMoeSkewHost& operator=(const QwenMoeSkewHost&) = delete;

  [[nodiscard]] const QwenMoeSkewDevice* device() const noexcept {
    return &device_;
  }

  /// Dumps the buffers accumulated since the last reset as one JSONL record
  /// (nothing is recorded if no router call happened) and re-arms the device
  /// buffers. Never throws: reset paths are noexcept.
  void OnReset(hipStream_t stream) noexcept;

  /// Writes the records still held in memory under a process-wide file mutex.
  /// Called on each recorded reset and from the executor destructor while its
  /// stream is alive.
  void Flush() noexcept;

private:
  bool AllocBuffers(hipStream_t stream);
  void FreeBuffers() noexcept;
  void CopyOut(hipStream_t stream) noexcept;
  void AppendRecord();
  void Rearm(hipStream_t stream) noexcept;

  QwenMoeSkewDevice device_{};
  std::uint32_t layers_;
  std::uint32_t experts_;
  std::uint32_t used_;
  std::string path_;
  /// Host mirror of the device buffers, read back on every dump.
  std::vector<std::int32_t> hist_;
  std::vector<float> slot_weights_;
  std::vector<unsigned long long> rows_;
  std::vector<unsigned long long> tokens_;
  std::vector<unsigned long long> unique_experts_;
  std::vector<unsigned long long> tail_lt1pct_;
  std::vector<unsigned long long> dup_;
  std::vector<unsigned long long> dup_pairs_;
  std::vector<std::int32_t> prev_ids_;
  std::vector<unsigned long long> overlap_;
  std::vector<unsigned long long> overlap_rows_;
  /// Probe of the call counters: a reset with no recorded call only re-arms
  /// the neighbor ids instead of dumping an empty record.
  std::vector<unsigned long long> rows_probe_;
  std::vector<std::string> records_;
  std::uint64_t ordinal_ = 0;
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MOE_SKEW_HPP_
