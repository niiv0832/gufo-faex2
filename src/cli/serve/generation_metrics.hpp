#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <ostream>
#include <span>
#include <sstream>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

// Process-wide `/metrics` values. Token counters advance per prefill chunk and
// per generated token, so scrapes see work while requests are still running.
namespace detail {
inline std::atomic<std::uint64_t>& DeviceLostTotal() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalGenTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Request totals below are added once per request when it finishes.
inline std::atomic<std::uint64_t>& TotalCachedPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& TotalPromptSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<double>& TotalGenSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<std::uint64_t>& MaxSequenceTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftRounds() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftAcceptedTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& LastPromptSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
inline std::atomic<double>& LastGenSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
// Schedulers add their own deltas, so a draining scheduler and its
// replacement can both contribute during a model reload.
inline std::atomic<std::int64_t>& RequestsProcessing() {
  static std::atomic<std::int64_t> count{0};
  return count;
}
inline std::atomic<std::int64_t>& RequestsDeferred() {
  static std::atomic<std::int64_t> count{0};
  return count;
}

inline constexpr std::array<std::size_t, 7> kPromptTokenBucketBounds{
    512, 2048, 8192, 16384, 32768, 65536, 131072};
inline constexpr std::array<double, 7> kCachedFractionBucketBounds{
    0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0};
inline constexpr std::array<const char*, 7> kCachedFractionBucketLabels{
    "0", "0.25", "0.5", "0.75", "0.9", "0.99", "1"};
inline constexpr std::array<std::size_t, 3> kActiveLongRequestThresholds{
    16384, 32768, 65536};

struct CacheMetricsSnapshot {
  std::array<std::uint64_t, 8> prompt_buckets{};
  std::uint64_t prompt_sum{0};
  std::array<std::uint64_t, 8> fraction_buckets{};
  double fraction_sum{0.0};
  std::uint64_t cache_miss{0};
  std::uint64_t cache_memory{0};
  std::uint64_t cache_disk{0};
};

struct CacheMetricsStore {
  std::mutex mutex;
  CacheMetricsSnapshot snapshot;
};

inline CacheMetricsStore& CacheMetrics() {
  static CacheMetricsStore store;
  return store;
}
}  // namespace detail

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
}

/// Adds one finished request to the request totals and speed gauges. The
/// scheduler calls this when a request ends, whether or not its client reads
/// the result; failed requests are not added.
inline void RecordRequestMetrics(const TextGenerationBackend::Result& result) {
  const auto add_seconds = [](std::atomic<double>& total, double ms) {
    double current = total.load(std::memory_order_relaxed);
    while (!total.compare_exchange_weak(current, current + ms / 1000.0,
                                        std::memory_order_relaxed)) {
    }
  };
  detail::TotalCachedPromptTokens().fetch_add(
      std::min(result.cached_prompt_tokens, result.prompt_tokens),
      std::memory_order_relaxed);
  add_seconds(detail::TotalPromptSeconds(), result.prefill_ms);
  add_seconds(detail::TotalGenSeconds(), result.decode_ms);
  const std::uint64_t sequence_tokens =
      result.prompt_tokens + result.completion_tokens;
  std::uint64_t max_tokens =
      detail::MaxSequenceTokens().load(std::memory_order_relaxed);
  while (max_tokens < sequence_tokens &&
         !detail::MaxSequenceTokens().compare_exchange_weak(
             max_tokens, sequence_tokens, std::memory_order_relaxed)) {
  }
  detail::TotalDraftRounds().fetch_add(result.draft_rounds,
                                       std::memory_order_relaxed);
  detail::TotalDraftTokens().fetch_add(result.draft_tokens,
                                       std::memory_order_relaxed);
  detail::TotalDraftAcceptedTokens().fetch_add(result.draft_accepted_tokens,
                                               std::memory_order_relaxed);
  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double tok_per_sec =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  if (prompt_per_second > 0.0) {
    detail::LastPromptSpeed().store(prompt_per_second,
                                    std::memory_order_relaxed);
  }
  if (tok_per_sec > 0.0) {
    detail::LastGenSpeed().store(tok_per_sec, std::memory_order_relaxed);
  }
}

/// Observes prompt length and cache reuse once per nonempty request that
/// completed cache lookup in the scheduler. Failures, pre-lookup cancellations,
/// and token-counting endpoints are excluded.
inline void RecordCacheRequestMetrics(
    const TextGenerationBackend::Result& result) {
  if (result.prompt_tokens == 0) {
    return;
  }
  std::size_t prompt_bin = detail::kPromptTokenBucketBounds.size();
  for (std::size_t i = 0; i < detail::kPromptTokenBucketBounds.size(); ++i) {
    if (result.prompt_tokens <= detail::kPromptTokenBucketBounds[i]) {
      prompt_bin = i;
      break;
    }
  }
  const std::size_t cached =
      std::min(result.cached_prompt_tokens, result.prompt_tokens);
  const double fraction =
      static_cast<double>(cached) / static_cast<double>(result.prompt_tokens);
  std::size_t fraction_bin = detail::kCachedFractionBucketBounds.size();
  for (std::size_t i = 0; i < detail::kCachedFractionBucketBounds.size(); ++i) {
    if (fraction <= detail::kCachedFractionBucketBounds[i]) {
      fraction_bin = i;
      break;
    }
  }

  auto& store = detail::CacheMetrics();
  const std::lock_guard<std::mutex> lock(store.mutex);
  ++store.snapshot.prompt_buckets[prompt_bin];
  store.snapshot.prompt_sum += result.prompt_tokens;
  ++store.snapshot.fraction_buckets[fraction_bin];
  store.snapshot.fraction_sum += fraction;
  if (result.cache_disk_hit) {
    ++store.snapshot.cache_disk;
  } else if (result.cache_hit) {
    ++store.snapshot.cache_memory;
  } else {
    ++store.snapshot.cache_miss;
  }
}

inline void WriteCacheRequestMetrics(std::ostream& out) {
  detail::CacheMetricsSnapshot snap;
  {
    auto& store = detail::CacheMetrics();
    const std::lock_guard<std::mutex> lock(store.mutex);
    snap = store.snapshot;
  }

  out << "# HELP gufo_request_prompt_tokens Prompt length in tokens for "
         "requests that completed cache lookup\n"
      << "# TYPE gufo_request_prompt_tokens histogram\n";
  std::uint64_t prompt_cum = 0;
  for (std::size_t i = 0; i < detail::kPromptTokenBucketBounds.size(); ++i) {
    prompt_cum += snap.prompt_buckets[i];
    out << "gufo_request_prompt_tokens_bucket{le=\""
        << detail::kPromptTokenBucketBounds[i] << "\"} " << prompt_cum << "\n";
  }
  prompt_cum += snap.prompt_buckets.back();
  out << "gufo_request_prompt_tokens_bucket{le=\"+Inf\"} " << prompt_cum << "\n"
      << "gufo_request_prompt_tokens_sum " << snap.prompt_sum << "\n"
      << "gufo_request_prompt_tokens_count " << prompt_cum << "\n";

  out << "# HELP gufo_request_cached_fraction Fraction of prompt tokens "
         "reused from cache for requests that completed cache lookup\n"
      << "# TYPE gufo_request_cached_fraction histogram\n";
  std::uint64_t frac_cum = 0;
  for (std::size_t i = 0; i < detail::kCachedFractionBucketLabels.size(); ++i) {
    frac_cum += snap.fraction_buckets[i];
    out << "gufo_request_cached_fraction_bucket{le=\""
        << detail::kCachedFractionBucketLabels[i] << "\"} " << frac_cum << "\n";
  }
  frac_cum += snap.fraction_buckets.back();
  out << "gufo_request_cached_fraction_bucket{le=\"+Inf\"} " << frac_cum << "\n"
      << "gufo_request_cached_fraction_sum " << snap.fraction_sum << "\n"
      << "gufo_request_cached_fraction_count " << frac_cum << "\n";

  out << "# HELP gufo_request_cache_total Completed-lookup requests by cache "
         "reuse source\n"
      << "# TYPE gufo_request_cache_total counter\n"
      << "gufo_request_cache_total{cache=\"miss\"} " << snap.cache_miss << "\n"
      << "gufo_request_cache_total{cache=\"memory\"} " << snap.cache_memory
      << "\n"
      << "gufo_request_cache_total{cache=\"disk\"} " << snap.cache_disk << "\n";
}

inline void WriteActiveLongRequestMetrics(
    std::ostream& out,
    std::span<const TextGenerationBackend::SessionState> sessions) {
  out << "# HELP gufo_active_long_requests Processing requests (including "
         "cache preparation and parked shared-prefix followers) with prompt "
         "tokens strictly above threshold\n"
      << "# TYPE gufo_active_long_requests gauge\n";
  for (const std::size_t threshold : detail::kActiveLongRequestThresholds) {
    std::size_t count = 0;
    for (const auto& session : sessions) {
      if (session.processing && session.prompt_tokens > threshold) {
        ++count;
      }
    }
    out << "gufo_active_long_requests{threshold_tokens=\"" << threshold
        << "\"} " << count << "\n";
  }
}

inline std::string GenerationLogDetails(
    const TextGenerationBackend::Result& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << "prompt_tokens=" << result.prompt_tokens
      << " prefill_tokens=" << result.prefill_tokens
      << " generated_tokens=" << result.completion_tokens << " finish="
      << (result.cancelled ? "cancelled"
          : result.finish_reason == TextGenerationBackend::FinishReason::kLength
              ? "length"
          : result.finish_reason ==
                  TextGenerationBackend::FinishReason::kStopSequence
              ? "stop_sequence"
              : "stop")
      << " cache="
      << (result.cache_disk_hit ? "disk"
          : result.cache_hit    ? "memory"
                                : "miss")
      << " cached_tokens=" << result.cached_prompt_tokens
      << " cache_restore_ms=" << result.cache_restore_ms
      << " queue_depth=" << result.queue_depth_at_submit
      << " resident_at_admission=" << result.resident_requests_at_admission
      << " queue_ms=" << result.queue_ms
      << " shared_prefix_wait_ms=" << result.shared_prefix_wait_ms
      << " ttft_ms=" << result.ttft_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result) << " decode_tps="
      << (result.decode_ms > 0
              ? 1000.0 * result.completion_tokens / result.decode_ms
              : 0.0)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens
      << " draft_rounds=" << result.draft_rounds;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens
        << " suffix_accepted=" << result.suffix_accepted_tokens
        << " suffix_proposed=" << result.suffix_draft_tokens;
  if (result.cache_snapshot_bytes > 0)
    out << " cache_snapshot_bytes=" << result.cache_snapshot_bytes;
  if (result.cache_disk_queued_bytes > 0)
    out << " cache_disk_queued_bytes=" << result.cache_disk_queued_bytes;
  if (!result.cache_hit && !result.cache_miss_reason.empty())
    out << " cache_miss_reason=" << result.cache_miss_reason
        << " common_prefix_tokens=" << result.cache_common_prefix_tokens
        << " nearest_checkpoint_tokens=" << result.cache_checkpoint_tokens;
  return out.str();
}

[[nodiscard]] inline bool CachePrefixLossIsWarn(
    const TextGenerationBackend::Result& result) noexcept {
  if (!result.cache_prompt_enabled ||
      result.cache_prefix_match.nearest_tokens == 0) {
    return false;
  }
  const auto& match = result.cache_prefix_match;
  const std::size_t lost =
      match.nearest_tokens > match.common_prefix_tokens
          ? match.nearest_tokens - match.common_prefix_tokens
          : 0;
  const bool input_changed = !match.input_matches;
  return (lost > 0 || input_changed) && result.prefill_tokens >= 8192;
}

/// Formats a bounded prefix-loss diagnostic line for `[cache]` without any
/// prompt text, token IDs, or guessed root causes.
[[nodiscard]] inline std::string CachePrefixLossLogDetails(
    const TextGenerationBackend::Result& result) {
  if (!result.cache_prompt_enabled ||
      result.cache_prefix_match.nearest_tokens == 0) {
    return {};
  }
  const auto& match = result.cache_prefix_match;
  const std::size_t common = match.common_prefix_tokens;
  const std::size_t nearest = match.nearest_tokens;
  const std::size_t lost = nearest > common ? nearest - common : 0;
  const std::size_t unreused = common > result.cached_prompt_tokens
                                   ? common - result.cached_prompt_tokens
                                   : 0;
  const bool input_changed = !match.input_matches;
  const bool warn = CachePrefixLossIsWarn(result);
  const bool info = !warn && common > 0 && (lost > 512 || unreused > 512);
  if (!warn && !info) {
    return {};
  }
  const std::string_view reason = input_changed ? "input_mismatch"
                                  : lost > 0    ? "prefix_diverged"
                                                : "prefix_unreused";
  std::ostringstream out;
  out << "event=cache_prefix_loss source=ram_live reason=" << reason
      << " common_prefix_tokens=" << common << " lost_tokens=" << lost
      << " nearest_tokens=" << nearest
      << " cached_tokens=" << result.cached_prompt_tokens
      << " unreused_prefix_tokens=" << unreused
      << " prefill_tokens=" << result.prefill_tokens;
  using Kind = TextPromptPosition::Kind;
  switch (result.cache_divergence.kind) {
    case Kind::kMessage:
      out << " divergence=message divergence_message_index="
          << result.cache_divergence.message_index
          << " divergence_token_offset="
          << result.cache_divergence.token_offset;
      break;
    case Kind::kBeforeMessages:
      out << " divergence=before_messages divergence_token_offset="
          << result.cache_divergence.token_offset;
      break;
    case Kind::kEnd:
      out << " divergence=end";
      break;
    case Kind::kUnknown:
      out << " divergence=unknown";
      break;
  }
  return out.str();
}

/// Timed prefill counts work actually executed; usage counts the full prompt.
inline json::Value GenerationTimings(
    const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  timings["prompt_n"] = result.prefill_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] =
      result.prefill_tokens > 0
          ? result.prefill_ms / static_cast<double>(result.prefill_tokens)
          : 0.0;
  timings["prompt_per_second"] = PrefillTokensPerSecond(result);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] =
      result.completion_tokens > 0
          ? result.decode_ms / static_cast<double>(result.completion_tokens)
          : 0.0;
  timings["predicted_per_second"] =
      result.decode_ms > 0.0 ? static_cast<double>(result.completion_tokens) *
                                   1000.0 / result.decode_ms
                             : 0.0;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  timings["draft_rounds"] = result.draft_rounds;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  return timings;
}

/// llama-server `prompt_progress` object for streamed `return_progress`.
inline json::Value PromptProgressJson(
    const TextGenerationBackend::PromptProgress& progress) {
  json::Value value = json::Value::object();
  value["total"] = progress.total;
  value["cache"] = progress.cache;
  value["processed"] = progress.processed;
  value["time_ms"] = progress.time_ms;
  return value;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_
