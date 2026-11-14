#ifndef GUFO_CORE_SNAPSHOT_STREAM_HPP_
#define GUFO_CORE_SNAPSHOT_STREAM_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace gufo::core {

// Sinks consume each borrowed span synchronously and must not retain it.
using PersistentSnapshotSink =
    std::function<void(std::span<const std::uint8_t>)>;
inline constexpr std::size_t kSnapshotStreamChunkBytes = 16U * 1024U * 1024U;

inline void StreamSnapshotBytes(std::span<const std::uint8_t> bytes,
                                const PersistentSnapshotSink& sink) {
  while (!bytes.empty()) {
    const auto count = std::min(bytes.size(), kSnapshotStreamChunkBytes);
    sink(bytes.first(count));
    bytes = bytes.subspan(count);
  }
}

}  // namespace gufo::core

#endif  // GUFO_CORE_SNAPSHOT_STREAM_HPP_
