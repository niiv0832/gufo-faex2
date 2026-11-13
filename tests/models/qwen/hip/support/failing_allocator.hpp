#ifndef GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_FAILING_ALLOCATOR_HPP_
#define GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_FAILING_ALLOCATOR_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <new>
#include <optional>

#include "src/core/hip/device_allocator.hpp"

namespace gufo::test {

/// Counting memory sources that refuse one chosen request without exhausting
/// real memory. The production seams take plain function pointers, so the
/// bookkeeping is process-wide; the tests using it are single-threaded.
class FailingAllocator {
public:
  [[nodiscard]] static hip::DeviceAllocator Device() {
    return {.allocate = &AllocateDevice, .release = &ReleaseDevice};
  }
  static void* AllocateHost(std::size_t bytes) {
    if (Refuse(host_failure_))
      throw std::bad_alloc();
    void* pointer = ::operator new(bytes);
    ++host_allocations_;
    return pointer;
  }
  static void ReleaseHost(void* pointer) {
    ++host_releases_;
    ::operator delete(pointer);
  }

  /// The request after `successes` further successful ones fails, once.
  static void FailDeviceAfter(std::size_t successes) {
    device_failure_ = successes;
  }
  static void FailHostAfter(std::size_t successes) {
    host_failure_ = successes;
  }
  /// True when an armed failure has not been consumed; disarms either way.
  static bool Disarm() {
    const bool pending =
        device_failure_.has_value() || host_failure_.has_value();
    device_failure_.reset();
    host_failure_.reset();
    return pending;
  }

  [[nodiscard]] static std::size_t device_allocations() {
    return device_allocations_;
  }
  [[nodiscard]] static std::size_t device_releases() {
    return device_releases_;
  }
  [[nodiscard]] static std::size_t host_allocations() {
    return host_allocations_;
  }
  [[nodiscard]] static std::size_t host_releases() { return host_releases_; }
  [[nodiscard]] static std::size_t live_device() { return live_.size(); }
  /// Identity of the live allocation starting at `pointer`; zero when the
  /// address is not currently allocated. A reused address gets a new serial.
  [[nodiscard]] static std::uint64_t Serial(const void* pointer) {
    const auto found = live_.find(pointer);
    return found == live_.end() ? 0 : found->second;
  }

private:
  static bool Refuse(std::optional<std::size_t>& countdown) {
    if (!countdown)
      return false;
    if (*countdown != 0) {
      --*countdown;
      return false;
    }
    countdown.reset();
    return true;
  }
  static hipError_t AllocateDevice(void** pointer, std::size_t bytes) {
    if (Refuse(device_failure_)) {
      *pointer = nullptr;
      return hipErrorOutOfMemory;
    }
    const hipError_t status = hipMalloc(pointer, bytes);
    if (status == hipSuccess)
      live_[*pointer] = ++device_allocations_;
    return status;
  }
  static hipError_t ReleaseDevice(void* pointer) {
    ++device_releases_;
    live_.erase(pointer);
    return hipFree(pointer);
  }

  static inline std::optional<std::size_t> device_failure_;
  static inline std::optional<std::size_t> host_failure_;
  static inline std::size_t device_allocations_{0};
  static inline std::size_t device_releases_{0};
  static inline std::size_t host_allocations_{0};
  static inline std::size_t host_releases_{0};
  static inline std::map<const void*, std::uint64_t> live_;
};

}  // namespace gufo::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_FAILING_ALLOCATOR_HPP_
