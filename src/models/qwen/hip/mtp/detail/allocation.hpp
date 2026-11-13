#ifndef GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::detail {

/// Device allocation entry points of the MTP head. Production keeps the
/// defaults; tests replace them to refuse a chosen allocation and to count
/// the calls.
struct DeviceAllocator {
  hipError_t (*allocate)(void** pointer, std::size_t bytes){
      [](void** pointer, std::size_t bytes) {
        return hipMalloc(pointer, bytes);
      }};
  hipError_t (*free)(void* pointer){
      [](void* pointer) { return hipFree(pointer); }};
};

[[nodiscard]] inline DeviceAllocator& MtpDeviceAllocator() noexcept {
  static DeviceAllocator allocator;
  return allocator;
}

[[nodiscard]] inline void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = MtpDeviceAllocator().allocate(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hipMalloc failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

inline void FreeDevice(void* pointer) noexcept {
  if (pointer != nullptr) {
    (void)MtpDeviceAllocator().free(pointer);
  }
}

/// The buffers of one lazily built workspace while it is being allocated.
/// They are freed together unless Publish() ran, so a refused allocation
/// leaves no part of the set for a later readiness check to find.
class PendingDeviceBuffers {
public:
  PendingDeviceBuffers() = default;
  PendingDeviceBuffers(const PendingDeviceBuffers&) = delete;
  PendingDeviceBuffers& operator=(const PendingDeviceBuffers&) = delete;
  ~PendingDeviceBuffers() {
    for (std::size_t index = 0; index < count_; ++index) {
      FreeDevice(buffers_[index]);
    }
  }

  template<typename T = void>
  [[nodiscard]] T* Allocate(std::size_t bytes) {
    if (count_ == kCapacity) {
      throw std::logic_error("MTP workspace has too many buffers");
    }
    void* const pointer = AllocateDevice(bytes);
    buffers_[count_++] = pointer;
    return static_cast<T*>(pointer);
  }

  /// The caller owns every buffer from here on.
  void Publish() noexcept { count_ = 0; }

private:
  static constexpr std::size_t kCapacity = 4;
  void* buffers_[kCapacity]{};
  std::size_t count_{0};
};

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
