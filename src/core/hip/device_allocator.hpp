#ifndef GUFO_CORE_HIP_DEVICE_ALLOCATOR_HPP_
#define GUFO_CORE_HIP_DEVICE_ALLOCATOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::hip {

/// Source of device memory. Production keeps the defaults; allocation-failure
/// tests substitute functions that fail on demand.
struct DeviceAllocator {
  hipError_t (*allocate)(void** pointer, std::size_t bytes){
      [](void** pointer, std::size_t bytes) {
        return hipMalloc(pointer, bytes);
      }};
  hipError_t (*release)(void* pointer){
      [](void* pointer) { return hipFree(pointer); }};
};

/// One device allocation, released on scope exit unless Release() hands it to
/// its final owner. A group of these lets a caller publish several buffers
/// only after every one of them exists.
class DeviceAllocation {
public:
  DeviceAllocation(const DeviceAllocator& allocator, std::size_t bytes)
      : release_(allocator.release) {
    const hipError_t status = allocator.allocate(&pointer_, bytes);
    if (status != hipSuccess) {
      pointer_ = nullptr;
      throw std::runtime_error(std::string("HIP allocation failed: ") +
                               hipGetErrorString(status));
    }
  }
  ~DeviceAllocation() {
    if (pointer_ == nullptr)
      return;
    const hipError_t status = release_(pointer_);
    if (status != hipSuccess) {
      std::fprintf(stderr, "HIP cleanup failed: %s\n",
                   hipGetErrorString(status));
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  template<typename T>
  [[nodiscard]] T* Get() const noexcept {
    return static_cast<T*>(pointer_);
  }
  template<typename T>
  [[nodiscard]] T* Release() noexcept {
    return static_cast<T*>(std::exchange(pointer_, nullptr));
  }

private:
  hipError_t (*release_)(void*);
  void* pointer_{nullptr};
};

}  // namespace gufo::hip
#endif  // GUFO_CORE_HIP_DEVICE_ALLOCATOR_HPP_
