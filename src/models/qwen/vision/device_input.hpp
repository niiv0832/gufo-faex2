#ifndef GUFO_MODELS_QWEN_VISION_DEVICE_INPUT_HPP_
#define GUFO_MODELS_QWEN_VISION_DEVICE_INPUT_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "src/core/hip/device_allocator.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen/vision/rope.hpp"

namespace gufo::models::qwen::vision {

/// Memory sources of a DeviceInput. Production keeps the defaults;
/// allocation-failure tests substitute functions that fail on demand.
struct DeviceInputMemory {
  hip::DeviceAllocator device;
  void* (*host_allocate)(std::size_t bytes){
      [](std::size_t bytes) { return ::operator new(bytes); }};
  void (*host_release)(void* pointer){
      [](void* pointer) { ::operator delete(pointer); }};
};

/// Per-session image inputs. The stable device descriptor lets captured
/// decode graphs follow a new request without capturing its image pointers.
class DeviceInput {
public:
  DeviceInput() = default;
  explicit DeviceInput(const DeviceInputMemory& memory) : memory_(memory) {}
  ~DeviceInput();
  DeviceInput(const DeviceInput&) = delete;
  DeviceInput& operator=(const DeviceInput&) = delete;
  void Configure(std::shared_ptr<const Prompt> prompt,
                 std::shared_ptr<Encoder> encoder, hipStream_t stream);
  void SetCancellationCheck(Encoder::CancellationCheck check) {
    is_cancelled_ = std::move(check);
  }
  void RestoreLayout(const RopeLayout& layout, hipStream_t stream);
  void Inject(float* hidden, std::uint32_t position, std::uint32_t count,
              std::uint32_t width, std::uint32_t hc, hipStream_t stream);
  [[nodiscard]] const DeviceRope* rope() const noexcept { return descriptor_; }
  [[nodiscard]] const RopeLayout& layout() const noexcept { return layout_; }
  [[nodiscard]] std::size_t Bytes() const noexcept {
    std::size_t bytes = capacity_ * 3 * sizeof(std::int32_t) +
                        (descriptor_ == nullptr ? 0 : sizeof(DeviceRope));
    for (const auto& embedding : embeddings_) {
      if (embedding)
        bytes +=
            std::size_t{embedding->rows()} * embedding->width() * sizeof(float);
    }
    return bytes;
  }

private:
  DeviceInputMemory memory_;
  RopeLayout layout_;
  std::shared_ptr<const Prompt> prompt_;
  std::shared_ptr<Encoder> encoder_;
  std::vector<std::shared_ptr<const Encoder::Embedding>> embeddings_;
  DeviceRope* descriptor_{nullptr};
  std::int32_t* positions_{nullptr};
  std::size_t capacity_{0};
  // False only when a failed device update could not leave the descriptor in
  // a known state; RestoreLayout then rebuilds it whatever the layout.
  bool published_{true};
  Encoder::CancellationCheck is_cancelled_;
};

}  // namespace gufo::models::qwen::vision
#endif
