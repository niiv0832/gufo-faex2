#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models::qwen::vision {
namespace {
void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

__global__ void InjectRows(float* hidden, const float* embedding,
                           std::uint32_t rows, std::uint32_t width,
                           std::uint32_t hc) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width * hc)
    return;
  const auto row = i / (width * hc);
  hidden[i] = embedding[row * width + i % width];
}
}  // namespace

DeviceInput::~DeviceInput() {
  if (descriptor_ != nullptr)
    (void)memory_.device.release(descriptor_);
  if (positions_ != nullptr)
    (void)memory_.device.release(positions_);
}

void DeviceInput::Configure(std::shared_ptr<const Prompt> prompt,
                            std::shared_ptr<Encoder> encoder,
                            hipStream_t stream) {
  if (prompt && !prompt->images.empty() && !encoder) {
    throw std::invalid_argument("image input has no vision encoder");
  }
  const bool reuse = prompt_ && prompt && !prompt->cache_identity.empty() &&
                     encoder_ == encoder &&
                     prompt_->cache_identity == prompt->cache_identity;
  // Everything that can fail runs before the request state is replaced, so a
  // failed Configure leaves the previous prompt with its own layout.
  if (prompt)
    embeddings_.reserve(prompt->images.size());
  RestoreLayout(prompt ? prompt->rope : RopeLayout{}, stream);
  prompt_ = std::move(prompt);
  encoder_ = std::move(encoder);
  if (!reuse)
    embeddings_.clear();
  if (prompt_)
    embeddings_.resize(prompt_->images.size());
}

void DeviceInput::RestoreLayout(const RopeLayout& layout, hipStream_t stream) {
  layout.Validate(std::numeric_limits<std::int32_t>::max());
  if (published_ && layout == layout_ &&
      (descriptor_ != nullptr || layout.images.empty()))
    return;

  // Prepare every new resource first. A failure up to the device writes below
  // leaves the published positions, layout and descriptor untouched.
  const auto length = layout.PrefixLength();
  const std::size_t bytes = std::size_t{length} * 3 * sizeof(std::int32_t);
  RopeLayout next = layout;
  std::optional<hip::DeviceAllocation> descriptor;
  if (descriptor_ == nullptr)
    descriptor.emplace(memory_.device, sizeof(DeviceRope));
  std::optional<hip::DeviceAllocation> grown;
  if (length > capacity_)
    grown.emplace(memory_.device, bytes);
  const std::unique_ptr<void, void (*)(void*)> staged(
      length == 0 ? nullptr : memory_.host_allocate(bytes),
      memory_.host_release);
  auto* const host = static_cast<std::int32_t*>(staged.get());
  for (std::uint32_t token = 0; token < length; ++token) {
    const auto position = layout.Position(token);
    std::copy(position.begin(), position.end(), host + std::size_t{token} * 3);
  }
  auto* const device_descriptor =
      descriptor ? descriptor->Get<DeviceRope>() : descriptor_;
  auto* const device_positions =
      grown ? grown->Get<std::int32_t>() : positions_;
  const DeviceRope value{device_positions, length, layout.Delta()};

  try {
    if (length != 0) {
      Check(hipMemcpyAsync(device_positions, host, bytes, hipMemcpyHostToDevice,
                           stream));
    }
    Check(hipMemcpyAsync(device_descriptor, &value, sizeof(value),
                         hipMemcpyHostToDevice, stream));
    // Also drains every kernel still reading the positions being replaced.
    Check(hipStreamSynchronize(stream));
  } catch (...) {
    // The device may hold a half-written state. Fall back to the empty layout,
    // whose descriptor reads no positions; if even that write fails, the next
    // call rebuilds everything whatever layout it is given.
    layout_ = RopeLayout{};
    const DeviceRope empty{};
    published_ = descriptor_ == nullptr ||
                 hipMemcpy(descriptor_, &empty, sizeof(empty),
                           hipMemcpyHostToDevice) == hipSuccess;
    throw;
  }

  if (descriptor)
    descriptor_ = descriptor->Release<DeviceRope>();
  std::int32_t* const outgrown = grown ? positions_ : nullptr;
  if (grown) {
    positions_ = grown->Release<std::int32_t>();
    capacity_ = length;
  }
  layout_ = std::move(next);
  published_ = true;
  if (outgrown != nullptr)
    Check(memory_.device.release(outgrown));
}

void DeviceInput::Inject(float* hidden, std::uint32_t position,
                         std::uint32_t count, std::uint32_t width,
                         std::uint32_t hc, hipStream_t stream) {
  if (!prompt_)
    return;
  for (std::size_t index = 0; index < prompt_->images.size(); ++index) {
    const auto& image = prompt_->images[index];
    const auto image_end =
        image.grid.offset + image.grid.height * image.grid.width;
    const auto begin = std::max(position, image.grid.offset);
    const auto end = std::min(position + count, image_end);
    if (begin >= end)
      continue;
    if (!embeddings_[index])
      embeddings_[index] = encoder_->Encode(image.pixels, {}, is_cancelled_);
    const auto& embedding = *embeddings_[index];
    if (embedding.width() != width ||
        embedding.rows() != image.grid.height * image.grid.width) {
      throw std::logic_error("vision embedding does not match language input");
    }
    const auto rows = end - begin;
    InjectRows<<<(rows * width * hc + 255) / 256, 256, 0, stream>>>(
        hidden + std::size_t{begin - position} * width * hc,
        embedding.data() + std::size_t{begin - image.grid.offset} * width, rows,
        width, hc);
    Check(hipGetLastError());
  }
}
}  // namespace gufo::models::qwen::vision
