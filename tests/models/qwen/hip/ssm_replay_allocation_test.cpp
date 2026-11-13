// Allocation-failure recovery of the SSM replay log. The failures are injected
// through QwenGpuArena::SetSsmReplayAllocator; no real memory pressure.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/failing_allocator.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

namespace {

using gufo::hip::QwenGpuArena;
using gufo::test::FailingAllocator;

constexpr std::uint32_t kContext = 32;
constexpr std::uint32_t kSavedContext = 4;
constexpr std::size_t kReplayBuffers = 4;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool HasNoLog(const QwenGpuArena& arena) {
  const auto capture = arena.GetSsmReplayCapture();
  return capture.qkv == nullptr && capture.alpha == nullptr &&
         capture.beta == nullptr && capture.enabled == nullptr;
}

// Fills one replay buffer to its full size and reads it back, so a buffer
// that is missing or shorter than the log layout cannot pass.
void CheckBuffer(float* device, std::size_t elements, float seed,
                 const std::string& name) {
  Expect(device != nullptr, name + ": buffer is missing");
  Expect(FailingAllocator::Serial(device) != 0,
         name + ": buffer is not a live allocation");
  std::vector<float> written(elements);
  for (std::size_t index = 0; index < elements; ++index)
    written[index] = seed + static_cast<float>(index % 251);
  HIP_CHECK(hipMemcpy(device, written.data(), elements * sizeof(float),
                      hipMemcpyHostToDevice));
  std::vector<float> read(elements);
  HIP_CHECK(hipMemcpy(read.data(), device, elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  Expect(read == written, name + ": buffer contents do not round-trip");
}

// A complete, armed replay capture over [kSavedContext, kSavedContext + 16).
void CheckWorkingCapture(QwenGpuArena& arena,
                         const gufo::core::ModelConfig& config,
                         bool expect_allocation, const std::string& stage) {
  arena.SaveState(kSavedContext);
  Expect(arena.BeginSsmReplayCapture() == expect_allocation,
         stage + ": unexpected replay allocation result");
  Expect(arena.IsSsmReplayCaptureActive(), stage + ": capture is not active");
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  const auto capture = arena.GetSsmReplayCapture();
  const std::size_t layer_slots =
      static_cast<std::size_t>(config.SsmLayerCount()) *
      gufo::hip::kSsmReplayCapacity;
  CheckBuffer(capture.qkv, layer_slots * config.SsmQkvSize(), 1.0F,
              stage + " qkv");
  CheckBuffer(capture.alpha, layer_slots * config.ssm_time_step_rank, 2.0F,
              stage + " alpha");
  CheckBuffer(capture.beta, layer_slots * config.ssm_time_step_rank, 3.0F,
              stage + " beta");
  Expect(capture.enabled != nullptr, stage + ": enable flag is missing");
  std::uint32_t enabled = 0;
  HIP_CHECK(hipMemcpy(&enabled, capture.enabled, sizeof(enabled),
                      hipMemcpyDeviceToHost));
  Expect(enabled != 0, stage + ": capture flag is not set on the device");

  Expect(!arena.CanReplaySsmPosition(kSavedContext),
         stage + ": empty capture is replayable");
  for (std::size_t offset = 0; offset < gufo::hip::kSsmReplayCapacity;
       ++offset) {
    arena.MarkSsmReplayPosition(
        static_cast<std::uint32_t>(kSavedContext + offset));
  }
  const auto last = static_cast<std::uint32_t>(
      kSavedContext + gufo::hip::kSsmReplayCapacity - 1);
  Expect(arena.CanReplaySsmPosition(kSavedContext) &&
             arena.CanReplaySsmPosition(last),
         stage + ": captured window is not replayable");
  Expect(arena.GetReplayQkv(0, kSavedContext) != nullptr &&
             arena.GetReplayAlpha(0, last) != nullptr &&
             arena.GetReplayBeta(0, last) != nullptr,
         stage + ": replay rows are missing");
  arena.RestoreState();
}

// The `failing`-th of the four allocations is refused. Nothing of the log may
// survive the failure, and the same arena must then capture normally, both
// when retried directly and after a Reset.
void TestFailedAllocation(const gufo::core::ModelConfig& config,
                          std::size_t failing, bool reset_before_retry) {
  const std::string stage = "allocation " + std::to_string(failing + 1) +
                            (reset_before_retry ? "/reset" : "/retry");
  const std::size_t live = FailingAllocator::live_device();
  {
    QwenGpuArena arena(config, kContext);
    arena.SetSsmReplayAllocator(FailingAllocator::Device());
    QwenGpuArena neighbour(config, kContext);
    neighbour.SetSsmReplayAllocator(FailingAllocator::Device());

    arena.SaveState(kSavedContext);
    FailingAllocator::FailDeviceAfter(failing);
    bool failed = false;
    try {
      (void)arena.BeginSsmReplayCapture();
    } catch (const std::exception&) {
      failed = true;
    }
    Expect(!FailingAllocator::Disarm(),
           stage + ": the armed allocation was never requested");
    Expect(failed, stage + ": allocation failure was swallowed");
    Expect(FailingAllocator::live_device() == live,
           stage + ": a partial replay log survived the failure");
    Expect(HasNoLog(arena), stage + ": a partial replay log is published");
    Expect(!arena.IsSsmReplayCaptureActive(),
           stage + ": capture is active without a log");
    Expect(!arena.CanReplaySsmPosition(kSavedContext),
           stage + ": replay is offered without a log");

    if (reset_before_retry) {
      arena.Reset();
      HIP_CHECK(hipStreamSynchronize(arena.stream));
      Expect(FailingAllocator::live_device() == live && HasNoLog(arena),
             stage + ": Reset left replay buffers behind");
    }
    CheckWorkingCapture(arena, config, true, stage + " recovered");
    Expect(FailingAllocator::live_device() == live + kReplayBuffers,
           stage + ": recovered log is not exactly four buffers");

    // Reuse of the same slot keeps the published log.
    arena.Reset();
    CheckWorkingCapture(arena, config, false, stage + " reused");
    Expect(FailingAllocator::live_device() == live + kReplayBuffers,
           stage + ": reuse reallocated the replay log");

    CheckWorkingCapture(neighbour, config, true, stage + " neighbour");
    Expect(FailingAllocator::live_device() == live + 2 * kReplayBuffers,
           stage + ": neighbour log is not exactly four buffers");
  }
  Expect(FailingAllocator::live_device() == live,
         stage + ": replay buffers leaked");
}

void Run() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  // Every scenario runs even after an earlier one fails, so one run reports
  // all broken recovery paths.
  std::size_t failures = 0;
  for (std::size_t failing = 0; failing < kReplayBuffers; ++failing) {
    for (const bool reset_before_retry : {false, true}) {
      try {
        TestFailedAllocation(config, failing, reset_before_retry);
      } catch (const std::exception& error) {
        (void)FailingAllocator::Disarm();
        ++failures;
        std::cerr << "FAILED " << error.what() << "\n";
      }
    }
  }
  Expect(failures == 0, std::to_string(failures) + " scenario(s) failed");
  Expect(FailingAllocator::live_device() == 0 &&
             FailingAllocator::device_allocations() ==
                 FailingAllocator::device_releases(),
         "replay allocations and releases do not balance");
  std::cout << "replay allocations=" << FailingAllocator::device_allocations()
            << " releases=" << FailingAllocator::device_releases() << "\n";
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen SSM replay allocation test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  try {
    Run();
  } catch (const std::exception& error) {
    std::cerr << "Qwen SSM replay allocation test FAILED: " << error.what()
              << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "Qwen SSM replay allocation test passed.\n";
  return EXIT_SUCCESS;
#else
  std::cout << "HIP disabled, skipping Qwen SSM replay allocation test.\n";
  return 77;
#endif
}
