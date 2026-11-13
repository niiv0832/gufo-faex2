// Allocation-failure recovery of the per-session image position state. The
// failures are injected through DeviceInputMemory; no real memory pressure.
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen/vision/rope.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/failing_allocator.hpp"

namespace {

using gufo::models::qwen::vision::DeviceInput;
using gufo::models::qwen::vision::DeviceInputMemory;
using gufo::models::qwen::vision::DeviceRope;
using gufo::models::qwen::vision::Prompt;
using gufo::models::qwen::vision::RopeLayout;
using gufo::test::FailingAllocator;

enum class Failure { kDevice, kHost };

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

DeviceInputMemory TestMemory() {
  return {.device = FailingAllocator::Device(),
          .host_allocate = &FailingAllocator::AllocateHost,
          .host_release = &FailingAllocator::ReleaseHost};
}

RopeLayout LayoutA() {
  return {.images = {{.offset = 2, .height = 3, .width = 4}}};
}
// Longer prefix than A, so restoring it over A must grow the device buffer.
RopeLayout LayoutB() {
  return {.images = {{.offset = 2, .height = 3, .width = 4},
                     {.offset = 20, .height = 8, .width = 8}}};
}
// Shorter prefix than A: reuses A's device buffer, still stages on the host.
RopeLayout LayoutC() {
  return {.images = {{.offset = 1, .height = 2, .width = 2}}};
}

/// What a kernel reading through the descriptor would see.
struct DeviceView {
  const std::int32_t* positions{nullptr};
  std::uint32_t prefix_length{0};
  std::int32_t delta{0};
  std::vector<std::int32_t> values;
};

bool SameContent(const DeviceView& left, const DeviceView& right) {
  return left.prefix_length == right.prefix_length &&
         left.delta == right.delta && left.values == right.values;
}

DeviceView Expected(const RopeLayout& layout) {
  DeviceView view;
  view.prefix_length = layout.PrefixLength();
  view.delta = layout.Delta();
  for (std::uint32_t token = 0; token < view.prefix_length; ++token) {
    const auto position = layout.Position(token);
    view.values.insert(view.values.end(), position.begin(), position.end());
  }
  return view;
}

DeviceView Read(const DeviceInput& input, const std::string& stage) {
  Expect(input.rope() != nullptr, stage + ": no device descriptor");
  DeviceRope rope;
  HIP_CHECK(
      hipMemcpy(&rope, input.rope(), sizeof(rope), hipMemcpyDeviceToHost));
  DeviceView view{.positions = rope.positions,
                  .prefix_length = rope.prefix_length,
                  .delta = rope.delta,
                  .values = {}};
  if (rope.prefix_length == 0)
    return view;
  // Never read through a descriptor that names released memory.
  Expect(FailingAllocator::Serial(rope.positions) != 0,
         stage + ": descriptor names released positions");
  view.values.resize(std::size_t{rope.prefix_length} * 3);
  HIP_CHECK(hipMemcpy(view.values.data(), rope.positions,
                      view.values.size() * sizeof(std::int32_t),
                      hipMemcpyDeviceToHost));
  return view;
}

void Arm(Failure failure, std::size_t successes) {
  if (failure == Failure::kDevice) {
    FailingAllocator::FailDeviceAfter(successes);
  } else {
    FailingAllocator::FailHostAfter(successes);
  }
}

template<typename Action>
void ExpectInjectedFailure(Action&& action, const std::string& stage) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  Expect(!FailingAllocator::Disarm(),
         stage + ": the armed allocation was never requested");
  Expect(failed, stage + ": allocation failure was swallowed");
}

std::string Name(Failure failure) {
  return failure == Failure::kDevice ? "device" : "host";
}

// A, then a failed larger B, then A again in the same object: the retry of A
// must see exactly what a cold A sees, through a descriptor that still names
// live memory.
void TestFailedGrowthKeepsPreviousLayout(hipStream_t stream, Failure failure,
                                         bool through_configure) {
  const std::string stage = "growth/" + Name(failure) +
                            (through_configure ? "/configure" : "/restore");
  const auto prompt_a = std::make_shared<Prompt>();
  prompt_a->rope = LayoutA();
  const auto prompt_b = std::make_shared<Prompt>();
  prompt_b->rope = LayoutB();
  const auto apply = [&](DeviceInput& input, bool larger) {
    if (through_configure) {
      input.Configure(larger ? prompt_b : prompt_a, nullptr, stream);
    } else {
      input.RestoreLayout(larger ? LayoutB() : LayoutA(), stream);
    }
  };

  DeviceInput cold(TestMemory());
  apply(cold, false);
  const DeviceView cold_a = Read(cold, stage + " cold A");
  Expect(SameContent(cold_a, Expected(LayoutA())),
         stage + ": cold A differs from the host oracle");

  DeviceInput input(TestMemory());
  apply(input, false);
  const DeviceView before = Read(input, stage + " A");
  const auto* descriptor = input.rope();
  const std::uint64_t positions_serial =
      FailingAllocator::Serial(before.positions);
  const std::size_t bytes = input.Bytes();

  Arm(failure, 0);
  ExpectInjectedFailure([&] { apply(input, true); }, stage);

  Expect(input.rope() == descriptor, stage + ": descriptor moved");
  Expect(input.layout() == LayoutA(), stage + ": layout changed by failure");
  const DeviceView after_failure = Read(input, stage + " after failure");
  Expect(input.Bytes() == bytes, stage + ": capacity changed by failure");
  Expect(
      after_failure.positions == before.positions &&
          FailingAllocator::Serial(after_failure.positions) == positions_serial,
      stage + ": positions of A were released by the failed restore");
  Expect(SameContent(after_failure, cold_a),
         stage + ": positions of A changed by the failed restore");

  apply(input, false);
  Expect(SameContent(Read(input, stage + " A retry"), cold_a),
         stage + ": A after a failed B differs from cold A");

  apply(input, true);
  const DeviceView grown = Read(input, stage + " B");
  Expect(SameContent(grown, Expected(LayoutB())),
         stage + ": B after recovery differs from the host oracle");
  Expect(FailingAllocator::Serial(before.positions) != positions_serial,
         stage + ": outgrown positions of A were not released");
  Expect(input.rope() == descriptor, stage + ": descriptor moved by growth");

  apply(input, false);
  Expect(SameContent(Read(input, stage + " A after B"), cold_a),
         stage + ": A after a successful B differs from cold A");
}

// A host staging failure while the device buffer is reused in place.
void TestFailedShrinkKeepsPreviousLayout(hipStream_t stream) {
  const std::string stage = "shrink/host";
  DeviceInput input(TestMemory());
  input.RestoreLayout(LayoutA(), stream);
  const DeviceView before = Read(input, stage + " A");

  Arm(Failure::kHost, 0);
  ExpectInjectedFailure([&] { input.RestoreLayout(LayoutC(), stream); }, stage);
  Expect(input.layout() == LayoutA(), stage + ": layout changed by failure");
  Expect(SameContent(Read(input, stage + " after failure"), before),
         stage + ": positions of A changed by the failed restore");

  input.RestoreLayout(LayoutC(), stream);
  Expect(SameContent(Read(input, stage + " C"), Expected(LayoutC())),
         stage + ": C after recovery differs from the host oracle");
}

// The very first restore fails at each of its allocations in turn.
void TestFailedFirstRestore(hipStream_t stream, Failure failure,
                            std::size_t successes) {
  const std::string stage =
      "first/" + Name(failure) + "/" + std::to_string(successes);
  const std::size_t live = FailingAllocator::live_device();
  {
    DeviceInput input(TestMemory());
    Arm(failure, successes);
    ExpectInjectedFailure([&] { input.RestoreLayout(LayoutA(), stream); },
                          stage);
    Expect(input.rope() == nullptr, stage + ": descriptor published");
    Expect(input.layout() == RopeLayout{}, stage + ": layout published");
    Expect(input.Bytes() == 0, stage + ": memory kept by a failed restore");
    Expect(FailingAllocator::live_device() == live,
           stage + ": device memory kept by a failed restore");

    input.RestoreLayout(LayoutA(), stream);
    Expect(SameContent(Read(input, stage + " A"), Expected(LayoutA())),
           stage + ": A after recovery differs from the host oracle");
    input.RestoreLayout(RopeLayout{}, stream);
    const DeviceView empty = Read(input, stage + " empty");
    Expect(empty.prefix_length == 0 && empty.delta == 0,
           stage + ": empty layout left image positions");
  }
  Expect(FailingAllocator::live_device() == live,
         stage + ": device memory leaked");
}

void Run() {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  // Every scenario runs even after an earlier one fails, so one run reports
  // all broken recovery paths.
  std::size_t failures = 0;
  const auto scenario = [&](auto&& body) {
    try {
      body();
    } catch (const std::exception& error) {
      (void)FailingAllocator::Disarm();
      ++failures;
      std::cerr << "FAILED " << error.what() << "\n";
    }
  };
  for (const Failure failure : {Failure::kDevice, Failure::kHost}) {
    for (const bool through_configure : {false, true}) {
      scenario([&] {
        TestFailedGrowthKeepsPreviousLayout(stream, failure, through_configure);
      });
    }
  }
  scenario([&] { TestFailedShrinkKeepsPreviousLayout(stream); });
  // Device allocations of a first restore: descriptor, then positions.
  scenario([&] { TestFailedFirstRestore(stream, Failure::kDevice, 0); });
  scenario([&] { TestFailedFirstRestore(stream, Failure::kDevice, 1); });
  scenario([&] { TestFailedFirstRestore(stream, Failure::kHost, 0); });
  HIP_CHECK(hipStreamDestroy(stream));
  Expect(failures == 0, std::to_string(failures) + " scenario(s) failed");

  Expect(FailingAllocator::live_device() == 0 &&
             FailingAllocator::device_allocations() ==
                 FailingAllocator::device_releases(),
         "device allocations and releases do not balance");
  Expect(
      FailingAllocator::host_allocations() == FailingAllocator::host_releases(),
      "host allocations and releases do not balance");
  std::cout << "device allocations=" << FailingAllocator::device_allocations()
            << " releases=" << FailingAllocator::device_releases()
            << " host allocations=" << FailingAllocator::host_allocations()
            << " releases=" << FailingAllocator::host_releases() << "\n";
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen vision device input test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  try {
    Run();
  } catch (const std::exception& error) {
    std::cerr << "Qwen vision device input test FAILED: " << error.what()
              << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "Qwen vision device input test passed.\n";
  return EXIT_SUCCESS;
#else
  std::cout << "HIP disabled, skipping Qwen vision device input test.\n";
  return 77;
#endif
}
