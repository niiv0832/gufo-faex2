#ifndef GUFO_TESTS_QWEN_SNAPSHOT_ALLOCATION_GUARD_HPP_
#define GUFO_TESTS_QWEN_SNAPSHOT_ALLOCATION_GUARD_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>

// Test-only observer of replaceable C++ allocations on the calling thread.
// This does not account for malloc/HIP/OS allocations or other threads.
namespace gufo::test::snapshot_alloc {
// Independent test limits; do not derive these from production chunk sizing.
inline constexpr std::size_t kChunkLimit = 16U * 1024U * 1024U;
inline constexpr std::size_t kLiveLimit = kChunkLimit + 4096U;

struct AllocationLedger {
  struct Entry {
    void* pointer{nullptr};
    std::size_t bytes{0};
  };
  std::array<Entry, 4096> entries{};
  std::size_t live_bytes{0};
  std::size_t peak_bytes{0};
  std::size_t largest_bytes{0};
  std::size_t allocations{0};
  std::size_t rejected_bytes{0};
  const char* rejection{nullptr};

  void Check(std::size_t bytes) {
    if (bytes > kChunkLimit) {
      Reject(bytes, "single-allocation");
    }
    if (bytes > kLiveLimit - live_bytes) {
      Reject(bytes, "aggregate-live-bytes");
    }
  }

  void Add(void* pointer, std::size_t bytes) {
    for (auto& entry : entries) {
      if (entry.pointer == nullptr) {
        entry = {pointer, bytes};
        live_bytes += bytes;
        peak_bytes = std::max(peak_bytes, live_bytes);
        largest_bytes = std::max(largest_bytes, bytes);
        ++allocations;
        return;
      }
    }
    Reject(bytes, "observer-capacity");
  }

  void Remove(void* pointer) noexcept {
    for (auto& entry : entries) {
      if (entry.pointer == pointer) {
        live_bytes -= entry.bytes;
        entry = {};
        return;
      }
    }
  }

  [[noreturn]] void Reject(std::size_t bytes, const char* reason) {
    rejected_bytes = bytes;
    rejection = reason;
    throw std::bad_alloc();
  }
};

inline thread_local AllocationLedger* active_ledger = nullptr;

class AllocationScope {
public:
  explicit AllocationScope(AllocationLedger& ledger) {
    if (active_ledger != nullptr)
      throw std::logic_error("allocation observer is already armed");
    active_ledger = &ledger;
  }
  ~AllocationScope() { active_ledger = nullptr; }
  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;
};

inline void* Allocate(std::size_t bytes, std::size_t alignment = 0) {
  if (active_ledger != nullptr)
    active_ledger->Check(bytes);
  void* pointer = nullptr;
  if (alignment == 0) {
    pointer = std::malloc(std::max(bytes, std::size_t{1}));
  } else {
    if ((alignment & (alignment - 1)) != 0)
      throw std::bad_alloc();
    // C++ callers (including LLVM during HIP startup) may request alignment
    // 1, 2 or 4. POSIX additionally requires a multiple of sizeof(void*).
    const auto posix_alignment = std::max(alignment, sizeof(void*));
    if (::posix_memalign(&pointer, posix_alignment,
                         std::max(bytes, std::size_t{1})) != 0)
      pointer = nullptr;
  }
  if (pointer == nullptr)
    throw std::bad_alloc();
  try {
    if (active_ledger != nullptr)
      active_ledger->Add(pointer, bytes);
  } catch (...) {
    std::free(pointer);
    throw;
  }
  return pointer;
}

inline void Deallocate(void* pointer) noexcept {
  if (pointer != nullptr && active_ledger != nullptr)
    active_ledger->Remove(pointer);
  std::free(pointer);
}

inline void CheckAllocatorAlignment() {
  for (const std::size_t alignment : {1U, 2U, 4U, 8U, 16U, 64U, 128U}) {
    void* pointer = ::operator new(32, std::align_val_t(alignment));
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
    ::operator delete(pointer, std::align_val_t(alignment));
    if (!aligned)
      throw std::runtime_error("test allocator violated a valid C++ alignment");
  }
}

}  // namespace gufo::test::snapshot_alloc

// Define in exactly one TU per standalone test executable, before its first
// include of this header. Never enable this in shared production libraries.
#if defined(GUFO_TEST_DEFINE_SNAPSHOT_ALLOCATION_OPERATORS)
void* operator new(std::size_t bytes) {
  return gufo::test::snapshot_alloc::Allocate(bytes);
}
void* operator new[](std::size_t bytes) {
  return gufo::test::snapshot_alloc::Allocate(bytes);
}
void* operator new(std::size_t bytes, std::align_val_t alignment) {
  return gufo::test::snapshot_alloc::Allocate(
      bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
  return gufo::test::snapshot_alloc::Allocate(
      bytes, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete[](void* pointer) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  gufo::test::snapshot_alloc::Deallocate(pointer);
}

#endif

#endif  // GUFO_TESTS_QWEN_SNAPSHOT_ALLOCATION_GUARD_HPP_
