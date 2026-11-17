#include "src/cli/serve/continuation_cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#include "src/cli/serve/logging.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct FakeState final : gufo::server::ContinuationState {
  FakeState(std::size_t state_id, std::vector<std::size_t>* invalidations)
      : id(state_id), invalidation_counts(invalidations) {}

  void Invalidate() noexcept override { ++invalidation_counts->at(id); }

  std::size_t id;
  std::size_t value{0};
  std::vector<std::size_t>* invalidation_counts;
};

struct FakeSnapshot final : gufo::server::ContinuationSnapshot {
  explicit FakeSnapshot(std::size_t value,
                        std::size_t payload_bytes = sizeof(std::size_t),
                        const gufo::server::ContinuationState* owner = nullptr)
      : value(value), payload_bytes(payload_bytes), owner(owner) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return payload_bytes;
  }
  [[nodiscard]] bool PrefersState(
      const gufo::server::ContinuationState& state) const noexcept override {
    return owner == &state;
  }

  std::size_t value;
  std::size_t payload_bytes;
  const gufo::server::ContinuationState* owner;
};

void TestBorrowedSnapshotPrefersAvailableOwner() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {.restore =
           [](gufo::server::ContinuationState& state,
              const gufo::server::ContinuationSnapshot& snapshot) {
             dynamic_cast<FakeState&>(state).value =
                 dynamic_cast<const FakeSnapshot&>(snapshot).value;
           },
       .capacity_bytes = [] { return 1024; },
       .on_event = {}});
  auto root = cache.Acquire(Tokens{1, 2, 3});
  auto* owner = &root.state();
  Expect(root.TryReserveSnapshot(sizeof(std::size_t), 3), "reserve root");
  root.Commit({1, 2, 3},
              std::make_unique<FakeSnapshot>(7, sizeof(std::size_t), owner));
  auto first = cache.Acquire(Tokens{1, 2, 3, 4});
  Expect(first.cache_hit() && &first.state() == owner &&
             dynamic_cast<FakeState&>(first.state()).value == 7,
         "restore prefers the owner over an older unused state");
  auto second = cache.Acquire(Tokens{1, 2, 3, 5});
  Expect(second.cache_hit() && &second.state() != owner &&
             dynamic_cast<FakeState&>(second.state()).value == 7,
         "a busy owner does not block a branch into another state");
  first.Invalidate();
  second.Invalidate();
}

void TestBorrowedSnapshotKeepsOwnersLiveFrontier() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {.restore =
           [](gufo::server::ContinuationState& state,
              const gufo::server::ContinuationSnapshot& snapshot) {
             dynamic_cast<FakeState&>(state).value =
                 dynamic_cast<const FakeSnapshot&>(snapshot).value;
           },
       .capacity_bytes = [] { return 1024; },
       .on_event = {}});
  // A conversation leaves its prompt checkpoint and, in the state that owns
  // the checkpoint's rows, its live frontier: prompt plus reply.
  auto conversation = cache.Acquire(Tokens{1, 2, 3});
  auto* owner = &conversation.state();
  dynamic_cast<FakeState&>(*owner).value = 9;
  Expect(conversation.TryReserveSnapshot(sizeof(std::size_t), 3),
         "reserve the prompt checkpoint");
  conversation.Commit(
      {1, 2, 3}, std::make_unique<FakeSnapshot>(7, sizeof(std::size_t), owner),
      {1, 2, 3, 4, 5});
  // A branch from the checkpoint takes the other, free state.
  auto branch = cache.Acquire(Tokens{1, 2, 3, 6});
  Expect(branch.cache_hit() && &branch.state() != owner &&
             dynamic_cast<FakeState&>(branch.state()).value == 7,
         "a branch restores into a free state, not the owner's live frontier");
  branch.Invalidate();
  // The conversation's next turn still continues from its live frontier.
  auto next = cache.Acquire(Tokens{1, 2, 3, 4, 5, 8});
  Expect(next.cache_hit() && next.cached_tokens() == 5 &&
             &next.state() == owner &&
             dynamic_cast<FakeState&>(next.state()).value == 9,
         "the next turn reuses the generated reply");
  next.Invalidate();
}

void TestBorrowedSnapshotRetryReplacesOwnersReply() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {.restore =
           [](gufo::server::ContinuationState& state,
              const gufo::server::ContinuationSnapshot& snapshot) {
             dynamic_cast<FakeState&>(state).value =
                 dynamic_cast<const FakeSnapshot&>(snapshot).value;
           },
       .capacity_bytes = [] { return 1024; },
       .on_event = {}});
  auto conversation = cache.Acquire(Tokens{1, 2, 3});
  auto* owner = &conversation.state();
  dynamic_cast<FakeState&>(*owner).value = 9;
  Expect(conversation.TryReserveSnapshot(sizeof(std::size_t), 3),
         "reserve the prompt checkpoint");
  conversation.Commit(
      {1, 2, 3}, std::make_unique<FakeSnapshot>(7, sizeof(std::size_t), owner),
      {1, 2, 3, 4, 5});
  // Retrying the prompt replaces the reply, so the owner's rows are reused
  // and the other state, which may hold another conversation, is kept.
  auto retry = cache.Acquire(Tokens{1, 2, 3});
  Expect(retry.cache_hit() && &retry.state() == owner &&
             dynamic_cast<FakeState&>(retry.state()).value == 7,
         "a retry restores into the owner whose reply it replaces");
  retry.Invalidate();
}

void TestColdMissThenExactExtensionHit() {
  std::vector<std::size_t> invalidations(1);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(1, [&] {
    return std::make_unique<FakeState>(next_id++, &invalidations);
  });

  const std::vector<gufo::server::ContinuationToken> first_prompt{1, 2, 3};
  {
    auto lease = cache.Acquire(first_prompt);
    Expect(static_cast<bool>(lease), "cold request acquires the slot");
    Expect(!lease.cache_hit(), "first request is a cache miss");
    Expect(lease.lookup().miss_reason == "no_checkpoint",
           "cold miss is distinguished from changed input");
    Expect(lease.cached_tokens() == 0, "cold request reuses no tokens");
    Expect(dynamic_cast<FakeState&>(lease.state()).id == 0,
           "lease exposes the opaque model state");
    lease.Commit({1, 2, 3, 4});
  }

  const std::vector<gufo::server::ContinuationToken> extension{1, 2, 3,
                                                               4, 5, 6};
  {
    auto lease = cache.Acquire(extension);
    Expect(lease.cache_hit(), "exact extension reuses the slot");
    Expect(lease.lookup().miss_reason.empty(), "hits carry no miss reason");
    Expect(lease.cached_tokens() == 4,
           "hit reports the complete retained prefix");
    Expect(dynamic_cast<FakeState&>(lease.state()).id == 0,
           "hit returns the same opaque state");
    Expect(invalidations[0] == 0, "hit does not invalidate retained state");
    lease.Commit(extension);
  }
}

void TestDivergenceInvalidatesOldState() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });

  {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    lease.Commit({1, 2, 3});
  }
  {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 9});
    Expect(!lease.cache_hit(), "divergent request is a miss");
    Expect(lease.lookup().miss_reason == "prefix_changed" &&
               lease.lookup().common_prefix_tokens == 1 &&
               lease.lookup().checkpoint_tokens == 3,
           "miss identifies the first changed token without exposing it");
    Expect(invalidations[0] == 1,
           "divergence invalidates the previous model state");
    lease.Commit({1, 9});
  }
}

void TestUncommittedLeaseIsInvalidated() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });

  {
    auto lease = cache.Acquire(std::vector<gufo::server::ContinuationToken>{7});
    Expect(static_cast<bool>(lease), "request acquires the slot");
  }
  Expect(invalidations[0] == 1,
         "abandoned request invalidates partially computed state");

  auto retry =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{7, 8});
  Expect(!retry.cache_hit(), "abandoned state is never reused");
  retry.Commit({7, 8});
}

void TestLongestAvailablePrefixWins() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(2, [&] {
    return std::make_unique<FakeState>(next_id++, &invalidations);
  });

  {
    auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    first.Commit({1, 2});
  }
  {
    auto second =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    second.Commit({9, 8, 7});
  }

  auto lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7, 6});
  Expect(lease.cache_hit(), "one available entry matches");
  Expect(lease.cached_tokens() == 3, "longest exact prefix is selected");
  Expect(dynamic_cast<FakeState&>(lease.state()).id == 1,
         "matching state is selected rather than an arbitrary slot");
  lease.Commit({9, 8, 7, 6});
}

void TestWaitingAcquireCanBeCancelled() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });
  auto held =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  std::atomic<bool> cancelled{true};

  auto cancelled_lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4},
                    [&] { return cancelled.load(); });
  Expect(!cancelled_lease, "cancelled waiter does not acquire a state");
  held.Commit({1, 2, 3});
}

void TestCachedPrefixTokensPeeksWithoutLeasing() {
  std::vector<std::size_t> invalidations(1);
  std::size_t next_id = 0;
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 1024; },
          .on_event = {},
      });
  Expect(cache.CachedPrefixTokens(Tokens{1, 2, 3}) == 0,
         "an empty cache has no reusable prefix");
  {
    auto root = cache.Acquire(Tokens{1, 2, 3});
    Expect(root.TryReserveSnapshot(sizeof(std::size_t), 3),
           "root snapshot reserves capacity");
    root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7));
  }
  Expect(cache.CachedPrefixTokens(Tokens{1, 2, 3, 4}) == 3,
         "a retained prefix of the prompt is reported");
  Expect(cache.CachedPrefixTokens(Tokens{1, 2}) == 0 &&
             cache.CachedPrefixTokens(Tokens{1, 9, 3, 4}) == 0,
         "shorter or diverging prompts do not reuse the checkpoint");
  const std::vector<std::uint8_t> image{5};
  Expect(cache.CachedPrefixTokens(Tokens{1, 2, 3, 4}, image) == 0,
         "a different input identity does not match");
  // Peeking leases nothing: the single state slot remains available.
  auto lease = cache.Acquire(Tokens{1, 2, 3, 4});
  Expect(lease.cache_hit() && lease.cached_tokens() == 3,
         "a peek does not consume the checkpoint or the state slot");
  lease.Invalidate();
}

void TestBranchPointOutlivesOlderTurnsUnderPressure() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  for (const bool branched : {false, true}) {
    std::vector<std::size_t> invalidations(1);
    std::size_t next_id = 0;
    // Room for four snapshots; the fifth evicts one.
    gufo::server::ContinuationCache cache(
        1,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        {
            .restore = [](gufo::server::ContinuationState&,
                          const gufo::server::ContinuationSnapshot&) {},
            .capacity_bytes = [] { return 4 * sizeof(std::size_t); },
            .on_event = {},
        },
        8);
    const auto retain = [&](const Tokens& tokens) {
      auto lease = cache.Acquire(tokens);
      Expect(lease.TryReserveSnapshot(sizeof(std::size_t), tokens.size()),
             "test snapshot is admitted");
      lease.Commit(tokens, std::make_unique<FakeSnapshot>(tokens.size()));
    };
    retain({9, 9, 9});        // An older, unrelated conversation.
    retain({1, 2, 3});        // The shared system prompt.
    retain({1, 2, 3, 4, 4});  // A conversation continuing from it.
    if (branched)
      retain({1, 2, 3, 5, 5});  // A second one diverging after it.
    else
      retain({7, 7});
    retain({8, 8, 8, 8});  // Pressure: one checkpoint must go.
    const auto prefix = cache.CachedPrefixTokens(Tokens{1, 2, 3, 6});
    const auto unrelated = cache.CachedPrefixTokens(Tokens{9, 9, 9, 1});
    Expect(branched ? prefix == 3 && unrelated == 0
                    : prefix == 0 && unrelated == 3,
           "a shared prefix outlives older checkpoints only once it branches");
  }
}

void TestLearnedBranchPointOutlivesItsOlderBranch() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  using gufo::server::SnapshotPurpose;
  // A chat bridge replays its last user message without the metadata it sent,
  // so each request diverges where the previous user turn starts. Its own new
  // boundary extends the learned point; neither a warm turn advancing from it
  // nor a cold turn publishing it may evict it before other families' copies.
  for (const bool warm : {true, false}) {
    std::vector<std::size_t> invalidations(1);
    gufo::server::ContinuationCache cache(
        1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
        {.restore =
             [](auto& state, const auto& snapshot) {
               dynamic_cast<FakeState&>(state).value =
                   dynamic_cast<const FakeSnapshot&>(snapshot).value;
             },
         .capacity_bytes = [] { return 4 * sizeof(std::size_t); },
         .on_event = {}},
        8);
    const auto retain = [&](const Tokens& tokens, std::size_t value,
                            SnapshotPurpose purpose) {
      auto lease = cache.Acquire(tokens);
      Expect(lease.TryReserveSnapshot(sizeof(std::size_t), tokens.size(), true,
                                      purpose),
             "initial checkpoint fits without eviction");
      lease.Commit(tokens, std::make_unique<FakeSnapshot>(value));
    };
    retain({9, 9, 9}, 9003, SnapshotPurpose::kContinuation);
    retain({7, 7, 7}, 7003, SnapshotPurpose::kContinuation);
    // The previous turn's boundary ends in the live message the next request
    // rewrites; the turn before it already diverged at the learned point.
    retain({1, 2, 3}, 1003, SnapshotPurpose::kBranchPoint);
    retain({1, 2, 3, 4, 4}, 2005, SnapshotPurpose::kContinuation);

    const Tokens incoming{1, 2, 3, 5, 5};
    auto turn = cache.Acquire(incoming, {}, {}, {}, warm);
    Expect(turn.cached_tokens() == (warm ? 3 : 0),
           "a warm turn restores the learned branch point");
    Expect(turn.TryReserveSnapshot(sizeof(std::size_t), incoming.size(), !warm,
                                   SnapshotPurpose::kContinuation, incoming),
           "the new boundary evicts another checkpoint");
    turn.Commit(incoming, std::make_unique<FakeSnapshot>(3005));

    auto next = cache.Acquire(Tokens{1, 2, 3, 6, 6});
    Expect(next.cached_tokens() == 3 &&
               dynamic_cast<FakeState&>(next.state()).value == 1003,
           "the learned branch point outlives its own family's new boundary");
    next.Invalidate();
    Expect(cache.CachedPrefixTokens(Tokens{9, 9, 9, 1}) == 0 &&
               cache.CachedPrefixTokens(Tokens{7, 7, 7, 1}) == 3,
           "the oldest other family's checkpoint yields instead");
  }
}

void TestDeeperLearnedBranchPointSupersedesShallower() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  using gufo::server::SnapshotPurpose;
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {.restore = [](gufo::server::ContinuationState&,
                     const gufo::server::ContinuationSnapshot&) {},
       .capacity_bytes = [] { return 4 * sizeof(std::size_t); },
       .on_event = {}},
      8);
  const auto retain = [&](const Tokens& tokens, SnapshotPurpose purpose) {
    auto lease = cache.Acquire(tokens);
    Expect(lease.TryReserveSnapshot(sizeof(std::size_t), tokens.size(), true,
                                    purpose),
           "test snapshot is admitted");
    lease.Commit(tokens, std::make_unique<FakeSnapshot>(tokens.size()));
  };
  retain({9, 9, 9}, SnapshotPurpose::kContinuation);
  retain({1, 2, 3}, SnapshotPurpose::kBranchPoint);
  retain({1, 2, 3, 4, 5}, SnapshotPurpose::kBranchPoint);
  retain({1, 2, 3, 4, 5, 6}, SnapshotPurpose::kContinuation);
  retain({8, 8, 8}, SnapshotPurpose::kContinuation);  // Pressure.
  Expect(cache.CachedPrefixTokens(Tokens{1, 2, 3, 7}) == 0 &&
             cache.CachedPrefixTokens(Tokens{1, 2, 3, 4, 5, 7}) == 5 &&
             cache.CachedPrefixTokens(Tokens{9, 9, 9, 1}) == 3,
         "only the deepest learned branch point of a family stays protected");
}

void TestRestoredSharedPrefixOutlivesFrozenFrontiers() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  using gufo::server::SnapshotPurpose;
  // Conversations share a system prompt whose grid checkpoint is history.
  // Once a second conversation restores it and diverges, it is a branch point:
  // freezing a continued frontier under byte pressure must evict an older
  // turn's checkpoint, not the prefix every new conversation restores.
  for (const bool restored : {false, true}) {
    std::vector<std::size_t> invalidations(2);
    std::size_t next_id = 0;
    gufo::server::ContinuationCache cache(
        2,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        {.restore = [](gufo::server::ContinuationState&,
                       const gufo::server::ContinuationSnapshot&) {},
         .capacity_bytes = [] { return 4 * sizeof(std::size_t); },
         .on_event = {}},
        8);
    const Tokens unrelated{9, 9, 9};
    auto other = cache.Acquire(unrelated);
    Expect(other.TryReserveSnapshot(sizeof(std::size_t), unrelated.size()),
           "the unrelated checkpoint fits");
    other.Commit(unrelated, std::make_unique<FakeSnapshot>(9));

    const Tokens shared{1, 2, 3};
    const Tokens first{1, 2, 3, 4, 4};
    auto a = cache.Acquire(first);
    Expect(a.TryReserveSnapshot(sizeof(std::size_t), shared.size(), true,
                                SnapshotPurpose::kHistory),
           "the shared grid checkpoint fits");
    a.PublishSnapshot(shared, std::make_unique<FakeSnapshot>(3), true);
    Expect(a.TryReserveSnapshot(sizeof(std::size_t), first.size(), false,
                                SnapshotPurpose::kContinuation, first),
           "the first prompt checkpoint fits");
    a.Commit(first, std::make_unique<FakeSnapshot>(5), {1, 2, 3, 4, 4, 8, 8});

    const Tokens second{1, 2, 3, 5, 5};
    auto b = cache.Acquire(second, {}, {}, {}, restored);
    Expect(b.cached_tokens() == (restored ? shared.size() : 0),
           "the second conversation restores the shared prefix when allowed");
    Expect(b.TryReserveSnapshot(sizeof(std::size_t), second.size(), false,
                                SnapshotPurpose::kContinuation, second),
           "the second prompt checkpoint fits");
    b.Commit(second, std::make_unique<FakeSnapshot>(6), {1, 2, 3, 5, 5, 8});

    // The first conversation continues from its live frontier and freezes it.
    const Tokens next{1, 2, 3, 4, 4, 8, 8, 6};
    auto turn = cache.Acquire(next);
    Expect(turn.cached_tokens() == 7, "the next turn continues the frontier");
    Expect(turn.TryReserveSnapshot(sizeof(std::size_t), next.size(), false,
                                   SnapshotPurpose::kContinuation, next),
           "the frozen frontier evicts one checkpoint");
    turn.Commit(next, std::make_unique<FakeSnapshot>(8));

    Expect(
        cache.CachedPrefixTokens(Tokens{1, 2, 3, 7, 7}) ==
            (restored ? shared.size() : 0),
        "a restored shared prefix outlives a frozen frontier under pressure");
    Expect(cache.CachedPrefixTokens(first) ==
               (restored ? shared.size() : first.size()),
           "the continued conversation's older checkpoint yields instead");
  }
}

void TestSnapshotCanBranchIntoTwoIndependentStateSlots() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                auto& fake = dynamic_cast<FakeState&>(state);
                const auto& saved = dynamic_cast<const FakeSnapshot&>(snapshot);
                fake.value = saved.value;
              },
          .capacity_bytes = [] { return 1024; },
          .on_event = {},
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    dynamic_cast<FakeState&>(root.state()).value = 7;
    Expect(root.TryReserveSnapshot(sizeof(std::size_t), 3),
           "root snapshot reserves aggregate capacity before allocation");
    root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7));
  }

  auto first =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4});
  auto second =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 5});
  Expect(first.cache_hit() && second.cache_hit(),
         "one snapshot can satisfy two simultaneous leases");
  auto& first_state = dynamic_cast<FakeState&>(first.state());
  auto& second_state = dynamic_cast<FakeState&>(second.state());
  Expect(first_state.id != second_state.id,
         "snapshot branches use different mutable state slots");
  Expect(first_state.value == 7 && second_state.value == 7,
         "both mutable states restore the root payload");

  first_state.value = 8;
  second_state.value = 9;
  Expect(first.TryReserveSnapshot(sizeof(std::size_t), 4),
         "first branch reserves snapshot capacity");
  Expect(second.TryReserveSnapshot(sizeof(std::size_t), 4),
         "second branch reserves snapshot capacity");
  first.Commit({1, 2, 3, 4}, std::make_unique<FakeSnapshot>(8));
  second.Commit({1, 2, 3, 5}, std::make_unique<FakeSnapshot>(9));

  auto root_again =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 6});
  Expect(root_again.cache_hit() && root_again.cached_tokens() == 3,
         "branch commits preserve the shared root snapshot");
  Expect(dynamic_cast<FakeState&>(root_again.state()).value == 7,
         "branch mutation never changes the immutable root payload");
  root_again.Invalidate();
}

void TestByteCapacityEvictsBeforeSnapshotAllocation() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 12; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    Expect(root.TryReserveSnapshot(8, 3),
           "initial snapshot fits the byte budget");
    Expect(root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7, 8)) == 8,
           "initial snapshot is retained");
  }
  Expect(cache.retained_snapshot_bytes() == 8,
         "retained bytes account the initial snapshot");

  auto replacement =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7});
  Expect(replacement.TryReserveSnapshot(12, 3),
         "reservation evicts stale bytes before snapshot allocation");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 12,
         "eviction transfers budget from retained to reserved bytes");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kRemoved &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 3,
         "byte-pressure removal reports sanitized reason and dimensions");
  Expect(replacement.Commit({9, 8, 7}, std::make_unique<FakeSnapshot>(9, 12)) ==
             12,
         "reserved replacement is retained");
  Expect(cache.retained_snapshot_bytes() == 12 &&
             cache.reserved_snapshot_bytes() == 0,
         "commit converts the reservation into exact retained bytes");
}

void TestConcurrentReservationsCannotOvercommitBudget() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 12; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  auto second = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(first.TryReserveSnapshot(8, 1),
         "first in-flight snapshot reserves bytes");
  Expect(!second.TryReserveSnapshot(8, 1),
         "second reservation is rejected instead of overcommitting");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 8,
         "only the admitted in-flight reservation is accounted");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kSkipped &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 1,
         "capacity refusal emits a sanitized skip event");

  second.Commit({2});
  first.Commit({1}, std::make_unique<FakeSnapshot>(1, 8));
  Expect(cache.retained_snapshot_bytes() == 8 &&
             cache.reserved_snapshot_bytes() == 0,
         "completed requests leave no leaked reservation");
}

void TestImpossibleReservationPreservesRetainedEntries() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 8; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    Expect(root.TryReserveSnapshot(8, 2), "root fills the byte budget");
    root.Commit({1, 2}, std::make_unique<FakeSnapshot>(7, 8));
  }
  {
    auto oversized =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    Expect(!oversized.TryReserveSnapshot(9, 1),
           "snapshot larger than the total budget is rejected");
    oversized.Commit({9});
  }

  Expect(cache.retained_snapshot_bytes() == 8,
         "impossible admission does not evict a useful retained entry");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kSkipped &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 9,
         "oversized refusal emits one skip and no removal");
  auto extension =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  Expect(extension.cache_hit() && extension.cached_tokens() == 2,
         "retained root remains reusable after oversized refusal");
  extension.Invalidate();
}

void TestAbandonedReservationIsReleased() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 8; },
          .on_event = {},
      });

  {
    auto abandoned =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    Expect(abandoned.TryReserveSnapshot(8, 1),
           "abandoned request owns the whole reservation");
  }
  Expect(cache.reserved_snapshot_bytes() == 0,
         "lease destruction releases its in-flight reservation");

  auto retry = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(retry.TryReserveSnapshot(8, 1),
         "released bytes are immediately available to another request");
  retry.Commit({2}, std::make_unique<FakeSnapshot>(2, 8));
}

void TestReservationMismatchSkipsRetentionWithoutFailingCommit() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 16; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  auto lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{4, 5});
  Expect(lease.TryReserveSnapshot(4, 2),
         "snapshot estimate reserves before allocation");
  Expect(lease.Commit({4, 5}, std::make_unique<FakeSnapshot>(1, 8)) == 0,
         "an underestimated snapshot is skipped without failing commit");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 0,
         "mismatch releases the complete reservation");
  Expect(
      events.size() == 1 &&
          events.front().action == SnapshotEventAction::kSkipped &&
          events.front().reason == SnapshotEventReason::kReservationMismatch &&
          events.front().snapshot_bytes == 8 && events.front().token_count == 2,
      "reservation mismatch is observable without prompt content");

  auto retry =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{4, 5, 6});
  Expect(!retry.cache_hit(), "skipped snapshot is a deterministic cache miss");
  retry.Invalidate();
}

void TestEntryReplacementLogsRemovedSnapshot() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 32; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto first =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    Expect(first.TryReserveSnapshot(8, 2), "first entry reserves bytes");
    first.Commit({1, 2}, std::make_unique<FakeSnapshot>(1, 8));
  }
  {
    auto second =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7});
    Expect(second.TryReserveSnapshot(8, 3), "replacement reserves bytes");
    second.Commit({9, 8, 7}, std::make_unique<FakeSnapshot>(2, 8));
  }

  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kRemoved &&
             events.front().reason == SnapshotEventReason::kEntryCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 2,
         "entry replacement logs the removed snapshot dimensions");
  Expect(cache.retained_snapshot_bytes() == 8,
         "entry replacement keeps exact aggregate accounting");
}

void TestRetentionPrefersExtraCopiesUnderPressure() {
  using gufo::server::ContinuationToken;
  using gufo::server::SnapshotPurpose;
  // With three records entry pressure binds; with eight records bytes bind.
  for (const std::size_t records : {3U, 8U}) {
    std::vector<std::size_t> invalidations(1);
    std::size_t created = 0;
    gufo::server::ContinuationCache cache(
        1,
        [&] { return std::make_unique<FakeState>(created++, &invalidations); },
        {.restore =
             [](auto& state, const auto& snapshot) {
               dynamic_cast<FakeState&>(state).value =
                   dynamic_cast<const FakeSnapshot&>(snapshot).value;
             },
         .capacity_bytes = [] { return 24; },
         .on_event = {}},
        records);
    auto save = [&](std::vector<ContinuationToken> tokens,
                    SnapshotPurpose purpose, std::size_t stable = 0) {
      auto lease = cache.Acquire(tokens, {}, {}, {}, true, stable);
      Expect(lease.TryReserveSnapshot(8, tokens.size(), false, purpose),
             "copy is admitted");
      lease.Commit(tokens, std::make_unique<FakeSnapshot>(tokens.front(), 8));
      Expect(cache.retained_snapshot_bytes() <= 24 &&
                 cache.reserved_snapshot_bytes() == 0,
             "retention stays bounded after every publication");
    };
    save({1, 2}, SnapshotPurpose::kContinuation);
    save({1, 2, 3}, SnapshotPurpose::kRetry, 2);
    save({8, 9}, SnapshotPurpose::kContinuation);
    save({6, 7}, SnapshotPurpose::kContinuation);
    for (const auto tag : {1U, 8U, 6U}) {
      auto returned = cache.Acquire(
          std::vector<ContinuationToken>{tag, tag == 1 ? 2U : tag + 1, 0});
      Expect(returned.cached_tokens() == 2 &&
                 dynamic_cast<FakeState&>(returned.state()).value == tag,
             "dropping a retry preserves independent conversation payloads");
      returned.Invalidate();
    }
    auto optional = cache.Acquire(std::vector<ContinuationToken>{4, 5});
    Expect(!optional.TryReserveSnapshot(8, 2, true, SnapshotPurpose::kHistory),
           "an optional copy cannot evict a last useful continuation");
    optional.Invalidate();
    Expect(cache.retained_snapshot_bytes() == 24 && created == 1 &&
               cache.entry_capacity() == records,
           "checkpoint records allocate no extra execution states");

    const std::vector<ContinuationToken> next{1, 2, 9};
    auto advancing = cache.Acquire(next);
    Expect(advancing.TryReserveSnapshot(8, 3, false,
                                        SnapshotPurpose::kContinuation, next),
           "a new boundary can replace its own old boundary under pressure");
    advancing.Commit(next, std::make_unique<FakeSnapshot>(1, 8));
    for (const auto& tokens : {std::vector<ContinuationToken>{1, 2, 9, 0},
                               std::vector<ContinuationToken>{8, 9, 0},
                               std::vector<ContinuationToken>{6, 7, 0}}) {
      auto returned = cache.Acquire(tokens);
      Expect(returned.cached_tokens() == tokens.size() - 1,
             "advancing one conversation does not evict another family's last "
             "checkpoint");
      returned.Invalidate();
    }
  }
}

void TestOptionalPublicationRechecksRecordPressure() {
  using gufo::server::ContinuationToken;
  using gufo::server::SnapshotPurpose;
  std::vector<std::size_t> invalidations(2);
  std::size_t created = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(created++, &invalidations); },
      {.restore = [](auto&, const auto&) {},
       .capacity_bytes = [] { return 64; },
       .on_event = [&](const auto& event) { events.push_back(event); }},
      2);
  auto initial = cache.Acquire(std::vector<ContinuationToken>{1});
  Expect(initial.TryReserveSnapshot(8, 1), "first boundary reserves bytes");
  initial.Commit({1}, std::make_unique<FakeSnapshot>(1, 8));

  auto pending = cache.Acquire(std::vector<ContinuationToken>{9});
  Expect(pending.TryReserveSnapshot(8, 1, true, SnapshotPurpose::kHistory),
         "optional capture sees a free record before a peer publishes");
  auto peer = cache.Acquire(std::vector<ContinuationToken>{8});
  Expect(peer.TryReserveSnapshot(8, 1),
         "peer capture fits aggregate reservations");
  peer.Commit({8}, std::make_unique<FakeSnapshot>(8, 8));
  Expect(pending.Commit({9}, std::make_unique<FakeSnapshot>(9, 8)) == 0,
         "optional publication refuses to evict a last copy after concurrent "
         "admission");
  Expect(cache.retained_snapshot_bytes() == 16 &&
             cache.reserved_snapshot_bytes() == 0 && events.size() == 1 &&
             events.front().reason ==
                 gufo::server::SnapshotEventReason::kEntryCapacity,
         "skipped publication releases its reservation and reports record "
         "pressure");
  for (const auto tag : {1U, 8U}) {
    auto returned = cache.Acquire(std::vector<ContinuationToken>{tag, 0});
    Expect(
        returned.cache_hit() && returned.cached_tokens() == 1,
        "concurrent optional publication preserves both continuation families");
    returned.Invalidate();
  }
}

void TestNewConversationReplacesItsOwnHistoryFirst() {
  using gufo::server::ContinuationToken;
  using gufo::server::SnapshotPurpose;
  // Exercise record pressure and byte pressure independently.
  for (const std::size_t records : {4U, 8U}) {
    std::vector<std::size_t> invalidations(1);
    gufo::server::ContinuationCache cache(
        1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
        {.restore = [](auto&, const auto&) {},
         .capacity_bytes = [=] { return records == 4 ? 1024U : 32U; },
         .on_event = {}},
        records);
    const std::vector<ContinuationToken> existing{1, 2};
    auto first = cache.Acquire(existing);
    Expect(first.TryReserveSnapshot(8, existing.size()),
           "the existing conversation fits");
    first.Commit(existing, std::make_unique<FakeSnapshot>(1, 8));

    const std::vector<ContinuationToken> prompt{9, 8, 7, 6};
    auto incoming = cache.Acquire(prompt);
    for (std::size_t count = 1; count != prompt.size(); ++count) {
      std::vector<ContinuationToken> prefix(prompt.begin(),
                                            prompt.begin() + count);
      Expect(incoming.TryReserveSnapshot(8, count, true,
                                         SnapshotPurpose::kHistory),
             "the incoming conversation's intermediate history fits");
      incoming.PublishSnapshot(prefix, std::make_unique<FakeSnapshot>(9, 8),
                               true);
    }
    Expect(incoming.TryReserveSnapshot(8, prompt.size(), false,
                                       SnapshotPurpose::kContinuation, prompt),
           "the new final checkpoint can replace redundant history");
    Expect(incoming.HasSnapshotFor(existing),
           "byte admission preserves another conversation's last checkpoint");
    incoming.Commit(prompt, std::make_unique<FakeSnapshot>(9, 8));

    auto previous = cache.Acquire(existing);
    Expect(previous.cached_tokens() == existing.size(),
           "publication preserves another conversation's last checkpoint");
    previous.Invalidate();
    auto next = cache.Acquire(prompt);
    Expect(next.cached_tokens() == prompt.size(),
           "the incoming conversation retains its new complete checkpoint");
    next.Invalidate();
  }
}

void TestRetryDoesNotDisplaceEarlierHistory() {
  using gufo::server::ContinuationToken;
  using gufo::server::SnapshotPurpose;
  for (const std::size_t records : {3U, 8U}) {
    std::vector<std::size_t> invalidations(1);
    gufo::server::ContinuationCache cache(
        1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
        {.restore = [](auto&, const auto&) {},
         .capacity_bytes = [] { return 24; },
         .on_event = {}},
        records);
    const auto save = [&](std::vector<ContinuationToken> tokens,
                          SnapshotPurpose purpose) {
      auto lease = cache.Acquire(tokens);
      Expect(lease.TryReserveSnapshot(8, tokens.size(), false, purpose),
             "initial checkpoints fit");
      lease.Commit(tokens, std::make_unique<FakeSnapshot>(1, 8));
    };
    save({1}, SnapshotPurpose::kHistory);
    save({1, 2}, SnapshotPurpose::kContinuation);
    save({8}, SnapshotPurpose::kContinuation);
    auto retry = cache.Acquire(std::vector<ContinuationToken>{1, 2, 3});
    Expect(!retry.TryReserveSnapshot(8, 3, true, SnapshotPurpose::kRetry),
           "an exact retry cannot discard the earlier edit checkpoint");
    retry.Invalidate();
    auto edited = cache.Acquire(std::vector<ContinuationToken>{1, 9});
    Expect(edited.cached_tokens() == 1,
           "earlier history remains reusable after skipped retry admission");
    edited.Invalidate();
  }
  std::vector<std::size_t> invalidations(2);
  std::size_t created = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(created++, &invalidations); },
      {.restore = [](auto&, const auto&) {},
       .capacity_bytes = [] { return 64; },
       .on_event = {}},
      3);
  for (const auto& tokens : {std::vector<ContinuationToken>{1}, {1, 2}}) {
    auto lease = cache.Acquire(tokens);
    Expect(lease.TryReserveSnapshot(8, tokens.size(), false,
                                    tokens.size() == 1
                                        ? SnapshotPurpose::kHistory
                                        : SnapshotPurpose::kContinuation),
           "history and stable boundary fit before concurrent admission");
    lease.Commit(tokens, std::make_unique<FakeSnapshot>(1, 8));
  }
  auto retry = cache.Acquire(std::vector<ContinuationToken>{1, 2, 3});
  Expect(retry.TryReserveSnapshot(8, 3, true, SnapshotPurpose::kRetry),
         "retry sees a free record before a peer publishes");
  auto peer = cache.Acquire(std::vector<ContinuationToken>{8});
  Expect(peer.TryReserveSnapshot(8, 1), "peer fits aggregate reservations");
  peer.Commit({8}, std::make_unique<FakeSnapshot>(8, 8));
  Expect(
      retry.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(1, 8)) == 0,
      "retry publication rechecks history priority after concurrent admission");
  auto edited = cache.Acquire(std::vector<ContinuationToken>{1, 9});
  Expect(
      edited.cached_tokens() == 1 && cache.reserved_snapshot_bytes() == 0,
      "concurrent retry publication preserves earlier history and accounting");
  edited.Invalidate();
}

void TestEditedTailReplacementPreservesSharedCheckpoints() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 64; },
          .on_event = {},
      },
      3);
  for (const std::vector<gufo::server::ContinuationToken>& prompt :
       {std::vector<gufo::server::ContinuationToken>{1}, {1, 2}, {1, 2, 3}}) {
    auto lease = cache.Acquire(prompt);
    Expect(lease.TryReserveSnapshot(8, prompt.size()),
           "shared prefix and original tail fit the byte budget");
    lease.Commit(prompt, std::make_unique<FakeSnapshot>(prompt.size(), 8));
  }
  {
    auto edited =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 9});
    Expect(edited.cached_tokens() == 2, "edited tail reuses the shared prefix");
    Expect(edited.TryReserveSnapshot(8, 3), "new tail fits the byte budget");
    edited.Commit({1, 2, 9}, std::make_unique<FakeSnapshot>(3, 8));
  }
  auto earlier =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 8});
  Expect(earlier.cached_tokens() == 1,
         "entry pressure replaces the incompatible tail, not the older prefix");
  earlier.Invalidate();
  Expect(cache.retained_snapshot_bytes() == 24,
         "tail replacement preserves exact byte accounting");
}

void TestReplacedSourceDoesNotEvictAnotherBranchTail() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 64; },
          .on_event = {},
      },
      3);
  const auto save = [&](std::vector<gufo::server::ContinuationToken> prompt) {
    auto lease = cache.Acquire(prompt);
    Expect(lease.TryReserveSnapshot(8, prompt.size()),
           "snapshot fits byte budget");
    lease.Commit(prompt, std::make_unique<FakeSnapshot>(prompt.size(), 8));
  };
  save({1});
  save({8});
  save({8, 2});

  auto pending =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 3});
  Expect(pending.cached_tokens() == 1, "pending branch reuses its own root");
  // Another execution slot makes the pending request's source the oldest
  // entry, then replaces it. The lease still has the original model state.
  for (const std::vector<gufo::server::ContinuationToken>& prompt :
       {std::vector<gufo::server::ContinuationToken>{8}, {8, 2}}) {
    auto touch = cache.Acquire(prompt);
    touch.Commit({});
  }
  save({7});
  save({7, 5});
  Expect(pending.TryReserveSnapshot(8, 2),
         "pending branch can retain its tail");
  pending.Commit({1, 3}, std::make_unique<FakeSnapshot>(2, 8));

  auto other =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{7, 5, 6});
  Expect(other.cached_tokens() == 2,
         "a replaced source index cannot identify another conversation's tail");
  other.Invalidate();
}

// The summarised cache lines stay at INFO/WARN; which candidate was rejected,
// and by which guard, is only worth the bytes when the operator asked for it.
void TestCacheCandidateDetailIsDebugTierOnly() {
  const auto run = [](gufo::server::LogLevel level) {
    std::vector<std::size_t> invalidations(1);
    std::size_t next_id = 0;
    const auto previous_level = gufo::server::Logger::Level();
    gufo::server::Logger::SetLevel(level);
    std::ostringstream output;
    auto* previous_sink = std::clog.rdbuf(output.rdbuf());
    {
      gufo::server::ContinuationCache cache(1, [&] {
        return std::make_unique<FakeState>(next_id++, &invalidations);
      });
      const std::vector<gufo::server::ContinuationToken> retained{1, 2, 3};
      const std::vector<gufo::server::ContinuationToken> diverged{9, 8, 7};
      {
        auto lease = cache.Acquire(retained);
        lease.Commit(retained);
      }
      auto other = cache.Acquire(diverged);
      Expect(!other.cache_hit(), "a different prompt cannot reuse the slot");
    }
    std::clog.rdbuf(previous_sink);
    gufo::server::Logger::SetLevel(previous_level);
    return output.str();
  };

  const auto quiet = run(gufo::server::LogLevel::kInfo);
  Expect(quiet.find("event=candidate_skip") == std::string::npos,
         "candidate detail is silent at the default level");
  Expect(quiet.find("event=capture_evicts") == std::string::npos,
         "eviction detail is silent at the default level");

  const auto verbose = run(gufo::server::LogLevel::kDebug);
  Expect(verbose.find("event=candidate_skip index=0 tokens=3 prompt_tokens=3 "
                      "reason=token_prefix") != std::string::npos,
         "debug tier names the rejected candidate and its failing guard");
  Expect(verbose.find("event=capture_evicts index=0 tokens=3") !=
             std::string::npos,
         "debug tier names the checkpoint a capture overwrites");
}

// A waiter that finds every slot busy still reports why each candidate was
// rejected: contention is exactly when that question gets asked, and the
// records used to die with the poll iteration that collected them.
void TestWaitingAcquireReportsCandidateSkips() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });
  {
    auto seed =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    seed.Commit({1, 2, 3});
  }
  auto busy =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4});
  Expect(busy.cache_hit(), "the sole slot is leased to a matching prompt");
  // The lease clears the saved tokens in the live (non-snapshot) mode, so the
  // waiter records the busy slot with zero checkpoint tokens.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{200};
  const auto previous_level = gufo::server::Logger::Level();
  gufo::server::Logger::SetLevel(gufo::server::LogLevel::kDebug);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    auto waiter = cache.Acquire(
        std::vector<gufo::server::ContinuationToken>{9, 8, 7},
        [&] { return std::chrono::steady_clock::now() >= deadline; });
    Expect(!waiter, "the cancelled waiter never acquires the busy slot");
  }
  std::clog.rdbuf(previous_sink);
  gufo::server::Logger::SetLevel(previous_level);
  const std::string log = output.str();
  Expect(log.find("event=candidate_skip index=0 tokens=0 prompt_tokens=3 "
                  "reason=unavailable") != std::string::npos,
         "a waiter names the guard that rejected the busy candidate");
  std::size_t bursts = 0;
  for (std::size_t at = log.find("event=candidate_skip");
       at != std::string::npos; at = log.find("event=candidate_skip", at + 1)) {
    ++bursts;
  }
  Expect(bursts == 1, "a long wait reports its first record set once");
}

}  // namespace

void TestImageIdentityIsolation() {
  for (const bool snapshot_mode : {false, true}) {
    std::vector<std::size_t> invalidations(2);
    std::size_t next_id = 0;
    gufo::server::ContinuationCache::SnapshotSupport support;
    if (snapshot_mode) {
      support.capacity_bytes = [] { return std::size_t{1024}; };
      support.restore = [](auto& state, const auto& snapshot) {
        dynamic_cast<FakeState&>(state).value =
            dynamic_cast<const FakeSnapshot&>(snapshot).value;
      };
    }
    gufo::server::ContinuationCache cache(
        2,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        std::move(support));
    const std::vector<gufo::server::ContinuationToken> prompt{1, 248056, 3};
    const std::vector<std::uint8_t> a{1, 2}, b{1, 3};
    const auto save = [&](auto identity, std::size_t value) {
      auto lease = cache.Acquire(prompt, {}, identity);
      Expect(!lease.cache_hit(), "different image is a cold input");
      dynamic_cast<FakeState&>(lease.state()).value = value;
      if (snapshot_mode) {
        Expect(lease.TryReserveSnapshot(sizeof(std::size_t), prompt.size()),
               "snapshot admitted");
        lease.Commit(prompt, std::make_unique<FakeSnapshot>(value));
      } else {
        lease.Commit(prompt);
      }
    };
    save(a, 10);
    save(b, 20);
    for (const auto& [identity, value] :
         std::vector<std::pair<std::vector<std::uint8_t>, std::size_t>>{
             {a, 10}, {b, 20}}) {
      auto lease = cache.Acquire(prompt, {}, identity);
      Expect(lease.cache_hit(), "same image reuses its own prefix");
      Expect(dynamic_cast<FakeState&>(lease.state()).value == value,
             "image state cannot cross requests");
      if (!snapshot_mode)
        lease.Commit(prompt);
    }
    auto text = cache.Acquire(prompt);
    Expect(!text.cache_hit(),
           "literal image-pad text cannot reuse image state");
  }
}

void TestAppendedImagesReuseOnlyCompatiblePrefixes() {
  using namespace gufo::server;
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {.restore =
           [](auto& state, const auto& snapshot) {
             dynamic_cast<FakeState&>(state).value =
                 dynamic_cast<const FakeSnapshot&>(snapshot).value;
           },
       .capacity_bytes = [] { return 1024; },
       .on_event = {}},
      4);
  const std::vector<ContinuationToken> text{1, 2, 3};
  const std::vector<ContinuationToken> image{1, 2, 3, 248056, 4};
  const std::vector<ContinuationToken> appended{1, 2, 3, 248056, 4, 248056, 5};
  const std::vector<std::uint8_t> a{10}, b{20}, ab{30};
  const std::vector<ContinuationInputPrefix> first{{3, {}}};
  const std::vector<ContinuationInputPrefix> second{{3, {}}, {5, a}};
  {
    auto lease = cache.Acquire(image, {}, a, {}, true, 0, first);
    Expect(lease.TryReserveSnapshot(sizeof(std::size_t), text.size()),
           "image request admits a checkpoint before its first image");
    lease.PublishSnapshot(text, std::make_unique<FakeSnapshot>(7));
    Expect(lease.HasSnapshotFor(text),
           "checkpoint lookup uses the identity at the saved position");
    dynamic_cast<FakeState&>(lease.state()).value = 70;
    lease.Commit({}, {}, image);
  }
  {
    auto lease = cache.Acquire(appended, {}, ab, {}, true, 0, second);
    Expect(lease.cache_hit() && lease.cached_tokens() == image.size() &&
               dynamic_cast<FakeState&>(lease.state()).value == 70,
           "new image reuses the live first-image state");
    Expect(lease.TryReserveSnapshot(sizeof(std::size_t), image.size()),
           "first-image snapshot admitted");
    lease.Commit(image, std::make_unique<FakeSnapshot>(70));
  }
  {
    int attachments = 0;
    auto attach = [&](ContinuationState&) { ++attachments; };
    auto matching = cache.Acquire(appended, {}, ab, attach, true, 0, second);
    auto changed = cache.Acquire(image, {}, b, {}, true, 0, first);
    Expect(matching.cached_tokens() == image.size() &&
               changed.cached_tokens() == text.size(),
           "simultaneous different images choose independent safe frontiers");
    Expect(attachments == 2, "request input is reattached after restoration");
    Expect(dynamic_cast<FakeState&>(matching.state()).value == 70 &&
               dynamic_cast<FakeState&>(changed.state()).value == 7 &&
               dynamic_cast<FakeState&>(matching.state()).id !=
                   dynamic_cast<FakeState&>(changed.state()).id,
           "branches restore separate payloads into separate sessions");
  }
  {
    auto removed = cache.Acquire(image);
    Expect(removed.cached_tokens() == text.size(),
           "removing an image cannot restore its computed embedding state");
  }
  Expect(cache.retained_snapshot_bytes() == 2 * sizeof(std::size_t),
         "prefix identity does not duplicate snapshot payloads");

  // Without an earlier checkpoint, expose token agreement even when image
  // identity makes the available state unusable.
  std::vector<std::size_t> direct_invalidations(1);
  ContinuationCache direct(
      1, [&] { return std::make_unique<FakeState>(0, &direct_invalidations); });
  auto original = direct.Acquire(image, {}, a, {}, true, 0, first);
  original.Commit(image);
  auto changed = direct.Acquire(image, {}, b, {}, true, 0, first);
  Expect(!changed.cache_hit() &&
             changed.lookup().miss_reason == "input_changed" &&
             changed.lookup().common_prefix_tokens == image.size() &&
             changed.lookup().checkpoint_tokens == image.size(),
         "image mismatch diagnostics retain the actual token agreement");
}

void TestNewBranchPreservesSharedSourceUnderBytePressure() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  using gufo::server::SnapshotPurpose;
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 4 * sizeof(std::size_t); },
          .on_event = {},
      },
      8);
  const auto retain = [&](const Tokens& tokens, std::size_t value) {
    auto lease = cache.Acquire(tokens);
    Expect(lease.TryReserveSnapshot(sizeof(std::size_t), tokens.size(), false,
                                    SnapshotPurpose::kContinuation, tokens),
           "initial checkpoint fits without eviction");
    lease.Commit(tokens, std::make_unique<FakeSnapshot>(value));
  };
  const Tokens unrelated{9, 9, 9};
  const Tokens shared{1, 2, 3};
  retain(unrelated, 9003);
  retain(shared, 1003);
  retain({1, 2, 3, 4, 4}, 2005);
  retain({1, 2, 3, 5, 5}, 3005);

  const Tokens incoming{1, 2, 3, 6, 6};
  auto branch = cache.Acquire(incoming);
  Expect(branch.cache_hit() && branch.cached_tokens() == shared.size() &&
             dynamic_cast<FakeState&>(branch.state()).value == 1003,
         "the new conversation restores the shared branch point");
  Expect(branch.TryReserveSnapshot(sizeof(std::size_t), incoming.size(), false,
                                   SnapshotPurpose::kContinuation, incoming),
         "the new continuation can replace an older unrelated checkpoint");
  branch.Commit(incoming, std::make_unique<FakeSnapshot>(4005));
  Expect(cache.retained_snapshot_bytes() == 4 * sizeof(std::size_t) &&
             cache.reserved_snapshot_bytes() == 0,
         "the new branch stays within the checkpoint byte budget");

  const Tokens next_prompt{1, 2, 3, 7, 7};
  auto next = cache.Acquire(next_prompt);
  Expect(next.cache_hit() && next.cached_tokens() == shared.size() &&
             dynamic_cast<FakeState&>(next.state()).value == 1003,
         "a new branch must not replace the shared source needed by its peers");
  next.Invalidate();
  Expect(cache.CachedPrefixTokens(unrelated) == 0,
         "the oldest unrelated continuation yields to the shared branch point");
}

// RAM misses name the cause when the prompt's own checkpoint existed and was
// lost to the byte budget, or exists and is unusable without its fallback.
void TestMissReasonNamesEvictionAndStableBoundary() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  const auto make_cache = [&](std::size_t capacity_bytes) {
    next_id = 0;
    return std::make_unique<gufo::server::ContinuationCache>(
        2,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        gufo::server::ContinuationCache::SnapshotSupport{
            .restore =
                [](gufo::server::ContinuationState& state,
                   const gufo::server::ContinuationSnapshot& snapshot) {
                  dynamic_cast<FakeState&>(state).value =
                      dynamic_cast<const FakeSnapshot&>(snapshot).value;
                },
            .capacity_bytes = [capacity_bytes] { return capacity_bytes; },
            .on_event = {},
        });
  };

  {
    auto cache = make_cache(12);
    {
      auto first = cache->Acquire(Tokens{1, 2, 3});
      Expect(first.TryReserveSnapshot(8, 3) &&
                 first.Commit({1, 2, 3},
                              std::make_unique<FakeSnapshot>(7, 8)) == 8,
             "the first conversation is retained");
    }
    {
      auto second = cache->Acquire(Tokens{9, 8, 7});
      Expect(second.TryReserveSnapshot(12, 3) &&
                 second.Commit({9, 8, 7},
                               std::make_unique<FakeSnapshot>(9, 12)) == 12,
             "the second conversation evicts the first by bytes");
    }
    {
      auto returning = cache->Acquire(Tokens{1, 2, 3, 4});
      Expect(!returning.cache_hit() &&
                 returning.lookup().miss_reason == "evicted" &&
                 returning.lookup().common_prefix_tokens == 3 &&
                 returning.lookup().checkpoint_tokens == 3,
             "a prompt whose checkpoint the byte budget removed reports "
             "evicted");
      returning.Invalidate();
    }
    {
      auto unrelated = cache->Acquire(Tokens{5, 5});
      Expect(!unrelated.cache_hit() &&
                 unrelated.lookup().miss_reason == "prefix_changed",
             "an unrelated prompt keeps its previous reason");
      unrelated.Invalidate();
    }
    {
      auto restored = cache->Acquire(Tokens{1, 2, 3});
      Expect(restored.TryReserveSnapshot(12, 3) &&
                 restored.Commit({1, 2, 3},
                                 std::make_unique<FakeSnapshot>(7, 12)) == 12,
             "the first conversation is retained again");
    }
    {
      auto hit = cache->Acquire(Tokens{1, 2, 3, 4});
      Expect(hit.cache_hit() && hit.lookup().miss_reason.empty(),
             "a retained checkpoint forgets its earlier eviction");
      hit.Invalidate();
    }
  }

  {
    auto cache = make_cache(64);
    {
      auto full = cache->Acquire(Tokens{1, 2, 3, 4});
      Expect(full.TryReserveSnapshot(8, 4) &&
                 full.Commit({1, 2, 3, 4},
                             std::make_unique<FakeSnapshot>(7, 8)) == 8,
             "a full-prompt checkpoint is retained without a fallback");
    }
    auto bounded = cache->Acquire(Tokens{1, 2, 3, 4, 5}, {}, {}, {}, true,
                                  /*stable_prefix_tokens=*/2);
    Expect(!bounded.cache_hit() &&
               bounded.lookup().miss_reason == "stable_prefix_boundary" &&
               bounded.lookup().common_prefix_tokens == 4 &&
               bounded.lookup().checkpoint_tokens == 4,
           "a checkpoint unusable without an earlier fallback reports the "
           "stable boundary");
    bounded.Invalidate();
  }
}

void TestRamOnlyKeepsLongConversation() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  std::vector<std::size_t> invalidations(4);
  std::size_t next_id = 0;
  const auto make_cache = [&](std::size_t capacity, std::size_t bytes,
                              std::size_t records) {
    next_id = 0;
    return std::make_unique<gufo::server::ContinuationCache>(
        capacity,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        gufo::server::ContinuationCache::SnapshotSupport{
            .restore =
                [](gufo::server::ContinuationState& state,
                   const gufo::server::ContinuationSnapshot& snapshot) {
                  dynamic_cast<FakeState&>(state).value =
                      dynamic_cast<const FakeSnapshot&>(snapshot).value;
                },
            .capacity_bytes = [bytes] { return bytes; },
            .on_event = {},
        },
        records);
  };

  // Record pressure + byte pressure: a 55K-token conversation is never evicted
  // by repeated aux (ram_only) requests, while aux repeats from RAM when room
  // remains and yields first when ordinary requests need capacity.
  {
    auto cache = make_cache(/*capacity=*/2, /*bytes=*/256, /*records=*/2);
    Tokens long_prompt(55000, 1);
    long_prompt[0] = 42;
    {
      auto ordinary = cache->Acquire(long_prompt);
      Expect(ordinary.TryReserveSnapshot(128, long_prompt.size()) &&
                 ordinary.Commit(long_prompt, std::make_unique<FakeSnapshot>(
                                                  100, 128)) == 128,
             "55K ordinary conversation snapshot is retained");
    }
    Tokens aux1{7, 1, 2};
    {
      auto aux =
          cache->Acquire(aux1, {}, {}, {}, true, 0, {}, /*ram_only=*/true);
      Expect(aux.TryReserveSnapshot(64, aux1.size()) &&
                 aux.Commit(aux1, std::make_unique<FakeSnapshot>(1, 64)) == 64,
             "aux request fits in remaining RAM record and byte budget");
    }
    {
      auto aux_hit =
          cache->Acquire(aux1, {}, {}, {}, true, 0, {}, /*ram_only=*/true);
      Expect(aux_hit.cache_hit() && aux_hit.cached_tokens() == aux1.size(),
             "aux request repeats from RAM while budget remains");
      aux_hit.Invalidate();
    }
    // A second aux requiring 128 bytes cannot fit alongside the 55K snapshot;
    // it may evict aux1 (64 bytes) or be skipped, never evicting the 55K entry.
    for (std::uint32_t tag = 10; tag < 14; ++tag) {
      Tokens aux_next{tag, 1, 2};
      auto aux = cache->Acquire(aux_next, {}, {}, {}, true, 0, {},
                                /*ram_only=*/true);
      if (aux.TryReserveSnapshot(128, aux_next.size())) {
        (void)aux.Commit(aux_next, std::make_unique<FakeSnapshot>(tag, 128));
      } else {
        aux.Invalidate();
      }
      Expect(cache->CachedPrefixTokens(long_prompt) == long_prompt.size(),
             "aux byte pressure never evicts the 55K ordinary snapshot");
    }
    // Record pressure with zero free slots: all 2 records filled by ordinary
    // snapshots; aux must neither reserve nor evict either ordinary record.
    Tokens second_ordinary{9, 9, 9};
    {
      auto ord2 = cache->Acquire(second_ordinary);
      Expect(ord2.TryReserveSnapshot(128, second_ordinary.size()) &&
                 ord2.Commit(second_ordinary,
                             std::make_unique<FakeSnapshot>(200, 128)) == 128,
             "ordinary request evicts aux first under byte/record pressure");
    }
    Expect(cache->CachedPrefixTokens(long_prompt) == long_prompt.size(),
           "ordinary eviction spared the older 55K ordinary conversation");
    {
      Tokens aux_blocked{8, 8, 8};
      auto aux = cache->Acquire(aux_blocked, {}, {}, {}, true, 0, {},
                                /*ram_only=*/true);
      Expect(!aux.TryReserveSnapshot(64, aux_blocked.size()),
             "aux cannot displace any ordinary snapshot under record pressure");
      aux.Invalidate();
    }
    Expect(cache->CachedPrefixTokens(long_prompt) == long_prompt.size() &&
               cache->CachedPrefixTokens(second_ordinary) ==
                   second_ordinary.size(),
           "both ordinary snapshots survive blocked aux admission");

    // Continuation, history, retry, and branch-point ordinary protection
    // against aux.
    auto continuation_hit = cache->Acquire(long_prompt);
    Expect(continuation_hit.cache_hit() &&
               continuation_hit.cached_tokens() == long_prompt.size(),
           "long conversation resumes warm from RAM after aux series");
    continuation_hit.Invalidate();
  }

  // Ordinary request promotes a matching ram_only record, and C4 execution
  // state selection prefers empty or ram_only slots over an ordinary frontier.
  {
    auto cache = make_cache(/*capacity=*/2, /*bytes=*/512, /*records=*/2);
    Tokens conv{10, 20, 30};
    {
      auto ord = cache->Acquire(conv);
      Expect(ord.TryReserveSnapshot(128, conv.size()) &&
                 ord.Commit(conv, std::make_unique<FakeSnapshot>(1, 128),
                            conv) == 128,
             "ordinary request sets live frontier on slot 0");
    }
    // Slot 1 is empty; aux must acquire slot 1 so slot 0 keeps its live
    // frontier (`restore_bytes == 0` for the next turn of `conv`).
    Tokens aux_tokens{99, 1};
    {
      auto aux = cache->Acquire(aux_tokens, {}, {}, {}, true, 0, {},
                                /*ram_only=*/true);
      Expect(aux.TryReserveSnapshot(64, aux_tokens.size()) &&
                 aux.Commit(aux_tokens, std::make_unique<FakeSnapshot>(2, 64),
                            aux_tokens) == 64,
             "aux runs on empty slot 1");
    }
    {
      auto conv_next = cache->Acquire(Tokens{10, 20, 30, 40});
      Expect(conv_next.cache_hit() && conv_next.cached_tokens() == 3 &&
                 conv_next.restored_snapshot_bytes() == 0,
             "aux left ordinary live frontier intact on slot 0");
      conv_next.Invalidate();
    }
    // Ordinary request on `aux_tokens` promotes the snapshot so a later aux
    // cannot evict it.
    {
      auto promote = cache->Acquire(aux_tokens);
      Expect(promote.cache_hit(), "ordinary request hits aux snapshot");
      Expect(promote.TryReserveSnapshot(256, aux_tokens.size()) &&
                 promote.Commit(aux_tokens,
                                std::make_unique<FakeSnapshot>(3, 256)) == 256,
             "ordinary request promotes aux prefix");
    }
    {
      Tokens aux_new{77, 77};
      auto aux =
          cache->Acquire(aux_new, {}, {}, {}, true, 0, {}, /*ram_only=*/true);
      Expect(!aux.TryReserveSnapshot(256, aux_new.size()),
             "aux cannot evict the promoted ordinary snapshot");
      aux.Invalidate();
    }
    Expect(cache->CachedPrefixTokens(aux_tokens) == aux_tokens.size(),
           "promoted snapshot remains protected from aux");
  }
}

void TestRamOnlyAdmissionDoesNotDowngradeOrdinary() {
  using Tokens = std::vector<gufo::server::ContinuationToken>;
  std::vector<std::size_t> invalidations(4);
  std::size_t next_id = 0;
  auto cache = std::make_unique<gufo::server::ContinuationCache>(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      gufo::server::ContinuationCache::SnapshotSupport{
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 256; },
          .on_event = {},
      },
      /*snapshot_capacity=*/2);

  // Two leases concurrently reserve/publish the same exact tokens: ordinary
  // commits first, then aux commits; aux must not downgrade ordinary to
  // ram_only or duplicate it.
  Tokens shared{1, 2, 3, 4};
  {
    auto ord_lease = cache->Acquire(shared);
    auto aux_lease =
        cache->Acquire(shared, {}, {}, {}, true, 0, {}, /*ram_only=*/true);
    Expect(ord_lease.TryReserveSnapshot(128, shared.size()),
           "ordinary reserves shared prefix");
    Expect(aux_lease.TryReserveSnapshot(128, shared.size()),
           "aux reserves concurrently before ordinary commits");
    Expect(ord_lease.Commit(shared, std::make_unique<FakeSnapshot>(10, 128)) ==
               128,
           "ordinary commits first");
    Expect(
        aux_lease.Commit(shared, std::make_unique<FakeSnapshot>(20, 128)) == 0,
        "aux commit skips instead of downgrading or duplicating ordinary");
    Expect(cache->reserved_snapshot_bytes() == 0 &&
               cache->retained_snapshot_bytes() == 128,
           "snapshot accounting stays exact after skipped concurrent aux");
  }
  // Verify the retained `shared` snapshot is still ordinary (cannot be evicted
  // by another aux, and even with reuse_prompt=false exact replacement is
  // blocked for aux).
  {
    auto aux_noreuse =
        cache->Acquire(shared, {}, {}, {}, /*reuse_prompt=*/false, 0, {},
                       /*ram_only=*/true);
    Expect(!aux_noreuse.TryReserveSnapshot(
               256, shared.size(), false, gufo::server::SnapshotPurpose::kRetry,
               shared),
           "aux cannot exact-replace an ordinary snapshot");
    aux_noreuse.Invalidate();
  }
  Expect(cache->CachedPrefixTokens(shared) == shared.size(),
         "ordinary snapshot survived concurrent aux commit and replacement");

  // Branch-point / redundancy guard: a ram_only extension must not make an
  // ordinary continuation/branch-point look redundant under ordinary pressure.
  {
    Tokens base{1, 2, 3, 4};
    Tokens aux_ext{1, 2, 3, 4, 5, 6};
    auto branch_source = cache->Acquire(Tokens{1, 2, 3, 4, 9});
    branch_source
        .Invalidate();  // RestoreDiverges already marked base kBranchPoint
    auto aux_child =
        cache->Acquire(aux_ext, {}, {}, {}, true, 0, {}, /*ram_only=*/true);
    Expect(aux_child.TryReserveSnapshot(64, aux_ext.size()) &&
               aux_child.Commit(aux_ext,
                                std::make_unique<FakeSnapshot>(30, 64)) == 64,
           "aux child extension fits in remaining space");
    // Now ordinary request needs 192 bytes: total is 256, so it must evict
    // aux_ext (64 bytes), leaving base (128 bytes) intact -> 128 + 192 > 256?
    // Try 128 bytes: 128 (base) + 128 = 256, evicts aux_ext, keeps base!
    Tokens other{8, 8, 8};
    auto ord_other = cache->Acquire(other);
    Expect(ord_other.TryReserveSnapshot(128, other.size()) &&
               ord_other.Commit(other,
                                std::make_unique<FakeSnapshot>(40, 128)) == 128,
           "ordinary request evicts aux extension rather than ordinary base");
    Expect(cache->CachedPrefixTokens(base) == base.size() &&
               cache->CachedPrefixTokens(aux_ext) == base.size(),
           "ordinary base stays retained while aux extension was evicted");
  }
}

void TestPrefixMatchSurvivesPartialHitAndDiskRestore() {
  using namespace gufo::server;
  using Tokens = std::vector<ContinuationToken>;
  std::vector<std::size_t> invalidations(4);
  std::size_t next_id = 0;
  const auto make_cache = [&](std::size_t capacity, std::size_t bytes = 1024,
                              std::size_t records = 8) {
    next_id = 0;
    return std::make_unique<ContinuationCache>(
        capacity,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        ContinuationCache::SnapshotSupport{
            .restore =
                [](ContinuationState& state,
                   const ContinuationSnapshot& snapshot) {
                  dynamic_cast<FakeState&>(state).value =
                      dynamic_cast<const FakeSnapshot&>(snapshot).value;
                },
            .capacity_bytes = [bytes] { return bytes; },
            .on_event = {},
        },
        records);
  };

  // Cold / no records -> zero match.
  {
    auto cache = make_cache(1);
    auto lease = cache->Acquire(Tokens{10, 20, 30});
    Expect(!lease.cache_hit() && lease.prefix_match().nearest_tokens == 0 &&
               lease.prefix_match().common_prefix_tokens == 0 &&
               !lease.prefix_match().input_matches,
           "cold acquire reports empty prefix match");
    lease.Invalidate();
  }

  // Disabled reuse -> zero match even with retained/live records.
  {
    auto cache = make_cache(1);
    {
      auto first = cache->Acquire(Tokens{1, 2, 3, 4});
      (void)first.TryReserveSnapshot(64, 4);
      (void)first.Commit(Tokens{1, 2, 3, 4},
                         std::make_unique<FakeSnapshot>(1, 64),
                         Tokens{1, 2, 3, 4, 5});
    }
    auto disabled = cache->Acquire(Tokens{1, 2, 3}, {}, {}, {},
                                   /*reuse_prompt=*/false);
    Expect(!disabled.cache_hit() &&
               disabled.prefix_match().nearest_tokens == 0 &&
               disabled.prefix_match().common_prefix_tokens == 0,
           "disabled reuse leaves prefix match empty");
    disabled.Invalidate();
  }

  // Full hit sets common == nearest == prompt.size() and input_matches == true.
  {
    auto cache = make_cache(1);
    Tokens p{1, 2, 3, 4};
    {
      auto first = cache->Acquire(p);
      (void)first.TryReserveSnapshot(64, p.size());
      (void)first.Commit(p, std::make_unique<FakeSnapshot>(1, 64));
    }
    auto hit = cache->Acquire(p);
    Expect(hit.cache_hit() && hit.cached_tokens() == 4 &&
               hit.prefix_match().common_prefix_tokens == 4 &&
               hit.prefix_match().nearest_tokens == 4 &&
               hit.prefix_match().input_matches,
           "full hit sets full prefix match");
    hit.Invalidate();
  }

  // Partial / short hit + deeper retained record + truncate + live_tokens
  // before invalidation/clear.
  {
    auto cache = make_cache(1, 2048, 8);
    // Retained short prefix [1, 2]
    {
      auto a = cache->Acquire(Tokens{1, 2});
      (void)a.TryReserveSnapshot(64, 2);
      (void)a.Commit(Tokens{1, 2}, std::make_unique<FakeSnapshot>(1, 64));
    }
    // Live longer frontier [1, 2, 3, 4, 5, 6] stored on state before clear.
    {
      auto b = cache->Acquire(Tokens{1, 2, 3, 4, 5});
      (void)b.TryReserveSnapshot(64, 5);
      (void)b.Commit(Tokens{1, 2, 3, 4, 5},
                     std::make_unique<FakeSnapshot>(2, 64),
                     Tokens{1, 2, 3, 4, 5, 6});
    }
    // Probe [1, 2, 3, 4, 99, 100]: captures live frontier [1..6] -> common=4,
    // nearest=6 before live_tokens.clear()!
    auto probe = cache->Acquire(Tokens{1, 2, 3, 4, 99, 100});
    Expect(
        probe.prefix_match().common_prefix_tokens == 4 &&
            probe.prefix_match().nearest_tokens == 6 &&
            probe.prefix_match().input_matches,
        "live frontier longer than retained snapshot is captured before clear");
    probe.Invalidate();

    // Truncated request [1, 2, 3]: matches retained/live, common == 3,
    // nearest >= 5.
    auto truncated = cache->Acquire(Tokens{1, 2, 3});
    Expect(truncated.prefix_match().common_prefix_tokens == 3 &&
               truncated.prefix_match().nearest_tokens >= 5 &&
               truncated.prefix_match().input_matches,
           "truncated request sees deeper nearest record");
    truncated.Invalidate();
  }

  // Identity mismatch and LCP tie-breaking: larger common wins; tie prefers
  // compatible identity, then longer nearest_tokens.
  {
    auto cache = make_cache(1, 2048, 8);
    const std::vector<std::uint8_t> id_a{1, 1, 1, 1};
    const std::vector<std::uint8_t> id_b{2, 2, 2, 2};
    const std::vector<ContinuationInputPrefix> pref_a{
        {.token_count = 3, .identity = id_a}};
    const std::vector<ContinuationInputPrefix> pref_b{
        {.token_count = 3, .identity = id_b}};
    {
      auto a =
          cache->Acquire(Tokens{10, 20, 30, 40}, {}, id_a, {}, true, 0, pref_a);
      (void)a.TryReserveSnapshot(64, 4);
      (void)a.Commit(Tokens{10, 20, 30, 40},
                     std::make_unique<FakeSnapshot>(10, 64));
    }
    {
      auto b = cache->Acquire(Tokens{10, 20, 30, 50, 60}, {}, id_b, {}, true, 0,
                              pref_b);
      (void)b.TryReserveSnapshot(64, 5);
      (void)b.Commit(Tokens{10, 20, 30, 50, 60},
                     std::make_unique<FakeSnapshot>(11, 64));
    }
    // Probe with id_a matches {10,20,30} for both (common=3): compatible id_a
    // beats mismatched id_b even though id_b has nearest=5 vs id_a's nearest=4.
    auto tie_id =
        cache->Acquire(Tokens{10, 20, 30, 99}, {}, id_a, {}, true, 0, pref_a);
    Expect(tie_id.prefix_match().common_prefix_tokens == 3 &&
               tie_id.prefix_match().nearest_tokens == 4 &&
               tie_id.prefix_match().input_matches,
           "tie in common prefix prefers compatible input identity");
    tie_id.Invalidate();

    // Mismatched identity id_c hits neither cleanly -> input_matches == false
    // and longer nearest (5) wins tie!
    const std::vector<std::uint8_t> id_c{3, 3, 3, 3};
    const std::vector<ContinuationInputPrefix> pref_c{
        {.token_count = 3, .identity = id_c}};
    auto mismatch =
        cache->Acquire(Tokens{10, 20, 30, 99}, {}, id_c, {}, true, 0, pref_c);
    Expect(
        mismatch.prefix_match().common_prefix_tokens == 3 &&
            mismatch.prefix_match().nearest_tokens == 5 &&
            !mismatch.prefix_match().input_matches,
        "identity mismatch reports input_matches=false and picks longest tie");
    mismatch.Invalidate();
  }
}

int main() {
  TestPrefixMatchSurvivesPartialHitAndDiskRestore();
  TestRamOnlyKeepsLongConversation();
  TestRamOnlyAdmissionDoesNotDowngradeOrdinary();
  TestMissReasonNamesEvictionAndStableBoundary();
  TestAppendedImagesReuseOnlyCompatiblePrefixes();
  TestImageIdentityIsolation();
  TestColdMissThenExactExtensionHit();
  TestDivergenceInvalidatesOldState();
  TestUncommittedLeaseIsInvalidated();
  TestLongestAvailablePrefixWins();
  TestWaitingAcquireCanBeCancelled();
  TestSnapshotCanBranchIntoTwoIndependentStateSlots();
  TestBorrowedSnapshotPrefersAvailableOwner();
  TestBorrowedSnapshotKeepsOwnersLiveFrontier();
  TestBorrowedSnapshotRetryReplacesOwnersReply();
  TestCachedPrefixTokensPeeksWithoutLeasing();
  TestBranchPointOutlivesOlderTurnsUnderPressure();
  TestNewBranchPreservesSharedSourceUnderBytePressure();
  TestLearnedBranchPointOutlivesItsOlderBranch();
  TestDeeperLearnedBranchPointSupersedesShallower();
  TestRestoredSharedPrefixOutlivesFrozenFrontiers();
  TestByteCapacityEvictsBeforeSnapshotAllocation();
  TestConcurrentReservationsCannotOvercommitBudget();
  TestImpossibleReservationPreservesRetainedEntries();
  TestAbandonedReservationIsReleased();
  TestReservationMismatchSkipsRetentionWithoutFailingCommit();
  TestEntryReplacementLogsRemovedSnapshot();
  TestRetentionPrefersExtraCopiesUnderPressure();
  TestOptionalPublicationRechecksRecordPressure();
  TestNewConversationReplacesItsOwnHistoryFirst();
  TestRetryDoesNotDisplaceEarlierHistory();
  TestEditedTailReplacementPreservesSharedCheckpoints();
  TestReplacedSourceDoesNotEvictAnotherBranchTail();
  TestCacheCandidateDetailIsDebugTierOnly();
  TestWaitingAcquireReportsCandidateSkips();
  std::cout << "All continuation cache tests passed\n";
  return 0;
}
