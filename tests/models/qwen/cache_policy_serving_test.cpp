// Disk continuation snapshots against the arithmetic overrides of a Qwen
// session (GUFO_QWEN_SPARSE_ATTN with its parameters, GUFO_QWEN_MOE_PREFILL),
// on real weights and one shared cache directory: a snapshot written under
// another override is a cache miss, never an error or a restored state; each
// policy restores its own snapshots; and a sparse checkpoint taken after
// sparse decoding continues exactly like the session that was never restored.
//
// qwen_cache_policy_serving_test MODEL.gguf CACHE_PARENT_DIRECTORY

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {
using namespace gufo;
using Backend = server::InferenceBackend;

constexpr std::uint32_t kContext = 4096;
constexpr std::size_t kReplyTokens = 24;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

constexpr const char* kOverrideVariables[] = {
    "GUFO_QWEN_SPARSE_ATTN",        "GUFO_QWEN_SPARSE_SINK",
    "GUFO_QWEN_SPARSE_LOCAL",       "GUFO_QWEN_SPARSE_TOPK",
    "GUFO_QWEN_SPARSE_MIN_CONTEXT", "GUFO_QWEN_MOE_PREFILL"};

struct Variable {
  const char* name;
  const char* value;
};

// Replaces every override of the process environment with `variables`.
void SetOverrides(std::initializer_list<Variable> variables) {
  for (const char* name : kOverrideVariables)
    Require(::unsetenv(name) == 0, "cannot clear an override");
  for (const auto& variable : variables)
    Require(::setenv(variable.name, variable.value, 1) == 0,
            "cannot set an override");
}

// Three blocks of 64 keys: fewer than the first prompt holds, so its reply is
// already decoded without most of them.
constexpr std::initializer_list<Variable> kSparse = {
    {"GUFO_QWEN_SPARSE_ATTN", "1"},
    {"GUFO_QWEN_SPARSE_SINK", "1"},
    {"GUFO_QWEN_SPARSE_LOCAL", "1"},
    {"GUFO_QWEN_SPARSE_TOPK", "1"},
    {"GUFO_QWEN_SPARSE_MIN_CONTEXT", "128"}};

std::string Lines(std::size_t first, std::size_t count) {
  std::string text;
  for (std::size_t i = first; i < first + count; ++i) {
    text += "Record " + std::to_string(i) + ": shelf " +
            std::to_string((i * 7) % 23) + " holds " +
            std::to_string((i * 13) % 97) + " boxes.\n";
  }
  return text;
}

server::ChatRequest Root() {
  server::ChatRequest request;
  request.reasoning.enabled = false;
  request.messages.emplace_back(
      tokenization::ChatRole::kUser,
      Lines(1, 24) + "List the shelves of the first three records.");
  return request;
}

// Longer than the store's minimum checkpoint step, so the prompt of this
// turn, which follows the decoded first reply, is persisted too.
server::ChatRequest SecondTurn(const std::string& reply) {
  auto request = Root();
  request.messages.emplace_back(tokenization::ChatRole::kAssistant, reply);
  request.messages.emplace_back(
      tokenization::ChatRole::kUser,
      Lines(100, 150) + "How many boxes does record 120 hold? Explain.");
  return request;
}

void Report(std::string_view label, const Backend::Result& result) {
  std::cout << label << ": disk_hit=" << result.cache_disk_hit
            << " cache_hit=" << result.cache_hit
            << " cached=" << result.cached_prompt_tokens
            << " prefill=" << result.prefill_tokens
            << " prompt=" << result.prompt_tokens
            << " generated=" << result.tokens.size()
            << " disk_queued_bytes=" << result.cache_disk_queued_bytes << '\n';
}

void RequireMiss(const Backend::Result& result, std::string_view message) {
  Require(!result.cache_disk_hit && !result.cache_hit &&
              result.cached_prompt_tokens == 0 &&
              result.prefill_tokens == result.prompt_tokens &&
              !result.tokens.empty() && !result.cancelled,
          message);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 1)
    return 77;
  try {
    Require(argc == 3,
            "usage: qwen_cache_policy_serving_test MODEL.gguf "
            "CACHE_PARENT_DIRECTORY");
    SetOverrides({});
    std::string error;
    auto reader = std::shared_ptr<const core::GgufReader>(
        core::GgufReader::OpenFile(argv[1], &error));
    Require(reader != nullptr, error);
    const auto model = hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Require(model != nullptr, error);

    std::string pattern =
        (std::filesystem::path(argv[2]) / "cache-policy-XXXXXX").string();
    Require(::mkdtemp(pattern.data()) != nullptr, "cannot create test cache");
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    } cleanup{pattern};
    server::TextDiskCacheConfig cache;
    cache.directory = cleanup.path;
    cache.capacity_bytes = std::size_t{8} << 30;
    cache.staging_capacity_bytes = std::size_t{2} << 30;
    cache.model_artifact_fingerprint = core::GgufIdentityHex(*reader);
    // Destroying a backend drains its queued snapshots to disk.
    const auto load = [&] {
      auto backend = std::make_unique<Backend>();
      Require(backend->load(model, &error, kContext, 1, {}, {}, {}, cache),
              error);
      return backend;
    };
    sampling::SamplingConfig greedy;
    greedy.temperature = 0;
    const auto root = Root();

    // Default policy: stored, then restored by a second backend.
    auto backend = load();
    const auto dense_root = backend->chat(root, kReplyTokens, greedy);
    Report("dense root", dense_root);
    RequireMiss(dense_root, "an empty cache directory produced a hit");
    Require(dense_root.cache_disk_queued_bytes > 0,
            "the dense root snapshot did not reach disk");
    const auto dense_second = SecondTurn(dense_root.text);
    const auto dense_turn = backend->chat(dense_second, kReplyTokens, greedy);
    Report("dense second turn", dense_turn);
    Require(dense_turn.prompt_tokens + kReplyTokens < kContext,
            "the fixture does not fit the test context");
    Require(dense_turn.cache_disk_queued_bytes > 0,
            "the dense second-turn snapshot did not reach disk");
    backend.reset();
    backend = load();
    const auto dense_restored = backend->chat(root, kReplyTokens, greedy);
    Report("dense root restored", dense_restored);
    Require(dense_restored.cache_disk_hit &&
                dense_restored.tokens == dense_root.tokens,
            "a compatible dense snapshot was not restored exactly");
    backend.reset();

    // Row-wise expert prefill must not accept the grouped snapshot.
    SetOverrides({{"GUFO_QWEN_MOE_PREFILL", "0"}});
    backend = load();
    const auto rowwise_root = backend->chat(root, kReplyTokens, greedy);
    Report("row-wise root", rowwise_root);
    RequireMiss(rowwise_root,
                "row-wise expert prefill accepted a grouped-prefill snapshot");
    backend.reset();
    backend = load();
    const auto rowwise_restored = backend->chat(root, kReplyTokens, greedy);
    Report("row-wise root restored", rowwise_restored);
    Require(rowwise_restored.cache_disk_hit &&
                rowwise_restored.tokens == rowwise_root.tokens,
            "a compatible row-wise snapshot was not restored exactly");
    backend.reset();

    // Sparse attention must not accept the dense snapshot.
    SetOverrides(kSparse);
    {
      const hip::QwenGpuExecutor probe(model, kContext);
      Require(probe.GetNumericalOverrides().sparse_attention,
              "the model geometry does not engage sparse attention");
    }
    backend = load();
    const auto sparse_root = backend->chat(root, kReplyTokens, greedy);
    Report("sparse root", sparse_root);
    RequireMiss(sparse_root, "sparse attention accepted a dense snapshot");
    // The reply above was decoded with sparse attention; this turn's prompt
    // snapshot is therefore a checkpoint after affected decoding.
    const auto sparse_second = SecondTurn(sparse_root.text);
    const auto sparse_turn = backend->chat(sparse_second, kReplyTokens, greedy);
    Report("sparse second turn", sparse_turn);
    Require(sparse_turn.cache_hit && !sparse_turn.cache_disk_hit &&
                sparse_turn.cached_prompt_tokens >= sparse_root.prompt_tokens,
            "the sparse second turn did not continue its live session");
    Require(sparse_turn.cache_disk_queued_bytes > 0,
            "the sparse second-turn snapshot did not reach disk");
    backend.reset();
    backend = load();
    const auto sparse_restored =
        backend->chat(sparse_second, kReplyTokens, greedy);
    Report("sparse second turn restored", sparse_restored);
    Require(sparse_restored.cache_disk_hit &&
                sparse_restored.cached_prompt_tokens >
                    sparse_root.prompt_tokens + sparse_root.tokens.size(),
            "the sparse checkpoint after decoding was not restored");
    Require(sparse_restored.tokens == sparse_turn.tokens,
            "a restored sparse checkpoint continued differently from the "
            "session that was never restored");
    const bool root_differs = sparse_root.tokens != dense_root.tokens;
    const bool turn_differs = sparse_turn.tokens != dense_turn.tokens;
    std::cout << "sparse differs from dense: root=" << root_differs
              << " second_turn=" << turn_differs << '\n';
    Require(root_differs || turn_differs,
            "sparse attention changed no reply: the fixture does not engage "
            "it");
    backend.reset();

    // Every sparse parameter is part of the identity.
    for (const Variable changed :
         {Variable{"GUFO_QWEN_SPARSE_SINK", "2"},
          Variable{"GUFO_QWEN_SPARSE_LOCAL", "2"},
          Variable{"GUFO_QWEN_SPARSE_TOPK", "2"},
          Variable{"GUFO_QWEN_SPARSE_MIN_CONTEXT", "192"}}) {
      SetOverrides(kSparse);
      Require(::setenv(changed.name, changed.value, 1) == 0,
              "cannot set an override");
      backend = load();
      const auto result = backend->chat(root, 4, greedy);
      Report(std::string(changed.name) + "=" + changed.value, result);
      RequireMiss(result, std::string("a snapshot of another ") + changed.name +
                              " was accepted");
      backend.reset();
    }

    // The snapshots of the other policies left the dense ones intact.
    SetOverrides({});
    backend = load();
    const auto dense_turn_restored =
        backend->chat(dense_second, kReplyTokens, greedy);
    Report("dense second turn restored", dense_turn_restored);
    Require(dense_turn_restored.cache_disk_hit &&
                dense_turn_restored.cached_prompt_tokens >
                    dense_root.prompt_tokens + dense_root.tokens.size() &&
                dense_turn_restored.tokens == dense_turn.tokens,
            "the dense checkpoint was not restored exactly after the other "
            "policies used the directory");
    backend.reset();
    std::cout << "cache identity of numerical overrides: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen_cache_policy_serving_test: " << e.what() << '\n';
    return 1;
  }
}
