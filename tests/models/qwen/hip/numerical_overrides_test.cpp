// The arithmetic overrides of a Qwen session and their part of the
// continuation-cache identity: defaults are explicit, supported policies have
// distinct identities, and external arithmetic without a content fingerprint
// cannot create persistent state.
#include "src/models/qwen/hip/numerical_overrides.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace gufo::hip;
using Environment = std::map<std::string, std::string>;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

gufo::core::ModelConfig Geometry(std::uint32_t heads, std::uint32_t kv_heads) {
  gufo::core::ModelConfig config;
  config.num_attention_heads = heads;
  config.num_key_value_heads = kv_heads;
  return config;
}

QwenNumericalOverrides Resolve(
    const Environment& environment,
    const gufo::core::ModelConfig& config = Geometry(16, 2),
    const QwenExecutionPolicy& policy = QwenExecutionPolicy::Production(),
    std::uint32_t max_context = 131072) {
  return ResolveQwenNumericalOverrides(
      config, policy, max_context, [&environment](const char* name) {
        const auto found = environment.find(name);
        return found == environment.end() ? nullptr : found->second.c_str();
      });
}

bool RejectsPersistence(const QwenNumericalOverrides& overrides,
                        std::string_view variable) {
  try {
    (void)overrides.CacheIdentity();
  } catch (const std::invalid_argument& error) {
    const std::string_view message = error.what();
    Expect(message.find(variable) != std::string_view::npos &&
               message.find("disk cache") != std::string_view::npos &&
               message.find("fingerprint") != std::string_view::npos,
           "persistence rejection identifies the external arithmetic and its "
           "missing fingerprint");
    return true;
  }
  std::cerr << "FAIL: persistence accepted unfingerprinted " << variable
            << '\n';
  return false;
}
}  // namespace

int main() {
  const QwenNumericalOverrides defaults;
  Expect(Resolve({}) == defaults &&
             defaults.CacheIdentity() ==
                 "attention=dense-v1\nmoe_prefill=grouped-v1\n"
                 "prefill_ffn=gpu-v1\nhipblaslt_plans=default\n",
         "an empty environment is the default policy, named explicitly so a "
         "state stored without these lines is not taken for it");
  // The tag of the arithmetic before verification rows equalled decoding.
  Expect(gufo::hip::kQwenNumericsIdentity != "qwen-bf16-fp32-prefill-v1",
         "states of the older row arithmetic have another numerics version");
  Expect(!defaults.sparse_attention && defaults.grouped_moe_prefill,
         "the default policy is dense attention with grouped expert prefill");
  Expect(Resolve({{"GUFO_QWEN_SPARSE_ATTN", "0"},
                  {"GUFO_QWEN_MOE_PREFILL", "1"},
                  {"GUFO_QWEN_SPARSE_TOPK", "7"}}) == defaults,
         "explicit defaults and parameters of a disabled sparse attention "
         "resolve to the default policy");

  const Environment sparse = {{"GUFO_QWEN_SPARSE_ATTN", "1"}};
  const auto sparse_defaults = Resolve(sparse);
  Expect(sparse_defaults.sparse_attention &&
             sparse_defaults.sparse_sink_blocks == 2 &&
             sparse_defaults.sparse_local_blocks == 32 &&
             sparse_defaults.sparse_top_blocks == 256 &&
             sparse_defaults.sparse_min_context == 8192,
         "sparse attention resolves its documented parameter defaults");
  Expect(Resolve(sparse) == sparse_defaults &&
             Resolve(sparse).CacheIdentity() == sparse_defaults.CacheIdentity(),
         "the same policy has the same identity");
  Expect(sparse_defaults.CacheIdentity() ==
             "attention=block-sparse-v1\nsparse_sink_blocks=2\n"
             "sparse_local_blocks=32\nsparse_top_blocks=256\n"
             "sparse_min_context=8192\nmoe_prefill=grouped-v1\n"
             "prefill_ffn=gpu-v1\nhipblaslt_plans=default\n",
         "the supported sparse policy keeps its stored identity");

  // A directory/boolean or plan pathname cannot identify its content. Resolve
  // still permits the experimental route without persistent caching, but a
  // request for a persistent identity must fail closed for every such route.
  std::size_t rejected = 0;
  for (const Environment& environment :
       {Environment{{"GUFO_NPU_DIR", "/npu-a"}},
        Environment{{"GUFO_NPU_DIR", "/npu-b"},
                    {"GUFO_QWEN_NPU_QKV_SLICES", "0"}},
        Environment{{"GUFO_NPU_DIR", ""}}}) {
    const auto overrides = Resolve(environment);
    Expect(overrides.npu_prefill,
           "resolving an NPU route does not itself require persistence");
    rejected += RejectsPersistence(overrides, "GUFO_NPU_DIR");
  }
  for (const std::string path : {"/plans-a", "/plans-b"}) {
    const auto overrides = Resolve({{"GUFO_HIPBLASLT_PLAN_CACHE", path}});
    Expect(overrides.hipblaslt_plan_cache == path,
           "resolving a plan route does not itself require persistence");
    rejected += RejectsPersistence(overrides, "GUFO_HIPBLASLT_PLAN_CACHE");
  }
  Expect(rejected == 5,
         "all arithmetic without a content fingerprint rejects persistence");
  Expect(Resolve({{"GUFO_HIPBLASLT_PLAN_CACHE", ""}}).CacheIdentity() ==
             defaults.CacheIdentity(),
         "an empty plan path keeps the default arithmetic identity");

  // Every distinct effective policy must have a distinct identity.
  std::set<std::string> identities = {defaults.CacheIdentity()};
  const auto distinct = [&identities](const QwenNumericalOverrides& overrides,
                                      const char* message) {
    Expect(identities.insert(overrides.CacheIdentity()).second, message);
  };
  distinct(sparse_defaults, "sparse attention changes the identity");
  struct Parameter {
    const char* name;
    std::uint32_t QwenNumericalOverrides::* field;
  };
  for (const Parameter parameter :
       {Parameter{"GUFO_QWEN_SPARSE_SINK",
                  &QwenNumericalOverrides::sparse_sink_blocks},
        Parameter{"GUFO_QWEN_SPARSE_LOCAL",
                  &QwenNumericalOverrides::sparse_local_blocks},
        Parameter{"GUFO_QWEN_SPARSE_TOPK",
                  &QwenNumericalOverrides::sparse_top_blocks},
        Parameter{"GUFO_QWEN_SPARSE_MIN_CONTEXT",
                  &QwenNumericalOverrides::sparse_min_context}}) {
    auto environment = sparse;
    // One value no parameter has by default, so swapped fields would collide.
    environment[parameter.name] = "77";
    const auto changed = Resolve(environment);
    Expect(changed.*parameter.field == 77 && changed != sparse_defaults,
           "a sparse parameter reaches its own field");
    distinct(changed, "each sparse parameter changes the identity");
  }
  const auto rowwise = Resolve({{"GUFO_QWEN_MOE_PREFILL", "0"}});
  Expect(!rowwise.grouped_moe_prefill && !rowwise.sparse_attention,
         "GUFO_QWEN_MOE_PREFILL=0 disables only the grouped expert prefill");
  Expect(rowwise.CacheIdentity() ==
             "attention=dense-v1\nmoe_prefill=row-wise-v1\n"
             "prefill_ffn=gpu-v1\nhipblaslt_plans=default\n",
         "the supported row-wise policy keeps its stored identity");
  distinct(rowwise, "the expert prefill switch changes the identity");
  auto both = sparse;
  both["GUFO_QWEN_MOE_PREFILL"] = "0";
  distinct(Resolve(both), "both overrides together are a policy of their own");

  // The identity names the policy in effect, not the request.
  Expect(Resolve(sparse, Geometry(32, 8)) == defaults,
         "an unsupported head geometry stays dense");
  Expect(Resolve(sparse, Geometry(24, 4)).sparse_attention,
         "the 24/4 head geometry supports sparse attention");
  auto fp32_kv = QwenExecutionPolicy::Production();
  fp32_kv.kv_cache_storage = QwenKvCacheStorage::kFp32;
  Expect(Resolve(sparse, Geometry(16, 2), fp32_kv) == defaults,
         "sparse attention needs the FP16 KV cache");
  Expect(Resolve(sparse, Geometry(16, 2), QwenExecutionPolicy::Production(),
                 262145) == defaults &&
             Resolve(sparse, Geometry(16, 2), QwenExecutionPolicy::Production(),
                     262144)
                 .sparse_attention,
         "sparse attention is bounded by the context its lists can index");
  Expect(
      Resolve({{"GUFO_QWEN_SPARSE_ATTN", "1"}, {"GUFO_QWEN_SPARSE_TOPK", "-5"}})
              .sparse_top_blocks == 0,
      "a negative parameter clamps to zero");
  std::cout << "Qwen numerical override identity checks passed.\n";
  return 0;
}
