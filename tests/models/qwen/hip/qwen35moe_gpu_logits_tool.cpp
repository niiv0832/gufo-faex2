// Development tool for the Qwen GPU MoE path. It runs either a single forward
// pass over a list of token ids to dump the last position's logits, or a greedy
// generation loop, so the GPU MoE path can be compared against the CPU
// reference and the llama.cpp CPU oracle without a full text harness.
//
// usage: qwen35moe_gpu_logits_tool <model.gguf> <ids.txt> <out> [n_gen]
//   without n_gen: writes <out> as raw float32 logits of the last position.
//   with n_gen:    writes <out> as n_gen greedy token ids, space separated.
//
// QUALITY_PREFILL=<n>: prefills the first n ids, feeds the rest one token at a
// time and writes the logits rows of positions n - 1 on (the row at position p
// predicts token p + 1). n = 0 reads the whole history through ForwardToken
// from position 0: the decode arithmetic, with no prompt pass at all.
//   QUALITY_STRIDE=<s>      every s-th row only.
//   QUALITY_WINDOWS=a:n,... only rows of positions a .. a + n - 1 of each
//                           window, in ascending order.
//   QUALITY_KV=fp32         FP32 attention keys and values instead of FP16.
//   QUALITY_BATCH=1         <ids.txt> is a list of "ids-file out-file" lines;
//                           every document runs after a Reset on one model.
//   QUALITY_TIME=<repeats>  no rows: after one warm-up pass, times reading
//                           the whole ids file <repeats> times and prints the
//                           median (prefill n = chunked prompt pass, n = 0 =
//                           token by token).
//
// TWO_SESSIONS=<ids2.txt> with TWO_SESSIONS_HEAD=<n>: two executors share one
// model, as serve sessions do. Each prefills the first n ids of its own
// sequence without a final synchronization, one right after the other, then
// decodes one token. Both logit rows must be bitwise equal to the serial
// reference; exits 1 otherwise. n >= 64 takes the grouped MoE prefill, a
// smaller n the per-token one.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using Ids = std::vector<gufo::tokenization::TokenId>;

Ids ReadIds(const std::string& path) {
  Ids ids;
  std::ifstream in(path);
  for (std::uint32_t id; in >> id;)
    ids.push_back(id);
  return ids;
}

// Positions [first, first + count) of each "first:count" item.
std::vector<std::pair<std::size_t, std::size_t>> ParseWindows(
    const char* text) {
  std::vector<std::pair<std::size_t, std::size_t>> windows;
  std::stringstream stream(text != nullptr ? text : "");
  for (std::string item; std::getline(stream, item, ',');) {
    const auto colon = item.find(':');
    if (colon == std::string::npos)
      throw std::invalid_argument("QUALITY_WINDOWS item needs first:count");
    windows.emplace_back(std::stoul(item.substr(0, colon)),
                         std::stoul(item.substr(colon + 1)));
  }
  return windows;
}

// Reads `ids` with `prefill` leading ids in one prompt pass (none for 0) and
// the rest token by token; returns the number of rows `on_row` accepted.
std::size_t ReadDocument(
    gufo::hip::QwenGpuExecutor& executor, const Ids& ids, std::size_t prefill,
    const std::function<bool(std::size_t position)>& wants_row,
    const std::function<void(std::span<const float>)>& on_row) {
  std::size_t rows = 0;
  const auto row = [&](std::size_t position) {
    if (wants_row(position)) {
      on_row(executor.CopyLastLogits());
      ++rows;
    }
  };
  if (prefill != 0) {
    (void)executor.ForwardPromptBatch(std::span(ids).first(prefill), 0, true);
    row(prefill - 1);
  }
  // The last id is only ever a gold label.
  for (std::size_t i = prefill; i + 1 < ids.size(); ++i) {
    (void)executor.ForwardToken(ids[i], static_cast<std::uint32_t>(i));
    row(i);
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " model.gguf ids.txt out.bin [n_gen]\n";
    return 2;
  }
  gufo::models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
  gufo::models::qwen38_flash_next::rocm::RegisterQwenNpuFfn();
  const bool generate = argc == 5;
  std::string error;
  auto reader_owner = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader_owner) {
    std::cerr << "open failed: " << error << '\n';
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));
  if (const char* second_env = std::getenv("TWO_SESSIONS")) {
    // One shared model only: this mode runs before the single-session executor
    // below would load a second copy of the weights.
    std::vector<gufo::tokenization::TokenId> ids;
    std::ifstream first(argv[2]);
    for (std::uint32_t id; first >> id;)
      ids.push_back(id);
    std::vector<gufo::tokenization::TokenId> other;
    std::ifstream second(second_env);
    for (std::uint32_t id; second >> id;)
      other.push_back(id);
    const std::size_t head = std::getenv("TWO_SESSIONS_HEAD") != nullptr
                                 ? std::stoul(std::getenv("TWO_SESSIONS_HEAD"))
                                 : 96;
    if (head == 0 || ids.size() <= head || other.size() <= head) {
      std::cerr << "both id lists need more than " << head << " ids\n";
      return 2;
    }
    const std::uint32_t context =
        std::getenv("QUALITY_CTX") != nullptr
            ? static_cast<std::uint32_t>(std::stoul(std::getenv("QUALITY_CTX")))
            : 4096U;
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    std::unique_ptr<gufo::hip::QwenGpuExecutor> sessions[2];
    for (auto& session : sessions) {
      session = model
                    ? gufo::hip::QwenGpuExecutor::Create(model, &error, context)
                    : nullptr;
      if (!session) {
        std::cerr << "executor failed: " << error << '\n';
        return 1;
      }
    }
    const std::vector<gufo::tokenization::TokenId>* inputs[2] = {&ids, &other};
    const auto prefill = [&](int s) {
      sessions[s]->Reset();
      (void)sessions[s]->ForwardPromptBatch(std::span(*inputs[s]).first(head),
                                            0, false);
    };
    const auto decode = [&](int s) {
      (void)sessions[s]->ForwardToken((*inputs[s])[head],
                                      static_cast<std::uint32_t>(head));
      const auto logits = sessions[s]->CopyLastLogits();
      return std::vector<float>(logits.begin(), logits.end());
    };
    std::vector<float> reference[2];
    int unstable = 0;
    for (int pass = 0; pass < 2; ++pass) {
      for (int s = 0; s < 2; ++s) {
        prefill(s);
        auto row = decode(s);
        if (pass == 1 && row != reference[s])
          ++unstable;
        reference[s] = std::move(row);
      }
    }
    const int rounds = std::getenv("TWO_SESSIONS_ROUNDS") != nullptr
                           ? std::stoi(std::getenv("TWO_SESSIONS_ROUNDS"))
                           : 20;
    int mismatches = 0;
    float worst = 0.0F;
    for (int round = 0; round < rounds; ++round) {
      prefill(0);
      prefill(1);
      for (int s = 0; s < 2; ++s) {
        const auto row = decode(s);
        if (row == reference[s])
          continue;
        ++mismatches;
        for (std::size_t i = 0; i < row.size(); ++i)
          worst = std::max(worst, std::abs(row[i] - reference[s][i]));
      }
    }
    std::printf(
        "two sessions head %zu: serial repeat mismatches %d, "
        "interleaved mismatches %d of %d, max|diff| %.4f\n",
        head, unstable, mismatches, rounds * 2, worst);
    return unstable == 0 && mismatches == 0 ? 0 : 1;
  }
  auto policy = gufo::hip::QwenExecutionPolicy::Production();
  if (const char* kv = std::getenv("QUALITY_KV")) {
    if (std::string(kv) != "fp32") {
      std::cerr << "QUALITY_KV accepts fp32 only\n";
      return 2;
    }
    policy.kv_cache_storage = gufo::hip::QwenKvCacheStorage::kFp32;
  }
  auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(
      reader, &error,
      std::getenv("QUALITY_CTX") != nullptr
          ? static_cast<std::uint32_t>(std::stoul(std::getenv("QUALITY_CTX")))
          : 4096U,
      policy);
  if (!executor) {
    std::cerr << "executor failed: " << error << '\n';
    return 1;
  }
  if (const char* prefill_env = std::getenv("QUALITY_PREFILL")) {
    const std::size_t prefill = std::stoul(prefill_env);
    const char* stride_env = std::getenv("QUALITY_STRIDE");
    const std::size_t stride =
        stride_env != nullptr ? std::stoul(stride_env) : 1;
    const auto windows = ParseWindows(std::getenv("QUALITY_WINDOWS"));
    const std::size_t first_row = prefill != 0 ? prefill - 1 : 0;
    const auto wants_row = [&](std::size_t position) {
      if ((position - first_row) % stride != 0)
        return false;
      if (windows.empty())
        return true;
      for (const auto& [first, count] : windows)
        if (position >= first && position - first < count)
          return true;
      return false;
    };
    if (const char* time_env = std::getenv("QUALITY_TIME")) {
      const Ids ids = ReadIds(argv[2]);
      if (prefill > ids.size() || ids.size() < 2)
        return 2;
      // One more id than is read, as ReadDocument keeps the last for a label.
      std::vector<double> seconds;
      const int repeats = std::stoi(time_env);
      for (int pass = 0; pass <= repeats; ++pass) {
        executor->Reset();
        const auto start = std::chrono::steady_clock::now();
        (void)ReadDocument(
            *executor, ids, prefill != 0 ? ids.size() - 1 : 0,
            [](std::size_t) { return false; }, [](std::span<const float>) {});
        // The copy waits for everything queued on the stream.
        (void)executor->CopyLastLogits();
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - start;
        if (pass != 0)
          seconds.push_back(elapsed.count());
      }
      std::sort(seconds.begin(), seconds.end());
      std::printf(
          "read %zu tokens %s: median %.4f s of %d (min %.4f max "
          "%.4f), %.1f tokens/s\n",
          ids.size() - 1, prefill != 0 ? "prompt pass" : "token by token",
          seconds[seconds.size() / 2], repeats, seconds.front(), seconds.back(),
          static_cast<double>(ids.size() - 1) / seconds[seconds.size() / 2]);
      return 0;
    }
    std::vector<std::pair<std::string, std::string>> documents;
    if (std::getenv("QUALITY_BATCH") != nullptr) {
      std::ifstream list(argv[2]);
      for (std::string ids_path, out_path; list >> ids_path >> out_path;)
        documents.emplace_back(ids_path, out_path);
    } else {
      documents.emplace_back(argv[2], argv[3]);
    }
    for (const auto& [ids_path, out_path] : documents) {
      const Ids ids = ReadIds(ids_path);
      if (prefill > ids.size() || ids.size() < 2) {
        std::cerr << ids_path << ": too few ids\n";
        return 2;
      }
      executor->Reset();
      std::ofstream out(out_path, std::ios::binary);
      const std::size_t rows = ReadDocument(
          *executor, ids, prefill, wants_row,
          [&](std::span<const float> logits) {
            out.write(
                reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
          });
      out.flush();
      if (!out) {
        std::cerr << out_path << ": write failed\n";
        return 1;
      }
      std::fprintf(stderr, "%s: prefill %zu, %zu ids, wrote %zu rows\n",
                   out_path.c_str(), prefill, ids.size(), rows);
    }
    return 0;
  }
  const Ids ids = ReadIds(argv[2]);
  if (const char* rows_env = std::getenv("VERIFY_ROWS")) {
    const std::size_t rows = std::stoul(rows_env);
    const std::vector<gufo::tokenization::TokenId> head(ids.begin(),
                                                        ids.end() - rows);
    const auto start = static_cast<std::uint32_t>(head.size());
    std::vector<std::vector<float>> sequential;
    for (int pass = 0; pass < 2; ++pass) {
      sequential.clear();
      executor->Reset();
      (void)executor->ForwardPromptBatch(head, 0, true);
      for (std::size_t row = 0; row < rows; ++row) {
        (void)executor->ForwardToken(ids[head.size() + row],
                                     start + static_cast<std::uint32_t>(row));
        const auto logits = executor->CopyLastLogits();
        sequential.emplace_back(logits.begin(), logits.end());
      }
    }
    executor->Reset();
    (void)executor->ForwardPromptBatch(head, 0, true);
    const gufo::hip::QwenGpuVerificationItem item{
        executor.get(), std::span(ids).subspan(head.size()), start, true};
    std::vector<std::vector<float>> again;
    if (std::getenv("VERIFY_CONTROL")) {
      for (std::size_t row = 0; row < rows; ++row) {
        (void)executor->ForwardToken(ids[head.size() + row],
                                     start + static_cast<std::uint32_t>(row));
        const auto logits = executor->CopyLastLogits();
        again.emplace_back(logits.begin(), logits.end());
      }
    } else {
      (void)gufo::hip::QwenGpuExecutor::ForwardVerificationBatch(
          std::span(&item, 1));
    }
    std::vector<std::vector<float>> oracle;
    for (std::size_t row = 0; row < rows; ++row) {
      const std::vector<gufo::tokenization::TokenId> prefix(
          ids.begin(),
          ids.begin() + static_cast<std::ptrdiff_t>(head.size() + row + 1));
      executor->Reset();
      (void)executor->ForwardPromptBatch(prefix, 0, true);
      const auto logits = executor->CopyLastLogits();
      oracle.emplace_back(logits.begin(), logits.end());
    }
    executor->Reset();
    (void)executor->ForwardPromptBatch(head, 0, true);
    (void)gufo::hip::QwenGpuExecutor::ForwardVerificationBatch(
        std::span(&item, 1));
    for (std::size_t row = 0; row < rows; ++row) {
      double d_seq = 0.0;
      double d_bat = 0.0;
      const auto batched_row = executor->CopyVerificationLogits(row);
      for (std::size_t i = 0; i < oracle[row].size(); ++i) {
        d_seq += std::abs(oracle[row][i] - sequential[row][i]);
        d_bat += std::abs(oracle[row][i] - batched_row[i]);
      }
      std::printf(
          "oracle row %zu: mean|prefill-decode| %.4f mean|prefill-batched| "
          "%.4f\n",
          row, d_seq / static_cast<double>(oracle[row].size()),
          d_bat / static_cast<double>(oracle[row].size()));
    }
    for (std::size_t row = 0; row < rows; ++row) {
      const std::span<const float> batched =
          again.empty() ? executor->CopyVerificationLogits(row)
                        : std::span<const float>(again[row]);
      const auto& ref = sequential[row];
      float max_diff = 0.0F;
      for (std::size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::abs(ref[i] - batched[i]));
      const auto top_ref = std::max_element(ref.begin(), ref.end());
      const auto top_bat = std::max_element(batched.begin(), batched.end());
      double sum_diff = 0.0;
      std::size_t worst = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        sum_diff += std::abs(ref[i] - batched[i]);
        if (std::abs(ref[i] - batched[i]) >
            std::abs(ref[worst] - batched[worst]))
          worst = i;
      }
      std::printf(
          "  mean|diff| %.4f worst idx %zu ref %.3f batched %.3f top ref %.3f "
          "batched %.3f\n",
          sum_diff / static_cast<double>(ref.size()), worst, ref[worst],
          batched[worst], *std::max_element(ref.begin(), ref.end()),
          *std::max_element(batched.begin(), batched.end()));
      std::vector<float> sorted(ref);
      std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(),
                        std::greater<>());
      std::printf(
          "row %zu max|diff| %.4f top ref %td batched %td margin %.4f\n", row,
          max_diff, top_ref - ref.begin(), top_bat - batched.begin(),
          sorted[0] - sorted[1]);
    }
    return 0;
  }
  (void)executor->ForwardPromptBatch(ids, 0, true);

  if (!generate) {
    const auto logits = executor->CopyLastLogits();
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    std::fprintf(stderr, "wrote %zu logits\n", logits.size());
    return 0;
  }

  const int n_gen = std::stoi(argv[4]);
  std::uint32_t pos = static_cast<std::uint32_t>(ids.size());
  std::vector<gufo::tokenization::TokenId> generated;
  for (int i = 0; i < n_gen; ++i) {
    const auto logits = executor->CopyLastLogits();
    const auto best = std::max_element(logits.begin(), logits.end());
    const auto next = static_cast<gufo::tokenization::TokenId>(
        std::distance(logits.begin(), best));
    generated.push_back(next);
    if (i + 1 < n_gen) {
      (void)executor->ForwardToken(next, pos++);
    }
  }
  std::ofstream out(argv[3]);
  for (std::size_t i = 0; i < generated.size(); ++i) {
    out << (i == 0 ? "" : " ") << generated[i];
  }
  out << '\n';
  std::fprintf(stderr, "wrote %zu tokens\n", generated.size());
  return 0;
}
