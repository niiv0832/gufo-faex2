// The grouped-expert prefill feed-forward against the row-wise one (the decode
// arithmetic) on the same real activation rows, with its causes separated.
//
// A prompt pass over human text runs with a tap around the grouped backend, so
// every layer's real feed-forward input is recovered: hidden-after-layer minus
// the feed-forward output is the input of the feed-forward norm, to one
// rounding of the residual add, and the production norm kernel turns it into
// the rows both paths then read.
//
// Exact contracts (a violation fails the test):
//   - split-BF16 router logits and the row-wise router logits are within the
//     provable error bound of their arithmetic (see kSplitBound, kRowBound);
//   - whenever the eighth and ninth logit of a row are further apart than
//     twice that bound, both routers select the expert set of the
//     double-precision reference;
//   - on identical logits the grouped and the row-wise top-k kernels select
//     the same experts, ties going to the lower index;
//   - reading tokens by decode alone is repeatable and unaffected by an
//     earlier prompt pass.
// Everything else is approximate arithmetic and is measured and printed with
// no threshold: router error and expert-set changes on real rows, the relative
// error of the grouped feed-forward output, and its split by cause from a
// double-precision emulation in which one approximation is applied at a time
// (router, Q6_K to Q8_0, F16 input, F16 weights, F16 intermediates, the
// quantized-activation shared expert). The row-wise path is not the exact
// value of the weights either: its expert kernels multiply int8-quantized
// activations. That is emulated too, so both paths are placed against the
// same double-precision reference.
//
// The same is measured outside the feed-forward for the projections that read
// a layer's input norm (attention q/k/v, the recurrent qkv, gate, alpha and
// beta): the prompt pass multiplies int8-quantized activations, decode
// multiplies the FP32 row. Their real input is exact: the hidden state after
// the previous layer.
//
// qwen_moe_prefill_rows_test [MODEL.gguf]
// Without an argument GUFO_QWEN_LAYERED_PREFILL_MODEL names the weights; the
// test is skipped without it or for a model without experts.
// GUFO_QWEN_PREFILL_ROWS_IDS names a file of token ids to read instead of the
// built-in text.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/moe_ffn.hpp"
#include "src/models/qwen/hip/moe_prefill.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#include "tests/models/qwen/hip/quant_reference.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::test_reference::DecodeRow;
using gufo::test_reference::QuantCode;
using gufo::test_reference::RoundToBf16;
using gufo::test_reference::RoundToHalf;
namespace rocm = gufo::models::qwen38_flash_next::rocm;

constexpr std::size_t kRows = 256;
constexpr std::size_t kEmulatedRows = 32;
constexpr std::size_t kCapturedLayersPerPass = 9;

// Public-domain English prose (1863 and 1776), tokenized with the model's own
// tokenizer when no id file is given.
constexpr const char* kText =
    "Four score and seven years ago our fathers brought forth on this "
    "continent, a new nation, conceived in Liberty, and dedicated to the "
    "proposition that all men are created equal. Now we are engaged in a "
    "great civil war, testing whether that nation, or any nation so "
    "conceived and so dedicated, can long endure. We are met on a great "
    "battle-field of that war. We have come to dedicate a portion of that "
    "field, as a final resting place for those who here gave their lives "
    "that that nation might live. It is altogether fitting and proper that "
    "we should do this. But, in a larger sense, we can not dedicate -- we "
    "can not consecrate -- we can not hallow -- this ground. The brave men, "
    "living and dead, who struggled here, have consecrated it, far above "
    "our poor power to add or detract. The world will little note, nor long "
    "remember what we say here, but it can never forget what they did here. "
    "It is for us the living, rather, to be dedicated here to the "
    "unfinished work which they who fought here have thus far so nobly "
    "advanced. It is rather for us to be here dedicated to the great task "
    "remaining before us -- that from these honored dead we take increased "
    "devotion to that cause for which they gave the last full measure of "
    "devotion -- that we here highly resolve that these dead shall not have "
    "died in vain -- that this nation, under God, shall have a new birth of "
    "freedom -- and that government of the people, by the people, for the "
    "people, shall not perish from the earth.\n\n"
    "When in the Course of human events, it becomes necessary for one "
    "people to dissolve the political bands which have connected them with "
    "another, and to assume among the powers of the earth, the separate and "
    "equal station to which the Laws of Nature and of Nature's God entitle "
    "them, a decent respect to the opinions of mankind requires that they "
    "should declare the causes which impel them to the separation. We hold "
    "these truths to be self-evident, that all men are created equal, that "
    "they are endowed by their Creator with certain unalienable Rights, "
    "that among these are Life, Liberty and the pursuit of Happiness.";

// |computed - exact| <= bound * sum_i |x_i| |w_i| for one router logit.
//
// Split router. Each float splits as hi + lo + r with hi its bfloat16
// rounding (|x - hi| <= 2^-8 |x|), lo the bfloat16 rounding of x - hi
// (|lo| <= 2^-8 (1 + 2^-8) |x|, |r| <= 2^-16 |x|). The GEMM sums
// hi*hi + lo*hi + hi*lo; against x*w it omits lo*lo + r_x*w + (hi + lo)*r_w,
// at most 3 * 2^-16 * (1 + 2^-7) |x||w|. Its 3K products of two 8-bit
// significands are exact in single precision and their sum, in any order,
// errs by at most (3K - 1) u / (1 - (3K - 1) u) times the sum of their
// magnitudes, u = 2^-24, itself at most (1 + 2^-6) sum |x||w|.
double SplitBound(std::size_t k) {
  const double u = std::ldexp(1.0, -24);
  const double n = static_cast<double>((3 * k) - 1);
  return (3.0 * std::ldexp(1.0, -16) * (1.0 + std::ldexp(1.0, -7))) +
         ((n * u / (1.0 - (n * u))) * (1.0 + std::ldexp(1.0, -6)));
}
// Row-wise router: K rounded single-precision products summed in any order,
// with or without fused multiply-add: at most K u / (1 - K u) sum |x||w|.
double RowBound(std::size_t k) {
  const double u = std::ldexp(1.0, -24);
  const double n = static_cast<double>(k);
  return n * u / (1.0 - (n * u));
}
// Both top-k kernels order experts by single-precision softmax values, exact
// to a few units of 2^-24. A logit gap of 1e-5 changes the ratio of two
// probabilities by 1e-5, a hundred times that, so above it the order of the
// probabilities is the order of the logits, as long as the probabilities do
// not underflow: exp(-80) is still a normal float.
constexpr double kTopKResolution = 1e-5;
constexpr double kTopKSpread = 80.0;

// Library code changes the stream's float format; every line sets its own.
std::ostream& Out() {
  return std::cout << std::scientific << std::setprecision(3);
}

template<typename T>
class Device {
public:
  explicit Device(std::size_t elements) : elements_(elements) {
    HIP_CHECK(hipMalloc(&data_, elements * sizeof(T)));
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { (void)hipFree(data_); }
  [[nodiscard]] T* data() const noexcept { return data_; }
  void Upload(const std::vector<T>& host) const {
    HIP_CHECK(hipMemcpy(data_, host.data(), host.size() * sizeof(T),
                        hipMemcpyHostToDevice));
  }
  [[nodiscard]] std::vector<T> Download(std::size_t elements) const {
    std::vector<T> host(elements);
    HIP_CHECK(hipMemcpy(host.data(), data_, elements * sizeof(T),
                        hipMemcpyDeviceToHost));
    return host;
  }

private:
  T* data_ = nullptr;
  std::size_t elements_;
};

template<typename T>
std::vector<T> DownloadFrom(const void* device, std::size_t elements) {
  std::vector<T> host(elements);
  HIP_CHECK(hipMemcpy(host.data(), device, elements * sizeof(T),
                      hipMemcpyDeviceToHost));
  return host;
}

template<typename F>
void ParallelFor(std::size_t count, F&& body) {
  const std::size_t threads = std::min<std::size_t>(
      {count, 8, std::max(1U, std::thread::hardware_concurrency())});
  std::atomic<std::size_t> next{0};
  std::exception_ptr failure;
  std::mutex mutex;
  std::vector<std::thread> pool;
  for (std::size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      try {
        for (std::size_t i = next++; i < count; i = next++)
          body(i);
      } catch (...) {
        const std::lock_guard lock(mutex);
        failure = std::current_exception();
      }
    });
  }
  for (auto& thread : pool)
    thread.join();
  if (failure)
    std::rethrow_exception(failure);
}

// What the production prompt pass handed the grouped backend for one layer,
// and what the backend returned.
struct LayerTap {
  std::size_t tokens = 0;
  std::vector<float> router_logits;
  std::vector<float> shared_logit;
  std::vector<std::uint16_t> x_half;
  std::vector<float> out;
};

class TapPrefill final : public gufo::hip::QwenMoePrefill {
public:
  TapPrefill(gufo::hip::QwenMoePrefill* real,
             std::map<std::uint32_t, LayerTap>* taps,
             const gufo::core::ModelConfig& config)
      : real_(real), taps_(taps), config_(config) {}

  std::uint32_t MaxTokens() const noexcept override {
    return real_->MaxTokens();
  }
  bool Supports(std::uint32_t layer) const noexcept override {
    return real_->Supports(layer);
  }
  bool Run(std::uint32_t layer, std::uint32_t n_tokens,
           const float* router_logits, const __half* x_half,
           const float* shared_out, const float* shared_logit, float* out,
           hipStream_t stream) override {
    if (!real_->Run(layer, n_tokens, router_logits, x_half, shared_out,
                    shared_logit, out, stream))
      return false;
    const auto tap = taps_->find(layer);
    if (tap == taps_->end())
      return true;
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::size_t n = n_tokens;
    tap->second.tokens = n;
    tap->second.router_logits =
        DownloadFrom<float>(router_logits, n * config_.num_experts);
    tap->second.shared_logit = DownloadFrom<float>(shared_logit, n);
    tap->second.x_half =
        DownloadFrom<std::uint16_t>(x_half, n * config_.hidden_size);
    tap->second.out = DownloadFrom<float>(out, n * config_.hidden_size);
    return true;
  }

  std::size_t ResidentBytes() const noexcept override {
    return real_->ResidentBytes();
  }
  std::size_t StreamScratchBytes(hipStream_t stream) override {
    return real_->StreamScratchBytes(stream);
  }

private:
  gufo::hip::QwenMoePrefill* real_;
  std::map<std::uint32_t, LayerTap>* taps_;
  gufo::core::ModelConfig config_;
};

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// Four partial sums: fast enough without reassociation flags.
double Dot(const double* a, const double* b, std::size_t n) {
  double s0 = 0.0;
  double s1 = 0.0;
  double s2 = 0.0;
  double s3 = 0.0;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i)
    s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

double RelativeError(const double* candidate, const double* reference,
                     std::size_t n) {
  double diff = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = candidate[i] - reference[i];
    diff += d * d;
    norm += reference[i] * reference[i];
  }
  return norm > 0.0 ? std::sqrt(diff / norm) : 0.0;
}

struct Routing {
  std::array<int, 64> ids{};
  std::array<double, 64> weights{};
  double shared = 0.0;  // sigmoid of the shared-expert gate logit
};

// Softmax over all logits, the `used` largest (ties to the lower index),
// renormalized to sum one.
Routing Route(const double* logits, std::size_t experts, std::size_t used,
              double shared_logit) {
  Routing r;
  std::vector<int> order(experts);
  for (std::size_t e = 0; e < experts; ++e)
    order[e] = static_cast<int>(e);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return logits[a] > logits[b]; });
  const double peak = logits[order[0]];
  double sum = 0.0;
  for (std::size_t s = 0; s < used; ++s) {
    r.ids[s] = order[s];
    r.weights[s] = std::exp(logits[order[s]] - peak);
    sum += r.weights[s];
  }
  for (std::size_t s = 0; s < used; ++s)
    r.weights[s] /= sum;
  r.shared = Sigmoid(shared_logit);
  return r;
}

std::set<int> IdSet(const std::int32_t* ids, std::size_t used) {
  return {ids, ids + used};
}

struct HostTensor {
  const std::uint8_t* data = nullptr;
  GgmlType type = GgmlType::kF32;
};

HostTensor Tensor(const gufo::core::GgufReader& reader, std::size_t layer,
                  const std::string& name) {
  const auto* info =
      reader.FindTensor("blk." + std::to_string(layer) + "." + name);
  if (info == nullptr)
    throw std::runtime_error("missing tensor " + name);
  return {static_cast<const std::uint8_t*>(info->data), info->type};
}

// One weight matrix [rows][cols] as doubles: the exact value of every weight
// and the value the grouped kernels use, an F16 affine of the code:
// half(code * half(scale) + half(-offset)).
void DecodeMatrix(const HostTensor& tensor, std::size_t first_row,
                  std::size_t rows, std::size_t cols,
                  std::vector<double>& exact, std::vector<double>& half) {
  const std::size_t row_bytes =
      gufo::test_reference::RowBytes(tensor.type, cols);
  std::vector<QuantCode> codes(cols);
  exact.resize(rows * cols);
  half.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    DecodeRow(tensor.type, tensor.data + ((first_row + r) * row_bytes), cols,
              codes.data());
    float scale = NAN;
    float offset = NAN;
    double scale_half = 0.0;
    double bias_half = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const QuantCode& w = codes[c];
      if (w.scale != scale || w.offset != offset) {
        scale = w.scale;
        offset = w.offset;
        scale_half = RoundToHalf(scale);
        bias_half = RoundToHalf(-static_cast<double>(offset));
      }
      exact[(r * cols) + c] = w.Value();
      half[(r * cols) + c] = RoundToHalf((w.code * scale_half) + bias_half);
    }
  }
}

// The activation form of the quantized-activation projections: bfloat16
// values in blocks of 32, each scaled to int8 by its largest magnitude. Float
// operations as in the kernel; the result is the value the GEMM multiplies.
void QuantizeActivation(const float* bf16_values, std::size_t n, double* out) {
  for (std::size_t block = 0; block < n / 32; ++block) {
    const float* v = bf16_values + (block * 32);
    float max_abs = 0.0F;
    for (int i = 0; i < 32; ++i)
      max_abs = std::fmax(max_abs, std::fabs(v[i]));
    const float d = max_abs / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    for (int i = 0; i < 32; ++i) {
      const auto q = static_cast<std::int8_t>(std::round(v[i] * id));
      out[(block * 32) + i] = static_cast<double>(d) * q;
    }
  }
}

// The activation form of the row-wise expert kernels: FP32 values in blocks
// of 32, each scaled to int8 by its largest magnitude.
void QuantizeActivationRows(const float* values, std::size_t n, double* out) {
  for (std::size_t block = 0; block < n / 32; ++block) {
    const float* v = values + (block * 32);
    float max_abs = 0.0F;
    for (int i = 0; i < 32; ++i)
      max_abs = std::fmax(max_abs, std::fabs(v[i]));
    const float d = max_abs / 127.0F;
    const float d_inv = 127.0F / max_abs;
    for (int i = 0; i < 32; ++i) {
      const auto q = static_cast<std::int8_t>(
          max_abs == 0.0F ? 0.0F : std::round(v[i] * d_inv));
      out[(block * 32) + i] = static_cast<double>(d) * q;
    }
  }
}

struct Stat {
  double sum = 0.0;
  double sum2 = 0.0;
  double max = 0.0;
  std::size_t n = 0;
  void Add(double v) {
    sum += v;
    sum2 += v * v;
    max = std::max(max, v);
    ++n;
  }
  [[nodiscard]] double Mean() const {
    return n != 0 ? sum / static_cast<double>(n) : 0.0;
  }
  [[nodiscard]] double MeanSquare() const {
    return n != 0 ? sum2 / static_cast<double>(n) : 0.0;
  }
};

enum Cause : std::size_t {
  kRouter,
  kRequant,
  kHalfInput,
  kHalfWeights,
  kHalfIntermediate,
  kSharedQuantized,
  kAll,
  kCauseCount
};
constexpr std::array<const char*, kCauseCount> kCauseNames = {
    "router split-BF16",
    "Q6_K->Q8_0 down weights",
    "F16 expert input",
    "F16 expert weights",
    "F16 expert intermediates",
    "shared expert int8 activations",
    "all together"};

struct Fixture {
  std::shared_ptr<const gufo::core::GgufReader> reader;
  std::shared_ptr<const gufo::hip::QwenGpuModel> model;
  gufo::hip::QwenMoePrefill* backend = nullptr;
  gufo::hip::QwenGpuArena* arena = nullptr;
  std::size_t hidden = 0;
  std::size_t experts = 0;
  std::size_t used = 0;
  std::size_t ff = 0;
  std::size_t shared_ff = 0;
  bool passed = true;

  void Check(bool ok, const std::string& what) {
    if (!ok) {
      Out() << "FAIL: " << what << '\n';
      passed = false;
    }
  }
};

struct RouterRows {
  std::vector<float> row_logits;      // row-wise kernel, [rows][experts]
  std::vector<float> row_shared;      // [rows]
  std::vector<float> split_logits;    // split-BF16 GEMM, [rows][experts]
  std::vector<float> split_shared;    // [rows]
  std::vector<std::int32_t> row_ids;  // row-wise top-k, [rows][used]
  std::vector<float> row_weights;
  std::vector<std::int32_t> split_ids;  // grouped top-k of the split logits
  std::vector<float> split_weights;
};

// Both production routers and both top-k kernels on the rows in
// arena.d_normed.
RouterRows RunRouters(Fixture& f, const gufo::models::QwenLayerWeights& layer,
                      std::uint32_t layer_index, std::size_t rows) {
  auto& arena = *f.arena;
  hipStream_t stream = arena.stream;
  const Device<float> logits(rows * f.experts);
  const Device<float> shared(rows);
  const Device<std::int32_t> ids(rows * f.used);
  const Device<float> weights(rows * f.used);
  const Device<float> shared_weight(rows);
  RouterRows r;
  gufo::hip::LaunchMoeRouterLogits(
      layer.ffn_gate_inp.data, layer.ffn_gate_inp.type,
      layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type,
      arena.d_normed, logits.data(), shared.data(), f.experts, f.hidden, stream,
      rows);
  gufo::hip::LaunchMoeRouterTopK(logits.data(), shared.data(), ids.data(),
                                 weights.data(), shared_weight.data(),
                                 static_cast<std::uint32_t>(f.experts),
                                 static_cast<std::uint32_t>(f.used),
                                 static_cast<std::uint32_t>(rows), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  r.row_logits = logits.Download(rows * f.experts);
  r.row_shared = shared.Download(rows);
  r.row_ids = ids.Download(rows * f.used);
  r.row_weights = weights.Download(rows * f.used);
  // The split router, as QwenGpuExecutor::MoePrefillFfn launches it.
  const std::uint16_t* split = f.model->RouterSplitWeights(layer_index);
  auto& scratch = arena.moe_prefill;
  if (split == nullptr || arena.hipblaslt_gemm == nullptr)
    throw std::runtime_error("the split router is unavailable");
  gufo::hip::LaunchRouterSplitActivations(arena.d_normed, scratch.router_x,
                                          rows, f.hidden, stream);
  if (!arena.hipblaslt_gemm->RunBf16(split, scratch.router_x, scratch.router_y,
                                     rows, gufo::hip::kRouterSplitRows,
                                     3 * f.hidden, stream))
    throw std::runtime_error("the split router GEMM has no plan");
  gufo::hip::LaunchRouterSplitLogits(scratch.router_y, scratch.router_logits,
                                     scratch.shared_logit, rows, f.experts,
                                     gufo::hip::kRouterSplitRows, stream);
  rocm::RouterTopK(scratch.router_logits, static_cast<std::uint32_t>(f.experts),
                   ids.data(), weights.data(), static_cast<std::uint32_t>(rows),
                   static_cast<std::uint32_t>(f.experts),
                   static_cast<std::uint32_t>(f.used), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  r.split_logits = DownloadFrom<float>(scratch.router_logits, rows * f.experts);
  r.split_shared = DownloadFrom<float>(scratch.shared_logit, rows);
  r.split_ids = ids.Download(rows * f.used);
  r.split_weights = weights.Download(rows * f.used);
  return r;
}

struct RouterReference {
  std::vector<double> logits;     // double precision, [rows][experts]
  std::vector<double> magnitude;  // sum |x||w|, [rows][experts]
};

RouterReference ReferenceRouter(const Fixture& f, std::size_t layer,
                                const std::vector<float>& x, std::size_t rows) {
  const HostTensor router = Tensor(*f.reader, layer, "ffn_gate_inp.weight");
  if (router.type != GgmlType::kF32)
    throw std::runtime_error("the router weights are not F32");
  std::vector<double> w(f.experts * f.hidden);
  std::vector<double> w_abs(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    float value = 0.0F;
    std::memcpy(&value, router.data + (i * sizeof(float)), sizeof(float));
    w[i] = value;
    w_abs[i] = std::fabs(value);
  }
  RouterReference ref;
  ref.logits.resize(rows * f.experts);
  ref.magnitude.resize(rows * f.experts);
  ParallelFor(rows, [&](std::size_t r) {
    std::vector<double> xd(f.hidden);
    std::vector<double> xa(f.hidden);
    for (std::size_t i = 0; i < f.hidden; ++i) {
      xd[i] = x[(r * f.hidden) + i];
      xa[i] = std::fabs(xd[i]);
    }
    for (std::size_t e = 0; e < f.experts; ++e) {
      ref.logits[(r * f.experts) + e] =
          Dot(&w[e * f.hidden], xd.data(), f.hidden);
      ref.magnitude[(r * f.experts) + e] =
          Dot(&w_abs[e * f.hidden], xa.data(), f.hidden);
    }
  });
  return ref;
}

struct RouterSummary {
  double max_row_error = 0.0;
  double max_split_error = 0.0;
  double max_row_ratio = 0.0;  // error / sum |x||w|
  double max_split_ratio = 0.0;
  std::size_t rows = 0;
  std::size_t rows_above_split_margin = 0;
  std::size_t split_set_changes = 0;  // split against the row-wise selection
  double max_dropped_weight = 0.0;
  double sum_dropped_weight = 0.0;
  double max_gap_of_change = 0.0;
  double max_weight_difference = 0.0;
};

// The exact contracts of the two routers on `rows` rows, and what the split
// router changes against the row-wise one.
RouterSummary CheckRouters(Fixture& f, const std::string& label,
                           const RouterRows& r, const RouterReference& ref,
                           std::size_t rows) {
  RouterSummary s;
  s.rows = rows;
  const double split_bound = SplitBound(f.hidden);
  const double row_bound = RowBound(f.hidden);
  std::size_t bound_failures = 0;
  std::size_t set_failures = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    const double* exact = &ref.logits[row * f.experts];
    double magnitude = 0.0;
    for (std::size_t e = 0; e < f.experts; ++e) {
      const double mag = ref.magnitude[(row * f.experts) + e];
      magnitude = std::max(magnitude, mag);
      const double row_error =
          std::fabs(r.row_logits[(row * f.experts) + e] - exact[e]);
      const double split_error =
          std::fabs(r.split_logits[(row * f.experts) + e] - exact[e]);
      s.max_row_error = std::max(s.max_row_error, row_error);
      s.max_split_error = std::max(s.max_split_error, split_error);
      if (mag > 0.0) {
        s.max_row_ratio = std::max(s.max_row_ratio, row_error / mag);
        s.max_split_ratio = std::max(s.max_split_ratio, split_error / mag);
      }
      bound_failures += !(row_error <= row_bound * mag);
      bound_failures += !(split_error <= split_bound * mag);
    }
    const Routing reference = Route(exact, f.experts, f.used + 1, 0.0);
    const double gap =
        exact[reference.ids[f.used - 1]] - exact[reference.ids[f.used]];
    const std::set<int> want(reference.ids.begin(),
                             reference.ids.begin() + f.used);
    const auto row_set = IdSet(&r.row_ids[row * f.used], f.used);
    const auto split_set = IdSet(&r.split_ids[row * f.used], f.used);
    const bool resolved =
        exact[reference.ids[0]] - exact[reference.ids[f.used]] < kTopKSpread;
    if (resolved &&
        gap > std::max(2.0 * row_bound * magnitude, kTopKResolution))
      set_failures += row_set != want;
    if (resolved &&
        gap > std::max(2.0 * split_bound * magnitude, kTopKResolution)) {
      ++s.rows_above_split_margin;
      set_failures += split_set != want;
    }
    if (split_set != row_set) {
      ++s.split_set_changes;
      s.max_gap_of_change = std::max(s.max_gap_of_change, gap);
      double dropped = 0.0;
      for (std::size_t k = 0; k < f.used; ++k) {
        if (split_set.count(r.row_ids[(row * f.used) + k]) == 0)
          dropped += r.row_weights[(row * f.used) + k];
      }
      s.max_dropped_weight = std::max(s.max_dropped_weight, dropped);
      s.sum_dropped_weight += dropped;
    } else {
      for (std::size_t a = 0; a < f.used; ++a) {
        for (std::size_t b = 0; b < f.used; ++b) {
          if (r.row_ids[(row * f.used) + a] == r.split_ids[(row * f.used) + b])
            s.max_weight_difference = std::max(
                s.max_weight_difference,
                std::fabs(
                    static_cast<double>(r.row_weights[(row * f.used) + a]) -
                    r.split_weights[(row * f.used) + b]));
        }
      }
    }
  }
  f.Check(bound_failures == 0, label + ": " + std::to_string(bound_failures) +
                                   " router logits beyond the provable bound");
  f.Check(set_failures == 0,
          label + ": " + std::to_string(set_failures) +
              " expert sets differ from the reference above the margin");
  return s;
}

// The grouped feed-forward on the rows in arena.d_normed (their BF16 copy in
// arena.d_scratch_bf16), launched as QwenGpuExecutor::MoePrefillFfn does after
// the router. Leaves the result in arena.d_ffn_out.
void RunGroupedAfterRouter(Fixture& f,
                           const gufo::models::QwenLayerWeights& layer,
                           std::uint32_t layer_index, std::size_t rows) {
  auto& arena = *f.arena;
  auto& scratch = arena.moe_prefill;
  hipStream_t stream = arena.stream;
  gufo::hip::LaunchFloatToFp16(arena.d_normed, scratch.x_half, rows * f.hidden,
                               stream);
  gufo::hip::LaunchBatchedQuantGEMM(layer.ffn_gate_shexp.type,
                                    layer.ffn_gate_shexp.data,
                                    arena.d_scratch_bf16, scratch.shared_gate,
                                    rows, f.shared_ff, f.hidden, stream);
  gufo::hip::LaunchBatchedQuantGEMM(
      layer.ffn_up_shexp.type, layer.ffn_up_shexp.data, arena.d_scratch_bf16,
      scratch.shared_up, rows, f.shared_ff, f.hidden, stream);
  gufo::hip::LaunchBatchedSwiGLUActivation(
      scratch.shared_gate, scratch.shared_up, scratch.shared_act,
      arena.d_scratch_bf16, rows * f.shared_ff, stream);
  gufo::hip::LaunchBatchedQuantGEMM(layer.ffn_down_shexp.type,
                                    layer.ffn_down_shexp.data,
                                    arena.d_scratch_bf16, scratch.shared_out,
                                    rows, f.hidden, f.shared_ff, stream);
  if (!f.backend->Run(layer_index, static_cast<std::uint32_t>(rows),
                      scratch.router_logits, scratch.x_half, scratch.shared_out,
                      scratch.shared_logit, arena.d_ffn_out, stream))
    throw std::runtime_error("the grouped backend refused the layer");
  HIP_CHECK(hipStreamSynchronize(stream));
}

struct CauseStats {
  std::array<Stat, kCauseCount> cause;
  Stat row_emulation;             // row-wise emulation against the reference
  Stat row_gpu_vs_emulation;      // row-wise device path against its emulation
  Stat grouped_gpu_vs_row_gpu;    // the two device paths against each other
  Stat row_gpu_vs_reference;      // row-wise device path against the host
  Stat grouped_gpu_vs_reference;  // grouped device path against the host
  Stat grouped_gpu_vs_emulation;  // grouped device path against "all"
};

// The host emulation of one layer on a subset of rows: the reference in
// double precision and one variant per cause.
CauseStats EmulateCauses(Fixture& f, std::size_t layer_index,
                         const gufo::models::QwenLayerWeights& layer,
                         const std::vector<float>& x,
                         const std::vector<float>& x_bf16,
                         const RouterRows& routers,
                         const std::vector<float>& row_out,
                         const std::vector<float>& grouped_out,
                         std::size_t rows) {
  const HostTensor gate =
      Tensor(*f.reader, layer_index, "ffn_gate_exps.weight");
  const HostTensor up = Tensor(*f.reader, layer_index, "ffn_up_exps.weight");
  const HostTensor down =
      Tensor(*f.reader, layer_index, "ffn_down_exps.weight");
  const bool requant = down.type == GgmlType::kQ6_K;
  std::vector<std::size_t> picked(std::min(kEmulatedRows, rows));
  for (std::size_t i = 0; i < picked.size(); ++i)
    picked[i] = i * rows / picked.size();

  // Shared expert: exact weights (Q8_0 values are exact in double).
  std::vector<double> shared_gate;
  std::vector<double> shared_up;
  std::vector<double> shared_down;
  std::vector<double> unused;
  DecodeMatrix(Tensor(*f.reader, layer_index, "ffn_gate_shexp.weight"), 0,
               f.shared_ff, f.hidden, shared_gate, unused);
  DecodeMatrix(Tensor(*f.reader, layer_index, "ffn_up_shexp.weight"), 0,
               f.shared_ff, f.hidden, shared_up, unused);
  DecodeMatrix(Tensor(*f.reader, layer_index, "ffn_down_shexp.weight"), 0,
               f.hidden, f.shared_ff, shared_down, unused);

  // The Q8_0 copy the engine really reads, per expert the rows need, from the
  // production conversion on the device.
  std::map<int, std::vector<std::uint8_t>> q8;
  if (requant) {
    std::set<int> needed;
    for (const std::size_t row : picked) {
      for (std::size_t k = 0; k < f.used; ++k) {
        needed.insert(routers.row_ids[(row * f.used) + k]);
        needed.insert(routers.split_ids[(row * f.used) + k]);
      }
    }
    const std::size_t expert_bytes =
        f.hidden * gufo::test_reference::RowBytes(GgmlType::kQ6_K, f.ff);
    const std::size_t q8_bytes = f.hidden * (f.ff / 32) * 34;
    const Device<std::uint8_t> device(q8_bytes);
    for (const int e : needed) {
      rocm::RequantToQ8_0(
          static_cast<const std::uint8_t*>(layer.ffn_down_exps.data) +
              (static_cast<std::size_t>(e) * expert_bytes),
          static_cast<std::uint32_t>(GgmlType::kQ6_K),
          static_cast<std::uint32_t>(f.ff), f.hidden, device.data(),
          f.arena->stream);
      HIP_CHECK(hipStreamSynchronize(f.arena->stream));
      q8[e] = device.Download(q8_bytes);
    }
  }

  CauseStats stats;
  std::mutex mutex;
  ParallelFor(picked.size(), [&](std::size_t index) {
    const std::size_t row = picked[index];
    const std::size_t hidden = f.hidden;
    const std::size_t ff = f.ff;
    std::vector<double> xd(hidden);
    std::vector<double> xh(hidden);
    std::vector<double> xr(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
      xd[i] = x[(row * hidden) + i];
      xh[i] = RoundToHalf(xd[i]);
    }
    QuantizeActivationRows(&x[row * hidden], hidden, xr.data());
    std::vector<double> row_logits(f.experts);
    std::vector<double> split_logits(f.experts);
    for (std::size_t e = 0; e < f.experts; ++e) {
      row_logits[e] = routers.row_logits[(row * f.experts) + e];
      split_logits[e] = routers.split_logits[(row * f.experts) + e];
    }
    const Routing exact_route =
        Route(row_logits.data(), f.experts, f.used, routers.row_shared[row]);
    const Routing split_route = Route(split_logits.data(), f.experts, f.used,
                                      routers.split_shared[row]);

    // Shared expert, exact and through int8 activations.
    std::vector<double> sg(f.shared_ff);
    std::vector<double> su(f.shared_ff);
    std::vector<double> act(f.shared_ff);
    std::vector<double> shared_exact(hidden);
    std::vector<double> shared_quantized(hidden);
    for (std::size_t j = 0; j < f.shared_ff; ++j) {
      sg[j] = Dot(&shared_gate[j * hidden], xd.data(), hidden);
      su[j] = Dot(&shared_up[j * hidden], xd.data(), hidden);
      act[j] = (sg[j] * Sigmoid(sg[j])) * su[j];
    }
    for (std::size_t i = 0; i < hidden; ++i)
      shared_exact[i] =
          Dot(&shared_down[i * f.shared_ff], act.data(), f.shared_ff);
    std::vector<double> xq(hidden);
    QuantizeActivation(&x_bf16[row * hidden], hidden, xq.data());
    std::vector<float> act_bf16(f.shared_ff);
    for (std::size_t j = 0; j < f.shared_ff; ++j) {
      const auto g =
          static_cast<float>(Dot(&shared_gate[j * hidden], xq.data(), hidden));
      const auto u =
          static_cast<float>(Dot(&shared_up[j * hidden], xq.data(), hidden));
      act_bf16[j] = RoundToBf16(static_cast<float>(
          (static_cast<double>(g) * Sigmoid(g)) * static_cast<double>(u)));
    }
    std::vector<double> act_q(f.shared_ff);
    QuantizeActivation(act_bf16.data(), f.shared_ff, act_q.data());
    for (std::size_t i = 0; i < hidden; ++i)
      shared_quantized[i] =
          Dot(&shared_down[i * f.shared_ff], act_q.data(), f.shared_ff);

    // Experts of either routing, every variant of each.
    enum : std::size_t {
      kExact,
      kQ8,
      kIn,
      kW,
      kMid,
      kEvery,
      kRowPath,
      kVariants
    };
    std::map<int, std::array<std::vector<double>, kVariants>> expert_out;
    std::vector<double> wg;
    std::vector<double> wg_half;
    std::vector<double> wu;
    std::vector<double> wu_half;
    std::vector<double> wd;
    std::vector<double> wd_half;
    std::vector<double> wd_q8(hidden * ff);
    std::vector<double> wd_q8_half(hidden * ff);
    // exact, F16 in, F16 w, both, int8 in
    std::array<std::vector<double>, 5> acts;
    for (auto& a : acts)
      a.resize(ff);
    std::set<int> needed(exact_route.ids.begin(),
                         exact_route.ids.begin() + f.used);
    needed.insert(split_route.ids.begin(), split_route.ids.begin() + f.used);
    for (const int e : needed) {
      const auto expert = static_cast<std::size_t>(e);
      DecodeMatrix(gate, expert * ff, ff, hidden, wg, wg_half);
      DecodeMatrix(up, expert * ff, ff, hidden, wu, wu_half);
      DecodeMatrix(down, expert * hidden, hidden, ff, wd, wd_half);
      if (requant) {
        // The stored copy is d * code; the grouped kernel reads it as
        // half(code * d).
        std::vector<QuantCode> codes(ff);
        const auto& bytes = q8.at(e);
        for (std::size_t r = 0; r < hidden; ++r) {
          DecodeRow(GgmlType::kQ8_0, bytes.data() + (r * (ff / 32) * 34), ff,
                    codes.data());
          for (std::size_t c = 0; c < ff; ++c) {
            wd_q8[(r * ff) + c] = codes[c].Value();
            wd_q8_half[(r * ff) + c] = RoundToHalf(codes[c].Value());
          }
        }
      }
      const double* const gate_w[5] = {wg.data(), wg.data(), wg_half.data(),
                                       wg_half.data(), wg.data()};
      const double* const up_w[5] = {wu.data(), wu.data(), wu_half.data(),
                                     wu_half.data(), wu.data()};
      const double* const input[5] = {xd.data(), xh.data(), xd.data(),
                                      xh.data(), xr.data()};
      for (int v = 0; v < 5; ++v) {
        for (std::size_t j = 0; j < ff; ++j) {
          const double g = Dot(gate_w[v] + (j * hidden), input[v], hidden);
          const double u = Dot(up_w[v] + (j * hidden), input[v], hidden);
          acts[v][j] = (u * g) * Sigmoid(g);
        }
      }
      std::vector<double> act_half(ff);
      std::vector<double> act_every(ff);
      std::vector<float> act_float(ff);
      std::vector<double> act_rows(ff);
      for (std::size_t j = 0; j < ff; ++j) {
        act_half[j] = RoundToHalf(acts[0][j]);
        act_every[j] = RoundToHalf(acts[3][j]);
        act_float[j] = static_cast<float>(acts[4][j]);
      }
      QuantizeActivationRows(act_float.data(), ff, act_rows.data());
      auto& out = expert_out[e];
      for (auto& o : out)
        o.resize(hidden);
      for (std::size_t i = 0; i < hidden; ++i) {
        const double* exact_row = &wd[i * ff];
        const double* q8_row = requant ? &wd_q8[i * ff] : exact_row;
        const double* every_row =
            requant ? &wd_q8_half[i * ff] : &wd_half[i * ff];
        out[kExact][i] = Dot(exact_row, acts[0].data(), ff);
        out[kQ8][i] = Dot(q8_row, acts[0].data(), ff);
        out[kIn][i] = Dot(exact_row, acts[1].data(), ff);
        // F16 weights alone: the original down weights displaced by the
        // rounding the grouped kernel adds to the weights it reads.
        out[kW][i] = Dot(exact_row, acts[2].data(), ff) +
                     Dot(every_row, acts[2].data(), ff) -
                     Dot(q8_row, acts[2].data(), ff);
        out[kMid][i] = RoundToHalf(Dot(exact_row, act_half.data(), ff));
        out[kEvery][i] = RoundToHalf(Dot(every_row, act_every.data(), ff));
        out[kRowPath][i] = Dot(exact_row, act_rows.data(), ff);
      }
    }
    const auto combine = [&](const Routing& route, std::size_t variant,
                             const std::vector<double>& shared) {
      std::vector<double> out(hidden, 0.0);
      for (std::size_t s = 0; s < f.used; ++s) {
        const auto& y = expert_out.at(route.ids[s])[variant];
        for (std::size_t i = 0; i < hidden; ++i)
          out[i] += route.weights[s] * y[i];
      }
      for (std::size_t i = 0; i < hidden; ++i)
        out[i] += route.shared * shared[i];
      return out;
    };
    const auto reference = combine(exact_route, kExact, shared_exact);
    std::array<std::vector<double>, kCauseCount> variants = {
        combine(split_route, kExact, shared_exact),
        combine(exact_route, kQ8, shared_exact),
        combine(exact_route, kIn, shared_exact),
        combine(exact_route, kW, shared_exact),
        combine(exact_route, kMid, shared_exact),
        combine(exact_route, kExact, shared_quantized),
        combine(split_route, kEvery, shared_quantized)};
    const auto row_emulation = combine(exact_route, kRowPath, shared_exact);
    std::vector<double> row_gpu(hidden);
    std::vector<double> grouped_gpu(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
      row_gpu[i] = row_out[(row * hidden) + i];
      grouped_gpu[i] = grouped_out[(row * hidden) + i];
    }
    const std::lock_guard lock(mutex);
    for (std::size_t c = 0; c < kCauseCount; ++c)
      stats.cause[c].Add(
          RelativeError(variants[c].data(), reference.data(), hidden));
    stats.row_emulation.Add(
        RelativeError(row_emulation.data(), reference.data(), hidden));
    stats.row_gpu_vs_emulation.Add(
        RelativeError(row_gpu.data(), row_emulation.data(), hidden));
    stats.grouped_gpu_vs_row_gpu.Add(
        RelativeError(grouped_gpu.data(), row_gpu.data(), hidden));
    stats.row_gpu_vs_reference.Add(
        RelativeError(row_gpu.data(), reference.data(), hidden));
    stats.grouped_gpu_vs_reference.Add(
        RelativeError(grouped_gpu.data(), reference.data(), hidden));
    stats.grouped_gpu_vs_emulation.Add(
        RelativeError(grouped_gpu.data(), variants[kAll].data(), hidden));
  });
  return stats;
}

// The projections reading the input norm of a layer, as the prompt pass
// launches them, against the decode projection on the same rows. `d_hidden`
// holds the hidden state after the previous layer.
void MeasureInputProjections(Fixture& f,
                             const gufo::models::QwenLayerWeights& layer,
                             const float* d_hidden, std::size_t rows,
                             std::map<std::string, Stat>& stats) {
  namespace route = gufo::models::qwen;
  auto& arena = *f.arena;
  hipStream_t stream = arena.stream;
  const std::size_t hidden = f.hidden;
  const auto* norm = static_cast<const float*>(layer.attn_norm.data);
  struct Item {
    const char* name;
    const gufo::models::QwenTensorRef* weight;
  };
  const std::vector<Item> items =
      layer.is_full_attention
          ? std::vector<Item>{{"attn_q", &layer.attn_q},
                              {"attn_k", &layer.attn_k},
                              {"attn_v", &layer.attn_v}}
          : std::vector<Item>{{"attn_qkv", &layer.attn_qkv},
                              {"attn_gate", &layer.attn_gate},
                              {"ssm_alpha", &layer.ssm_alpha},
                              {"ssm_beta", &layer.ssm_beta}};
  const auto reads_q8 = [](const gufo::models::QwenTensorRef& w) {
    return w.type == GgmlType::kQ8_0 ||
           gufo::hip::detail::IsNativeWmmaQuant(w.type);
  };
  bool all_q8 = gufo::hip::IsFusedRMSNormQuantizeQ8_1Supported(hidden);
  bool any_q8 = false;
  for (const auto& item : items) {
    all_q8 = all_q8 && reads_q8(*item.weight);
    any_q8 = any_q8 || reads_q8(*item.weight);
  }
  // As in the prompt pass: the activation is quantized from the FP32 row when
  // every reader is quantized, from its BF16 copy otherwise.
  gufo::hip::LaunchBatchedRMSNorm(d_hidden, norm, arena.d_normed,
                                  arena.d_scratch_bf16, rows, hidden, 1e-6F,
                                  stream);
  if (all_q8) {
    gufo::hip::LaunchBatchedFusedRMSNormQuantizeQ8_1(
        d_hidden, nullptr, norm, nullptr, arena.d_scratch_q8_act, rows, hidden,
        1e-6F, stream);
  } else if (any_q8) {
    gufo::hip::LaunchQuantizeActivationQ8_1(
        arena.d_scratch_bf16, arena.d_scratch_q8_act, rows, hidden, stream);
  }
  for (const auto& item : items) {
    const auto& w = *item.weight;
    if (w.empty())
      continue;
    const std::size_t m = w.num_elements / hidden;
    const Device<float> prefill(rows * m);
    const Device<float> decode(rows * m);
    if (reads_q8(w)) {
      gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
          w.type, w.data, arena.d_scratch_q8_act, prefill.data(), rows, m,
          hidden, stream);
    } else {
      const auto resolution = route::ResolveQwenGemmRoute(
          {.type = w.type,
           .batch_size = rows,
           .m = m,
           .k = hidden,
           .mode = route::QwenGemmMode::kHipPrefill});
      if (resolution.route == route::QwenGemmRoute::kHipPrefillBf16Fp32) {
        gufo::hip::LaunchExactBf16GEMMFp32SmallBatch(
            w.data, arena.d_normed, prefill.data(), rows, m, hidden, stream);
      } else if (resolution.route == route::QwenGemmRoute::kHipPrefillF32Blas) {
        const Device<std::uint16_t> conversion(rows * hidden);
        gufo::hip::LaunchHipblasGEMM(arena.hipblas_handle, w.data, false,
                                     arena.d_normed, prefill.data(), rows, m,
                                     hidden, conversion.data(), stream);
        HIP_CHECK(hipStreamSynchronize(stream));
      } else {
        continue;
      }
    }
    for (std::size_t first = 0; first < rows; first += 8) {
      gufo::hip::LaunchDecodeRowsProjection(
          w, arena.d_normed + (first * hidden), decode.data() + (first * m),
          std::min<std::size_t>(8, rows - first), m, hidden, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto a = prefill.Download(rows * m);
    const auto b = decode.Download(rows * m);
    std::vector<double> ad(m);
    std::vector<double> bd(m);
    Stat& stat = stats[std::string(item.name) + " (" +
                       std::string(gufo::core::ToString(w.type)) + ")"];
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t i = 0; i < m; ++i) {
        ad[i] = a[(r * m) + i];
        bd[i] = b[(r * m) + i];
      }
      stat.Add(RelativeError(ad.data(), bd.data(), m));
    }
  }
}

// Constructed router rows: values on bfloat16 rounding boundaries, and a real
// row moved until its eighth and ninth experts nearly tie.
void CheckConstructedRouterRows(Fixture& f, std::size_t layer_index,
                                const gufo::models::QwenLayerWeights& layer,
                                const std::vector<float>& real_row) {
  const std::size_t hidden = f.hidden;
  const HostTensor router =
      Tensor(*f.reader, layer_index, "ffn_gate_inp.weight");
  const auto weight = [&](int e, std::size_t i) {
    float value = 0.0F;
    std::memcpy(&value,
                router.data + ((static_cast<std::size_t>(e) * hidden + i) *
                               sizeof(float)),
                sizeof(float));
    return value;
  };
  std::vector<float> x;
  std::vector<std::string> names;
  const auto add = [&](const std::string& name, const std::vector<float>& row) {
    names.push_back(name);
    x.insert(x.end(), row.begin(), row.end());
  };
  std::vector<float> row(hidden);
  // Exactly halfway between two bfloat16 neighbours (1 + 2^-8 scaled), and
  // one float step to either side.
  for (const int step : {0, 1, -1}) {
    for (std::size_t i = 0; i < hidden; ++i) {
      float value =
          std::ldexp(1.0F + std::ldexp(1.0F, -8), static_cast<int>(i % 9) - 6);
      if (step != 0)
        value = std::nextafter(value, step > 0 ? INFINITY : 0.0F);
      row[i] = (i % 2 == 0) ? value : -value;
    }
    add(step == 0  ? "bf16 midpoints"
        : step > 0 ? "midpoints + 1 ulp"
                   : "midpoints - 1 ulp",
        row);
  }
  // The largest bfloat16 low part: all mantissa bits below the top eight set.
  for (std::size_t i = 0; i < hidden; ++i) {
    const float value =
        std::ldexp(std::nextafter(1.0F + std::ldexp(1.0F, -7), 0.0F),
                   static_cast<int>(i % 5) - 2);
    row[i] = (i % 3 == 0) ? -value : value;
  }
  add("full low mantissa", row);
  for (const int scale : {12, -12}) {
    for (std::size_t i = 0; i < hidden; ++i)
      row[i] = std::ldexp(real_row[i], scale);
    add(scale > 0 ? "real row * 2^12" : "real row * 2^-12", row);
  }
  // Near ties: move the real row along w_ninth - w_eighth until the gap is
  // the target; rounding the moved row to float leaves a gap close to it.
  std::vector<double> logits(f.experts);
  for (std::size_t e = 0; e < f.experts; ++e) {
    double sum = 0.0;
    for (std::size_t i = 0; i < hidden; ++i)
      sum += static_cast<double>(weight(static_cast<int>(e), i)) * real_row[i];
    logits[e] = sum;
  }
  const Routing order = Route(logits.data(), f.experts, f.used + 1, 0.0);
  const int eighth = order.ids[f.used - 1];
  const int ninth = order.ids[f.used];
  double direction2 = 0.0;
  for (std::size_t i = 0; i < hidden; ++i) {
    const double d = static_cast<double>(weight(ninth, i)) - weight(eighth, i);
    direction2 += d * d;
  }
  const double gap0 = logits[eighth] - logits[ninth];
  for (const double target : {0.0, 1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5, 1e-4,
                              -1e-4, 1e-3, -1e-3, 1e-2, -1e-2}) {
    const double t = (gap0 - target) / direction2;
    for (std::size_t i = 0; i < hidden; ++i)
      row[i] = static_cast<float>(
          real_row[i] +
          (t * (static_cast<double>(weight(ninth, i)) - weight(eighth, i))));
    add("near tie, target gap " + std::to_string(target), row);
  }
  const std::size_t rows = names.size();
  HIP_CHECK(hipMemcpy(f.arena->d_normed, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const RouterRows routers =
      RunRouters(f, layer, static_cast<std::uint32_t>(layer_index), rows);
  const RouterReference ref = ReferenceRouter(f, layer_index, x, rows);
  (void)CheckRouters(f,
                     "constructed rows, layer " + std::to_string(layer_index),
                     routers, ref, rows);
  Out() << "constructed router rows, layer " << layer_index
        << " (bound per unit of sum|x||w|: split " << SplitBound(hidden)
        << ", row-wise " << RowBound(hidden) << ")\n";
  for (std::size_t r = 0; r < rows; ++r) {
    const double* exact = &ref.logits[r * f.experts];
    const Routing reference = Route(exact, f.experts, f.used + 1, 0.0);
    const double gap =
        exact[reference.ids[f.used - 1]] - exact[reference.ids[f.used]];
    double magnitude = 0.0;
    double split_ratio = 0.0;
    double row_ratio = 0.0;
    for (std::size_t e = 0; e < f.experts; ++e) {
      const double mag = ref.magnitude[(r * f.experts) + e];
      magnitude = std::max(magnitude, mag);
      split_ratio = std::max(
          split_ratio,
          std::fabs(routers.split_logits[(r * f.experts) + e] - exact[e]) /
              mag);
      row_ratio = std::max(
          row_ratio,
          std::fabs(routers.row_logits[(r * f.experts) + e] - exact[e]) / mag);
    }
    const std::set<int> want(reference.ids.begin(),
                             reference.ids.begin() + f.used);
    Out() << "  " << names[r] << ": gap(8,9) " << gap << ", split margin "
          << (2.0 * SplitBound(hidden) * magnitude)
          << ", error/sum|x||w| split " << split_ratio << " row-wise "
          << row_ratio << ", set equals reference: split "
          << (IdSet(&routers.split_ids[r * f.used], f.used) == want)
          << " row-wise "
          << (IdSet(&routers.row_ids[r * f.used], f.used) == want) << '\n';
  }
}

// Both top-k kernels on identical logits, with exact ties.
void CheckTopKParity(Fixture& f, const std::vector<float>& real_logits,
                     std::size_t real_rows) {
  const std::size_t experts = f.experts;
  const std::size_t used = f.used;
  std::vector<float> logits;
  std::vector<std::vector<int>> expected;  // empty: no fixed expectation
  const auto add = [&](const std::vector<float>& row, std::vector<int> want) {
    logits.insert(logits.end(), row.begin(), row.end());
    expected.push_back(std::move(want));
  };
  std::vector<int> first(used);
  for (std::size_t s = 0; s < used; ++s)
    first[s] = static_cast<int>(s);
  // All equal: the lowest indices win.
  add(std::vector<float>(experts, 0.25F), first);
  // A clear top seven and several candidates tied for the last slot, placed
  // across the 32-lane groups the kernels reduce over.
  for (const auto& tied :
       {std::vector<int>{40, 41}, std::vector<int>{31, 32},
        std::vector<int>{63, 64, 200}, std::vector<int>{255, 128, 33},
        std::vector<int>{96, 224}}) {
    std::vector<float> row(experts, -3.0F);
    std::vector<int> want;
    for (std::size_t s = 0; s + 1 < used; ++s) {
      const int e = static_cast<int>((s * 37) + 5) % static_cast<int>(experts);
      row[e] = 2.0F + static_cast<float>(s);
      want.push_back(e);
    }
    for (const int e : tied)
      row[e] = 1.0F;
    want.push_back(*std::min_element(tied.begin(), tied.end()));
    add(row, want);
  }
  // Every selected expert tied with others: nine equal leaders.
  {
    std::vector<float> row(experts, -1.0F);
    std::vector<int> want;
    for (int j = 0; j < 9; ++j)
      row[static_cast<std::size_t>((j * 29) + 3)] = 0.5F;
    for (int j = 0; j < 8; ++j)
      want.push_back((j * 29) + 3);
    add(row, want);
  }
  const std::size_t constructed = expected.size();
  for (std::size_t r = 0; r < real_rows; ++r) {
    add({real_logits.begin() + static_cast<std::ptrdiff_t>(r * experts),
         real_logits.begin() + static_cast<std::ptrdiff_t>((r + 1) * experts)},
        {});
  }
  const std::size_t rows = expected.size();
  const Device<float> d_logits(rows * experts);
  const Device<float> d_shared(rows);
  const Device<std::int32_t> d_ids(rows * used);
  const Device<float> d_weights(rows * used);
  const Device<float> d_shared_weight(rows);
  d_logits.Upload(logits);
  d_shared.Upload(std::vector<float>(rows, 0.0F));
  hipStream_t stream = f.arena->stream;
  gufo::hip::LaunchMoeRouterTopK(d_logits.data(), d_shared.data(), d_ids.data(),
                                 d_weights.data(), d_shared_weight.data(),
                                 static_cast<std::uint32_t>(experts),
                                 static_cast<std::uint32_t>(used),
                                 static_cast<std::uint32_t>(rows), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto row_ids = d_ids.Download(rows * used);
  const auto row_weights = d_weights.Download(rows * used);
  rocm::RouterTopK(d_logits.data(), static_cast<std::uint32_t>(experts),
                   d_ids.data(), d_weights.data(),
                   static_cast<std::uint32_t>(rows),
                   static_cast<std::uint32_t>(experts),
                   static_cast<std::uint32_t>(used), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto grouped_ids = d_ids.Download(rows * used);
  const auto grouped_weights = d_weights.Download(rows * used);
  std::size_t failures = 0;
  std::size_t real_checked = 0;
  std::size_t real_differ = 0;
  double max_weight_difference = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    const auto a = IdSet(&row_ids[r * used], used);
    const auto b = IdSet(&grouped_ids[r * used], used);
    if (r < constructed) {
      const std::set<int> want(expected[r].begin(), expected[r].end());
      failures += a != want;
      failures += b != want;
      continue;
    }
    std::vector<double> row(experts);
    for (std::size_t e = 0; e < experts; ++e)
      row[e] = logits[(r * experts) + e];
    const Routing order = Route(row.data(), experts, used + 1, 0.0);
    real_differ += a != b;
    if (row[order.ids[used - 1]] - row[order.ids[used]] > kTopKResolution) {
      ++real_checked;
      failures += a != b;
    }
    if (a == b) {
      for (std::size_t i = 0; i < used; ++i) {
        for (std::size_t j = 0; j < used; ++j) {
          if (row_ids[(r * used) + i] == grouped_ids[(r * used) + j])
            max_weight_difference = std::max(
                max_weight_difference,
                std::fabs(static_cast<double>(row_weights[(r * used) + i]) -
                          grouped_weights[(r * used) + j]));
        }
      }
    }
  }
  f.Check(failures == 0, "top-k kernels: " + std::to_string(failures) +
                             " selections differ on identical logits");
  Out() << "top-k on identical logits: " << constructed << " tie fixtures, "
        << real_rows << " real rows (" << real_checked
        << " above the resolution), sets differing on "
        << "real rows " << real_differ << ", max weight difference "
        << max_weight_difference << ',' << (failures == 0 ? "PASS" : "FAIL")
        << '\n';
}

// Reading by decode alone: repeatable, clean after a prompt pass, and how a
// one-token prompt pass differs from it.
void CheckDecodeReference(Fixture& f, gufo::hip::QwenGpuExecutor& executor,
                          std::span<const gufo::tokenization::TokenId> tokens) {
  const std::size_t count = std::min<std::size_t>(tokens.size(), 24);
  const auto decode_from = [&](std::size_t first) {
    std::vector<std::vector<float>> rows;
    for (std::size_t i = first; i < count; ++i) {
      (void)executor.ForwardToken(tokens[i], static_cast<std::uint32_t>(i));
      const auto logits = executor.CopyLastLogits();
      rows.emplace_back(logits.begin(), logits.end());
    }
    return rows;
  };
  executor.Reset();
  const auto first = decode_from(0);
  executor.Reset();
  const auto second = decode_from(0);
  // A grouped prompt pass in between must leave nothing behind.
  executor.Reset();
  (void)executor.ForwardPromptBatch(tokens, 0, true);
  executor.Reset();
  const auto third = decode_from(0);
  f.Check(first == second, "decode-only reading is not repeatable");
  f.Check(first == third, "decode-only reading depends on an earlier pass");
  executor.Reset();
  (void)executor.ForwardPromptBatch(tokens.first(1), 0, true);
  const auto after_prompt = decode_from(1);
  std::size_t differing = 0;
  double max_difference = 0.0;
  for (std::size_t i = 1; i < count; ++i) {
    const auto& a = first[i];
    const auto& b = after_prompt[i - 1];
    differing += std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0;
    for (std::size_t v = 0; v < a.size(); ++v)
      max_difference =
          std::max(max_difference, std::fabs(static_cast<double>(a[v]) - b[v]));
  }
  Out() << "decode-only reading of " << count
        << " tokens: repeatable and clean after a prompt pass,"
        << (first == second && first == third ? "PASS" : "FAIL") << '\n'
        << "  a one-token prompt pass followed by decode differs from it "
        << "in " << differing << " of " << (count - 1)
        << " rows, max |logit difference| " << max_difference << '\n';
}

// A history whose tail was produced by decode: restoring its snapshot is
// exact; reading the same tokens again in one prompt pass is a different
// computation and its difference is only measured.
void CheckSnapshotAgainstReread(
    Fixture& f, gufo::hip::QwenGpuExecutor& executor,
    std::span<const gufo::tokenization::TokenId> tokens) {
  constexpr std::size_t kPrompt = 48;
  constexpr std::size_t kReply = 16;
  constexpr std::size_t kTail = 8;
  if (tokens.size() < kPrompt + kReply + kTail)
    return;
  const std::size_t history = kPrompt + kReply;
  const auto tail = [&] {
    std::vector<std::vector<float>> rows;
    for (std::size_t i = history; i < history + kTail; ++i) {
      (void)executor.ForwardToken(tokens[i], static_cast<std::uint32_t>(i));
      const auto logits = executor.CopyLastLogits();
      rows.emplace_back(logits.begin(), logits.end());
    }
    return rows;
  };
  executor.Reset();
  (void)executor.ForwardPromptBatch(tokens.first(kPrompt), 0, true);
  for (std::size_t i = kPrompt; i < history; ++i)
    (void)executor.ForwardToken(tokens[i], static_cast<std::uint32_t>(i));
  const auto snapshot =
      executor.SaveSnapshot(static_cast<std::uint32_t>(history));
  const auto before = tail();
  executor.Reset();
  executor.RestoreSnapshot(*snapshot);
  const auto restored = tail();
  f.Check(before == restored,
          "a restored snapshot does not continue as the saved state did");
  executor.Reset();
  (void)executor.ForwardPromptBatch(tokens.first(history), 0, true);
  const auto reread = tail();
  std::size_t differing = 0;
  std::size_t top_changes = 0;
  double max_difference = 0.0;
  double min_margin = INFINITY;
  for (std::size_t r = 0; r < kTail; ++r) {
    const auto& a = before[r];
    const auto& b = reread[r];
    differing += a != b;
    for (std::size_t v = 0; v < a.size(); ++v)
      max_difference =
          std::max(max_difference, std::fabs(static_cast<double>(a[v]) - b[v]));
    top_changes += (std::max_element(a.begin(), a.end()) - a.begin()) !=
                   (std::max_element(b.begin(), b.end()) - b.begin());
    std::vector<float> sorted(a);
    std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(),
                      std::greater<>());
    min_margin = std::min<double>(min_margin, sorted[0] - sorted[1]);
  }
  Out() << "snapshot of a history with " << kReply
        << " decoded tokens: restored continuation bitwise equal,"
        << (before == restored ? "PASS" : "FAIL") << '\n'
        << "  the same " << history
        << " tokens read again by one prompt pass: " << differing << " of "
        << kTail << " following rows differ, max |logit difference| "
        << max_difference << ", top-1 changes " << top_changes
        << ", smallest top-1 margin of the saved state " << min_margin << '\n';
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(ENGINE_ENABLE_HIP)
  if (argc > 2) {
    std::cerr << "Usage: qwen_moe_prefill_rows_test [MODEL.gguf]\n";
    return 2;
  }
  const char* env = std::getenv("GUFO_QWEN_LAYERED_PREFILL_MODEL");
  const std::string path = argc > 1 ? argv[1] : env != nullptr ? env : "";
  if (path.empty()) {
    std::cerr << "GUFO_QWEN_LAYERED_PREFILL_MODEL is not set; skipping\n";
    return 77;
  }
  try {
    rocm::RegisterQwenMoePrefill();
    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(path, &error);
    if (!reader_owner)
      throw std::runtime_error("open: " + error);
    Fixture f;
    f.reader = std::move(reader_owner);
    f.model = gufo::hip::QwenGpuModel::CreateFromGguf(f.reader, &error);
    if (!f.model)
      throw std::runtime_error("model: " + error);
    const auto& config = f.model->GetConfig();
    if (!config.IsMoe()) {
      std::cerr << "the model has no experts; skipping\n";
      return 77;
    }
    const auto& weights = f.model->GetWeights();
    f.hidden = config.hidden_size;
    f.experts = config.num_experts;
    f.used = config.num_experts_used;
    f.ff = config.expert_ff;
    f.shared_ff = config.shared_expert_ff;

    // The production backend, and a tap around it for the model to use.
    constexpr std::uint32_t kContext = 4096;
    const auto backend =
        gufo::hip::CreateQwenMoePrefill(weights, kContext, &error);
    if (!backend) {
      std::cerr << "no grouped-expert prefill for this model; skipping\n";
      return 77;
    }
    f.backend = backend.get();
    std::map<std::uint32_t, LayerTap> taps;
    gufo::hip::SetQwenMoePrefillFactory(
        [&](const gufo::models::QwenModelWeights&, std::uint32_t,
            std::string*) -> std::unique_ptr<gufo::hip::QwenMoePrefill> {
          return std::make_unique<TapPrefill>(backend.get(), &taps, config);
        },
        {.resident_bytes =
             [](const gufo::models::QwenModelWeights& w) {
               return gufo::hip::EstimateQwenMoePrefillResidentBytes(w);
             },
         .stream_scratch_bytes = [](const gufo::core::ModelConfig&,
                                    std::uint32_t) -> std::size_t {
           return 0;
         }});
    auto executor =
        gufo::hip::QwenGpuExecutor::Create(f.model, &error, kContext);
    if (!executor)
      throw std::runtime_error("executor: " + error);
    gufo::hip::QwenGpuArena arena(config, kContext);
    f.arena = &arena;

    std::vector<gufo::tokenization::TokenId> tokens;
    if (const char* ids_path = std::getenv("GUFO_QWEN_PREFILL_ROWS_IDS")) {
      std::ifstream in(ids_path);
      for (std::uint32_t id; in >> id;)
        tokens.push_back(id);
    } else {
      tokens = f.model->GetTokenizer().Encode(kText);
    }
    if (tokens.size() < 128)
      throw std::runtime_error("fewer than 128 tokens to read");
    tokens.resize(std::min(tokens.size(), kRows));
    const std::size_t rows = tokens.size();
    const std::size_t hidden = f.hidden;
    Out() << "rows=" << rows << " hidden=" << hidden << " experts=" << f.experts
          << " used=" << f.used << '\n';

    const std::size_t layer_count = weights.layers.size();
    std::set<std::size_t> emulated = {1, layer_count / 2, layer_count - 1};
    for (std::size_t l = 0; l < layer_count; ++l) {
      if (weights.layers[l].ffn_down_exps.type == GgmlType::kQ6_K)
        emulated.insert(l);
    }
    const Device<float> d_hidden(rows * hidden);
    const Device<float> d_row_out(rows * hidden);
    gufo::hip::WarmMoeDecode(weights, config, arena.moe, d_hidden.data(),
                             arena.stream);

    RouterSummary total;
    std::map<std::string, Stat> projection_stats;
    Stat grouped_vs_row_all;
    std::vector<float> saved_row;
    std::vector<float> saved_logits;
    for (std::size_t first_layer = 0; first_layer < layer_count;
         first_layer += kCapturedLayersPerPass) {
      std::vector<std::uint32_t> layers;
      taps.clear();
      for (std::size_t l = first_layer;
           l < std::min(layer_count, first_layer + kCapturedLayersPerPass);
           ++l) {
        if (!backend->Supports(static_cast<std::uint32_t>(l)))
          continue;
        layers.push_back(static_cast<std::uint32_t>(l));
        taps[static_cast<std::uint32_t>(l)];
      }
      if (layers.empty())
        continue;
      executor->Reset();
      executor->SetPromptHiddenCapture(true, layers);
      (void)executor->ForwardPromptBatch(tokens, 0, false);
      const auto captured_span = executor->GetPromptHiddenStates();
      const std::vector<float> captured(captured_span.begin(),
                                        captured_span.end());
      executor->SetPromptHiddenCapture(false);
      if (captured.size() != rows * layers.size() * hidden)
        throw std::runtime_error("unexpected hidden capture size");
      for (std::size_t li = 0; li < layers.size(); ++li) {
        const std::uint32_t l = layers[li];
        const auto& layer = weights.layers[l];
        const LayerTap& tap = taps.at(l);
        if (tap.tokens != rows)
          throw std::runtime_error("the grouped path did not run in the pass");
        if (li != 0 && layers[li - 1] + 1 == l) {
          std::vector<float> previous(rows * hidden);
          for (std::size_t r = 0; r < rows; ++r)
            std::copy_n(&captured[((r * layers.size()) + li - 1) * hidden],
                        hidden, &previous[r * hidden]);
          d_hidden.Upload(previous);
          MeasureInputProjections(f, layer, d_hidden.data(), rows,
                                  projection_stats);
        }
        // Input of the feed-forward norm: hidden after the layer minus the
        // feed-forward output the pass added to it.
        std::vector<float> mid(rows * hidden);
        for (std::size_t r = 0; r < rows; ++r) {
          for (std::size_t i = 0; i < hidden; ++i)
            mid[(r * hidden) + i] =
                captured[(((r * layers.size()) + li) * hidden) + i] -
                tap.out[(r * hidden) + i];
        }
        d_hidden.Upload(mid);
        gufo::hip::LaunchBatchedRMSNorm(
            d_hidden.data(), static_cast<const float*>(layer.ffn_norm.data),
            arena.d_normed, arena.d_scratch_bf16, rows, hidden, 1e-6F,
            arena.stream);
        HIP_CHECK(hipStreamSynchronize(arena.stream));
        const auto x = DownloadFrom<float>(arena.d_normed, rows * hidden);
        const auto x_bf16_bits =
            DownloadFrom<std::uint16_t>(arena.d_scratch_bf16, rows * hidden);
        std::vector<float> x_bf16(rows * hidden);
        std::size_t half_equal = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
          const std::uint32_t bits = static_cast<std::uint32_t>(x_bf16_bits[i])
                                     << 16;
          std::memcpy(&x_bf16[i], &bits, sizeof(float));
          half_equal +=
              gufo::test_reference::DoubleToHalfBits(x[i]) == tap.x_half[i];
        }
        const double recovered =
            static_cast<double>(half_equal) / static_cast<double>(x.size());
        // Rows of another layer or position would agree in almost no
        // element; the recovered rows differ by a rounding of the residual
        // add in a few.
        if (recovered < 0.9)
          throw std::runtime_error("the feed-forward input was not recovered");

        gufo::hip::MoeFfnRows(layer, config, arena, arena.d_normed,
                              d_row_out.data(), rows);
        HIP_CHECK(hipStreamSynchronize(arena.stream));
        const auto row_out = d_row_out.Download(rows * hidden);
        const RouterRows routers = RunRouters(f, layer, l, rows);
        RunGroupedAfterRouter(f, layer, l, rows);
        const auto grouped_out =
            DownloadFrom<float>(arena.d_ffn_out, rows * hidden);

        // The mirrored launches against the production pass itself.
        double pass_router = 0.0;
        for (std::size_t i = 0; i < routers.split_logits.size(); ++i)
          pass_router =
              std::max(pass_router,
                       std::fabs(static_cast<double>(routers.split_logits[i]) -
                                 tap.router_logits[i]));
        Stat grouped_vs_pass;
        Stat grouped_vs_row;
        std::vector<double> a(hidden);
        std::vector<double> b(hidden);
        std::vector<double> c(hidden);
        for (std::size_t r = 0; r < rows; ++r) {
          for (std::size_t i = 0; i < hidden; ++i) {
            a[i] = grouped_out[(r * hidden) + i];
            b[i] = tap.out[(r * hidden) + i];
            c[i] = row_out[(r * hidden) + i];
          }
          grouped_vs_pass.Add(RelativeError(a.data(), b.data(), hidden));
          grouped_vs_row.Add(RelativeError(a.data(), c.data(), hidden));
          grouped_vs_row_all.Add(RelativeError(a.data(), c.data(), hidden));
        }
        const RouterReference ref = ReferenceRouter(f, l, x, rows);
        const RouterSummary s =
            CheckRouters(f, "layer " + std::to_string(l), routers, ref, rows);
        total.rows += s.rows;
        total.rows_above_split_margin += s.rows_above_split_margin;
        total.split_set_changes += s.split_set_changes;
        total.sum_dropped_weight += s.sum_dropped_weight;
        total.max_dropped_weight =
            std::max(total.max_dropped_weight, s.max_dropped_weight);
        total.max_gap_of_change =
            std::max(total.max_gap_of_change, s.max_gap_of_change);
        total.max_split_error =
            std::max(total.max_split_error, s.max_split_error);
        total.max_row_error = std::max(total.max_row_error, s.max_row_error);
        total.max_split_ratio =
            std::max(total.max_split_ratio, s.max_split_ratio);
        total.max_row_ratio = std::max(total.max_row_ratio, s.max_row_ratio);
        total.max_weight_difference =
            std::max(total.max_weight_difference, s.max_weight_difference);
        Out() << "layer=" << l
              << " down=" << gufo::core::ToString(layer.ffn_down_exps.type)
              << " input recovered " << (100.0 * recovered)
              << " % of F16 values; mirror vs pass: router max|d| "
              << pass_router << ", output rel mean " << grouped_vs_pass.Mean()
              << '\n'
              << "  router max|error| split " << s.max_split_error
              << " row-wise " << s.max_row_error << "; error/sum|x||w| split "
              << s.max_split_ratio << " row-wise " << s.max_row_ratio
              << "; expert sets changed by split " << s.split_set_changes << '/'
              << rows << " (dropped weight max " << s.max_dropped_weight
              << "), rows above split margin " << s.rows_above_split_margin
              << '\n'
              << "  grouped vs row-wise output, relative L2 per row: mean "
              << grouped_vs_row.Mean() << " max " << grouped_vs_row.max << '\n';
        if (emulated.count(l) != 0) {
          const CauseStats causes = EmulateCauses(
              f, l, layer, x, x_bf16, routers, row_out, grouped_out, rows);
          double sum2 = 0.0;
          for (std::size_t cidx = 0; cidx < kAll; ++cidx)
            sum2 += causes.cause[cidx].MeanSquare();
          Out() << "  causes on " << causes.cause[0].n
                << " rows, relative L2 against the double reference "
                   "(mean, max, share of summed squares):\n";
          for (std::size_t cidx = 0; cidx < kCauseCount; ++cidx) {
            const Stat& st = causes.cause[cidx];
            Out() << "    " << kCauseNames[cidx] << ": " << st.Mean() << ", "
                  << st.max;
            if (cidx < kAll && sum2 > 0.0)
              Out() << ", " << (100.0 * st.MeanSquare() / sum2) << " %";
            Out() << '\n';
          }
          Out() << "    row-wise path, int8 expert activations (emulated): "
                << causes.row_emulation.Mean() << ", "
                << causes.row_emulation.max << '\n'
                << "    device row-wise path vs its emulation: mean "
                << causes.row_gpu_vs_emulation.Mean() << " max "
                << causes.row_gpu_vs_emulation.max << '\n'
                << "    device grouped path vs device row-wise path: mean "
                << causes.grouped_gpu_vs_row_gpu.Mean() << " max "
                << causes.grouped_gpu_vs_row_gpu.max << '\n'
                << "    device row-wise path vs reference: mean "
                << causes.row_gpu_vs_reference.Mean() << " max "
                << causes.row_gpu_vs_reference.max << '\n'
                << "    device grouped path vs reference: mean "
                << causes.grouped_gpu_vs_reference.Mean() << " max "
                << causes.grouped_gpu_vs_reference.max << '\n'
                << "    device grouped path vs \"all together\": mean "
                << causes.grouped_gpu_vs_emulation.Mean() << " max "
                << causes.grouped_gpu_vs_emulation.max << '\n';
          // Validity of the host side, not a quality gate: each emulation
          // must account for most of its device path's distance from the
          // reference. Wrong decoders or a wrong model of a path leave the
          // emulation as far from the device as the reference is.
          f.Check(causes.row_gpu_vs_emulation.Mean() <
                          0.5 * causes.row_gpu_vs_reference.Mean() &&
                      causes.grouped_gpu_vs_emulation.Mean() <
                          0.5 * causes.grouped_gpu_vs_reference.Mean(),
                  "layer " + std::to_string(l) +
                      ": the host emulation does not account for the device "
                      "paths");
        }
        if (l == layer_count / 2) {
          saved_row.assign(x.begin() + static_cast<std::ptrdiff_t>(hidden),
                           x.begin() + static_cast<std::ptrdiff_t>(2 * hidden));
          saved_logits = routers.row_logits;
        }
      }
    }
    Out() << "router, all layers: " << total.rows << " rows, "
          << total.rows_above_split_margin
          << " above the split margin; expert set changed by the split "
             "router in "
          << total.split_set_changes << " rows ("
          << (100.0 * static_cast<double>(total.split_set_changes) /
              static_cast<double>(std::max<std::size_t>(total.rows, 1)))
          << " %), dropped weight mean "
          << (total.split_set_changes != 0
                  ? total.sum_dropped_weight /
                        static_cast<double>(total.split_set_changes)
                  : 0.0)
          << " max " << total.max_dropped_weight
          << ", largest gap of a changed row " << total.max_gap_of_change
          << "\n  max|error| split " << total.max_split_error << " row-wise "
          << total.max_row_error << "; max error/sum|x||w| split "
          << total.max_split_ratio << " (bound " << SplitBound(hidden)
          << ") row-wise " << total.max_row_ratio << " (bound "
          << RowBound(hidden) << "); max weight difference on equal sets "
          << total.max_weight_difference << '\n'
          << "grouped vs row-wise output, all layers: mean "
          << grouped_vs_row_all.Mean() << " max " << grouped_vs_row_all.max
          << '\n';
    Out() << "input projections, prompt pass against decode on the same "
             "rows, relative L2 per row (mean, max, rows):\n";
    for (const auto& [name, stat] : projection_stats)
      Out() << "  " << name << ": " << stat.Mean() << ", " << stat.max << ", "
            << stat.n << '\n';
    if (!saved_row.empty()) {
      CheckConstructedRouterRows(f, layer_count / 2,
                                 weights.layers[layer_count / 2], saved_row);
      CheckTopKParity(f, saved_logits, std::min<std::size_t>(rows, 64));
    }
    CheckDecodeReference(f, *executor, tokens);
    CheckSnapshotAgainstReread(f, *executor, tokens);
    Out() << (f.passed ? "PASS" : "FAIL") << '\n';
    return f.passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "qwen_moe_prefill_rows_test: " << failure.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  Out() << "HIP disabled, skipping Qwen MoE prefill rows test.\n";
  return 77;
#endif
}
