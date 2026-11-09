// Vision-tower attention: both kernels against a CPU reference with the
// encoder's rounding (BF16 scores, BF16 scaled scores, FP32 softmax), and the
// fused kernel against the tiled one on a full screenshot-sized shape.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/vision/attention.hpp"

namespace {
namespace vision = gufo::models::qwen::vision;
using Bf16 = hip_bfloat16;
constexpr unsigned H = vision::kVisionHeads, D = vision::kVisionHeadDim;

void Expect(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
float R(float x) { return static_cast<float>(Bf16(x)); }

struct Inputs {
  std::vector<Bf16> q, k, v;
};
Inputs Random(unsigned n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.6F);
  Inputs in;
  for (auto* x : {&in.q, &in.k, &in.v}) {
    x->resize(std::size_t{H} * n * D);
    for (auto& value : *x)
      value = Bf16(normal(rng));
  }
  return in;
}

std::vector<float> Reference(const Inputs& in, unsigned n) {
  std::vector<float> out(std::size_t{n} * H * D);
  std::vector<float> s(n);
  for (unsigned h = 0; h < H; ++h)
    for (unsigned i = 0; i < n; ++i) {
      float m = -INFINITY;
      for (unsigned j = 0; j < n; ++j) {
        float dot = 0;
        for (unsigned d = 0; d < D; ++d)
          dot += static_cast<float>(in.q[(std::size_t{h} * n + i) * D + d]) *
                 static_cast<float>(in.k[(std::size_t{h} * n + j) * D + d]);
        s[j] = R(R(dot) * 0.1178511301977579F);
        m = std::max(m, s[j]);
      }
      double sum = 0;
      for (unsigned j = 0; j < n; ++j) {
        s[j] = std::exp(s[j] - m);
        sum += s[j];
      }
      for (unsigned d = 0; d < D; ++d) {
        double acc = 0;
        for (unsigned j = 0; j < n; ++j)
          acc += s[j] / sum *
                 static_cast<float>(in.v[(std::size_t{h} * n + j) * D + d]);
        out[std::size_t{i} * H * D + h * D + d] = static_cast<float>(acc);
      }
    }
  return out;
}

std::vector<float> Run(vision::AttentionKernel kind, hipblasHandle_t blas,
                       const Inputs& in, unsigned n) {
  const std::size_t bytes = std::size_t{H} * n * D * sizeof(Bf16);
  Bf16 *q, *k, *v, *o;
  Check(hipMalloc(&q, bytes));
  Check(hipMalloc(&k, bytes));
  Check(hipMalloc(&v, bytes));
  Check(hipMalloc(&o, bytes));
  Check(hipMemcpy(q, in.q.data(), bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(k, in.k.data(), bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(v, in.v.data(), bytes, hipMemcpyHostToDevice));
  Check(hipMemset(o, 0xff, bytes));  // NaN pattern: every element must be written
  vision::Attention(kind, blas, q, k, v, o, n, nullptr);
  Check(hipDeviceSynchronize());
  std::vector<Bf16> host(std::size_t{H} * n * D);
  Check(hipMemcpy(host.data(), o, bytes, hipMemcpyDeviceToHost));
  for (auto* p : {q, k, v, o})
    Check(hipFree(p));
  std::vector<float> out(host.size());
  for (std::size_t i = 0; i < host.size(); ++i)
    out[i] = static_cast<float>(host[i]);
  return out;
}

void Compare(const std::vector<float>& got, const std::vector<float>& want,
             double max_tol, double mean_tol, const std::string& what) {
  double worst = 0, total = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double e = std::abs(double{got[i]} - want[i]);
    Expect(std::isfinite(got[i]), what + ": non-finite output");
    worst = std::max(worst, e);
    total += e;
  }
  const double mean = total / got.size();
  std::cout << what << ": max " << worst << " mean " << mean << '\n';
  Expect(worst <= max_tol && mean <= mean_tol, what + ": error too large");
}

}  // namespace

// --bench N: time one attention call of each kernel (warm, best of 5).
void Bench(hipblasHandle_t blas, unsigned n) {
  const auto in = Random(n, 3);
  const std::size_t bytes = std::size_t{H} * n * D * sizeof(Bf16);
  Bf16 *q, *k, *v, *o;
  Check(hipMalloc(&q, bytes));
  Check(hipMalloc(&k, bytes));
  Check(hipMalloc(&v, bytes));
  Check(hipMalloc(&o, bytes));
  Check(hipMemcpy(q, in.q.data(), bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(k, in.k.data(), bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(v, in.v.data(), bytes, hipMemcpyHostToDevice));
  for (const auto kind : {vision::AttentionKernel::kTiled, vision::AttentionKernel::kFused}) {
    float best = 1e30F;
    for (int i = 0; i < 6; ++i) {
      hipEvent_t a, b;
      Check(hipEventCreate(&a));
      Check(hipEventCreate(&b));
      Check(hipEventRecord(a, nullptr));
      vision::Attention(kind, blas, q, k, v, o, n, nullptr);
      Check(hipEventRecord(b, nullptr));
      Check(hipEventSynchronize(b));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, a, b));
      if (i > 0)
        best = std::min(best, ms);
    }
    std::cout << (kind == vision::AttentionKernel::kTiled ? "tiled" : "fused")
              << " n=" << n << " ms=" << best << " x27=" << best * 27 << '\n';
  }
  for (auto* p : {q, k, v, o})
    Check(hipFree(p));
}

int main(int argc, char** argv) {
  try {
    hipblasHandle_t blas;
    if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("hipblasCreate failed");
    if (argc == 3 && std::string(argv[1]) == "--bench") {
      Bench(blas, static_cast<unsigned>(std::stoul(argv[2])));
      return 0;
    }
    // Ragged shapes: fewer rows than a block, block tails and key tails.
    for (const unsigned n : {20U, 129U, 1000U}) {
      const auto in = Random(n, n);
      const auto want = Reference(in, n);
      // One BF16 rounding of the output (values up to ~5) bounds the error.
      Compare(Run(vision::AttentionKernel::kTiled, blas, in, n), want, 0.04,
              2e-3, "tiled n=" + std::to_string(n));
      Compare(Run(vision::AttentionKernel::kFused, blas, in, n), want, 0.04,
              2e-3, "fused n=" + std::to_string(n));
    }
    // Screenshot shape (2923 merged rows x 4 patches): fused against tiled.
    const unsigned n = 11692;
    const auto in = Random(n, 7);
    Compare(Run(vision::AttentionKernel::kFused, blas, in, n),
            Run(vision::AttentionKernel::kTiled, blas, in, n), 0.06, 2e-3,
            "fused vs tiled n=11692");
    hipblasDestroy(blas);
  } catch (const std::exception& e) {
    std::cerr << "vision attention test failed: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen_vision_attention_test: ok\n";
  return 0;
}
