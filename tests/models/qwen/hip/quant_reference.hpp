// Host references for the MoE prefill arithmetic tests: K-quant and Q8_0 rows
// decoded from the GGUF bytes by the format definition (ggml-quants.c), with
// no engine decoder involved, and exact conversions to the narrow float
// formats the grouped-expert prefill rounds through.
#ifndef GUFO_TESTS_MODELS_QWEN_HIP_QUANT_REFERENCE_HPP_
#define GUFO_TESTS_MODELS_QWEN_HIP_QUANT_REFERENCE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::test_reference {

/// The value of an IEEE binary16 bit pattern.
inline double HalfBitsToDouble(std::uint16_t bits) {
  const int exponent = (bits >> 10) & 0x1F;
  const int fraction = bits & 0x3FF;
  double value = 0.0;
  if (exponent == 0) {
    value = std::ldexp(static_cast<double>(fraction), -24);
  } else if (exponent == 31) {
    value = fraction == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(static_cast<double>(fraction + 1024), exponent - 25);
  }
  return (bits & 0x8000U) != 0 ? -value : value;
}

/// Round to nearest even binary16 in one step from a double.
inline std::uint16_t DoubleToHalfBits(double value) {
  const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
  const double magnitude = std::fabs(value);
  if (std::isnan(magnitude)) {
    return 0x7E00U;
  }
  if (magnitude == 0.0) {
    return sign;
  }
  int exponent = 0;
  (void)std::frexp(magnitude, &exponent);  // magnitude in [2^(e-1), 2^e)
  // Spacing of binary16 around the value: 2^-24 for subnormals.
  const int unit = exponent - 11 < -24 ? -24 : exponent - 11;
  // nearbyint rounds half to even in the default rounding mode, and the
  // scaling by a power of two is exact.
  const double scaled = std::nearbyint(std::ldexp(magnitude, -unit));
  const double rounded = std::ldexp(scaled, unit);
  if (rounded >= 65520.0) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (rounded < std::ldexp(1.0, -14)) {
    return static_cast<std::uint16_t>(
        sign | static_cast<std::uint16_t>(std::ldexp(rounded, 24)));
  }
  int rounded_exponent = 0;
  const double mantissa = std::frexp(rounded, &rounded_exponent);  // [0.5, 1)
  const auto fraction =
      static_cast<std::uint16_t>(std::ldexp(mantissa, 11) - 1024.0);
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>((rounded_exponent + 14) << 10) |
      fraction);
}

inline double RoundToHalf(double value) {
  return HalfBitsToDouble(DoubleToHalfBits(value));
}

/// bfloat16 keeps the top sixteen bits of a float, rounded to nearest even:
/// the conversion of the router split and of the BF16 activation copies.
inline float RoundToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  bits &= 0xFFFF0000U;
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline double ReadHalf(const std::uint8_t* bytes) {
  return HalfBitsToDouble(
      static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8)));
}

/// One weight of a quantized row in the terms of its format: the integer code
/// and the float scale and offset of its sub-block; value = scale * code -
/// offset. Scale and offset are exact in float (a half times a small integer).
struct QuantCode {
  float scale;
  float offset;
  std::int16_t code;

  [[nodiscard]] double Value() const {
    return (static_cast<double>(scale) * code) - static_cast<double>(offset);
  }
};

inline std::size_t RowBytes(core::GgmlType type, std::size_t k) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return k / 256 * 144;
    case core::GgmlType::kQ5_K:
      return k / 256 * 176;
    case core::GgmlType::kQ6_K:
      return k / 256 * 210;
    case core::GgmlType::kQ8_0:
      return k / 32 * 34;
    default:
      throw std::invalid_argument("no host reference for this weight type");
  }
}

// The 6-bit scale and minimum of sub-block j of a Q4_K or Q5_K super-block
// (get_scale_min_k4).
inline void ScaleMinK4(int j, const std::uint8_t* q, int& scale, int& minimum) {
  if (j < 4) {
    scale = q[j] & 63;
    minimum = q[j + 4] & 63;
  } else {
    scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    minimum = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

/// Decodes one row of `k` weights (dequantize_row_q4_K, _q5_K, _q6_K, _q8_0).
inline void DecodeRow(core::GgmlType type, const std::uint8_t* row,
                      std::size_t k, QuantCode* out) {
  switch (type) {
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K: {
      const bool q5 = type == core::GgmlType::kQ5_K;
      const std::size_t block_bytes = q5 ? 176 : 144;
      for (std::size_t block = 0; block < k / 256; ++block) {
        const std::uint8_t* b = row + (block * block_bytes);
        const double d = ReadHalf(b);
        const double dmin = ReadHalf(b + 2);
        const std::uint8_t* scales = b + 4;
        const std::uint8_t* qh = b + 16;  // Q5_K only
        const std::uint8_t* ql = b + (q5 ? 16 + 32 : 16);
        QuantCode* y = out + (block * 256);
        for (int pair = 0; pair < 4; ++pair) {
          int sc[2];
          int mn[2];
          ScaleMinK4(2 * pair, scales, sc[0], mn[0]);
          ScaleMinK4((2 * pair) + 1, scales, sc[1], mn[1]);
          for (int l = 0; l < 32; ++l) {
            const std::uint8_t byte = ql[(pair * 32) + l];
            for (int half = 0; half < 2; ++half) {
              int code = half == 0 ? (byte & 0xF) : (byte >> 4);
              if (q5 && ((qh[l] >> ((2 * pair) + half)) & 1) != 0) {
                code += 16;
              }
              QuantCode& w = y[(pair * 64) + (half * 32) + l];
              w.scale = static_cast<float>(d * sc[half]);
              w.offset = static_cast<float>(dmin * mn[half]);
              w.code = static_cast<std::int16_t>(code);
            }
          }
        }
      }
      return;
    }
    case core::GgmlType::kQ6_K: {
      for (std::size_t block = 0; block < k / 256; ++block) {
        const std::uint8_t* ql = row + (block * 210);
        const std::uint8_t* qh = ql + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(qh + 64);
        const double d = ReadHalf(qh + 64 + 16);
        QuantCode* y = out + (block * 256);
        for (int n = 0; n < 2; ++n) {
          for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const std::uint8_t h = qh[(n * 32) + l];
            const std::uint8_t lo0 = ql[(n * 64) + l];
            const std::uint8_t lo1 = ql[(n * 64) + l + 32];
            const int codes[4] = {((lo0 & 0xF) | (((h >> 0) & 3) << 4)) - 32,
                                  ((lo1 & 0xF) | (((h >> 2) & 3) << 4)) - 32,
                                  ((lo0 >> 4) | (((h >> 4) & 3) << 4)) - 32,
                                  ((lo1 >> 4) | (((h >> 6) & 3) << 4)) - 32};
            for (int quarter = 0; quarter < 4; ++quarter) {
              QuantCode& w = y[(n * 128) + (quarter * 32) + l];
              w.scale =
                  static_cast<float>(d * scales[(n * 8) + is + (quarter * 2)]);
              w.offset = 0.0F;
              w.code = static_cast<std::int16_t>(codes[quarter]);
            }
          }
        }
      }
      return;
    }
    case core::GgmlType::kQ8_0: {
      for (std::size_t block = 0; block < k / 32; ++block) {
        const std::uint8_t* b = row + (block * 34);
        const auto d = static_cast<float>(ReadHalf(b));
        for (int i = 0; i < 32; ++i) {
          QuantCode& w = out[(block * 32) + i];
          w.scale = d;
          w.offset = 0.0F;
          w.code = static_cast<std::int8_t>(b[2 + i]);
        }
      }
      return;
    }
    default:
      throw std::invalid_argument("no host reference for this weight type");
  }
}

/// The Q8_0 re-encoding of 32 weights the grouped down projection reads for
/// Q6_K experts, in IEEE single precision: scale = half(max|v| / 127), codes
/// = round-to-nearest-even(v / scale) clamped to +-127. Writes 34 bytes.
inline void RequantBlockQ8_0(const QuantCode* weights, std::uint8_t* out) {
  float v[32];
  float amax = 0.0F;
  for (int i = 0; i < 32; ++i) {
    // scale * code is exact in float for Q6_K (18-bit scale, 6-bit code).
    v[i] = static_cast<float>(weights[i].Value());
    amax = std::fmax(amax, std::fabs(v[i]));
  }
  const std::uint16_t scale_bits =
      DoubleToHalfBits(static_cast<double>(amax / 127.0F));
  const auto scale = static_cast<float>(HalfBitsToDouble(scale_bits));
  const float inverse = amax > 0.0F ? 1.0F / scale : 0.0F;
  out[0] = static_cast<std::uint8_t>(scale_bits & 0xFF);
  out[1] = static_cast<std::uint8_t>(scale_bits >> 8);
  for (int i = 0; i < 32; ++i) {
    const float code =
        std::fmin(127.0F, std::fmax(-127.0F, std::nearbyint(v[i] * inverse)));
    out[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
  }
}

}  // namespace gufo::test_reference

#endif  // GUFO_TESTS_MODELS_QWEN_HIP_QUANT_REFERENCE_HPP_
