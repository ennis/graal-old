#pragma once
#include <cassert>
#include <graal/glad.h>
#include <string_view>

namespace graal {

/// @brief
enum class data_type {
  float_,  ///< 32-bit floats
  unorm32, ///< normalized 32-bit unsigned integer
  snorm32, ///< normalized 32-bit signed integer
  unorm16, ///< normalized 16-bit unsigned integer
  snorm16, ///< normalized 16-bit signed integer
  unorm8,  ///< normalized 8-bit unsigned integer
  snorm8,  ///< normalized 8-bit signed integer

  uscaled32, ///< scaled 32-bit unsigned integer
  sscaled32, ///< scaled 32-bit signed integer
  uscaled16, ///< scaled 16-bit unsigned integer
  sscaled16, ///< scaled 16-bit signed integer
  uscaled8,  ///< scaled 8-bit unsigned integer
  sscaled8,  ///< scaled 8-bit signed integer

  uint32, ///< 32-bit unsigned integer
  sint32, ///< 32-bit signed integer
  uint16, ///< 16-bit unsigned integer
  sint16, ///< 16-bit signed integer
  uint8,  ///< 8-bit unsigned integer
  sint8,  ///< 8-bit signed integer

  unorm2_10_10_10_pack32, ///< 4 normalized unsigned integers packed into one
                          ///< 32-bit integer
  snorm2_10_10_10_pack32, ///< 4 normalized signed integers packed into one
                          ///< 32-bit integer
  float10_11_11_pack32,   ///< 3 low-bitdepth floats packed into one 32-bit
                          ///< integer
};

/// @brief
/// See https://www.khronos.org/opengl/wiki/Normalized_Integer.
enum class data_type_class {
  float_,              ///< Floating-point data
  unsigned_normalized, ///< Normalized unsigned integers
  signed_normalized,   ///< Normalized signed integers
  unsigned_scaled,     ///< Scaled unsigned integers
  signed_scaled,       ///< Scaled signed integers
  unsigned_integer,    ///< Unsigned integers
  signed_integer,      ///< Signed integers
};

inline constexpr data_type_class get_data_type_class(data_type ty) noexcept {
  // clang-format off
  switch (ty) {
    case data_type::float_: return data_type_class::float_;
    case data_type::unorm32: return data_type_class::unsigned_normalized;
    case data_type::snorm32: return data_type_class::signed_normalized;
    case data_type::unorm16: return data_type_class::unsigned_normalized;
    case data_type::snorm16: return data_type_class::signed_normalized;
    case data_type::unorm8: return data_type_class::unsigned_normalized;
    case data_type::snorm8: return data_type_class::signed_normalized;
    case data_type::uscaled32: return data_type_class::unsigned_scaled;
    case data_type::sscaled32: return data_type_class::signed_scaled;
    case data_type::uscaled16: return data_type_class::unsigned_scaled;
    case data_type::sscaled16: return data_type_class::signed_scaled;
    case data_type::uscaled8: return data_type_class::unsigned_scaled;
    case data_type::sscaled8: return data_type_class::signed_scaled;
    case data_type::uint32: return data_type_class::unsigned_integer;
    case data_type::sint32: return data_type_class::signed_integer;
    case data_type::uint16: return data_type_class::unsigned_integer;
    case data_type::sint16: return data_type_class::signed_integer;
    case data_type::uint8: return data_type_class::unsigned_integer;
    case data_type::sint8: return data_type_class::signed_integer;
    case data_type::unorm2_10_10_10_pack32: return data_type_class::unsigned_normalized;
    case data_type::snorm2_10_10_10_pack32: return data_type_class::signed_normalized;
    case data_type::float10_11_11_pack32: return data_type_class::float_;
    default: assert(false);
  }
  // clang-format on
}

inline constexpr std::string_view get_data_type_name(data_type ty) noexcept {
  using namespace std::literals;
  // clang-format off
  switch (ty) {
    case data_type::float_: return "float_"sv;
    case data_type::unorm32: return "unorm32"sv;
    case data_type::snorm32: return "snorm32"sv;
    case data_type::unorm16: return "unorm16"sv;
    case data_type::snorm16: return "snorm16"sv;
    case data_type::unorm8: return "unorm8"sv;
    case data_type::snorm8: return "snorm8"sv;
    case data_type::uscaled32: return "uscaled32"sv;
    case data_type::sscaled32: return "sscaled32"sv;
    case data_type::uscaled16: return "uscaled16"sv;
    case data_type::sscaled16: return "sscaled16"sv;
    case data_type::uscaled8: return "uscaled8"sv;
    case data_type::sscaled8: return "sscaled8"sv;
    case data_type::uint32: return "uint32"sv;
    case data_type::sint32: return "sint32"sv;
    case data_type::uint16: return "uint16"sv;
    case data_type::sint16: return "sint16"sv;
    case data_type::uint8: return "uint8"sv;
    case data_type::sint8: return "sint8"sv;
    case data_type::unorm2_10_10_10_pack32: return "unorm2_10_10_10_pack32"sv;
    case data_type::snorm2_10_10_10_pack32: return "snorm2_10_10_10_pack32"sv;
    case data_type::float10_11_11_pack32: return "float10_11_11_pack32"sv;
    default: assert(false);
  }
  // clang-format on
}

inline constexpr GLenum get_gl_component_type(data_type ty) noexcept {
  // clang-format off
  switch (ty) {
    case data_type::float_: return GL_FLOAT;
    case data_type::unorm32: return GL_UNSIGNED_INT;
    case data_type::snorm32: return GL_INT;
    case data_type::unorm16: return GL_UNSIGNED_SHORT;
    case data_type::snorm16: return GL_SHORT;
    case data_type::unorm8: return GL_UNSIGNED_BYTE;
    case data_type::snorm8: return GL_BYTE;
    case data_type::uscaled32: return GL_UNSIGNED_INT;
    case data_type::sscaled32: return GL_INT;
    case data_type::uscaled16: return GL_UNSIGNED_SHORT;
    case data_type::sscaled16: return GL_SHORT;
    case data_type::uscaled8: return GL_UNSIGNED_BYTE;
    case data_type::sscaled8: return GL_BYTE;
    case data_type::uint32: return GL_UNSIGNED_INT;
    case data_type::sint32: return GL_INT;
    case data_type::uint16: return GL_UNSIGNED_SHORT;
    case data_type::sint16: return GL_SHORT;
    case data_type::uint8: return GL_UNSIGNED_BYTE;
    case data_type::sint8: return GL_BYTE;
    case data_type::unorm2_10_10_10_pack32: return GL_UNSIGNED_INT_2_10_10_10_REV;
    case data_type::snorm2_10_10_10_pack32: return GL_INT_2_10_10_10_REV;
    case data_type::float10_11_11_pack32: return GL_UNSIGNED_INT_10F_11F_11F_REV;
    default: assert(false);
  }
  // clang-format on
}

} // namespace graal