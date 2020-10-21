#pragma once
#include <string_view>
#include <graal/errors.hpp>

namespace graal {

/// @brief
enum class image_format {
  r8_unorm,
  r8g8_unorm,
  r8g8b8_unorm,
  r8g8b8a8_unorm,
  r16_unorm,
  r16g16_unorm,
  r16g16b16_unorm,
  r16g16b16a16_unorm,

  r16_sfloat,
  r16g16_sfloat,
  r16g16b16_sfloat,
  r16g16b16a16_sfloat,

  r32_sfloat,
  r32g32_sfloat,
  r32g32b32_sfloat,
  r32g32b32a32_sfloat,
};

/// @brief
/// @param fmt
/// @return
inline constexpr std::string_view get_image_format_name(image_format fmt) {
  using namespace std::literals;
  switch (fmt) {
  case image_format::r8_unorm:
    return "r8_unorm"sv;
  case image_format::r8g8_unorm:
    return "r8g8_unorm"sv;
  case image_format::r8g8b8_unorm:
    return "r8g8b8_unorm"sv;
  case image_format::r8g8b8a8_unorm:
    return "r8g8b8a8_unorm"sv;
  case image_format::r16_unorm:
    return "r16_unorm"sv;
  case image_format::r16g16_unorm:
    return "r16g16_unorm"sv;
  case image_format::r16g16b16_unorm:
    return "r16g16b16_unorm"sv;
  case image_format::r16g16b16a16_unorm:
    return "r16g16b16a16_unorm"sv;
  case image_format::r16_sfloat:
    return "r16_sfloat"sv;
  case image_format::r16g16_sfloat:
    return "r16g16_sfloat"sv;
  case image_format::r16g16b16_sfloat:
    return "r16g16b16_sfloat"sv;
  case image_format::r16g16b16a16_sfloat:
    return "r16g16b16a16_sfloat"sv;
  case image_format::r32_sfloat:
    return "r32_sfloat"sv;
  case image_format::r32g32_sfloat:
    return "r32g32_sfloat"sv;
  case image_format::r32g32b32_sfloat:
    return "r32g32b32_sfloat"sv;
  case image_format::r32g32b32a32_sfloat:
    return "r32g32b32a32_sfloat"sv;
  default:
    throw unimplemented_error{};
  }
}

} // namespace graal