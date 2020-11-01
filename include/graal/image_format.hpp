#pragma once
#include <cassert>
#include <graal/errors.hpp>
#include <string_view>
#include <vulkan/vulkan.hpp>

namespace graal {

/// @brief
// These formats correspond to vulkan formats
enum class image_format {
  r8_unorm = VK_FORMAT_R8_UNORM,
  r8g8_unorm = VK_FORMAT_R8G8_UNORM,
  r8g8b8_unorm = VK_FORMAT_R8G8B8_UNORM,
  r8g8b8a8_unorm = VK_FORMAT_R8G8B8A8_UNORM,
  r16_unorm = VK_FORMAT_R16_UNORM,
  r16g16_unorm = VK_FORMAT_R16G16_UNORM,
  r16g16b16_unorm = VK_FORMAT_R16G16B16_UNORM,
  r16g16b16a16_unorm = VK_FORMAT_R16G16B16A16_UNORM,

  r16_sfloat = VK_FORMAT_R16_SFLOAT,
  r16g16_sfloat = VK_FORMAT_R16G16_SFLOAT,
  r16g16b16_sfloat = VK_FORMAT_R16G16B16_SFLOAT,
  r16g16b16a16_sfloat = VK_FORMAT_R16G16B16A16_SFLOAT,

  r32_sfloat = VK_FORMAT_R32_SFLOAT,
  r32g32_sfloat = VK_FORMAT_R32G32_SFLOAT,
  r32g32b32_sfloat = VK_FORMAT_R32G32B32_SFLOAT,
  r32g32b32a32_sfloat = VK_FORMAT_R32G32B32A32_SFLOAT,
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
    // TODO
    assert(false);
  }
}

inline constexpr vk::Format get_vk_format(image_format fmt) {
  return static_cast<vk::Format>(fmt);
}

} // namespace graal