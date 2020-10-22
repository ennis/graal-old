#pragma once
#include <graal/glad.h>
#include <graal/image_format.hpp>
#include <stdexcept>

namespace graal {

/// @brief
struct gl_format_info {
  GLenum internal_format; ///< Corresponding internal format
  GLenum external_format; ///< Preferred external format for uploads/reads
  GLenum type;            ///< Preferred element type for uploads/reads
  int    component_count; ///< number of components (channels) (TODO redundant)
  int    size;            ///< Size of one pixel in bytes
};

/// @brief Returns information about the OpenGL internal format corresponding to
/// the specified Format
/// @param fmt
/// @return
inline constexpr gl_format_info get_gl_format_info(image_format fmt) {
  // clang-format off
  switch (fmt) {
    case image_format::r8g8b8a8_unorm: return { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 4 };
    case image_format::r8_unorm: return { GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, 1 };
    case image_format::r32_sfloat: return { GL_R32F, GL_RED, GL_FLOAT, 1, 4 };
    case image_format::r32g32_sfloat: return { GL_RG32F, GL_RG, GL_FLOAT, 2, 8 };
    case image_format::r32g32b32_sfloat: return { GL_RGB32F, GL_RGB, GL_FLOAT, 3, 12 };
    case image_format::r16g16b16a16_sfloat: return { GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 4, 8 };
    case image_format::r32g32b32a32_sfloat: return { GL_RGBA32F, GL_RGBA, GL_FLOAT, 4, 16 };
    case image_format::r16g16_sfloat: return { GL_RG16F, GL_RG, GL_FLOAT, 2, 4 };
    default: throw std::logic_error{"unimplemented"};
  }
  // clang-format on
}

} // namespace graal::gl