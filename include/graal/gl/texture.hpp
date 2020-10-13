#pragma once
#include "handle.hpp"
#include <graal/image_format.hpp>

namespace graal::gl {

struct texture_deleter {
  void operator()(GLuint tex_obj);
};

using texture_handle = handle<texture_deleter>;

[[nodiscard]] texture_handle create_texture_1d(GLenum       target,
                                               image_format format,
                                               size_t width, size_t height,
                                               size_t num_mipmaps);

[[nodiscard]] texture_handle create_texture_2d(GLenum       target,
                                               image_format format,
                                               size_t width, size_t height,
                                               size_t num_mipmaps);

[[nodiscard]] texture_handle
create_texture_3d(GLenum target, image_format format, size_t width,
                  size_t height, size_t depth, size_t num_mipmaps);

[[nodiscard]] texture_handle
create_texture_2d_multisample(GLenum target, image_format format, size_t width,
                              size_t height, size_t num_samples);

[[nodiscard]] texture_handle
create_texture_3d_multisample(GLenum target, image_format format, size_t width,
                              size_t height, size_t depth, size_t num_samples);

} // namespace graal::gl