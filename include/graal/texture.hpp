#pragma once
#include <graal/detail/gl_handle.hpp>
#include <graal/image_format.hpp>
#include <graal/gl_types.hpp>

namespace graal {

struct texture_deleter {
  void operator()(GLuint tex_obj) const noexcept;
};

using texture_handle = detail::gl_handle<texture_deleter>;

[[nodiscard]] texture_handle create_texture_1d(GLenum       target,
                                               image_format format,
                                               size_t width,
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