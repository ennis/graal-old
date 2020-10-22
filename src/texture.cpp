#include <fmt/core.h>
#include <graal/gl_format_info.hpp>
#include <graal/glad.h>
#include <graal/texture.hpp>

namespace graal {

namespace {
void set_default_texture_parameters(GLuint tex) {
  /*glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);*/
}
} // namespace

void texture_deleter::operator()(GLuint tex_obj) const noexcept {
  if (tex_obj) {
    glDeleteTextures(1, &tex_obj);
#ifdef GRAAL_TRACE_RESOURCES
    fmt::print("delete_texture {}\n", tex_obj);
#endif
  }
}

texture_handle create_texture_1d(GLenum target, image_format format,
                                 size_t width, size_t num_mipmaps) {
  GLuint obj = 0;
  auto &&fmtinfo = get_gl_format_info(format);
  glCreateTextures(target, 1, &obj);
  glTextureStorage1D(obj, 1, fmtinfo.internal_format, width);
  set_default_texture_parameters(obj);
#ifdef GRAAL_TRACE_RESOURCES
  fmt::print(
      "create_texture_1d width={} target={} format={} num_mipmaps={} -> {}\n",
      width, target, get_image_format_name(format), num_mipmaps, obj);
#endif
  return obj;
}

texture_handle create_texture_2d(GLenum target, image_format format,
                                 size_t width, size_t height,
                                 size_t num_mipmaps) {
  GLuint obj = 0;
  auto &&fmtinfo = get_gl_format_info(format);
  glCreateTextures(target, 1, &obj);
  glTextureStorage2D(obj, 1, fmtinfo.internal_format, width, height);
  set_default_texture_parameters(obj);
#ifdef GRAAL_TRACE_RESOURCES
  fmt::print("create_texture_2d width={} height={} target={} format={} "
             "num_mipmaps={} -> {}\n",
             width, height, target, get_image_format_name(format), num_mipmaps,
             obj);
#endif
  return obj;
}

texture_handle create_texture_3d(GLenum target, image_format format,
                                 size_t width, size_t height, size_t depth,
                                 size_t num_mipmaps) {

  GLuint obj = 0;
  auto &&fmtinfo = get_gl_format_info(format);
  glCreateTextures(target, 1, &obj);
  glTextureStorage3D(obj, 1, fmtinfo.internal_format, width, height, depth);
  set_default_texture_parameters(obj);
#ifdef GRAAL_TRACE_RESOURCES
  fmt::print("create_texture_3d width={} height={} depth={} target={} "
             "format={} num_mipmaps={} -> {}\n",
             width, height, depth, target, get_image_format_name(format),
             num_mipmaps, obj);
#endif
  return obj;
}

texture_handle create_texture_2d_multisample(GLenum target, image_format format,
                                             size_t width, size_t height,
                                             size_t num_samples) {
  GLuint obj = 0;
  auto &&fmtinfo = get_gl_format_info(format);
  glCreateTextures(target, 1, &obj);
  glTextureStorage2DMultisample(obj, num_samples, fmtinfo.internal_format,
                                width, height, true);
  set_default_texture_parameters(obj);
#ifdef GRAAL_TRACE_RESOURCES
  fmt::print("create_texture_2d_multisample width={} height={} target={} "
             "format={} num_samples={} -> {}\n",
             width, height, target, get_image_format_name(format), num_samples,
             obj);
#endif
  return obj;
}

texture_handle create_texture_3d_multisample(GLenum target, image_format format,
                                             size_t width, size_t height,
                                             size_t depth, size_t num_samples) {
  GLuint obj = 0;
  auto &&fmtinfo = get_gl_format_info(format);
  glCreateTextures(target, 1, &obj);
  glTextureStorage3DMultisample(obj, num_samples, fmtinfo.internal_format,
                                width, height, depth, true);
  set_default_texture_parameters(obj);
#ifdef GRAAL_TRACE_RESOURCES
  fmt::print("create_texture_3d_multisample width={} height={} depth={} "
             "target={} format={} num_samples={} -> {}\n",
             width, height, depth, target, get_image_format_name(format),
             num_samples, obj);
#endif
  return obj;
}

} // namespace graal::gl