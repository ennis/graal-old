#include <graal/gl/format_info.hpp>
#include <graal/gl/glad.h>
#include <graal/gl/texture.hpp>

namespace graal::gl {

namespace {
void set_default_texture_parameters(GLuint tex) {
  /*glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);*/
}
} // namespace

void texture_deleter::operator()(GLuint tex_obj) {
  if (tex_obj)
    glDeleteTextures(1, &tex_obj);
}

texture_handle create_texture_1d(GLenum target, image_format format,
                                 size_t width, size_t height,
                                 size_t num_mipmaps) {
  GLuint obj = 0;
  auto &&fmtinfo = get_format_info(format);
  // glCreateTextures(target, 1, &obj);
  // glTextureStorage2D(obj, 1, fmtinfo.internal_format, width);
  set_default_texture_parameters(obj);
  return obj;
}

texture_handle create_texture_2d(GLenum target, image_format format,
                                 size_t width, size_t height,
                                 size_t num_mipmaps) {
  GLuint obj = 0;
  auto &&fmtinfo = get_format_info(format);
  // glCreateTextures(target, 1, &obj);
  // glTextureStorage2D(obj, 1, fmtinfo.internal_format, width, height);
  set_default_texture_parameters(obj);
  return obj;
}

texture_handle create_texture_3d(GLenum target, image_format format,
                                 size_t width, size_t height, size_t depth,
                                 size_t num_mipmaps) {
  GLuint obj = 0;
  auto &&fmtinfo = get_format_info(format);
  // glCreateTextures(target, 1, &obj);
  // glTextureStorage3D(obj, 1, fmtinfo.internal_format, width, height, depth);
  set_default_texture_parameters(obj);
  return obj;
}


texture_handle create_texture_2d_multisample(GLenum target, image_format format,
    size_t width, size_t height,size_t num_samples) {
    GLuint obj = 0;
    auto&& fmtinfo = get_format_info(format);
    // glCreateTextures(target, 1, &obj);
    // glTextureStorage2DMultisample(obj, 1, fmtinfo.internal_format, width, height);
    set_default_texture_parameters(obj);
    return obj;
}

texture_handle create_texture_3d_multisample(GLenum target, image_format format,
    size_t width, size_t height, size_t depth, size_t num_samples) 
{
    GLuint obj = 0;
    auto&& fmtinfo = get_format_info(format);
    // glCreateTextures(target, 1, &obj);
    // glTextureStorage3DMultisample(obj, 1, fmtinfo.internal_format, width, height, depth);
    set_default_texture_parameters(obj);
    return obj;
}


} // namespace graal::gl