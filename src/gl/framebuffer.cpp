#include <graal/gpu/framebuffer.hpp>
#include <graal/gpu/glad.h>
#include <graal/gpu/image.hpp>

namespace graal::gpu {

void RenderbufferDeleter::operator()(GLuint rb) {
  glDeleteRenderbuffers(1, &rb);
}

void FramebufferDeleter::operator()(GLuint fbo) {
  glDeleteFramebuffers(1, &fbo);
}

/*
Framebuffer::Framebuffer(util::span<ImageHandle> colors, GpuImage *depth) {
int    n_color_att = (int)colors.len;
GLuint fbo;
glCreateFramebuffers(1, &fbo);
static const GLenum drawBuffers[8] = {
  GL_COLOR_ATTACHMENT0,     GL_COLOR_ATTACHMENT0 + 1,
  GL_COLOR_ATTACHMENT0 + 2, GL_COLOR_ATTACHMENT0 + 3,
  GL_COLOR_ATTACHMENT0 + 4, GL_COLOR_ATTACHMENT0 + 5,
  GL_COLOR_ATTACHMENT0 + 6, GL_COLOR_ATTACHMENT0 + 7};
glNamedFramebufferDrawBuffers(fbo, n_color_att, drawBuffers);

if (depth != nullptr) {
UT_UNIMPLEMENTED;
}

// attach all render targets
for (int i = 0; i < n_color_att; ++i) {
int att_w = colors[i]->w;
int att_h = colors[i]->h;
if (w == 0) {
  w = att_w;
  h = att_h;
} else {
  assert(w == att_w && h == att_h);
}
auto tex = colors[i]->tex_obj.get();
glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0 + i, tex, 0);
}

auto status = glCheckNamedFramebufferStatus(fbo, GL_DRAW_FRAMEBUFFER);
if (status != GL_FRAMEBUFFER_COMPLETE) {
glDeleteFramebuffers(1, &fbo);
GLenum error = glGetError();
UT_ERROR("glCheckNamedFramebufferStatus returned: {}", error);
}

obj = fbo;
}*/

} // namespace gpu