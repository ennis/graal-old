#pragma once
#include <graal/glad.h>
#include <fmt/core.h>

namespace graal {

namespace {
void APIENTRY debug_callback(GLenum source, GLenum type, GLuint id,
                             GLenum severity, GLsizei length,
                             const GLubyte *msg, void *data) {
  //if (severity != GL_DEBUG_SEVERITY_LOW &&
  //    severity != GL_DEBUG_SEVERITY_NOTIFICATION)
    fmt::print(stderr, "GL: {}\n", msg);
}
} // namespace

void setup_debug_output() {
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageCallback((GLDEBUGPROC)debug_callback, nullptr);
  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr,
                        true);
  glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 1111,
                       GL_DEBUG_SEVERITY_NOTIFICATION, -1,
                       "Started logging OpenGL messages");
}

} // namespace graal::gl