#pragma once
#include <graal/detail/gl_handle.hpp>
#include <graal/glad.h>
#include <iosfwd>

namespace graal {

struct program_deleter {
  void operator()(GLuint shader_obj) const noexcept {
    glDeleteProgram(shader_obj);
  }
};

using program_handle = detail::gl_handle<program_deleter>;

/// @brief Creates and links a program for the graphics pipeline using the given
/// shader stages.
/// @param vertex_shader
/// @param fragment_shader
/// @param tess_control_shader
/// @param tess_eval_shader
/// @param geometry_shader
/// @return
[[nodiscard]] program_handle
create_program(GLuint vertex_shader, GLuint fragment_shader,
               GLuint tess_control_shader, GLuint tess_eval_shader,
               GLuint geometry_shader, std::ostream &link_log);

/// @brief Creates and links a compute program from the given compute shader.
/// @param compute_shader
/// @return
[[nodiscard]] program_handle create_compute_program(GLuint compute_shader,
                                                    std::ostream &link_log);

} // namespace graal