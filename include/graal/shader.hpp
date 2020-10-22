#pragma once
#include <exception>
#include <graal/detail/gl_handle.hpp>
#include <graal/glad.h>
#include <ostream>
#include <string_view>

namespace graal {
struct shader_deleter {
  void operator()(GLuint shader_obj) const noexcept {
    glDeleteShader(shader_obj);
  }
};

using shader_handle = detail::gl_handle<shader_deleter>;

/// @brief
enum class shader_stage : GLenum {
  vertex = GL_VERTEX_SHADER,
  fragment = GL_FRAGMENT_SHADER,
  tess_control = GL_TESS_CONTROL_SHADER,
  tess_evaluation = GL_TESS_EVALUATION_SHADER,
  geometry = GL_GEOMETRY_SHADER,
  compute = GL_COMPUTE_SHADER
};

inline constexpr GLenum shader_stage_to_glenum(shader_stage stage) noexcept {
  return static_cast<GLenum>(stage);
}

/// @brief Thrown when a shader fails to compile.
class shader_compilation_error : public std::exception {
public:
  shader_compilation_error(shader_stage stage, std::string log)
      : stage_{stage}, log_{std::move(log)} {}

  /// @brief Compilation log
  std::string_view log() const noexcept { return log_; }

  /// @brief The stage of the shader that was being compiled.
  shader_stage stage() const noexcept { return stage_; }

private:
  shader_stage stage_;
  std::string  log_;
};

/// @brief
/// @param stage
/// @param source
/// @return
[[nodiscard]] shader_handle compile_shader(shader_stage     stage,
                                           std::string_view source);

/// @brief
/// @param stage
/// @param source
/// @param out_log
/// @return
[[nodiscard]] shader_handle compile_shader(shader_stage     stage,
                                           std::string_view source,
                                           std::ostream &   out_log);

} // namespace graal