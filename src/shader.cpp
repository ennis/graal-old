#include <graal/errors.hpp>
#include <graal/shader.hpp>
#include <ostream>
#include <sstream>
#include <vector>

namespace graal {

shader_handle create_shader(shader_stage stage, std::string_view source,
                            std::ostream &out_log) {
  GLuint      obj = glCreateShader(shader_stage_to_glenum(stage));
  std::string source_str = std::string{source};
  const char *shader_sources[1] = {source_str.c_str()};
  glShaderSource(obj, 1, shader_sources, NULL);
  glCompileShader(obj);

  // get log
  GLint logsize = 0;
  glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &logsize);
  std::string logbuf;
  if (logsize != 0) {
    logbuf.resize(logsize);
    glGetShaderInfoLog(obj, logsize, &logsize, logbuf.data());
    out_log << logbuf.data();
  }

  // if failure, delete and throw
  GLint status = GL_TRUE;
  glGetShaderiv(obj, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    glDeleteShader(obj);
    throw shader_compilation_error{std::move(logbuf)};
  }

  return shader_handle{obj};
}

shader_handle compile_shader(shader_stage stage, std::string_view source) {
  std::ostringstream ss;
  return create_shader(stage, source, ss);
}

} // namespace graal::gl