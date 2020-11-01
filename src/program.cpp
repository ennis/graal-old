#include <cassert>
#include <graal/errors.hpp>
#include <graal/program.hpp>
#include <ostream>
#include <string>

namespace graal {

namespace {

// Tries to link the program. If it fails, deletes the program object and throws
// a program_link_error exception.
void link_program(GLuint prog, std::ostream &link_log) {
  glLinkProgram(prog);

  // retrieve log
  GLint logsize = 0;
  glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logsize);
  std::string logbuf;
  if (logsize != 0) {
    logbuf.resize(logsize);
    glGetProgramInfoLog(prog, logsize, &logsize, logbuf.data());
    link_log << logbuf.data();
  }

  // throw on link error
  GLint status = GL_TRUE;
  glGetProgramiv(prog, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    glDeleteProgram(prog);
    throw program_link_error{std::move(logbuf)};
  }
}

} // namespace

program_handle create_program(GLuint vertex_shader, GLuint fragment_shader,
                              GLuint tess_control_shader,
                              GLuint tess_eval_shader, GLuint geometry_shader,
                              std::ostream &link_log) 
{
  GLuint prog = glCreateProgram();

  assert(vertex_shader);
  assert(fragment_shader);
  glAttachShader(prog, vertex_shader);
  glAttachShader(prog, fragment_shader);
  if (tess_control_shader)
    glAttachShader(prog, tess_control_shader);
  if (tess_eval_shader)
    glAttachShader(prog, tess_eval_shader);
  if (geometry_shader)
    glAttachShader(prog, geometry_shader);

  link_program(prog, link_log);

  glDetachShader(prog, vertex_shader);
  glDetachShader(prog, fragment_shader);
  if (tess_control_shader)
    glDetachShader(prog, tess_control_shader);
  if (tess_eval_shader)
    glDetachShader(prog, tess_eval_shader);
  if (geometry_shader)
    glDetachShader(prog, geometry_shader);

  return program_handle{prog};
}

program_handle create_compute_program(GLuint        compute_shader,
                                      std::ostream &link_log) {
  GLuint prog = glCreateProgram();
  assert(compute_shader);
  glAttachShader(prog, compute_shader);
  link_program(prog, link_log);
  glDetachShader(prog, compute_shader);
  return program_handle{prog};
}

} // namespace graal