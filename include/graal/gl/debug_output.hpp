#pragma once
#include <graal/gl/glad.h>

namespace graal::gl {

enum class debug_source : GLenum { application, third_party };

enum class debug_type : GLenum {
  error = GL_DEBUG_TYPE_ERROR,
  deprecated_behavior = GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR,
  undefined_behavior = GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR,
  portability = GL_DEBUG_TYPE_PORTABILITY,
  performance = GL_DEBUG_TYPE_PERFORMANCE,
  marker = GL_DEBUG_TYPE_MARKER,
  push_group = GL_DEBUG_TYPE_PUSH_GROUP,
  pop_group = GL_DEBUG_TYPE_POP_GROUP,
  other = GL_DEBUG_TYPE_OTHER
};

enum class severity {
  // TODO
};

void setup_debug_output();
//void push_debug_group(std::string_view group_name);
//void pop_debug_group();

} // namespace graal::gl