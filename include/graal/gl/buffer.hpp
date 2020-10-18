#pragma once
#include <graal/gl/glad.h>
#include <graal/gl/handle.hpp>

namespace graal::gl {

struct buffer_deleter {
  void operator()(GLuint buf_obj) { glDeleteBuffers(1, &buf_obj); }
};

using buffer_handle = handle<buffer_deleter>;

buffer_handle create_buffer(size_t size, const void *data, GLbitfield flags);

} // namespace graal::gl