#pragma once
#include <graal/glad.h>
#include <graal/detail/gl_handle.hpp>

namespace graal {

struct buffer_deleter {
  void operator()(GLuint buf_obj) const noexcept { glDeleteBuffers(1, &buf_obj); }
};

using buffer_handle = detail::gl_handle<buffer_deleter>;

buffer_handle get_unbound_vk_buffer(size_t size, const void *data, GLbitfield flags);

} // namespace graal::gl