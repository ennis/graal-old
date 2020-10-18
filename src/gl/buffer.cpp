#include <graal/gl/buffer.hpp>

namespace graal::gl {

buffer_handle create_buffer(size_t size, const void *data, GLbitfield flags) {
  GLuint obj = 0;
  glCreateBuffers(1, &obj);
  glNamedBufferStorage(obj, size, data, flags);
  return buffer_handle{obj};
}

} // namespace graal::gl