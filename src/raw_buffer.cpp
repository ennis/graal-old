#include <graal/raw_buffer.hpp>

namespace graal {

buffer_handle get_unbound_vk_buffer(size_t size, const void *data, GLbitfield flags) {
  GLuint obj = 0;
  glCreateBuffers(1, &obj);
  glNamedBufferStorage(obj, size, data, flags);
  return buffer_handle{obj};
}

} // namespace graal::gl