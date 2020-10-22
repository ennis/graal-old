#pragma once
#include <graal/data_type.hpp>
#include <graal/detail/gl_handle.hpp>
#include <graal/glad.h>
#include <span>
#include <utility>

namespace graal {

struct vertex_array_deleter {
  void operator()(GLuint vaobj) const noexcept {
    glDeleteVertexArrays(1, &vaobj);
  }
};

using vertex_array_handle = detail::gl_handle<vertex_array_deleter>;

/// @brief
struct vertex_attribute {
  data_type component_type;
  int       size;
  size_t    offset;
};

class vertex_array_builder {
public:
  /// @brief 
  /// @param base_attrib_index 
  /// @param binding 
  /// @param attributes 
  void set_attributes(int base_attrib_index, int binding,
                      std::span<const vertex_attribute> attributes);

  /// @brief 
  /// @return 
  [[nodiscard]] vertex_array_handle get_vertex_array() { return std::move(vao_); }

private:
  vertex_array_handle vao_;
};

} // namespace graal