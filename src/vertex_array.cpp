#include <graal/vertex_array.hpp>

namespace graal {

void vertex_array_builder::set_attributes(
    int base_attrib_index, int binding,
    std::span<const vertex_attribute> attributes) {
  const auto vaobj = vao_.get();

  int i = base_attrib_index;
  for (auto &&attrib : attributes) {
    glEnableVertexArrayAttrib(vaobj, i);
    const auto tyclass = get_data_type_class(attrib.component_type);
    switch (tyclass) {
    case data_type_class::float_:
    case data_type_class::unsigned_normalized:
    case data_type_class::signed_normalized:
      glVertexArrayAttribFormat(vaobj, i, attrib.size,
                                get_gl_component_type(attrib.component_type),
                                true, (GLuint)attrib.offset);
      break;
    case data_type_class::unsigned_scaled:
    case data_type_class::signed_scaled:
      glVertexArrayAttribFormat(vaobj, i, attrib.size,
                                get_gl_component_type(attrib.component_type),
                                false, (GLuint)attrib.offset);
      break;
    case data_type_class::unsigned_integer:
    case data_type_class::signed_integer:
      glVertexArrayAttribIFormat(vaobj, i, attrib.size,
                                 get_gl_component_type(attrib.component_type),
                                 (GLuint)attrib.offset);
      break;
    }
    glVertexArrayAttribBinding(vaobj, i, binding);
    ++i;
  }
}
} // namespace graal