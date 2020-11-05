#pragma once
#include <graal/buffer.hpp>
#include <graal/ext/vertex_traits.hpp>
#include <graal/vertex_array.hpp>

#include <fstream>
#include <glm/glm.hpp>
#include <optional>
#include <tiny_obj_loader.h>
#include <vector>
#include <span>
#include <string_view>

namespace graal {

enum class topology {
  points,
  lines,
  line_strips,
  triangles,
  triangle_strips,
};

struct vertex_2d {
  glm::vec2 position;
  glm::vec2 texcoord;
};

template <> struct vertex_traits<vertex_2d> {
  static constexpr std::array<vertex_attribute, 2> attributes{
      {{data_type::float_, 2, offsetof(vertex_2d, position)},
       {data_type::float_, 2, offsetof(vertex_2d, texcoord)}}};

  static constexpr void set_attribute(vertex_2d &vtx, semantic s, float x,
                                      float y, float z, float w) {
    if (s == semantic::position) {
      vtx.position = glm::vec2{x, y};
    } else if (s == semantic::texcoord) {
      vtx.texcoord = glm::vec2{x, y};
    }
  }
};

/// @brief Represents some geometry.
// Geometry objects are immutable. They can't change after having been created.
//
template <typename VertexType> class geometry {
public:
  struct draw_item {
    topology topology;
    size_t   start_vertex;
    size_t   num_vertices;
  };

  /// @brief Binds the vertex array and vertex buffers to the OpenGL pipeline.
  void bind() const;

  std::span<const draw_item> draw_items() const noexcept { return std::span{draw_items_}; }

  /// @brief Loads geometry from an OBJ file.
  /// @param path path to the OBJ file to load.
  /// @return
  static geometry<VertexType> load_obj(std::string_view file_name);

private:
  // uninitialized constructor, do not expose
  geometry() = default;

  topology           topo_;
  size_t             vertex_stride_;
  buffer<VertexType> vertex_buffer_; 
  std::vector<draw_item> draw_items_;
  // std::optional<buffer<uint32_t>> index_data_;
};

//=============================================================================

template <typename VertexType>
geometry<VertexType>
geometry<VertexType>::load_obj(std::string_view file_name) {
  std::ifstream file{std::string{file_name}};
  if (!file) {
    throw std::runtime_error{
        fmt::format("could not open file: `{}`", file_name)};
  }

  std::string       err;
  std::string       warn;
  tinyobj::attrib_t attrib;

  std::vector<tinyobj::shape_t>    shapes;
  std::vector<tinyobj::material_t> materials;

  tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, &file);

  // create VAO
  vertex_array_builder vao_builder;

  size_t offset = 0;
  size_t normals_offset = 0;
  size_t texcoords_offset = 0;
  int    attrib_index = 0;

  const bool has_normals = !attrib.normals.empty();
  const bool has_texcoords = !attrib.texcoords.empty();

  vao_builder.set_attribute(attrib_index++, 0, {data_type::float_, 3, offset});
  offset += 3 * sizeof(float);
  if (has_normals) {
    // got normals
    normals_offset = offset;
    vao_builder.set_attribute(attrib_index++, 0,
                              {data_type::float_, 3, offset});
    offset += 3 * sizeof(float);
  }
  if (has_texcoords) {
    // got texcoords
    texcoords_offset = offset;
    vao_builder.set_attribute(attrib_index++, 0,
                              {data_type::float_, 2, offset});
    offset += 2 * sizeof(float);
  }

  const size_t stride = offset;
  const int    num_attribs = attrib_index;
  const size_t num_vertices = attrib.vertices.size() / 3; // close enough

  std::vector<draw_item> draw_items;
  draw_items.reserve(shapes.size());
  std::vector<VertexType> vertices;
  vertices.reserve(num_vertices);

  for (int i = 0; i < shapes.size(); i++) {
    const auto &shape = shapes[i];
    const auto & mesh = shape.mesh;
    const auto start_vertex = vertices.size();

    for (int j = 0; j < mesh.indices.size(); j++) {
        auto indices = mesh.indices[j];
      // yes, we don't de-duplicate vertices here, but whatever. OBJ is a shitty
      // format anyway; eventually I'll use a better one
      VertexType v;
      vertex_traits<VertexType>::set_attribute(v, 
          semantic::position, attrib.vertices[indices.vertex_index * 3 + 0],
          attrib.vertices[indices.vertex_index * 3 + 1],
          attrib.vertices[indices.vertex_index * 3 + 2], 1.0f);
      vertex_traits<VertexType>::set_attribute(v,
          semantic::normal, attrib.normals[indices.normal_index * 3 + 0],
          attrib.normals[indices.normal_index * 3 + 1],
          attrib.normals[indices.normal_index * 3 + 2], 0.0f);
      vertex_traits<VertexType>::set_attribute(v,
          semantic::texcoord, attrib.texcoords[indices.texcoord_index * 2 + 0],
          attrib.texcoords[indices.texcoord_index * 2 + 1], 0.0f, 0.0f);
      vertices.push_back(v);
    }

    draw_items.push_back(draw_item{
        .topo = topology::triangle,
        .start_vertex = start_vertex,
        .num_vertices = mesh.indices.size(),
        });
  }

  // create buffer
  buffer vertex_buffer{std::as_bytes(std::span{vertices_interleaved})};

  geometry<VertexType> geo;
  geo.topo_ = topology::triangles;
  geo.vao_ = vao_builder.get_vertex_array();
  geo.vertex_buffer_ = std::move(vertex_buffer);
  geo.vertex_stride_ = stride;
  
  return geo;
}
} // namespace graal