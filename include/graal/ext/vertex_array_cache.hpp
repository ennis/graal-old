#pragma once
#include <graal/vertex_array.hpp>
#include <graal/ext/vertex_traits.hpp>
#include <typeindex>
#include <unordered_map>

namespace graal {

class vertex_array_cache {
public:
  template <typename VertexType> std::shared_ptr<vertex_array_handle> get() {
    static_assert(std::is_standard_layout_v<VertexType>,
                  "VertexType must be standard-layout");
    const std::type_index vtx_type_index{typeid(VertexType)};
    if (auto it = vertex_arrays_.find(vtx_type_index);
        it != vertex_arrays_.end()) {
      return it->second;
    } else {
      vertex_array_builder vao_builder;
      vao_builder.set_attributes(0, 0, vertex_traits<VertexType>::attributes);
      auto vao =
          std::make_shared<vertex_array_handle>(vao_builder.get_vertex_array());
      vertex_arrays_.insert({vtx_type_index, std::move(vao)});
    }
  }

  static vertex_array_cache &global_cache() noexcept {
    static vertex_array_cache instance;
    return instance;
  }

private:
  std::unordered_map<std::type_index, std::shared_ptr<vertex_array_handle>>
      vertex_arrays_;
};

} // namespace graal