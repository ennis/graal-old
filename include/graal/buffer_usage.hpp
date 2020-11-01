#pragma once
//#include <vulkan/vulkan.hpp>

namespace graal {

enum class buffer_usage {
  vertex_buffer,
  index_buffer,
  transform_feedback_buffer,
  uniform_buffer,
  storage_buffer,
  mapped_buffer, // buffer mapped in host memory
};

} // namespace graal