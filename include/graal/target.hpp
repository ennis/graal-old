#pragma once

namespace graal {

/// @brief 
enum class target {
  vertex_buffer,
  index_buffer,
  transform_feedback_buffer,
  uniform_buffer,
  storage_buffer,
  sampled_image,
  storage_image,
  framebuffer_attachment,

  pixel_transfer_source,      // == GPU image readback
  pixel_transfer_destination, // == GPU image upload

  mapped_buffer, // buffer mapped in host memory
};

}