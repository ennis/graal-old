#pragma once
//#include <vulkan/vulkan.hpp>

namespace graal {

/// @brief How the resource is going to be accessed
enum class image_usage {
  sampled_image,
  storage_image,
  color_attachment,
  depth_stencil_attachment,
  pixel_transfer_source,      // == GPU image readback
  pixel_transfer_destination, // == GPU image upload
  presentation,		// vkQueuePresent
};

} // namespace graal