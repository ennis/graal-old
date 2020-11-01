#pragma once
#include <graal/range.hpp>
#include <vulkan/vulkan.hpp>

namespace graal {

/// @brief
enum class image_type {
  image_1d = VK_IMAGE_TYPE_1D,
  image_2d = VK_IMAGE_TYPE_2D,
  image_3d = VK_IMAGE_TYPE_3D,
  // image_cube_map,
  // image_1d_array,
  // image_2d_array,
  // image_2d_multisample,
  // image_2d_multisample_array,
};

inline constexpr int num_extents(image_type ty) noexcept {
  switch (ty) {
  case image_type::image_1d:
    return 1;
  case image_type::image_2d:
    return 2;
  case image_type::image_3d:
    return 3;
    /*case image_type::image_cube_map:
      return 2;
    case image_type::image_1d_array:
      return 2;
    case image_type::image_2d_array:
      return 3;
    case image_type::image_2d_multisample_array:
      return 3;
    case image_type::image_2d_multisample:
      return 2;*/
  }
}

inline constexpr image_type extents_to_image_type(int ext) {
  switch (ext) {
  case 1:
    return image_type::image_1d;
  case 2:
    return image_type::image_2d;
  case 3:
    return image_type::image_3d;
  default:
    throw std::runtime_error{"invalid dimensionality"};
  }
}

inline constexpr vk::ImageType get_vk_image_type(image_type ty) noexcept {
  return static_cast<vk::ImageType>(ty);
}

/*inline constexpr bool image_type_has_mipmaps(image_type ty) {
    return
        (ty != image_type::image_2d_multisample) &&
        (ty != image_type::image_2d_multisample_array);
}

inline constexpr bool image_type_is_multisample(image_type ty) {
    return
        (ty == image_type::image_2d_multisample) ||
        (ty == image_type::image_2d_multisample_array);
}*/

template <image_type Type> using image_size = range<num_extents(Type)>;

} // namespace graal