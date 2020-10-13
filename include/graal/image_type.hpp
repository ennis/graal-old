#pragma once
#include <graal/range.hpp>

namespace graal {

/// @brief
enum class image_type {
  image_1d,
  image_2d,
  image_3d,
  image_cube_map,
  image_1d_array,
  image_2d_array,
  image_2d_multisample,
  image_2d_multisample_array,
};

inline constexpr int num_extents(image_type ty) {
  switch (ty) {
  case image_type::image_1d:
    return 1;
  case image_type::image_2d:
    return 2;
  case image_type::image_3d:
    return 3;
  case image_type::image_cube_map:
    return 2;
  case image_type::image_1d_array:
    return 2;
  case image_type::image_2d_array:
    return 3;
  case image_type::image_2d_multisample_array:
    return 3;
  case image_type::image_2d_multisample:
    return 2;
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
  }
}

inline constexpr bool image_type_has_mipmaps(image_type ty) {
    return
        (ty != image_type::image_2d_multisample) &&
        (ty != image_type::image_2d_multisample_array);
}

inline constexpr bool image_type_is_multisample(image_type ty) {
    return
        (ty == image_type::image_2d_multisample) ||
        (ty == image_type::image_2d_multisample_array);
}

template <image_type Type> using image_size = range<num_extents(Type)>;

} // namespace graal