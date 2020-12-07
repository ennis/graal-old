#pragma once
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/range.hpp>

namespace graal::detail {

struct image_desc {

  constexpr explicit image_desc(image_type type, image_format format,
                                range<1> size, int num_mipmaps, int num_samples)
      : image_desc{type, format, range{size[0], 1, 1}, num_mipmaps,
                   num_samples} {}

  constexpr explicit image_desc(image_type type, image_format format,
                                range<2> size, int num_mipmaps, int num_samples)
      : image_desc{type, format, range{size[0], size[1], 1}, num_mipmaps,
                   num_samples} {}
  constexpr explicit image_desc(image_type type, image_format format,
                                range<3> size, int num_mipmaps, int num_samples)
      : type{type}, format{format}, size{size}, num_mipmaps{num_mipmaps},
        num_samples{num_samples} {}

  constexpr bool operator==(const image_desc &rhs) const {
    return type == rhs.type && format == rhs.format && size == rhs.size &&
           num_mipmaps == rhs.num_mipmaps && num_samples == rhs.num_samples;
  }

  constexpr bool operator!=(const image_desc &rhs) const {
    return type != rhs.type || format != rhs.format || size != rhs.size ||
           num_mipmaps != rhs.num_mipmaps || num_samples != rhs.num_samples;
  }

  image_type   type;
  image_format format;
  range<3>     size;
  int          num_mipmaps = 1;
  int          num_samples = 1;
};

} // namespace graal::detail