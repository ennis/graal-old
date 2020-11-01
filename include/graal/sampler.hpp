#pragma once
#include <array>
#include <boost/functional/hash.hpp>
#include <graal/glad.h>

namespace graal {

/// @brief 
enum class sampler_address_mode : GLenum {
  repeat = GL_REPEAT,
  clamp_to_edge = GL_CLAMP_TO_EDGE,
  clamp_to_border = GL_CLAMP_TO_BORDER,
  mirrored_repeat = GL_MIRRORED_REPEAT
};

/// @brief 
enum class sampler_filter : GLenum {
  nearest = GL_NEAREST,
  linear = GL_LINEAR,
  nearest_mipmap_nearest = GL_NEAREST_MIPMAP_NEAREST,
  linear_mipmap_nearest = GL_LINEAR_MIPMAP_NEAREST,
  nearest_mipmap_linear = GL_NEAREST_MIPMAP_LINEAR,
  linear_mipmap_linear = GL_LINEAR_MIPMAP_LINEAR,
};

/// @brief 
enum class compare_func : GLenum {
  less_or_equal = GL_LEQUAL,
  greater_or_equal = GL_GEQUAL,
  less = GL_LESS,
  greater = GL_GREATER,
  equal = GL_EQUAL,
  not_equal = GL_NOTEQUAL,
  always = GL_ALWAYS,
  never = GL_NEVER,
};

/// @brief 
enum class sampler_compare_mode : GLenum {
  none = GL_NONE,
  compare_ref_to_texture = GL_COMPARE_REF_TO_TEXTURE
};

/// @brief 
/// @param filter 
/// @return 
inline constexpr GLenum get_gl_sampler_filter(sampler_filter filter) noexcept {
  return static_cast<GLenum>(filter);
}

/// @brief 
struct sampler_desc {
  sampler_address_mode wrap_s =
      sampler_address_mode::repeat; ///< GL_TEXTURE_WRAP_S
  sampler_address_mode wrap_t =
      sampler_address_mode::repeat; ///< GL_TEXTURE_WRAP_T
  sampler_address_mode wrap_r =
      sampler_address_mode::repeat; ///< GL_TEXTURE_WRAP_R
  sampler_filter min_filter =
      sampler_filter::nearest; ///< GL_TEXTURE_MAG_FILTER
  sampler_filter mag_filter =
      sampler_filter::nearest;                ///< GL_TEXTURE_MIN_FILTER
  float                max_anisotropy = 1.0f; ///< GL_TEXTURE_MAX_ANISOTROPY
  float                min_lod = -1000.0f;    ///< GL_TEXTURE_MIN_LOD
  float                max_lod = 1000.0f;     ///< GL_TEXTURE_MAX_LOD
  float                lod_bias = 0.0f;       ///< GL_TEXTURE_LOD_BIAS
  sampler_compare_mode compare_mode = sampler_compare_mode::none;
  compare_func         compare_func = compare_func::always;

  std::array<float, 4> border_color;

  /// Comparison operator
  constexpr bool operator==(const sampler_desc &rhs) const noexcept {
    return wrap_s == rhs.wrap_s && wrap_t == rhs.wrap_t &&
           wrap_r == rhs.wrap_r && min_filter == rhs.min_filter &&
           mag_filter == rhs.mag_filter && border_color == rhs.border_color &&
           max_anisotropy == rhs.max_anisotropy && min_lod == rhs.min_lod &&
           max_lod == rhs.max_lod && lod_bias == rhs.lod_bias &&
           compare_mode == rhs.compare_mode && compare_func == rhs.compare_func;
  }

  struct Hash {
    std::size_t operator()(sampler_desc const &s) const noexcept {
      using boost::hash_combine;
      using boost::hash_value;
      std::size_t res = 0;
      hash_combine(res, hash_value(s.wrap_s));
      hash_combine(res, hash_value(s.wrap_t));
      hash_combine(res, hash_value(s.wrap_r));
      hash_combine(res, hash_value(s.min_filter));
      hash_combine(res, hash_value(s.mag_filter));
      hash_combine(res, hash_value(s.border_color[0]));
      hash_combine(res, hash_value(s.border_color[1]));
      hash_combine(res, hash_value(s.border_color[2]));
      hash_combine(res, hash_value(s.border_color[3]));
      hash_combine(res, hash_value(s.max_anisotropy));
      hash_combine(res, hash_value(s.min_lod));
      hash_combine(res, hash_value(s.max_lod));
      hash_combine(res, hash_value(s.lod_bias));
      hash_combine(res, hash_value(s.compare_mode));
      hash_combine(res, hash_value(s.compare_func));
      return res;
    }
  };
};
} // namespace graal