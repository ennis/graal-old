#pragma once
#include <graal/buffer.hpp>
#include <graal/image.hpp>
#include <graal/image_type.hpp>
#include <graal/queue.hpp>
#include <type_traits>

namespace graal {

template <access_mode> struct mode_tag_t { explicit mode_tag_t() = default; };
template <target> struct target_tag_t { explicit target_tag_t() = default; };

template <target, access_mode> struct mode_target_tag_t {
  explicit mode_target_tag_t() = default;
};

inline constexpr mode_tag_t<access_mode::read_only>  read_only{};
inline constexpr mode_tag_t<access_mode::read_write> read_write{};
inline constexpr mode_tag_t<access_mode::write_only> write_only{};

struct framebuffer_attachment_tag_t {
  explicit framebuffer_attachment_tag_t() = default;
};
struct framebuffer_load_op_clear_t {
  explicit constexpr framebuffer_load_op_clear_t(float x)
      : clear_values{x, x, x, x} {}

  explicit constexpr framebuffer_load_op_clear_t(float r, float g, float b,
                                                 float a)
      : clear_values{r, g, b, a} {}

  std::array<float, 4> clear_values;
};
struct framebuffer_load_op_keep_t {
  explicit framebuffer_load_op_keep_t() = default;
};
struct framebuffer_load_op_discard_t {
  explicit framebuffer_load_op_discard_t() = default;
};

inline constexpr framebuffer_attachment_tag_t framebuffer_attachment{};
using clear_init = framebuffer_load_op_clear_t;
inline constexpr framebuffer_load_op_keep_t    keep{};
inline constexpr framebuffer_load_op_discard_t discard{};

struct pixel_transfer_destination_t {
  explicit pixel_transfer_destination_t() = default;
};
struct pixel_transfer_source_t {
  explicit pixel_transfer_source_t() = default;
};

inline constexpr pixel_transfer_destination_t pixel_transfer_destination;
inline constexpr pixel_transfer_source_t      pixel_transfer_source;

struct sampled_image_tag_t {
  explicit sampled_image_tag_t() = default;
};
inline constexpr sampled_image_tag_t sampled_image{};

struct storage_image_tag_t {
  explicit storage_image_tag_t() = default;
};
inline constexpr storage_image_tag_t storage_image{};

namespace detail {
//-----------------------------------------------------
inline constexpr bool is_image_target(target target) {
  return target == target::framebuffer_attachment ||
         target == target::pixel_transfer_source ||
         target == target::pixel_transfer_destination ||
         target == target::storage_image || target == target::sampled_image;
}

inline constexpr bool is_buffer_target(target target) {
  return target == target::mapped_buffer || target == target::index_buffer ||
         target == target::vertex_buffer || target == target::uniform_buffer ||
         target == target::storage_buffer ||
         target == target::transform_feedback_buffer;
}

//-----------------------------------------------------

template <typename DataT, image_type Type, target Target,
          access_mode AccessMode, bool ExternalAccess>
class image_accessor;

// image_accessor for virtual images
template <typename DataT, image_type Type, target Target,
          access_mode AccessMode>
class image_accessor<DataT, Type, Target, AccessMode, false> {
  static_assert(is_image_target(Target), "invalid target for image_accessor");

public:
  image_accessor(image<Type, false> &img, handler &h)
      : virt_img_{img.get_virtual_image()} {
    h.add_resource_access(img.get_resource_tracker(), AccessMode, virt_img_);
  }

  /// @brief
  /// @return
  GLuint get_gl_object() const { throw unimplemented_error{}; }

  /// @brief
  /// @return
  image_size<Type> size() const { throw unimplemented_error{}; }

  /// @brief
  /// @return
  image_format format() const { throw unimplemented_error{}; }

private:
  std::shared_ptr<detail::virtual_image_resource> virt_img_;
};

// image_accessor for concrete images
template <typename DataT, image_type Type, target Target,
          access_mode AccessMode>
class image_accessor<DataT, Type, Target, AccessMode, true> {
  static_assert(is_image_target(Target), "invalid target for image_accessor");

public:
  image_accessor(image<Type, true> &img, handler &h) : image_{img} {
    h.add_resource_access(img.get_resource_tracker(), AccessMode);
  }

  /// @brief
  /// @return
  GLuint get_gl_object() const { return image_.get_gl_object(); }

  /// @brief
  /// @return
  image_size<Type> size() const { return image_.size(); }

  /// @brief
  /// @return
  image_format format() const { return image_.format(); }

private:
  image<Type, true> image_;
};

template <typename DataT, target Target, access_mode AccessMode>
class buffer_accessor {
  static_assert(is_buffer_target(Target), "invalid target for buffer_accessor");

public:
  GLuint get_gl_object() const { return buffer_.get_gl_object(); }

private:
  buffer<DataT> buffer_;
};
} // namespace detail

//-----------------------------------------------------------------------------
/// @brief Primary accessor template
/// @tparam DataT
template <typename DataT,
          image_type  Type,       // image dimensions (1D for buffers)
          target      Target,     // target
          access_mode AccessMode, // access mode (RW/RO/WO)
          bool ExternalAccess     // false if no external access (virtual), true
                                  // otherwise
          >
class accessor {};

// now buffers and images are sufficiently different to warrant two different
// accessor types?

//-----------------------------------------------------------------------------
// accessor(sampled_image)
template <typename DataT, image_type Type, access_mode AccessMode,
          bool ExternalAccess>
class accessor<DataT, Type, target::sampled_image, AccessMode, ExternalAccess>
    : public detail::image_accessor<DataT, Type, target::sampled_image,
                                    AccessMode, ExternalAccess> {
public:
  using base_t = detail::image_accessor<void, Type, target::sampled_image,
                                        AccessMode, ExternalAccess>;

  accessor(image<Type, ExternalAccess> &image, sampled_image_tag_t, handler &h)
      : base_t{image, h} {}
};

//-----------------------------------------------------------------------------
// accessor<storage_image>
template <image_type Type, access_mode AccessMode, bool ExternalAccess>
class accessor<void, Type, target::storage_image, AccessMode, ExternalAccess>
    : public detail::image_accessor<void, Type, target::storage_image,
                                    AccessMode, ExternalAccess> {
public:
  using base_t =
      detail::image_accessor<void, Type, target::framebuffer_attachment,
                             AccessMode, ExternalAccess>;

  accessor(image<Type, ExternalAccess> &image, storage_image_tag_t, handler &h)
      : base_t{image, h} {}

  accessor(image<Type, ExternalAccess> &image, storage_image_tag_t,
           mode_tag_t<AccessMode>, handler &h)
      : base_t{image, h} {}
};

//-----------------------------------------------------------------------------
// accessor<framebuffer_attachment>
template <image_type Type, access_mode AccessMode, bool ExternalAccess>
class accessor<void, Type, target::framebuffer_attachment, AccessMode,
               ExternalAccess>
    : public detail::image_accessor<void, Type, target::framebuffer_attachment,
                                    AccessMode, ExternalAccess>

{
public:
  using base_t =
      detail::image_accessor<void, Type, target::framebuffer_attachment,
                             AccessMode, ExternalAccess>;

  accessor(image<Type, ExternalAccess> &img, framebuffer_attachment_tag_t,
           handler &                    h)
      : base_t{img, h} {}

  // framebuffer_attachment, discard (implies write-only)
  accessor(image<Type, ExternalAccess> &img, framebuffer_attachment_tag_t,
           framebuffer_load_op_discard_t, handler &h)
      : base_t{img, h} {}

  // framebuffer_attachment, keep (implies read-write)
  accessor(image<Type, ExternalAccess> &img, framebuffer_attachment_tag_t,
           framebuffer_load_op_keep_t, handler &h)
      : base_t{img, h} {}

  // framebuffer_attachment, clear (implies write-only)
  accessor(image<Type, ExternalAccess> &img, framebuffer_attachment_tag_t,
           framebuffer_load_op_clear_t clear_color, handler &h)
      : base_t{img, h} {}
};

//-----------------------------------------------------------------------------
// accessor<pixel_transfer_destination>
template <image_type Type, bool ExternalAccess>
class accessor<void, Type, target::pixel_transfer_destination,
               access_mode::write_only, ExternalAccess>
    : public detail::image_accessor<void, Type,
                                    target::pixel_transfer_destination,
                                    access_mode::write_only, ExternalAccess> {
public:
  accessor(const image<Type, ExternalAccess> &img, pixel_transfer_destination_t,
           handler &                          h)
      : detail::image_accessor{img, h} {}

  /*// TODO maybe find a safer way to specify the format of the data?
  void upload_image_data(const range<Dimensions> &offset,
                         const image_size<Dimensions> &size, GLenum
  externalFormat, GLenum type, const void *data) {
    // get the texture object,
    // auto tex = get_gl_object();
  }*/
};

// --- deduction guides -------------------------------------------------------

// clang-format off

// --- sampled image ---
template <image_type D, bool Ext> accessor(const image<D, Ext>&, sampled_image_tag_t, handler&)
    -> accessor<void, D, target::sampled_image, access_mode::read_only, Ext>;


// --- storage image ---
template <image_type D, bool Ext, access_mode AccessMode> accessor(image<D, Ext>&, storage_image_tag_t, mode_tag_t<AccessMode>, handler&)
    -> accessor<void, D, target::storage_image, AccessMode, Ext>;

// --- framebuffer attachment ---
template <image_type D, bool Ext> accessor(image<D, Ext>&, framebuffer_attachment_tag_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::read_write, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, framebuffer_attachment_tag_t, framebuffer_load_op_keep_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::read_write, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, framebuffer_attachment_tag_t, framebuffer_load_op_discard_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::write_only, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, framebuffer_attachment_tag_t, framebuffer_load_op_clear_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::write_only, Ext>;

// --- pixel transfer ---
template <image_type D, bool Ext> accessor(image<D, Ext>&, pixel_transfer_destination_t, handler&)
    -> accessor<void, D, target::pixel_transfer_destination, access_mode::write_only, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, pixel_transfer_source_t, handler&)
    -> accessor<void, D, target::pixel_transfer_source, access_mode::read_only, Ext>;

// clang-format on

} // namespace graal