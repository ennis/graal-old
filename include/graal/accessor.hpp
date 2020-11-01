#pragma once
#include <cassert>
#include <graal/buffer.hpp>
#include <graal/image.hpp>
#include <graal/image_type.hpp>
#include <graal/queue.hpp>
#include <type_traits>

namespace graal {

template <access_mode> struct mode_tag_t { explicit mode_tag_t() = default; };

inline constexpr mode_tag_t<access_mode::read_only>  read_only{};
inline constexpr mode_tag_t<access_mode::read_write> read_write{};
inline constexpr mode_tag_t<access_mode::write_only> write_only{};

struct color_attachment_tag_t {
  explicit color_attachment_tag_t() = default;
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

inline constexpr color_attachment_tag_t framebuffer_attachment{};
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

// image_accessor 
template < 
    image_type Type, 
    image_usage Usage,
    access_mode AccessMode, 
    bool HostVisible>
class image_accessor_base
{
public:
    image_accessor_base(image<Type, false> &img, handler &h) {
    h.add_image_access(img.impl_, Usage, AccessMode);
  }

  /// @brief
  /// @return
  image_size<Type> size() const {
    // TODO
    assert(false);
  }

  /// @brief
  /// @return
  image_format format() const {
    // TODO
    assert(false);
  }

private:
  std::shared_ptr<detail::image_impl> img_;
};

template <typename DataT, buffer_usage Usage, access_mode AccessMode>
class buffer_accessor_base
{
public:
    buffer_accessor_base(buffer<DataT, false>& buf, handler& h) {
        h.add_buffer_access(buf.impl_, Usage, AccessMode);
    }

private:
  std::shared_ptr<detail::buffer_impl> buffer_;
};
} // namespace detail

//-----------------------------------------------------------------------------
/// @brief Primary image accessor template
/// @tparam DataT
template <image_type  Type,       // image dimensions (1D for buffers)
          image_usage      Usage,     // target
          access_mode AccessMode, // access mode (RW/RO/WO)
          bool HostVisible     // true if can be mapped
>
class image_accessor {};

// now buffers and images are sufficiently different to warrant two different
// accessor types?

//-----------------------------------------------------------------------------
// image_accessor(sampled_image)
template <image_type Type, access_mode AccessMode, bool HostVisible>
class image_accessor<Type, image_usage::sampled_image, AccessMode, HostVisible>
    : public detail::image_accessor_base<Type, image_usage::sampled_image,
                                         AccessMode, HostVisible> {
public:
  using base_t = detail::image_accessor_base<Type, image_usage::sampled_image,
                                             AccessMode, HostVisible>;

  image_accessor(image<Type, HostVisible> &image, sampled_image_tag_t,
                 handler &                 h)
      : base_t{image, h} {
    image.impl_->add_usage(vk::ImageUsageFlagBits::eSampled);
  }
};

//-----------------------------------------------------------------------------
// accessor<storage_image>
template <image_type Type, access_mode AccessMode, bool HostVisible>
class image_accessor<Type, image_usage::storage_image, AccessMode, HostVisible>
    : public detail::image_accessor_base<Type, image_usage::storage_image,
    AccessMode, HostVisible> {
public:
    using base_t = detail::image_accessor_base<Type, image_usage::storage_image,
        AccessMode, HostVisible>;

    image_accessor(image<Type, HostVisible>& image, storage_image_tag_t, handler& h)
        : base_t{ image, h } 
    {
        image.impl_->add_usage(vk::ImageUsageFlagBits::eStorage);
    }

    image_accessor(image<Type, HostVisible>& image, storage_image_tag_t,
        mode_tag_t<AccessMode>, handler& h)
        : base_t{ image, h } 
    {
        image.impl_->add_usage(vk::ImageUsageFlagBits::eStorage);
    }
};

//-----------------------------------------------------------------------------
// accessor<color_attachment>
template <image_type Type, access_mode AccessMode, bool HostVisible>
class image_accessor<Type, image_usage::color_attachment, AccessMode,
    HostVisible>
    : public detail::image_accessor_base<Type, image_usage::color_attachment, AccessMode,
    HostVisible>
{
public:
    using base_t = detail::image_accessor_base<Type, image_usage::color_attachment, AccessMode,
        HostVisible>;

    image_accessor(image<Type, HostVisible>& img, color_attachment_tag_t,
        handler& h)
        : base_t{ img, h } 
    {
    }

    // framebuffer_attachment, discard (implies write-only)
    image_accessor(image<Type, HostVisible>& img, color_attachment_tag_t,
        framebuffer_load_op_discard_t, handler& h)
        : base_t{ img, h } {}

    // framebuffer_attachment, keep (implies read-write)
    image_accessor(image<Type, HostVisible>& img, color_attachment_tag_t,
        framebuffer_load_op_keep_t, handler& h)
        : base_t{ img, h } {}

    // framebuffer_attachment, clear (implies write-only)
    image_accessor(image<Type, HostVisible>& img, color_attachment_tag_t,
        framebuffer_load_op_clear_t clear_color, handler& h)
        : base_t{ img, h } {}
};

//-----------------------------------------------------------------------------
// accessor<pixel_transfer_destination>
template <image_type Type, bool HostVisible>
class accessor<void, Type, target::pixel_transfer_destination,
    access_mode::write_only, HostVisible>
    : public detail::image_accessor<void, Type,
    target::pixel_transfer_destination,
    access_mode::write_only, HostVisible> {
public:
    accessor(const image<Type, HostVisible> &img, pixel_transfer_destination_t,
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
template <image_type D, bool Ext> accessor(image<D, Ext>&, color_attachment_tag_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::read_write, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, color_attachment_tag_t, framebuffer_load_op_keep_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::read_write, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, color_attachment_tag_t, framebuffer_load_op_discard_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::write_only, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, color_attachment_tag_t, framebuffer_load_op_clear_t, handler&)
    -> accessor<void, D, target::framebuffer_attachment, access_mode::write_only, Ext>;

// --- pixel transfer ---
template <image_type D, bool Ext> accessor(image<D, Ext>&, pixel_transfer_destination_t, handler&)
    -> accessor<void, D, target::pixel_transfer_destination, access_mode::write_only, Ext>;

template <image_type D, bool Ext> accessor(image<D, Ext>&, pixel_transfer_source_t, handler&)
    -> accessor<void, D, target::pixel_transfer_source, access_mode::read_only, Ext>;

// clang-format on

} // namespace graal