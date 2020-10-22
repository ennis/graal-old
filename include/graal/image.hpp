#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>

#include <graal/detail/image_resource.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/task.hpp>
#include <graal/detail/virtual_resource.hpp>
#include <graal/errors.hpp>
#include <graal/texture.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/queue.hpp>
#include <graal/range.hpp>
#include <graal/visibility.hpp>

namespace graal {

template <image_type Type, bool ExternalAccess> class image;

template <typename... Props> class image_properties {
public:
  image_properties(Props... props) : props_{props...} {}

  template <image_type Type, bool ExternalAccess>
  void apply(image<Type, ExternalAccess> &img) const;

private:
  std::tuple<Props...> props_;
};

struct virtual_tag_t {
  explicit virtual_tag_t() = default;
};
inline constexpr virtual_tag_t virtual_image;

namespace detail {

// is there a use for partially-specified width and height?
// e.g. for array texture: specify width, height, but not depth (array length)

template <image_type Type> class image_impl_base : public named_object {
public:
  static constexpr int  Extents = num_extents(Type);
  static constexpr bool HasMipmaps = image_type_has_mipmaps(Type);
  static constexpr bool HasMultisample = image_type_is_multisample(Type);

  using image_size_t = image_size<Type>;

  image_impl_base() {}
  image_impl_base(image_format format) : format_{format} {}
  image_impl_base(image_size_t size) : size_{size} { set_size_specified(); }

  image_impl_base(image_format format, image_size_t size)
      : format_{format}, size_{size} {
    set_size_specified();
  }

  //-------------------------------------------------------
  void set_width(size_t width) { set_size<0>(width); }

  template <int D = Extents>
  std::enable_if_t<D >= 2> set_height(size_t height) {
    set_size<1>(height);
  }

  template <int D = Extents> std::enable_if_t<D >= 3> set_depth(size_t depth) {
    set_size<2>(depth);
  }

  void set_format(image_format format) {
    if (format_.has_value() && format_.value() != format) {
      throw std::logic_error{"format already specified"};
    }
    format_ = format;
  }

  template <bool HM = HasMipmaps>
  std::enable_if_t<HM> set_mipmaps(int num_mipmaps) {
    if (mipmaps_.has_value() && mipmaps_.value() != num_mipmaps) {
      throw std::logic_error{"mipmaps already specified"};
    }
    mipmaps_ = num_mipmaps;
  }

  template <bool HM = HasMultisample>
  std::enable_if_t<HM> set_samples(int num_samples) {
    if (samples_.has_value() && samples_.value() != num_samples) {
      throw std::logic_error{"samples already specified"};
    }
    samples_ = num_samples;
  }

  //-------------------------------------------------------
  bool has_width() const noexcept { return has_size<0>(); }
  template <int D = Extents>
  std::enable_if_t<D >= 2, bool> has_height() const noexcept {
    return has_size<1>();
  }
  template <int D = Extents>
  std::enable_if_t<D >= 3, bool> has_depth() const noexcept {
    return has_size<2>();
  }
  bool has_format() const noexcept { return format_.has_value(); }

  template <bool HM = HasMipmaps>
  std::enable_if_t<HM, bool> has_mipmaps() const noexcept {
    return mipmaps_.has_value();
  }

  template <bool HM = HasMultisample>
  std::enable_if_t<HM, bool> has_samples() const {
    return samples_.has_value();
  }

  //-------------------------------------------------------
  int width() const { return get_size<0>(); }
  template <int D = Extents> std::enable_if_t<D >= 2, int> height() const {
    return get_size<1>();
  }
  template <int D = Extents> std::enable_if_t<D >= 3, int> depth() const {
    return get_size<2>();
  }

  image_size_t size() const {
    if (!std::all_of(has_size_.begin(), has_size_.end(),
                     [](bool x) { return x; })) {
      // not all dimensions were specified
      throw std::logic_error("size was unspecified");
    }
    return size_;
  }

  image_format format() const {
    if (!has_format()) {
      throw std::logic_error("format was unspecified");
    }
    return format_.value();
  }

  template <bool HM = HasMipmaps> std::enable_if_t<HM, int> mipmaps() const {
    return mipmaps_.value();
  }

  template <bool HM = HasMultisample>
  std::enable_if_t<HM, int> samples() const {
    return samples_.value();
  }

  bool is_fully_specified() const noexcept {
    bool spec = has_format();
    if constexpr (Extents == 1) {
      spec = spec && has_width();
    } else if constexpr (Extents == 2) {
      spec = spec && has_width() && has_height();
    } else if constexpr (Extents == 3) {
      spec = spec && has_width() && has_height() && has_depth();
    }
    return spec;
  }

  detail::resource_tracker &get_resource_tracker() noexcept { return tracker_; }
  const detail::resource_tracker &get_resource_tracker() const noexcept {
    return tracker_;
  }

protected:
  template <int Dim> size_t has_size() const noexcept { return has_size_[Dim]; }

  template <int Dim> size_t get_size() {
    if (!has_size_[Dim]) {
      throw std::logic_error("dimension was unspecified");
    }
    return size_[Dim];
  }

  template <int Dim> void specify_size(std::size_t s) {
    if (has_size_[Dim] && size_[Dim] != s) {
      throw std::logic_error("dimension was already specified");
    }
    has_size_[Dim] = true;
    size_[Dim] = s;
  }

  void set_size_specified() noexcept {
    std::fill(has_size_.begin(), has_size_.end(), true);
  }

  std::string                 name_;
  image_size_t                size_;
  std::array<bool, Extents>   has_size_{}; // value-initialized to false
  std::optional<image_format> format_;
  std::optional<int>          samples_; // multisample count
  std::optional<int>          mipmaps_; // mipmap count
  detail::resource_tracker    tracker_;
};

template <image_type Type, bool ExternalAccess> class image_impl;

template <image_type Type>
class image_impl<Type, true> : public image_impl_base<Type> {
public:
  using image_impl_base<Type>::image_impl_base;

  GLuint get_gl_object() {
    if (image_) {
      // resource already created
      return image_->get_gl_object();
    }

    // resource not created, create now
    auto fmt = image_impl_base<Type>::format();
    auto s = image_impl_base<Type>::size();

    int mipmaps = 1;
    if constexpr (image_type_has_mipmaps(Type)) {
      mipmaps = this->mipmaps_.value_or(1);
    }

    int samples = 1;
    if constexpr (image_type_is_multisample(Type)) {
      samples = this->samples_.value_or(1);
    }

    image_ = std::make_optional<image_resource>(
        image_desc{Type, fmt, s, mipmaps, samples});
    return image_->get_gl_object();
  }

private:
  std::optional<image_resource> image_;
};

template <image_type Type>
class image_impl<Type, false> : public image_impl_base<Type> {
public:
  using image_impl_base<Type>::image_impl_base;

  ~image_impl() {
    // last external ref to this image going away
    discard();
  }

  /// @brief
  /// @param queue
  /// @param task
  /// @param target
  /// @param mode
  std::shared_ptr<virtual_image_resource> get_virtual_image() {
    if (!virt_image_) {
      // create a virtual image now
      auto fmt = image_impl_base<Type>::format();
      auto s = image_impl_base<Type>::size();

      int mipmaps = 1;
      if constexpr (image_type_has_mipmaps(Type)) {
        mipmaps = this->mipmaps_.value_or(1);
      }

      int samples = 1;
      if constexpr (image_type_is_multisample(Type)) {
        samples = this->samples_.value_or(1);
      }

      virt_image_ = std::make_shared<virtual_image_resource>(
          image_desc{Type, fmt, s, mipmaps, samples});
      // propagate the name to the virtual resource
      virt_image_->set_name(std::string{named_object::name()});
    }
    return virt_image_;
  }

  void discard() {
    if (virt_image_) {
      virt_image_->discard();
      // virt_image_ = nullptr;
    }
  }

private:
  std::shared_ptr<detail::virtual_image_resource> virt_image_;
};

} // namespace detail

// the delayed image specification is purely an ergonomic decision
// this is because we expect to support functions of the form:
//
//		void filter(const image& input, image& output);
//
// where the filter (the callee) decides the size of the image,
// but the *caller* decides other properties of the image, such as its required
// access (evaluation only, or externally visible)

/// @brief Represents an image.
/// @tparam Dimensions
/// @tparam ExternalAccess
template <image_type Type, bool ExternalAccess = true> class image final {
public:
  using impl_t = detail::image_impl<Type, ExternalAccess>;
  using image_size_t = image_size<Type>;

  static constexpr int  Extents = num_extents(Type);
  static constexpr bool HasMipmaps = image_type_has_mipmaps(Type);
  static constexpr bool HasMultisample = image_type_is_multisample(Type);

  template <typename... Props>
  image(const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>()} {
    props.apply(*this);
  }

  template <typename... Props>
  image(image_format format, const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>(format)} {
    props.apply(*this);
  }

  template <typename... Props>
  image(image_size_t size, const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>(size)} {
    props.apply(*this);
  }

  template <typename... Props>
  image(image_format format, image_size_t size,
        const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>(format, size)} {
    props.apply(*this);
  }

  template <typename... Props>
  image(virtual_tag_t, const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>()} {
    props.apply(*this);
  }

  template <typename... Props>
  image(virtual_tag_t, image_format format,
        const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>(format)} {
    props.apply(*this);
  }

  template <typename... Props>
  image(virtual_tag_t, image_size_t size,
        const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>(size)} {
    props.apply(*this);
  }

  template <typename... Props>
  image(virtual_tag_t, image_format format, image_size_t size,
        const image_properties<Props...> &props = {})
      : impl_{std::make_shared<impl_t>(format, size)} {
    props.apply(*this);
  }

  /// @brief
  /// @param width
  void set_width(int width) { impl_->set_width(width); }

  /// @brief
  /// @param height
  /// @return
  template <int D = Extents> std::enable_if_t<D >= 2> set_height(int height) {
    impl_->set_height(height);
  }

  /// @brief
  /// @param depth
  /// @return
  template <int D = Extents> std::enable_if_t<D >= 3> set_depth(int depth) {
    impl_->set_depth(depth);
  }

  /// @brief
  /// @param format
  void set_format(image_format format) { impl_->set_format(format); }

  template <bool HM = HasMipmaps>
  [[nodiscard]] std::enable_if_t<HM> set_mipmaps(int num_mipmaps) {
    impl_->set_mipmaps(num_mipmaps);
  }

  template <bool HM = HasMultisample>
  [[nodiscard]] std::enable_if_t<HM> set_samples(int num_samples) {
    impl_->set_samples(num_samples);
  }

  /// @brief
  /// @return
  [[nodiscard]] bool has_width() const noexcept { return impl_->has_width(); }

  /// @brief
  /// @return
  template <int D = Extents>
  [[nodiscard]] std::enable_if_t<D >= 2, bool> has_height() const noexcept {
    return impl_->has_height();
  }

  /// @brief
  /// @return
  template <int D = Extents>
  [[nodiscard]] std::enable_if_t<D >= 3, bool> has_depth() const noexcept {
    return impl_->has_depth();
  }

  /// @brief
  /// @return
  [[nodiscard]] bool has_format() const noexcept { return impl_->has_format(); }

  template <bool HM = HasMipmaps>
  [[nodiscard]] std::enable_if_t<HM, bool> has_mipmaps() const noexcept {
    return impl_->has_mipmaps();
  }

  template <bool HM = HasMultisample>
  [[nodiscard]] std::enable_if_t<HM, bool> has_samples() const noexcept {
    return impl_->has_samples();
  }

  /// @brief
  /// @return
  [[nodiscard]] int width() const { return impl_->width(); }

  /// @brief
  /// @return
  template <int D = Extents>
  [[nodiscard]] std::enable_if_t<D >= 2, int> height() const {
    return impl_->height();
  }

  /// @brief
  /// @return
  template <int D = Extents>
  [[nodiscard]] std::enable_if_t<D >= 3, int> depth() const {
    return impl_->depth();
  }

  /// @brief
  /// @return
  [[nodiscard]] image_format format() const { return impl_->format(); }

  /// @brief
  /// @return
  template <bool HM = HasMipmaps>
  [[nodiscard]] std::enable_if_t<HM, int> mipmaps() const {
    return impl_->mipmaps();
  }

  /// @brief
  /// @return
  template <bool HM = HasMultisample>
  [[nodiscard]] std::enable_if_t<HM, int> samples() const {
    return impl_->samples();
  }

  /// @brief
  /// @return
  template <bool Ext = ExternalAccess>
  [[nodiscard]] std::enable_if_t<Ext, GLuint> get_gl_object() const {
    return impl_->get_gl_object();
  }

  template <bool Ext = ExternalAccess>
  [[nodiscard]] std::enable_if_t<
      !Ext, std::shared_ptr<detail::virtual_image_resource>>
  get_virtual_image() {
    return impl_->get_virtual_image();
  }

  template <bool Ext = ExternalAccess> std::enable_if_t<!Ext> discard() {
    return impl_->discard();
  }

  /// @brief
  /// @return
  [[nodiscard]] image_size_t size() const { return impl_->size(); }

  /// @brief
  /// @return
  [[nodiscard]] std::string_view name() const noexcept { return impl_->name(); }

  /// @brief
  /// @param name
  void set_name(std::string name) { impl_->set_name(std::move(name)); }

  /// @brief
  /// @return
  [[nodiscard]] detail::resource_tracker &get_resource_tracker() noexcept {
    return impl_->get_resource_tracker();
  }

  /// @brief
  /// @return
  [[nodiscard]] const detail::resource_tracker &
  get_resource_tracker() const noexcept  {
    return impl_->get_resource_tracker();
  }

private:
  std::shared_ptr<impl_t> impl_;
};

using image_1d = image<image_type::image_1d>;
using image_2d = image<image_type::image_2d>;
using image_3d = image<image_type::image_3d>;
using image_1d_array = image<image_type::image_1d_array>;
using image_2d_array = image<image_type::image_2d_array>;
using image_cube_map = image<image_type::image_cube_map>;
using multisample_image_2d = image<image_type::image_2d_multisample>;
using multisample_image_2d_array =
    image<image_type::image_2d_multisample_array>;

/*
template <image_dimensions Dimensions>
using virtual_image = image<Dimensions, false>;
using virtual_image_1d = image<image_dimensions::image_1d, false>;
using virtual_image_2d = image<image_dimensions::image_2d, false>;
using virtual_image_3d = image<image_dimensions::image_3d, false>;
using virtual_image_1d_array = image<image_dimensions::image_1d_array, false>;
using virtual_image_2d_array = image<image_dimensions::image_2d_array, false>;
using virtual_image_cube_map = image<image_dimensions::image_cube_map, false>;*/

// deduction guides

// clang-format off
template <int D> image(range<D>)
    -> image<extents_to_image_type(D), true>;
template <int D, typename... Props> image(range<D>, const image_properties<Props...>&)
    -> image<extents_to_image_type(D), true>;
template <int D> image(image_format, range<D>)
    -> image<extents_to_image_type(D), true>;
template <int D, typename... Props> image(image_format, range<D>, const image_properties<Props...>&)       
    -> image<extents_to_image_type(D), true>;
template <int D> image(virtual_tag_t, range<D>)
    -> image<extents_to_image_type(D), false>;
template <int D, typename... Props> image(virtual_tag_t, range<D>, const image_properties<Props...>&)
    -> image<extents_to_image_type(D), false>;
template <int D> image(virtual_tag_t, image_format, range<D>) 
    -> image<extents_to_image_type(D), false>;
template <int D, typename... Props> image(virtual_tag_t, image_format, range<D>, const image_properties<Props...>&)
    -> image<extents_to_image_type(D), false>;
// clang-format on

template <typename... Props>
template <image_type Type, bool ExternalAccess>
void image_properties<Props...>::apply(image<Type, ExternalAccess> &img) const {
  std::apply([&](auto... props) { (props(img), ...); }, props_);
}

// Image properties
struct mipmaps {
  int num_mipmaps;

  template <image_type Type, bool ExternalAccess>
  void operator()(image<Type, ExternalAccess> &img) const {
    static_assert(image_type_has_mipmaps(Type),
                  "Cannot set mipmaps for this image type");
    img.set_mipmaps(num_mipmaps);
  }
};

struct multisample {
  int num_samples;

  template <image_type Type, bool ExternalAccess>
  void operator()(image<Type, ExternalAccess> &img) const {
    static_assert(
        image_type_is_multisample(Type),
        "Cannot set the number of samples on a non-multisample image");
    img.set_samples(num_samples);
  }
};

struct sparse {
  template <image_type Type, bool ExternalAccess>
  void operator()(image<Type, ExternalAccess> &img) const {
    // TODO
  }
};

} // namespace graal
