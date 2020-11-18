#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>

#include <graal/access_mode.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/resource.hpp>
#include <graal/detail/task.hpp>
#include <graal/errors.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/image_usage.hpp>
#include <graal/range.hpp>
#include <graal/texture.hpp>
#include <graal/visibility.hpp>

namespace graal {

template<image_type Type, bool HostVisible>
class image;

struct image_properties {
    std::optional<unsigned int> mip_levels = std::nullopt;
    std::optional<unsigned int> array_layers = std::nullopt;
    std::optional<unsigned int> num_samples = std::nullopt;
    bool aliasable = false;
};

struct virtual_tag_t {
    explicit virtual_tag_t() = default;
};
inline constexpr virtual_tag_t virtual_image;

namespace detail {

inline constexpr allocation_flags get_allocation_flags(
        bool host_visible, const image_properties& props) noexcept {
    allocation_flags f{};
    if (props.aliasable) { f |= allocation_flag::aliasable; }
    if (host_visible) { f |= allocation_flag::host_visible; }
    return f;
}

template<int Dimensions>
inline constexpr range<3> extend_size(const range<Dimensions>& s) {
    if constexpr (Dimensions == 1) {
        return range{s[0], 1, 1};
    } else if constexpr (Dimensions == 2) {
        return range{s[0], s[1], 1};
    } else {
        return s;
    }
}

template<int Dimensions>
inline constexpr range<Dimensions> cast_size(const range<3>& s) {
    if constexpr (Dimensions == 1) {
        return range{s[0]};
    } else if constexpr (Dimensions == 2) {
        return range{s[0], s[1]};
    } else {
        return s;
    }
}

// TODO is there a use for partially-specified width and height?
// e.g. for array texture: specify width, height, but not depth (array length)

/// @brief
inline constexpr vk::SampleCountFlagBits get_vk_sample_count(unsigned samples) {
    switch (samples) {
        case 1: return vk::SampleCountFlagBits::e1;
        case 2: return vk::SampleCountFlagBits::e2;
        case 4: return vk::SampleCountFlagBits::e4;
        case 8: return vk::SampleCountFlagBits::e8;
        case 16: return vk::SampleCountFlagBits::e16;
        case 32: return vk::SampleCountFlagBits::e32;
        case 64: return vk::SampleCountFlagBits::e64;
        default: throw std::runtime_error{"invalid sample count: not a power of two"};
    }
}

// TODO: should we pass the device to the constructor, or only when first using
// the object?

class image_impl : public virtual_resource {
public:
    image_impl(image_type type, allocation_flags flags) :
        virtual_resource{resource_type::image}, type_{type} {
    }
    image_impl(image_type type, image_format format, allocation_flags flags) :
        virtual_resource{resource_type::image}, type_{type}, format_{format} {
    }
    image_impl(image_type type, range<3> size, allocation_flags flags) :
        virtual_resource{resource_type::image}, type_{type}, size_{size} {
    }

    image_impl(image_type type, image_format format, range<3> size, allocation_flags flags) :
        virtual_resource{ resource_type::image }, type_{type}, format_{format}, size_{size} {
    }

    //-------------------------------------------------------
    void set_size(const range<3>& s) {
        if (size_ != s) { throw std::logic_error{"size already specified"}; }
        size_ = s;
    }

    void set_format(image_format format) {
        if (format_ != format) { throw std::logic_error{"format already specified"}; }
        format_ = format;
    }

    void set_mipmaps(unsigned num_mipmaps) {
        if (mipmaps_ != num_mipmaps) { throw std::logic_error{"mipmaps already specified"}; }
        mipmaps_ = num_mipmaps;
    }

    void set_samples(unsigned num_samples) {
        if (samples_ != num_samples) { throw std::logic_error{"samples already specified"}; }
        samples_ = num_samples;
    }

    void set_array_layers(unsigned array_layers) {
        if (array_layers_ != array_layers) {
            throw std::logic_error{"array layers already specified"};
        }
        array_layers_ = array_layers;
    }

    //-------------------------------------------------------
    bool has_size() const noexcept {
        return size_.has_value();
    }

    bool has_format() const noexcept {
        return format_.has_value();
    }

    bool has_mipmaps() const noexcept {
        return mipmaps_.has_value();
    }

    bool has_samples() const noexcept {
        return samples_.has_value();
    }

    bool has_array_layers() const noexcept {
        return array_layers_.has_value();
    }

    //-------------------------------------------------------
    size_t width() const {
        return size_.value()[0];
    }
    size_t height() const {
        return size_.value()[1];
    }
    size_t depth() const {
        return size_.value()[2];
    }

    range<3> size() const {
        if (!size_.has_value()) { throw std::logic_error("size was unspecified"); }
        return size_.value();
    }

    image_format format() const {
        if (!has_format()) { throw std::logic_error("format was unspecified"); }
        return format_.value();
    }

    unsigned mipmaps() const {
        return mipmaps_.value();
    }

    unsigned samples() const {
        return samples_.value();
    }

    unsigned array_layers() const {
        return array_layers_.value();
    }

    bool is_fully_specified() const noexcept {
        // the only things we need are the size and the format; for the rest, we can
        // use default values
        return has_size() && has_format();
    }

    void add_usage_flags(vk::ImageUsageFlags flags) {
        if (image_) {
            // can't set usage once the resource is created
            throw std::logic_error{"image already created"};
        }
        usage_ = usage_ | flags;
    }

    vk::Image get_vk_image(device& dev);

    allocation_requirements get_allocation_requirements(device_impl_ptr device) override;

    /// @brief See resource::bind_memory
    void bind_memory(device_impl_ptr device, VmaAllocation allocation,
            VmaAllocationInfo allocation_info) override;

protected:
    // Creates the VkImage without binding memory
    vk::Image get_unbound_vk_image(device_impl_ptr dev);

    device_impl_ptr device_impl_;
    image_type type_;
    allocation_flags allocation_flags_;
    std::optional<range<3>> size_;
    std::optional<image_format> format_;
    std::optional<unsigned> samples_;  // multisample count
    std::optional<unsigned> mipmaps_;  // mipmap count
    std::optional<unsigned> array_layers_;  // num array layers
    vk::Image image_;  // nullptr if not created yet
    VmaAllocation allocation_;  // nullptr if not allocated yet
    VmaAllocationInfo allocation_info_;  // allocation info
    vk::ImageUsageFlags usage_;  // vulkan image usage flags; can add usages until
            // the image is created
};

}  // namespace detail

// the delayed image specification is purely an ergonomic decision
// this is because we expect to support functions of the form:
//
//		void filter(const image& input, image& output);
//
// where the filter (the callee) decides the size of the image,
// but the *caller* decides other properties of the image, such as its required
// access (evaluation only, or externally visible)
 

/// @brief Represents an image.
/// @tparam Type
/// @tparam ExternalAccess
template<image_type Type, bool HostVisible = false>
class image final 
{

    template<image_type Type, image_usage Usage, access_mode AccessMode, bool HostVisible>
    friend class image_accessor_base;
    friend class handler;

public:
    using impl_t = detail::image_impl;
    using image_size_t = image_size<Type>;

    static constexpr int Extents = num_extents(Type);

    image(const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(Type, detail::get_allocation_flags(HostVisible, props))} 
    {
        apply_image_properties(props);
    }

    image(image_format format, const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                Type, format, detail::get_allocation_flags(HostVisible, props))} 
    {
        apply_image_properties(props);
    }

    image(image_size_t size, const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                Type, size, detail::get_allocation_flags(HostVisible, props))}
    {
        apply_image_properties(props);
    }

    image(image_format format, image_size_t size, const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(Type, format, detail::extend_size(size),
                detail::get_allocation_flags(HostVisible, props))}
    {
        apply_image_properties(props);
    }

    image(virtual_tag_t, const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(Type, detail::get_allocation_flags(HostVisible, props))}
    {
        apply_image_properties(props);
    }

    image(virtual_tag_t, image_format format, const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                Type, format, detail::get_allocation_flags(HostVisible, props))} 
    {
        apply_image_properties(props);
    }

    image(virtual_tag_t, image_size_t size, const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(Type, detail::extend_size(size),
                detail::get_allocation_flags(HostVisible, props))}
    {
        apply_image_properties(props);
    }

    image(virtual_tag_t, image_format format, image_size_t size,
            const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(Type, format, detail::extend_size(size),
                detail::get_allocation_flags(HostVisible, props))} 
    {
        apply_image_properties(props);
    }



    /// @brief
    /// @param size
    void set_size(const image_size_t& size) {
        impl_->set_size(detail::extend_size(size));
    }

    /// @brief
    /// @param format
    void set_format(image_format format) {
        impl_->set_format(format);
    }

    void set_mipmaps(unsigned num_mipmaps) {
        impl_->set_mipmaps(num_mipmaps);
    }

    void set_samples(unsigned num_samples) {
        impl_->set_samples(num_samples);
    }

    void set_array_layers(unsigned array_layers) {
        impl_->set_array_layers(array_layers);
    }

    /// @brief Combines the specified usage with the current image usage flags.
    /// @param usage
    /// You don't need to call this function when using images with accessors,
    /// as the accessors will set automatically set proper image flags.
    void add_usage_flags(vk::ImageUsageFlags usage) {
        impl_->add_usage_flags(usage);
    }

    /// @brief
    /// @return
    [[nodiscard]] bool has_size() const noexcept {
        return impl_->has_size();
    }

    /// @brief
    /// @return
    [[nodiscard]] bool has_format() const noexcept {
        return impl_->has_format();
    }

    [[nodiscard]] bool has_mipmaps() const noexcept {
        return impl_->has_mipmaps();
    }

    [[nodiscard]] bool has_samples() const noexcept {
        return impl_->has_samples();
    }

    [[nodiscard]] bool has_array_layers() const noexcept {
        return impl_->has_array_layers();
    }

    /// @brief
    /// @return
    [[nodiscard]] size_t width() const {
        return impl_->width();
    }

    /// @brief
    /// @return
    template<int D = Extents>
    [[nodiscard]] std::enable_if_t<D >= 2, size_t> height() const {
        return impl_->height();
    }

    /// @brief
    /// @return
    template<int D = Extents>
    [[nodiscard]] std::enable_if_t<D >= 3, size_t> depth() const {
        return impl_->depth();
    }

    /// @brief
    /// @return
    [[nodiscard]] image_format format() const {
        return impl_->format();
    }

    /// @brief
    /// @return
    [[nodiscard]] unsigned mipmaps() const {
        return impl_->mipmaps();
    }

    /// @brief
    /// @return
    [[nodiscard]] unsigned samples() const {
        return impl_->samples();
    }

    [[nodiscard]] unsigned array_layers() const {
        return impl_->array_layers();
    }

    /*template <bool Ext = HostVisible> std::enable_if_t<!Ext> discard() {
    return impl_->discard();
  }*/

    /// @brief
    /// @return
    [[nodiscard]] image_size_t size() const {
        return detail::cast_size<Extents>(impl_->size());
    }

    /// @brief
    /// @return
    [[nodiscard]] std::string_view name() const noexcept {
        return impl_->name();
    }

    /// @brief
    /// @param name
    void set_name(std::string name) {
        impl_->set_name(std::move(name));
    }

private:
    void apply_image_properties(const image_properties& props) {
        if ((bool) props.array_layers) { set_array_layers(props.array_layers.value()); }
        if ((bool) props.mip_levels) { set_mipmaps(props.mip_levels.value()); }
        if ((bool) props.num_samples) { set_samples(props.num_samples.value()); }
    }

    detail::user_resource_ptr<impl_t> impl_;
};

using image_1d = image<image_type::image_1d>;
using image_2d = image<image_type::image_2d>;
using image_3d = image<image_type::image_3d>;
/*using image_1d_array = image<image_type::image_1d_array>;
using image_2d_array = image<image_type::image_2d_array>;
using image_cube_map = image<image_type::image_cube_map>;
using multisample_image_2d = image<image_type::image_2d_multisample>;
using multisample_image_2d_array =
    image<image_type::image_2d_multisample_array>;*/

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
template <int D> image(range<D>, const image_properties&)
    -> image<extents_to_image_type(D), true>;
template <int D> image(image_format, range<D>)
    -> image<extents_to_image_type(D), true>;
template <int D> image(image_format, range<D>, const image_properties&)       
    -> image<extents_to_image_type(D), true>;
template <int D> image(virtual_tag_t, range<D>)
    -> image<extents_to_image_type(D), false>;
template <int D> image(virtual_tag_t, range<D>, const image_properties&)
    -> image<extents_to_image_type(D), false>;
template <int D> image(virtual_tag_t, image_format, range<D>) 
    -> image<extents_to_image_type(D), false>;
template <int D> image(virtual_tag_t, image_format, range<D>, const image_properties&)
    -> image<extents_to_image_type(D), false>;
// clang-format on

}  // namespace graal
