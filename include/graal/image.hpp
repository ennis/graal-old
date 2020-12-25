#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <cmath>

#include <graal/access_mode.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/resource.hpp>
#include <graal/errors.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/image_usage.hpp>
#include <graal/range.hpp>
#include <graal/visibility.hpp>

namespace graal {

template<image_type Type>
class image;

struct image_properties {
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;
    uint32_t samples = 1;
    vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
    vk::MemoryPropertyFlags required_flags{};
    vk::MemoryPropertyFlags preferred_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
};

inline uint32_t get_mip_level_count(size_t width, size_t height) noexcept {
    return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

struct virtual_tag_t {
    explicit virtual_tag_t() = default;
};
inline constexpr virtual_tag_t virtual_image;

namespace detail {

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
inline constexpr range<Dimensions> cast_size(const range<3>& s) noexcept {
    if constexpr (Dimensions == 1) {
        return range{s[0]};
    } else if constexpr (Dimensions == 2) {
        return range{s[0], s[1]};
    } else {
        return s;
    }
}

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

class image_impl : public image_resource, public virtual_resource {
public:
    image_impl(device dev, image_type type, image_usage usage, image_format format,
            range<3> size, const image_properties& props);
    ~image_impl();

    //-------------------------------------------------------
    [[nodiscard]] size_t width() const noexcept {
        return size_[0];
    }
    [[nodiscard]] size_t height() const noexcept {
        return size_[1];
    }
    [[nodiscard]] size_t depth() const noexcept {
        return size_[2];
    }

    [[nodiscard]] range<3> size() const noexcept {
        return size_;
    }

    /*[[nodiscard]] image_format format() const noexcept {
        return format_;
    }*/

    [[nodiscard]] unsigned mip_levels() const noexcept {
        return props_.mip_levels;
    }

    [[nodiscard]] unsigned samples() const noexcept {
        return props_.samples;
    }

    [[nodiscard]] unsigned array_layers() const noexcept {
        return props_.array_layers;
    }

    /*[[nodiscard]] vk::Image vk_image() const noexcept {
        return image_resource::image_;
    }*/

    allocation_requirements get_allocation_requirements(vk::Device device) override;

    /// @brief See resource::bind_memory
    void bind_memory(vk::Device device, VmaAllocation allocation,
            const VmaAllocationInfo& allocation_info) override;

protected:
    device device_;
    image_type type_;
    image_usage usage_{};  // can add usages until the image is created
    range<3> size_;
    //image_format format_;
    image_properties props_;
    VmaAllocation allocation_;  // nullptr if not allocated yet
    VmaAllocationInfo allocation_info_;  // allocation info
};

}  // namespace detail

/// @brief Represents an image.
/// @tparam Type
/// @tparam ExternalAccess
template<image_type Type>
class image final {
    friend class handler;
    friend class attachment;

public:
    using impl_t = detail::image_impl;
    using image_size_t = image_size<Type>;

    static constexpr int Extents = num_extents(Type);

    image(device device, image_usage usage, image_format format, image_size_t size,
            const image_properties& props = {}) :
        impl_{std::make_shared<impl_t>(std::move(device), Type, usage, format, detail::extend_size(size), props)} {
    }

    /// @brief
    /// @return
    [[nodiscard]] size_t width() const noexcept {
        return impl_->width();
    }

    /// @brief
    /// @return
    template<int D = Extents>
    [[nodiscard]] std::enable_if_t<D >= 2, size_t> height() const noexcept {
        return impl_->height();
    }

    /// @brief
    /// @return
    template<int D = Extents>
    [[nodiscard]] std::enable_if_t<D >= 3, size_t> depth() const noexcept {
        return impl_->depth();
    }

    /// @brief
    /// @return
    [[nodiscard]] image_format format() const noexcept {
        return impl_->format();
    }

    /// @brief
    /// @return
    [[nodiscard]] unsigned mipmaps() const noexcept {
        return impl_->mipmaps();
    }

    /// @brief
    /// @return
    [[nodiscard]] unsigned samples() const noexcept {
        return impl_->samples();
    }

    [[nodiscard]] unsigned array_layers() const noexcept {
        return impl_->array_layers();
    }

    /// @brief
    /// @return
    [[nodiscard]] image_size_t size() const noexcept {
        return detail::cast_size<Extents>(impl_->size());
    }

    /// @brief
    /// @return
    [[nodiscard]] std::string_view name() const noexcept {
        return impl_->name();
    }

    /// @brief
    /// @return
    [[nodiscard]] vk::Image vk_image() const noexcept {
        return impl_->vk_image();
    }

    /// @brief
    /// @param name
    void set_name(std::string name) {
        impl_->set_name(std::move(name));
    }

private:
    detail::user_resource_ptr<impl_t> impl_;
};

using image_1d = image<image_type::image_1d>;
using image_2d = image<image_type::image_2d>;
using image_3d = image<image_type::image_3d>;

template<int D>
image(device, image_usage, image_format, range<D>)
        -> image<extents_to_image_type(D)>;

template<int D>
image(device, image_usage, image_format, range<D>, const image_properties&)
->image<extents_to_image_type(D)>;

}  // namespace graal
