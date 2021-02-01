#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <type_traits>

#include <graal/access_mode.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/resource.hpp>
#include <graal/errors.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/image_usage.hpp>
#include <graal/instance.hpp>
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

}  // namespace detail

class image_base {
public:
    image_base(context ctx, image_type type, image_usage usage, image_format format, range<3> size,
            const image_properties& props = {});

    /// @brief
    /// @return
    [[nodiscard]] std::string_view name() const noexcept;

    /// @brief
    /// @param name
    void set_name(std::string name);

    /// @brief
    /// @return
    [[nodiscard]] vk::Image get_handle() const noexcept;

private:
    detail::image_resource_ptr impl_;
};

/// @brief Represents an image.
/// @tparam Type
/// @tparam ExternalAccess
template<image_type Type>
class image final : public image_base {
    friend class handler;
    friend class attachment;

public:
    using impl_t = detail::image_resource;
    using image_size_t = image_size<Type>;

    static constexpr int Extents = num_extents(Type);

    image(context ctx, image_usage usage, image_format format, image_size_t size,
            const image_properties& props = {}) :
        image_base{std::move(ctx), Type, usage, format, detail::extend_size(size), props} {
    }
};

using image_1d = image<image_type::image_1d>;
using image_2d = image<image_type::image_2d>;
using image_3d = image<image_type::image_3d>;

template<int D>
image(context, image_usage, image_format, range<D>) -> image<extents_to_image_type(D)>;

template<int D>
image(context, image_usage, image_format, range<D>, const image_properties&)
        -> image<extents_to_image_type(D)>;

}  // namespace graal
