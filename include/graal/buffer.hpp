#pragma once
#include <graal/access_mode.hpp>
#include <graal/buffer_usage.hpp>
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/instance.hpp>

#include <memory>
#include <span>
#include <stdexcept>

namespace graal {

struct buffer_properties {
    vk::MemoryPropertyFlags required_flags{};
    vk::MemoryPropertyFlags preferred_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
};


/// @brief
/// @tparam T
template<typename T>
class buffer {
    template<typename DataT, buffer_usage Usage, access_mode AccessMode>
    friend class buffer_accessor_base;
    friend class handler;

public:
    /// @brief
    /// @param size
    buffer(device dev, buffer_usage usage, std::size_t size, const buffer_properties& props = {}) :
        impl_{std::make_shared<detail::buffer_impl>(
                std::move(dev), usage, size * sizeof(T), props)} {
    }

    /*/// @brief
    /// @param data
    template<size_t Extent>
    buffer(buffer_usage usage, std::span<const T, Extent> data,
            const buffer_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                usage, data.size(), props)} {
        // TODO allocate and upload data
        // if the memory requirements do not include HOST_VISIBLE, then 
        // we must use a staging buffer, and thus we require a queue (upload is asynchronous)
    }*/

    /// @brief
    /// @return
    [[nodiscard]] std::size_t size() const noexcept {
        return impl_->byte_size() / sizeof(T);
    }

private:
    detail::user_resource_ptr<detail::buffer_impl> impl_;
};

// deduction guides
/*template<typename T>
buffer(device dev, buffer_usage, std::span<const T>) -> buffer<T>;
template<typename T>
buffer(device dev, buffer_usage, std::span<const T>, const buffer_properties&)->buffer<T>;*/

}  // namespace graal