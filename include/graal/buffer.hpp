#pragma once
#include <graal/access_mode.hpp>
#include <graal/buffer_usage.hpp>
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <memory>
#include <span>
#include <stdexcept>

namespace graal {

struct buffer_properties {
    vk::MemoryPropertyFlags required_flags{};
    vk::MemoryPropertyFlags preferred_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
};

namespace detail {

class buffer_impl : public buffer_resource, public virtual_resource {
public:
    // construct uninitialized from size
    buffer_impl(device dev, buffer_usage usage, std::size_t byte_size, const buffer_properties& properties);

    // construct with initial data
    //buffer_impl(device_impl_ptr device, buffer_usage usage, std::size_t byte_size, const buffer_properties& properties, std::span<std::byte> bytes);

    ~buffer_impl();

    [[nodiscard]] std::size_t byte_size() const noexcept {
        return byte_size_;
    }

    void bind_memory(vk::Device device, VmaAllocation allocation,
        const VmaAllocationInfo& allocation_info) override;

    [[nodiscard]] allocation_requirements get_allocation_requirements(vk::Device device) override;

private:
    device device_;
    std::size_t byte_size_ = -1;
    bool own_allocation_ = false;
    VmaAllocation allocation_ = nullptr;
    VmaAllocationInfo allocation_info_{};
    buffer_properties props_{};
    buffer_usage usage_{};
};

}  // namespace detail

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
        impl_{std::make_shared<detail::buffer_impl>(std::move(dev),
                usage, size * sizeof(T), props)} {
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