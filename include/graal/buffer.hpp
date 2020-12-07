#pragma once
#include <graal/glad.h>
#include <graal/access_mode.hpp>
#include <graal/buffer_usage.hpp>
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/raw_buffer.hpp>
#include <memory>
#include <span>
#include <stdexcept>

namespace graal {

struct buffer_properties {
    bool aliasable = false;
};

namespace detail {

inline constexpr allocation_flags get_buffer_allocation_flags(
        bool host_visible, const buffer_properties& props) noexcept {
    allocation_flags f = {};
    if (props.aliasable) { f |= allocation_flags::aliasable; }
    if (host_visible) { f |= allocation_flags::host_visible; }
    return f;
}

class buffer_impl : public buffer_resource, public virtual_resource {
public:
    // construct with unspecified size
    buffer_impl(buffer_usage usage, allocation_flags flags) :
        buffer_resource{}, usage_{usage}, allocation_flags_{flags} {
    }

    // construct uninitialized from size
    buffer_impl(buffer_usage usage, std::size_t byte_size, allocation_flags flags) :
        buffer_resource{}, usage_{usage}, allocation_flags_{flags}, byte_size_{byte_size} {
    }

    ~buffer_impl() {
    }

    void set_byte_size(std::size_t size) {
        if (has_byte_size() && byte_size_ != size) {
            throw std::logic_error{"size already specified"};
        }
        byte_size_ = size;
    }

    std::size_t byte_size() const {
        if (!has_byte_size()) { throw std::logic_error{"size was unspecified"}; }
        return byte_size_;
    }

    bool has_byte_size() const noexcept {
        return byte_size_ != -1;
    }

    void bind_memory(device_impl_ptr dev, VmaAllocation allocation,
        const VmaAllocationInfo& allocation_info) override;

    allocation_requirements get_allocation_requirements(device_impl_ptr dev) override;

    void add_usage(buffer_usage usage) noexcept {
        usage_ |= usage;
    }

    vk::Buffer get_vk_buffer(device_impl_ptr dev);

private:
    allocation_flags allocation_flags_{};
    std::size_t byte_size_ = -1;
    VmaAllocation allocation_ = nullptr;
    VmaAllocationInfo allocation_info_{};
    buffer_usage usage_{};
};

}  // namespace detail

/// @brief
/// @tparam T
// HostVisible enables the "map" methods.
template<typename T, bool HostVisible = true>
class buffer {
    template<typename DataT, buffer_usage Usage, access_mode AccessMode>
    friend class buffer_accessor_base;
    friend class handler;

public:
    using impl_t = detail::buffer_impl;
    /// @brief
    buffer(const buffer_properties& props = {}) :
        impl_{ std::make_shared<impl_t>({}, detail::get_buffer_allocation_flags(HostVisible, props)) } {
    }

    /// @brief
    buffer(buffer_usage usage, const buffer_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                usage, detail::get_buffer_allocation_flags(HostVisible, props))} {
    }

    /// @brief
    /// @param size
    buffer(buffer_usage usage, std::size_t size, const buffer_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                usage, size * sizeof(T), detail::get_buffer_allocation_flags(HostVisible, props))} {
    }

    /// @brief
    /// @param data
    template<size_t Extent>
    buffer(buffer_usage usage, std::span<const T, Extent> data,
            const buffer_properties& props = {}) :
        impl_{std::make_shared<impl_t>(
                usage, data.size(), detail::get_buffer_allocation_flags(HostVisible, props))} {
        // TODO allocate and upload data
    }

    /// @brief
    /// @param size
    void set_size(std::size_t size) {
        impl_->set_byte_size(size * sizeof(T));
    }

    /// @brief
    /// @return
    std::size_t size() const {
        return impl_->byte_size() / sizeof(T);
    }

    /// @brief
    /// @return
    bool has_size() const noexcept {
        return impl_->has_byte_size();
    }

private:
    detail::user_resource_ptr<impl_t> impl_;
};

// deduction guides
template<typename T>
buffer(buffer_usage, std::span<const T>) -> buffer<T, true>;

}  // namespace graal