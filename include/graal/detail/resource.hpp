#pragma once
#include <graal/detail/named_object.hpp>
#include <graal/detail/task.hpp>
#include <graal/device.hpp>
#include <graal/flags.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <memory>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

/// @brief Flags common to all resource types.
enum class allocation_flag {
    host_visible = (1 << 0),  ///< Resource can be mapped in host memory.
    aliasable = (1 << 1),  ///< The memory of the resource can be aliased to other resources.
};
}  // namespace graal::detail

GRAAL_ALLOW_FLAGS_FOR_ENUM(graal::detail::allocation_flag)

namespace graal::detail {

using allocation_flags = flags<allocation_flag>;

/// @brief
struct allocation_requirements {
    allocation_flags flags;
    uint32_t memory_type_bits;
    size_t size;
    size_t alignment;
};

enum class resource_type {
    swapchain_image,  // can cast to detail::swapchain_image_impl
    image,  // can cast to detail::virtual_resource
    buffer,  // can cast to detail::buffer_resource
};

/// @brief Base class for tracked resources.
class resource : public named_object {
    friend class queue_impl;

public:
    resource(resource_type type) : type_{type} {
    }

    [[nodiscard]] bool is_virtual() const noexcept {
        return type_ == resource_type::image || type_ == resource_type::buffer;
    }
    [[nodiscard]] resource_type type() const noexcept {
        return type_;
    }

    bool allocated = false;  // set to true once bind_memory has been called successfully
    bool discarded = false;  // no more external references to this resource via user handles
    temporary_index tmp_index = -1;  // assigned temporary index
    sequence_number last_write_sequence_number = 0;

private:
    resource_type type_;
};

using resource_ptr = std::shared_ptr<resource>;

/// @brief Base class for resources whose memory is managed by the queue.
class virtual_resource : public resource {
    friend class queue_impl;

public:
    virtual_resource(resource_type type) : resource{type} {
    }

    /// @brief Returns the memory requirements of the resource.
    virtual allocation_requirements get_allocation_requirements(device_impl_ptr dev) = 0;

    /// @brief
    /// @param other
    virtual void bind_memory(
            device_impl_ptr dev, VmaAllocation allocation, VmaAllocationInfo allocation_info) = 0;
};

using virtual_resource_ptr = std::shared_ptr<virtual_resource>;

}  // namespace graal::detail