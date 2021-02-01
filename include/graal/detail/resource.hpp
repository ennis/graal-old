#pragma once
#include <graal/bitmask.hpp>
#include <graal/detail/sequence_number.hpp>
#include <graal/device.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>

#include <vk_mem_alloc.h>
#include <atomic>
#include <memory>
#include <vulkan/vulkan.hpp>

#include <boost/intrusive_ptr.hpp>

namespace graal::detail {

/// @brief Flags common to all resource types.
enum class allocation_flags {
    host_visible = (1 << 0),  ///< Resource can be mapped in host memory.
    aliasable = (1 << 1),  ///< The memory of the resource can be aliased to other resources.
};

GRAAL_BITMASK(allocation_flags)

/// @brief
struct allocation_requirements {
    VkMemoryRequirements memreq;
    vk::MemoryPropertyFlags required_flags{};
    vk::MemoryPropertyFlags preferred_flags{};
};

enum class resource_type {
    image,
    buffer,
};

class resource;
class image_resource;
class buffer_resource;
class virtual_resource;

struct resource_access_tracking_info {
    /// @brief Readers. There can be concurrent readers on different queues.
    std::array<serial_number, max_queues> readers{};
    /// @brief Last writer.
    submission_number writer;
    /// @brief Current image layout.
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    /// @brief Flags that describe access types (writes) that have not yet been made available (caches that have not been flushed yet).
    vk::AccessFlags availability_mask{};
    /// @brief Flags that describe access types (reads) that can see the last write to the resources (invalidated caches).
    vk::AccessFlags visibility_mask{};
    /// @brief Stages accessing the resource.
    vk::PipelineStageFlags stages{};
    /// @brief Binary semaphore guarding access to the resource. Only used by swapchain images.
    // TODO who is responsible of recycling that if it's unconsumed?
    handle<vk::Semaphore> wait_semaphore;

    [[nodiscard]] bool has_readers() const noexcept {
        for (auto r : readers) {
            if (r) return true;
        }
        return false;
    }

    void clear_readers() noexcept {
        for (auto& r : readers) {
            r = 0;
        }
    }
};

/// @brief Base class for tracked resources.
struct resource {
    resource(resource_type type) : type{type} {
    }

    void add_ref() const noexcept {
        ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    void release() const noexcept {
        // relaxed should be sufficient since we're not using the counter for synchronization
        ref_count.fetch_sub(1, std::memory_order_relaxed);
    }

    void add_user_ref() const noexcept {
        user_ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    void release_user_ref() const noexcept {
        // relaxed should be sufficient since we're not using the counter for synchronization
        user_ref_count.fetch_sub(1, std::memory_order_relaxed);
    }

    [[nodiscard]] vk::Image get_image_handle() const noexcept;
    [[nodiscard]] vk::Buffer get_buffer_handle() const noexcept;

    void bind_memory(
            vk::Device vkd, VmaAllocation allocation, const VmaAllocationInfo& allocation_info) 
    {
        assert(!allocation);
        assert(owned);

        this->allocation = allocation;
        this->allocation_info = allocation_info;

        if (type == resource_type::image) {
            vkBindImageMemory(
                    vkd, get_image_handle(), allocation_info.deviceMemory, allocation_info.offset);
        } else {
            vkBindBufferMemory(
                    vkd, get_buffer_handle(), allocation_info.deviceMemory, allocation_info.offset);
        }
    }

    allocation_requirements get_allocation_requirements(vk::Device vkd) const
    {
        vk::MemoryRequirements mem_req;
        if (type == resource_type::image) {
            mem_req = vkd.getImageMemoryRequirements(get_image_handle());
        } else {
            mem_req = vkd.getBufferMemoryRequirements(get_buffer_handle());
        }
        return allocation_requirements{.memreq = mem_req,
                .required_flags = mem_required_flags,
                .preferred_flags = mem_preferred_flags};
    }

    bool owned = true;
    vk::MemoryPropertyFlags mem_required_flags{};
    vk::MemoryPropertyFlags mem_preferred_flags{};
    VmaAllocation allocation = nullptr;  // nullptr if not allocated yet
    VmaAllocationInfo allocation_info{};  // allocation info
    resource_access_tracking_info access;
    std::string name;
    resource_type type;
    size_t tmp_index = static_cast<size_t>(-1);
    mutable std::atomic_uint32_t ref_count = 0;
    mutable std::atomic_uint32_t user_ref_count = 0;
};

void intrusive_ptr_add_ref(resource* r) {
    r->add_ref();
}

void intrusive_ptr_release(resource* r) {
    r->release();
}

using resource_ptr = boost::intrusive_ptr<resource>;

struct image_resource : resource {
    image_resource() : resource{resource_type::image} {
    }

    vk::Image image = nullptr;
};

using image_resource_ptr = boost::intrusive_ptr<image_resource>;

struct buffer_resource : resource {
    buffer_resource() : resource{resource_type::buffer} {
    }

    vk::Buffer buffer = nullptr;
};

using buffer_resource_ptr = boost::intrusive_ptr<buffer_resource>;

inline vk::Image resource::get_image_handle() const noexcept {
    return static_cast<const image_resource*>(this)->image;
}

inline vk::Buffer resource::get_buffer_handle() const noexcept {
    return static_cast<const buffer_resource*>(this)->buffer;
}

/*
/// @brief
/// @tparam T
template<typename T>
class user_resource_ptr {
    static_assert(std::is_base_of_v<resource, T>);

public:
    constexpr user_resource_ptr() noexcept = default;
    constexpr user_resource_ptr(std::nullptr_t) noexcept {
    }

    user_resource_ptr(std::shared_ptr<T> ptr) noexcept : ptr_{std::move(ptr)} {
        add_ref();
    }

    user_resource_ptr(const user_resource_ptr<T>& rhs) noexcept : ptr_{rhs.ptr_} {
        add_ref();
    }

    user_resource_ptr(user_resource_ptr<T>&& rhs) noexcept : ptr_{std::move(rhs.ptr_)} {
    }

    ~user_resource_ptr() {
        release();
    }

    user_resource_ptr<T>& operator=(const std::shared_ptr<T>& rhs) noexcept {
        release();
        ptr_ = rhs.ptr_;
        add_ref();
    }

    user_resource_ptr<T>& operator=(const user_resource_ptr<T>& rhs) noexcept {
        release();
        ptr_ = rhs.ptr_;
        add_ref();
    }

    user_resource_ptr& operator=(std::shared_ptr<T>&& rhs) noexcept {
        release();
        ptr_ = std::move(rhs.ptr_);
        add_ref();
    }

    user_resource_ptr& operator=(user_resource_ptr<T>&& rhs) noexcept {
        release();
        ptr_ = std::move(rhs.ptr_);
    }

    void swap(user_resource_ptr& r) noexcept {
        // no changes in user ref count
        ptr_.swap(r.ptr_);
    }

    T* get() const noexcept {
        return ptr_.get();
    }
    T& operator*() const noexcept {
        return *get();
    }
    T* operator->() const noexcept {
        return get();
    }
    explicit operator bool() const noexcept {
        return get() != nullptr;
    }
    operator std::shared_ptr<T>() const noexcept {
        return ptr_;
    }

private:
    void add_ref() const noexcept {
        if (ptr_) ptr_->add_user_ref();
    }

    void release() const noexcept {
        if (ptr_) ptr_->release_user_ref();
    }

    std::shared_ptr<T> ptr_;
};
*/

}  // namespace graal::detail