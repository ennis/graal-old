#pragma once
#include <graal/detail/named_object.hpp>
#include <graal/detail/sequence_number.hpp>
#include <graal/device.hpp>
#include <graal/bitmask.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>

#include <vk_mem_alloc.h>
#include <atomic>
#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

/// @brief Flags common to all resource types.
enum class allocation_flags {
    host_visible = (1 << 0),  ///< Resource can be mapped in host memory.
    aliasable = (1 << 1),  ///< The memory of the resource can be aliased to other resources.
};

GRAAL_BITMASK(allocation_flags)

/// @brief
struct allocation_requirements {
    allocation_flags flags;
    VkMemoryRequirements memreq;
};

enum class resource_type {
    swapchain_image,  // can cast to detail::swapchain_image_impl
    image,  // can cast to detail::virtual_resource
    buffer,  // can cast to detail::buffer_resource
};

class resource;
class image_resource;
class buffer_resource;
class virtual_resource;

/// @brief Base class for tracked resources.
class resource : public named_object {
    friend class queue_impl;

public:
    resource(resource_type type) : type_{type} {
        std::fill(last_access.begin(), last_access.end(), 0);
    }

    /// @brief If the resource is an image, returns the corresponding VkImage object.
    /// @param device The device used to create the VkImage object, if it was not created yet.
    /// @return nullptr if the resource is not an image.
    image_resource& as_image();
    const image_resource& as_image() const;

    /// @brief If the resource is an image, returns the corresponding VkImage object.
    /// @param device The device used to create the VkImage object, if it was not created yet.
    /// @return nullptr if the resource is not an image.
    buffer_resource& as_buffer();
    const buffer_resource& as_buffer() const;

    virtual_resource& as_virtual_resource();

    [[nodiscard]] bool is_virtual() const noexcept {
        return type_ == resource_type::image || type_ == resource_type::buffer;
    }

    [[nodiscard]] resource_type type() const noexcept {
        return type_;
    }

    void realize(device_impl_ptr device);

    void add_user_ref() const noexcept {
        user_ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] uint32_t user_ref_count() const noexcept { return user_ref_count_; }

    void release_user_ref() const noexcept {
        // relaxed should be sufficient since we're not using the counter for synchronization
        user_ref_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    [[nodiscard]] bool has_user_refs() const noexcept {
        return user_ref_count_.load(std::memory_order_relaxed) != 0;
    }

    bool allocated = false;  // set to true once bind_memory has been called successfully
    submission_number last_write;   // submission number of the last write
    std::array<serial_number, max_queues> last_access;    // serial numbers of the last read accesses for each queue

    vk::ImageLayout last_layout = vk::ImageLayout::eUndefined;            // last known image layout, ignored for buffers
    vk::AccessFlags last_access_flags = {};
    vk::PipelineStageFlags last_pipeline_stages;      
    vk::Semaphore wait_semaphore =
            nullptr;  // semaphore to synchronize on before using the resource (updated as the resource is used in a queue)
    // TODO use a unique_handle pattern to signal ownership. Not the one provided by vulkan-hpp though because
    // it bundles pointers to the device and allocation infos to allow ad-hoc deletion.
private:
    // For each resource, we maintain an "user reference count" which represents the number of
    // user-facing objects (image<>, buffer<>, etc.) referencing the resource. This is used
    // when submitting a batch to determine whether there exists user-facing references to the resource.
    // If not, then the queue can perform optimizations knowing that the program will not use the resource in following batches.
    mutable std::atomic_uint32_t user_ref_count_ = 0;
    resource_type type_;
};

using resource_ptr = std::shared_ptr<resource>;

/// @brief Base class for image resources.
class image_resource : public resource {
public:
    image_resource(resource_type type) : resource{ type }
    {}

    vk::Image image = nullptr;
    image_format format = image_format::undefined;
};


/// @brief Base class for buffer resources.
class buffer_resource : public resource {
public:
    buffer_resource() : resource{ resource_type::buffer }
    {}

    vk::Buffer buffer = nullptr;
    size_t size = 0;
};


/// @brief Base interface for resources whose memory is managed by the queue.
class virtual_resource {
    friend class queue_impl;
public:
    /// @brief Returns the memory requirements of the resource.
    virtual allocation_requirements get_allocation_requirements(device_impl_ptr dev) = 0;

    /// @brief
    /// @param other
    virtual void bind_memory(
        device_impl_ptr dev, VmaAllocation allocation, const VmaAllocationInfo& allocation_info) = 0;
};

using virtual_resource_ptr = std::shared_ptr<virtual_resource>;


/// @brief 
/// @tparam T 
template<typename T>
class user_resource_ptr {
    static_assert(std::is_base_of_v<resource, T>);

public:
    constexpr user_resource_ptr() noexcept = default;
    constexpr user_resource_ptr(std::nullptr_t) noexcept {}

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

}  // namespace graal::detail