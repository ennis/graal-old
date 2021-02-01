#pragma once
#include <graal/detail/recycler.hpp>
#include <graal/detail/vk_handle.hpp>
#include <graal/image_type.hpp>
#include <graal/queue_class.hpp>

#include <vk_mem_alloc.h>
#include <memory>
#include <span>
#include <vulkan/vulkan.hpp>

namespace graal {

extern vk::DispatchLoaderDynamic vk_default_dynamic_loader;

struct queue_indices {
    uint8_t graphics;
    uint8_t compute;
    uint8_t transfer;
    uint8_t present;
};

struct queues_info {
    size_t queue_count;
    queue_indices indices;
    std::array<uint32_t, max_queues> families;
    std::array<vk::Queue, max_queues> queues;
};

namespace detail {

class swapchain_impl;
class queue_impl;

class device_impl {
public:
    device_impl(vk::SurfaceKHR present_surface);
    ~device_impl();

    [[nodiscard]] vk::Instance get_instance_handle() const noexcept {
        return instance_;
    }
    [[nodiscard]] vk::Device get_handle() const noexcept {
        return device_;
    }
    [[nodiscard]] vk::PhysicalDevice get_physical_device_handle() const noexcept {
        return physical_device_;
    }

    [[nodiscard]] VmaAllocator get_allocator() const noexcept {
        return allocator_;
    }

    [[nodiscard]] handle<vk::Semaphore> create_binary_semaphore();

    void recycle_binary_semaphore(handle<vk::Semaphore> semaphore);

    [[nodiscard]] queues_info get_queues_info() const noexcept {
        return queues_info_;
    }

private:
    void create_vk_device_and_queues(vk::SurfaceKHR present_surface);

    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::PhysicalDeviceProperties physical_device_properties_;
    vk::PhysicalDeviceFeatures physical_device_features_;
    vk::Device device_;
    uint32_t graphics_queue_family_;
    uint32_t compute_queue_family_;
    uint32_t transfer_queue_family_;
    queues_info queues_info_;
    VmaAllocator allocator_;
    handle_vector<vk::Semaphore> free_semaphores_;
};

using device_impl_ptr = std::shared_ptr<device_impl>;

}  // namespace detail

/// @brief Vulkan instance and device
class device {
    friend class queue;
    friend class detail::queue_impl;
    friend class detail::swapchain_impl;

    template<image_type Type>
    friend class image;

    template<typename T>
    friend class buffer;

public:
    using impl_t = std::shared_ptr<detail::device_impl>;

    device(vk::SurfaceKHR present_surface) :
        impl_{std::make_shared<detail::device_impl>(present_surface)} {
    }

    [[nodiscard]] vk::Instance get_instance_handle() const noexcept {
        return impl_->get_instance_handle();
    }
    [[nodiscard]] vk::Device get_handle() const noexcept {
        return impl_->get_handle();
    }
    [[nodiscard]] vk::PhysicalDevice get_physical_device_handle() const noexcept {
        return impl_->get_physical_device_handle();
    }
    [[nodiscard]] VmaAllocator get_allocator() const noexcept {
        return impl_->get_allocator();
    }

    [[nodiscard]] queues_info get_queues_info() const noexcept {
        return impl_->get_queues_info();
    }

    [[nodiscard]] detail::handle<vk::Semaphore> create_binary_semaphore() {
        return impl_->create_binary_semaphore();
    }
    void recycle_binary_semaphore(detail::handle<vk::Semaphore> semaphore) {
        impl_->recycle_binary_semaphore(std::move(semaphore));
    }

private:
    impl_t impl_;
};

}  // namespace graal