#pragma once
#include <graal/detail/recycler.hpp>
#include <graal/image_type.hpp>
#include <graal/queue_class.hpp>

#include <vk_mem_alloc.h>
#include <memory>
#include <span>
#include <vulkan/vulkan.hpp>

namespace graal {

struct queue_indices {
    uint8_t graphics;
    uint8_t compute;
    uint8_t transfer;
    uint8_t present;
};

namespace detail {

class swapchain_impl;
class queue_impl;

class device_impl {
public:
    device_impl(vk::SurfaceKHR present_surface);
    ~device_impl();

    [[nodiscard]] vk::Instance get_vk_instance() const noexcept {
        return instance_;
    }
    [[nodiscard]] vk::Device get_vk_device() const noexcept {
        return device_;
    }
    [[nodiscard]] vk::PhysicalDevice get_vk_physical_device() const noexcept {
        return physical_device_;
    }

    [[nodiscard]] VmaAllocator get_allocator() const noexcept {
        return allocator_;
    }

    [[nodiscard]] vk::Semaphore create_binary_semaphore();

    void recycle_binary_semaphore(vk::Semaphore semaphore);

    [[nodiscard]] queue_indices get_queue_indices() const noexcept {
        return queue_indices_;
    }

    [[nodiscard]] vk::Queue get_queue_by_index(uint8_t index) const noexcept {
        return queues_[(size_t) index];
    }

    [[nodiscard]] uint32_t get_queue_family_by_index(uint8_t index) const noexcept {
        return queue_family_indices_[(size_t)index];
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
    vk::Queue queues_[max_queues];
    uint32_t queue_family_indices_[max_queues];
    queue_indices queue_indices_;
    VmaAllocator allocator_;
    recycler<vk::Semaphore> free_semaphores_;
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

    [[nodiscard]] vk::Instance get_vk_instance() const noexcept {
        return impl_->get_vk_instance();
    }
    [[nodiscard]] vk::Device get_vk_device() const noexcept {
        return impl_->get_vk_device();
    }
    [[nodiscard]] vk::PhysicalDevice get_vk_physical_device() const noexcept {
        return impl_->get_vk_physical_device();
    }
    [[nodiscard]] VmaAllocator get_allocator() const noexcept {
        return impl_->get_allocator();
    }

    [[nodiscard]] queue_indices get_queue_indices() const noexcept {
        return impl_->get_queue_indices();
    }
    [[nodiscard]] vk::Queue get_queue_by_index(uint8_t index) const noexcept {
        return impl_->get_queue_by_index(index);
    }
    [[nodiscard]] uint32_t get_queue_family_by_index(uint8_t index) const noexcept {
        return impl_->get_queue_family_by_index(index);
    }
    [[nodiscard]] vk::Semaphore create_binary_semaphore() {
        return impl_->create_binary_semaphore();
    }
    void recycle_binary_semaphore(vk::Semaphore semaphore) {
        impl_->recycle_binary_semaphore(semaphore);
    }

private:
    impl_t impl_;
};

}  // namespace graal