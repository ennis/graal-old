#pragma once
#include <graal/device.hpp>
#include <graal/range.hpp>

#include <vulkan/vulkan.hpp>

namespace graal {
namespace detail {
class queue_impl;
class swapchain_impl;
class swapchain_image_impl;
}  // namespace detail

/// @brief
class swapchain_image {
    friend class detail::queue_impl;

public:
    // semaphore to synchronize with the presentation engine on first access
    vk::Semaphore image_available_semaphore() const;

private:
    std::shared_ptr<detail::swapchain_image_impl> impl_;
};

/// @brief
class swapchain {
    friend class detail::queue_impl;

public:
    swapchain(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface);

    void resize(range_2d framebuffer_size, vk::SurfaceKHR surface);
    swapchain_image acquire_next_image();

private:
    std::shared_ptr<detail::swapchain_impl> impl_;
};

}  // namespace graal