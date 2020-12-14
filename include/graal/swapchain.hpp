#pragma once
#include <graal/device.hpp>
#include <graal/range.hpp>
#include <graal/detail/swapchain_impl.hpp>

#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal {

namespace detail {
class queue_impl;
class swapchain_image_impl;
class swapchain_impl;
}  // namespace detail

/// @brief
class swapchain_image {
    friend class swapchain;
    friend class handler;
    friend class attachment;
    friend class detail::queue_impl;

public:
    [[nodiscard]] uint32_t index() const noexcept;
    [[nodiscard]] vk::SwapchainKHR get_vk_swapchain() const noexcept;

private:
    swapchain_image(std::shared_ptr<detail::swapchain_image_impl> impl) : impl_{std::move(impl)} {}
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