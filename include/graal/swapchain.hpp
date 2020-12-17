#pragma once
#include <graal/device.hpp>
#include <graal/range.hpp>
#include <graal/detail/swapchain_impl.hpp>

#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal {

namespace detail {
class queue_impl;
class swapchain_impl;
}  // namespace detail

/// @brief
class swapchain {
    friend class detail::queue_impl;
    friend class attachment;
    friend class handler;

public:
    swapchain(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface);

    /// @brief Resizes the swapchain
    /// @param framebuffer_size 
    /// @param surface 
    void resize(range_2d framebuffer_size, vk::SurfaceKHR surface);
    
    /// Returns the VkSwapchainKHR object
    [[nodiscard]] vk::SwapchainKHR vk_swapchain() const noexcept;

/// @brief Returns the current swapchain image.
    [[nodiscard]] vk::Image current_image() const noexcept;

    [[nodiscard]] uint32_t current_image_index() const noexcept;

private:
    std::shared_ptr<detail::swapchain_impl> impl_;
};

}  // namespace graal