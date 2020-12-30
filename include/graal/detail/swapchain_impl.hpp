#pragma once
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/range.hpp>
#include <graal/image_format.hpp>

#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

class swapchain_impl;

//-----------------------------------------------------------------------------
class swapchain_impl : public image_resource {
    friend class queue_impl;

public:
    swapchain_impl(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface);

    ~swapchain_impl();

    void resize(range_2d framebuffer_size, vk::SurfaceKHR surface);

    [[nodiscard]] uint32_t current_image_index() const noexcept {
        return current_image_;
    }

    [[nodiscard]] vk::Image current_image() const noexcept {
        return images_[current_image_];
    }

    [[nodiscard]] vk::SwapchainKHR vk_swapchain() const noexcept {
        return swapchain_;
    }

    [[nodiscard]] image_format format() const noexcept {
        return format_;
    }

    void acquire_next_image();

private:
    device device_;
    vk::SwapchainKHR swapchain_;
    image_format format_;
    uint32_t current_image_ = 0;
    std::vector<vk::Image> images_;
};

}  // namespace graal::detail