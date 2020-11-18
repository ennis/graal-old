#pragma once
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/range.hpp>

#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

class swapchain_image_impl : public resource {
    friend class swapchain_impl;

public:
    swapchain_image_impl(vk::Image image, size_t index) :
        resource{resource_type::swapchain_image}, image_{image} {
    }

private:
    vk::Image image_;
    size_t index_;
    //std::shared_ptr<detail::swapchain_impl> swapchain_;
};

class swapchain_impl {
public:
    swapchain_impl(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface);

    ~swapchain_impl();

    void resize(range_2d framebuffer_size, vk::SurfaceKHR surface);

    std::shared_ptr<swapchain_image_impl> acquire_next_image();

private:
    device device_;
    vk::SwapchainKHR swapchain_;
    std::vector<std::shared_ptr<swapchain_image_impl>> swapchain_images_;
};

}  // namespace graal::detail