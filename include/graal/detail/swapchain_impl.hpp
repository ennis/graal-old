#pragma once
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/range.hpp>
#include <graal/image_format.hpp>

#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

class swapchain_impl;
class swapchain_image_impl;

//-----------------------------------------------------------------------------
class swapchain_impl {
    friend class queue_impl;

public:
    swapchain_impl(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface);

    ~swapchain_impl();

    void resize(range_2d framebuffer_size, vk::SurfaceKHR surface);

    [[nodiscard]] vk::SwapchainKHR get_vk_swapchain() const noexcept {
        return swapchain_;
    }

    [[nodiscard]] image_format format() const noexcept {
        return format_;
    }

    [[nodiscard]] static std::shared_ptr<swapchain_image_impl> acquire_next_image(
            std::shared_ptr<swapchain_impl> impl);

private:
    device device_;
    vk::SwapchainKHR swapchain_;
    image_format format_;
    std::vector<vk::Image> images_;
};

//-----------------------------------------------------------------------------
class swapchain_image_impl : public image_resource {
    friend class swapchain_impl;

public:
    swapchain_image_impl(
            std::shared_ptr<detail::swapchain_impl> swapchain, vk::Image image, uint32_t index) :
        image_resource{resource_type::swapchain_image, image, swapchain->format()},
        swapchain_{std::move(swapchain)}, index_{index} 
    {
    }

    [[nodiscard]] uint32_t index() const noexcept {
        return index_;
    }

    [[nodiscard]] vk::Image get_vk_image() const noexcept {
        return image_;
    }

    [[nodiscard]] vk::SwapchainKHR get_vk_swapchain() const noexcept {
        return swapchain_->get_vk_swapchain();
    }

private:
    std::shared_ptr<detail::swapchain_impl> swapchain_;
    uint32_t index_;
};

}  // namespace graal::detail