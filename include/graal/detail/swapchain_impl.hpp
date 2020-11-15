#pragma once
#include <graal/detail/recycler.hpp>
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/range.hpp>

#include <memory>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

class swapchain_image_impl : public resource {
    friend class swapchain_impl;

public:
    /// @brief Returns the semaphore signalling that the swapchain image is available for use. The ownership of the semaphore is transferred
    /// to the caller and subsequent calls will return nullptr (until the image is handed back to the presentation engine and acquired again).
    /// @return
    vk::Semaphore consume_semaphore(vk::Semaphore new_semaphore);

private:
    swapchain_image_impl(vk::Image image) :
        resource{resource_type::swapchain_image}, image_{image} {
    }

    vk::Image image_;
    vk::Semaphore semaphore_;  // swapchain images use binary semaphores for synchronization because presentation doesn't support timelines
    // TODO use a unique_handle pattern to signal ownership. Not the one provided by vulkan-hpp though because
    // it bundles pointers to the device and allocation infos to allow ad-hoc deletion.

    //std::shared_ptr<detail::swapchain_impl> swapchain_;
};

class swapchain_impl {
public:
    swapchain_impl(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface);

    ~swapchain_impl();

    void resize(range_2d framebuffer_size, vk::SurfaceKHR surface);

    std::shared_ptr<swapchain_image_impl> acquire_next_image(
            recycler<vk::Semaphore> semaphore_recycler);

private:
    device device_;
    vk::SwapchainKHR swapchain_;
    std::vector<std::shared_ptr<swapchain_image_impl>> swapchain_images_;
};

}  // namespace graal::detail