#include <graal/detail/swapchain_impl.hpp>

#include <utility>

namespace graal::detail {

namespace {

vk::SurfaceFormatKHR get_preferred_swapchain_surface_format(
        std::span<const vk::SurfaceFormat2KHR> surface_formats) {
    for (const auto& sfmt : surface_formats) {
        if (sfmt.surfaceFormat.format == vk::Format::eB8G8R8A8Srgb
                && sfmt.surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return sfmt.surfaceFormat;
        }
    }
    throw std::runtime_error{"no suitable surface format available"};
}

vk::PresentModeKHR get_preferred_present_mode(std::span<const vk::PresentModeKHR> present_modes) {
    if (auto it = std::find(
                present_modes.begin(), present_modes.end(), vk::PresentModeKHR::eMailbox);
            it != present_modes.end()) {
        return vk::PresentModeKHR::eMailbox;
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D get_preferred_swap_extent(
        range_2d framebuffer_size, const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        vk::Extent2D extent = {static_cast<uint32_t>(framebuffer_size[0]),
                static_cast<uint32_t>(framebuffer_size[1])};

        extent.width = std::clamp(
                extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);

        return extent;
    }
}

}  // namespace

// using a swapchain image will be like this:
// - 0. obtain next image from swapchain (Signal semaphore, signal fence)
// - 0.1. wait for fence on returned image to ensure present finished and semaphore signalled
// - 1. enqueue wait on presentation finished semaphore (always done after acquiring an image from the swapchain)
// - 2. enqueue uses of swapchain image
// - 3. enqueue wait on render finished
// - 4. enqueue present, signal fence

// Problem: getting a semaphore to signal for vkAcquireNextImageKHR
// - ideally, re-use semaphore that's already signalled
// - problem: we can acquire images as fast as possible, all vkAcquireNextImageKHR calls need different semaphores
//   (no guarantee that any of the previous semaphores were signalled)
//      - can't use a semaphore

// - obtain a cleared semaphore (#1), use it for vkAcquireNextImageKHR
// - wait for fence associated with the newly acquired image
//    - then, reclaim previous semaphore (#0) associated to the image, assign the new one (#1)
//    - problem: the previous semaphore may be in a signaled state, because we may call vkAcquireNextImageKHR,
//      and never use the acquired image (we will not wait on the semaphore and it will be left in a signalled state!)
//

//-----------------------------------------------------------------------------
swapchain_impl::swapchain_impl(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface) :
    device_{device} {
    resize(framebuffer_size, surface);
}

swapchain_impl::~swapchain_impl() {
    // XXX destroy swapchain
    const auto vk_device = device_.get_vk_device();
    vk_device.destroySwapchainKHR(swapchain_);
}

vk::Semaphore swapchain_image_impl::consume_semaphore(vk::Semaphore semaphore) {
    return std::exchange(semaphore_, semaphore);
}

//-----------------------------------------------------------------------------
void swapchain_impl::resize(range_2d framebuffer_size, vk::SurfaceKHR surface) {
    const auto phy = device_.get_vk_physical_device();
    const auto vk_device = device_.get_vk_device();

    vk::PhysicalDeviceSurfaceInfo2KHR surface_info{.surface = surface};
    const auto caps = phy.getSurfaceCapabilities2KHR(surface_info);
    const auto formats = phy.getSurfaceFormats2KHR(surface_info);
    const auto present_modes = phy.getSurfacePresentModesKHR(surface);

    const auto swap_format = get_preferred_swapchain_surface_format(formats);
    const auto present_mode = get_preferred_present_mode(present_modes);
    const auto swap_extent = get_preferred_swap_extent(framebuffer_size, caps.surfaceCapabilities);
    const auto max_image_count = caps.surfaceCapabilities.maxImageCount;
    auto image_count = caps.surfaceCapabilities.minImageCount + 1;
    if (max_image_count > 0 && image_count > max_image_count) { image_count = max_image_count; }

    const auto graphics_queue_family = device_.get_graphics_queue_family();
    const vk::SharingMode swapchain_image_sharing_mode = vk::SharingMode::eExclusive;
    const uint32_t share_queue_families[] = {graphics_queue_family};

    const vk::SwapchainCreateInfoKHR create_info{
            .surface = surface,
            .minImageCount = image_count,
            .imageFormat = swap_format.format,
            .imageColorSpace = swap_format.colorSpace,
            .imageExtent = swap_extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = swapchain_image_sharing_mode,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = share_queue_families,
            .preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = present_mode,
            .clipped = true,
            .oldSwapchain = swapchain_ ? swapchain_ : nullptr,
    };

    const auto new_swapchain = vk_device.createSwapchainKHR(create_info);

    if (swapchain_) {
        vk_device.destroySwapchainKHR(swapchain_);
        swapchain_images_.clear();
    }
    swapchain_ = new_swapchain;

    // query swapchain images
    auto swapchain_images = vk_device.getSwapchainImagesKHR(swapchain_);
}

std::shared_ptr<swapchain_image_impl> swapchain_impl::acquire_next_image(
        recycler<vk::Semaphore> semaphore_recycler) {
    auto vk_device = device_.get_vk_device();

    auto [result, image_index] =
            vk_device.acquireNextImageKHR(swapchain_, 1000000000, nullptr, nullptr);

    // Two options:
    // - either the image was never used before
    // - or the image was handed back to the presentation engine (via queue_impl::present and QueuePresent)
    // AcquireNextImage cannot return already acquired images.
    //
    // Since queue_impl::present consumes the image_available_ semaphore if it was not consumed earlier in the frame,
    // we can be sure that image_available_ is consumed.
    assert(!swapchain_images_[image_index]->semaphore_);

    vk::Semaphore image_available;
    if (!semaphore_recycler.fetch(image_available)) {
        vk::SemaphoreCreateInfo sci;
        image_available = vk_device.createSemaphore(sci);
    }
    swapchain_images_[image_index]->semaphore_ = image_available;
    return swapchain_images_[image_index];
}

}  // namespace graal::detail