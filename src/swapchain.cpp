#include <graal/detail/resource.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/swapchain.hpp>
#include <vector>

namespace graal {

swapchain::swapchain(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface) :
    impl_{std::make_shared<detail::swapchain_impl>(device, framebuffer_size, surface)} {
}

vk::SwapchainKHR swapchain::vk_swapchain() const noexcept {
    return impl_->vk_swapchain();
}

void swapchain::resize(range_2d framebuffer_size, vk::SurfaceKHR surface) {
    impl_->resize(framebuffer_size, surface);
}

vk::Image swapchain::current_image() const noexcept {
    return impl_->current_image();
}

uint32_t swapchain::current_image_index() const noexcept {
    return impl_->current_image_index();
}

}  // namespace graal