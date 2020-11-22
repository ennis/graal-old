#include <graal/detail/resource.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/swapchain.hpp>
#include <vector>

namespace graal {

uint32_t swapchain_image::index() const noexcept {
    return impl_->index();
}

vk::SwapchainKHR swapchain_image::get_vk_swapchain() const noexcept {
    return impl_->get_vk_swapchain();
}

swapchain::swapchain(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface) :
    impl_{std::make_shared<detail::swapchain_impl>(device, framebuffer_size, surface)} {
}

swapchain_image swapchain::acquire_next_image() {
    return swapchain_image{detail::swapchain_impl::acquire_next_image(impl_)};
}

void swapchain::resize(range_2d framebuffer_size, vk::SurfaceKHR surface) {
    impl_->resize(framebuffer_size, surface);
}

}  // namespace graal