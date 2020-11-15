#include <graal/detail/resource.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/swapchain.hpp>
#include <vector>

namespace graal {
namespace detail {}  // namespace detail

swapchain::swapchain(device& device, range_2d framebuffer_size, vk::SurfaceKHR surface) :
    impl_{std::make_shared<detail::swapchain_impl>(device, framebuffer_size, surface)} {
}

swapchain_image swapchain::acquire_next_image() {
    return swapchain_image{ impl_->acquire_next_image() };
}

void swapchain::resize(range_2d framebuffer_size, vk::SurfaceKHR surface) {
    impl_->resize(framebuffer_size, surface);
}

}  // namespace graal