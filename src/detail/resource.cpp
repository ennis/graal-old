#include <graal/buffer.hpp>
#include <graal/detail/resource.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/image.hpp>

namespace graal::detail {

vk::Image resource::as_vk_image(device_impl_ptr device) {
    if (type_ == resource_type::image) {
        return static_cast<image_impl*>(this)->get_unbound_vk_image(std::move(device));
    } else if (type_ == resource_type::swapchain_image) {
        return static_cast<swapchain_image_impl*>(this)->get_vk_image();
    } else {
        return nullptr;
    }
}

vk::Buffer resource::as_vk_buffer(device_impl_ptr device) {
    if (type_ == resource_type::buffer) {
        return static_cast<buffer_impl*>(this)->get_vk_buffer(std::move(device));
    } else {
        return nullptr;
    }
}

}  // namespace graal::detail