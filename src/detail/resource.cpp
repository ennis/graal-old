#include <graal/buffer.hpp>
#include <graal/detail/resource.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/image.hpp>

namespace graal::detail {

image_resource& resource::as_image() {
    assert(type_ == resource_type::image || type_ == resource_type::swapchain_image);
    return static_cast<image_resource&>(*this);
}

buffer_resource& resource::as_buffer() {
    assert(type_ == resource_type::buffer);
    return static_cast<buffer_resource&>(*this);
}

virtual_resource& resource::as_virtual_resource() {
    assert(type_ == resource_type::buffer || type_ == resource_type::image);

    if (type_ == resource_type::buffer) {
        return static_cast<virtual_resource&>(static_cast<buffer_impl&>(*this));
    } else {
        return static_cast<virtual_resource&>(static_cast<image_impl&>(*this));
    }
}

void resource::realize(device_impl_ptr device) {
    if (type_ == resource_type::image) {
        (void) static_cast<image_impl*>(this)->get_vk_image(std::move(device));
    } else if (type_ == resource_type::buffer) {
        (void) static_cast<buffer_impl*>(this)->get_vk_buffer(std::move(device));
    }
}

}  // namespace graal::detail