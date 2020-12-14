#include <graal/device.hpp>
#include <graal/image.hpp>

namespace graal {
namespace detail {

namespace {
vk::Image create_image(vk::Device vk_device, image_type type, image_usage usage,
        image_format format, range<3> size, const image_properties& props) {
    const auto vk_format = get_vk_format(format);
    const auto vk_extent = to_vk_extent_3d(size);
    const auto vk_image_type = get_vk_image_type(type);

    vk::ImageCreateInfo create_info{
            .imageType = vk_image_type,
            .format = vk_format,
            .extent = vk_extent,
            .mipLevels = props.mip_levels,
            .arrayLayers = props.array_layers,
            .samples = get_vk_sample_count(props.samples),
            .usage = static_cast<vk::ImageUsageFlags>(static_cast<int>(usage)),
    };

    const auto image = vk_device.createImage(create_info);
    return image;
}

}  // namespace

image_impl::image_impl(device dev, image_type type, image_usage usage,
        image_format format, range<3> size, const image_properties& props) :
    image_resource{resource_type::image,
            create_image(dev.get_vk_device(), type, usage, format, size, props), format},
    device_{std::move(dev)}, type_{type}, usage_{usage}, size_{size}, props_{props} {
}

image_impl::~image_impl() {
    if (image_) { device_.get_vk_device().destroyImage(image_); }
}

/// @brief See resource::bind_memory
void image_impl::bind_memory(vk::Device device, VmaAllocation allocation,
        const VmaAllocationInfo& allocation_info)
{
    allocation_ = allocation;
    allocation_info_ = allocation_info;
    device.bindImageMemory(image_, allocation_info.deviceMemory, allocation_info.offset);
    allocated = true;
}

allocation_requirements image_impl::get_allocation_requirements(vk::Device device) {
    const vk::MemoryRequirements mem_req = device.getImageMemoryRequirements(image_);
    return allocation_requirements{.memreq = mem_req,
            .required_flags = props_.required_flags,
            .preferred_flags = props_.preferred_flags};
}

}  // namespace detail
}  // namespace graal