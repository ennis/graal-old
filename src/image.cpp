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

image_resource::image_resource(device dev, image_type type, image_usage usage, image_format format,
        range<3> size, const image_properties& props) :
    resource{std::move(dev), resource_type::image},
    owned_{true} {
    const auto vkd = get_device().get_handle();
    image_ = create_image(vkd, type, usage, format, size, props);
    mem_preferred_flags = props.preferred_flags;
    mem_required_flags = props.required_flags;
}

image_resource::image_resource(
    device dev,
    handle<vk::Image> image,
    handle<VmaAllocation> allocation,
    const VmaAllocationInfo& alloc_info) : 
    resource{ std::move(dev), resource_type::image },
    owned_{ false },
    image_{image.release()},
    allocation_{allocation.release()},
    allocation_info_{alloc_info}
{}

image_resource::~image_resource() {
    if (owned_ && image_) {
        const auto vkd = get_device().get_handle();
        vkd.waitIdle();
        vkd.destroyImage(image_);
    }
}

/// @brief See resource::bind_memory
void image_resource::bind_memory(
        VmaAllocation allocation, const VmaAllocationInfo& allocation_info) {
    allocation_ = allocation;
    allocation_info_ = allocation_info;
    const auto vkd = get_device().get_handle();
    vkBindImageMemory(vkd, image_, allocation_info.deviceMemory, allocation_info.offset);
}

allocation_requirements image_resource::get_allocation_requirements() {
    const auto vkd = get_device().get_handle();
    const vk::MemoryRequirements mem_req = vkd.getImageMemoryRequirements(image_);
    return allocation_requirements{.memreq = mem_req,
            .required_flags = mem_required_flags,
            .preferred_flags = mem_preferred_flags};
}

void image_resource::set_name(std::string name) {
    const auto vkd = get_device().get_handle();
    vk::DebugUtilsObjectNameInfoEXT object_name_info{
         .objectType = vk::ObjectType::eImage,
         .objectHandle = (uint64_t)(VkImage)image_,
         .pObjectName = name.c_str(),
    };
    vkd.setDebugUtilsObjectNameEXT(object_name_info, vk_default_dynamic_loader);
    resource::set_name(name);
}

}  // namespace detail
}  // namespace graal