#include <graal/buffer.hpp>

namespace graal::detail {

namespace {
vk::Buffer create_buffer(vk::Device vk_device, buffer_usage usage, std::size_t byte_size,
        const buffer_properties& props) {
    vk::BufferCreateInfo create_info{.size = byte_size,
            .usage = static_cast<vk::BufferUsageFlags>(static_cast<int>(usage)),
            .sharingMode = vk::SharingMode::eConcurrent,  // TODO
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};
    return vk_device.createBuffer(create_info);
}

}  // namespace

// construct uninitialized from size
buffer_resource::buffer_resource(device dev, buffer_usage usage, std::size_t byte_size,
        const buffer_properties& properties) :
    resource{std::move(dev), resource_type::buffer},
    device_{std::move(dev)} 
{
    const auto vkd = get_device().get_handle();
    buffer_ = handle{ create_buffer(vkd, usage, byte_size, properties) };
}

/*buffer_impl::buffer_impl(device_impl_ptr device, buffer_usage usage, std::size_t byte_size,
        const buffer_properties& properties, std::span<std::byte> initial_bytes) :
    buffer_resource{
            create_buffer(device->get_vk_device(), usage, byte_size, properties), byte_size},
    device_{std::move(device)}, usage_{usage}, byte_size_{byte_size}, props_{properties}
{
    // force memory visible in host
}*/

buffer_resource::~buffer_resource() {
    if (buffer_) { device_.get_handle().destroyBuffer(buffer_); }
}

void buffer_resource::bind_memory(
        vk::Device device, VmaAllocation allocation, const VmaAllocationInfo& allocation_info) {
    allocation_ = allocation;
    allocation_info_ = allocation_info;
    device.bindBufferMemory(buffer_, allocation_info.deviceMemory, allocation_info.offset);
    allocated = true;
}

allocation_requirements buffer_resource::get_allocation_requirements(vk::Device device) {
    const vk::MemoryRequirements mem_req = device.getBufferMemoryRequirements(buffer_);
    return allocation_requirements{.memreq = mem_req,
            .required_flags = props_.required_flags,
            .preferred_flags = props_.preferred_flags};
}

void buffer_resource::set_name(std::string name) {
    const auto vk_device = device_.get_handle();
    vk::DebugUtilsObjectNameInfoEXT object_name_info{
            .objectType = vk::ObjectType::eBuffer,
            .objectHandle = (uint64_t)(VkBuffer)buffer_,
            .pObjectName = name.c_str(),
    };
    vk_device.setDebugUtilsObjectNameEXT(object_name_info, vk_default_dynamic_loader);
    resource::set_name(name);
}

}  // namespace graal::detail