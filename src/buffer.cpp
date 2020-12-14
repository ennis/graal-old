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

/*void create_buffer_with_allocation(vk::Device vk_device, VmaAllocator allocator,
        buffer_usage& usage, std::size_t byte_size, const buffer_properties& props,
        //std::span<std::byte> initial_bytes,
        vk::Buffer& out_buffer, VmaAllocation& out_alloc, VmaAllocationInfo& out_alloc_info) 
{
    vk::BufferCreateInfo create_info{.size = byte_size,
            .usage = static_cast<vk::BufferUsageFlags>(static_cast<int>(usage)),
            .sharingMode = vk::SharingMode::eConcurrent,  // TODO
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};

    VmaAllocationCreateInfo alloc_create_info{.flags = 0,
            .usage = VMA_MEMORY_USAGE_UNKNOWN,
            .requiredFlags = static_cast<VkMemoryPropertyFlags>(props.required_flags),
            .preferredFlags = static_cast<VkMemoryPropertyFlags>(props.preferred_flags),
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr};

    if (auto result = vmaCreateBuffer(allocator,
                reinterpret_cast<const VkBufferCreateInfo*>(&create_info), &alloc_create_info,
                reinterpret_cast<VkBuffer*>(&out_buffer), &out_alloc, &out_alloc_info);
            result != VK_SUCCESS) {
        throw std::runtime_error{"vmaCreateBuffer failed"};
    }
}*/

}  // namespace

// construct uninitialized from size
buffer_impl::buffer_impl(device dev, buffer_usage usage, std::size_t byte_size,
        const buffer_properties& properties) :
    buffer_resource{create_buffer(dev.get_vk_device(), usage, byte_size, properties), byte_size},
    device_{std::move(dev)}, usage_{usage}, byte_size_{byte_size} {
}

/*buffer_impl::buffer_impl(device_impl_ptr device, buffer_usage usage, std::size_t byte_size,
        const buffer_properties& properties, std::span<std::byte> initial_bytes) :
    buffer_resource{
            create_buffer(device->get_vk_device(), usage, byte_size, properties), byte_size},
    device_{std::move(device)}, usage_{usage}, byte_size_{byte_size}, props_{properties}
{
    // force memory visible in host
}*/

buffer_impl::~buffer_impl() {
    if (buffer_) { device_.get_vk_device().destroyBuffer(buffer_); }
}

void buffer_impl::bind_memory(
        vk::Device device, VmaAllocation allocation, const VmaAllocationInfo& allocation_info) {
    allocation_ = allocation;
    allocation_info_ = allocation_info;
    device.bindBufferMemory(buffer_, allocation_info.deviceMemory, allocation_info.offset);
    allocated = true;
}

allocation_requirements buffer_impl::get_allocation_requirements(vk::Device device) {
    const vk::MemoryRequirements mem_req = device.getBufferMemoryRequirements(buffer_);
    return allocation_requirements{.memreq = mem_req,
            .required_flags = props_.required_flags,
            .preferred_flags = props_.preferred_flags};
}

}  // namespace graal::detail