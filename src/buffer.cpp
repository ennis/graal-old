#include <graal/buffer.hpp>

namespace graal::detail {
void buffer_impl::bind_memory(device_impl_ptr dev, VmaAllocation allocation,
                              VmaAllocationInfo allocation_info) {
  const auto vk_buffer = get_unbound_vk_buffer(dev);
  const auto vk_device = dev->get_vk_device();
  allocation_ = allocation;
  allocation_info_ = allocation_info;
  vk_device.bindBufferMemory(vk_buffer, allocation_info.deviceMemory,
                             allocation_info.offset);
  allocated = true;
}

allocation_requirements
buffer_impl::get_allocation_requirements(device_impl_ptr dev) {
  const auto                   vk_device = dev->get_vk_device();
  const auto                   vk_buffer = get_unbound_vk_buffer(dev);
  const vk::MemoryRequirements mem_req =
      vk_device.getBufferMemoryRequirements(vk_buffer);
  return allocation_requirements{.flags = allocation_flags_,
                                 .memory_type_bits = mem_req.memoryTypeBits,
                                 .size = mem_req.size,
                                 .alignment = mem_req.alignment};
}

vk::Buffer buffer_impl::get_unbound_vk_buffer(device_impl_ptr dev) {

  if (buffer_) {
    return buffer_;
  }

  vk::BufferCreateInfo create_info{.size = byte_size_,
                                   .usage = usage_,
                                   .sharingMode =
                                       vk::SharingMode::eConcurrent, // TODO
                                   .queueFamilyIndexCount = 0,
                                   .pQueueFamilyIndices = nullptr};

  auto vk_device = dev->get_vk_device();
  buffer_ = vk_device.createBuffer(create_info);
  return buffer_;
}
} // namespace graal::detail