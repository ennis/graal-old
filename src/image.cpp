#include <graal/device.hpp>
#include <graal/image.hpp>

namespace graal {
namespace detail {

/*vk::Image
create_vk_image_with_allocation(device                     device,
                                const vk::ImageCreateInfo &create_info) {
  const auto vk_device = device.get_vk_device();
  const auto image = vk_device.createImage(create_info);
  const auto mem_req = vk_device.getImageMemoryRequirements(image);

  const auto              allocator = device.get_allocator();
  VmaAllocationCreateInfo allocation_create_info = {};
  // allocation_create_info.usage = ;

  // vmaCreateImage(allocator, &create_info, )
}*/

/// @brief See resource::bind_memory
void image_impl::bind_memory(device_impl_ptr device, VmaAllocation allocation,
                             VmaAllocationInfo allocation_info) {
  auto vk_image = get_unbound_vk_image(device);
  auto vk_device = device->get_vk_device();
  allocation_ = allocation;
  allocation_info_ = allocation_info;
  vk_device.bindImageMemory(image_, allocation_info.deviceMemory,
                            allocation_info.offset);
  allocated = true;
}

allocation_requirements
image_impl::get_allocation_requirements(device_impl_ptr dev) {

  auto image = get_unbound_vk_image(dev);

  return allocation_requirements{
      // TODO
  };
}

// Creates a VkImage without bound memory
vk::Image image_impl::get_unbound_vk_image(device_impl_ptr dev) {

  if (image_) {
    // resource already created
    return image_;
  }

  // resource not created, create now
  const auto format = get_vk_format(this->format());
  const auto extent = to_vk_extent_3d(size());
  const auto image_type = get_vk_image_type(type_);

  const unsigned mip_levels = mipmaps_.value_or(1);
  const unsigned samples = samples_.value_or(1);
  const unsigned array_layers = array_layers_.value_or(1);

  vk::ImageCreateInfo create_info{
      .imageType = image_type,
      .format = format,
      .extent = extent,
      .mipLevels = mip_levels,
      .arrayLayers = array_layers,
      .samples = get_vk_sample_count(samples),
      .usage = usage_,
  };

  auto vk_device = dev->get_vk_device();
  image_ = vk_device.createImage(create_info);
  device_impl_ = std::move(dev);
  return image_;
}
} // namespace detail
} // namespace graal