#pragma once
#include <memory>
#include <span>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace graal {

namespace detail {

class device_impl {
public:
  device_impl(std::span<const char *const> required_instance_extensions);
  ~device_impl();

  [[nodiscard]] vk::Instance get_vk_instance() const noexcept {
    return instance_;
  }
  [[nodiscard]] vk::Device get_vk_device() const noexcept { return device_; }
  [[nodiscard]] vk::PhysicalDevice get_vk_physical_device() const noexcept {
    return physical_device_;
  }

  [[nodiscard]] VmaAllocator get_allocator() const noexcept {
    return allocator_;
  }

private:
  vk::Instance                 instance_;
  vk::PhysicalDevice           physical_device_;
  vk::PhysicalDeviceProperties physical_device_properties_;
  vk::PhysicalDeviceFeatures   physical_device_features_;
  vk::Device                   device_;
  size_t                       graphics_queue_family_;
  size_t                       compute_queue_family_;
  size_t                       transfer_queue_family_;
  vk::Queue                    graphics_queue_;
  vk::Queue                    compute_queue_;
  vk::Queue                    transfer_queue_;
  VmaAllocator                 allocator_;
};

using device_impl_ptr = std::shared_ptr<device_impl>;

} // namespace detail

/// @brief Vulkan instance and device
class device {
    friend class queue;
public:
  using impl_t = std::shared_ptr<detail::device_impl>;

  device(std::span<const char *const> required_instance_extensions)
      : impl_{std::make_shared<detail::device_impl>(
            required_instance_extensions)} {}

  [[nodiscard]] vk::Instance get_vk_instance() const noexcept {
    return impl_->get_vk_instance();
  }
  [[nodiscard]] vk::Device get_vk_device() const noexcept {
    return impl_->get_vk_device();
  }
  [[nodiscard]] vk::PhysicalDevice get_vk_physical_device() const noexcept {
    return impl_->get_vk_physical_device();
  }
  [[nodiscard]] VmaAllocator get_allocator() const noexcept {
    return impl_->get_allocator();
  }

private:
  impl_t impl_;
};

} // namespace graal