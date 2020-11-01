#pragma once
#include <graal/detail/task.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/device.hpp>
#include <graal/flags.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <memory>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

/// @brief Flags common to all resource types.
enum class allocation_flag {
  host_visible = (1 << 0), ///< Resource can be mapped in host memory.
  aliasable =
      (1
       << 1), ///< The memory of the resource can be aliased to other resources.
};
} // namespace graal::detail

GRAAL_ALLOW_FLAGS_FOR_ENUM(graal::detail::allocation_flag)

namespace graal::detail {


using allocation_flags = flags<allocation_flag>;

/// @brief
struct allocation_requirements {
  allocation_flags flags;
  uint32_t         memory_type_bits;
  size_t           size;
  size_t           alignment;
};

/// @brief Base class for resources (buffer and images)
class resource : public named_object {
  friend class queue_impl;

public:
  /// @brief Returns the memory requirements of the resource.
  virtual allocation_requirements
  get_allocation_requirements(device_impl_ptr dev) = 0;

  /// @brief
  /// @param other
  virtual void bind_memory(device_impl_ptr dev, VmaAllocation allocation,
                           VmaAllocationInfo allocation_info) = 0;

  bool allocated = false;   // set to true once bind_memory has been called successfully
  bool discarded =
      false; // no more external references to this resource via user handles
  temporary_index tmp_index =
      invalid_temporary_index; // assigned temporary index
  batch_index batch =
      invalid_batch_index; // the last batch in which the resource was used
  task_index producer =
      invalid_task_index; // the last producer (last task that wrote to this
                          // resource in the batch)
};

using resource_ptr = std::shared_ptr<resource>;
} // namespace graal::detail