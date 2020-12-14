#pragma once
#include <graal/detail/recycler.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

struct staging_buffer {
    void* try_allocate(
            size_t align, size_t size, vk::Buffer& out_buffer, vk::DeviceSize& out_offset);

    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocation_info;
    vk::Buffer buffer = nullptr;
    void* base = nullptr;
    void* ptr = nullptr;
    size_t space = 0;
};

/// @brief A pool of staging buffers.
class staging_pool {
public:
    ~staging_pool();

    void* get_staging_buffer(vk::Device device, VmaAllocator allocator,
            recycler<staging_buffer>& buffers, size_t align, size_t size, vk::Buffer& out_buffer,
            vk::DeviceSize& out_offset);

    void recycle(recycler<staging_buffer>& buffers);

private:
    std::vector<staging_buffer> buffers_;
};

}  // namespace graal::detail