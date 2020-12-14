#include <graal/detail/staging_pool.hpp>
#include <memory>

namespace graal::detail {

    void* staging_buffer::try_allocate(size_t align, size_t size, vk::Buffer& out_buffer, vk::DeviceSize& out_offset) {
        if (auto p = std::align(align, size, ptr, space)) {
            // enough space
            out_buffer = buffer;
            out_offset = static_cast<std::byte*>(ptr) - static_cast<std::byte*>(base);
            return p;
        }
        return nullptr;
    }

    void* staging_pool::get_staging_buffer(
        vk::Device device,
        VmaAllocator allocator,
        recycler<staging_buffer>& buffers,
        size_t align,
        size_t size,
        vk::Buffer& out_buffer,
        vk::DeviceSize& out_offset)
    {
        // look for a buffer with enough free space
        for (auto& b : buffers_) {
            if (auto ptr = b.try_allocate(align, size, out_buffer, out_offset)) {
                return ptr;
            }
            // not enough space, continue
        }

        // not enough space in any buffer, allocate a new buffer
        staging_buffer sb;
        if (!buffers.fetch_if(sb, [&](const staging_buffer& sb) { return sb.space >= align + size; })) {
            // allocate new
            constexpr size_t min_staging_buffer_block_size = 1024 * 1024;
            const size_t alloc_size = std::max(min_staging_buffer_block_size, align + size);

            const VmaAllocationCreateInfo alloc_create_info{
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = VMA_MEMORY_USAGE_CPU_ONLY,
                .requiredFlags = 0,
                .preferredFlags = 0,
                .memoryTypeBits = 0,
                .pool = VK_NULL_HANDLE,
                .pUserData = nullptr
            };

            const vk::BufferCreateInfo buffer_create_info{
                .size = alloc_size,
                .usage = vk::BufferUsageFlagBits::eTransferSrc,
                .sharingMode = vk::SharingMode::eConcurrent
            };

            VkBuffer vk_buffer;
            if (auto result = vmaCreateBuffer(
                allocator,
                &static_cast<const VkBufferCreateInfo&>(buffer_create_info),
                &alloc_create_info,
                &vk_buffer,
                &sb.allocation,
                &sb.allocation_info); result != VK_SUCCESS)
            {
                throw std::runtime_error{ "failed to allocate staging buffer" };
            }
            sb.buffer = vk_buffer;

            // map the buffer
            sb.base = device.mapMemory(
                sb.allocation_info.deviceMemory,
                sb.allocation_info.offset,
                sb.allocation_info.size);
            sb.ptr = sb.base;
            sb.space = sb.allocation_info.size;
        }

        auto ptr = sb.try_allocate(align, size, out_buffer, out_offset);
        buffers_.push_back(std::move(sb));
        return ptr;
    }

    void staging_pool::recycle(recycler<staging_buffer>& buffers) {
        buffers.recycle_vector(std::move(buffers_));
    }

    staging_pool::~staging_pool() {
        assert(buffers_.empty());
    }
}