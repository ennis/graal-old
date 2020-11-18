#pragma once
#include <graal/detail/recycler.hpp>

#include <vulkan/vulkan.hpp>

namespace graal::detail {
struct command_buffer_pool {
    vk::CommandPool command_pool;
    recycler<vk::CommandBuffer> free_cbs;
    std::vector<vk::CommandBuffer> used_cbs;

    void reset(vk::Device vk_device) {
        vk_device.resetCommandPool(command_pool, {});
        free_cbs.recycle_vector(std::move(used_cbs));
    }

    vk::CommandBuffer fetch_command_buffer(vk::Device vk_device) {
        vk::CommandBuffer cb;

        if (!free_cbs.fetch(cb)) {
            vk::CommandBufferAllocateInfo alloc_info{.commandPool = command_pool,
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1};
            auto cmdbufs = vk_device.allocateCommandBuffers(alloc_info);
            cb = cmdbufs[0];
        }

        used_cbs.push_back(cb);
        return cb;
    }

    void destroy(vk::Device vk_device) {
        const auto list = free_cbs.free_list();
        vk_device.freeCommandBuffers(command_pool, static_cast<uint32_t>(list.size()), list.data());
        free_cbs.clear();
    }
};
}  // namespace graal::detail