#pragma once
#include <graal/detail/recycler.hpp>
#include <graal/detail/vk_handle.hpp>
#include <vulkan/vulkan.hpp>

namespace graal::detail {
class command_buffer_pool {
public:
    /// @brief
    /// @param vk_device
    /// @param queue_family
    command_buffer_pool(vk::Device vk_device, uint32_t queue_family) : queue_family_{queue_family} 
    {
        vk::CommandPoolCreateInfo create_info{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                .queueFamilyIndex = queue_family_};
        command_pool_ = handle{vk_device.createCommandPool(create_info)};
    }

    ///
    [[nodiscard]] uint32_t queue_family() const noexcept {
        return queue_family_;
    }

    /// @brief
    /// @param vk_device
    void reset(vk::Device vk_device) {
        vk_device.resetCommandPool(command_pool_.get(), {});
        while (!used_.empty()) {
            free_.push_back(used_.pop_back());
        }
    }

    ///
    [[nodiscard]] vk::CommandBuffer fetch_command_buffer(vk::Device vk_device) {
        handle<vk::CommandBuffer> cb;

        if (!free_.empty()) {
            cb = free_.pop_back();
        } else {
            vk::CommandBufferAllocateInfo alloc_info{.commandPool = command_pool_.get(),
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1};
            auto cmdbufs = vk_device.allocateCommandBuffers(alloc_info);
            cb = handle{cmdbufs[0]};
        }

        return used_.push_back(std::move(cb));
    }

    /// @brief
    /// @param vk_device
    void destroy(vk::Device vk_device) {
        auto list = free_.release_all();
        vk_device.freeCommandBuffers(
                command_pool_.get(), static_cast<uint32_t>(list.size()), list.data());
        vk_device.destroyCommandPool(command_pool_.release());
    }

private:
    uint32_t queue_family_;
    handle<vk::CommandPool> command_pool_;
    handle_vector<vk::CommandBuffer> free_;
    handle_vector<vk::CommandBuffer> used_;
};
}  // namespace graal::detail