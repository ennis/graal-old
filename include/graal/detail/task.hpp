#pragma once
#include <graal/detail/sequence_number.hpp>
#include <graal/detail/swapchain_impl.hpp>

#include <cstddef>
#include <functional>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

/// @brief Index of the temporary entry in the current batch.
using temporary_index = size_t;
constexpr inline temporary_index invalid_temporary_index = (temporary_index) -1;

/// @brief Index of a task in a batch.
using task_index = std::size_t;
inline constexpr task_index invalid_task_index = (std::size_t) -1;

/// @brief Task callback
using submit_callback_fn = void(vk::CommandBuffer);

/// @brief Maximum number of queues for the application
constexpr inline size_t max_queues = 4;

/// @brief Represents a resource access in a task.
struct resource_access_details {
    vk::ImageLayout layout;  // only valid if resource is an image
    vk::AccessFlags access_flags;
    vk::PipelineStageFlags input_stage; // stage that reads the resource (can be empty for write-only)
    vk::PipelineStageFlags output_stage; // stage that writes to the resource (can be empty for read-only)
};

struct task {
    struct resource_access {
        size_t index;
        resource_access_details details;
    };

    std::string name;
    serial_number serial;
    std::vector<uint64_t> preds;
    std::vector<uint64_t> succs;
    uint64_t waits[max_queues] = {};

    struct {
        uint64_t enabled : 1 =
                0;  // whether we should signal the task sequence number on the timeline
        uint64_t queue : 2 = 0;  // the queue of the timeline to signal
        uint64_t serial : 61 = 0;  // task sequence number
    } signal;

    std::vector<vk::Semaphore> wait_binary;  // TODO unique_handle
    std::vector<vk::Semaphore> signal_binary;
    std::vector<resource_access> accesses;
};

}  // namespace graal::detail