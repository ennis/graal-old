#pragma once
#include <graal/detail/sequence_number.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/queue_class.hpp>

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

/// @brief Represents a resource access in a task.
struct resource_access_details {
    /// @brief The layout into which the image must be before beginning the task.
    vk::ImageLayout layout;  
    /// @brief How the task is going to access the resource.
    vk::AccessFlags access_flags;
    vk::PipelineStageFlags input_stage;
    vk::PipelineStageFlags output_stage;
};

enum class task_type {
    render_pass,
    compute_pass,
    transfer,
    present
};

struct task {
    struct resource_access {
        size_t index;
        resource_access_details details;
    };

    std::string name;
    task_type type = task_type::render_pass;

    /// @brief The submission number (SNN).
    /// NOTE a first serial is assigned when the task is created, without a queue, for the purposes of 
    /// DAG building. However, the serial might change after submission, due to task reordering.
    submission_number snn;

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