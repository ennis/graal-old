#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <vulkan/vulkan.hpp>

namespace graal::detail {

/// @brief Sequence numbers uniquely identify a task.
using sequence_number = uint64_t;

/// @brief Index of the temporary entry in the current batch.
using temporary_index = size_t;
constexpr inline temporary_index invalid_temporary_index = (temporary_index)-1;

/// @brief Number that encodes a wait operation on a task.
/// To synchronize tasks, their sequence numbers are signalled on timeline semaphores.
/// There is one timeline per queue. A task is scheduled to run on a particular queue.
/// The sync number encodes both the sequence number to wait for, and on which timeline (queue)
/// the corresponding task has been submitted, and thus where the sync number should be awaited.
struct sync_number {
    /// @brief
    uint64_t timeline : 2 = 0;
    uint64_t sequence_number : 62 = 0;
};

/// @brief Index of a task in a batch.
using task_index = std::size_t;
inline constexpr task_index invalid_task_index = (std::size_t) - 1;

/// @brief Task callback
using submit_callback_fn = void(vk::CommandBuffer);
using present_callback_fn = void(vk::Queue);

enum class task_type {
    submit,       // submit_task
    present         // present_task
};

/// @brief Maximum number of queues for the application
constexpr inline size_t max_queues = 4;

struct task {
    void add_read(uint64_t r);
    void add_write(uint64_t w);

    bool needs_synchronization() const noexcept {
        return !wait_binary.empty() || !signal_binary.empty() || signal.enabled || std::any_of(waits, waits + max_queues, [](uint64_t x) { return x > 0; });
    }

    task_type type = task_type::submit;
    std::string name;
       
    std::vector<temporary_index> reads;
    std::vector<temporary_index> writes;
    std::vector<uint64_t> preds;
    std::vector<uint64_t> succs;

    uint64_t waits[max_queues] = {};

    struct {
        uint64_t enabled : 1 = 0;           // whether we should signal the task sequence number on the timeline
        uint64_t queue : 2 = 0;             // the queue of the timeline to signal
        uint64_t sequence_number : 61 = 0;  // task sequence number
    } signal;

    // binary semaphores to wait for (owned)
    std::vector<vk::Semaphore> wait_binary; // TODO unique_handle
    // binary semaphores to signal (non-owned)
    std::vector<vk::Semaphore> signal_binary;
};

struct submit_task final : task {
    std::vector<std::function<present_callback_fn>> callbacks;
};

struct present_task final : task {
    std::vector<std::function<present_callback_fn>> callbacks;
};

}  // namespace graal::detail