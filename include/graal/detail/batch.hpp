#pragma once
#include <graal/detail/recycler.hpp>
#include <graal/detail/resource.hpp>
#include <graal/access_mode.hpp>
#include <graal/buffer_usage.hpp>
#include <graal/image_usage.hpp>
#include <graal/detail/task.hpp>
#include <graal/detail/swapchain_impl.hpp>

#include <vector>
#include <vulkan/vulkan.hpp>

namespace graal::detail {



struct recycled_batch_resources {
    recycler<vk::Semaphore> semaphores;
};

struct batch_resource_access {
    detail::resource_ptr resource;
    access_mode mode;
    union {
        buffer_usage buffer;
        image_usage image;
    } usage_flags;
};

/// @brief Represents a set of interdependent tasks (forming a DAG) and resources.
/// Batches automatically memory allocation of resources (discovering memory aliasing opportunities)
/// scheduling of tasks, and synchronization.
///
/// A batch can be in one of three states:
/// - building: the task DAG is being built
/// - submittable: the task DAG has been built and resources have been allocated
/// - pending: command buffers have been built and submitted
/// - retired: the batch has finished execution, and its resources can be recycled
///
///
class batch {
    friend class queue_impl;
public:
    bool is_empty() const noexcept {
        return tasks.empty();
    }

    /// @brief Adds a task.
    /// @param task
    /// This is noexcept because we assume that the resources referenced within the task have
    /// already been modified to track their last use: throwing an exception here would leave the
    /// resources in an invalid state.
    void add_task(std::unique_ptr<task> task) noexcept;


    /// @brief Adds a managed temporary
    /// @param resource
    /// @return
    size_t add_temporary(resource_ptr resource);

    /// @brief Schedules and submit the batch to the device, and signals the provided timeline semaphore
    void submit(vk::Semaphore timeline, recycled_batch_resources& recycled_resources);

    /// @brief Waits for the batch to finish.
    /// @param timeline_semaphore
    /// NOTE due to limitations of present operations concerning timeline semaphore signalling,
    /// this will only wait for all submit-type operations to finish, but **not** present operations.
    /// This means that present operations may still be in progress when wait returns. However,
    /// this does not prevent the caller to recycle the batch resources, since present operations only

    void wait(vk::Semaphore timeline);


    const task* get_task(uint64_t sequence_number) const noexcept {
        if (start_sn < sequence_number) {
            const auto i = sequence_number - start_sn - 1;
            if (i < tasks.size()) { return tasks[i].get(); }
        }
        return nullptr;
    }

    task* get_task(uint64_t sequence_number) {
        return const_cast<task*>(static_cast<const batch&>(*this).get_task(sequence_number));
    }

private:
    batch(device_impl_ptr device, serial_number start_sn);


    device_impl_ptr device_;
    /// @brief starting sequence number of the batch
    /// (base for submissions)
    serial_number start_sn = 0;
    /// @brief all resources referenced in the batch (called "temporaries" for historical reasons)
    std::vector<resource_ptr> temporaries;
    /// @brief tasks
    std::vector<std::unique_ptr<task>> tasks;
    /// @brief Per-thread resources
    /// @brief Semaphores used within the batch
    std::vector<vk::Semaphore> semaphores;
};

}  // namespace graal::detail