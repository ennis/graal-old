#pragma once
#include <graal/detail/recycler.hpp>
#include <graal/detail/resource.hpp>
#include <graal/detail/task.hpp>

#include <vector>
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

/// @brief Per-batch+thread resources
struct batch_thread_resources {
    command_buffer_pool cb_pool;
};

struct recycled_batch_resources {
    recycler<command_buffer_pool> cb_pools;
    recycler<vk::Semaphore> semaphores;
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
        return tasks_.empty();
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

    /// @brief Returns the first sequence number of the batch.
    sequence_number start_sequence_number() const noexcept {
        return start_sn_;
    }

    /// @brief Returns the last sequence number of the batch.
    sequence_number finish_sequence_number() const noexcept {
        return start_sn_ + tasks_.size();
    }

    /// @brief Sequence number of the next task.
    sequence_number next_sequence_number() const noexcept {
        return finish_sequence_number()+1;
    }

    const task* get_task(uint64_t sequence_number) const noexcept {
        if (start_sn_ <= sequence_number) {
            const auto i = sequence_number - start_sn_;
            if (i < tasks_.size()) { return tasks_[i].get(); }
        }
        return nullptr;
    }

    task* get_task(uint64_t sequence_number) {
        return const_cast<task*>(static_cast<const batch&>(*this).get_task(sequence_number));
    }

private:
    batch(device_impl_ptr device, sequence_number start_sn);

    void finalize_task_dag();
    command_buffer_pool get_command_buffer_pool(recycled_batch_resources& recycled_resources);

    /// @brief Returns whether writes to the specified resource in this batch could be seen 
    /// by uses of the resource in subsequent batches.
    /// @return 
    bool writes_are_user_observable(resource& r) const noexcept;

    device_impl_ptr device_;
    /// @brief starting sequence number of the batch
    /// (base for submissions)
    sequence_number start_sn_ = 0;
    /// @brief all resources referenced in the batch (called "temporaries" for historical reasons)
    std::vector<resource_ptr> temporaries_;
    /// @brief tasks
    std::vector<std::unique_ptr<task>> tasks_;
    /// @brief Per-thread resources
    std::vector<batch_thread_resources> threads_;
    /// @brief Semaphores used within the batch
    std::vector<vk::Semaphore> semaphores_;
};

}  // namespace graal::detail