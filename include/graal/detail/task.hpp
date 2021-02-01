#pragma once
#include <graal/detail/sequence_number.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/detail/vk_handle.hpp>
#include <graal/queue_class.hpp>
#include <graal/render_pass.hpp>

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
    vk::AccessFlags access_mask;
    vk::PipelineStageFlags input_stage;
    vk::PipelineStageFlags output_stage;
    /// @brief Whether sync needs a binary semaphore
    //bool binary_semaphore = false;
};

enum class task_type { render_pass, compute_pass, transfer, present };

using per_queue_wait_serials = std::array<serial_number, max_queues>;
using per_queue_wait_dst_stages = std::array<vk::PipelineStageFlags, max_queues>;

// maybe "pass" would be more appropriate?
struct task {
    struct present_details {
        vk::SwapchainKHR swapchain;
        uint32_t image_index;
    };

    struct render_details {
        std::vector<attachment> color_attachments;
        std::vector<attachment> input_attachments;
        std::optional<attachment> depth_attachment;
        std::function<void(vk::RenderPass, vk::CommandBuffer)> callback;
    };

    struct compute_details {
        std::function<void(vk::CommandBuffer)> callback;
    };

    struct details {
        /// @brief Constructs a compute task
        details() {
            type_ = task_type::compute_pass;
            new (&u.compute) compute_details;
        }

        /// @brief Constructs a present task
        /// @param swapchain
        /// @param image
        details(vk::SwapchainKHR swapchain, uint32_t image_index) {
            type_ = task_type::present;
            new (&u.present) present_details;
            u.present.swapchain = swapchain;
            u.present.image_index = image_index;
        }

        /// @brief Constructs a renderpass task
        details(const render_pass_desc& rpd) {
            type_ = task_type::render_pass;
            new (&u.render) render_details;

            for (auto&& a : rpd.color_attachments) {
                u.render.color_attachments.push_back(a);
            }
            for (auto&& a : rpd.input_attachments) {
                u.render.input_attachments.push_back(a);
            }
            if (rpd.depth_attachment) { u.render.depth_attachment = *rpd.depth_attachment; }
        }

        ~details() {
            destroy();
        }

        void destroy() {
            switch (type_) {
                case task_type::render_pass: u.render.~render_details(); break;
                case task_type::present: u.present.~present_details(); break;
                case task_type::compute_pass: u.compute.~compute_details(); break;
            }
        }

        details(details&& t) noexcept : type_{t.type_} {
            switch (t.type_) {
                case task_type::render_pass:
                    new (&u.render) render_details(std::move(t.u.render));
                    break;
                case task_type::compute_pass:
                    new (&u.compute) compute_details(std::move(t.u.compute));
                    break;
                case task_type::present:
                    new (&u.present) present_details(std::move(t.u.present));
                    break;
            }
        }

        details& operator=(details&& t) noexcept {
            destroy();

            type_ = t.type_;
            switch (t.type_) {
                case task_type::render_pass:
                    new (&u.render) render_details(std::move(t.u.render));
                    break;
                case task_type::compute_pass:
                    new (&u.compute) compute_details(std::move(t.u.compute));
                    break;
                case task_type::present:
                    new (&u.present) present_details(std::move(t.u.present));
                    break;
            }

            return *this;
        }

        [[nodiscard]] task_type type() const noexcept {
            return type_;
        }

        // data specific to the task type
        union U {
            U() {
            }
            ~U() {
            }
            present_details present;
            render_details render;
            compute_details compute;
        } u;

    private:
        task_type type_ = task_type::render_pass;
    };

    /// @brief Constructs a compute task
    task() : d{} {
    }

    /// @brief Constructs a present task
    /// @param swapchain
    /// @param image
    task(vk::SwapchainKHR swapchain, uint32_t image_index) : d{swapchain, image_index} {
    }

    /// @brief Constructs a renderpass task
    task(const render_pass_desc& rpd) : d{rpd} {
    }

    [[nodiscard]] task_type type() const noexcept {
        return d.type();
    }

    /// Whether the task needs a pipeline barrier before executing.
    [[nodiscard]] bool needs_barrier() const noexcept {
        return src_stage_mask != vk::PipelineStageFlagBits::eTopOfPipe
            || input_stage_mask != vk::PipelineStageFlagBits::eBottomOfPipe
            || !buffer_memory_barriers.empty()
            || !image_memory_barriers.empty();
    }

    struct resource_access {
        size_t index;
        vk::AccessFlags access_mask;
    };

    /// @brief Name of the task.
    std::string name;
    /// @brief Submission number of the task.
    submission_number snn;
    /// @brief Index of the command buffer batch (assigned during scheduling).
    size_t submit_batch_index = 0;
    /// @brief Predecessors of the task (all tasks that must happen before this one).
    std::vector<size_t> preds;
    /// @brief Successors of the task (all tasks for which this task is a predecessor).
    std::vector<size_t> succs;
    /// @brief List of accesses made by the task to resources.
    /// FIXME Right now, this is used only for debugging purposes, and when allocating memory for the resources.
    /// It probably could be removed.
    std::vector<resource_access> accesses;

    /// @brief The list of image memory barriers that must be applied before executing the task.
    std::vector<vk::ImageMemoryBarrier> image_memory_barriers;
    /// @brief The list of buffer memory barriers that must be applied before executing the task.
    std::vector<vk::BufferMemoryBarrier> buffer_memory_barriers;

    /// @brief Source stage mask for the pre-execution barrier.
    vk::PipelineStageFlags src_stage_mask{};
    /// @brief Destination stage mask for the pre-execution barrier.
    vk::PipelineStageFlags input_stage_mask{};
    /// @brief 
    vk::PipelineStageFlags output_stage_mask{};
    /// @brief Whether a signal operation must be performed on the queue after the task.
    bool signal = false;
    /// @brief Whether the task should wait on semaphores.
    bool wait = false;
    /// @brief If wait == true, the serials that the task should wait for (on the corresponding timelines).
    per_queue_wait_serials wait_serials{};
    /// @brief Corresponding destination stages for each wait defined by wait_serials.
    per_queue_wait_dst_stages wait_dst_stages{};
    /// @brief If wait == true, the binary semaphores that the task should wait on,
    /// in addition to the waits specified by wait_serials.
    handle_vector<vk::Semaphore> wait_semaphores;

    // data specific to the task type
    details d;

private:
};

}  // namespace graal::detail