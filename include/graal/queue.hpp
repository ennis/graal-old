#pragma once
#include <deque>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <graal/access_mode.hpp>
#include <graal/buffer.hpp>
#include <graal/detail/image_resource.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/resource.hpp>
#include <graal/device.hpp>
#include <graal/image.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/image_usage.hpp>
#include <graal/range.hpp>
#include <graal/swapchain.hpp>

namespace graal {

namespace detail {
class queue_impl;
struct task;
}  // namespace detail

//-----------------------------------------------------------------------------
/// @brief Command handler.
// TODO remove this class and expose submit_task directly instead?
class handler {
    friend class queue;
    friend class detail::queue_impl;

    template<image_type, image_usage, access_mode, bool>
    friend class image_accessor;

    template<typename, buffer_usage, access_mode, bool>
    friend class buffer_accessor;

public:
    /// @brief Adds a callback function that is executed during submission.
    template<typename F>
    void add_command(F&& command_callback) {
        static_assert(std::is_invocable_v<F, vk::CommandBuffer>,
                "command callback has an invalid signature");
        task_->callbacks.push_back(std::move(command_callback));
    }

    // Called by buffer accessors to register an use of a buffer in a task
    template<typename T, bool HostVisible>
    void add_buffer_access(buffer<T, HostVisible>& buffer, vk::AccessFlags access_flags,
            vk::PipelineStageFlags input_stage, vk::PipelineStageFlags output_stage) const {
        add_buffer_access(buffer.impl_, access_flags, input_stage, output_stage);
    }

    // Called by image accessors to register an use of an image in a task
    template<image_type Type, bool HostVisible>
    void add_image_access(image<Type, HostVisible> image, vk::AccessFlags access_flags,
            vk::PipelineStageFlags input_stage, vk::PipelineStageFlags output_stage,
            vk::ImageLayout layout) const {
        add_image_access(image.impl_, access_flags, input_stage, output_stage, layout);
    }

private:
    handler(detail::queue_impl& queue, detail::task& task) : queue_{queue}, task_{task} {
    }

    // Called by buffer accessors to register an use of a buffer in a task
    void add_buffer_access(std::shared_ptr<detail::buffer_impl> buffer,
            vk::AccessFlags access_flags, vk::PipelineStageFlags input_stage,
            vk::PipelineStageFlags output_stage) const;

    // Called by image accessors to register an use of an image in a task
    void add_image_access(std::shared_ptr<detail::image_impl> image, vk::AccessFlags access_flags,
            vk::PipelineStageFlags input_stage, vk::PipelineStageFlags output_stage,
            vk::ImageLayout layout) const;

    // TODO add_direct_image_access for raw VkImages
    // TODO add_direct_buffer_access for raw VkBuffer

    detail::queue_impl& queue_;
    detail::task& task_;
};

//-----------------------------------------------------------------------------

struct queue_properties {
    int max_batches_in_flight = 1;
};

/// @brief
class queue {
public:
    queue(device& device, const queue_properties& props = {});

    template<typename F>
    void schedule(F f) {
        schedule(typeid(f).name(), f);
    }

    /// @brief
    /// @tparam F
    /// @param f
    ///
    /// @par Exception Safety
    /// Basic guarantee: if the handler callback throws, the current state of the queue might be affected
    /// (a task might be partially constructed), but no resource is leaked.
    /// In reasonable programs, the callback probably should not throw exceptions.
    template<typename F>
    void schedule(std::string_view name, F f) {
        // create task
        auto& task = create_task(name);
        handler h{ *impl_, task };
        f(h);
    }

    void enqueue_pending_tasks();

    void present(swapchain_image&& image);

private:
    [[nodiscard]] detail::task& create_task(std::string_view name) noexcept;

    std::shared_ptr<detail::queue_impl> impl_;
};

}  // namespace graal