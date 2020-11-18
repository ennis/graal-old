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
#include <graal/detail/task.hpp>
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
    void add_buffer_access(buffer<T, HostVisible>& buffer, access_mode mode, buffer_usage usage) {
        add_buffer_access(buffer.impl_, mode, usage);
    }

    // Called by image accessors to register an use of an image in a task
    template<image_type Type, bool HostVisible>
    void add_image_access(image<Type, HostVisible> image, access_mode mode, image_usage usage) {
        add_image_access(image.impl_, mode, usage);
    }

private:
    handler(std::unique_ptr<detail::submit_task> task) : task_{std::move(task)} {
    }

    // Called by buffer accessors to register an use of a buffer in a task
    void add_buffer_access(
            std::shared_ptr<detail::buffer_impl> buffer, access_mode mode, buffer_usage usage);

    // Called by image accessors to register an use of an image in a task
    void add_image_access(
            std::shared_ptr<detail::image_impl> image, access_mode mode, image_usage usage);

    // TODO add_direct_image_access for raw VkImages
    // TODO add_direct_buffer_access for raw VkBuffer
    void add_resource_access(std::shared_ptr<detail::virtual_resource> resource, access_mode mode);

    std::unique_ptr<detail::submit_task> task_;
    struct resource_access {
        detail::resource_ptr resource;
        access_mode mode;
        union {
            buffer_usage buffer;
            image_usage image;
        } usage_flags;
    };
    std::vector<resource_access> accesses_;
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
    template<typename F>
    void schedule(std::string_view name, F f) {
        // create task
        auto h = begin_build_task(name);
        f(h);
        end_build_task(std::move(h));
    }

    void enqueue_pending_tasks();

    void present(swapchain_image&& image);

private:
    /// @brief Called to start building a task. Returns the task sequence number.
    [[nodiscard]] handler begin_build_task(std::string_view name) const noexcept;

    /// @brief Called to finish building a task. Adds the task to the current batch.
    void end_build_task(handler&& handler);

    std::shared_ptr<detail::queue_impl> impl_;
};

}  // namespace graal