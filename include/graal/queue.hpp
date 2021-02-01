#pragma once
#include <deque>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <graal/access_mode.hpp>
#include <graal/buffer.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/resource.hpp>
#include <graal/detail/task.hpp>
#include <graal/device.hpp>
#include <graal/image.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/image_usage.hpp>
#include <graal/range.hpp>
#include <graal/render_pass.hpp>
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
    friend class context;
    friend class detail::context_impl;

    template<image_type, image_usage, access_mode, bool>
    friend class image_accessor;

    template<typename, buffer_usage, access_mode, bool>
    friend class buffer_accessor;

public:
    // Called by buffer accessors to register an use of a buffer in a task
    template<typename T>
    void add_buffer_access(buffer<T>& buffer, vk::AccessFlags access_mask,
            vk::PipelineStageFlags input_stage, vk::PipelineStageFlags output_stage) const {
        add_buffer_access(static_cast<std::shared_ptr<detail::buffer_impl>>(buffer.impl_),
                access_mask, input_stage, output_stage);
    }

    // Called by image accessors to register an use of an image in a task
    template<image_type Type>
    void add_image_access(image<Type> image, vk::AccessFlags access_mask,
            vk::PipelineStageFlags input_stage, vk::PipelineStageFlags output_stage,
            vk::ImageLayout layout) const {
        add_image_access(static_cast<std::shared_ptr<detail::image_impl>>(image.impl_), access_mask,
                input_stage, output_stage, layout);
    }

    void add_swapchain_access(swapchain swapchain, vk::AccessFlags access_mask,
            vk::PipelineStageFlags input_stage, vk::PipelineStageFlags output_stage,
            vk::ImageLayout layout) const {
        add_image_access(swapchain.impl_, access_mask, input_stage, output_stage, layout);
    }

private:
    handler(detail::context_impl& context, detail::task& task) : context_{ context}, task_{task} {
    }

    // Called by buffer accessors to register an use of a buffer in a task
    void add_buffer_access(std::shared_ptr<detail::buffer_resource> buffer,
            vk::AccessFlags access_mask, vk::PipelineStageFlags input_stage,
            vk::PipelineStageFlags output_stage) const;

    // Called by image accessors to register an use of an image in a task
    void add_image_access(std::shared_ptr<detail::image_resource> image,
            vk::AccessFlags access_mask, vk::PipelineStageFlags input_stage,
            vk::PipelineStageFlags output_stage, vk::ImageLayout layout) const;

    detail::context_impl& context_;
    detail::task& task_;
};

//-----------------------------------------------------------------------------

/// @brief
class context {
public:
    context(device& dev);

    /// @brief Schedule a render pass
    /// @tparam F
    /// @param f
    template<typename F>
    void render_pass(const render_pass_desc& desc, F f) {
        render_pass("", desc, std::move(f));
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
    void render_pass(std::string_view name, const render_pass_desc& desc, F f) {
        static_assert(std::is_invocable_v<std::invoke_result_t<F, handler&>, vk::RenderPass,
                              vk::CommandBuffer>,
                "render pass evaluation callback has an invalid signature: expected "
                "`void(vk::RenderPass, vk::CommandBuffer)`");

        auto& task = create_render_pass_task(name, desc);
        handler h{*impl_, task};
        task.d.u.render.callback = f(h);
    }

    /// @brief Schedule a compute pass
    /// @tparam F
    /// @param f
    template<typename F>
    void compute_pass(F f) {
        compute_pass_internal("", false, std::move(f));
    }

    /// @brief Schedule a compute pass
    /// @tparam F
    /// @param f
    template<typename F>
    void compute_pass(std::string_view name, F f) {
        compute_pass_internal(name, false, std::move(f));
    }

    /// @brief Schedule a compute pass
    /// @tparam F
    /// @param f
    template<typename F>
    void compute_pass_async(F f) {
        compute_pass_internal("", true, std::move(f));
    }

    /// @brief Schedule a compute pass
    /// @tparam F
    /// @param f
    template<typename F>
    void compute_pass_async(std::string_view name, F f) {
        compute_pass_internal(name, true, std::move(f));
    }

    /*template<typename T>
    staging_buffer<T> get_staging_buffer(size_t size = 1) {
        vk::Buffer buffer;
        vk::DeviceSize offset;
        void* data = get_staging_buffer(size * sizeof(T), alignof(T), buffer, offset);
    }*/

    void* get_staging_buffer(
            size_t align, size_t size, vk::Buffer& out_buffer, vk::DeviceSize& out_offset);

    void present(swapchain swapchain);
    void enqueue_pending_tasks();

    [[nodiscard]] device get_device();

private:
    template<typename F>
    void compute_pass_internal(std::string_view name, bool async, F f) {
        static_assert(std::is_invocable_v<std::invoke_result_t<F, handler&>, vk::CommandBuffer>,
                "compute pass evaluation callback has an invalid signature: expected "
                "`void(vk::CommandBuffer)`");

        auto& task = create_compute_pass_task(name, async);
        handler h{*impl_, task};
        task.d.u.compute.callback = f(h);
    }

    [[nodiscard]] detail::task& create_render_pass_task(
            std::string_view name, const render_pass_desc& rpd) noexcept;

    [[nodiscard]] detail::task& create_compute_pass_task(
            std::string_view name, bool async) noexcept;

    std::shared_ptr<detail::context_impl> impl_;
};

}  // namespace graal