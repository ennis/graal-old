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

/*
enum class attachment_load_op {
    clear = VK_ATTACHMENT_LOAD_OP_CLEAR,
    load = VK_ATTACHMENT_LOAD_OP_LOAD,
    dont_care = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
};

enum class attachment_store_op {
    store = VK_ATTACHMENT_STORE_OP_STORE,
    dont_care = VK_ATTACHMENT_STORE_OP_DONT_CARE,
};

struct clear_color_value {
    clear_color_value(float r, float g, float b, float a) {
        value.float32[0] = r;
        value.float32[1] = g;
        value.float32[2] = b;
        value.float32[3] = a;
    }

    clear_color_value(int32_t r, int32_t g, int32_t b, int32_t a) {
        value.int32[0] = r;
        value.int32[1] = g;
        value.int32[2] = b;
        value.int32[3] = a;
    }

    clear_color_value(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
        value.uint32[0] = r;
        value.uint32[1] = g;
        value.uint32[2] = b;
        value.uint32[3] = a;
    }

    VkClearColorValue value;
};

struct clear_depth_stencil_value {};

struct clear_value {
    VkClearValue cv;
};

struct attachment_desc {
    // color_attachment, color format
    template<image_type D, bool Ext>
    attachment_desc(image<D, Ext> image, attachment_load_op load_op, attachment_store_op store_op,
        clear_color_value clear_value = {});

    // color_attachment, depth-stencil format
    template<image_type D, bool Ext>
    attachment_desc(image<D, Ext> image, attachment_load_op load_op, attachment_store_op store_op,
        attachment_load_op stencil_load_op, attachment_store_op stencil_store_op,
        clear_depth_stencil_value clear_value = {});
};

struct render_pass_desc {
    std::span<const attachment_desc> color_attachments;
    std::span<const attachment_desc> input_attachments;
    const attachment_desc* depth_attachment = nullptr;
};*/

enum class task_type {
    render_pass,
    compute_pass,
    transfer,
    present
};

struct task {
    struct present_details {
        vk::SwapchainKHR swapchain;
        vk::Image image;
    };

    struct render_details {
        vk::RenderPass render_pass;
        std::function<void(vk::RenderPass, vk::CommandBuffer)> callback;
    };

    struct compute_details {
        std::function<void(vk::CommandBuffer)> callback;
    };

    /// @brief Constructs a compute task
    task() 
    {
        type_ = task_type::compute_pass;
        new (&detail.compute) compute_details;
    }

    /// @brief Constructs a present task
    /// @param swapchain 
    /// @param image 
    task(vk::SwapchainKHR swapchain, vk::Image image) {
        type_ = task_type::present;
        new (&detail.present) present_details;
        detail.present.swapchain = swapchain;
        detail.present.image = image;
    }

    /// @brief Constructs a renderpass task
    task(vk::RenderPass render_pass) {
        type_ = task_type::render_pass;
        new (&detail.render) render_details;
        detail.render.render_pass = render_pass;
    }

    ~task() {
        destroy_detail();
    }

    task(task&& t) : 
        name{ std::move(t.name) }, 
        type_{ t.type_ },
        sn{ t.sn },
        preds{std::move(t.preds)},
        succs{std::move(t.succs)},
        wait_binary{ std::move(t.wait_binary) },
        signal_binary{ std::move(t.signal_binary) },
        accesses{ std::move(t.accesses) }
    {
        switch (t.type_) {
        case task_type::render_pass: 
            new (&detail.render) render_details(std::move(t.detail.render));
            break;
        case task_type::compute_pass:
            new (&detail.compute) compute_details(std::move(t.detail.compute));
            break;
        case task_type::present:
            new (&detail.present) present_details(std::move(t.detail.present));
            break;
        }
    }

    task& operator=(task&& t) {
        destroy_detail();

        name = std::move(t.name);
        type_ = std::move(t.type_);
        sn = std::move(t.sn);
        preds = std::move(t.preds);
        succs = std::move(t.succs);
        wait_binary = std::move(t.wait_binary);
        signal_binary = std::move(t.signal_binary);
        accesses = std::move(t.accesses);

        switch (t.type_) {
        case task_type::render_pass:
            new (&detail.render) render_details(std::move(t.detail.render));
            break;
        case task_type::compute_pass:
            new (&detail.compute) compute_details(std::move(t.detail.compute));
            break;
        case task_type::present:
            new (&detail.present) present_details(std::move(t.detail.present));
            break;
        }

        return *this;
    }

    [[nodiscard]] task_type type() const noexcept { return type_; }

    /*task(const task& t) :
        name{ std::move(t.name) },
        type{ std::move(t.type) },
        snn{ std::move(t.snn) },
        preds{ std::move(t.preds) },
        succs{ std::move(t.succs) },
        wait_binary{ std::move(t.wait_binary) },
        signal_binary{ std::move(t.signal_binary) },
        accesses{ std::move(t.accesses) }
    {
        switch (t.type) {
        case task_type::render_pass:
            new (&detail.render) render_details(std::move(t.detail.render));
            break;
        case task_type::compute_pass:
            new (&detail.compute) compute_details(std::move(t.detail.compute));
            break;
        case task_type::present:
            new (&detail.present) present_details(std::move(t.detail.present));
            break;
        }
    }*/

    struct resource_access {
        size_t index;
        resource_access_details details;
    };

    std::string name;

    /// @brief The submission number (SNN).
    /// NOTE a first serial is assigned when the task is created, without a queue, for the purposes of 
    /// DAG building. However, the serial might change after submission, due to task reordering.
    serial_number sn;

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

    union U {
        U() {}
        ~U() {}
        present_details present;
        render_details render;
        compute_details compute;
    } detail;

private:
    task_type type_ = task_type::render_pass;

    void destroy_detail() {
        switch (type_) {
        case task_type::render_pass:
            detail.render.~render_details();
            break;
        case task_type::present:
            detail.present.~present_details();
            break;
        case task_type::compute_pass:
            detail.compute.~compute_details();
            break;
        }
    }
};

}  // namespace graal::detail