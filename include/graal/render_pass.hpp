#pragma once
#include <graal/image.hpp>
#include <graal/swapchain.hpp>

#include <span>
#include <vulkan/vulkan.hpp>

namespace graal {

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
    clear_color_value() : clear_color_value{0.0f, 0.0f, 0.0f, 1.0f} {
    }

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

struct clear_depth_stencil_value {
    VkClearDepthStencilValue v;
};

class attachment {
public:
    // color_attachment, color format
    attachment(swapchain_image image, attachment_load_op load_op, attachment_store_op store_op,
            clear_color_value clear_value = {}) :
        image_resource_{static_cast<std::shared_ptr<detail::swapchain_image_impl>>(image.impl_)} {
    }

    // color_attachment, color format
    template<image_type D>
    attachment(image<D> image, attachment_load_op load_op, attachment_store_op store_op,
            clear_color_value clear_value = {}) :
        image_resource_{static_cast<std::shared_ptr<detail::image_impl>>(image.impl_)},
        load_op_{load_op}, store_op_{store_op} {
    }

    // color_attachment, depth-stencil format
    template<image_type D>
    attachment(image<D> image, attachment_load_op load_op, attachment_store_op store_op,
            attachment_load_op stencil_load_op, attachment_store_op stencil_store_op,
            clear_depth_stencil_value clear_value = {}) :
        image_resource_{static_cast<std::shared_ptr<detail::image_impl>>(image.impl_)},
        load_op_{load_op}, store_op_{store_op}, load_op_{stencil_load_op},
        store_op_{stencil_store_op} {
    }

private:
    detail::resource_ptr image_resource_;
    attachment_load_op load_op_;
    attachment_store_op store_op_;
    attachment_load_op stencil_load_op_;
    attachment_store_op stencil_store_op_;
    VkClearValue clear_value_;
};

struct render_pass_desc {
    std::span<const attachment> color_attachments;
    std::span<const attachment> input_attachments;
    const attachment* depth_attachment = nullptr;
};
}  // namespace graal