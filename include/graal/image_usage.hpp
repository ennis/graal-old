#pragma once
#include <graal/bitmask.hpp>
#include <vulkan/vulkan.hpp>

namespace graal {

enum class image_usage {
    transfer_src = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    transfer_dst = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    sampled = VK_IMAGE_USAGE_SAMPLED_BIT,
    storage = VK_IMAGE_USAGE_STORAGE_BIT,
    color_attachment = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    depth_stencil_attachment = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    transient_attachment = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
    input_attachment = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
};

GRAAL_BITMASK(image_usage)

}  // namespace graal