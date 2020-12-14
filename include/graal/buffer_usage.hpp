#pragma once
#include <graal/bitmask.hpp>
#include <vulkan/vulkan.hpp>

namespace graal {

enum class buffer_usage {
    transfer_src = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    transfer_dst = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    uniform_texel_buffer = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
    storage_texel_buffer = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
    uniform_buffer = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    storage_buffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    index_buffer = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    vertex_buffer = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    indirect_buffer = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
};

GRAAL_BITMASK(buffer_usage)

}  // namespace graal