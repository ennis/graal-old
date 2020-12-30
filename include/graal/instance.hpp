#pragma once
#include <span>
#include <vulkan/vulkan.hpp>

namespace graal {
constexpr uint32_t vk_api_version = VK_API_VERSION_1_2;

vk::Instance initialize_vulkan_instance(
    std::span<const char *const> required_instance_extensions);
vk::Instance get_vulkan_instance();
void         release_vulkan_instance();

} // namespace graal