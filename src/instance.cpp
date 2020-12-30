#include <graal/instance.hpp>

namespace graal {

namespace {

vk::Instance vulkan_instance;

/// @brief List of validation layers to enable
constexpr std::array<const char*, 1> validation_layers = {"VK_LAYER_KHRONOS_validation"};

/// @brief Checks if all validation layers are supported
/// @return
bool check_validation_layer_support() {
    auto available_layers = vk::enumerateInstanceLayerProperties();

    for (auto layer : validation_layers) {
        bool found = std::any_of(available_layers.begin(), available_layers.end(),
                [&](const vk::LayerProperties& layer_props) {
                    return std::strcmp(layer_props.layerName, layer) == 0;
                });
        if (!found) { return false; }
    }

    return true;
}

}  // namespace

vk::Instance initialize_vulkan_instance(std::span<const char* const> required_instance_extensions) {
    if (vulkan_instance) { throw std::logic_error{"vulkan instance already initialized"}; }

    bool validation_available = check_validation_layer_support();
    if (!validation_available) { throw std::runtime_error{"validation layer not available"}; }

    std::vector<const char*> instance_extensions;
    instance_extensions.assign(
            required_instance_extensions.begin(), required_instance_extensions.end());
    instance_extensions.push_back("VK_KHR_get_surface_capabilities2");
    instance_extensions.push_back("VK_EXT_debug_utils");

    vk::ApplicationInfo app_info{.pApplicationName = "graal",
            .applicationVersion = 1,
            .pEngineName = "graal",
            .engineVersion = 1,
            .apiVersion = vk_api_version};
    vk::InstanceCreateInfo instance_create_info{.pApplicationInfo = &app_info,
            .enabledExtensionCount = (uint32_t) instance_extensions.size(),
            .ppEnabledExtensionNames = instance_extensions.data()};
    if (validation_available) {
        instance_create_info.enabledLayerCount = validation_layers.size();
        instance_create_info.ppEnabledLayerNames = validation_layers.data();
    }

    vulkan_instance = vk::createInstance(instance_create_info);
    return vulkan_instance;
}

vk::Instance get_vulkan_instance() {
    if (!vulkan_instance) { throw std::logic_error{"vulkan instance not initialized"}; }
    return vulkan_instance;
}

void release_vulkan_instance() {
    // TODO
}

}  // namespace graal