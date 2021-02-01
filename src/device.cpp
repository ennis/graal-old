#include <graal/device.hpp>
#include <graal/instance.hpp>

#include <fmt/format.h>
#include <optional>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace graal {

vk::DispatchLoaderDynamic vk_default_dynamic_loader;

namespace detail {

namespace {

struct physical_device_and_properties {
    vk::PhysicalDevice device;
    vk::PhysicalDeviceProperties properties;
    vk::PhysicalDeviceFeatures features;
};

struct device_and_queues {
    physical_device_and_properties phy;
    vk::Device device;
    size_t queue_class_map[max_queues];
    vk::Queue queues[max_queues];
};

// TODO remove once MSVC supports std::popcount
constexpr uint32_t popcount(uint32_t x) noexcept {
    uint32_t c = 0;
    for (; x != 0; x >>= 1)
        if (x & 1) c++;
    return c;
}

// finds the best queue family with the specified flags
uint32_t find_queue_family(vk::PhysicalDevice phy,
        const std::vector<vk::QueueFamilyProperties>& queue_family_properties,
        vk::QueueFlagBits target_flag, vk::SurfaceKHR present_surface = nullptr) {
    std::optional<uint32_t> best_queue_family;

    uint32_t index = 0;
    for (auto queue_family : queue_family_properties) {
        if (queue_family.queueFlags & target_flag) {
            // matches the intended usage

            // if present_surface != nullptr, check that it also supports presentation
            // to the given surface
            if (present_surface) {
                if (!phy.getSurfaceSupportKHR(index, present_surface)) {
                    // does not support presentation, skip it
                    continue;
                }
            }

            if (best_queue_family.has_value()) {
                // there was already a queue for the specified usage,
                // change it only if it is more specialized.
                // to determine if it is more specialized, count number of bits (XXX
                // sketchy?)
                uint32_t last = best_queue_family.value();
                if (popcount(static_cast<VkQueueFlags>(queue_family.queueFlags)) < popcount(
                            static_cast<VkQueueFlags>(queue_family_properties[last].queueFlags))) {
                    best_queue_family = index;
                }
            } else {
                best_queue_family = index;
            }
        }
        ++index;
    }

    if (!best_queue_family.has_value()) {
        throw std::runtime_error{"could not find a compatible queue"};
    }
    return best_queue_family.value();
}

/// @brief Selects an appropriate physical device
/// @param instance
/// @return
physical_device_and_properties select_physical_device(vk::Instance instance) {
    auto physical_devices = instance.enumeratePhysicalDevices();
    if (physical_devices.size() == 0) { throw std::runtime_error{"no device with vulkan support"}; }

    vk::PhysicalDevice sel_phy;
    vk::PhysicalDeviceProperties sel_phy_properties;
    vk::PhysicalDeviceFeatures sel_phy_features;
    for (auto phy : physical_devices) {
        auto props = phy.getProperties();
        auto features = phy.getFeatures();
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            sel_phy = phy;
            sel_phy_properties = props;
            sel_phy_features = features;
        }
    }
    // TODO fallbacks

    if (!sel_phy) { throw std::runtime_error{"no suitable physical device"}; }
    return physical_device_and_properties{
            .device = sel_phy, .properties = sel_phy_properties, .features = sel_phy_features};
}

}  // namespace

/// @brief Creates the vulkan device and default queues
/// @param instance
/// @return
void device_impl::create_vk_device_and_queues(vk::SurfaceKHR present_surface) {
    auto phy = select_physical_device(instance_);

    fmt::print("Selected physical device: {}\n", phy.properties.deviceName);

    // --- find the best queue families for graphics, compute, and transfer
    auto queue_family_props = phy.device.getQueueFamilyProperties();

    graphics_queue_family_ = find_queue_family(
            phy.device, queue_family_props, vk::QueueFlagBits::eGraphics, present_surface);
    compute_queue_family_ =
            find_queue_family(phy.device, queue_family_props, vk::QueueFlagBits::eCompute, nullptr);
    transfer_queue_family_ = find_queue_family(
            phy.device, queue_family_props, vk::QueueFlagBits::eTransfer, nullptr);

    fmt::print("Graphics queue family: {} ({})\n", graphics_queue_family_,
            to_string(queue_family_props[graphics_queue_family_].queueFlags));
    fmt::print("Compute queue family: {} ({})\n", compute_queue_family_,
            to_string(queue_family_props[compute_queue_family_].queueFlags));
    fmt::print("Transfer queue family: {} ({})\n", transfer_queue_family_,
            to_string(queue_family_props[transfer_queue_family_].queueFlags));

    std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
    const float queue_priority = 1.0f;
    for (auto family : {graphics_queue_family_, compute_queue_family_, transfer_queue_family_}) {
        // only create multiple queues if the families are different; otherwise use
        // the same queue
        bool already_created = false;
        for (const auto& ci : device_queue_create_infos) {
            if (ci.queueFamilyIndex == family) {
                already_created = true;
                break;
            }
        }
        if (already_created) continue;
        device_queue_create_infos.push_back(vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &queue_priority});
    }

    vk::PhysicalDeviceTimelineSemaphoreFeatures timeline_features{.timelineSemaphore = true};

    vk::PhysicalDeviceFeatures2 features2{.pNext = &timeline_features,
            .features{.tessellationShader = true,
                    .fillModeNonSolid = true,
                    .samplerAnisotropy = true,
                    .shaderStorageImageExtendedFormats = true}};

    // auto props2 = phy.device.getProperties2<vk::PhysicalDeviceProperties2,
    // vk::PhysicalDeviceTimelineSemaphoreProperties>();

    const char* const device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::DeviceCreateInfo dci{
            .pNext = &features2,
            .queueCreateInfoCount = (uint32_t) device_queue_create_infos.size(),
            .pQueueCreateInfos = device_queue_create_infos.data(),
            .enabledLayerCount = 0,
            .enabledExtensionCount = 1,
            .ppEnabledExtensionNames = device_extensions,
            .pEnabledFeatures = nullptr,
    };

    // create device
    device_ = phy.device.createDevice(dci);

    const auto graphics_queue = device_.getQueue(graphics_queue_family_, 0);
    const auto compute_queue = device_.getQueue(compute_queue_family_, 0);
    const auto transfer_queue = device_.getQueue(transfer_queue_family_, 0);

    const size_t graphics_queue_index = 0;
    const size_t compute_queue_index = compute_queue == graphics_queue ? 0 : 1;
    const size_t transfer_queue_index =
            transfer_queue == graphics_queue ? 0 : (transfer_queue == compute_queue ? 1 : 2);
    const size_t present_queue_index = 0;

    // initialize
    physical_device_ = phy.device;
    physical_device_properties_ = phy.properties;
    physical_device_features_ = phy.features;

    queues_info_.queues[graphics_queue_index] = graphics_queue;
    queues_info_.queues[compute_queue_index] = compute_queue;
    queues_info_.queues[transfer_queue_index] = transfer_queue;

    queues_info_.families[graphics_queue_index] = graphics_queue_family_;
    queues_info_.families[compute_queue_index] = compute_queue_family_;
    queues_info_.families[transfer_queue_index] = transfer_queue_family_;
    queues_info_.families[present_queue_index] = graphics_queue_family_;

    queues_info_.indices.graphics = graphics_queue_index;
    queues_info_.indices.compute = compute_queue_index;
    queues_info_.indices.transfer = transfer_queue_index;
    queues_info_.indices.present = present_queue_index;

    vk_default_dynamic_loader.init(instance_, device_);
}

device_impl::device_impl(vk::SurfaceKHR present_surface) 
{
    instance_ = get_vulkan_instance();
    create_vk_device_and_queues(present_surface);

    // create a memory allocator instance
    VmaAllocatorCreateInfo allocator_create_info{
            .flags = 0,
            .physicalDevice = physical_device_,
            .device = device_,
            .preferredLargeHeapBlockSize = 0,  // default
            .pAllocationCallbacks = nullptr,
            .pDeviceMemoryCallbacks = nullptr,
            .frameInUseCount = 2,
            .pHeapSizeLimit = nullptr,
            .pVulkanFunctions = nullptr,
            .pRecordSettings = nullptr,
            .instance = instance_,
            .vulkanApiVersion = vk_api_version,
    };

    if (auto status = vmaCreateAllocator(&allocator_create_info, &allocator_);
            status != VK_SUCCESS) {
        throw std::runtime_error{"vmaCreateAllocator failed"};
    }
}

device_impl::~device_impl() {
    // TODO wait?
    device_.destroy();
    instance_.destroy();
}

[[nodiscard]] handle<vk::Semaphore> device_impl::create_binary_semaphore() {

    if (free_semaphores_.empty()) {
        vk::SemaphoreCreateInfo sci;
        auto sem = device_.createSemaphore(sci);
        return handle{ sem };
    }
    return free_semaphores_.pop_back();
}

void device_impl::recycle_binary_semaphore(handle<vk::Semaphore> semaphore) {
    free_semaphores_.push_back(std::move(semaphore));
}

}  // namespace detail
}  // namespace graal