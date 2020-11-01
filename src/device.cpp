#include <fmt/format.h>
#include <graal/device.hpp>
#include <optional>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace graal {
namespace detail {

namespace {

constexpr uint32_t vk_api_version = VK_API_VERSION_1_1;

struct physical_device_and_properties {
  vk::PhysicalDevice           device;
  vk::PhysicalDeviceProperties properties;
  vk::PhysicalDeviceFeatures   features;
};

struct device_and_queues {
  physical_device_and_properties phy;
  vk::Device                     device;
  size_t                         graphics_queue_family;
  vk::Queue                      graphics_queue;
  size_t                         compute_queue_family;
  vk::Queue                      compute_queue;
  size_t                         transfer_queue_family;
  vk::Queue                      transfer_queue;
};

/// @brief List of validation layers to enable
constexpr std::array<const char *, 1> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

/// @brief Checks if all validation layers are supported
/// @return
bool check_validation_layer_support() {
  auto available_layers = vk::enumerateInstanceLayerProperties();

  for (auto layer : validation_layers) {
    bool found =
        std::any_of(available_layers.begin(), available_layers.end(),
                    [&](const vk::LayerProperties &layer_props) {
                      return std::strcmp(layer_props.layerName, layer) == 0;
                    });
    if (!found) {
      return false;
    }
  }

  return true;
}

// TODO remove once MSVC supports std::popcount
constexpr uint32_t popcount(uint32_t x) noexcept {
  uint32_t c = 0;
  for (; x != 0; x >>= 1)
    if (x & 1)
      c++;
  return c;
}

// finds the best queue family with the specified flags
uint32_t find_queue_family(
    const std::vector<vk::QueueFamilyProperties> &queue_family_properties,
    vk::QueueFlagBits                             target_flag) {
  std::optional<uint32_t> best_queue_family;

  uint32_t index = 0;
  for (auto queue_family : queue_family_properties) {
    if (queue_family.queueFlags & target_flag) {
      // matches the intended usage
      if (best_queue_family.has_value()) {
        // there was already a queue for the specified usage,
        // change it only if it is more specialized.
        // to determine if it is more specialized, count number of bits (XXX
        // sketchy?)
        uint32_t last = best_queue_family.value();
        if (popcount(static_cast<VkQueueFlags>(queue_family.queueFlags)) <
            popcount(static_cast<VkQueueFlags>(
                queue_family_properties[last].queueFlags))) {
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

vk::Instance
create_vk_instance(std::span<const char *const> required_instance_extensions) {
  bool validation_available = check_validation_layer_support();
  if (!validation_available) {
    throw std::runtime_error{"validation layer not available"};
  }

  vk::ApplicationInfo    app_info{.pApplicationName = "graal",
                               .applicationVersion = 1,
                               .pEngineName = "graal",
                               .engineVersion = 1,
                               .apiVersion = vk_api_version};
  vk::InstanceCreateInfo instance_create_info{
      .pApplicationInfo = &app_info,
      .enabledExtensionCount = (uint32_t)required_instance_extensions.size(),
      .ppEnabledExtensionNames = required_instance_extensions.data()};
  if (validation_available) {
    instance_create_info.enabledLayerCount = validation_layers.size();
    instance_create_info.ppEnabledLayerNames = validation_layers.data();
  }
  return vk::createInstance(instance_create_info);
}

/// @brief Selects an appropriate physical device
/// @param instance
/// @return
physical_device_and_properties select_physical_device(vk::Instance instance) {
  auto physical_devices = instance.enumeratePhysicalDevices();
  if (physical_devices.size() == 0) {
    throw std::runtime_error{"no device with vulkan support"};
  }

  vk::PhysicalDevice           sel_phy;
  vk::PhysicalDeviceProperties sel_phy_properties;
  vk::PhysicalDeviceFeatures   sel_phy_features;
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

  if (!sel_phy) {
    throw std::runtime_error{"no suitable physical device"};
  }
  return physical_device_and_properties{.device = sel_phy,
                                        .properties = sel_phy_properties,
                                        .features = sel_phy_features};
}

/// @brief Creates the vulkan device and default queues
/// @param instance
/// @return
device_and_queues create_vk_device_and_queues(vk::Instance instance) {
  auto phy = select_physical_device(instance);

  fmt::print("Selected physical device: {}\n", phy.properties.deviceName);

  // --- find the best queue families for graphics, compute, and transfer
  auto queue_family_props = phy.device.getQueueFamilyProperties();

  uint32_t graphics_queue_family =
      find_queue_family(queue_family_props, vk::QueueFlagBits::eGraphics);
  uint32_t compute_queue_family =
      find_queue_family(queue_family_props, vk::QueueFlagBits::eCompute);
  uint32_t transfer_queue_family =
      find_queue_family(queue_family_props, vk::QueueFlagBits::eTransfer);

  fmt::print("Graphics queue family: {} ({})\n", graphics_queue_family,
             to_string(queue_family_props[graphics_queue_family].queueFlags));
  fmt::print("Compute queue family: {} ({})\n", compute_queue_family,
             to_string(queue_family_props[compute_queue_family].queueFlags));
  fmt::print("Transfer queue family: {} ({})\n", transfer_queue_family,
             to_string(queue_family_props[transfer_queue_family].queueFlags));

  std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
  const float                            queue_priority = 1.0f;
  for (auto family :
       {graphics_queue_family, compute_queue_family, transfer_queue_family}) {
    // only create multiple queues if the families are different; otherwise use
    // the same queue
    bool already_created = false;
    for (const auto &ci : device_queue_create_infos) {
      if (ci.queueFamilyIndex == family) {
        already_created = true;
        break;
      }
    }
    if (already_created)
      break;
    device_queue_create_infos.push_back(
        vk::DeviceQueueCreateInfo{.queueFamilyIndex = family,
                                  .queueCount = 1,
                                  .pQueuePriorities = &queue_priority});
  }

  vk::PhysicalDeviceFeatures features{
      .tessellationShader = true,
      .fillModeNonSolid = true,
      .samplerAnisotropy = true,
      .shaderStorageImageExtendedFormats = true,
  };

  vk::DeviceCreateInfo dci{
      .queueCreateInfoCount = (uint32_t)device_queue_create_infos.size(),
      .pQueueCreateInfos = device_queue_create_infos.data(),
      .enabledLayerCount = 0,
      .enabledExtensionCount = 0,
      .pEnabledFeatures = &features,
  };

  auto device = phy.device.createDevice(dci);

  auto graphics_queue = device.getQueue(graphics_queue_family, 0);
  auto compute_queue = device.getQueue(compute_queue_family, 0);
  auto transfer_queue = device.getQueue(transfer_queue_family, 0);

  return device_and_queues{.phy = phy,
                           .device = device,
                           .graphics_queue_family = graphics_queue_family,
                           .graphics_queue = graphics_queue,
                           .compute_queue_family = compute_queue_family,
                           .compute_queue = compute_queue,
                           .transfer_queue_family = transfer_queue_family,
                           .transfer_queue = transfer_queue};
}

} // namespace

device_impl::device_impl(
    std::span<const char *const> required_instance_extensions) {
  instance_ = create_vk_instance(required_instance_extensions);
  auto device_and_queues = create_vk_device_and_queues(instance_);

  physical_device_ = device_and_queues.phy.device;
  physical_device_properties_ = device_and_queues.phy.properties;
  physical_device_features_ = device_and_queues.phy.features;
  device_ = device_and_queues.device;
  graphics_queue_family_ = device_and_queues.graphics_queue_family;
  graphics_queue_ = device_and_queues.graphics_queue;
  compute_queue_family_ = device_and_queues.compute_queue_family;
  compute_queue_ = device_and_queues.compute_queue;
  transfer_queue_family_ = device_and_queues.transfer_queue_family;
  transfer_queue_ = device_and_queues.transfer_queue;

  // create a memory allocator instance
  VmaAllocatorCreateInfo allocator_create_info{
      .flags = 0,
      .physicalDevice = physical_device_,
      .device = device_,
      .preferredLargeHeapBlockSize = 0, // default
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

} // namespace detail
} // namespace graal