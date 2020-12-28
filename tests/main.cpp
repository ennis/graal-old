//#include <graal/accessor.hpp>
#include <graal/buffer.hpp>
#include <graal/ext/vertex_traits.hpp>
#include <graal/image.hpp>
#include <graal/instance.hpp>
#include <graal/program.hpp>
#include <graal/queue.hpp>
#include <graal/shader.hpp>

#include <vulkan/vulkan_win32.h>
#include <vulkan/vulkan.hpp>

#define IMGUI_IMPLEMENTATION
#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <iterator>
#include <span>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "load_image.hpp"

using namespace graal;

vk::SurfaceKHR create_surface(GLFWwindow* window) {
    VkSurfaceKHR surface;
    auto result = glfwCreateWindowSurface(get_vulkan_instance(), window, nullptr, &surface);
    if (result != VK_SUCCESS) {
        fmt::print(stderr, "glfwCreateWindowSurface failed (VkResult {:08x})\n", (uint32_t) result);
        throw std::runtime_error{"failed to create window surface"};
    }
    return surface;
}

//-----------------------------------------------------------------------------
// Vertex input

struct vertex_3d {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec2 texcoord;
};

constexpr size_t vtx_stride = sizeof(vertex_3d);

constexpr vk::VertexInputBindingDescription vtx_input_bindings[] = {
        {.binding = 0, .stride = vtx_stride, .inputRate = vk::VertexInputRate::eVertex}};

constexpr vk::VertexInputAttributeDescription vtx_input_attributes[] = {
        {.location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(vertex_3d, position)},
        {.location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(vertex_3d, normal)},
        {.location = 2,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(vertex_3d, tangent)},
        {.location = 3,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(vertex_3d, texcoord)},
};

constexpr vk::PipelineVertexInputStateCreateInfo vtx_input_state = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = vtx_input_bindings,
        .vertexAttributeDescriptionCount = 4,
        .pVertexAttributeDescriptions = vtx_input_attributes};

//-----------------------------------------------------------------------------
// Vertex assembly
constexpr vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false,
};

//-----------------------------------------------------------------------------
// Viewport state
constexpr vk::PipelineViewportStateCreateInfo viewport_state = {
        .viewportCount = 1,
        .pViewports = nullptr,  // dynamic
        .scissorCount = 1,
        .pScissors = nullptr,  // dynamic
};

//-----------------------------------------------------------------------------
// Rasterization
constexpr vk::PipelineRasterizationStateCreateInfo rasterization_state = {.depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .depthBiasConstantFactor = 0.0,
        .depthBiasClamp = 0.0,
        .depthBiasSlopeFactor = 0.0,
        .lineWidth = 1.0};

constexpr vk::PipelineMultisampleStateCreateInfo multisample_state = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
        .minSampleShading = 0.0,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = false,
        .alphaToOneEnable = false};

//-----------------------------------------------------------------------------
// Depth stencil
constexpr vk::PipelineDepthStencilStateCreateInfo depth_stencil_state = {
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false,
        .front =
                {
                        .failOp = vk::StencilOp::eKeep,
                        .passOp = vk::StencilOp::eKeep,
                        .depthFailOp = vk::StencilOp::eKeep,
                        .compareOp = vk::CompareOp::eAlways,
                        .compareMask = 0x0,
                        .writeMask = 0x0,
                        .reference = 0x0,
                },
        .back =
                {
                        .failOp = vk::StencilOp::eKeep,
                        .passOp = vk::StencilOp::eKeep,
                        .depthFailOp = vk::StencilOp::eKeep,
                        .compareOp = vk::CompareOp::eAlways,
                        .compareMask = 0x0,
                        .writeMask = 0x0,
                        .reference = 0x0,
                },
};

//-----------------------------------------------------------------------------
// color blending
constexpr vk::PipelineColorBlendAttachmentState color_blend_attachment_states[] = {{
        .blendEnable = false,
}};

constexpr vk::PipelineColorBlendStateCreateInfo color_blend_state = {.logicOpEnable = false,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = color_blend_attachment_states,
        .blendConstants = {{0.0f, 0.0f, 0.0f, 0.0f}}};

//-----------------------------------------------------------------------------
// Dynamic states
constexpr vk::DynamicState dynamic_states[] = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
};

constexpr vk::PipelineDynamicStateCreateInfo dynamic_state_create_info = {
        .dynamicStateCount = 2, .pDynamicStates = dynamic_states};

//-----------------------------------------------------------------------------
// Descriptor set layouts

struct frame_uniforms {
    glm::mat4 view;
    glm::mat4 view_inverse;
    glm::mat4 view_inverse_transpose;

    glm::mat4 projection;
    glm::mat4 projection_inverse;
    glm::mat4 projection_inverse_transpose;

    glm::mat4 view_projection;
    glm::mat4 view_projection_inverse;
    glm::mat4 view_projection_inverse_transpose;
};

struct object_uniforms {
    glm::mat4 model;
    glm::mat4 model_inverse;
    glm::mat4 model_inverse_transpose;
};

constexpr vk::DescriptorSetLayoutBinding frame_uniforms_descriptor_set_layout_bindings[] = {{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eAllGraphics,
}};

constexpr vk::DescriptorSetLayoutBinding object_uniforms_descriptor_set_layout_bindings[] = {
        {.binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBufferDynamic,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eAllGraphics}};

constexpr vk::DescriptorSetLayoutCreateInfo frame_uniforms_descriptor_set_layout_create_info = {
        .bindingCount = 1,
        .pBindings = frame_uniforms_descriptor_set_layout_bindings,
};

constexpr vk::DescriptorSetLayoutCreateInfo object_uniforms_descriptor_set_layout_create_info = {
        .bindingCount = 1, .pBindings = object_uniforms_descriptor_set_layout_bindings};

//-----------------------------------------------------------------------------
// Render pass
constexpr vk::AttachmentDescription render_pass_attachments[] = {
        // color
        {
                .flags = vk::AttachmentDescriptionFlagBits::eMayAlias,
                .format = vk::Format::eR8G8B8A8Snorm,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .initialLayout = vk::ImageLayout::eGeneral,
                .finalLayout = vk::ImageLayout::eGeneral,
        },
        // depth buffer
        {.flags = vk::AttachmentDescriptionFlagBits::eMayAlias,
                .format = vk::Format::eD32Sfloat,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .initialLayout = vk::ImageLayout::eGeneral,
                .finalLayout = vk::ImageLayout::eGeneral}};

constexpr vk::AttachmentReference color_attachment_refs[] = {{
        .attachment = 0,
        .layout = vk::ImageLayout::eGeneral,
}};

constexpr vk::AttachmentReference depth_attachment_ref = {
        .attachment = 1,
        .layout = vk::ImageLayout::eGeneral,
};

constexpr vk::SubpassDescription subpasses[] = {{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments = color_attachment_refs,
        .pDepthStencilAttachment = &depth_attachment_ref,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
}};

constexpr vk::RenderPassCreateInfo render_pass_create_info = {
        .attachmentCount = 2,
        .pAttachments = render_pass_attachments,
        .subpassCount = 1,
        .pSubpasses = subpasses,
        .dependencyCount = 0,
};

//
struct pipeline_states {
    vk::DescriptorSetLayout frame_uniforms_set_layout;
    vk::DescriptorSetLayout object_uniforms_set_layout;
    vk::Pipeline pipeline;
    vk::RenderPass render_pass;
};

std::vector<uint32_t> read_spirv_binary(const std::filesystem::path& file_path) {
    std::ifstream file{file_path, std::ios::ate | std::ios::binary};
    if (!file) { throw std::runtime_error{"failed to open file SPIR-V file"}; }
    std::size_t file_size = (std::size_t) file.tellg();
    std::vector<uint32_t> buffer;
    buffer.resize(file_size / 4);
    file.seekg(0);
    file.read((char*) buffer.data(), file_size);
    return buffer;
}

const std::filesystem::path project_root_path = "../../../";  // TODO
const std::filesystem::path spirv_path = ".";  // TODO

vk::ShaderModule create_shader_module_from_spirv(
        vk::Device vk_device, std::span<const uint32_t> spirv) {
    const vk::ShaderModuleCreateInfo smci = {.codeSize = spirv.size_bytes(), .pCode = spirv.data()};

    return vk_device.createShaderModule(smci);
}

vk::ShaderModule create_shader_module_from_spirv_file(
        vk::Device vk_device, std::filesystem::path relative_path) {
    auto spv = read_spirv_binary(spirv_path / relative_path);
    return create_shader_module_from_spirv(vk_device, std::span{spv});
}

pipeline_states init_pipeline(device& dev) {
    const auto vk_device = dev.get_vk_device();
    // create descriptor set layouts
    const auto frame_uniforms_set_layout =
            vk_device.createDescriptorSetLayout(frame_uniforms_descriptor_set_layout_create_info);
    const auto object_uniforms_set_layout =
            vk_device.createDescriptorSetLayout(object_uniforms_descriptor_set_layout_create_info);

    // create pipeline layout
    const vk::DescriptorSetLayout set_layouts[] = {
            frame_uniforms_set_layout, object_uniforms_set_layout};
    const vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
            .setLayoutCount = 2, .pSetLayouts = set_layouts};
    const auto pipeline_layout = vk_device.createPipelineLayout(pipeline_layout_create_info);

    // create shader modules
    const auto vertex_shader_module =
            create_shader_module_from_spirv_file(vk_device, "data/shaders/background.vert.spv");
    const auto fragment_shader_module =
            create_shader_module_from_spirv_file(vk_device, "data/shaders/background.frag.spv");

    const vk::PipelineShaderStageCreateInfo stages[] = {
            {.stage = vk::ShaderStageFlagBits::eVertex,
                    .module = vertex_shader_module,
                    .pName = "main"},
            {.stage = vk::ShaderStageFlagBits::eFragment,
                    .module = fragment_shader_module,
                    .pName = "main"},
    };

    // create render pass
    const auto render_pass = vk_device.createRenderPass(render_pass_create_info);

    // create pipeline
    vk::GraphicsPipelineCreateInfo graphics_pipeline_create_info = {
            .stageCount = 2,
            .pStages = stages,
            .pVertexInputState = &vtx_input_state,
            .pInputAssemblyState = &input_assembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization_state,
            .pMultisampleState = &multisample_state,
            .pDepthStencilState = &depth_stencil_state,
            .pColorBlendState = &color_blend_state,
            .pDynamicState = &dynamic_state_create_info,
            .layout = pipeline_layout,
            .renderPass = render_pass,
            .subpass = 0,
    };
    const auto pipeline = vk_device.createGraphicsPipeline(nullptr, graphics_pipeline_create_info);

    return pipeline_states{
            .frame_uniforms_set_layout = frame_uniforms_set_layout,
            .object_uniforms_set_layout = object_uniforms_set_layout,
            .pipeline = pipeline.value,
            .render_pass = render_pass,
    };
}

inline constexpr image_usage default_image_usage =
        image_usage::color_attachment | image_usage::sampled | image_usage::input_attachment
        | image_usage::storage | image_usage::transfer_dst;

#define CIMG(NAME)                                                                       \
    image NAME{dev, default_image_usage, image_format::r16g16_sfloat, range{1280, 720}}; \
    NAME.set_name(#NAME);

#define VIMG(NAME)                                                                       \
    image NAME{dev, default_image_usage, image_format::r16g16_sfloat, range{1280, 720}}; \
    NAME.set_name(#NAME);

#define COMPUTE_READ(img)                                    \
    h.add_image_access(img, vk::AccessFlagBits::eShaderRead, \
            vk::PipelineStageFlagBits::eComputeShader, {}, vk::ImageLayout::eGeneral);

#define COMPUTE_WRITE(img)                                                                        \
    h.add_image_access(img, vk::AccessFlagBits::eShaderWrite,                                     \
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, \
            vk::ImageLayout::eGeneral);

#define SAMPLE(img)                                                                                \
    h.add_image_access(img, vk::AccessFlagBits::eShaderRead,                                       \
            vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader, \
            {}, vk::ImageLayout::eShaderReadOnlyOptimal);

#define DRAW(img)                                                      \
    h.add_image_access(img, vk::AccessFlagBits::eColorAttachmentWrite, \
            vk::PipelineStageFlagBits::eColorAttachmentOutput,         \
            vk::PipelineStageFlagBits::eColorAttachmentOutput,         \
            vk::ImageLayout::eColorAttachmentOptimal);

#define DRAW_SWAPCHAIN(img)                                                \
    h.add_swapchain_access(img, vk::AccessFlagBits::eColorAttachmentWrite, \
            vk::PipelineStageFlagBits::eColorAttachmentOutput,             \
            vk::PipelineStageFlagBits::eColorAttachmentOutput,             \
            vk::ImageLayout::eColorAttachmentOptimal);

void test_shader(device& device);
void test_case_1(graal::queue& q);
void test_parallel_branches(graal::queue& q);
void test_aliasing_pipelining_conflicts(graal::queue& q);

/// @brief Returns the contents of a text file as a string
/// @param path
/// @return
static std::string read_text_file(std::string_view path) {
    std::ifstream file_in{std::string{path}};
    std::string source;

    if (!file_in) { throw std::runtime_error{fmt::format("could not open file `{}` for reading")}; }

    file_in.seekg(0, std::ios::end);
    source.reserve(file_in.tellg());
    file_in.seekg(0, std::ios::beg);

    source.assign(std::istreambuf_iterator<char>{file_in}, std::istreambuf_iterator<char>{});
    return source;
}

struct vertex_2d {
    std::array<float, 2> pos;
    std::array<float, 2> tex;
};

/*template<>
struct vertex_traits<vertex_2d> {
    static constexpr vertex_attribute attributes[] = {
            {data_type::float_, 2, offsetof(vertex_2d, pos)},
            {data_type::float_, 2, offsetof(vertex_2d, tex)}};
};*/

/// @brief Application state
class app_state {
public:
    app_state() {
        reload_shaders();
    }

    void reload_shaders() {
        auto vertex_src = read_text_file("data/shaders/background.vert");
        auto fragment_src = read_text_file("data/shaders/background.frag");
    }

    void setup() {
        // build buffer for screen rect
        const float left = -1.0f;
        const float right = -1.0f;
        const float top = 1.0f;
        const float bottom = 1.0f;

        const vertex_2d data[] = {{{left, top}, {0.0, 0.0}}, {{right, top}, {1.0, 0.0}},
                {{left, bottom}, {0.0, 1.0}}, {{left, bottom}, {0.0, 1.0}},
                {{right, top}, {1.0, 0.0}}, {{right, bottom}, {1.0, 1.0}}};

        // build VAO for 2D vertices
        //vertex_array_builder vao_builder{};
        //vao_builder.set_attributes(0, 0, vertex_traits<vertex_2d>::attributes);
        //vao = vao_builder.get_vertex_array();
    }

private:
    //vertex_array_handle vao;
    //program_handle bg_program;
};

int main() {
    // must split instance and device because of surfaces:
    // - get required instance extensions, provided by GLFW
    // - instance (needs extensions), provided by us
    // - surface (needs instance), provided by GLFW
    // - device (needs surface), provided by us
    // So we must ping-pong information a lot during initialization

    //=======================================================
    // SETUP

    GLFWwindow* window;
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void) io;
    ImGui::StyleColorsDark();

    //=======================================================
    // Vulkan setup

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    // --- create instance
    initialize_vulkan_instance(std::span{glfwExtensions, (size_t) glfwExtensionCount});
    // --- create surface
    const auto surface = create_surface(window);
    // --- create device
    device dev{surface};

    auto vk_device = dev.get_vk_device();

    const vk::PipelineCacheCreateInfo pcci{
            .initialDataSize = 0,
            .pInitialData = nullptr,
    };

    const auto pipeline_cache = vk_device.createPipelineCache(pcci);

    const vk::DescriptorPoolSize descriptor_pool_sizes[] = {
            {.type = vk::DescriptorType::eSampler, .descriptorCount = 128},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 128},
            {.type = vk::DescriptorType::eSampledImage, .descriptorCount = 128},
            {.type = vk::DescriptorType::eStorageImage, .descriptorCount = 128},
            {.type = vk::DescriptorType::eUniformTexelBuffer, .descriptorCount = 128},
            {.type = vk::DescriptorType::eStorageTexelBuffer, .descriptorCount = 128},
            {.type = vk::DescriptorType::eUniformBufferDynamic, .descriptorCount = 128},
            {.type = vk::DescriptorType::eStorageBufferDynamic, .descriptorCount = 128},
            {.type = vk::DescriptorType::eInputAttachment, .descriptorCount = 128}};
    // big-ass pool
    const vk::DescriptorPoolCreateInfo dpci{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1024,
            .poolSizeCount = 9,
            .pPoolSizes = descriptor_pool_sizes};
    const auto descriptor_pool = vk_device.createDescriptorPool(dpci);

    //=======================================================
    // ImGui setup
    const ImGui_ImplVulkan_InitInfo imgui_vk_init = {
            .Instance = dev.get_vk_instance(),
            .PhysicalDevice = dev.get_vk_physical_device(),
            .Device = dev.get_vk_device(),
            .QueueFamily = (uint32_t) dev.get_graphics_queue_family(),
            .Queue = dev.get_graphics_queue(),
            .PipelineCache = pipeline_cache,
            .DescriptorPool = descriptor_pool,
            .MinImageCount = 2,
    };

    //=======================================================
    queue q{dev};

    test_case_1(q);
    test_parallel_branches(q);
    test_aliasing_pipelining_conflicts(q);

    load_texture(q, project_root_path / "data/images/El4KUGDU0AAW64U.jpg");

    auto pp_state = init_pipeline(dev);

    //=======================================================
    //  Main loop
    bool reload_shaders = true;
    float clear_color[4] = {0.0, 0.0, 0.0, 1.0};

    // --- initial swapchain
    int prev_display_w = 0, prev_display_h = 0;
    glfwGetFramebufferSize(window, &prev_display_w, &prev_display_h);
    swapchain swapchain{dev,
            {static_cast<size_t>(prev_display_w), static_cast<size_t>(prev_display_h)}, surface};

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        if (display_w != prev_display_w || display_h != prev_display_h) {
            prev_display_w = display_w;
            prev_display_h = display_h;
            swapchain.resize(
                    {static_cast<size_t>(display_w), static_cast<size_t>(display_h)}, surface);
        }

        {
            attachment color[] = {
                    attachment{swapchain, attachment_load_op::clear, attachment_store_op::store}};
            render_pass_desc pass_desc{
                    .color_attachments = color,
            };

            q.render_pass(pass_desc, [&](handler& h) {
                DRAW_SWAPCHAIN(swapchain)
                return [](vk::RenderPass, vk::CommandBuffer) {};
            });
        }

        q.present(swapchain);
        q.enqueue_pending_tasks();

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

//=============================================================================

void draw_frame(graal::queue& q) {
}

namespace graal {}  // namespace graal

//=============================================================================
void test_case_1(graal::queue& q) {
    graal::device dev = q.get_device();
    VIMG(A)
    VIMG(B)
    VIMG(C)
    VIMG(D1)
    VIMG(D2)
    VIMG(E)
    VIMG(F)
    VIMG(G)
    VIMG(H)
    VIMG(I)
    VIMG(J)
    VIMG(K)

    {
        attachment color[] = {attachment{A, attachment_load_op::clear, attachment_store_op::store}};
        render_pass_desc pass_desc{
                .color_attachments = color,
        };

        q.render_pass("T0", pass_desc, [&](handler& h) {
            DRAW(A);
            return [](vk::RenderPass, vk::CommandBuffer) {};
        });
    }

    {
        attachment color[] = {attachment{B, attachment_load_op::clear, attachment_store_op::store}};
        render_pass_desc pass_desc{
                .color_attachments = color,
        };

        q.render_pass("T1", pass_desc, [&](handler& h) {
            DRAW(B);
            return [](vk::RenderPass, vk::CommandBuffer) {};
        });
    }

    q.compute_pass("T2", [&](handler& h) {
        COMPUTE_READ(A)
        COMPUTE_READ(B)
        COMPUTE_WRITE(D1)
        COMPUTE_WRITE(D2)
        return [](vk::CommandBuffer) {};
    });

    {
        attachment color[] = {attachment{C, attachment_load_op::clear, attachment_store_op::store}};
        render_pass_desc pass_desc{
                .color_attachments = color,
        };

        q.render_pass("T3", pass_desc, [&](handler& h) {
            DRAW(C);
            return [](vk::RenderPass, vk::CommandBuffer) {};
        });
    }

    q.compute_pass_async("T4", [&](handler& h) {
        COMPUTE_READ(D2)
        COMPUTE_READ(C)
        COMPUTE_WRITE(E)
        return [](vk::CommandBuffer) {};
    });

    q.compute_pass("T5", [&](handler& h) {
        COMPUTE_READ(D1)
        COMPUTE_WRITE(F)
        return [](vk::CommandBuffer) {};
    });

    q.compute_pass("T6", [&](handler& h) {
        COMPUTE_READ(E)
        COMPUTE_READ(F)
        COMPUTE_WRITE(G)
        return [](vk::CommandBuffer) {};
    });

    q.compute_pass("T7", [&](handler& h) {
        COMPUTE_READ(G) COMPUTE_WRITE(H) return [](vk::CommandBuffer) {};
    });
    q.compute_pass("T8", [&](handler& h) {
        COMPUTE_READ(H) COMPUTE_WRITE(I) return [](vk::CommandBuffer) {};
    });
    q.compute_pass("T9", [&](handler& h) {
        COMPUTE_READ(I) COMPUTE_READ(G) COMPUTE_WRITE(J) return [](vk::CommandBuffer) {};
    });
    q.compute_pass("T10", [&](handler& h) {
        COMPUTE_READ(J) COMPUTE_WRITE(K) return [](vk::CommandBuffer) {};
    });

    /*A.discard();
  B.discard();
  C.discard();
  D1.discard();
  D2.discard();
  E.discard();
  F.discard();
  G.discard();
  H.discard();
  I.discard();
  J.discard();*/

    q.enqueue_pending_tasks();

    q.compute_pass("T10", [&](handler& h) {
        // color_attachment_accessor write_K{K, discard, h};
        // sampled_image_accessor sample_I{ I, h };
        return [](vk::CommandBuffer) {};
    });
    q.enqueue_pending_tasks();
}

void test_parallel_branches(queue& q) {
    device dev = q.get_device();

    VIMG(S);
    VIMG(C);

    {
        VIMG(A1);
        VIMG(A2);
        VIMG(A3);
        VIMG(B1);
        VIMG(B2);
        VIMG(B3);
        q.compute_pass([&](handler& h) {
            COMPUTE_WRITE(S)
            return [](vk::CommandBuffer) {};
        });
        // A branch
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(S) COMPUTE_WRITE(A1) return [](vk::CommandBuffer) {};
        });
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(A1) COMPUTE_WRITE(A2) return [](vk::CommandBuffer) {};
        });
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(A2) COMPUTE_WRITE(A3) return [](vk::CommandBuffer) {};
        });
        // B branch
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(S) COMPUTE_WRITE(B1) return [](vk::CommandBuffer) {};
        });
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(B1) COMPUTE_WRITE(B2) return [](vk::CommandBuffer) {};
        });
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(B2) COMPUTE_WRITE(B3) return [](vk::CommandBuffer) {};
        });
        // C
        q.compute_pass([&](handler& h) {
            COMPUTE_READ(A3) COMPUTE_READ(B3) COMPUTE_WRITE(C) return [](vk::CommandBuffer) {};
        });
    }
    q.enqueue_pending_tasks();
}

void test_aliasing_pipelining_conflicts(graal::queue& q) {
    device dev = q.get_device();

    {
        VIMG(A);
        VIMG(B);
        VIMG(C);
        VIMG(D);

        // B must not alias with C or D

        {
            attachment color[] = {
                    attachment{B, attachment_load_op::clear, attachment_store_op::store}};
            render_pass_desc pass_desc{
                    .color_attachments = color,
            };

            q.render_pass(pass_desc, [&](handler& h) {
                h.add_image_access(A, vk::AccessFlagBits::eShaderWrite,
                        vk::PipelineStageFlagBits::eVertexShader,
                        vk::PipelineStageFlagBits::eVertexShader, vk::ImageLayout::eGeneral);
                h.add_image_access(B, vk::AccessFlagBits::eColorAttachmentWrite,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput,
                        vk::ImageLayout::eColorAttachmentOptimal);
                return [](vk::RenderPass, vk::CommandBuffer) {};
            });
        }

        q.compute_pass([&](handler& h) {
            h.add_image_access(A, vk::AccessFlagBits::eShaderRead,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader, vk::ImageLayout::eGeneral);

            h.add_image_access(C, vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader, vk::ImageLayout::eGeneral);
            return [](vk::CommandBuffer) {};
        });

        q.compute_pass([&](handler& h) {
            h.add_image_access(C, vk::AccessFlagBits::eShaderRead,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader, vk::ImageLayout::eGeneral);

            h.add_image_access(D, vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader, vk::ImageLayout::eGeneral);
            return [](vk::CommandBuffer) {};
        });
    }

    q.enqueue_pending_tasks();
}