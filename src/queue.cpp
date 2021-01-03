#include <graal/detail/command_buffer_pool.hpp>
#include <graal/detail/pipeline_stage_tracker.hpp>
#include <graal/detail/staging_pool.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/detail/task.hpp>
#include <graal/queue.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <fstream>
#include <numeric>
#include <queue>
#include <span>

namespace graal {
namespace detail {
namespace {

inline constexpr bool is_read_access(vk::AccessFlags flags) {
    return !!(flags
              & (vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eIndexRead
                      | vk::AccessFlagBits::eVertexAttributeRead | vk::AccessFlagBits::eUniformRead
                      | vk::AccessFlagBits::eInputAttachmentRead | vk::AccessFlagBits::eShaderRead
                      | vk::AccessFlagBits::eColorAttachmentRead
                      | vk::AccessFlagBits::eDepthStencilAttachmentRead
                      | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eHostRead
                      | vk::AccessFlagBits::eMemoryRead
                      | vk::AccessFlagBits::eTransformFeedbackCounterReadEXT
                      | vk::AccessFlagBits::eConditionalRenderingReadEXT
                      | vk::AccessFlagBits::eColorAttachmentReadNoncoherentEXT
                      | vk::AccessFlagBits::eAccelerationStructureReadKHR
                      | vk::AccessFlagBits::eShadingRateImageReadNV
                      | vk::AccessFlagBits::eFragmentDensityMapReadEXT
                      | vk::AccessFlagBits::eCommandPreprocessReadNV
                      | vk::AccessFlagBits::eAccelerationStructureReadNV));
}

inline constexpr bool is_write_access(vk::AccessFlags flags) {
    return !!(flags
              & (vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eColorAttachmentWrite
                      | vk::AccessFlagBits::eDepthStencilAttachmentWrite
                      | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eHostWrite
                      | vk::AccessFlagBits::eMemoryWrite
                      | vk::AccessFlagBits::eTransformFeedbackWriteEXT
                      | vk::AccessFlagBits::eTransformFeedbackCounterWriteEXT
                      | vk::AccessFlagBits::eAccelerationStructureWriteKHR
                      | vk::AccessFlagBits::eCommandPreprocessWriteNV));
}

/// Converts a vector of intrusive_ptr to a vector of raw pointers.
template<typename T>
std::vector<T*> to_raw_ptr_vector(const std::vector<std::shared_ptr<T>>& v) {
    std::vector<T*> result;
    std::transform(
            v.begin(), v.end(), std::back_inserter(result), [](auto&& x) { return x.get(); });
    return result;
}

bool adjust_allocation_requirements(
        allocation_requirements& req_a, const allocation_requirements& req_b) {
    if (req_a.required_flags != req_b.required_flags) return false;
    // XXX what about preferred flags?
    // don't alias if the memory type bits are not strictly the same
    if (req_a.memreq.memoryTypeBits != req_b.memreq.memoryTypeBits) return false;

    req_a.memreq.alignment = std::max(req_a.memreq.alignment, req_b.memreq.alignment);
    req_a.memreq.size = std::max(req_a.memreq.size, req_b.memreq.size);
    return true;
}

/// @brief A reference to a resource used in the batch (called "temporary" for historical reasons).
/// Also tracks the current layout and accesses during submission.
struct temporary {
    /// @brief Constructs a temporary from an existing resource
    /// @param device
    /// @param resource
    explicit temporary(resource_ptr r) : resource{std::move(r)} {
    }

    resource_ptr resource;
};

std::string_view find_image_name(std::span<const temporary> temps, vk::Image image) {
    for (auto&& t : temps) {
        if (auto img = t.resource->as_image(); img && img->vk_image() == image) {
            return t.resource->name();
        }
    }
    return "<unknown>";
}

std::string pipeline_stages_to_string_compact(vk::PipelineStageFlags value) {
    if (!value) return "{}";
    std::string result;
    if (value & vk::PipelineStageFlagBits::eTopOfPipe) result += "TOP_OF_PIPE,";
    if (value & vk::PipelineStageFlagBits::eDrawIndirect) result += "DI,";
    if (value & vk::PipelineStageFlagBits::eVertexInput) result += "VI,";
    if (value & vk::PipelineStageFlagBits::eVertexShader) result += "VS,";
    if (value & vk::PipelineStageFlagBits::eTessellationControlShader) result += "TCS,";
    if (value & vk::PipelineStageFlagBits::eTessellationEvaluationShader) result += "TES,";
    if (value & vk::PipelineStageFlagBits::eGeometryShader) result += "GS,";
    if (value & vk::PipelineStageFlagBits::eFragmentShader) result += "FS,";
    if (value & vk::PipelineStageFlagBits::eEarlyFragmentTests) result += "EFT,";
    if (value & vk::PipelineStageFlagBits::eLateFragmentTests) result += "LFT,";
    if (value & vk::PipelineStageFlagBits::eColorAttachmentOutput) result += "CAO,";
    if (value & vk::PipelineStageFlagBits::eComputeShader) result += "CS,";
    if (value & vk::PipelineStageFlagBits::eTransfer) result += "TR,";
    if (value & vk::PipelineStageFlagBits::eBottomOfPipe) result += "BOTTOM,";
    if (value & vk::PipelineStageFlagBits::eHost) result += "HST,";
    if (value & vk::PipelineStageFlagBits::eAllGraphics) result += "ALL_GRAPHICS,";
    if (value & vk::PipelineStageFlagBits::eAllCommands) result += "ALL_COMMANDS,";
    if (value & vk::PipelineStageFlagBits::eTransformFeedbackEXT) result += "TF,";
    if (value & vk::PipelineStageFlagBits::eConditionalRenderingEXT) result += "CR,";
    if (value & vk::PipelineStageFlagBits::eRayTracingShaderKHR) result += "RTS,";
    if (value & vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR) result += "ASB,";
    if (value & vk::PipelineStageFlagBits::eShadingRateImageNV) result += "FSR,";
    if (value & vk::PipelineStageFlagBits::eTaskShaderNV) result += "TS,";
    if (value & vk::PipelineStageFlagBits::eMeshShaderNV) result += "MS,";
    if (value & vk::PipelineStageFlagBits::eFragmentDensityProcessEXT) result += "FDP,";
    if (value & vk::PipelineStageFlagBits::eCommandPreprocessNV) result += "CPR,";
    return result.substr(0, result.size() - 1);
}

std::string pipeline_stages_to_string(vk::PipelineStageFlags value) {
    if (!value) return "empty";
    std::string result;
    if (value & vk::PipelineStageFlagBits::eTopOfPipe) result += "TOP_OF_PIPE|";
    if (value & vk::PipelineStageFlagBits::eDrawIndirect) result += "DRAW_INDIRECT|";
    if (value & vk::PipelineStageFlagBits::eVertexInput) result += "VERTEX_INPUT|";
    if (value & vk::PipelineStageFlagBits::eVertexShader) result += "VERTEX_SHADER|";
    if (value & vk::PipelineStageFlagBits::eTessellationControlShader)
        result += "TESSELLATION_CONTROL_SHADER|";
    if (value & vk::PipelineStageFlagBits::eTessellationEvaluationShader)
        result += "TESSELLATION_EVALUATION_SHADER|";
    if (value & vk::PipelineStageFlagBits::eGeometryShader) result += "GEOMETRY_SHADER|";
    if (value & vk::PipelineStageFlagBits::eFragmentShader) result += "FRAGMENT_SHADER|";
    if (value & vk::PipelineStageFlagBits::eEarlyFragmentTests) result += "EARLY_FRAGMENT_TESTS|";
    if (value & vk::PipelineStageFlagBits::eLateFragmentTests) result += "LATE_FRAGMENT_TESTS|";
    if (value & vk::PipelineStageFlagBits::eColorAttachmentOutput)
        result += "COLOR_ATTACHMENT_OUTPUT|";
    if (value & vk::PipelineStageFlagBits::eComputeShader) result += "COMPUTE_SHADER|";
    if (value & vk::PipelineStageFlagBits::eTransfer) result += "TRANSFER|";
    if (value & vk::PipelineStageFlagBits::eBottomOfPipe) result += "BOTTOM_OF_PIPE|";
    if (value & vk::PipelineStageFlagBits::eHost) result += "HOST|";
    if (value & vk::PipelineStageFlagBits::eAllGraphics) result += "ALL_GRAPHICS|";
    if (value & vk::PipelineStageFlagBits::eAllCommands) result += "ALL_COMMANDS|";
    if (value & vk::PipelineStageFlagBits::eTransformFeedbackEXT)
        result += "TRANSFORM_FEEDBACK_EXT|";
    if (value & vk::PipelineStageFlagBits::eConditionalRenderingEXT)
        result += "CONDITIONAL_RENDERING_EXT|";
    if (value & vk::PipelineStageFlagBits::eRayTracingShaderKHR)
        result += "ACCELERATION_STRUCTURE_BUILD_KHR|";
    if (value & vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR)
        result += "RAY_TRACING_SHADER_KHR|";
    if (value & vk::PipelineStageFlagBits::eShadingRateImageNV) result += "SHADING_RATE_IMAGE_NV|";
    if (value & vk::PipelineStageFlagBits::eTaskShaderNV) result += "TASK_SHADER_NV|";
    if (value & vk::PipelineStageFlagBits::eMeshShaderNV) result += "MESH_SHADER_NV|";
    if (value & vk::PipelineStageFlagBits::eFragmentDensityProcessEXT)
        result += "FRAGMENT_DENSITY_PROCESS_EXT|";
    if (value & vk::PipelineStageFlagBits::eCommandPreprocessNV) result += "COMMAND_PREPROCESS_NV|";
    return result.substr(0, result.size() - 1);
}

std::string access_mask_to_string(vk::AccessFlags value) {
    if (!value) return "empty";
    std::string result;
    if (value & vk::AccessFlagBits::eIndirectCommandRead) result += "INDIRECT_COMMAND_READ|";
    if (value & vk::AccessFlagBits::eIndexRead) result += "INDEX_READ|";
    if (value & vk::AccessFlagBits::eVertexAttributeRead) result += "VERTEX_ATTRIBUTE_READ|";
    if (value & vk::AccessFlagBits::eUniformRead) result += "UNIFORM_READ|";
    if (value & vk::AccessFlagBits::eInputAttachmentRead) result += "INPUT_ATTACHMENT_READ|";
    if (value & vk::AccessFlagBits::eShaderRead) result += "SHADER_READ|";
    if (value & vk::AccessFlagBits::eShaderWrite) result += "SHADER_WRITE|";
    if (value & vk::AccessFlagBits::eColorAttachmentRead) result += "COLOR_ATTACHMENT_READ|";
    if (value & vk::AccessFlagBits::eColorAttachmentWrite) result += "COLOR_ATTACHMENT_WRITE|";
    if (value & vk::AccessFlagBits::eDepthStencilAttachmentRead)
        result += "DEPTH_STENCIL_ATTACHMENT_READ|";
    if (value & vk::AccessFlagBits::eDepthStencilAttachmentWrite)
        result += "DEPTH_STENCIL_ATTACHMENT_WRITE|";
    if (value & vk::AccessFlagBits::eTransferRead) result += "TRANSFER_READ|";
    if (value & vk::AccessFlagBits::eTransferWrite) result += "TRANSFER_WRITE|";
    if (value & vk::AccessFlagBits::eHostRead) result += "HOST_READ|";
    if (value & vk::AccessFlagBits::eHostWrite) result += "HOST_WRITE|";
    if (value & vk::AccessFlagBits::eMemoryRead) result += "MEMORY_READ|";
    if (value & vk::AccessFlagBits::eMemoryWrite) result += "MEMORY_WRITE|";
    if (value & vk::AccessFlagBits::eTransformFeedbackWriteEXT)
        result += "TRANSFORM_FEEDBACK_WRITE_EXT|";
    if (value & vk::AccessFlagBits::eTransformFeedbackCounterReadEXT)
        result += "TRANSFORM_FEEDBACK_COUNTER_READ_EXT|";
    if (value & vk::AccessFlagBits::eTransformFeedbackCounterWriteEXT)
        result += "TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT|";
    if (value & vk::AccessFlagBits::eConditionalRenderingReadEXT)
        result += "CONDITIONAL_RENDERING_READ_EXT|";
    if (value & vk::AccessFlagBits::eColorAttachmentReadNoncoherentEXT)
        result += "COLOR_ATTACHMENT_READ_NONCOHERENT_EXT|";
    if (value & vk::AccessFlagBits::eAccelerationStructureReadKHR)
        result += "ACCELERATION_STRUCTURE_READ_KHR|";
    if (value & vk::AccessFlagBits::eAccelerationStructureWriteKHR)
        result += "ACCELERATION_STRUCTURE_WRITE_KHR|";
    if (value & vk::AccessFlagBits::eShadingRateImageReadNV)
        result += "SHADING_RATE_IMAGE_READ_NV|";
    if (value & vk::AccessFlagBits::eFragmentDensityMapReadEXT)
        result += "FRAGMENT_DENSITY_MAP_READ_EXT|";
    if (value & vk::AccessFlagBits::eCommandPreprocessReadNV)
        result += "COMMAND_PREPROCESS_READ_NV|";
    if (value & vk::AccessFlagBits::eCommandPreprocessWriteNV)
        result += "COMMAND_PREPROCESS_WRITE_NV|";
    return result.substr(0, result.size() - 1);
}

std::string access_mask_to_string_compact(vk::AccessFlags value) {
    if (!value) return "{}";
    std::string result;
    if (value & vk::AccessFlagBits::eIndirectCommandRead) result += "ICr,";
    if (value & vk::AccessFlagBits::eIndexRead) result += "IXr,";
    if (value & vk::AccessFlagBits::eVertexAttributeRead) result += "VAr";
    if (value & vk::AccessFlagBits::eUniformRead) result += "Ur,";
    if (value & vk::AccessFlagBits::eInputAttachmentRead) result += "IAr,";
    if (value & vk::AccessFlagBits::eShaderRead) result += "SHr,";
    if (value & vk::AccessFlagBits::eShaderWrite) result += "SHw,";
    if (value & vk::AccessFlagBits::eColorAttachmentRead) result += "CAr,";
    if (value & vk::AccessFlagBits::eColorAttachmentWrite) result += "CAw,";
    if (value & vk::AccessFlagBits::eDepthStencilAttachmentRead) result += "DSAr,";
    if (value & vk::AccessFlagBits::eDepthStencilAttachmentWrite) result += "DSAw,";
    if (value & vk::AccessFlagBits::eTransferRead) result += "Tr,";
    if (value & vk::AccessFlagBits::eTransferWrite) result += "Tw,";
    if (value & vk::AccessFlagBits::eHostRead) result += "Hr,";
    if (value & vk::AccessFlagBits::eHostWrite) result += "Hw,";
    if (value & vk::AccessFlagBits::eMemoryRead) result += "Mr,";
    if (value & vk::AccessFlagBits::eMemoryWrite) result += "Mw,";
    if (value & vk::AccessFlagBits::eTransformFeedbackWriteEXT) result += "TFw,";
    if (value & vk::AccessFlagBits::eTransformFeedbackCounterReadEXT) result += "TFCr,";
    if (value & vk::AccessFlagBits::eTransformFeedbackCounterWriteEXT) result += "TFCw,";
    if (value & vk::AccessFlagBits::eConditionalRenderingReadEXT) result += "CRr,";
    if (value & vk::AccessFlagBits::eColorAttachmentReadNoncoherentEXT) result += "CANCr,";
    if (value & vk::AccessFlagBits::eAccelerationStructureReadKHR) result += "ASr,";
    if (value & vk::AccessFlagBits::eAccelerationStructureWriteKHR) result += "ASw,";
    if (value & vk::AccessFlagBits::eShadingRateImageReadNV) result += "SRIr,";
    if (value & vk::AccessFlagBits::eFragmentDensityMapReadEXT) result += "FDMr,";
    if (value & vk::AccessFlagBits::eCommandPreprocessReadNV) result += "CPr,";
    if (value & vk::AccessFlagBits::eCommandPreprocessWriteNV) result += "CPw,";
    return result.substr(0, result.size() - 1);
}

std::string_view layout_to_string(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined: return "UNDEFINED";
        case vk::ImageLayout::eGeneral: return "GENERAL";
        case vk::ImageLayout::eColorAttachmentOptimal: return "COLOR_ATTACHMENT_OPTIMAL";
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
        case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
            return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
        case vk::ImageLayout::eShaderReadOnlyOptimal: return "SHADER_READ_ONLY_OPTIMAL";
        case vk::ImageLayout::eTransferSrcOptimal: return "TRANSFER_SRC_OPTIMAL";
        case vk::ImageLayout::eTransferDstOptimal: return "TRANSFER_DST_OPTIMAL";
        case vk::ImageLayout::ePreinitialized: return "PREINITIALIZED";
        case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal: return "DROSA";
        case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal: return "DASRO";
        case vk::ImageLayout::eDepthAttachmentOptimal: return "DA";
        case vk::ImageLayout::eDepthReadOnlyOptimal: return "DRO";
        case vk::ImageLayout::eStencilAttachmentOptimal: return "SAO";
        case vk::ImageLayout::eStencilReadOnlyOptimal: return "SRO";
        case vk::ImageLayout::ePresentSrcKHR: return "PRESENT_KHR";
        case vk::ImageLayout::eSharedPresentKHR: return "SHARED_PRESENT_KHR";
        case vk::ImageLayout::eShadingRateOptimalNV: return "SR";
        case vk::ImageLayout::eFragmentDensityMapOptimalEXT: return "FDM";
        default: return "<invalid>";
    }
}

std::string_view layout_to_string_compact(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined: return "UND";
        case vk::ImageLayout::eGeneral: return "GEN";
        case vk::ImageLayout::eColorAttachmentOptimal: return "CA";
        case vk::ImageLayout::eDepthStencilAttachmentOptimal: return "DSA";
        case vk::ImageLayout::eDepthStencilReadOnlyOptimal: return "DSRO";
        case vk::ImageLayout::eShaderReadOnlyOptimal: return "SHRO";
        case vk::ImageLayout::eTransferSrcOptimal: return "TSRC";
        case vk::ImageLayout::eTransferDstOptimal: return "TDST";
        case vk::ImageLayout::ePreinitialized: return "PREINIT";
        case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal: return "DROSA";
        case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal: return "DASRO";
        case vk::ImageLayout::eDepthAttachmentOptimal: return "DA";
        case vk::ImageLayout::eDepthReadOnlyOptimal: return "DRO";
        case vk::ImageLayout::eStencilAttachmentOptimal: return "SAO";
        case vk::ImageLayout::eStencilReadOnlyOptimal: return "SRO";
        case vk::ImageLayout::ePresentSrcKHR: return "PR";
        case vk::ImageLayout::eSharedPresentKHR: return "SPR";
        case vk::ImageLayout::eShadingRateOptimalNV: return "SR";
        case vk::ImageLayout::eFragmentDensityMapOptimalEXT: return "FDM";
        default: return "<invalid>";
    }
}

using bitset = boost::dynamic_bitset<uint64_t>;

// Computes the transitive closure of the task graph, which tells us
// whether there's a path between two tasks in the graph.
// This is used later during liveness analysis for querying whether two
// tasks can be run in parallel (if there exists no path between two tasks,
// then they can be run in parallel).
std::vector<bitset> transitive_closure(std::span<const task> tasks) {
    std::vector<bitset> m;
    m.resize(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        // tasks already in topological order
        m[i].resize(tasks.size());
        for (auto pred : tasks[i].preds) {
            m[i].set(pred);
            m[i] |= m[pred];
        }
    }
    return m;
}

}  // namespace

template<typename T>
void sorted_vector_insert(std::vector<T>& vec, T elem) {
    auto it = std::lower_bound(vec.begin(), vec.end(), elem);
    vec.insert(it, std::move(elem));
}

static void dump_vector_set(std::span<const serial_number> s) {
    bool first = true;
    for (size_t i = 0; i < s.size(); ++i) {
        if (first) {
            fmt::print("{{{:d}", s[i]);
            first = false;
        } else {
            fmt::print(",{:d}", s[i]);
        }
    }
    if (first) {
        fmt::print("{{}}");
    } else {
        fmt::print("}}");
    }
}

static void dump_vector_set(std::span<const submission_number> s) {
    bool first = true;
    for (size_t i = 0; i < s.size(); ++i) {
        if (first) {
            fmt::print("{{{:d}:{:d}", (uint64_t) s[i].queue, (uint64_t) s[i].serial);
            first = false;
        } else {
            fmt::print(",{:d}:{:d}", (uint64_t) s[i].queue, (uint64_t) s[i].serial);
        }
    }
    if (first) {
        fmt::print("{{}}");
    } else {
        fmt::print("}}");
    }
}

vk::ImageAspectFlags format_aspect_mask(image_format format) noexcept {
    vk::ImageAspectFlags aspect_mask;
    if (is_depth_only_format(format)) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
    } else if (is_stencil_only_format(format)) {
        aspect_mask = vk::ImageAspectFlagBits::eStencil;
    } else if (is_depth_and_stencil_format(format)) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    } else {
        aspect_mask = vk::ImageAspectFlagBits::eColor;
    }
    return aspect_mask;
}

/// @brief Per-batch+thread resources
struct batch_thread_resources {
    command_buffer_pool cb_pool;
};

struct submitted_batch {
    serial_number start_serial = 0;
    /// @brief Serials to wait for
    per_queue_wait_serials end_serials{};
    /// @brief Per-thread resources
    std::vector<command_buffer_pool> cb_pools;
    /// @brief All resources referenced in the batch.
    std::vector<resource_ptr> resources;
    /// @brief Semaphores used within the batch
    std::vector<vk::Semaphore> semaphores;
};

//-----------------------------------------------------------------------------
class queue_impl {
    friend class graal::queue;

public:
    queue_impl(device device, const queue_properties& props);
    ~queue_impl();

    void enqueue_pending_tasks();
    void present(swapchain swapchain);

    void wait_for_batch();

    const task* get_unsubmitted_task(uint64_t sequence_number) const noexcept;
    task* get_unsubmitted_task(uint64_t sequence_number) noexcept {
        return const_cast<task*>(
                static_cast<const queue_impl&>(*this).get_unsubmitted_task(sequence_number));
    }

    // returned reference only valid until next call to create_task or present.
    [[nodiscard]] task& create_render_pass_task(
            std::string_view name, const render_pass_desc& rpd) noexcept {
        return init_task(tasks_.emplace_back(rpd), name, false);
    }

    // returned reference only valid until next call to create_task or present.
    [[nodiscard]] task& create_compute_pass_task(std::string_view name, bool async) noexcept {
        return init_task(tasks_.emplace_back(), name, async);
    }

    void add_task_dependency(
            submission_number before_snn, task& after, vk::PipelineStageFlags wait_dst_stages);

    void add_resource_dependency(
            task& t, resource_ptr resource, const resource_access_details& access);

    [[nodiscard]] void* get_staging_buffer(
            size_t align, size_t size, vk::Buffer& out_buffer, vk::DeviceSize& out_offset) {
        return staging_.get_staging_buffer(device_.get_vk_device(), device_.get_allocator(),
                staging_buffers_, align, size, out_buffer, out_offset);
    }

private:
    task& init_task(task& t, std::string_view name, bool async) {
        last_serial_++;
        t.async = async;
        switch (t.type()) {
            case task_type::render_pass: t.snn.queue = qinfo_.indices.graphics; break;
            case task_type::compute_pass:
                if (t.async) {
                    t.snn.queue = qinfo_.indices.compute;
                } else {
                    t.snn.queue = qinfo_.indices.graphics;
                }
                break;
            case task_type::transfer:
                if (t.async) {
                    t.snn.queue = qinfo_.indices.transfer;
                } else {
                    t.snn.queue = qinfo_.indices.graphics;
                }
                break;
            case task_type::present: t.snn.queue = qinfo_.indices.compute; break;
        }
        t.snn.serial = last_serial_;  // temporary SNN for DAG creation
        if (!name.empty()) {
            t.name = name;
        } else {
            t.name = fmt::format("T_{}", (uint64_t) t.snn.serial);
        }
        return t;
    }

    size_t get_resource_tmp_index(resource_ptr resource);

    [[nodiscard]] command_buffer_pool get_command_buffer_pool(uint32_t queue);
    //[[nodiscard]] vk::CommandBuffer get_command_buffer(uint32_t queue);

    /// @brief Returns whether writes to the specified resource in this batch could be seen
    /// by uses of the resource in subsequent batches.
    /// @return
    [[nodiscard]] bool writes_are_user_observable(resource& r) const noexcept;

    using resource_to_temporary_map = std::unordered_map<resource*, size_t>;

    device device_;
    queue_properties props_;
    queues_info qinfo_;
    vk::Semaphore timelines_[max_queues];
    staging_pool staging_;
    recycler<staging_buffer> staging_buffers_;
    recycler<command_buffer_pool> cb_pools_;

    serial_number last_serial_ = 0;
    serial_number completed_serial_ = 0;

    serial_number batch_start_serial_ = 0;
    std::vector<temporary> temporaries_;
    std::vector<task> tasks_;
    resource_to_temporary_map tmp_indices;  // for the current batch

    std::deque<submitted_batch> in_flight_batches_;
};

queue_impl::queue_impl(device device, const queue_properties& props) :
    device_{std::move(device)}, props_{props}, qinfo_{device_.get_queues_info()} {
    auto vk_device = device_.get_vk_device();

    vk::SemaphoreTypeCreateInfo timeline_create_info{
            .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
    vk::SemaphoreCreateInfo semaphore_create_info{.pNext = &timeline_create_info};
    for (size_t i = 0; i < max_queues; ++i) {
        timelines_[i] = vk_device.createSemaphore(semaphore_create_info);
    }
}

queue_impl::~queue_impl() {
    const auto vk_device = device_.get_vk_device();
    vk_device.waitIdle();
    for (size_t i = 0; i < max_queues; ++i) {
        vk_device.destroySemaphore(timelines_[i]);
    }
}

size_t queue_impl::get_resource_tmp_index(resource_ptr resource) {
    const auto next_tmp_index = temporaries_.size();
    const auto result = tmp_indices.insert({resource.get(), next_tmp_index});
    if (result.second) {
        // init temporary
        temporaries_.emplace_back(std::move(resource));
    }
    return result.first->second;
}

command_buffer_pool queue_impl::get_command_buffer_pool(uint32_t queue) {
    const auto vk_device = device_.get_vk_device();
    const auto queue_family_index = qinfo_.families[queue];
    command_buffer_pool cbp;

    if (!cb_pools_.fetch_if(
                cbp, [=](const auto& cbp) { return cbp.queue_family_index == queue_family_index; })) {
        // TODO other queues?
        vk::CommandPoolCreateInfo create_info{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                .queueFamilyIndex = queue_family_index};
        const auto pool = vk_device.createCommandPool(create_info);
        cbp = command_buffer_pool{
                .queue_family_index = queue_family_index,
                .command_pool = pool,
        };
        fmt::print("created new command buffer pool for queue index {}, family {}\n", queue,
                queue_family_index);
    }
    return cbp;
}

bool queue_impl::writes_are_user_observable(resource& r) const noexcept {
    // writes to the resources might be seen if there are still user handles to the resource
    return r.state.writer.serial > batch_start_serial_ && !r.has_user_refs();
}

const task* queue_impl::get_unsubmitted_task(uint64_t sequence_number) const noexcept {
    if (sequence_number <= batch_start_serial_) { return nullptr; }
    if (sequence_number < last_serial_) {
        const auto i = sequence_number - batch_start_serial_ - 1;
        return &tasks_[i];
    }
    return nullptr;
}

//using task_priority_queue =
//        std::priority_queue<ready_task, std::vector<ready_task>, ready_task::compare>;

static void dump_set(bitset s) {
    bool first = true;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i]) {
            if (first) {
                fmt::print("{{{:02d}", i);
                first = false;
            } else {
                fmt::print(",{:02d}", i);
            }
        }
    }
    if (first) {
        fmt::print("{{}}");
    } else {
        fmt::print("}}");
    }
}

/// @brief Represents a queue operation (either a call to vkQueueSubmit or vkQueuePresent)
struct queue_submission {
    per_queue_wait_serials wait_sn{};
    per_queue_wait_dst_stages wait_dst_stages{};
    submission_number first_snn;
    submission_number signal_snn;
    std::vector<vk_handle<vk::Semaphore>> wait_binary_semaphores;
    std::vector<vk_handle<vk::Semaphore>> signal_binary_semaphores;
    vk_handle<vk::Semaphore> render_finished;
    std::vector<vk::SwapchainKHR> swapchains;
    std::vector<uint32_t> swapchain_image_indices;
    vk::CommandBuffer command_buffer;

    size_t cb_offset = 0;
    size_t cb_count = 0;

    [[nodiscard]] bool is_empty() const noexcept {
        return cb_count == 0 && signal_binary_semaphores.empty() && !render_finished;
    }

    void submit(
        vk::Queue queue, 
        std::span<vk::Semaphore> timelines, 
        per_queue_wait_serials& out_signals,
        std::vector<vk::Semaphore> pending_semaphores)
    {
        const auto queue_index = signal_snn.queue;
        std::vector<vk::Semaphore> signal_semaphores;
        signal_semaphores.push_back(timelines[queue_index]);
        signal_semaphores.insert(signal_semaphores.end(), signal_binary_semaphores.begin(),
                signal_binary_semaphores.end());
        if (render_finished) { 
            signal_semaphores.push_back(render_finished); 
            pending_semaphores.push_back(render_finished);
        }

        std::vector<vk::Semaphore> wait_semaphores;
        std::vector<vk::PipelineStageFlags> wait_semaphore_dst_stages;
        std::vector<uint64_t> wait_semaphore_values;
        for (size_t i = 0; i < wait_sn.size(); ++i) {
            if (wait_sn[i]) {
                wait_semaphores.push_back(timelines[i]);
                wait_semaphore_values.push_back(wait_sn[i]);
                wait_semaphore_dst_stages.push_back(wait_dst_stages[i]);
            }
        }
        for (auto s : wait_binary_semaphores) {
            wait_semaphores.push_back(s);
            wait_semaphore_values.push_back(0); // dummy
            wait_semaphore_dst_stages.push_back(vk::PipelineStageFlagBits::eTopOfPipe); // TODO
        }

        // pSignalSemaphoreValues must contain dummy entries for binary semaphores as well, wtf?
        std::vector<uint64_t> signal_semaphore_values;
        signal_semaphore_values.resize(signal_semaphores.size());
        signal_semaphore_values[0] = signal_snn.serial;

        const vk::TimelineSemaphoreSubmitInfo timeline_submit_info{
                .waitSemaphoreValueCount = (uint32_t) wait_semaphore_values.size(),
                .pWaitSemaphoreValues = wait_semaphore_values.data(),
                // must be equal to submit_info.signalSemaphoreCount (then why the fuck do we need to specify it here?)
                .signalSemaphoreValueCount =
                        (uint32_t)signal_semaphore_values.size(),
                .pSignalSemaphoreValues = signal_semaphore_values.data()};

        vk::SubmitInfo submit_info{.pNext = &timeline_submit_info,
                .waitSemaphoreCount = (uint32_t) wait_semaphores.size(),
                .pWaitSemaphores = wait_semaphores.data(),
                .pWaitDstStageMask = wait_semaphore_dst_stages.data(),
                .commandBufferCount = 1,
                .pCommandBuffers = &command_buffer,
                .signalSemaphoreCount = (uint32_t) signal_semaphores.size(),
                .pSignalSemaphores = signal_semaphores.data()};

        if (auto result = queue.submit(1, &submit_info, nullptr); result != vk::Result::eSuccess) {
            fmt::print("vkQueueSubmit failed: {}\n", result);
        }

        if (render_finished) {
            // terminate with a present
            vk::PresentInfoKHR present_info{.waitSemaphoreCount = 1,
                    .pWaitSemaphores = &render_finished,
                    .swapchainCount = (uint32_t) swapchains.size(),
                    .pSwapchains = swapchains.data(),
                    .pImageIndices = swapchain_image_indices.data()};

            queue.presentKHR(present_info);
        }

        out_signals[signal_snn.queue] = signal_snn.serial;
    }
};

struct device_memory_allocation {
    allocation_requirements req;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
};

struct submission_builder {
    std::array<size_t, max_queues> pending{};
    std::vector<queue_submission> submits;
    std::vector<vk::CommandBuffer> command_buffers;
    size_t cb_count = 0;

    submission_builder() {
        for (auto& p : pending) {
            p = (size_t) -1;
        }
    }

    queue_submission& get_pending(uint32_t queue) {
        if (pending[queue] == -1) {
            submits.emplace_back();
            pending[queue] = submits.size() - 1;
        }
        return submits[pending[queue]];
    }

    void start_batch(uint32_t queue, per_queue_wait_serials wait_sn,
            per_queue_wait_dst_stages wait_dst_stages,
            std::vector<vk::Semaphore> wait_binary_semaphores) {
        for (size_t iq = 0; iq < max_queues; ++iq) {
            if (wait_sn[iq] || iq == queue) { end_batch(iq); }
        }

        auto& qs = get_pending(queue);
        qs.wait_sn = wait_sn;
        qs.wait_dst_stages = wait_dst_stages;
        qs.wait_binary_semaphores = std::move(wait_binary_semaphores);
    }

    void end_batch(uint32_t queue) {
        if (pending[queue] == -1) return;
        auto& qs = get_pending(queue);
        if (qs.is_empty()) { return; }
        qs.cb_offset = cb_count;
        cb_count += qs.cb_count;
        pending[queue] = (size_t) -1;
    }

    void add_task(task& t) {
        auto& qs = get_pending(t.snn.queue);
        if (qs.first_snn.serial == 0) {
            // set submission SN
            qs.first_snn = t.snn;
        }

        qs.signal_snn.serial = std::max(qs.signal_snn.serial, t.snn.serial);
        t.submit_batch_cb_index = qs.cb_count;
        qs.cb_count++;
        t.submit_batch_index = pending[t.snn.queue];
    }

    void end_batch_and_present(uint32_t queue, std::vector<vk::SwapchainKHR> swapchains,
            std::vector<uint32_t> swpachain_image_indices, vk::Semaphore render_finished) {
        auto& qs = get_pending(queue);
        // end the batch with a present (how considerate!)
        qs.render_finished = std::move(render_finished);
        qs.swapchains = std::move(swapchains);
        qs.swapchain_image_indices = std::move(swpachain_image_indices);
        end_batch(queue);
    }

    void finish() {
        for (size_t i = 0; i < max_queues; ++i) {
            end_batch(i);
        }
        command_buffers.resize(cb_count);
    }
};

struct schedule_ctx {
    device& dev;
    queue_indices queues;
    std::span<task> tasks;
    std::span<temporary> temporaries;

    // --- scheduling state
    serial_number base_sn;  // serial number of the first task
    serial_number next_sn;  // serial number of the next task to be scheduled
    bitset done_tasks;  // set of submitted tasks
    bitset ready_tasks;  // set of tasks ready to be submitted

    // --- liveness
    std::vector<bitset> reachability;  // reachability matrix [to][from]
    bitset dead_and_recycled;  // set of dead temporaries after the last scheduled task

    // --- device memory allocations
    std::vector<device_memory_allocation> allocations;
    std::vector<size_t> allocation_map;  // tmp index -> allocations

    // --- queue batches
    submission_builder submission_builder;

    // --- submits
    schedule_ctx(device& d, queue_indices queues, serial_number base_sn, std::span<task> tasks,
            std::span<temporary> temporaries) :
        dev{d},
        queues{queues}, base_sn{base_sn}, next_sn{base_sn}, tasks{tasks}, temporaries{temporaries} {
        const auto n_task = tasks.size();
        const auto n_tmp = temporaries.size();
        reachability = transitive_closure(tasks);
        done_tasks.resize(n_task, false);
        ready_tasks.resize(n_task, false);
        allocation_map.resize(n_tmp, (size_t) -1);
        dead_and_recycled.resize(n_tmp);
    }

    void dump() {
        /*fmt::print("Scheduling state dump:\nnext_sn={}\n", next_sn);
        fmt::print("Pipeline:\n");
        fmt::print("{:4d} {:4d} {:4d} {:4d} | DRAW_INDIRECT\n",
                pstate[0].stages[pipeline_stage_tracker::i_DI],
                pstate[1].stages[pipeline_stage_tracker::i_DI],
                pstate[2].stages[pipeline_stage_tracker::i_DI],
                pstate[3].stages[pipeline_stage_tracker::i_DI]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | VERTEX_INPUT\n",
                pstate[0].stages[pipeline_stage_tracker::i_VI],
                pstate[1].stages[pipeline_stage_tracker::i_VI],
                pstate[2].stages[pipeline_stage_tracker::i_VI],
                pstate[3].stages[pipeline_stage_tracker::i_VI]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | VERTEX_SHADER\n",
                pstate[0].stages[pipeline_stage_tracker::i_VS],
                pstate[1].stages[pipeline_stage_tracker::i_VS],
                pstate[2].stages[pipeline_stage_tracker::i_VS],
                pstate[3].stages[pipeline_stage_tracker::i_VS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | TESSELLATION_CONTROL_SHADER\n",
                pstate[0].stages[pipeline_stage_tracker::i_TCS],
                pstate[1].stages[pipeline_stage_tracker::i_TCS],
                pstate[2].stages[pipeline_stage_tracker::i_TCS],
                pstate[3].stages[pipeline_stage_tracker::i_TCS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | TESSELLATION_EVALUATION_SHADER\n",
                pstate[0].stages[pipeline_stage_tracker::i_TES],
                pstate[1].stages[pipeline_stage_tracker::i_TES],
                pstate[2].stages[pipeline_stage_tracker::i_TES],
                pstate[3].stages[pipeline_stage_tracker::i_TES]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | GEOMETRY_SHADER\n",
                pstate[0].stages[pipeline_stage_tracker::i_GS],
                pstate[1].stages[pipeline_stage_tracker::i_GS],
                pstate[2].stages[pipeline_stage_tracker::i_GS],
                pstate[3].stages[pipeline_stage_tracker::i_GS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | TRANSFORM_FEEDBACK_EXT\n",
                pstate[0].stages[pipeline_stage_tracker::i_TF],
                pstate[1].stages[pipeline_stage_tracker::i_TF],
                pstate[2].stages[pipeline_stage_tracker::i_TF],
                pstate[3].stages[pipeline_stage_tracker::i_TF]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | TASK_SHADER_NV\n",
                pstate[0].stages[pipeline_stage_tracker::i_TS],
                pstate[1].stages[pipeline_stage_tracker::i_TS],
                pstate[2].stages[pipeline_stage_tracker::i_TS],
                pstate[3].stages[pipeline_stage_tracker::i_TS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | MESH_SHADER_NV\n",
                pstate[0].stages[pipeline_stage_tracker::i_MS],
                pstate[1].stages[pipeline_stage_tracker::i_MS],
                pstate[2].stages[pipeline_stage_tracker::i_MS],
                pstate[3].stages[pipeline_stage_tracker::i_MS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | SHADING_RATE_IMAGE_NV\n",
                pstate[0].stages[pipeline_stage_tracker::i_FSR],
                pstate[1].stages[pipeline_stage_tracker::i_FSR],
                pstate[2].stages[pipeline_stage_tracker::i_FSR],
                pstate[3].stages[pipeline_stage_tracker::i_FSR]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | EARLY_FRAGMENT_TESTS\n",
                pstate[0].stages[pipeline_stage_tracker::i_EFT],
                pstate[1].stages[pipeline_stage_tracker::i_EFT],
                pstate[2].stages[pipeline_stage_tracker::i_EFT],
                pstate[3].stages[pipeline_stage_tracker::i_EFT]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | FRAGMENT_SHADER\n",
                pstate[0].stages[pipeline_stage_tracker::i_FS],
                pstate[1].stages[pipeline_stage_tracker::i_FS],
                pstate[2].stages[pipeline_stage_tracker::i_FS],
                pstate[3].stages[pipeline_stage_tracker::i_FS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | LATE_FRAGMENT_TESTS\n",
                pstate[0].stages[pipeline_stage_tracker::i_LFT],
                pstate[1].stages[pipeline_stage_tracker::i_LFT],
                pstate[2].stages[pipeline_stage_tracker::i_LFT],
                pstate[3].stages[pipeline_stage_tracker::i_LFT]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | COLOR_ATTACHMENT_OUTPUT\n",
                pstate[0].stages[pipeline_stage_tracker::i_CAO],
                pstate[1].stages[pipeline_stage_tracker::i_CAO],
                pstate[2].stages[pipeline_stage_tracker::i_CAO],
                pstate[3].stages[pipeline_stage_tracker::i_CAO]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | FRAGMENT_DENSITY_PROCESS_EXT\n",
                pstate[0].stages[pipeline_stage_tracker::i_FDP],
                pstate[1].stages[pipeline_stage_tracker::i_FDP],
                pstate[2].stages[pipeline_stage_tracker::i_FDP],
                pstate[3].stages[pipeline_stage_tracker::i_FDP]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | COMPUTE_SHADER\n",
                pstate[0].stages[pipeline_stage_tracker::i_CS],
                pstate[1].stages[pipeline_stage_tracker::i_CS],
                pstate[2].stages[pipeline_stage_tracker::i_CS],
                pstate[3].stages[pipeline_stage_tracker::i_CS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | RAY_TRACING_SHADER_KHR\n",
                pstate[0].stages[pipeline_stage_tracker::i_RTS],
                pstate[1].stages[pipeline_stage_tracker::i_RTS],
                pstate[2].stages[pipeline_stage_tracker::i_RTS],
                pstate[3].stages[pipeline_stage_tracker::i_RTS]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | HOST\n",
                pstate[0].stages[pipeline_stage_tracker::i_HST],
                pstate[1].stages[pipeline_stage_tracker::i_HST],
                pstate[2].stages[pipeline_stage_tracker::i_HST],
                pstate[3].stages[pipeline_stage_tracker::i_HST]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | COMMAND_PREPROCESS_NV\n",
                pstate[0].stages[pipeline_stage_tracker::i_CPR],
                pstate[1].stages[pipeline_stage_tracker::i_CPR],
                pstate[2].stages[pipeline_stage_tracker::i_CPR],
                pstate[3].stages[pipeline_stage_tracker::i_CPR]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | ACCELERATION_STRUCTURE_BUILD_KHR\n",
                pstate[0].stages[pipeline_stage_tracker::i_ASB],
                pstate[1].stages[pipeline_stage_tracker::i_ASB],
                pstate[2].stages[pipeline_stage_tracker::i_ASB],
                pstate[3].stages[pipeline_stage_tracker::i_ASB]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | TRANSFER\n",
                pstate[0].stages[pipeline_stage_tracker::i_TR],
                pstate[1].stages[pipeline_stage_tracker::i_TR],
                pstate[2].stages[pipeline_stage_tracker::i_TR],
                pstate[3].stages[pipeline_stage_tracker::i_TR]);
        fmt::print("{:4d} {:4d} {:4d} {:4d} | CONDITIONAL_RENDERING_EXT\n",
                pstate[0].stages[pipeline_stage_tracker::i_CR],
                pstate[1].stages[pipeline_stage_tracker::i_CR],
                pstate[2].stages[pipeline_stage_tracker::i_CR],
                pstate[3].stages[pipeline_stage_tracker::i_CR]);*/
        /*fmt::print("Resources:\n");
        for (size_t i = 0; i < temporaries.size(); ++i) {
            fmt::print("    {:<10} {} write={}:{} access=0:{},1:{},2:{},3:{} layout={} avail={} vis={} out_stage={}\n",
                    temporaries[i].resource->name(), live[i] ? "live" : "dead",
                    (uint64_t) temporaries[i].state.write_snn.queue,
                    (uint64_t) temporaries[i].state.write_snn.serial,
                    (uint64_t) temporaries[i].state.access_sn[0],
                    (uint64_t) temporaries[i].state.access_sn[1],
                    (uint64_t) temporaries[i].state.access_sn[2],
                    (uint64_t) temporaries[i].state.access_sn[3],
                    layout_to_string_compact(temporaries[i].state.layout),
                    access_mask_to_string_compact(temporaries[i].state.availability_mask),
                    access_mask_to_string_compact(temporaries[i].state.visibility_mask),
                    pipeline_stages_to_string_compact(temporaries[i].state.stages));
        }*/
        fmt::print("\n");
    }

    /// @brief Assigns a memory block for the given temporary,
    /// possibly aliasing with a temporary that can be proven dead in the current scheduling state.
    /// @param tmp_index the index of the temporary to assign a memory block to
    void assign_memory(size_t task_index, size_t tmp_index) {
        const auto& t = tasks[task_index];
        const auto n_task = tasks.size();
        const auto n_tmp = temporaries.size();

        auto allocator = dev.get_allocator();
        auto tmp = temporaries[tmp_index].resource.get();

        // skip if it doesn't need allocation
        if (!tmp->is_virtual() || tmp->allocated || allocation_map[tmp_index] != (size_t) -1)
            return;

        virtual_resource& tmp_vr = tmp->as_virtual_resource();

        const auto requirements = tmp_vr.get_allocation_requirements(dev.get_vk_device());

        // resource became alive, if possible, alias with a dead
        // temporary, otherwise allocate a new one.
        bool aliased = false;

        for (size_t i = 0; i < temporaries.size(); ++i) {
            auto tmp_alias = temporaries[i].resource.get();

            // skip if the resource isn't virtual, or is not allocated yet
            if (!tmp_alias->is_virtual() || allocation_map[i] == (size_t) -1) continue;

            // skip if the resource is already dead, and its memory was already reused
            if (dead_and_recycled[i]) continue;

            // skip if the resource has user handles pointing to it that may live beyond the current batch
            if (tmp_alias->has_user_refs()) continue;

            // if we want to use the resource, the resource must be dead (no more uses in subsequent tasks),
            // and there must be an execution dependency chain between the current task and all tasks that last accessed the resource

            bool live = false;
            for (auto reader : tmp_alias->state.readers) {
                if (reader
                        && (reader < base_sn || reader >= t.snn.serial
                                || !reachability[task_index][reader - base_sn])) {
                    // 1. the reader is in a previous batch, there's no way to know if the
                    // current task has an execution dependency on it, so exclude this resource.
                    // 2. the reader is in a future serial
                    // 3. there's no execution dependency chain from the reader to the current task.
                    live = true;  // still uses
                    break;
                }
            }
            const auto writer = tmp_alias->state.writer.serial;
            live = live
                   || (writer
                           && (writer < base_sn || writer >= t.snn.serial
                                   || !reachability[task_index][writer - base_sn]));

            if (live) continue;  // still alive

            // the resource is dead, try to reuse
            auto i_alloc = allocation_map[i];
            auto& dead_alloc = allocations[i_alloc];

            if (!adjust_allocation_requirements(dead_alloc.req, requirements)) continue;

            // the two resources may alias; the requirements
            // have been adjusted
            allocation_map[tmp_index] = i_alloc;
            dead_and_recycled.set(i);
            aliased = true;
            break;
        }

        // no aliasing opportunities
        if (!aliased) {
            // new allocation
            allocations.push_back(device_memory_allocation{.req = requirements, .alloc = nullptr});
            allocation_map[tmp_index] = allocations.size() - 1;
        }
    }

    /// @brief
    /// @param ctx Context kept across calls to schedule_task
    /// @param task_index
    /// @param sn
    /// @param ready_queue_size
    void schedule_task(size_t task_index) {
        const auto n_tmp = temporaries.size();
        const auto n_task = tasks.size();

        task& t = tasks[task_index];

        // --- assign memory for all resources accessed in this task
        for (const auto& access : t.accesses) {
            assign_memory(task_index, access.index);
        }

        // --- allocate command buffer
        if (t.wait) {
            submission_builder.start_batch(
                    t.snn.queue, t.input_wait_serials, t.input_wait_dst_stages, std::move(t.input_wait_semaphores));
        }

        submission_builder.add_task(t);

        if (t.type() == task_type::present) {
            // present task, end the batch
            submission_builder.end_batch_and_present(t.snn.queue, {t.d.u.present.swapchain},
                    {t.d.u.present.image_index}, dev.create_binary_semaphore());
        }

        done_tasks.set(task_index, true);
    }

    void allocate_memory() {
        const auto allocator = dev.get_allocator();

        fmt::print("Memory blocks:\n");
        for (size_t i = 0; i < allocations.size(); ++i) {
            const auto& alloc = allocations[i];
            fmt::print("   MEM_{}: size={} \n"
                       "           align={}\n"
                       "           preferred_flags={}\n"
                       "           required_flags={}\n"
                       "           memory_type_bits={}\n",
                    i, alloc.req.memreq.size, alloc.req.memreq.alignment,
                    to_string(alloc.req.preferred_flags), to_string(alloc.req.required_flags),
                    alloc.req.memreq.memoryTypeBits);
        }

        fmt::print("\nMemory block assignments:\n");
        for (size_t i = 0; i < temporaries.size(); ++i) {
            if (allocation_map[i] != (size_t) -1) {
                fmt::print("   {} => MEM_{}\n", temporaries[i].resource->name(), allocation_map[i]);
            } else {
                fmt::print("   {} => no memory\n", temporaries[i].resource->name());
            }
        }

        for (auto& a : allocations) {
            const VmaAllocationCreateInfo create_info{.flags = 0,
                    .usage = VMA_MEMORY_USAGE_UNKNOWN,
                    .requiredFlags = 0,
                    .preferredFlags = 0,
                    .memoryTypeBits = 0,
                    .pool = VK_NULL_HANDLE,
                    .pUserData = nullptr};

            if (auto r = vmaAllocateMemory(
                        allocator, &a.req.memreq, &create_info, &a.alloc, &a.alloc_info);
                    r != VK_SUCCESS) {
                fmt::print("vmaAllocateMemory failed, VkResult({})\n", r);
            }
        }

        for (size_t i = 0; i < temporaries.size(); ++i) {
            if (allocation_map[i] != (size_t) -1) {
                const auto i_alloc = allocation_map[i];
                temporaries[i].resource->as_virtual_resource().bind_memory(dev.get_vk_device(),
                        allocations[i_alloc].alloc, allocations[i_alloc].alloc_info);
            }
        }
    }

    void finish_pending_cb_batches() {
        submission_builder.finish();

        fmt::print(
                "====================================================\nCommand buffer batches:\n");
        for (auto&& submit : submission_builder.submits) {
            fmt::print("- W={},{},{},{} WDST={},{},{},{} S={} cb_offset={} cb_count={}\n",
                    submit.wait_sn[0], submit.wait_sn[1], submit.wait_sn[2], submit.wait_sn[3],
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[0]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[1]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[2]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[3]),
                    (uint64_t) submit.signal_snn.serial, submit.cb_offset, submit.cb_count);
        }
    }

    void schedule_all() {
        for (size_t i = 0; i < tasks.size(); ++i) {
            schedule_task(i);
        }
    }
};

void dump_tasks(
        std::ostream& out, std::span<const task> tasks, std::span<const temporary> temporaries) {
    out << "digraph G {\n";
    out << "node [shape=record fontname=Consolas];\n";

    for (task_index index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        //const auto& submitted_task = submitted_tasks[index];

        out << "t_" << index << " [shape=record style=filled fillcolor=\"";
        switch (task.snn.queue) {
            case 0: out << "#ffff99"; break;
            case 1: out << "#7fc97f"; break;
            case 2: out << "#fdc086"; break;
            case 3:
            default: out << "#beaed4"; break;
        }

        out << "\" label=\"{";
        out << task.name;
        out << "|sn=" << task.snn.serial << "(q=" << task.snn.queue << ")";
        out << "\\lreads=";
        {
            bool first = true;
            for (const auto& a : task.accesses) {
                if (is_read_access(a.access_mask)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.access_mask) << ")";
                }
            }
        }
        out << "\\lwrites=";
        {
            bool first = true;
            for (const auto& a : task.accesses) {
                if (is_write_access(a.access_mask)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.access_mask) << ")";
                }
            }
        }
        out << "}\"]\n";
    }

    for (task_index i = 0; i < tasks.size(); ++i) {
        for (auto pred : tasks[i].preds) {
            out << "t_" << pred << " -> "
                << "t_" << i << "\n";
        }
    }

    out << "}\n";
}

void queue_impl::enqueue_pending_tasks() {
    const auto vk_device = device_.get_vk_device();

    // --- short-circuit if no tasks
    if (tasks_.empty()) { return; }

    // --- await previous batches
    wait_for_batch();

    // dump
    fmt::print(" === Tasks: === \n");
    for (const auto& t : tasks_) {
        fmt::print("task: {} ({}:{})\n", t.name, (uint64_t) t.snn.queue, (uint64_t) t.snn.serial);
        fmt::print("    input execution barrier: {}->{}\n",
                pipeline_stages_to_string(t.src_stage_mask),
                pipeline_stages_to_string(t.input_stage_mask));
        fmt::print("    input memory barriers: \n");
        for (const auto& imb : t.image_memory_barriers) {
            fmt::print("        {} access:{}->{} layout:{}->{}\n",
                    find_image_name(temporaries_, imb.image),
                    access_mask_to_string(imb.srcAccessMask),
                    access_mask_to_string(imb.dstAccessMask), layout_to_string(imb.oldLayout),
                    layout_to_string(imb.newLayout));
        }
        if (t.wait) {
            fmt::print("    wait: \n");
            if (t.input_wait_serials[0]) {
                fmt::print("        0:{}|{}\n", t.input_wait_serials[0],
                        pipeline_stages_to_string(t.input_wait_dst_stages[0]));
            }
            if (t.input_wait_serials[1]) {
                fmt::print("        1:{}|{}\n", t.input_wait_serials[1],
                        pipeline_stages_to_string(t.input_wait_dst_stages[1]));
            }
            if (t.input_wait_serials[2]) {
                fmt::print("        2:{}|{}\n", t.input_wait_serials[2],
                        pipeline_stages_to_string(t.input_wait_dst_stages[2]));
            }
            if (t.input_wait_serials[3]) {
                fmt::print("        3:{}|{}\n", t.input_wait_serials[3],
                        pipeline_stages_to_string(t.input_wait_dst_stages[3]));
            }
        }

        fmt::print("    output stage: {}\n", pipeline_stages_to_string(t.output_stage_mask));
        if (t.signal) {
            fmt::print("    signal: {}:{}\n", (uint64_t) t.snn.queue, (uint64_t) t.snn.serial);
        }
        fmt::print("\n");
    }
    fmt::print(" === Final resource states: === \n");
    for (const auto& r : temporaries_) {
        const auto& s = r.resource->state;
        fmt::print("{}\n", r.resource->name());
        fmt::print("    stages={}\n", pipeline_stages_to_string(s.stages));
        fmt::print("    avail={}\n", access_mask_to_string(s.availability_mask));
        fmt::print("    vis={}\n", access_mask_to_string(s.visibility_mask));
        fmt::print("    layout={}\n", layout_to_string(s.layout));

        if (s.has_readers()) {
            fmt::print("    readers: \n");
            if (s.readers[0]) { fmt::print("        0:{}\n", s.readers[0]); }
            if (s.readers[1]) { fmt::print("        1:{}\n", s.readers[1]); }
            if (s.readers[2]) { fmt::print("        2:{}\n", s.readers[2]); }
            if (s.readers[3]) { fmt::print("        3:{}\n", s.readers[3]); }
        }
        if (s.writer.serial) {
            fmt::print(
                    "    writer: {}:{}\n", (uint64_t) s.writer.queue, (uint64_t) s.writer.serial);
        }
    }

    schedule_ctx ctx{device_, qinfo_.indices, batch_start_serial_ + 1, tasks_, temporaries_};
    ctx.schedule_all();
    ctx.finish_pending_cb_batches();
    ctx.allocate_memory();

    {
        std::ofstream out_graphviz{
                fmt::format("graal_test_{}.dot", batch_start_serial_), std::ios::trunc};
        dump_tasks(out_graphviz, tasks_, temporaries_);
    }

    std::array<std::optional<command_buffer_pool>, max_queues> cb_pools{};

    //---------------------------------------------------
    // allocate command buffers for each submission
    for (auto& s : ctx.submission_builder.submits) {
        const auto queue = s.signal_snn.queue;
        if (!cb_pools[queue]) {
            cb_pools[queue] = get_command_buffer_pool(queue);
        }
        s.command_buffer = cb_pools[queue]->fetch_command_buffer(vk_device);
        s.command_buffer.begin(vk::CommandBufferBeginInfo{});
    }

    //---------------------------------------------------
    // fill command buffers

    for (size_t i = 0; i < tasks_.size(); ++i) {
        auto& t = tasks_[i];
        auto& batch = ctx.submission_builder.submits[t.submit_batch_index];

        vk::CommandBuffer cb = batch.command_buffer;
        if (t.needs_barrier()) {
            const auto src_stage_mask =
                    t.src_stage_mask ? t.src_stage_mask : vk::PipelineStageFlagBits::eTopOfPipe;
            const auto dst_stage_mask = t.input_stage_mask
                                                ? t.input_stage_mask
                                                : vk::PipelineStageFlagBits::eBottomOfPipe;
            cb.pipelineBarrier(src_stage_mask, dst_stage_mask, {}, 0, nullptr,
                    (uint32_t) t.buffer_memory_barriers.size(), t.buffer_memory_barriers.data(),
                    (uint32_t) t.image_memory_barriers.size(), t.image_memory_barriers.data());
        }

        if (t.type() == task_type::render_pass) {
            // TODO render pass
            if (t.d.u.render.callback) t.d.u.render.callback(nullptr, cb);
        } else if (t.type() == task_type::compute_pass) {
            if (t.d.u.compute.callback) t.d.u.compute.callback(cb);
        }
    }

    for (auto& s : ctx.submission_builder.submits) {
        s.command_buffer.end();
    }

    //---------------------------------------------------
    // queue submission
    per_queue_wait_serials signals{};
    for (auto& b : ctx.submission_builder.submits) {
        b.submit(qinfo_.queues[b.signal_snn.queue], timelines_, signals);
    }

    //---------------------------------------------------
    // stash all resources belonging to the batch

    auto& ifb = in_flight_batches_.emplace_back();
    ifb.start_serial = batch_start_serial_;
    ifb.end_serials = signals;
    // stash cb pools
    for (auto& p : cb_pools) {
        if (p) { ifb.cb_pools.push_back(std::move(*p)); }
    }
    // stash refs to resources
    ifb.resources.reserve(temporaries_.size());
    for (auto tmp : temporaries_) {
        ifb.resources.push_back(std::move(tmp.resource));
    }

    // start new batch
    batch_start_serial_ = last_serial_;
    temporaries_.clear();
    tasks_.clear();
    tmp_indices.clear();
}

void queue_impl::present(swapchain swapchain) {
    auto& t = init_task(
            tasks_.emplace_back(swapchain.vk_swapchain(), swapchain.current_image_index()),
            "present", false);
    // XXX the memory barrier should not be necessary here? according to the spec:
    // "Any writes to memory backing the images referenced by the pImageIndices
    // and pSwapchains members of pPresentInfo, that are available before vkQueuePresentKHR is executed,
    // are automatically made visible to the read access performed by the presentation engine.
    // This automatic visibility operation for an image happens-after the semaphore signal operation,
    // and happens-before the presentation engine accesses the image."
    add_resource_dependency(t, swapchain.impl_,
            resource_access_details{.layout = vk::ImageLayout::ePresentSrcKHR,
                    .access_mask = vk::AccessFlagBits::eMemoryRead,
                    .input_stage = vk::PipelineStageFlagBits::eAllCommands});
    swapchain.acquire_next_image();
}

void queue_impl::wait_for_batch() 
{
    //assert(snn.serial <= batch_start_serial_);
    if (in_flight_batches_.size() < 2) {
        return;
    }

    // wait for the first batch
    const auto vk_device = device_.get_vk_device();
    auto& b = in_flight_batches_.front();

    const vk::SemaphoreWaitInfo wait_info{
            .semaphoreCount = max_queues, 
            .pSemaphores = timelines_, 
            .pValues = b.end_serials.data()};
    vk_device.waitSemaphores(wait_info, 10000000000);  // 10sec batch timeout
    
    for (auto& cbp : b.cb_pools) {
        cbp.reset(vk_device);
        cb_pools_.recycle(std::move(cbp));
        for (auto s : b.semaphores) {
            device_.recycle_binary_semaphore(s);
        }
    }

    in_flight_batches_.pop_front();
}

/// @brief synchronizes two tasks
/// @param before
/// @param after Sequence number of the task to execute after. Must correspond to an unsubmitted task.
/// @param use_binary_semaphore whether to synchronize using a binary semaphore.
/// before < after; if use_binary_semaphore == true, then before must also correspond to an unsubmitted task.
void queue_impl::add_task_dependency(
        submission_number before_snn, task& after, vk::PipelineStageFlags dst_stage_mask) {
    if (before_snn.serial <= batch_start_serial_) {
        // --- Inter-batch synchronization w/ timeline semaphore
        after.wait = true;
        after.input_wait_serials[before_snn.queue] =
                std::max(after.input_wait_serials[before_snn.queue], before_snn.serial);
        after.input_wait_dst_stages[before_snn.queue] |= dst_stage_mask;
    } else {
        const auto local_task_index =
                before_snn.serial - batch_start_serial_ - 1;  // SN relative to start of batch
        auto& before = tasks_[local_task_index];

        // --- Intra-batch synchronization
        if (after.snn.queue != before_snn.queue) {
            // cross-queue dependency w/ timeline semaphore
            before.signal = true;
            after.wait = true;
            after.input_wait_serials[before_snn.queue] =
                    std::max(after.input_wait_serials[before_snn.queue], before_snn.serial);
            after.input_wait_dst_stages[before_snn.queue] |= dst_stage_mask;
        } else {
            after.src_stage_mask |= before.output_stage_mask;
        }

        after.preds.push_back(local_task_index);
        before.succs.push_back(after.snn.serial - batch_start_serial_ - 1);
    }
}

/// @brief Registers a resource access in a submit task
/// @param resource
/// @param mode
void queue_impl::add_resource_dependency(
        task& t, resource_ptr resource, const resource_access_details& access)
{
    const auto tmp_index = get_resource_tmp_index(resource);
    auto& tmp = *temporaries_[tmp_index].resource;
    const bool writing = !!access.output_stage || tmp.state.layout != access.layout;

    auto old_layout = tmp.state.layout;
    auto src_access_mask = tmp.state.availability_mask;
    t.input_stage_mask |= access.input_stage;

    // external semaphore dependency
    if (auto semaphore = std::exchange(resource->state.wait_semaphore, nullptr)) {
        t.input_wait_semaphores.push_back(std::move(semaphore));
        t.wait = true;
    }

    if (writing) {
        if (!tmp.state.has_readers()) {
            // WAW
            if (tmp.state.writer.serial) {
                add_task_dependency(tmp.state.writer, t, access.input_stage);
            }
        } else {
            // WAR
            for (size_t i = 0; i < max_queues; ++i) {
                if (auto r = tmp.state.readers[i]) {
                    add_task_dependency(
                            submission_number{.queue = i, .serial = r}, t, access.input_stage);
                }
            }
        }
        tmp.state.clear_readers();
        tmp.state.writer = t.snn;
        t.output_stage_mask |= access.output_stage;
    } else {
        // RAW
        // a read without a write is probably an uninitialized access
        if (tmp.state.writer.serial) {
            add_task_dependency(tmp.state.writer, t, access.input_stage);
        }
        tmp.state.readers[t.snn.queue] = std::max(tmp.state.readers[t.snn.queue], t.snn.serial);
    }

    // are all writes to the resource visible to the requested access type?
    const bool writes_visible =
            (tmp.state.visibility_mask & vk::AccessFlagBits::eMemoryRead)
            || ((tmp.state.visibility_mask & access.access_mask) == access.access_mask);

    // is the layout of the resource different?
    const bool layout_transition = tmp.state.layout != access.layout;

    // is there a possible write-after-write hazard, that requires a memory dependency?
    const bool write_after_write_hazard = writing && is_write_access(tmp.state.availability_mask);

    if (!writes_visible || layout_transition || write_after_write_hazard) {
        // no need to make memory visible if we're writing to the resource
        const auto dst_access_mask =
                !is_read_access(access.access_mask) ? vk::AccessFlags{} : access.access_mask;

        // the resource access needs a memory barrier
        if (auto img = tmp.as_image()) {
            // image barrier
            const auto format = img->format();
            const auto vk_image = img->vk_image();
            vk::ImageAspectFlags aspect_mask = format_aspect_mask(format);
            const vk::ImageSubresourceRange subresource_range =
                    vk::ImageSubresourceRange{.aspectMask = aspect_mask,
                            .baseMipLevel = 0,
                            .levelCount = VK_REMAINING_MIP_LEVELS,
                            .baseArrayLayer = 0,
                            .layerCount = VK_REMAINING_ARRAY_LAYERS};

            t.image_memory_barriers.push_back(
                    vk::ImageMemoryBarrier{.srcAccessMask = tmp.state.availability_mask,
                            .dstAccessMask = dst_access_mask,
                            .oldLayout = tmp.state.layout,
                            .newLayout = access.layout,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = vk_image,
                            .subresourceRange = subresource_range});

        } else if (auto buf = tmp.as_buffer()) {
            t.buffer_memory_barriers.push_back(
                    vk::BufferMemoryBarrier{.srcAccessMask = tmp.state.availability_mask,
                            .dstAccessMask = dst_access_mask,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .buffer = buf->vk_buffer(),
                            .offset = 0,
                            .size = VK_WHOLE_SIZE});
        }

        tmp.state.availability_mask = {};
        // update the access types that can now see the resource
        tmp.state.visibility_mask |= access.access_mask;
        tmp.state.layout = access.layout;
    }

    // all previous writes are flushed
    if (is_write_access(access.access_mask)) { tmp.state.availability_mask |= access.access_mask; }

    t.accesses.push_back(
            task::resource_access{.index = tmp_index, .access_mask = access.access_mask});
}

}  // namespace detail

//-----------------------------------------------------------------------------

// Called by buffer accessors to register an use of a buffer in a task
void handler::add_buffer_access(std::shared_ptr<detail::buffer_resource> buffer,
        vk::AccessFlags access_mask, vk::PipelineStageFlags input_stage,
        vk::PipelineStageFlags output_stage) const {
    detail::resource_access_details details;
    details.layout = vk::ImageLayout::eUndefined;
    details.access_mask = access_mask;
    details.input_stage |= input_stage;
    details.output_stage |= output_stage;
    queue_.add_resource_dependency(task_, buffer, details);
}

// Called by image accessors to register an use of an image in a task
void handler::add_image_access(std::shared_ptr<detail::image_resource> image,
        vk::AccessFlags access_mask, vk::PipelineStageFlags input_stage,
        vk::PipelineStageFlags output_stage, vk::ImageLayout layout) const {
    detail::resource_access_details details;
    details.layout = layout;
    details.access_mask = access_mask;
    details.input_stage |= input_stage;
    details.output_stage |= output_stage;
    queue_.add_resource_dependency(task_, image, details);
}

//-----------------------------------------------------------------------------
queue::queue(device& dev, const queue_properties& props) :
    impl_{std::make_unique<detail::queue_impl>(dev, props)} {
}

void queue::enqueue_pending_tasks() {
    impl_->enqueue_pending_tasks();
}

void queue::present(swapchain swapchain) {
    impl_->present(std::move(swapchain));
}

detail::task& queue::create_render_pass_task(
        std::string_view name, const render_pass_desc& rpd) noexcept {
    return impl_->create_render_pass_task(name, rpd);
}

detail::task& queue::create_compute_pass_task(std::string_view name, bool async) noexcept {
    return impl_->create_compute_pass_task(name, async);
}

void* queue::get_staging_buffer(
        size_t align, size_t size, vk::Buffer& out_buffer, vk::DeviceSize& out_offset) {
    return impl_->get_staging_buffer(align, size, out_buffer, out_offset);
}

[[nodiscard]] device queue::get_device() {
    return impl_->device_;
}

/*
size_t queue::next_task_index() const {
    return impl_->next_task_index();
}
size_t queue::current_batch_index() const {
    return impl_->current_batch_index();
}*/

}  // namespace graal