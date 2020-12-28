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
    explicit temporary(resource_ptr r) : resource{std::move(r)}
    {
    }

    /// @brief The resource
    resource_ptr resource;
    //std::vector<submission_number> readers;
    //submission_number writer{};
    //vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    //resource_last_access_info state;
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
    if (value & vk::PipelineStageFlagBits::eTessellationControlShader) result += "TESSELLATION_CONTROL_SHADER|";
    if (value & vk::PipelineStageFlagBits::eTessellationEvaluationShader) result += "TESSELLATION_EVALUATION_SHADER|";
    if (value & vk::PipelineStageFlagBits::eGeometryShader) result += "GEOMETRY_SHADER|";
    if (value & vk::PipelineStageFlagBits::eFragmentShader) result += "FRAGMENT_SHADER|";
    if (value & vk::PipelineStageFlagBits::eEarlyFragmentTests) result += "EARLY_FRAGMENT_TESTS|";
    if (value & vk::PipelineStageFlagBits::eLateFragmentTests) result += "LATE_FRAGMENT_TESTS|";
    if (value & vk::PipelineStageFlagBits::eColorAttachmentOutput) result += "COLOR_ATTACHMENT_OUTPUT|";
    if (value & vk::PipelineStageFlagBits::eComputeShader) result += "COMPUTE_SHADER|";
    if (value & vk::PipelineStageFlagBits::eTransfer) result += "TRANSFER|";
    if (value & vk::PipelineStageFlagBits::eBottomOfPipe) result += "BOTTOM_OF_PIPE|";
    if (value & vk::PipelineStageFlagBits::eHost) result += "HOST|";
    if (value & vk::PipelineStageFlagBits::eAllGraphics) result += "ALL_GRAPHICS|";
    if (value & vk::PipelineStageFlagBits::eAllCommands) result += "ALL_COMMANDS|";
    if (value & vk::PipelineStageFlagBits::eTransformFeedbackEXT) result += "TRANSFORM_FEEDBACK_EXT|";
    if (value & vk::PipelineStageFlagBits::eConditionalRenderingEXT) result += "CONDITIONAL_RENDERING_EXT|";
    if (value & vk::PipelineStageFlagBits::eRayTracingShaderKHR) result += "ACCELERATION_STRUCTURE_BUILD_KHR|";
    if (value & vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR) result += "RAY_TRACING_SHADER_KHR|";
    if (value & vk::PipelineStageFlagBits::eShadingRateImageNV) result += "SHADING_RATE_IMAGE_NV|";
    if (value & vk::PipelineStageFlagBits::eTaskShaderNV) result += "TASK_SHADER_NV|";
    if (value & vk::PipelineStageFlagBits::eMeshShaderNV) result += "MESH_SHADER_NV|";
    if (value & vk::PipelineStageFlagBits::eFragmentDensityProcessEXT) result += "FRAGMENT_DENSITY_PROCESS_EXT|";
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
    if (value & vk::AccessFlagBits::eDepthStencilAttachmentRead) result += "DEPTH_STENCIL_ATTACHMENT_READ|";
    if (value & vk::AccessFlagBits::eDepthStencilAttachmentWrite) result += "DEPTH_STENCIL_ATTACHMENT_WRITE|";
    if (value & vk::AccessFlagBits::eTransferRead) result += "TRANSFER_READ|";
    if (value & vk::AccessFlagBits::eTransferWrite) result += "TRANSFER_WRITE|";
    if (value & vk::AccessFlagBits::eHostRead) result += "HOST_READ|";
    if (value & vk::AccessFlagBits::eHostWrite) result += "HOST_WRITE|";
    if (value & vk::AccessFlagBits::eMemoryRead) result += "MEMORY_READ|";
    if (value & vk::AccessFlagBits::eMemoryWrite) result += "MEMORY_WRITE|";
    if (value & vk::AccessFlagBits::eTransformFeedbackWriteEXT) result += "TRANSFORM_FEEDBACK_WRITE_EXT|";
    if (value & vk::AccessFlagBits::eTransformFeedbackCounterReadEXT) result += "TRANSFORM_FEEDBACK_COUNTER_READ_EXT|";
    if (value & vk::AccessFlagBits::eTransformFeedbackCounterWriteEXT) result += "TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT|";
    if (value & vk::AccessFlagBits::eConditionalRenderingReadEXT) result += "CONDITIONAL_RENDERING_READ_EXT|";
    if (value & vk::AccessFlagBits::eColorAttachmentReadNoncoherentEXT) result += "COLOR_ATTACHMENT_READ_NONCOHERENT_EXT|";
    if (value & vk::AccessFlagBits::eAccelerationStructureReadKHR) result += "ACCELERATION_STRUCTURE_READ_KHR|";
    if (value & vk::AccessFlagBits::eAccelerationStructureWriteKHR) result += "ACCELERATION_STRUCTURE_WRITE_KHR|";
    if (value & vk::AccessFlagBits::eShadingRateImageReadNV) result += "SHADING_RATE_IMAGE_READ_NV|";
    if (value & vk::AccessFlagBits::eFragmentDensityMapReadEXT) result += "FRAGMENT_DENSITY_MAP_READ_EXT|";
    if (value & vk::AccessFlagBits::eCommandPreprocessReadNV) result += "COMMAND_PREPROCESS_READ_NV|";
    if (value & vk::AccessFlagBits::eCommandPreprocessWriteNV) result += "COMMAND_PREPROCESS_WRITE_NV|";
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
    case vk::ImageLayout::eDepthStencilAttachmentOptimal: return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
    case vk::ImageLayout::eDepthStencilReadOnlyOptimal: return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
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
    case vk::ImageLayout::ePresentSrcKHR: return "PR";
    case vk::ImageLayout::eSharedPresentKHR: return "SPR";
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

/// @brief Per-batch+thread resources
struct batch_thread_resources {
    command_buffer_pool cb_pool;
};

struct submitted_batch {
    serial_number start_serial = 0;
    /// @brief Serials to wait for
    per_queue_wait_serials end_serials{};
    /// @brief Per-thread resources
    std::vector<batch_thread_resources> threads;
    /// @brief All resources referenced in the batch.
    std::vector<resource_ptr> resources;
    /// @brief Semaphores used within the batch
    std::vector<vk::Semaphore> semaphores;
};

/*static void dump_batch(batch& b) {
    fmt::print("=== batch (SN{} .. SN{}) ===\n", b.start_serial, b.finish_serial());

    fmt::print("Temporaries:\n");
    size_t tmp_index = 0;
    for (const auto& tmp : b.temporaries) {
        fmt::print(" - #{}: {} user_ref_count={} last_write={}\n", tmp_index++,
                tmp.resource->name().empty() ? "<unnamed>" : tmp.resource->name(),
                tmp.resource->user_ref_count(), tmp.write_snn.serial);
    }

    fmt::print("Tasks:\n");
    size_t task_index = 0;
    for (auto&& t : b.tasks) {
        fmt::print(" - task #{} {}\n", task_index, t.name.empty() ? "<unnamed>" : t.name);

        if (!t.preds.empty()) {
            fmt::print("   preds: ");
            for (auto pred : t.preds) {
                fmt::print("{},", pred);
            }
            fmt::print("\n");
        }
        if (!t.succs.empty()) {
            fmt::print("   succs: ");
            for (auto s : t.succs) {
                fmt::print("{},", s);
            }
            fmt::print("\n");
        }

        if (!t.accesses.empty()) {
            fmt::print("   accesses: \n");
            for (const auto& access : t.accesses) {
                fmt::print("      #{} flags={}\n", access.index,
                        to_string(access.details.access_flags));
                fmt::print("      input_stage={}\n"
                           "      output_stage={}\n",
                        to_string(access.details.input_stage),
                        to_string(access.details.output_stage));
                const auto ty = b.temporaries[access.index].resource->type();
                if (ty == resource_type::image || ty == resource_type::swapchain_image) {
                    fmt::print("          layout={}\n", to_string(access.details.layout));
                }
            }
            fmt::print("\n");
        }
        task_index++;
    }
}*/

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
            fmt::print("{{{:d}:{:d}", (uint64_t)s[i].queue, (uint64_t)s[i].serial);
            first = false;
        }
        else {
            fmt::print(",{:d}:{:d}", (uint64_t)s[i].queue, (uint64_t)s[i].serial);
        }
    }
    if (first) {
        fmt::print("{{}}");
    }
    else {
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

enum class sync_type {
    none,
    pipeline_barrier,  // CmdPipelineBarrier
    semaphore,          // QueueSubmit, signal/wait semaphore
    event,          // CmdSetEvent/CmdWaitEvent
};

struct barrier {
    vk::PipelineStageFlags src_stage_mask;
    vk::PipelineStageFlags dst_stage_mask;
    std::vector<vk::ImageMemoryBarrier> image_memory_barriers;
    std::vector<vk::BufferMemoryBarrier> buffer_memory_barriers;

    /*void merge_with(barrier&& other) {
        src_stage_mask |= other.src_stage_mask;
        //dst_stage_mask |= other.dst_stage_mask;
        image_memory_barriers.insert(image_memory_barriers.end(),
            other.image_memory_barriers.begin(), other.image_memory_barriers.end());
        buffer_memory_barriers.insert(buffer_memory_barriers.end(),
            other.buffer_memory_barriers.begin(), other.buffer_memory_barriers.end());
        std::vector<serial_number> new_source;
        std::vector<serial_number> new_destination;
        std::set_union(source.begin(), source.end(), other.source.begin(), other.source.end(),
            std::back_inserter(new_source));
        std::set_union(destination.begin(), destination.end(), other.destination.begin(),
            other.destination.end(), std::back_inserter(new_destination));
        source = std::move(new_source);
        destination = std::move(new_destination);
    }*/

};

/*
struct task_sync {
    bool signal = false;
    bool wait = false;
    per_queue_wait_serials input_wait_serials;
    per_queue_wait_dst_stages input_wait_dst_stages;

    std::vector<serial_number> inputs;
    std::vector<serial_number> outputs;
    vk::PipelineStageFlags src_stage_mask;  
    vk::PipelineStageFlags dst_stage_mask;
    std::vector<image_memory_sync> image_memory_barriers;
    //std::vector<vk::BufferMemoryBarrier> buffer_memory_barriers;
    

    image_memory_sync* find_image_memory_barrier(vk::Image image) {
        for (auto& b : image_memory_barriers) {
            if (b.image == image) { return &b; }
        }
        return nullptr;
    }

    image_memory_sync& get_image_memory_barrier(const image_resource& resource) {
        if (auto b = find_image_memory_barrier(resource.vk_image())) { return *b; }
        auto& b = image_memory_barriers.emplace_back();
        const auto format = resource.format();
        const auto aspect_mask = format_aspect_mask(format);
        const vk::ImageSubresourceRange subresource_range{ .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS };
        // setup the source side of the memory barrier
        b.input_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.input_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.input_barrier.image = resource.vk_image();
        b.input_barrier.subresourceRange.aspectMask = aspect_mask;
        b.input_barrier.subresourceRange.baseMipLevel = 0;
        b.input_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        b.input_barrier.subresourceRange.baseArrayLayer = 0;
        b.input_barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
        return b;
    }

    vk::BufferMemoryBarrier& get_buffer_memory_barrier(const buffer_resource& resource) {
        if (auto b = find_buffer_memory_barrier(resource.vk_buffer())) { return *b; }
        auto& b = buffer_memory_barriers.emplace_back();
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = resource.vk_buffer();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        return b;
    }
};
*/

/*struct barrier_builder {
    std::vector<task_sync> task_syncs;
    serial_number start_sn;

    void image_memory_dependency(
        submission_number source,
        submission_number destination,
        image_resource& image,
        const resource_access_details& access) 
    {
        const auto dst_local_index = destination.serial - start_sn;
        auto& dst_sync = task_syncs[dst_local_index];

        if (source.serial < start_sn) {
            // external dependency
            //auto& imb = dst_sync.get_pre_image_memory_barrier();
        }

        const auto src_local_index = source.serial-start_sn;

        auto& src_sync = task_syncs[src_local_index];

        sorted_vector_insert(src_sync.outputs, destination.serial);
        sorted_vector_insert(dst_sync.inputs, source.serial);
        
        src_sync.dst_stage_mask |= access.input_stage;
        dst_sync.src_stage_mask |= access.output_stage;

        if (source.queue != destination.queue || src_sync.signal || dst_sync.wait) {
            src_sync.signal = true;
            dst_sync.wait = true;
            for (auto input : dst_sync.inputs) {
                task_syncs[input-start_sn].signal = true;
            }
            dst_sync.input_wait_serials[source.queue] = std::max(dst_sync.input_wait_serials[source.queue], source.serial);
            dst_sync.input_wait_dst_stages[source.queue] |= access.input_stage;
        }
        else {
            auto& imb = dst_sync.get_image_memory_barrier(image);
            imb.dstAccessMask |= access.access_mask;
            imb.newLayout = access.layout;
        }

        if (!dst_sync.signal && access.output_stage) {
            //auto& imb = dst_sync.get_image_memory_barrier(image);
            //imb.srcAccessMask |= access.access_mask;
            //imb.oldLayout = access.layout;
        }
    }

    void dump(std::span<const temporary> temporaries) {
        // dump
        for (size_t i = 0; i < task_syncs.size(); ++i) {
            auto&& ts = task_syncs[i];
            fmt::print("task {} (", i+1);
            dump_vector_set(ts.inputs);
            fmt::print("->");
            dump_vector_set(ts.outputs);
            fmt::print(")\n    EX: {}->{}", pipeline_stages_to_string_compact(ts.src_stage_mask),
                    pipeline_stages_to_string_compact(ts.dst_stage_mask));

            if (ts.wait) {
                fmt::print("\n    WAIT: _->{},{},{},{}({},{},{},{})", ts.input_wait_serials[0],
                    ts.input_wait_serials[1], ts.input_wait_serials[2], ts.input_wait_serials[3],
                    pipeline_stages_to_string_compact(ts.input_wait_dst_stages[0]),
                    pipeline_stages_to_string_compact(ts.input_wait_dst_stages[1]),
                    pipeline_stages_to_string_compact(ts.input_wait_dst_stages[2]),
                    pipeline_stages_to_string_compact(ts.input_wait_dst_stages[3]));
            }

            if (ts.signal) {
                fmt::print("\n    SIGNAL: {}\n", i+1);
                for (auto&& imb : ts.image_memory_barriers) {
                    fmt::print("    IMB: {}\n"
                        "        trans : {}->{}\n", find_image_name(temporaries, imb.image),
                        layout_to_string_compact(imb.oldLayout),
                        layout_to_string_compact(imb.newLayout));
                }
            }
            else {
                fmt::print("\n");
                for (auto&& imb : ts.image_memory_barriers) {
                    fmt::print("    IMB: {}\n"
                        "        flush : {}\n"
                        "        inv   : {}\n"
                        "        trans : {}->{}\n", find_image_name(temporaries, imb.image),
                        access_mask_to_string_compact(imb.srcAccessMask),
                        access_mask_to_string_compact(imb.dstAccessMask),
                        layout_to_string_compact(imb.oldLayout),
                        layout_to_string_compact(imb.newLayout));
                }
            }
            fmt::print("\n");
        }
    }
};*/

//-----------------------------------------------------------------------------
class queue_impl {
    friend class graal::queue;

public:
    queue_impl(device device, const queue_properties& props);
    ~queue_impl();

    void enqueue_pending_tasks();
    void present(swapchain swapchain);

    void wait_for_task(uint64_t sequence_number);

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

    void add_task_dependency(submission_number before_snn, task& after, vk::PipelineStageFlags wait_dst_stages);

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
        case task_type::render_pass:
            t.snn.queue = queue_indices_.graphics;
            break;
        case task_type::compute_pass:
            if (t.async) {
                t.snn.queue = queue_indices_.compute;
            }
            else {
                t.snn.queue = queue_indices_.graphics; 
            }
            break;
        case task_type::transfer:
            if (t.async) {
                t.snn.queue = queue_indices_.transfer;
            }
            else {
                t.snn.queue = queue_indices_.graphics;
            }
            break;
        case task_type::present:
            t.snn.queue = queue_indices_.compute;   
            break;
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
    [[nodiscard]] command_buffer_pool get_command_buffer_pool();

    /// @brief Returns whether writes to the specified resource in this batch could be seen
    /// by uses of the resource in subsequent batches.
    /// @return
    [[nodiscard]] bool writes_are_user_observable(resource& r) const noexcept;

    void propagate_signal_wait();

    using resource_to_temporary_map = std::unordered_map<resource*, size_t>;

    device device_;
    queue_properties props_;
    queue_indices queue_indices_;
    vk::Semaphore timelines_[max_queues];
    staging_pool staging_;
    recycler<staging_buffer> staging_buffers_;
    recycler<command_buffer_pool> cb_pools;

    serial_number last_serial_ = 0;
    serial_number completed_serial_ = 0;
    serial_number batch_start_serial_ = 0;

    //barrier_builder barriers_;
    std::vector<temporary> temporaries_;
    std::vector<task> tasks_;
    resource_to_temporary_map tmp_indices;  // for the current batch

    std::deque<submitted_batch> in_flight_batches_;
};

queue_impl::queue_impl(device device, const queue_properties& props) :
    device_{std::move(device)}, props_{props}, queue_indices_{device_.get_queue_indices()} {
    auto vk_device = device_.get_vk_device();

    vk::SemaphoreTypeCreateInfo timeline_create_info{
            .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
    vk::SemaphoreCreateInfo semaphore_create_info{.pNext = &timeline_create_info};
    for (size_t i = 0; i < max_queues; ++i) {
        timelines_[i] = vk_device.createSemaphore(semaphore_create_info);
    }
}

queue_impl::~queue_impl() {
    auto vk_device = device_.get_vk_device();
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

command_buffer_pool queue_impl::get_command_buffer_pool() {
    const auto vk_device = device_.get_vk_device();
    command_buffer_pool cbp;
    if (!cb_pools.fetch(cbp)) {
        // TODO other queues?
        vk::CommandPoolCreateInfo create_info{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                .queueFamilyIndex = device_.get_graphics_queue_family()};
        const auto pool = vk_device.createCommandPool(create_info);
        cbp = command_buffer_pool{
                .command_pool = pool,
        };
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

struct ready_task {
    size_t latest_pred_serial;
    size_t index;

    struct compare {
        inline constexpr bool operator()(const ready_task& lhs, const ready_task& rhs) {
            return lhs.latest_pred_serial > rhs.latest_pred_serial;
        }
    };
};

using task_priority_queue =
        std::priority_queue<ready_task, std::vector<ready_task>, ready_task::compare>;

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

struct image_memory_barrier {
    size_t resource_index;
    vk::AccessFlags src_access_mask;
    vk::AccessFlags dst_access_mask;
    vk::ImageLayout old_layout;
    vk::ImageLayout new_layout;
};

struct buffer_memory_barrier {
    size_t resource_index;
    vk::AccessFlags src_access_mask;
    vk::AccessFlags dst_access_mask;
};

/// @brief Represents a queue operation (either a call to vkQueueSubmit or vkQueuePresent)
struct queue_submission {
    per_queue_wait_serials wait_sn{};
    per_queue_wait_dst_stages wait_dst_stages{};
    serial_number first_sn;
    serial_number signal_sn;
    std::vector<serial_number> serials;  // not necessarily in order
    std::vector<vk::CommandBuffer> command_buffers;
    std::vector<vk::Semaphore> wait_binary_semaphores;
    std::vector<vk::Semaphore> signal_binary_semaphores;
    vk::Semaphore render_finished;
    std::vector<vk::SwapchainKHR> swapchains;
    std::vector<uint32_t> swpachain_image_indices;

    [[nodiscard]] bool is_empty() const noexcept {
        return command_buffers.empty() && signal_binary_semaphores.empty() && !render_finished;
    }

    void submit(vk::Queue queue, uint8_t queue_index, std::span<vk::CommandBuffer> command_buffers,
            std::span<vk::Semaphore> timelines) {
        const auto signal_semaphore_value = signal_sn;

        std::vector<vk::Semaphore> signal_semaphores;
        signal_semaphores.push_back(timelines[queue_index]);
        signal_semaphores.insert(signal_semaphores.end(), signal_binary_semaphores.begin(),
                signal_binary_semaphores.end());
        if (render_finished) { signal_semaphores.push_back(render_finished); }

        std::vector<vk::Semaphore> wait_semaphores;
        std::vector<uint64_t> wait_semaphore_values;
        for (size_t i = 0; i < wait_sn.size(); ++i) {
            if (wait_sn[i]) {
                wait_semaphores.push_back(timelines[i]);
                wait_semaphore_values.push_back(wait_sn[i]);
            }
        }
        wait_semaphores.insert(wait_semaphores.end(), wait_binary_semaphores.begin(),
                wait_binary_semaphores.end());

        const vk::TimelineSemaphoreSubmitInfo timeline_submit_info{
                .waitSemaphoreValueCount = (uint32_t) wait_semaphore_values.size(),
                .pWaitSemaphoreValues = wait_semaphore_values.data(),
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &signal_semaphore_value};

        vk::SubmitInfo submit_info{.pNext = &timeline_submit_info,
                .waitSemaphoreCount = (uint32_t) wait_semaphores.size(),
                .pWaitSemaphores = wait_semaphores.data(),
                .pWaitDstStageMask = wait_dst_stages.data(),
                .commandBufferCount = (uint32_t) command_buffers.size(),
                .pCommandBuffers = command_buffers.data(),
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
                    .pImageIndices = swpachain_image_indices.data()};

            queue.presentKHR(present_info);
        }
    }
};

// batch (and submission_builder)
// command_buffer_pool
// submitted_task
// task
// temporary
// queue_submission

struct submitted_task {
    vk::PipelineStageFlags src_stage_mask;
    vk::PipelineStageFlags dst_stage_mask;
    submission_number snn;
    size_t cb_index;
    uint32_t num_image_memory_barriers;
    size_t image_memory_barriers_offset;  // offset into the vector of image barriers
    uint32_t num_buffer_memory_barriers;
    size_t buffer_memory_barriers_offset;  // offset into the vector of buffer barriers

    bool needs_cmd_pipeline_barrier() const noexcept {
        return src_stage_mask != vk::PipelineStageFlagBits::eTopOfPipe
               || dst_stage_mask != vk::PipelineStageFlagBits::eBottomOfPipe
               || num_buffer_memory_barriers != 0 || num_image_memory_barriers != 0;
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

    queue_submission& get_pending(uint32_t queue) {
        return submits[pending[queue]];
    }

    void start_batch(uint32_t queue, per_queue_wait_serials wait_sn,
            per_queue_wait_dst_stages wait_dst_stages,
            std::vector<vk::Semaphore> wait_binary_semaphores) {
        for (size_t iq = 0; iq < max_queues; ++iq) {
            queue_submission& qs = get_pending(iq);
            if (wait_sn[iq] || iq == queue) { end_batch(iq); }
        }

        auto& qs = get_pending(queue);
        qs.wait_sn = wait_sn;
        qs.wait_dst_stages = wait_dst_stages;
        qs.wait_binary_semaphores = std::move(wait_binary_semaphores);
    }

    void end_batch(uint32_t queue) {
        auto& qs = get_pending(queue);
        if (qs.is_empty()) { return; }
        pending[queue] = (size_t)-1;
    }

    void add_task(task& t) {
        auto& qs = get_pending(t.snn.queue);
        if (qs.first_sn == 0) {
            // set submission SN
            qs.first_sn = t.snn.serial;
        }
        qs.serials.push_back(t.snn.serial);
        qs.command_buffers.push_back(nullptr);
        qs.signal_sn = std::max(qs.signal_sn, t.snn.serial);
        t.submission_index = pending[t.snn.queue];
    }

    void end_batch_and_present(uint32_t queue, std::vector<vk::SwapchainKHR> swapchains,
            std::vector<uint32_t> swpachain_image_indices, vk::Semaphore render_finished) 
    {
        auto& qs = get_pending(queue);
        // end the batch with a present (how considerate!)
        qs.render_finished = render_finished;
        qs.swapchains = std::move(swapchains);
        qs.swpachain_image_indices = std::move(swpachain_image_indices);
        end_batch(queue);
    }

    void finish() {
        for (size_t i = 0; i < max_queues; ++i) {
            end_batch(i);
        }
    }
};

struct schedule_ctx {
    device& dev;
    queue_indices queues;
    std::span<task> tasks;
    std::span<temporary> temporaries;
    std::span<barrier> barriers;

    // --- scheduling state
    serial_number base_sn;  // serial number of the first task
    serial_number next_sn;  // serial number of the next task to be scheduled
    bitset done_tasks;  // set of submitted tasks
    bitset ready_tasks;  // set of tasks ready to be submitted
    task_priority_queue ready_queue;  // queue of tasks ready to be submitted

    // --- pipeline state tracking
    pipeline_stage_tracker pstate[max_queues];

    // --- liveness
    std::vector<bitset> reachability;  // reachability matrix [to][from]
    std::vector<bitset> live_sets;  // live-sets for each task
    std::vector<bitset> use_sets;  // use-sets for each task
    std::vector<bitset> def_sets;  // def-sets for each task
    bitset live;  // set of live temporaries after the last scheduled task
    bitset dead;  // set of dead temporaries after the last scheduled task
    bitset gen;  // temporaries that just came alive after the last scheduled task
    bitset kill;  // temporaries that just dropped dead after the last scheduled task
    bitset mask;  // auxiliary set
    bitset tmp;  //

    // --- device memory allocations
    std::vector<device_memory_allocation> allocations;
    std::vector<size_t> allocation_map;  // tmp index -> allocations

    // memory_dependency state for the task being submitted
    //std::vector<submission_number> current_queue_syncs;

    // --- queue batches
    submission_builder submission_builder;

    // --- submits
    std::vector<submitted_task> submitted_tasks;
    std::vector<vk::BufferMemoryBarrier> buffer_memory_barriers;
    std::vector<vk::ImageMemoryBarrier> image_memory_barriers;
    std::vector<size_t>
            buffer_memory_barriers_temporaries;  // resources referenced in buffer_memory_barriers
    std::vector<size_t>
            image_memory_barriers_temporaries;  // resources referenced in image_memory_barriers

    schedule_ctx(device& d, queue_indices queues, serial_number base_sn, std::span<task> tasks,
            std::span<temporary> temporaries, std::span<barrier> barriers) :
        dev{d},
        queues{queues}, base_sn{base_sn}, next_sn{base_sn}, tasks{tasks},
        temporaries{temporaries}, barriers{barriers} {
        const auto n_task = tasks.size();
        const auto n_tmp = temporaries.size();
        reachability = transitive_closure(tasks);
        done_tasks.resize(n_task, false);
        ready_tasks.resize(n_task, false);
        submitted_tasks.resize(n_task);
        allocation_map.resize(n_tmp, (size_t) -1);

        live.resize(n_tmp);
        dead.resize(n_tmp);
        gen.resize(n_tmp);
        kill.resize(n_tmp);
        mask.resize(n_tmp);
        tmp.resize(n_tmp);
        use_sets.resize(n_task);
        def_sets.resize(n_task);
        live_sets.resize(n_task);

        for (size_t i = 0; i < n_task; ++i) {
            use_sets[i].resize(n_tmp);
            def_sets[i].resize(n_tmp);
            live_sets[i].resize(n_tmp);

            for (const auto& r : tasks[i].accesses) {
                if (is_read_access(r.dst_access_mask)) { use_sets[i].set(r.index); }
                if (is_write_access(r.dst_access_mask)) { def_sets[i].set(r.index); }
            }
        }
    }

    void dump() {
        fmt::print("Scheduling state dump:\nnext_sn={}\n", next_sn);
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
                pstate[3].stages[pipeline_stage_tracker::i_CR]);
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
    void assign_memory(size_t tmp_index) {
        auto allocator = dev.get_allocator();
        auto r = temporaries[tmp_index].resource.get();

        // skip if it doesn't need allocation
        if (!r->is_virtual() || r->allocated || allocation_map[tmp_index] != (size_t) -1) return;

        virtual_resource& vr = r->as_virtual_resource();

        const auto requirements = vr.get_allocation_requirements(dev.get_vk_device());

        // resource became alive, if possible, alias with a dead
        // temporary, otherwise allocate a new one.
        bool aliased = false;
        for (size_t t1 = 0; t1 < temporaries.size(); ++t1) {
            // filter out live resources
            if (!dead[t1]) continue;

            auto i_alloc = allocation_map[t1];
            auto& dead_alloc = allocations[i_alloc];

            if (adjust_allocation_requirements(dead_alloc.req, requirements)) {
                // the two resources may alias; the requirements
                // have been adjusted
                allocation_map[tmp_index] = i_alloc;
                // not dead anymore
                dead[t1] = false;
                aliased = true;
                break;
            }

            // otherwise continue
        }

        // no aliasing opportunities
        if (!aliased) {
            // new allocation
            allocations.push_back(device_memory_allocation{.req = requirements, .alloc = nullptr});
            allocation_map[tmp_index] = allocations.size() - 1;
        }
    }

    /*/// @brief Infers the necessary barriers for the given access in the given task.
    /// @param tmp_index index of the temporary
    /// @param access access details
    /// @param snn task submission number (queue and serial)
    /// @param wait_sn (output) serials to wait for on each queue
    /// @param wait_dst_stages (output) destination stages for each queue wait
    /// @param num_image_memory_barriers (output) number of image memory barriers written (to image_memory_barriers)
    /// @param num_buffer_memory_barriers (output) number of buffer memory barriers written (to buffer_memory_barriers)
    /// @param src_stage_flags (output) source stage mask for the pipeline barrier on the submission queue
    /// @param dst_stage_flags (output) destination stage mask for the pipeline barrier
    /// @param out_wait_semaphores (output) semaphores to wait for
    void memory_dependency(size_t tmp_index, const resource_access_details& access,
            submission_number snn, per_queue_wait_serials& wait_sn,
            per_queue_wait_dst_stages& wait_dst_stages, uint32_t& num_image_memory_barriers,
            uint32_t& num_buffer_memory_barriers, vk::PipelineStageFlags& src_stage_mask,
            vk::PipelineStageFlags& dst_stage_mask,
            std::vector<vk::Semaphore>& out_wait_semaphores)
    {
        auto& tmp = temporaries[tmp_index];
        const bool reading = is_read_access(access.access_mask);

        // HACK: there can be only one wait operation per binary semaphore, so only this task can wait on the resource using
        // the binary semaphore.
        // Other tasks that read from the resource concurrently (on other queues) cannot synchronize on the semaphore directly,
        // so instead we force those to synchronize on this task, *even if the task is not writing to the resource*.
        // We do so by "upgrading" the access to a write access.
        // This is suboptimal, as it introduces a false dependency between this task and concurrent
        // tasks accessing the resource. The "proper" solution here would be to have multiple semaphores,
        // but we can't know in advance (i.e. across batches) how many of those we would need.
        //
        // Note that binary semaphores are only used for presentation images currently, and usually the first thing that we do
        // is write to them, so this particular situation is not expected to come up often.
        const bool writing = is_write_access(access.access_mask)
                             || access.layout != tmp.state.layout || tmp.state.wait_semaphore;

        // --- is an execution barrier necessary for this access?
        if (writing) {
            for (size_t iq = 0; iq < max_queues; ++iq) {
                // WA* hazard: need to sync w.r.t. last accesses across ALL queues
                if (tmp.state.access_sn[iq]) {
                    // resource accessed on queue iq
                    wait_sn[iq] = std::max(wait_sn[iq], tmp.state.access_sn[iq]);
                    if (iq == snn.queue) {
                        // access on same queue, will use a pipeline barrier
                        src_stage_mask |= tmp.state.stages;
                    }
                }
                wait_dst_stages[iq] |= access.input_stage;
            }
        } else {
            // RAW hazard: need queue sync w.r.t last write
            wait_sn[tmp.state.write_snn.queue] =
                    std::max(wait_sn[tmp.state.write_snn.queue], tmp.state.write_snn.serial);
            if (tmp.state.write_snn.queue == snn.queue) { src_stage_mask |= tmp.state.stages; }
            wait_dst_stages[tmp.state.write_snn.queue] |= access.input_stage;
        }

        // --- is a semaphore wait necessary for this access?
        if (tmp.state.wait_semaphore) {
            // consume the semaphore
            out_wait_semaphores.push_back(std::exchange(tmp.state.wait_semaphore, nullptr));
        }

        dst_stage_mask |= access.input_stage;

        // --- is a memory barrier necessary for this access?

        
        // are all writes to the resource visible to the requested access type?
        const bool writes_visible =
            (tmp.state.visibility_mask & vk::AccessFlagBits::eMemoryRead)
            || ((tmp.state.visibility_mask & access.access_mask) == access.access_mask);

        // is the layout of the resource different?
        const bool different_layout = tmp.state.layout != access.layout;

        // is there a possible write-after-write hazard, that requires a memory dependency?
        const bool write_after_write_hazard = writing && is_write_access(tmp.state.availability_mask);

        if (!writes_visible || different_layout || write_after_write_hazard) 
        {
            // the resource access needs a memory barrier
            if (auto img = tmp.resource->as_image()) {
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

                image_memory_barriers.push_back(
                        vk::ImageMemoryBarrier{
                                .srcAccessMask = tmp.state.availability_mask,
                                .dstAccessMask = access.access_mask,
                                .oldLayout = tmp.state.layout,
                                .newLayout = access.layout,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = vk_image,
                                .subresourceRange = subresource_range});
                image_memory_barriers_temporaries.push_back(tmp_index);
                num_image_memory_barriers++;

            } else if (auto buf = tmp.resource->as_buffer()) {
                buffer_memory_barriers.push_back(
                        vk::BufferMemoryBarrier{
                                .srcAccessMask = tmp.state.availability_mask,
                                .dstAccessMask = access.access_mask,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .buffer = buf->vk_buffer(),
                                .offset = 0,
                                .size = VK_WHOLE_SIZE});
                buffer_memory_barriers_temporaries.push_back(tmp_index);
                num_buffer_memory_barriers++;
            }

            // all previous writes are flushed
            tmp.state.availability_mask = {};
            // update the access types that can now see the resource
            tmp.state.visibility_mask |= access.access_mask;
            tmp.state.layout = access.layout;
        }

        // --- update what we know about the resource after applying the barriers
        tmp.state.stages = access.output_stage;
        if (writing) {
            tmp.state.write_snn = snn;
            for (size_t iq = 0; iq < max_queues; ++iq) {
                if (iq == snn.queue) {
                    tmp.state.access_sn[iq] = snn.serial;
                } else {
                    tmp.state.access_sn[iq] = 0;
                }
            }
        } else {
            tmp.state.access_sn[snn.queue] = snn.serial;
        }
    }*/

    void update_liveness(size_t task_index) {
        // update live/dead sets
        //
        // a resource becomes live if it wasn't live before and it's used in this task (use or def).
        // a resource becomes dead if
        // - it was live before
        // - it's not used in any task that isn't a predecessor of the current task (no use between defs)
        // - we know, because of previous execution barriers, that all pipeline stages accessing the resource have finished
        //      - NOTE: never insert a barrier just to improve aliasing
        // determine the resources live before this task
        const auto& t = tasks[task_index];
        const auto n_task = tasks.size();
        const auto n_tmp = temporaries.size();

        //live.reset();
        //for (auto p : t.preds) {
        //    live |= live_sets[p];
        //}

        // determine the resources that come alive in this task (
        gen.reset();
        gen |= use_sets[task_index];  // "use" == read access
        gen |= def_sets[task_index];  // "def" == write access
        gen -= live;  // exclude those that were already live

        // add uses & defs to the live set
        live |= use_sets[task_index];
        live |= def_sets[task_index];

        // initialize kill set to all live temps, and progressively remove
        kill = live;
        // don't kill defs and uses of the current task
        kill -= use_sets[task_index];
        kill -= def_sets[task_index];

        mask.reset();

        // kill set calculation: remove temps accessed in parallel branches
        for (size_t j = 0; j < n_task; ++j) {
            if (j == task_index || reachability[j][task_index] || reachability[task_index][j])
                continue;
            kill -= use_sets[j];
            kill -= def_sets[j];
        }

        // now look for uses and defs on the successor branches
        // if there's a def before any use, or no use at all, then consider the
        // resource dead (its contents are not going to be used anymore on this
        // branch).
        for (size_t succ = task_index + 1; succ < n_task; ++succ) {
            if (!reachability[succ][task_index]) continue;

            // def use mask kill mask
            // 0   0   0    kill 0
            // 1   0   0    1    1
            // 0   1   0    0    1
            // 1   1   0    0    1

            // tmp = bits to clear
            tmp = use_sets[succ];
            tmp.flip();
            tmp |= mask;
            kill &= tmp;
            mask |= def_sets[succ];
        }

        // now ensure that all pipeline stages have finished accessing the potentially killed resource
        for (size_t i = 0; i < n_tmp; ++i) {
            if (!kill[i]) continue;
            const auto& tmp = temporaries[i];
            // XXX what about queues here?
            // actually, if the resource was accessed in a different queue(s),
            // then there must be an execution dependency between the current task and all other queues
            for (size_t iq = 0; iq < max_queues; ++iq) {
                /*if (tmp.state.access_sn[iq]) {
                    if (pstate[iq].needs_execution_barrier(
                                tmp.state.access_sn[iq], tmp.state.stages)) {
                        // can't prove that the queue has finished accessing the resource, can't consider it dead
                        kill.reset(i);
                        break;
                    }
                }*/
            }
        }

        // add resources to the dead set.
        dead |= kill;

        // update live sets
        live -= kill;
        live_sets[task_index] = live;
    }

    /// @brief builds set of tasks ready to be submitted
    /// @return
    void update_ready_queue() {
        const auto n_task = tasks.size();

        for (size_t i = 0; i < n_task; ++i) {
            const auto& t = tasks[i];
            if (done_tasks[i] || ready_tasks[i]) continue;

            if (t.preds.size() == 0 || std::all_of(t.preds.begin(), t.preds.end(), [&](size_t i) {
                    return done_tasks[i];
                })) {
                // 1. max overlap
                size_t latest_pred_serial = 0;
                for (auto pred : t.preds) {
                    latest_pred_serial = std::max(latest_pred_serial, pred);
                }

                ready_tasks.set(i);
                ready_queue.push(ready_task{.latest_pred_serial = latest_pred_serial, .index = i});
            }
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

        per_queue_wait_serials wait_sn{};
        per_queue_wait_dst_stages wait_dst_stages{};
        uint32_t num_image_memory_barriers;
        uint32_t num_buffer_memory_barriers;
        vk::PipelineStageFlags src_stage_mask;
        vk::PipelineStageFlags dst_stage_mask;
        std::vector<vk::Semaphore> wait_semaphores;

        // build source stage mask
        for (auto p : t.preds) {
            src_stage_mask |= tasks[p].output_stage_mask;
        }

        // build memory barriers
        for (const auto& access : t.accesses) 
        {
            // issue: for each resource, 
            // we need the flush access mask, which we get from the accesses in predecessors
            // either:
            // - 1. scan the pred for the corresponding resource access and get the flush mask here (but might be flushed twice?)
            // - 2. track the flush mask on DAG creation (OK to track early since there's no reordering)

            assign_memory(access.index);
            //src_stage_mask |=

           /* memory_dependency(
                access.index, 
                access.details, 
                t.snn, 
                wait_sn,
                wait_dst_stages, 
                num_image_memory_barriers, 
                num_buffer_memory_barriers, 
                src_stage_mask,
                dst_stage_mask,
                wait_semaphores
                );*/
        }

        // --- update liveness sets
        update_liveness(task_index);


        // --- apply queue syncs
       /* bool sem_wait = false;  // cross queue sync
        for (size_t iq = 0; iq < max_queues; ++iq) {
            if (wait_sn[iq]) {
                // execution barrier necessary
                // if the exec barrier is on a different queue, then it's going to be a semaphore wait,
                // and this ensures that all stages have finished
                pstate[iq].apply_execution_barrier(
                        wait_sn[iq], iq != t.snn.queue ? vk::PipelineStageFlagBits::eBottomOfPipe
                                                     : st.src_stage_mask);
                sem_wait |= iq != t.snn.queue;
            }
        }*/
        //sem_wait |= !wait_binary_semaphores.empty();

        /*
        if (sem_wait) {
            // if a semaphore wait is necessary, must start a new batch
            submission_builder.start_batch(t.snn.queue, wait_sn, wait_dst_stages, {});
        }
        if (t.type() != task_type::present) {
            submission_builder.add_task(t.snn);
        } else {
            // present task, end the batch
            submission_builder.end_batch_and_present(t.snn.queue, {t.detail.present.swapchain},
                    {t.detail.present.image_index}, dev.create_binary_semaphore());
        }*/

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
                fmt::print("   {:02d} => MEM_{}\n", i, allocation_map[i]);
            } else {
                fmt::print("   {:02d} => no memory\n", i);
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

    size_t finish_pending_cb_batches() {
        submission_builder.finish();

        fmt::print(
                "====================================================\nCommand buffer batches:\n");
        for (auto&& submit : submission_builder.submits) 
        {
            fmt::print("- W={},{},{},{} WDST={},{},{},{} S={} ncb={}\n",
                    submit.wait_sn[0], submit.wait_sn[1], submit.wait_sn[2], submit.wait_sn[3],
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[0]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[1]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[2]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[3]),
                    (uint64_t) submit.signal_sn,
                    submit.command_buffers.size());
        }

        fmt::print(
                "====================================================\nCommand buffer indices:\n");
        for (size_t i = 0; i < tasks.size(); ++i) {
            fmt::print("- {} => {}\n", tasks[i].name, submitted_tasks[i].cb_index);
        }

        return 0;
    }

    bool schedule_next() {
        update_ready_queue();
        if (ready_queue.empty()) return false;
        size_t next_task = ready_queue.top().index;
        ready_queue.pop();
        schedule_task(next_task);
        return true;
    }
};

void dump_tasks(std::ostream& out, std::span<const task> tasks,
        std::span<const temporary> temporaries, std::span<const submitted_task> submitted_tasks) {
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
                if (is_read_access(a.dst_access_mask)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.dst_access_mask) << ")";
                }
            }
        }
        out << "\\lwrites=";
        {
            bool first = true;
            for (const auto& a : task.accesses) {
                if (is_write_access(a.dst_access_mask)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.dst_access_mask) << ")";
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

    //---------------------------------------------------
    // main scheduling loop

    // - determine next task
    // - determine execution and memory barriers and cross-queue barriers
    // - assign memory to resources
    // - update resource liveness
    // - allocate and bind memory to resources
    // - update resources with the known final states
    // - generate command buffers
    // - submit batches
    // - move all referenced resources into the wait queue
    // - pacing (wait for batch N-2)
    // - dump states

    //barriers_.merge_by_destination();
    //barriers_.dump(temporaries_);


    // dump
    for (const auto& t : tasks_) {
        fmt::print("task: {} ({}:{})\n", t.name, (uint64_t)t.snn.queue, (uint64_t)t.snn.serial);
        fmt::print("    input execution barrier: {}->{}\n", pipeline_stages_to_string(t.src_stage_mask), pipeline_stages_to_string(t.input_stage_mask));
        fmt::print("    input memory barriers: \n");
        for (const auto& imb : t.image_memory_barriers) 
        {
            fmt::print("        {} access:{}->{} layout:{}->{}\n",
                find_image_name(temporaries_, imb.image),
                access_mask_to_string(imb.srcAccessMask),
                access_mask_to_string(imb.dstAccessMask),
                layout_to_string(imb.oldLayout),
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
            fmt::print("    signal: {}:{}\n", (uint64_t)t.snn.queue, (uint64_t)t.snn.serial);
        }
        fmt::print("\n");
    }

    schedule_ctx ctx{device_, queue_indices_, batch_start_serial_ + 1, tasks_, temporaries_,
        {} };
    while (ctx.schedule_next()) {}

    {
        std::ofstream out_graphviz{
                fmt::format("graal_test_{}.dot", batch_start_serial_), std::ios::trunc };
        dump_tasks(out_graphviz, tasks_, temporaries_, ctx.submitted_tasks);
    }

    size_t cb_count = ctx.finish_pending_cb_batches();
    ctx.allocate_memory();

    //---------------------------------------------------
    // fill command buffers
    std::vector<vk::CommandBuffer> command_buffers;
    command_buffers.resize(cb_count, nullptr);
    batch_thread_resources thread_resources{.cb_pool = get_command_buffer_pool()};

    // fill command buffers
    for (size_t i = 0; i < tasks_.size(); ++i) {
        auto& t = tasks_[i];
        auto& st = ctx.submitted_tasks[i];

        const auto cb = thread_resources.cb_pool.fetch_command_buffer(vk_device);
        command_buffers[st.cb_index] = cb;

        cb.begin(vk::CommandBufferBeginInfo{});

        // emit the necessary pipeline barriers
        if (st.needs_cmd_pipeline_barrier()) {
            cb.pipelineBarrier(st.src_stage_mask, st.dst_stage_mask, {}, 0, nullptr,
                    st.num_buffer_memory_barriers,
                    ctx.buffer_memory_barriers.data() + st.buffer_memory_barriers_offset,
                    st.num_image_memory_barriers,
                    ctx.image_memory_barriers.data() + st.image_memory_barriers_offset);
        }

        if (t.type() == task_type::render_pass) {
            // TODO render pass
            if (t.d.u.render.callback) t.d.u.render.callback(nullptr, cb);
        } else if (t.type() == task_type::compute_pass) {
            if (t.d.u.compute.callback) t.d.u.compute.callback(cb);
        }

        cb.end();
    }

    //---------------------------------------------------
    // queue submission
    for (auto& b : ctx.submission_builder.submits) {
        //b.submit(device_.get_queue_by_index(b.signal_snn.queue), command_buffers, timelines_);
    }

    std::vector<resource_ptr> batch_resources;
    batch_resources.reserve(temporaries_.size());
    for (auto tmp : temporaries_) {
        // store last known state of the resource objects
        batch_resources.push_back(std::move(tmp.resource));
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
}

void queue_impl::wait_for_task(uint64_t sequence_number) {
    /*assert(batch_to_wait <= current_batch_);
    const auto vk_device = device_->get_vk_device();
    const vk::Semaphore semaphores[] = {batch_index_semaphore_};
    const uint64_t values[] = {batch_to_wait};
    const vk::SemaphoreWaitInfo wait_info{
            .semaphoreCount = 1, .pSemaphores = semaphores, .pValues = values};
    vk_device.waitSemaphores(wait_info, 10000000000);  // 10sec batch timeout

    // reclaim resources of all batches with index <= batch_to_wait (we proved
    // that they are finished by waiting on the semaphore)
    while (!in_flight_batches_.empty() && in_flight_batches_.front().batch_index <= batch_to_wait) {
        auto& b = in_flight_batches_.front();

        // reclaim command buffer pools
        for (auto& t : b.threads) {
            t.cb_pool.reset(vk_device);
            free_cb_pools_.recycle(std::move(t.cb_pool));
        }

        // reclaim semaphores
        free_semaphores_.recycle_vector(std::move(b.semaphores));
        in_flight_batches_.pop_front();
    }*/
}


/// @brief synchronizes two tasks
/// @param before
/// @param after Sequence number of the task to execute after. Must correspond to an unsubmitted task.
/// @param use_binary_semaphore whether to synchronize using a binary semaphore.
/// before < after; if use_binary_semaphore == true, then before must also correspond to an unsubmitted task.
void queue_impl::add_task_dependency(submission_number before_snn, task& after, vk::PipelineStageFlags wait_dst_stages) {
   
    if (before_snn.serial <= batch_start_serial_) {
        // --- Inter-batch synchronization w/ timeline semaphore
        after.wait = true;
        after.input_wait_serials[before_snn.queue] = std::max(after.input_wait_serials[before_snn.queue], before_snn.serial);
        after.input_wait_dst_stages[before_snn.queue] |= wait_dst_stages;
    } else {
        const auto local_task_index =
            before_snn.serial - batch_start_serial_ - 1;  // SN relative to start of batch
        auto& before = tasks_[local_task_index];

        // --- Intra-batch synchronization
        if (after.snn.queue != before_snn.queue) {
            // cross-queue dependency w/ timeline semaphore
            before.signal = true;
            after.wait = true;
            after.input_wait_serials[before_snn.queue] = std::max(after.input_wait_serials[before_snn.queue], before_snn.serial);
            after.input_wait_dst_stages[before_snn.queue] |= wait_dst_stages;
        }

        after.preds.push_back(local_task_index);
        before.succs.push_back(after.snn.serial - batch_start_serial_ - 1);
        after.src_stage_mask |= before.output_stage_mask;
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

    if (writing) {
        if (tmp.state.has_readers()) {
            // WAW
            if (tmp.state.writer.serial) {
                add_task_dependency(tmp.state.writer, t, access.input_stage);
            }
        }
        else {
            // WAR
            for (size_t i = 0; i < max_queues; ++i) {
                if (auto r = tmp.state.readers[i]) {
                    add_task_dependency(submission_number{ .queue = i, .serial = r }, t, access.input_stage);
                }
            }
        }
        tmp.state.clear_readers();
        tmp.state.writer = t.snn;
        t.output_stage_mask |= access.output_stage;
    }
    else {
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

    if (!writes_visible || layout_transition || write_after_write_hazard)
    {
        // no need to make memory visible if we're writing to the resource
        const auto dst_access_mask = !is_read_access(access.access_mask) ? vk::AccessFlags{} : access.access_mask;

        // the resource access needs a memory barrier
        if (auto img = tmp.as_image()) {
            // image barrier
            const auto format = img->format();
            const auto vk_image = img->vk_image();
            vk::ImageAspectFlags aspect_mask = format_aspect_mask(format);
            const vk::ImageSubresourceRange subresource_range =
                vk::ImageSubresourceRange{ .aspectMask = aspect_mask,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS };

            t.image_memory_barriers.push_back(
                vk::ImageMemoryBarrier{
                        .srcAccessMask = tmp.state.availability_mask,
                        .dstAccessMask = dst_access_mask,
                        .oldLayout = tmp.state.layout,
                        .newLayout = access.layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = vk_image,
                        .subresourceRange = subresource_range });

        }
        else if (auto buf = tmp.as_buffer()) {
            t.buffer_memory_barriers.push_back(
                vk::BufferMemoryBarrier{
                        .srcAccessMask = tmp.state.availability_mask,
                        .dstAccessMask = dst_access_mask,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = buf->vk_buffer(),
                        .offset = 0,
                        .size = VK_WHOLE_SIZE });
        }

        tmp.state.availability_mask = {};
        // update the access types that can now see the resource
        tmp.state.visibility_mask |= access.access_mask;
        tmp.state.layout = access.layout;
    }

    // all previous writes are flushed
    if (writing) {
        tmp.state.availability_mask |= access.access_mask;
    }
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