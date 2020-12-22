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
    explicit temporary(resource_ptr r) : resource{std::move(r)}, state{resource->last_state} {
        resource->last_state.wait_semaphore =
                nullptr;  // we take ownership of the semaphore, if there's one
    }

    /// @brief The resource
    resource_ptr resource;
    std::vector<submission_number> readers;
    submission_number writer{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    resource_last_access_info state;
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
using per_queue_wait_serials = std::array<serial_number, max_queues>;
using per_queue_wait_dst_stages = std::array<vk::PipelineStageFlags, max_queues>;

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

struct barrier {
    std::vector<serial_number> source;  // "before", or "left"
    std::vector<serial_number> destination;  // "after", or "right"

    // whether this is a semaphore wait or signal operation 
    // it's either a wait or a signal depending on whether dst_stage_mask is null or not
    bool semaphore_sync = false;      
    
    vk::PipelineStageFlags src_stage_mask;
    vk::PipelineStageFlags dst_stage_mask;
    std::vector<vk::ImageMemoryBarrier> image_memory_barriers;
    std::vector<vk::BufferMemoryBarrier> buffer_memory_barriers;

    void merge_with(barrier&& other) {
        src_stage_mask |= other.src_stage_mask;
        dst_stage_mask |= other.dst_stage_mask;
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
    }

    vk::ImageMemoryBarrier* find_image_memory_barrier(vk::Image image) {
        for (auto& b : image_memory_barriers) {
            if (b.image == image) { return &b; }
        }
        return nullptr;
    }

    vk::BufferMemoryBarrier* find_buffer_memory_barrier(vk::Buffer buffer) {
        for (auto& b : buffer_memory_barriers) {
            if (b.buffer == buffer) { return &b; }
        }
        return nullptr;
    }

    vk::ImageMemoryBarrier& get_image_memory_barrier(const image_resource& resource) {
        if (auto b = find_image_memory_barrier(resource.vk_image())) { return *b; }
        auto& b = image_memory_barriers.emplace_back();
        const auto format = resource.format();
        const auto aspect_mask = format_aspect_mask(format);
        const vk::ImageSubresourceRange subresource_range{.aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS};
        // setup the source side of the memory barrier
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = resource.vk_image();
        b.subresourceRange.aspectMask = aspect_mask;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
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

struct half_image_memory_barrier {
    vk::Image image;
    vk::AccessFlags access_mask;
    vk::ImageLayout layout;
};

struct half_buffer_memory_barrier {
    vk::Buffer buffer;
    vk::AccessFlags access_mask;
};

struct half_barrier {
    serial_number task;
    vk::PipelineStageFlags stage_mask;
    std::vector<half_image_memory_barrier> image_memory_barriers;
    std::vector<half_buffer_memory_barrier> buffer_memory_barriers;

    half_image_memory_barrier* find_image_memory_barrier(vk::Image image) {
        for (auto& b : image_memory_barriers) {
            if (b.image == image) { return &b; }
        }
        return nullptr;
    }

    half_buffer_memory_barrier* find_buffer_memory_barrier(vk::Buffer buffer) {
        for (auto& b : buffer_memory_barriers) {
            if (b.buffer == buffer) { return &b; }
        }
        return nullptr;
    }

    half_image_memory_barrier& get_image_memory_barrier(const image_resource& resource)
    {
        if (auto b = find_image_memory_barrier(resource.vk_image())) { return *b; }
        auto& b = image_memory_barriers.emplace_back();
        b.image = resource.vk_image();
        return b;
    }

    half_buffer_memory_barrier& get_buffer_memory_barrier(const buffer_resource& resource) {
        if (auto b = find_buffer_memory_barrier(resource.vk_buffer())) { return *b; }
        auto& b = buffer_memory_barriers.emplace_back();
        b.buffer = resource.vk_buffer();
        return b;
    }
};

struct destination_barrier {
    submission_number source;
    half_barrier dest;
};

struct barrier_builder {

    std::vector<half_barrier> source_barriers;
    std::vector<destination_barrier> destination_barriers;
    serial_number start_sn = 0;
    
    void reset(serial_number ssn) {
        start_sn = ssn;
        source_barriers.clear();
        destination_barriers.clear();
    }

    half_barrier& get_or_create_source_barrier(submission_number source) {
        if (auto it = std::find_if(source_barriers.begin(), source_barriers.end(), [=](const half_barrier& hb) { return hb.task == source.serial; });
            it != source_barriers.end()) {
            return *it; 
        }
        auto& b = source_barriers.emplace_back();
        b.task = source.serial;
        return b;
    }

    destination_barrier& get_or_create_destination_barrier(submission_number source, submission_number destination) {
        if (auto it = std::find_if(destination_barriers.begin(), destination_barriers.end(), [=](const destination_barrier& hb) {
            return hb.source.serial == source.serial && hb.dest.task == destination.serial; });
            it != destination_barriers.end()) {
            return *it;
        }
        auto& b = destination_barriers.emplace_back();
        b.dest.task = destination.serial;
        return b;
    }

    /// @brief Sets up the source half of an image memory dependency.
    /// @param resource
    /// @param producer
    /// @param src_stage_mask
    /// @param src_access_mask
    /// @param layout
    void image_memory_source_barrier(const image_resource& resource, submission_number producer,
            vk::PipelineStageFlags src_stage_mask, vk::AccessFlags src_access_mask,
            vk::ImageLayout layout) 
    {
        half_barrier& b = get_or_create_source_barrier(producer);
        b.stage_mask |= src_stage_mask;
        auto& imb = b.get_image_memory_barrier(resource);
        imb.access_mask |= src_access_mask;
        imb.layout = layout;
    }

    /// @brief Sets up the destination half of an image memory dependency.
    /// @param resource
    /// @param producer
    /// @param consumer
    /// @param dst_stage_mask
    /// @param dst_access_mask
    /// @param layout
    void image_memory_destination_barrier(
            const image_resource& resource, 
            submission_number producer,
            submission_number consumer,
            vk::PipelineStageFlags dst_stage_mask,
            vk::AccessFlags dst_access_mask, vk::ImageLayout layout)
    {
        destination_barrier& b = get_or_create_destination_barrier(producer, consumer);
        b.dest.stage_mask |= dst_stage_mask;
        auto& imb = b.dest.get_image_memory_barrier(resource);
        imb.access_mask |= dst_access_mask;
        imb.layout = layout;
    }

    /*void merge_by_destination() {
        // merge by common destination
        for (size_t i = 0; i < barriers.size(); ++i) {
            if (!barriers[i].source.empty()) {
                for (size_t j = i + 1; j < barriers.size(); ++j) {
                    if (barriers[i].destination == barriers[j].destination) {
                        barriers[i].merge_with(std::move(barriers[j]));
                        barriers[j].source.clear();
                    }
                }
            }
        }

        auto it = std::remove_if(barriers.begin(), barriers.end(),
                [](const barrier& b) { return b.source.empty(); });
        barriers.erase(it, barriers.end());
    }*/

    void dump(std::span<const temporary> temporaries) {
        // dump
        for (const auto& b : source_barriers) {
            fmt::print("SRC: {} ", b.task);
            fmt::print("\n    EX: {}->", pipeline_stages_to_string_compact(b.stage_mask));
            fmt::print("\n    IMB: \n");
            bool begin = true;
            for (size_t i = 0; i < b.image_memory_barriers.size(); ++i) {
                const auto& imb = b.image_memory_barriers[i];
                fmt::print("        {}({}->_)({}->_)\n", find_image_name(temporaries, imb.image),
                        access_mask_to_string_compact(imb.access_mask),
                        layout_to_string_compact(imb.layout));
            }
            fmt::print("\n");
        }
        for (const auto& b : destination_barriers) {
            fmt::print("DST: {}->{}", b.source.serial, b.dest.task);
            fmt::print("\n    EX: ->{}", pipeline_stages_to_string_compact(b.dest.stage_mask));
            fmt::print("\n    IMB: \n");
            bool begin = true;
            for (size_t i = 0; i < b.dest.image_memory_barriers.size(); ++i) {
                const auto& imb = b.dest.image_memory_barriers[i];
                fmt::print("        {}(_->{})(_->{})\n", find_image_name(temporaries, imb.image),
                    access_mask_to_string_compact(imb.access_mask),
                    layout_to_string_compact(imb.layout));
            }
            fmt::print("\n");
        }
    }
};

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
        return init_task(tasks_.emplace_back(rpd), name);
    }

    // returned reference only valid until next call to create_task or present.
    [[nodiscard]] task& create_compute_pass_task(std::string_view name) noexcept {
        return init_task(tasks_.emplace_back(), name);
    }

    void add_task_dependency(task& t, submission_number before);

    void add_resource_dependency(
            task& t, resource_ptr resource, const resource_access_details& access);

    [[nodiscard]] void* get_staging_buffer(
            size_t align, size_t size, vk::Buffer& out_buffer, vk::DeviceSize& out_offset) {
        return staging_.get_staging_buffer(device_.get_vk_device(), device_.get_allocator(),
                staging_buffers_, align, size, out_buffer, out_offset);
    }

private:
    task& init_task(task& t, std::string_view name) {
        last_serial_++;
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

    barrier_builder barriers_;
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
    return r.last_state.write_snn.serial > batch_start_serial_ && !r.has_user_refs();
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
    std::array<queue_submission, max_queues> pending{};
    std::vector<queue_submission> submits;

    void start_batch(uint32_t queue, per_queue_wait_serials wait_sn,
            per_queue_wait_dst_stages wait_dst_stages,
            std::vector<vk::Semaphore> wait_binary_semaphores) {
        for (size_t iq = 0; iq < max_queues; ++iq) {
            queue_submission& qs = pending[iq];
            if (wait_sn[iq] || iq == queue) { end_batch(iq); }
        }

        pending[queue].wait_sn = wait_sn;
        pending[queue].wait_dst_stages = wait_dst_stages;
        pending[queue].wait_binary_semaphores = std::move(wait_binary_semaphores);
    }

    void end_batch(uint32_t queue) {
        if (pending[queue].is_empty()) { return; }

        submits.push_back(pending[queue]);
        pending[queue] = {};
    }

    void batch_command_buffer(submission_number snn) {
        if (pending[snn.queue].first_sn == 0) {
            // set submission SN
            pending[snn.queue].first_sn = snn.serial;
        }
        pending[snn.queue].serials.push_back(snn.serial);
        pending[snn.queue].command_buffers.push_back(nullptr);
        pending[snn.queue].signal_sn = std::max(pending[snn.queue].signal_sn, snn.serial);
    }

    void end_batch_and_present(uint32_t queue, std::vector<vk::SwapchainKHR> swapchains,
            std::vector<uint32_t> swpachain_image_indices, vk::Semaphore render_finished) {
        // end the batch with a present (how considerate!)
        pending[queue].render_finished = render_finished;
        pending[queue].swapchains = std::move(swapchains);
        pending[queue].swpachain_image_indices = std::move(swpachain_image_indices);
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

        // prepare resources
        for (size_t i = 0; i < n_tmp; ++i) {
            // reset the temporary last write serial used to build the DAG
            temporaries[i].state.write_snn.serial = 0;
        }

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
                if (is_read_access(r.details.access_mask)) { use_sets[i].set(r.index); }
                if (is_write_access(r.details.access_mask)) { def_sets[i].set(r.index); }
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
        fmt::print("Resources:\n");
        for (size_t i = 0; i < temporaries.size(); ++i) {
            fmt::print("{:<10} {} W={}:{} A={},{},{},{} L={} AM={} S={}\n",
                    temporaries[i].resource->name(), live[i] ? "live" : "dead",
                    (uint64_t) temporaries[i].state.write_snn.queue,
                    (uint64_t) temporaries[i].state.write_snn.serial,
                    (uint64_t) temporaries[i].state.access_sn[0],
                    (uint64_t) temporaries[i].state.access_sn[1],
                    (uint64_t) temporaries[i].state.access_sn[2],
                    (uint64_t) temporaries[i].state.access_sn[3],
                    layout_to_string_compact(temporaries[i].state.layout),
                    access_mask_to_string_compact(temporaries[i].state.access_mask),
                    pipeline_stages_to_string_compact(temporaries[i].state.stages));
        }
        fmt::print("\n");
    }

    void infer_barriers() {
        struct sync_scope {
            bitset src;
            bitset dest;
        };
        std::vector<sync_scope> scopes;

        // pass 1: one barrier per edge, try to merge with common sources
        for (size_t i = 0; i < tasks.size(); ++i) {
            for (auto p : tasks[i].preds) {
                bool merged = false;
                for (auto& s : scopes) {
                    if (s.src[p]) {
                        s.dest.set(i);
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    auto& b = scopes.emplace_back();
                    b.src.resize(tasks.size() + 1);
                    b.dest.resize(tasks.size() + 1);
                    b.src.set(p);
                    b.dest.set(i);
                }
            }
        }

        // merge by common destination
        for (size_t i = 0; i < scopes.size(); ++i) {
            if (!scopes[i].src.none()) {
                for (size_t j = i + 1; j < scopes.size(); ++j) {
                    if (scopes[i].dest == scopes[j].dest) {
                        scopes[i].src |= scopes[j].src;
                        scopes[j].src.reset();
                    }
                }
            }
        }

        // remove empty scopes
        auto it = std::remove_if(
                scopes.begin(), scopes.end(), [](const sync_scope& ss) { return ss.src.none(); });
        scopes.erase(it, scopes.end());

        const auto n_scopes = scopes.size();

        // infer barriers
        std::vector<barrier> barriers;
        barriers.reserve(n_scopes);

        barrier_builder bb;

        for (size_t i = 0; i < tasks.size(); ++i) {
            for (const auto& a : tasks[i].accesses) {
                size_t producer;

                // reading (read-after-write dependency)
                if (a.details.input_stage) {
                    // need memory barrier from producer to current task
                    //bb.image_memory_destination_barrier()
                }
            }
        }

        for (auto& s : scopes) {
            auto& b = barriers.emplace_back();
            //b.src = s.src;
            //b.dest = s.dest;

            // setup the source side of the barriers
            for (size_t i = 0; i < tasks.size(); ++i) {
                if (s.src[i]) {
                    for (const auto& a : tasks[i].accesses) {
                        if (a.details.output_stage) {
                            b.src_stage_mask |= a.details.output_stage;
                            auto& resource = temporaries[a.index].resource;
                            if (auto img = resource->as_image()) {
                                auto& imb = b.get_image_memory_barrier(*img);
                                imb.srcAccessMask = a.details.access_mask;
                                imb.oldLayout = a.details.layout;
                            } else if (auto buf = resource->as_buffer()) {
                                auto& bmb = b.get_buffer_memory_barrier(*buf);
                                bmb.srcAccessMask = a.details.access_mask;
                            }
                        }
                    }
                }
            }

            // setup the destination side of the barriers
            for (size_t i = 0; i < tasks.size(); ++i) {
                if (s.dest[i]) {
                    for (const auto& a : tasks[i].accesses) {
                        b.dst_stage_mask |= a.details.input_stage;
                        auto& resource = temporaries[a.index].resource;
                        if (auto img = resource->as_image()) {
                            if (auto imb = b.find_image_memory_barrier(img->vk_image())) {
                                imb->dstAccessMask = a.details.access_mask;
                                imb->newLayout = a.details.layout;
                            } else {
                                fmt::print("{}, access {}: no barrier\n", tasks[i].name,
                                        temporaries[a.index].resource->name());
                            }
                        } else if (auto buf = resource->as_buffer()) {
                            if (auto bmb = b.find_buffer_memory_barrier(buf->vk_buffer())) {
                                bmb->dstAccessMask = a.details.access_mask;
                            }
                        }
                    }
                }
            }
        }

        
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

    /// @brief Infers the necessary barriers for the given access in the given task.
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
            std::vector<vk::Semaphore>& out_wait_semaphores) {
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
        // yes, if layout is different
        const bool needs_layout_transition = tmp.state.layout != access.layout;
        // covers RAW hazards, and visibility across different access types
        const bool visibility_hazard =
                (tmp.state.access_mask & access.access_mask) != access.access_mask;
        // covers WAW hazards across the same access type (writes must happen in order)
        const bool waw_hazard = writing && is_write_access(tmp.state.access_mask);

        if (needs_layout_transition || visibility_hazard || waw_hazard) {
            // the resource access needs a memory barrier
            if (auto img = tmp.resource->as_image()) {
                // image barrier

                // determine aspect mask
                const auto format = img->format();
                const auto vk_image = img->vk_image();
                assert(vk_image && "image not realized");

                // TODO move into a function (aspect_mask_from_format)
                vk::ImageAspectFlags aspect_mask;
                if (is_depth_only_format(format)) {
                    aspect_mask = vk::ImageAspectFlagBits::eDepth;
                } else if (is_stencil_only_format(format)) {
                    aspect_mask = vk::ImageAspectFlagBits::eStencil;
                } else if (is_depth_and_stencil_format(format)) {
                    aspect_mask =
                            vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
                } else {
                    aspect_mask = vk::ImageAspectFlagBits::eColor;
                }
                // TODO metadata? planes?

                const vk::ImageSubresourceRange subresource_range =
                        vk::ImageSubresourceRange{.aspectMask = aspect_mask,
                                .baseMipLevel = 0,
                                .levelCount = VK_REMAINING_MIP_LEVELS,
                                .baseArrayLayer = 0,
                                .layerCount = VK_REMAINING_ARRAY_LAYERS};

                image_memory_barriers.push_back(
                        vk::ImageMemoryBarrier{.srcAccessMask = tmp.state.access_mask,
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
                        vk::BufferMemoryBarrier{.srcAccessMask = tmp.state.access_mask,
                                .dstAccessMask = access.access_mask,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .buffer = buf->vk_buffer(),
                                .offset = 0,
                                .size = VK_WHOLE_SIZE});
                buffer_memory_barriers_temporaries.push_back(tmp_index);
                num_buffer_memory_barriers++;
            }
        }

        // --- update what we know about the resource after applying the barriers
        tmp.state.layout = access.layout;
        tmp.state.stages = access.output_stage;
        if (writing) {
            // The resource is modified, either through an actual write or a layout transition.
            // The last version of the contents is visible only for the access flags given in this task.
            tmp.state.write_snn = snn;
            for (size_t iq = 0; iq < max_queues; ++iq) {
                if (iq == snn.queue) {
                    tmp.state.access_sn[iq] = snn.serial;
                } else {
                    tmp.state.access_sn[iq] = 0;
                }
            }
            tmp.state.access_mask = access.access_mask;
        } else {
            // Read-only access, a barrier has been inserted so that all writes to the resource are visible.
            // Combine with the previous access flags.
            tmp.state.access_sn[snn.queue] = snn.serial;
            tmp.state.access_mask |= access.access_mask;
        }
    }

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
                if (tmp.state.access_sn[iq]) {
                    if (pstate[iq].needs_execution_barrier(
                                tmp.state.access_sn[iq], tmp.state.stages)) {
                        // can't prove that the queue has finished accessing the resource, can't consider it dead
                        kill.reset(i);
                        break;
                    }
                }
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

        const auto sn = next_sn++;
        task& t = tasks[task_index];

        // --- assign memory for all resources accessed in this task
        for (const auto& access : t.accesses) {
            assign_memory(access.index);
        }

        // --- update liveness sets
        update_liveness(task_index);

        per_queue_wait_serials wait_sn{};
        per_queue_wait_dst_stages wait_dst_stages{};

        // all barriers ready to be submitted
        for (auto& b : barriers) {
            bool ready = true;
            bool has_source = false;
            for (auto src : b.source) {
                if (src == t.snn.serial) { has_source = true; }
                if (done_tasks[src]) {
                    ready = false;
                    break;
                }
            }
        }

        // --- apply queue syncs
        bool sem_wait = false;  // cross queue sync
        for (size_t iq = 0; iq < max_queues; ++iq) {
            if (wait_sn[iq]) {
                /*// execution barrier necessary
                // if the exec barrier is on a different queue, then it's going to be a semaphore wait,
                // and this ensures that all stages have finished
                pstate[iq].apply_execution_barrier(
                        wait_sn[iq], iq != snn.queue ? vk::PipelineStageFlagBits::eBottomOfPipe
                                                     : st.src_stage_mask);*/
                sem_wait |= iq != t.snn.queue;
            }
        }
        //sem_wait |= !wait_binary_semaphores.empty();

        if (sem_wait) {
            // if a semaphore wait is necessary, must start a new batch
            submission_builder.start_batch(t.snn.queue, wait_sn, wait_dst_stages, {});
        }
        if (t.type() != task_type::present) {
            submission_builder.batch_command_buffer(t.snn);
        } else {
            // present task, end the batch
            submission_builder.end_batch_and_present(t.snn.queue, {t.detail.present.swapchain},
                    {t.detail.present.image_index}, dev.create_binary_semaphore());
        }

        // --- state dump
        /*fmt::print("====================================================\n"
                   "Task {} SNN {}:{}",
                t.name, q, sn);
        fmt::print(" | WAIT: ");
        {
            bool begin = true;
            for (size_t iq = 0; iq < max_queues; ++iq) {
                if (wait_sn[iq] != 0 && iq != q) {
                    // cross-queue semaphore wait
                    if (!begin) {
                        fmt::print(",");
                    } else
                        begin = false;
                    fmt::print("{}:{}->{}", (uint64_t) iq, (uint64_t) wait_sn[iq],
                            pipeline_stages_to_string_compact(wait_dst_stages[iq]));
                }
            }
        }
        if (st.src_stage_mask || st.dst_stage_mask) {
            fmt::print(" | EX: {} -> {}", pipeline_stages_to_string_compact(st.src_stage_mask),
                    pipeline_stages_to_string_compact(st.dst_stage_mask));
        }
        if (st.num_buffer_memory_barriers) {
            fmt::print(" | BMB: ");
            bool begin = true;
            for (size_t i = 0; i < st.num_buffer_memory_barriers; ++i) {
                const auto& s = buffer_memory_barriers[st.buffer_memory_barriers_offset + i];
                if (!begin) {
                    fmt::print(",");
                } else
                    begin = false;
                fmt::print("{}({}->{})",
                        temporaries[buffer_memory_barriers_temporaries
                                            [st.buffer_memory_barriers_offset + i]]
                                .resource->name(),
                        access_mask_to_string_compact(s.srcAccessMask),
                        access_mask_to_string_compact(s.dstAccessMask));
            }
        }
        if (!st.num_image_memory_barriers) {
            fmt::print(" | IMB: ");
            bool begin = true;
            for (size_t i = 0; i < st.num_image_memory_barriers; ++i) {
                const auto& s = image_memory_barriers[st.image_memory_barriers_offset + i];
                if (!begin) {
                    fmt::print(",");
                } else
                    begin = false;
                fmt::print("{}({}->{})({}->{})",
                        temporaries[image_memory_barriers_temporaries
                                            [st.buffer_memory_barriers_offset + i]]
                                .resource->name(),
                        access_mask_to_string_compact(s.srcAccessMask),
                        access_mask_to_string_compact(s.dstAccessMask),
                        layout_to_string_compact(s.oldLayout),
                        layout_to_string_compact(s.newLayout));
            }
        }
        fmt::print("\n");
        dump();*/
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
        const auto& submitted_task = submitted_tasks[index];

        out << "t_" << index << " [shape=record style=filled fillcolor=\"";
        switch (submitted_task.snn.queue) {
            case 0: out << "#ffff99"; break;
            case 1: out << "#7fc97f"; break;
            case 2: out << "#fdc086"; break;
            case 3:
            default: out << "#beaed4"; break;
        }

        out << "\" label=\"{";
        out << task.name;
        out << "|sn=" << submitted_task.snn.serial << "(q=" << submitted_task.snn.queue << ")";
        out << "\\lreads=";
        {
            bool first = true;
            for (const auto& a : task.accesses) {
                if (is_read_access(a.details.access_mask)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.details.access_mask) << ")";
                }
            }
        }
        out << "\\lwrites=";
        {
            bool first = true;
            for (const auto& a : task.accesses) {
                if (is_write_access(a.details.access_mask)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.details.access_mask) << ")";
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
    barriers_.dump(temporaries_);

    // dump
    for (const auto& t : tasks_) {
        fmt::print("task: {} ({}:{})", t.name, (uint64_t)t.snn.queue, (uint64_t)t.snn.serial);
        fmt::print("\n    accesses: \n");
        for (const auto& a : t.accesses) {
            fmt::print("        {} ({},{}) in:{} out:{} pred:",
                temporaries_[a.index].resource->name(),
                access_mask_to_string_compact(a.details.access_mask),
                layout_to_string_compact(a.details.layout),
                pipeline_stages_to_string_compact(a.details.input_stage),
                pipeline_stages_to_string_compact(a.details.output_stage));
            dump_vector_set(a.preds);
            fmt::print("\n");
        }
        fmt::print("\n");
    }

    schedule_ctx ctx{device_, queue_indices_, batch_start_serial_ + 1, tasks_, temporaries_,
        {} };
    while (ctx.schedule_next()) {}
    size_t cb_count = ctx.finish_pending_cb_batches();
    ctx.allocate_memory();

    // the state so far:
    // - `ctx.submitted_tasks` contains a list of all submitted tasks, in order
    // - each entry in `ctx.submits` corresponds to a vkQueueSubmit
    // - `cb_count` is the total number of command buffers necessary
    // - `cctx.submitted_tasks[i].cb_index` is the index of the command buffer for task i

    {
        std::ofstream out_graphviz{
                fmt::format("graal_test_{}.dot", batch_start_serial_), std::ios::trunc};
        dump_tasks(out_graphviz, tasks_, temporaries_, ctx.submitted_tasks);
    }

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
            if (t.detail.render.callback) t.detail.render.callback(nullptr, cb);
        } else if (t.type() == task_type::compute_pass) {
            if (t.detail.compute.callback) t.detail.compute.callback(cb);
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
        tmp.resource->last_state = tmp.state;
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
            "present");
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


/// @brief Determines on which queue the task will be running on
/*void queue_impl::determine_queues() {
    for (size_t i = 0; i < tasks.size(); ++i) {
        task& t = tasks[i];
        uint8_t q = queues.graphics;

        switch (t.type()) {
        case task_type::render_pass: q = queues.graphics; break;
        case task_type::compute_pass: {
            // this is a bit more complicated: we might want to run the compute pass asynchronously, but
            // it's useless if the graphics queue is starved in the meantime (i.e. if the task is on a critical path).
            // we need to determine if there's another task that can run in parallel on the main queue.
            bool is_critical_pass = true;
            for (size_t j = 0; j < tasks.size(); ++j) {
                if (i != j && !reachability[i][j] && !reachability[j][i]) {
                    // parallel execution possible
                    if (tasks[j].signal.queue == queues.graphics) {
                        // already scheduled on the graphics queue
                        is_critical_pass = false;
                        break;
                    }
                    else {
                        // not yet scheduled
                        if (auto ty = tasks[j].type(); ty == task_type::render_pass
                            || ty == task_type::compute_pass) {
                            // but it's OK, because there's another pass that can run in parallel
                            // on the main queue
                            is_critical_pass = false;
                            break;
                        }
                    }
                }
            }
            if (is_critical_pass) {
                q = queues.graphics;
            }
            else {
                q = queues.compute;
            }
            break;
        }
        case task_type::present: q = queues.present; break;
        case task_type::transfer: q = queues.transfer; break;
        }

        t.snn.queue = q;
    }
}*/


/// @brief synchronizes two tasks
/// @param before
/// @param after Sequence number of the task to execute after. Must correspond to an unsubmitted task.
/// @param use_binary_semaphore whether to synchronize using a binary semaphore.
/// before < after; if use_binary_semaphore == true, then before must also correspond to an unsubmitted task.
void queue_impl::add_task_dependency(task& t, submission_number before) {
    //assert(t.sn > before.serial);

    /*if (before <= completed_serial_) {
        // no sync needed, task has already completed
        return;
    }*/

    if (before.serial <= batch_start_serial_) {
        // --- Inter-batch synchronization w/ timeline
        t.waits[before.queue] = std::max(t.waits[before.queue], before.serial);
    } else {
        // --- Intra-batch synchronization
        const auto local_task_index =
                before.serial - batch_start_serial_ - 1;  // SN relative to start of batch
        auto& before_task = tasks_[local_task_index];
        t.preds.push_back(local_task_index);
        before_task.succs.push_back(t.snn.serial - batch_start_serial_ - 1);
    }
}

/// @brief Registers a resource access in a submit task
/// @param resource
/// @param mode
void queue_impl::add_resource_dependency(
        task& t, resource_ptr resource, const resource_access_details& access) {
    const auto tmp_index = get_resource_tmp_index(resource);
    auto& tmp = temporaries_[tmp_index];

    const auto producer = tmp.state.write_snn;

    const bool writing = !!access.output_stage;

    /*if (auto img = resource->as_image()) {
        if (writing) {
            // writing, barrier on all readers and writer
            for (auto reader : tmp.readers) {
                barriers_.image_memory_destination_barrier(*img, reader, t.snn,
                        access.input_stage, access.access_mask, access.layout);
            }
            if (tmp.writer.serial) {
                barriers_.image_memory_destination_barrier(*img, tmp.writer, t.snn,
                        access.input_stage, access.access_mask, access.layout);
            }
            if (access.output_stage) {
                barriers_.image_memory_source_barrier(
                        *img, t.snn, access.output_stage, access.access_mask, access.layout);
            }
        } else {
            add_task_dependency(t, tmp.writer);
            barriers_.image_memory_destination_barrier(*img, tmp.writer, t.snn,
                    access.input_stage, access.access_mask, access.layout);
        }
    }
    else if (auto buf = resource->as_buffer()) {

    }*/

    std::vector<submission_number> preds;
    if (writing) {
        if (tmp.readers.empty()) {
            // WAW
            if (tmp.writer.serial) {
                add_task_dependency(t, tmp.writer);
                preds.push_back(tmp.writer);
            }
        }
        else {
            // WAR
            for (auto r : tmp.readers) {
                add_task_dependency(t, r);
            }
            preds = std::move(tmp.readers);
        }
        tmp.readers.clear();
        tmp.writer = t.snn;
        tmp.layout = access.layout;
    }
    else {
        // RAW
        // a read without a write is probably an uninitialized access
        if (tmp.writer.serial) {
            add_task_dependency(t, tmp.writer);
            preds.push_back(tmp.writer);
        } 
        tmp.readers.push_back(t.snn);
    }

    // add resource dependency to the task and update resource usage flags
    t.accesses.push_back(
         task::resource_access{
            .index = tmp_index, 
            .preds = std::move(preds),
            .details = access});
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

detail::task& queue::create_compute_pass_task(std::string_view name) noexcept {
    return impl_->create_compute_pass_task(name);
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