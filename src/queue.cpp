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
        //type = resource->type();
        access_sn = resource->last_access;
        write_snn = resource->last_write;
        layout = resource->last_layout;
        access_flags = resource->last_access_flags;
        stages = resource->last_pipeline_stages;
    }

    /// @brief The resource
    resource_ptr resource;
    // last access SN
    std::array<serial_number, max_queues> access_sn;
    // last write SN (a dummy one is first assigned when building the DAG)
    submission_number write_snn;
    // last known image layout, ignored for buffers
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    // binary semaphore to wait for
    vk::Semaphore wait_semaphore;
    vk::AccessFlags access_flags{};
    vk::PipelineStageFlags stages{};
};

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

    void add_task_dependency(task& t, submission_number before, bool use_binary_semaphore);

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
        t.sn = last_serial_;  // temporary SNN for DAG creation
        if (!name.empty()) {
            t.name = name;
        } else {
            t.name = fmt::format("T_{}", t.sn);
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
    serial_number last_serial_ = 0;
    serial_number completed_serial_ = 0;
    vk::Semaphore timelines_[max_queues];

    // --- current batch ---
    serial_number batch_start_serial_;
    std::vector<temporary> temporaries_;
    std::vector<task> tasks_;

    std::deque<submitted_batch> in_flight_batches_;

    staging_pool staging_;
    recycler<staging_buffer> staging_buffers_;

    recycler<command_buffer_pool> cb_pools;
    resource_to_temporary_map tmp_indices;  // for the current batch
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
        // just inserted, create temporary
        const auto last_write = resource->last_write.serial;
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
    return r.last_write.serial > batch_start_serial_ && !r.has_user_refs();
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

using per_queue_wait_serials = std::array<serial_number, max_queues>;
using per_queue_wait_dst_stages = std::array<vk::PipelineStageFlags, max_queues>;

/// @brief Represents a queue operation (either a call to vkQueueSubmit or vkQueuePresent)
struct queue_submission {
    per_queue_wait_serials wait_sn{};
    per_queue_wait_dst_stages wait_dst_stages{};
    serial_number first_sn;
    submission_number signal_snn;
    size_t cb_offset = 0;
    size_t cb_count = 0;
    std::vector<vk::Semaphore> wait_binary_semaphores;
    std::vector<vk::Semaphore> signal_binary_semaphores;
    vk::Semaphore render_finished;
    std::vector<vk::SwapchainKHR> swapchains;
    std::vector<uint32_t> swpachain_image_indices;

    [[nodiscard]] bool is_empty() const noexcept {
        return (cb_count == 0) && signal_binary_semaphores.empty() && render_finished;
    }

    void submit(vk::Queue queue, std::span<vk::CommandBuffer> command_buffers,
            std::span<vk::Semaphore> timelines) {
        const auto signal_semaphore_value = signal_snn.serial;

        std::vector<vk::Semaphore> signal_semaphores;
        signal_semaphores.push_back(timelines[signal_snn.queue]);
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
                .commandBufferCount = (uint32_t) cb_count,
                .pCommandBuffers = command_buffers.data() + cb_offset,
                .signalSemaphoreCount = (uint32_t) signal_semaphores.size(),
                .pSignalSemaphores = signal_semaphores.data()};

        if (auto result = queue.submit(1, &submit_info, nullptr); result != vk::Result::eSuccess) {
            fmt::print("vkQueueSubmit failed: {}\n", result);
        }

        if (render_finished) {
            // terminate with a present
            vk::PresentInfoKHR present_info{.waitSemaphoreCount = 1,
                    .pWaitSemaphores = &render_finished,
                    .swapchainCount = swapchains.size(),
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

        pending[queue].cb_offset = next_cb_index;
        submits.push_back(pending[queue]);
        next_cb_index += pending[queue].cb_count;
        pending[queue] = {};
    }

    void batch_command_buffer(submission_number snn) {
        pending[snn.queue].cb_count++;
        pending[snn.queue].signal_snn = snn;
        if (pending[snn.queue].cb_count == 0) {
            // set submission SN
            pending[snn.queue].first_sn = snn.serial;
        }
    }

    void end_batch_and_present(uint32_t queue, std::vector<vk::SwapchainKHR> swapchains,
            std::vector<uint32_t> swpachain_image_indices, vk::Semaphore render_finished) {
        // end the batch with a present (how considerate!)
        pending[queue].render_finished = render_finished;
        pending[queue].swapchains = std::move(swapchains);
        pending[queue].swpachain_image_indices = std::move(swpachain_image_indices);
        end_batch(queue);
    }

    size_t finish() {
        for (size_t i = 0; i < max_queues; ++i) {
            if (!pending[i].is_empty()) { end_batch(i); }
        }
        return next_cb_index;
    }

    size_t next_cb_index = 0;
    std::array<queue_submission, max_queues> pending{};
    std::vector<queue_submission> submits;
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
            std::span<temporary> temporaries) :
        dev{d},
        queues{queues}, base_sn{base_sn}, next_sn{base_sn}, tasks{tasks}, temporaries{temporaries} {
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
            temporaries[i].write_snn.serial = 0;
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
                if (is_read_access(r.details.access_flags)) { use_sets[i].set(r.index); }
                if (is_write_access(r.details.access_flags)) { def_sets[i].set(r.index); }
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
                    (uint64_t) temporaries[i].write_snn.queue,
                    (uint64_t) temporaries[i].write_snn.serial,
                    (uint64_t) temporaries[i].access_sn[0], (uint64_t) temporaries[i].access_sn[1],
                    (uint64_t) temporaries[i].access_sn[2], (uint64_t) temporaries[i].access_sn[3],
                    layout_to_string_compact(temporaries[i].layout),
                    access_mask_to_string_compact(temporaries[i].access_flags),
                    pipeline_stages_to_string_compact(temporaries[i].stages));
        }
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
        const bool reading = is_read_access(access.access_flags);

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
        const bool writing = is_write_access(access.access_flags) || access.layout != tmp.layout
                             || tmp.wait_semaphore;

        // --- is an execution barrier necessary for this access?
        if (writing) {
            for (size_t iq = 0; iq < max_queues; ++iq) {
                // WA* hazard: need to sync w.r.t. last accesses across ALL queues
                if (tmp.access_sn[iq]) {
                    // resource accessed on queue iq
                    wait_sn[iq] = std::max(wait_sn[iq], tmp.access_sn[iq]);
                    if (iq == snn.queue) {
                        // access on same queue, will use a pipeline barrier
                        src_stage_mask |= tmp.stages;
                    }
                }
                wait_dst_stages[iq] |= access.input_stage;
            }
        } else {
            // RAW hazard: need queue sync w.r.t last write
            wait_sn[tmp.write_snn.queue] =
                    std::max(wait_sn[tmp.write_snn.queue], tmp.write_snn.serial);
            if (tmp.write_snn.queue == snn.queue) { src_stage_mask |= tmp.stages; }
            wait_dst_stages[tmp.write_snn.queue] |= access.input_stage;
        }

        // --- is a semaphore wait necessary for this access?
        if (tmp.wait_semaphore) {
            // consume the semaphore
            out_wait_semaphores.push_back(std::exchange(tmp.wait_semaphore, nullptr));
        }

        dst_stage_mask |= access.input_stage;

        // --- is a memory barrier necessary for this access?
        // yes, if layout is different
        const bool needs_layout_transition = tmp.layout != access.layout;
        // covers RAW hazards, and visibility across different access types
        const bool visibility_hazard =
                (tmp.access_flags & access.access_flags) != access.access_flags;
        // covers WAW hazards across the same access type (writes must happen in order)
        const bool waw_hazard = writing && is_write_access(tmp.access_flags);

        if (needs_layout_transition || visibility_hazard || waw_hazard) {
            // the resource access needs a memory barrier
            if (tmp.resource->type() == resource_type::image
                    || tmp.resource->type() == resource_type::swapchain_image) {
                // image barrier

                // determine aspect mask
                const auto format = tmp.resource->as_image().format();
                const auto vk_image = tmp.resource->as_image().vk_image();
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
                        vk::ImageMemoryBarrier{.srcAccessMask = tmp.access_flags,
                                .dstAccessMask = access.access_flags,
                                .oldLayout = tmp.layout,
                                .newLayout = access.layout,
                                .image = tmp.resource->as_image().vk_image(),
                                .subresourceRange = subresource_range});
                image_memory_barriers_temporaries.push_back(tmp_index);
                num_image_memory_barriers++;

            } else if (tmp.resource->type() == resource_type::buffer) {
                buffer_memory_barriers.push_back(
                        vk::BufferMemoryBarrier{.srcAccessMask = tmp.access_flags,
                                .dstAccessMask = access.access_flags,
                                .buffer = tmp.resource->as_buffer().vk_buffer(),
                                .offset = 0,
                                .size = VK_WHOLE_SIZE});
                buffer_memory_barriers_temporaries.push_back(tmp_index);
                num_buffer_memory_barriers++;
            }
        }

        // --- update what we know about the resource after applying the barriers
        tmp.layout = access.layout;
        tmp.stages = access.output_stage;
        if (writing) {
            // The resource is modified, either through an actual write or a layout transition.
            // The last version of the contents is visible only for the access flags given in this task.
            tmp.write_snn = snn;
            for (size_t iq = 0; iq < max_queues; ++iq) {
                if (iq == snn.queue) {
                    tmp.access_sn[iq] = snn.serial;
                } else {
                    tmp.access_sn[iq] = 0;
                }
            }
            tmp.access_flags = access.access_flags;
        } else {
            // Read-only access, a barrier has been inserted so that all writes to the resource are visible.
            // Combine with the previous access flags.
            tmp.access_sn[snn.queue] = snn.serial;
            tmp.access_flags |= access.access_flags;
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
            const auto& rs = temporaries[i];
            // XXX what about queues here?
            // actually, if the resource was accessed in a different queue(s),
            // then there must be an execution dependency between the current task and all other queues
            for (size_t iq = 0; iq < max_queues; ++iq) {
                if (rs.access_sn[iq]) {
                    if (pstate[iq].needs_execution_barrier(rs.access_sn[iq], rs.stages)) {
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

        //t.snn.serial = sn;  // replace the temporary serial with the final one

        // --- determine which on which queue to run the task
        size_t q = queues.graphics;
        switch (t.type()) {
            case task_type::compute_pass:
                // a compute pass is eligible for async compute, but it must not be
                // a critical task (i.e. the graphics queue must not starve in the meantime).
                q = !ready_queue.empty() ? queues.compute : queues.graphics;
                break;
            case task_type::render_pass: q = queues.graphics; break;
            case task_type::present: q = queues.present; break;
            case task_type::transfer: q = queues.transfer; break;
        }

        submission_number snn{.queue = q, .serial = sn};

        submitted_task& st = submitted_tasks[sn - base_sn];
        st.snn = snn;

        // --- assign memory for all resources accessed in this task
        for (const auto& access : t.accesses) {
            assign_memory(access.index);
        }

        // --- infer execution & memory dependencies from resource accesses
        per_queue_wait_serials wait_sn{};
        per_queue_wait_dst_stages wait_dst_stages{};
        st.image_memory_barriers_offset = image_memory_barriers.size();
        st.buffer_memory_barriers_offset = buffer_memory_barriers.size();
        std::vector<vk::Semaphore> wait_binary_semaphores;
        for (const auto& access : t.accesses) {
            memory_dependency(access.index, access.details, snn, wait_sn, wait_dst_stages,
                    st.num_image_memory_barriers, st.num_buffer_memory_barriers, st.src_stage_mask,
                    st.dst_stage_mask, wait_binary_semaphores);
        }

        if (!st.src_stage_mask) { st.src_stage_mask = vk::PipelineStageFlagBits::eTopOfPipe; }
        if (!st.dst_stage_mask) { st.src_stage_mask = vk::PipelineStageFlagBits::eBottomOfPipe; }

        // --- update liveness sets
        update_liveness(task_index);

        // --- apply queue syncs
        bool sem_wait = false;  // cross queue sync
        for (size_t iq = 0; iq < max_queues; ++iq) {
            if (wait_sn[iq]) {
                // execution barrier necessary
                // if the exec barrier is on a different queue, then it's going to be a semaphore wait,
                // and this ensures that all stages have finished
                pstate[iq].apply_execution_barrier(
                        wait_sn[iq], iq != snn.queue ? vk::PipelineStageFlagBits::eBottomOfPipe
                                                     : st.src_stage_mask);
                sem_wait |= iq != snn.queue;
            }
        }
        sem_wait |= !wait_binary_semaphores.empty();

        if (sem_wait) {
            // if a semaphore wait is necessary, must start a new batch
            submission_builder.start_batch(
                    snn.queue, wait_sn, wait_dst_stages, wait_binary_semaphores);
        }
        if (t.type() != task_type::present) {
            submission_builder.batch_command_buffer(snn);
        } else {
            // present task, end the batch
            submission_builder.end_batch_and_present(snn.queue, {t.detail.present.swapchain},
                    {t.detail.present.image_index}, dev.create_binary_semaphore());
        }

        // --- state dump
        fmt::print("====================================================\n"
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
        dump();
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
        const auto num_cb = submission_builder.finish();

        fmt::print(
                "====================================================\nCommand buffer batches:\n");
        for (auto&& submit : submission_builder.submits) {
            fmt::print("- W={},{},{},{} WDST={},{},{},{} S={}:{} cb_offset={} ncb={}\n",
                    submit.wait_sn[0], submit.wait_sn[1], submit.wait_sn[2], submit.wait_sn[3],
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[0]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[1]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[2]),
                    pipeline_stages_to_string_compact(submit.wait_dst_stages[3]),
                    (uint64_t) submit.signal_snn.queue, (uint64_t) submit.signal_snn.serial,
                    submit.cb_offset, submit.cb_count);

            size_t i_first = submit.first_sn - base_sn;
            size_t i_last = submit.signal_snn.serial - base_sn;
            size_t icb = submit.cb_offset;
            for (size_t i = i_first; i <= i_last; ++i) {
                if (submitted_tasks[i].snn.queue == submit.signal_snn.queue) {
                    submitted_tasks[i].cb_index = icb++;
                }
            }
        }

        fmt::print(
                "====================================================\nCommand buffer indices:\n");
        for (size_t i = 0; i < tasks.size(); ++i) {
            fmt::print("- {} => {}\n", tasks[i].name, submitted_tasks[i].cb_index);
        }

        return num_cb;
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
                if (is_read_access(a.details.access_flags)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.details.access_flags) << ")";
                }
            }
        }
        out << "\\lwrites=";
        {
            bool first = true;
            for (const auto& a : task.accesses) {
                if (is_write_access(a.details.access_flags)) {
                    if (!first) {
                        out << ",";
                    } else {
                        first = false;
                    }
                    out << temporaries[a.index].resource->name();
                    out << "(" << access_mask_to_string_compact(a.details.access_flags) << ")";
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

    schedule_ctx ctx{device_, queue_indices_, batch_start_serial_ + 1, tasks_, temporaries_};
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
        b.submit(device_.get_queue_by_index(b.signal_snn.queue), command_buffers, timelines_);
    }

    std::vector<resource_ptr> batch_resources;
    batch_resources.reserve(temporaries_.size());
    for (auto tmp : temporaries_) {
        // store last know state of the resource objects
        tmp.resource->last_access = tmp.access_sn;
        tmp.resource->last_access_flags = tmp.access_flags;
        tmp.resource->last_pipeline_stages = tmp.stages;
        tmp.resource->last_write = tmp.write_snn;
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
            resource_access_details{
                    .layout = vk::ImageLayout::ePresentSrcKHR,
                    .access_flags = vk::AccessFlagBits::eMemoryRead,
                    .input_stage = vk::PipelineStageFlagBits::eAllCommands,
                    .binary_semaphore = true,
            });
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
void queue_impl::add_task_dependency(task& t, submission_number before, bool use_binary_semaphore) {
    assert(t.sn > before.serial);

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
        before_task.succs.push_back(t.sn - batch_start_serial_ - 1);
    }
}

/// @brief Registers a resource access in a submit task
/// @param resource
/// @param mode
void queue_impl::add_resource_dependency(
        task& t, resource_ptr resource, const resource_access_details& access) {
    const auto tmp_index = get_resource_tmp_index(resource);
    auto& tmp = temporaries_[tmp_index];

    // does the resource carry a semaphore? if so, extract it and sync on that
    if (auto semaphore = std::exchange(resource->wait_semaphore, nullptr)) {
        t.wait_binary.push_back(semaphore);
    } else {
        // the resource does not carry a semaphore, sync on timeline
        add_task_dependency(t, tmp.write_snn, false);
    }

    //const bool reading = is_read_access(access.access_flags);
    const bool writing = is_write_access(access.access_flags);

    if (writing) {
        // we're writing to this resource: set last write SN
        tmp.write_snn.serial = t.sn;
    }

    // add resource dependency to the task and update resource usage flags
    t.accesses.push_back(task::resource_access{.index = tmp_index, .details = access});
}

}  // namespace detail

//-----------------------------------------------------------------------------

// Called by buffer accessors to register an use of a buffer in a task
void handler::add_buffer_access(std::shared_ptr<detail::buffer_resource> buffer,
        vk::AccessFlags access_flags, vk::PipelineStageFlags input_stage,
        vk::PipelineStageFlags output_stage) const {
    detail::resource_access_details details;
    details.layout = vk::ImageLayout::eUndefined;
    details.access_flags = access_flags;
    details.input_stage |= input_stage;
    details.output_stage |= output_stage;
    queue_.add_resource_dependency(task_, buffer, details);
}

// Called by image accessors to register an use of an image in a task
void handler::add_image_access(std::shared_ptr<detail::image_resource> image,
        vk::AccessFlags access_flags, vk::PipelineStageFlags input_stage,
        vk::PipelineStageFlags output_stage, vk::ImageLayout layout) const {
    detail::resource_access_details details;
    details.layout = layout;
    details.access_flags = access_flags;
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