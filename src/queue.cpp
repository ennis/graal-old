#include <graal/detail/command_buffer_pool.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/queue.hpp>

#include <fstream>
#include <chrono>
#include <fmt/format.h>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <fstream>
#include <numeric>
#include <span>

namespace graal {
namespace detail {
namespace {

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
    // TODO dedicated allocs?
    if (req_a.flags != req_b.flags) return false;
    // don't alias if the memory type bits are not strictly the same
    if (req_a.memory_type_bits != req_b.memory_type_bits) return false;

    req_a.alignment = std::max(req_a.alignment, req_b.alignment);
    req_a.size = std::max(req_a.size, req_b.size);
    return true;
}

/// @brief Whether two tasks can be merged inside the same submission
/// @param
/// @return
bool needs_synchronization(const task& a, const task& b) {
    return a.signal.queue != b.signal.queue || a.type != b.type;  // TODO async compute
}

std::string get_task_name(task_index index, const task& task) {
    if (task.name.empty()) { return fmt::format("#{}", index); }
    return task.name;
}

std::string get_object_name(std::size_t index, const named_object& obj) {
    if (obj.name().empty()) { return fmt::format("#{}", index); }
    return std::string{obj.name()};
}


using variable_set = boost::dynamic_bitset<uint64_t>;
void dump_tasks(std::ostream& out, std::span<std::unique_ptr<task>> tasks,
        std::span<resource_ptr> temporaries, std::span<variable_set> live_sets) {
    out << "digraph G {\n";
    out << "node [shape=record fontname=Consolas];\n";

    for (task_index index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];

        out << "t_" << index << " [shape=record label=\"{";
        out << index;
        if (!task->name.empty()) { out << "(" << task->name << ")"; }
        out << "|{{reads\\l|writes\\l|live\\l}|{";
        {
            int i = 0;
            for (auto r : task->reads) {
                out << get_object_name(r, *temporaries[r]);
                if (i != task->reads.size() - 1) { out << ","; }
                ++i;
            }
        }
        out << "\\l|";
        {
            int i = 0;
            for (auto w : task->writes) {
                out << get_object_name(w, *temporaries[w]);
                if (i != task->writes.size() - 1) { out << ","; }
                ++i;
            }
        }
        out << "\\l|";

        {
            for (size_t t = 0; t < temporaries.size(); ++t) {
                if (live_sets[index].test(t)) {
                    out << get_object_name(t, *temporaries[t]);
                    if (t != live_sets[index].size() - 1) { out << ","; }
                }
            }
        }

        out << "\\l}}}\"]\n";
    }

    for (task_index i = 0; i < tasks.size(); ++i) {
        for (auto pred : tasks[i]->preds) {
            out << "t_" << pred << " -> "
                << "t_" << i << "\n";
        }
    }

    out << "}\n";
}


struct allocate_batch_memory_results
{

};

allocate_batch_memory_results allocate_batch_memory(
    device_impl_ptr device,
    serial_number start_sn,
    std::span<std::unique_ptr<task>> tasks,
    std::span<resource_ptr> temporaries)
{
    // --- start perf timer for submission
    namespace chrono = std::chrono;
    const auto start = chrono::high_resolution_clock::now();

    const size_t n_task = tasks.size();
    const size_t n_tmp = temporaries.size();

    // Before submitting a task, we assign concrete resources
    // to all virtual resources used in the batch (=="temporaries"). We strive
    // to minimize total memory usage by "aliasing" a single concrete resource
    // to multiple temporaries if we can determine that those are never in use
    // ("alive") at the same time during execution. This can be done by keeping
    // track of the "liveness" of temporaries during submisson (see
    // https://en.wikipedia.org/wiki/Live_variable_analysis).
    //
    // When computing liveness of virtual resources, we must also take into
    // account the fact that tasks without data dependencies between them could
    // be run in parallel. Otherwise, we may alias the same resource to two
    // potentially parallel tasks, creating a false dependency that forces
    // serial execution and may reduce performance. Since we keep track of data
    // dependencies between tasks during construction of the task list, we can
    // determine which tasks can be run in parallel.

    // --- 1. Compute the transitive closure of the task graph, which tells us
    // whether there's a path between two tasks in the graph.
    // This is used later during liveness analysis for querying whether two
    // tasks can be run in parallel (if there exists no path between two tasks,
    // then they can be run in parallel).
    std::vector<variable_set> reachability;
    reachability.resize(n_task);
    for (size_t i = 0; i < n_task; ++i) {
        // tasks already in topological order
        reachability[i].resize(n_task);
        for (auto pred : tasks[i]->preds) {
            reachability[i].set(pred);
            reachability[i] |= reachability[pred];
        }
    }

    // --- 2. allocate resources for temporaries, keeping track of live and dead
    // temporaries on the fly
    std::vector<variable_set> live_sets;  // live-sets for each task
    std::vector<variable_set> use;  // use-sets for each task
    std::vector<variable_set> def;  // def-sets for each task
    use.resize(n_task);
    def.resize(n_task);
    live_sets.resize(n_task);
    for (size_t i = 0; i < n_task; ++i) {
        use[i].resize(n_tmp);
        def[i].resize(n_tmp);
        live_sets[i].resize(n_tmp);
        for (auto r : tasks[i]->reads) {
            use[i].set(r);
        }
        for (auto w : tasks[i]->writes) {
            def[i].set(w);
        }
    }

    variable_set live;  // in the loop, set of live temporaries
    variable_set dead;  // dead temporaries (kept across iterations)
    variable_set gen;  // in the loop, temporaries that just came alive
    variable_set kill;  // in the loop, temporaries that just dropped dead
    variable_set mask;  // auxiliary set
    variable_set tmp;  //
    live.resize(n_tmp);
    dead.resize(n_tmp);
    gen.resize(n_tmp);
    kill.resize(n_tmp);
    mask.resize(n_tmp);
    tmp.resize(n_tmp);

    // std::vector<detail::task_index> kill_task; // vector indicating in which
    // task
    // the resource was determined dead
    // kill_task.resize(num_temporaries, 0);

    std::vector<allocation_requirements> allocations;   // OUTPUT
    std::vector<size_t> alloc_map;                      // OUTPUT
    alloc_map.resize(n_tmp, (size_t)-1);

    for (size_t i = 0; i < n_task; ++i) {
        // determine the resources live before this task
        live.reset();
        for (auto p : tasks[i]->preds) {
            live |= live_sets[p];
        }

        // determine the resources that come alive in this task
        gen.reset();
        gen |= use[i];
        gen |= def[i];
        gen -= live;

        // update the live set
        live |= use[i];
        live |= def[i];

        // determine the kill set
        kill = live;
        mask.reset();

        // do not kill vars used on parallel branches
        for (size_t j = 0; j < n_task; ++j) {
            if (!(i != j && !reachability[j][i] && !reachability[i][j])) continue;
            kill -= use[j];
            kill -= def[j];
        }

        // now look for uses and defs on the successor branches
        // if there's a def before any use, or no use at all, then consider the
        // resource dead (its contents are not going to be used anymore on this
        // branch).
        for (size_t succ = i + 1; succ < n_task; ++succ) {
            if (!reachability[succ][i]) continue;

            // def use mask kill mask
            // 0   0   0    kill 0
            // 1   0   0    1    1
            // 0   1   0    0    1
            // 1   1   0    0    1

            // tmp = bits to clear
            tmp = use[succ];
            tmp.flip();
            tmp |= mask;
            kill &= tmp;
            mask |= def[succ];
        }

        // assign a concrete resource to each virtual resource of this task
        for (size_t t0 = 0; t0 < n_tmp; ++t0) {
            if (gen[t0]) {
#ifdef GRAAL_TRACE_BATCH_SUBMIT
                fmt::print("{:02d}: Live({})\n", i, temporaries[t0]->name());
#endif
                auto& tmp0 = temporaries[t0];

                // skip if it doesn't need allocation
                if (!tmp0->is_virtual()) continue;
                if (tmp0->allocated) continue;

                auto alloc_requirements =
                    static_cast<virtual_resource&>(*tmp0).get_allocation_requirements(device);
                // resource became alive, if possible, alias with a dead
                // temporary, otherwise allocate a new one.

                if (!(alloc_requirements.flags & allocation_flag::aliasable)) {
                    // resource not aliasable, because it was explicitly
                    // requested to be not aliasable, or because there's still
                    // an external handle to it.
                    allocations.push_back(alloc_requirements);
                    alloc_map.push_back(allocations.size() - 1);
                }
                else {
                    // whether we managed to find a dead resource to alias with
                    bool aliased = false;
                    for (size_t t1 = 0; t1 < n_tmp; ++t1) {
                        // filter out live resources
                        if (!dead[t1]) continue;

#ifdef GRAAL_TRACE_BATCH_SUBMIT
                        fmt::print(" | {}", temporaries[t1]->name());
#endif
                        auto i_alloc = alloc_map[t1];
                        auto& dead_alloc_requirements = allocations[i_alloc];

                        if (adjust_allocation_requirements(
                            dead_alloc_requirements, alloc_requirements)) {
                            // the two resources may alias; the requirements
                            // have been adjusted
                            alloc_map[t0] = i_alloc;
                            // not dead anymore
                            dead[t1] = false;
                            aliased = true;
                            break;
                        }

                        // otherwise continue
                    }

#ifdef GRAAL_TRACE_BATCH_SUBMIT
                    fmt::print("\n");
#endif
                    // no aliasing opportunities
                    if (!aliased) {
                        // new allocation
                        allocations.push_back(alloc_requirements);
                        alloc_map.push_back(allocations.size() - 1);
                    }
                }
            }
        }

        // add resources to the dead set.
        // do it after assigning resource because we don't want to assign a
        // just-killed resource to a just-live resource (can't both read and
        // write to the same GPU texture, except in some circumstances which are
        // better dealt with explicitly anyway).
        dead |= kill;

        // for (size_t t = 0; t < num_temporaries; ++t) {
        //  if (kill[t]) {
        // kill_task[t] = i;
        // }
        // }

#ifdef GRAAL_TRACE_BATCH_SUBMIT
        for (size_t t = 0; t < n_tmp; ++t) {
            if (kill[t]) {
                fmt::print("{:02d}: Kill({})\n", i, temporaries[t]->name());
                // kill_task[t] = i;
            }
        }
#endif

        // update live
        live -= kill;
        live_sets[i] = live;
    }

    const auto stop = chrono::high_resolution_clock::now();
    const auto us = chrono::duration_cast<chrono::microseconds>(stop - start);

#ifdef GRAAL_TRACE_BATCH_SUBMIT
    fmt::print("pre-submission took {}us\n", us.count());
    /*for (size_t t = 0; t < num_tasks; ++t) {
          fmt::print("live set for task #{}:", t);
          for (size_t i = 0; i < num_temporaries; ++i) {
            if (live_sets[t].test(i)) {
              fmt::print("{},", temporaries_[i]->name());
            }
          }
          fmt::print("\n");
        }*/

        // dump task graph with live-variable analysis to a graphviz file
    {
        std::ofstream out_graphviz{ fmt::format("graal_test_{}.dot", start_sn), std::ios::trunc };
        dump_tasks(out_graphviz, tasks, temporaries, live_sets);
    }
#endif
    return{};
}

}  // namespace

/// @brief Per-batch+thread resources
struct batch_thread_resources {
    command_buffer_pool cb_pool;
};

/// @brief Represents a batch
struct batch {
    serial_number start_serial = 0;
    /// @brief all resources referenced in the batch (called "temporaries" for historical reasons)
    std::vector<resource_ptr> temporaries;
    /// @brief tasks
    std::vector<std::unique_ptr<task>> tasks;
    /// @brief Per-thread resources
    std::vector<batch_thread_resources> threads;
    /// @brief Semaphores used within the batch
    std::vector<vk::Semaphore> semaphores;

    /// @brief Returns the last sequence number of the batch.
    serial_number finish_serial() const noexcept {
        return start_serial + tasks.size();
    }

    /// @brief Sequence number of the next task.
    serial_number next_serial() const noexcept {
        return finish_serial() + 1;
    }
};

//-----------------------------------------------------------------------------
class queue_impl {
public:
    queue_impl(device_impl_ptr device, const queue_properties& props);
    ~queue_impl();

    void enqueue_pending_tasks();
    void present(swapchain_image&& image);

    void wait_for_task(uint64_t sequence_number);

    const task* get_submitted_task(uint64_t sequence_number) const noexcept;

    const task* get_unsubmitted_task(uint64_t sequence_number) const noexcept;

    task* get_unsubmitted_task(uint64_t sequence_number) noexcept {
        return const_cast<task*>(static_cast<const queue_impl&>(*this).get_unsubmitted_task(sequence_number));
    }


    [[nodiscard]] handler begin_build_task(std::string_view name) {
        if (building_task_) { throw std::logic_error{"a task is already being built"}; }
        building_task_ = true;
        const auto task_sn = last_serial_ + 1;
        auto t = std::make_unique<detail::submit_task>();
        t->name = name;
        t->serial = task_sn;
        return handler{std::move(t)};
    }

    void end_build_task(handler&& h) {
        for (auto& a : h.accesses_) {
            add_resource_dependency(*h.task_, a.resource, a.mode);
        }
        batch_.tasks.push_back(std::move(h.task_));
        building_task_ = false;
        last_serial_++;
    }

private:
    size_t get_resource_tmp_index(resource_ptr resource);
    void finalize_batch();
    [[nodiscard]] command_buffer_pool get_command_buffer_pool();
    void add_task_dependency(task& t, serial_number before, bool use_binary_semaphore);
    void add_resource_dependency(task& t, resource_ptr resource, access_mode mode);

    /// @brief Returns whether writes to the specified resource in this batch could be seen
    /// by uses of the resource in subsequent batches.
    /// @return
    [[nodiscard]] bool writes_are_user_observable(resource& r) const noexcept;

    bool building_task_ = false;
    queue_properties props_;
    device_impl_ptr device_;
    batch batch_;
    serial_number last_serial_ = 0;
    serial_number completed_serial_ = 0;
    std::deque<batch> in_flight_batches_;
    vk::Semaphore timelines_[max_queues];
    recycler<command_buffer_pool> cb_pools;
};

queue_impl::queue_impl(device_impl_ptr device, const queue_properties& props) :
    device_{device}, props_{props} {
    auto vk_device = device_->get_vk_device();

    vk::SemaphoreTypeCreateInfo timeline_create_info{
            .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
    vk::SemaphoreCreateInfo semaphore_create_info{.pNext = &timeline_create_info};
    for (size_t i = 0; i < max_queues; ++i) {
        timelines_[i] = vk_device.createSemaphore(semaphore_create_info);
    }
}

queue_impl::~queue_impl() {
    auto vk_device = device_->get_vk_device();
    for (size_t i = 0; i < max_queues; ++i) {
        vk_device.destroySemaphore(timelines_[i]);
    }
}

size_t queue_impl::get_resource_tmp_index(resource_ptr resource) {
    if (resource->tmp_index != invalid_temporary_index) {
        // already added
        return resource->tmp_index;
    }
    const auto tmp_index = batch_.temporaries.size();
    resource->tmp_index = tmp_index;
    batch_.temporaries.push_back(resource);
    return tmp_index;
}

command_buffer_pool queue_impl::get_command_buffer_pool() {
    const auto vk_device = device_->get_vk_device();
    command_buffer_pool cbp;
    if (!cb_pools.fetch(cbp)) {
        // TODO other queues?
        vk::CommandPoolCreateInfo create_info{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                .queueFamilyIndex = device_->get_graphics_queue_family()};
        const auto pool = vk_device.createCommandPool(create_info);
        cbp = command_buffer_pool{
                .command_pool = pool,
        };
    }
    return cbp;
}

bool queue_impl::writes_are_user_observable(resource& r) const noexcept {
    // writes to the resources might be seen if there are still user handles to the resource
    return r.last_write_serial > batch_.start_serial && !r.has_user_refs();
}

/*void queue_impl::finalize_batch() {
    // fill successors array for each task
    for (size_t i = 0; i < batch_.tasks.size(); ++i) {
        auto& t = batch_.tasks[i];
        for (auto pred : t->preds) {
            auto& pred_succ = batch_.tasks[pred]->succs;
            if (std::find(pred_succ.begin(), pred_succ.end(), i) == pred_succ.end()) {
                pred_succ.push_back(i);
            }
        }
    }
}*/

const task* queue_impl::get_submitted_task(uint64_t sequence_number) const noexcept {
    uint64_t start;
    uint64_t finish;

    for (auto&& b : in_flight_batches_) {
        start = b.start_serial;
        finish = b.finish_serial();
        if (sequence_number <= start) { return nullptr; }
        if (sequence_number > start && sequence_number <= finish) {
            const auto i = sequence_number - start - 1;
            return b.tasks[i].get();
        }
    }
    
    return nullptr;
}

const task* queue_impl::get_unsubmitted_task(uint64_t sequence_number) const noexcept 
{
    if (sequence_number <= batch_.start_serial) { return nullptr; }
    if (sequence_number <= batch_.finish_serial()) {
        const auto i = sequence_number - batch_.start_serial - 1;
        return batch_.tasks[i].get();
    }
    return nullptr;
}

void queue_impl::enqueue_pending_tasks() {
    const auto vk_device = device_->get_vk_device();

    // --- short-circuit if no tasks
    if (batch_.tasks.empty()) { return; }

    const auto n_tmp = batch_.temporaries.size();
    const auto n_task = batch_.tasks.size();

    // --- print debug information
#ifdef GRAAL_TRACE_BATCH_SUBMIT
    fmt::print("=== submitting batch (SN{} .. SN{}) ===\n", batch_.start_serial,
            batch_.finish_serial());

    fmt::print("Temporaries:\n");
    for (auto tmp : batch_.temporaries) {
        if (tmp->name().empty()) {
            fmt::print(" - temporary <unnamed> (discarded={})\n", tmp->has_user_refs());
        } else {
            fmt::print(" - temporary `{}` (discarded={})\n", tmp->name(), tmp->has_user_refs());
        }
    }

    fmt::print("Tasks:\n");
    int task_index = 0;
    for (auto&& t : batch_.tasks) {
        if (t->name.empty()) {
            fmt::print(" - task #{} (unnamed)\n", task_index);
        } else {
            fmt::print(" - task #{} `{}`\n", task_index, t->name);
        }
        if (!t->preds.empty()) {
            fmt::print("   preds: ");
            for (auto pred : t->preds) {
                fmt::print("{},", pred);
            }
            fmt::print("\n");
        }
        if (!t->succs.empty()) {
            fmt::print("   succs: ");
            for (auto s : t->succs) {
                fmt::print("{},", s);
            }
            fmt::print("\n");
        }
        task_index++;
    }
#endif

    // 
    auto batch_resource_memory = allocate_batch_memory(device_, batch_.start_serial, batch_.tasks, batch_.temporaries);
   

    // --- 3. set up external synchronization for resources that are could be accessed in the next batch
    // This can affect the creation of the submission batches (next step).
    for (size_t i = 0; i < n_tmp; ++i) {
        auto& tmp = *batch_.temporaries[i];

        if (writes_are_user_observable(tmp))
        {
            fmt::print("Writes to {} are observable by the user.\n", get_object_name(i, tmp));
            // writes to this resource in the batch can be observed by user code afterwards
            auto producer = get_unsubmitted_task(tmp.last_write_serial);
            fmt::print("\t producer task {} (SN{}).\n",
                    get_task_name(tmp.last_write_serial, *producer),
                    tmp.last_write_serial);

            if (tmp.type() == resource_type::swapchain_image) {
                // Special case: user observable swapchain images are synchronized with binary semaphores because presentation does not support timelines
                auto semaphore = device_->create_binary_semaphore();
                producer->signal_binary.push_back(semaphore);
                auto prev_semaphore = std::exchange(tmp.wait_semaphore, semaphore);
                // FIXME not sure what we know about the previous semaphore here (there shouldn't be one)
                if (prev_semaphore) {
                    device_->recycle_binary_semaphore(prev_semaphore);
                }
            } else {
                // use timeline
                producer->signal.enabled = true;
            }
        }
    }

    // --- 4. set up synchronization for cross-queue dependencies (TODO)
    for (size_t i = 0; i < n_task; ++i) {
        auto& task = batch_.tasks[i];

        for (auto pred : batch_.tasks[i]->preds) {
            if (needs_synchronization(*batch_.tasks[i], *batch_.tasks[pred])) {
                //pre_sync = true;
                //break;
            }
        }

        /*for (auto succ : tasks_[i]->succs) {
            if (needs_synchronization(*tasks_[i], *tasks_[succ])) {
                post_sync = true;
                break;
            }
        }*/
    }

    // --- 3. partition the graph into submissions
    struct submission {
        std::vector<size_t> tasks;
        serial_number wait_sn;  // 0 means no wait
        serial_number signal_sn;  // 0 means no signal
    };

    // build batches
    std::vector<submission> submissions[max_queues];
    std::vector<size_t> traverse_stack;

    for (size_t i = 0; i < n_task; ++i) {
        bool pre_sync = false;
        bool post_sync = false;

        for (auto pred : batch_.tasks[i]->preds) {
            if (needs_synchronization(*batch_.tasks[i], *batch_.tasks[pred])) {
                pre_sync = true;
                break;
            }
        }
        for (auto succ : batch_.tasks[i]->succs) {
            if (needs_synchronization(*batch_.tasks[i], *batch_.tasks[succ])) {
                post_sync = true;
                break;
            }
        }
    }

    // two tasks are in different submissions if:
    // - they are not scheduled into the same queue
    // - they cannot be merged into a single vkQueueSubmit
    // - one of the tasks is a submission and the other is a present operation

    // --- 3. submit tasks
    // --- 3.1 get command pool/allocate command buffers

    // TODO parallel submission
    for (size_t i = 0; i < n_task; ++i) {
        /*for (auto&& cmd : tasks_[i]->callbacks) {
            cmd(cb);
        }*/
        auto task = batch_.tasks[i].get();
        if (task->type == task_type::present) {
            auto present = static_cast<present_task*>(task);
            vk::PresentInfoKHR present_info{
                    .waitSemaphoreCount = 1,
                    .pWaitSemaphores = nullptr,
                    .swapchainCount = {},
                    .pSwapchains = {},
                    .pImageIndices = {},
                    .pResults = {},
            };

            const auto graphics_queue = device_->get_graphics_queue();
            graphics_queue.presentKHR(present_info);
        }
    }

    // pacing: wait for frame N - (max_batches_in_flight+1) before starting
    // submission pacing is important because we have to reclaim resources on
    // the CPU side at some point

    // With NFLIGHT=1, BATCH=1 (first batch)
    // do not wait
    // BATCH=2, do not wait
    // BATCH=3, wait on 1
    //

    /*// if there are too many in-flight batches, wait
    if (in_flight_batches_.size() >= props_.max_batches_in_flight) {
        wait_for_batch(current_batch_ - (props_.max_batches_in_flight + 1));
    }

    // Wait for batch `N-max_batches_in_flight` to finish
    const uint64_t wait_values[] = {
            should_wait ? (current_batch_ - props_.max_batches_in_flight) : 0};
    // Signal current batch
    const uint64_t signal_values[] = {current_batch_};
    const vk::Semaphore semaphores[] = {batch_index_semaphore_};
    const vk::CommandBuffer cbs[] = {cb};

    const vk::TimelineSemaphoreSubmitInfo timeline_submit_info{.waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = wait_values,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = signal_values};

    const vk::PipelineStageFlags wait_stages[] = {vk::PipelineStageFlagBits::eAllCommands};

    const vk::SubmitInfo submit_info = {.pNext = &timeline_submit_info,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = semaphores,
            .pWaitDstStageMask = wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = cbs,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = semaphores};

    const auto queue = device_->get_graphics_queue();
    queue.submit(submit_info, nullptr);
    //queue.

    //temporaries_.clear();
    //tasks_.clear();
    current_batch_++;*/

    // stash the current batch
    in_flight_batches_.push_back(std::move(batch_));
    // start new batch
    batch_.start_serial = last_serial_;
    batch_.semaphores.clear();
    batch_.temporaries.clear();
    batch_.threads.clear();
    batch_.tasks.clear();
}

void queue_impl::present(swapchain_image&& image) 
{
    auto image_impl = image.impl_.get();

    // --- build task
    auto t = std::make_unique<present_task>();
    t->name = "present";
    t->type = task_type::present;
    t->serial = last_serial_++;
    //t->swapchain = image.swapchain();
    //t->image_index = ...

    add_resource_dependency(*t, image.impl_, access_mode::read_only);
    
    batch_.tasks.push_back(std::move(t));
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
void queue_impl::add_task_dependency(task& t, serial_number before,  bool use_binary_semaphore)
{
    assert(t.serial > before);

    if (before <= completed_serial_) {
        // no sync needed, task has already completed
        return;     
    }

    if (before <= batch_.start_serial) {
        // --- Inter-batch synchronization w/ timeline
        auto before_task = get_submitted_task(before);
        size_t q = before_task->signal.queue;
        t.waits[q] = std::max(t.waits[q], before);
    }
    else {
        // --- Intra-batch synchronization
        const auto local_task_index = before - batch_.start_serial;  // SN relative to start of batch
        auto before_task = batch_.tasks[local_task_index].get();
        t.preds.push_back(local_task_index);
        // successors are updated 
        before_task->succs.push_back(t.serial);
    }
}

/// @brief Registers a resource access in a submit task
/// @param resource
/// @param mode
void queue_impl::add_resource_dependency(task& t, resource_ptr resource, access_mode mode)
{
    // does the resource carry a semaphore? if so, extract it and sync on that
    if (auto semaphore = std::exchange(resource->wait_semaphore, nullptr)) {
        t.wait_binary.push_back(semaphore);
    }
    else {
        // the resource does not carry a semaphore, sync on timeline
        add_task_dependency(t, resource->last_write_serial, false);
    }

    if (mode == access_mode::read_write || mode == access_mode::write_only) {
        // we're writing to this resource: set last write SN
        resource->last_write_serial = t.serial;
    }

    const auto tmp_index = get_resource_tmp_index(resource);

    switch (mode) {
    case access_mode::read_only: t.add_read(tmp_index); break;
    case access_mode::write_only: t.add_write(tmp_index); break;
    case access_mode::read_write:
        t.add_read(tmp_index);
        t.add_write(tmp_index);
        break;
    }
}

}  // namespace detail


//-----------------------------------------------------------------------------

void handler::add_buffer_access(
        std::shared_ptr<detail::buffer_impl> buffer, access_mode mode, buffer_usage usage) {
    add_resource_access(buffer, mode);
}

void handler::add_image_access(
        std::shared_ptr<detail::image_impl> image, access_mode mode, image_usage usage) {
    add_resource_access(image, mode);
    // TODO do something with usage? (layout transitions)
}

void handler::add_resource_access(
        std::shared_ptr<detail::virtual_resource> resource, access_mode mode) {
    accesses_.push_back(resource_access{.resource = std::move(resource), .mode = mode});
}

//-----------------------------------------------------------------------------
queue::queue(device& device, const queue_properties& props) :
    impl_{std::make_unique<detail::queue_impl>(device.impl_, props)} {
}

void queue::enqueue_pending_tasks() {
    impl_->enqueue_pending_tasks();
}

void queue::present(swapchain_image&& image) {
    impl_->present(std::move(image));
}

handler queue::begin_build_task(std::string_view name) const noexcept {
    // auto sequence_number = impl_->begin_build_task();
    // NOTE begin_build_task is not really necessary right now since the task object
    // is not created before the handler callback is executed,
    // but keep it anyway for future-proofing (we might want to know e.g. the task sequence number in the handler)
    return impl_->begin_build_task(name);
}

/// @brief Called to finish building a task. Adds the task to the current batch.
void queue::end_build_task(handler&& handler) {
    impl_->end_build_task(std::move(handler));
}

/*
size_t queue::next_task_index() const {
    return impl_->next_task_index();
}
size_t queue::current_batch_index() const {
    return impl_->current_batch_index();
}*/

}  // namespace graal