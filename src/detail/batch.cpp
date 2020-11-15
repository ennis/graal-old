#include <graal/detail/batch.hpp>
#include <graal/detail/swapchain_impl.hpp>

#include <boost/dynamic_bitset.hpp>
#include <fstream>
#include <chrono>

namespace graal::detail {
namespace {

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

template<typename T>
void sorted_vector_insert(std::vector<T>& vec, T elem) {
    auto it = std::lower_bound(vec.begin(), vec.end(), elem);
    vec.insert(it, std::move(elem));
}

using variable_set = boost::dynamic_bitset<uint64_t>;
void dump_tasks(std::ostream& out, const std::vector<std::unique_ptr<task>>& tasks,
        const std::vector<resource_ptr>& temporaries, const std::vector<variable_set>& live_sets) {
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


}  // namespace

void task::add_read(temporary_index r) {
    sorted_vector_insert(reads, std::move(r));
}

void task::add_write(temporary_index w) {
    sorted_vector_insert(writes, std::move(w));
}

//-------------------------------------------------------------------------
batch::batch(device_impl_ptr device, sequence_number start_sn) : device_{ std::move(device) }, start_sn_{ start_sn }
{}


command_buffer_pool batch::get_command_buffer_pool(recycled_batch_resources& recycled_resources)
{
    const auto vk_device = device_->get_vk_device();
    command_buffer_pool cbp;
    if (!recycled_resources.cb_pools.fetch(cbp)) {
        // TODO other queues?
        vk::CommandPoolCreateInfo create_info{ .flags = vk::CommandPoolCreateFlagBits::eTransient,
                .queueFamilyIndex = device_->get_graphics_queue_family() };
        const auto pool = vk_device.createCommandPool(create_info);
        cbp = command_buffer_pool{
                .command_pool = pool,
        };
    }
    return cbp;
}

bool batch::writes_are_user_observable(resource& r) const noexcept {
    // writes to the resources might be seen if there are still user handles to the resource
    return r.last_write_sequence_number > start_sn_ && !r.discarded;
}

void batch::finalize_task_dag() {
    // fill successors array for each task
    for (size_t i = 0; i < tasks_.size(); ++i) {
        auto& t = tasks_[i];
        for (auto pred : t->preds) {
            auto& pred_succ = tasks_[pred]->succs;
            if (std::find(pred_succ.begin(), pred_succ.end(), i) == pred_succ.end()) {
                pred_succ.push_back(i);
            }
        }
    }
}

void batch::add_task(std::unique_ptr<task> task) noexcept {
    tasks_.push_back(std::move(task));
}

size_t batch::add_temporary(resource_ptr resource) {
    if (resource->tmp_index != invalid_temporary_index) {
        // already added
        return resource->tmp_index;
    }
    const auto tmp_index = temporaries_.size();
    resource->tmp_index = tmp_index;
    temporaries_.push_back(resource);
    return tmp_index;
}

void batch::submit(vk::Semaphore timeline_semaphore, recycled_batch_resources& recycled_resources)
{
    const auto vk_device = device_->get_vk_device();

    // --- short-circuit if no tasks
    if (tasks_.empty()) { return; }

    // --- print debug information
#ifdef GRAAL_TRACE_BATCH_SUBMIT
    fmt::print("=== submitting batch (start_sn={}) ===\n", start_sn_);

    fmt::print("Temporaries:\n");
    for (auto tmp : temporaries_) {
        if (tmp->name().empty()) {
            fmt::print(" - temporary <unnamed> (discarded={})\n", tmp->discarded);
        } else {
            fmt::print(" - temporary `{}` (discarded={})\n", tmp->name(), tmp->discarded);
        }
    }

    fmt::print("Tasks:\n");
    int task_index = 0;
    for (auto&& t : tasks_) {
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

    // --- prepare the resources for this batch

    // --- start perf timer for submission
    namespace chrono = std::chrono;
    const auto start = chrono::high_resolution_clock::now();

    const size_t num_tasks = tasks_.size();
    const size_t num_temporaries = temporaries_.size();

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
    reachability.resize(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
        // tasks already in topological order
        reachability[i].resize(num_tasks);
        for (auto pred : tasks_[i]->preds) {
            reachability[i].set(pred);
            reachability[i] |= reachability[pred];
        }
    }

    // --- 2. allocate resources for temporaries, keeping track of live and dead
    // temporaries on the fly
    std::vector<variable_set> live_sets;  // live-sets for each task
    std::vector<variable_set> use;  // use-sets for each task
    std::vector<variable_set> def;  // def-sets for each task
    use.resize(num_tasks);
    def.resize(num_tasks);
    live_sets.resize(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
        use[i].resize(num_temporaries);
        def[i].resize(num_temporaries);
        live_sets[i].resize(num_temporaries);
        for (auto r : tasks_[i]->reads) {
            use[i].set(r);
        }
        for (auto w : tasks_[i]->writes) {
            def[i].set(w);
        }
    }

    variable_set live;  // in the loop, set of live temporaries
    variable_set dead;  // dead temporaries (kept across iterations)
    variable_set gen;  // in the loop, temporaries that just came alive
    variable_set kill;  // in the loop, temporaries that just dropped dead
    variable_set mask;  // auxiliary set
    variable_set tmp;  //
    live.resize(num_temporaries);
    dead.resize(num_temporaries);
    gen.resize(num_temporaries);
    kill.resize(num_temporaries);
    mask.resize(num_temporaries);
    tmp.resize(num_temporaries);

    // std::vector<detail::task_index> kill_task; // vector indicating in which
    // task
    // the resource was determined dead
    // kill_task.resize(num_temporaries, 0);

    std::vector<allocation_requirements> allocations;
    std::vector<size_t> alloc_map;
    alloc_map.resize(num_temporaries, (size_t) -1);

    for (size_t i = 0; i < num_tasks; ++i) {
        // determine the resources live before this task
        live.reset();
        for (auto p : tasks_[i]->preds) {
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
        for (size_t j = 0; j < num_tasks; ++j) {
            if (!(i != j && !reachability[j][i] && !reachability[i][j])) continue;
            kill -= use[j];
            kill -= def[j];
        }

        // now look for uses and defs on the successor branches
        // if there's a def before any use, or no use at all, then consider the
        // resource dead (its contents are not going to be used anymore on this
        // branch).
        for (size_t succ = i + 1; succ < num_tasks; ++succ) {
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
        for (size_t t0 = 0; t0 < num_temporaries; ++t0) {
            if (gen[t0]) {
#ifdef GRAAL_TRACE_BATCH_SUBMIT
                fmt::print("{:02d}: Live({})", i, temporaries_[t0]->name());
#endif
                auto& tmp0 = temporaries_[t0];

                // skip if it doesn't need allocation
                if (!tmp0->is_virtual()) continue;
                if (tmp0->allocated) continue;

                auto alloc_requirements = static_cast<virtual_resource&>(*tmp0).get_allocation_requirements(device_);
                // resource became alive, if possible, alias with a dead
                // temporary, otherwise allocate a new one.

                if (!(alloc_requirements.flags & allocation_flag::aliasable)) {
                    // resource not aliasable, because it was explicitly
                    // requested to be not aliasable, or because there's still
                    // an external handle to it.
                    allocations.push_back(alloc_requirements);
                    alloc_map.push_back(allocations.size() - 1);
                } else {
                    // whether we managed to find a dead resource to alias with
                    bool aliased = false;
                    for (size_t t1 = 0; t1 < num_temporaries; ++t1) {
                        // filter out live resources
                        if (!dead[t1]) continue;

#ifdef GRAAL_TRACE_BATCH_SUBMIT
                        fmt::print(" | {}", temporaries_[t1]->name());
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
        for (size_t t = 0; t < num_temporaries; ++t) {
            if (kill[t]) {
                fmt::print("{:02d}: Kill({})\n", i, temporaries_[t]->name());
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
        std::ofstream out_graphviz{
                fmt::format("graal_test_{}.dot", start_sn_), std::ios::trunc};
        dump_tasks(out_graphviz, tasks_, temporaries_, live_sets);
    }
#endif


    // --- 3. set up external synchronization for resources that are could be accessed in the next batch
    // This can affect the creation of the submission batches (next step).
    for (size_t i = 0; i < num_temporaries; ++i) {
        auto &tmp = *temporaries_[i];

        if (writes_are_user_observable(tmp)) 
        {
            // writes to this resource in the batch can be observed by user code afterwards
            auto producer = get_task(tmp.last_write_sequence_number);

            if (tmp.type() == resource_type::swapchain_image) {
                // needs a binary semaphore
                vk::Semaphore semaphore;
                if (!recycled_resources.semaphores.fetch(semaphore)) {
                    vk::SemaphoreCreateInfo sci{};
                    semaphore = vk_device.createSemaphore(sci);
                }
                // set wait on producer
                producer->signal_binary.push_back(semaphore);
                // set semaphore on resource (owner)
                auto prev_semaphore = static_cast<swapchain_image_impl&>(tmp).consume_semaphore(semaphore);
                // FIXME not sure what we know about the previous semaphore here
                if (prev_semaphore) {
                    recycled_resources.semaphores.recycle(std::move(prev_semaphore));
                }
            }
            else {
                // use timeline
                producer->signal.enabled = true;
            }
        }
    }

    // --- 4. set up synchronization for cross-queue dependencies and rendering/presentation 
    for (size_t i = 0; i < num_tasks; ++i) {
        auto& task = tasks_[i];

        for (auto pred : tasks_[i]->preds) {
            
            if (tasks_[i]->)
            
            if (needs_synchronization(*tasks_[i], *tasks_[pred])) 
            {
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
        sequence_number wait_sn;   // 0 means no wait
        sequence_number signal_sn; // 0 means no signal
    };

    // build batches
    std::vector<submission> submissions[max_queues];
    std::vector<size_t> traverse_stack; 

    for (size_t i = 0; i < num_tasks; ++i) 
    {
        bool pre_sync = false;
        bool post_sync = false;

        for (auto pred : tasks_[i]->preds) {
            if (needs_synchronization(*tasks_[i], *tasks_[pred])) {
                pre_sync = true;
                break;
            }
        }
        for (auto succ : tasks_[i]->succs) {
            if (needs_synchronization(*tasks_[i], *tasks_[succ])) {
                post_sync = true;
                break;
            }
        }

        
    }


    // --- 4. set outgoing sequence numbers for all submissions and associated resources


    // two tasks are in different submissions if:
    // - they are not scheduled into the same queue
    // - they cannot be merged into a single vkQueueSubmit
    // - one of the tasks is a submission and the other is a present operation

    // --- 3. submit tasks
    // --- 3.1 get command pool/allocate command buffers

    // TODO parallel submission
    for (size_t i = 0; i < num_tasks; ++i) {
        /*for (auto&& cmd : tasks_[i]->callbacks) {
            cmd(cb);
        }*/
    }

    // reset temporary indices that were assigned during queuing.
    for (auto&& tmp : temporaries_) {
        tmp->tmp_index = invalid_temporary_index;
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
}
}  // namespace graal::detail