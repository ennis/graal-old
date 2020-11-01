#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <graal/queue.hpp>
#include <numeric>
#include <span>
#include <unordered_set>

namespace graal {
namespace detail {
namespace {

bool adjust_allocation_requirements(allocation_requirements &      req_a,
                                    const allocation_requirements &req_b) {
  // TODO dedicated allocs?
  if (req_a.flags != req_b.flags)
    return false;
  // don't alias if the memory type bits are not strictly the same
  if (req_a.memory_type_bits != req_b.memory_type_bits)
    return false;

  req_a.alignment = std::max(req_a.alignment, req_b.alignment);
  req_a.size = std::max(req_a.size, req_b.size);
  return true;
}

std::string get_task_name(task_index index, const task &task) {
  if (task.name.empty()) {
    return fmt::format("#{}", index);
  }
  return task.name;
}

std::string get_object_name(std::size_t index, const named_object &obj) {
  if (obj.name().empty()) {
    return fmt::format("#{}", index);
  }
  return std::string{obj.name()};
}

/// Converts a vector of intrusive_ptr to a vector of raw pointers.
template <typename T>
std::vector<T *> to_raw_ptr_vector(const std::vector<std::shared_ptr<T>> &v) {
  std::vector<T *> result;
  std::transform(v.begin(), v.end(), std::back_inserter(result),
                 [](auto &&x) { return x.get(); });
  return result;
}

template <typename T> void sorted_vector_insert(std::vector<T> &vec, T elem) {
  auto it = std::lower_bound(vec.begin(), vec.end(), elem);
  vec.insert(it, std::move(elem));
}

/*class adjacency_matrix {
public:
  adjacency_matrix(std::size_t n) : n_{n} {
    mat_.resize((n_ * (n_ + 1)) / 2, false);
  }

  void add_edge(std::size_t a, std::size_t b) { mat_[idx(a, b)] = true; }

  bool operator()(std::size_t a, std::size_t b) const {
    return mat_[idx(a, b)];
  }

  void dump() {
    fmt::print("  |");
    for (std::size_t i = 0; i < n_; ++i) {
      fmt::print("{:02d} ", i);
    }
    fmt::print("\n");
    for (std::size_t i = 0; i < n_; ++i) {
      fmt::print("{:02d}|", i);
      for (std::size_t j = 0; j < n_; ++j) {
        if (this->operator()(i, j)) {
          fmt::print("1  ");
        } else {
          fmt::print("0  ");
        }
      }
      fmt::print("\n");
    }
  }

private:
  std::size_t idx(std::size_t a, std::size_t b) const {
    if (b > a) {
      std::swap(a, b);
    }
    return (a * (a + 1)) / 2 + b;
  }

  std::size_t       n_;
  std::vector<bool> mat_;
};*/

using variable_set = boost::dynamic_bitset<uint64_t>;

void dump_tasks(std::ostream &out, const std::vector<task> &tasks,
                const std::vector<resource_ptr> &temporaries,
                const std::vector<variable_set> &live_sets) {
  out << "digraph G {\n";
  out << "node [shape=record fontname=Consolas];\n";

  for (task_index index = 0; index < tasks.size(); ++index) {
    const auto &task = tasks[index];

    out << "t_" << index << " [shape=record label=\"{";
    out << index;
    if (!task.name.empty()) {
      out << "(" << task.name << ")";
    }
    out << "|{{reads\\l|writes\\l|live\\l}|{";
    {
      int i = 0;
      for (auto r : task.reads) {
        out << get_object_name(r, *temporaries[r]);
        if (i != task.reads.size() - 1) {
          out << ",";
        }
        ++i;
      }
    }
    out << "\\l|";
    {
      int i = 0;
      for (auto w : task.writes) {
        out << get_object_name(w, *temporaries[w]);
        if (i != task.writes.size() - 1) {
          out << ",";
        }
        ++i;
      }
    }
    out << "\\l|";

    {
      for (size_t t = 0; t < temporaries.size(); ++t) {
        if (live_sets[index].test(t)) {
          out << get_object_name(t, *temporaries[t]);
          if (t != live_sets[index].size() - 1) {
            out << ",";
          }
        }
      }
    }

    out << "\\l}}}\"]\n";
  }

  for (task_index i = 0; i < tasks.size(); ++i) {
    for (auto pred : tasks[i].preds) {
      out << "t_" << pred << " -> "
          << "t_" << i << "\n";
    }
  }

  out << "}\n";
}

} // namespace

void task::add_read(temporary_index r) {
  sorted_vector_insert(reads, std::move(r));
}

void task::add_write(temporary_index w) {
  sorted_vector_insert(writes, std::move(w));
}

void queue_impl::enqueue_pending_tasks() {
  // --- short-circuit if no tasks
  if (tasks_.empty()) {
    return;
  }

  // --- print debug information
#ifdef GRAAL_TRACE_BATCH_SUBMIT
  fmt::print("=== submitting batch #{} ===\n", current_batch_);

  fmt::print("Temporaries:\n");
  for (auto tmp : temporaries_) {
    if (tmp->name().empty()) {
      fmt::print(" - temporary <unnamed> (discarded={})\n", tmp->discarded);
    } else {
      fmt::print(" - temporary `{}` (discarded={})\n", tmp->name(),
                 tmp->discarded);
    }
  }

  fmt::print("Tasks:\n");
  int task_index = 0;
  for (auto &&t : tasks_) {
    if (t.name.empty()) {
      fmt::print(" - task #{} (unnamed)\n", task_index);
    } else {
      fmt::print(" - task #{} `{}`\n", task_index, t.name);
    }
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
    task_index++;
  }
#endif

  // --- start perf timer for submission
  namespace chrono = std::chrono;
  const auto start = chrono::high_resolution_clock::now();

  const size_t num_tasks = tasks_.size();
  const size_t num_temporaries = temporaries_.size();

  // Before submitting a task, we assign concrete resources
  // to all virtual resources used in the batch (=="temporaries"). We strive to
  // minimize total memory usage by "aliasing" a single concrete resource to
  // multiple temporaries if we can determine that those are never in use
  // ("alive") at the same time during execution. This can be done by keeping
  // track of the "liveness" of temporaries during submisson (see
  // https://en.wikipedia.org/wiki/Live_variable_analysis).
  //
  // When computing liveness of virtual resources, we must also take into
  // account the fact that tasks without data dependencies between them could be
  // run in parallel. Otherwise, we may alias the same resource to two
  // potentially parallel tasks, creating a false dependency that forces serial
  // execution and may reduce performance. Since we keep track of data
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
    for (auto pred : tasks_[i].preds) {
      reachability[i].set(pred);
      reachability[i] |= reachability[pred];
    }
  }

  // --- 2. allocate resources for temporaries, keeping track of live and dead
  // temporaries on the fly
  std::vector<variable_set> live_sets; // live-sets for each task
  std::vector<variable_set> use;       // use-sets for each task
  std::vector<variable_set> def;       // def-sets for each task
  use.resize(num_tasks);
  def.resize(num_tasks);
  live_sets.resize(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    use[i].resize(num_temporaries);
    def[i].resize(num_temporaries);
    live_sets[i].resize(num_temporaries);
    for (auto r : tasks_[i].reads) {
      use[i].set(r);
    }
    for (auto w : tasks_[i].writes) {
      def[i].set(w);
    }
  }

  variable_set live; // in the loop, set of live temporaries
  variable_set dead; // dead temporaries (kept across iterations)
  variable_set gen;  // in the loop, temporaries that just came alive
  variable_set kill; // in the loop, temporaries that just dropped dead
  variable_set mask; // auxiliary set
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
  std::vector<size_t>                  alloc_map;
  alloc_map.resize(num_temporaries, (size_t)-1);

  for (size_t i = 0; i < num_tasks; ++i) {
    // determine the resources live before this task
    live.reset();
    for (auto p : tasks_[i].preds) {
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
      if (!(i != j && !reachability[j][i] && !reachability[i][j]))
        continue;
      kill -= use[j];
      kill -= def[j];
    }

    // now look for uses and defs on the successor branches
    // if there's a def before any use, or no use at all, then consider the
    // resource dead (its contents are not going to be used anymore on this
    // branch).
    for (size_t succ = i + 1; succ < num_tasks; ++succ) {
      if (!reachability[succ][i])
        continue;

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
        auto &tmp0 = temporaries_[t0];

        // skip if it doesn't need allocation
        if (tmp0->allocated)
          continue;

        auto alloc_requirements = tmp0->get_allocation_requirements(device_);
        // resource became alive, if possible, alias with a dead temporary,
        // otherwise allocate a new one.

        if (!(alloc_requirements.flags & allocation_flag::aliasable)) {
          // resource not aliasable, because it was explicitly requested to be
          // not aliasable, or because there's still an external handle to it.
          allocations.push_back(alloc_requirements);
          alloc_map.push_back(allocations.size() - 1);
        } else {

          // whether we managed to find a dead resource to alias with
          bool aliased = false;
          for (size_t t1 = 0; t1 < num_temporaries; ++t1) {
            // filter out live resources
            if (!dead[t1])
              continue;

#ifdef GRAAL_TRACE_BATCH_SUBMIT
            fmt::print(" | {}", temporaries_[t1]->name());
#endif
            auto  i_alloc = alloc_map[t1];
            auto &dead_alloc_requirements = allocations[i_alloc];

            if (adjust_allocation_requirements(dead_alloc_requirements,
                                               alloc_requirements)) {
              // the two resources may alias; the requirements have been
              // adjusted
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
    // just-killed resource to a just-live resource (can't both read and write
    // to the same GPU texture, except in some circumstances which are better
    // dealt with explicitly anyway).
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
  fmt::print("submission took {}us\n", us.count());
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
    std::ofstream out_graphviz{fmt::format("graal_test_{}.dot", current_batch_),
                               std::ios::trunc};
    dump_tasks(out_graphviz, tasks_, temporaries_, live_sets);
  }
#endif

  // --- 3. submit tasks
  for (size_t i = 0; i < num_tasks; ++i) {
    for (auto &&cmd : tasks_[i].callbacks) {
      cmd();
    }
  }

  // reset temporary indices that were assigned during queuing.
  for (auto &&tmp : temporaries_) {
    tmp->tmp_index = invalid_temporary_index;
  }

  temporaries_.clear();
  tasks_.clear();
  current_batch_++;
}

temporary_index queue_impl::add_temporary(resource_ptr resource) {
  if (resource->tmp_index != invalid_temporary_index) {
    // already added
    return resource->tmp_index;
  }
  const auto tmp_index = temporaries_.size();
  resource->tmp_index = tmp_index;
  temporaries_.push_back(resource);
  return tmp_index;
}

} // namespace detail

void handler::add_buffer_access(std::shared_ptr<detail::buffer_impl> buffer,
    access_mode mode, buffer_usage usage) {
  add_resource_access(buffer, mode);
  // TODO do something with usage?
}

void handler::add_image_access(std::shared_ptr<detail::image_impl> image, access_mode mode,
                               image_usage usage) {
  add_resource_access(image, mode);
  // TODO do something with usage? (layout transitions)
}

void handler::add_resource_access(std::shared_ptr<detail::resource> resource,
                                  access_mode                       mode) {
  if (resource->batch == batch_index_ &&
      (mode == access_mode::read_only || mode == access_mode::read_write)) {

    // if the last write was in a different batch, then we depend on the
    // previous batch to be finished.
    // TODO add a dependency to the task that was aleady submitted?

    // we know that the producer is a task in this batch
    auto &pred_succ = queue_.tasks_[resource->producer].succs;
    if (std::find(pred_succ.begin(), pred_succ.end(), task_index_) ==
        pred_succ.end()) {
      // add current task in the list of successors
      pred_succ.push_back(task_index_);
    }
    // add the producer to the current list of predecessors
    task_.preds.push_back(resource->producer);
  }

  if (mode == access_mode::read_write || mode == access_mode::write_only) {
    // we're writing to this resource: set producer to this task and set the
    // last write batch
    resource->batch = batch_index_;
    resource->producer = task_index_;
  }

  const auto tmp_index = queue_.add_temporary(resource);

  switch (mode) {
  case access_mode::read_only:
    task_.add_read(tmp_index);
    break;
  case access_mode::write_only:
    task_.add_write(tmp_index);
    break;
  case access_mode::read_write:
    task_.add_read(tmp_index);
    task_.add_write(tmp_index);
    break;
  }
}

} // namespace graal