#include <algorithm>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <graal/queue.hpp>
#include <numeric>
#include <span>
#include <unordered_set>

#include <boost/dynamic_bitset.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/transitive_closure.hpp>

namespace graal {
namespace detail {
namespace {

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

class adjacency_matrix {
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
};

using live_set = boost::dynamic_bitset<uint64_t>;

/// @brief magic
/// @param tasks
/// @param task_index
/// @return magic
std::vector<live_set>
compute_liveness(const std::vector<virtual_resource_ptr> &temporaries,
                 const std::vector<detail::task> &        tasks)

{
  namespace chrono = std::chrono;
  auto start = chrono::high_resolution_clock::now();

  const size_t num_tasks = tasks.size();
  const size_t num_temporaries = temporaries.size();

  // 1. compute the transitive closure of the task graph, as this will help us
  // determine the tasks that can be run in parallel
  std::vector<live_set> reachability;
  reachability.resize(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    // tasks already in topological order
    reachability[i].resize(num_tasks);
    for (auto pred : tasks[i].preds) {
      reachability[i].set(pred);
      reachability[i] |= reachability[pred];
    }
  }

  // init use/def sets
  std::vector<live_set> live_sets;
  std::vector<live_set> in_sets;
  std::vector<live_set> out_sets;
  std::vector<live_set> use;
  std::vector<live_set> def;
  use.resize(num_tasks);
  def.resize(num_tasks);
  in_sets.resize(num_tasks);
  out_sets.resize(num_tasks);
  live_sets.resize(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    use[i].resize(num_temporaries);
    def[i].resize(num_temporaries);
    in_sets[i].resize(num_temporaries);
    out_sets[i].resize(num_temporaries);
    live_sets[i].resize(num_temporaries);
    for (auto r : tasks[i].reads) {
      use[i].set(r);
    }
    for (auto w : tasks[i].writes) {
      def[i].set(w);
    }
  }

  /*live_set prev_in;
  live_set prev_out;

  bool end;
  do {
    end = true;
    for (size_t i = 0; i < num_tasks; ++i) {
      auto &in = in_sets[i];
      prev_in = in;
      auto &out = out_sets[i];
      prev_out = out;
      auto &task = tasks[i];

      in = out;
      out.reset();
      for (auto w : task.writes) {
        in.reset(w);
      }
      for (auto r : task.reads) {
        in.set(r);
        // out.set(r);
      }

       for (auto s : task.succs) {
      //if (i < num_tasks - 1) {
        // assume sequential execution, single successor
        out |= in_sets[s];
      //}
      }

      fmt::print("in [{:2d}]: ", i);
      for (size_t j = 0; j < num_temporaries; ++j) {
        if (in[j]) {
          fmt::print("{:02d} ", j);
        } else {
          fmt::print("   ");
        }
      }
      fmt::print("\n");

      fmt::print("out[{:2d}]: ", i);
      for (size_t j = 0; j < num_temporaries; ++j) {
        if (out[j]) {
          fmt::print("{:02d} ", j);
        } else {
          fmt::print("   ");
        }
      }
      fmt::print("\n");

      end &= (prev_in == in) && (prev_out == out);
    }
    // fmt::print("\n");

  } while (!end);*/

  /*for (size_t i = 0; i < num_tasks; ++i) {
    out_sets[i] |= in_sets[i];
  }*/

  live_set live;
  live_set kill;
  live_set mask;
  live_set tmp;
  live.resize(num_temporaries);
  kill.resize(num_temporaries);
  mask.resize(num_temporaries);
  tmp.resize(num_temporaries);

  for (size_t i = 0; i < num_tasks; ++i) {
    live.reset();
    for (auto p : tasks[i].preds) {
      live |= live_sets[p];
    }

    live |= use[i];
    live |= def[i];

    // Some tasks may run in parallel because they have no data-dependencies
    // between them. However, the liveness analysis performed above assumes
    // serial execution of all tasks in the order of which they appear in the
    // list. Following this analysis, resource allocation may map two tasks on
    // the same resource, even if those two tasks could run in parallel. This
    // forces the two tasks to run serially, which negatively impacts
    // performance.
    //
    // In other words, we only want to map the same resource to two different
    // tasks if we can prove that they can't possibly run in parallel. To
    // account for that, we modify the live-sets of each task by adding the
    // union of the live sets of all tasks that could run in parallel.

    // determine the kill set

    // init kill, which basically says "don't kill the variables we access in
    // the task". kill = ~(use[i] | def[i])
    kill = use[i];
    kill |= def[i];
    kill.flip();
    mask.reset();

    /*for (size_t j = 0; j < num_tasks; ++j) {
      if (i != j && !reachability[i][j] && !reachability[j][i]) {
        kill -= use[j];
        kill -= def[j];
        live |= use[j];
        live |= def[j];
      }
    }*/

    for (size_t succ = i + 1; succ < num_tasks; ++succ) {
      if (!reachability[succ][i])
        continue;
      // def use kill mask
      // 0   0   1    0
      // 1   0   1    1
      // 0   1   0    0 (no need for mask)
      // 1   1   0    1

      // TODO explain this
      tmp = use[succ];
      tmp.flip();
      tmp |= mask;
      kill &= tmp;
      mask |= def[succ];
    }

    live -= kill;
    live_sets[i] = live;
  }

  auto stop = chrono::high_resolution_clock::now();
  auto us = chrono::duration_cast<chrono::microseconds>(stop - start);

  fmt::print("liveness analysis took {}us\n", us.count());
  for (size_t t = 0; t < num_tasks; ++t) {
    fmt::print("live set for task #{}:", t);
    for (size_t i = 0; i < num_temporaries; ++i) {
      if (live_sets[t].test(i)) {
        fmt::print("{},", temporaries[i]->name());
      }
    }
    fmt::print("\n");
  }

  //----------------------------------------------
  // build interference graph
  adjacency_matrix g{num_temporaries};

  for (size_t i = 0; i < num_tasks; ++i) {
    const auto &live = live_sets[i];
    // for each task, add an edge between values that live at the same time
    for (size_t ta = 0; ta < num_temporaries; ++ta) {
      for (size_t tb = 0; tb < ta; ++tb) {
        if (live[ta] && live[tb]) {
          g.add_edge(ta, tb);
        }
      }
    }
    // add edges between live vars in parallel tasks
    for (auto j = 0; j < i; ++j) {
      const auto &live_b = live_sets[j];
      if (!reachability[i][j]) {
        for (size_t ta = 0; ta < num_temporaries; ++ta) {
          for (size_t tb = 0; tb < num_temporaries; ++tb) {
            if (ta != tb && live[ta] && live_b[tb]) {
              g.add_edge(ta, tb);
            }
          }
        }
      }
    }
  }

  g.dump();

  // return g;

  return live_sets;
} // namespace

adjacency_matrix
build_interference_graph(const std::vector<virtual_resource_ptr> &temporaries,
                         const std::vector<live_set> &            live_sets) {
  const auto       num_temporaries = temporaries.size();
  adjacency_matrix g{num_temporaries};

  for (auto &&lset : live_sets) {
    // for each task, add an edge between values that live at the same time
    for (size_t i = 0; i < num_temporaries; ++i) {
      for (size_t j = 0; j < num_temporaries; ++j) {
        if (i != j) {
          if (lset.test(i) && lset.test(j)) {
            g.add_edge(i, j);
          }
        }
      }
    }
  }

  // add edges between parallel nodes

  return g;
}

void allocate_resources(const std::vector<virtual_resource_ptr> &temporaries,
                        const adjacency_matrix &                 interference) {
  // compute the degree of the interference graph: this gives us a higher
  // bound on the number of colors
  auto num_temporaries = temporaries.size();

  // compute the degree of the interference graph: this gives us a higher
  // bound on the number of colors
  int              nmax = 0;
  temporary_index  inmax = 0;
  std::vector<int> degrees;
  degrees.reserve(num_temporaries);

  for (temporary_index i = 0; i < num_temporaries; ++i) {
    int nn = 0;
    for (temporary_index j = 0; j < num_temporaries; ++j) {
      if (interference(i, j)) {
        ++nn;
      }
    }
    degrees.push_back(nn);
    if (nn > nmax) {
      nmax = nn;
      inmax = i;
    }
  }
  nmax = nmax + 1;

  // sort temporaries by interference degree
  std::vector<temporary_index> sorted_by_degree;
  sorted_by_degree.resize(num_temporaries, 0);
  std::iota(sorted_by_degree.begin(), sorted_by_degree.end(), 0);
  std::sort(sorted_by_degree.begin(), sorted_by_degree.end(),
            [&degrees](temporary_index a, temporary_index b) {
              return degrees[a] > degrees[b];
            });

  // color the graph
  std::vector<int> coloring;
  coloring.resize(num_temporaries, -1);
  int ncolors = 0;

  for (auto tmp : sorted_by_degree) {
    // look at the already colored neighbors and choose the minimum color not
    // attributed to them
    std::unordered_set<int> neighbor_colors;
    for (int i = 0; i < num_temporaries; ++i) {
      if (interference(tmp, i)) {
        if (coloring[i] != -1) {
          neighbor_colors.insert(coloring[i]);
        }
      }
    }

    for (int c = 0; c < nmax; ++c) {
      if (neighbor_colors.count(c) == 0) {
        coloring[tmp] = c;
        ncolors = std::max(c + 1, ncolors);
        break;
      }
    }
  }

  fmt::print("Coloring:\n");
  for (temporary_index i = 0; i < num_temporaries; ++i) {
    fmt::print(" - `{}` => {}\n", temporaries[i]->name(), coloring[i]);
  }

  // allocate resources: for each color, allocate a resource that fits the use
  for (int c = 0; c < ncolors; ++c) {
    temporary_index first_colored = invalid_temporary_index;
    for (temporary_index i = 0; i < num_temporaries; ++i) {
      if (coloring[i] == c) {
        if (first_colored == invalid_temporary_index) {
          // first time we see this color, allocate the resource
          first_colored = i;
          temporaries[i]->allocate();
        } else {
          // alias the resource with the one already created for this color
          temporaries[i]->alias_with(*temporaries[first_colored]);
        }
      }
    }
  }
}

void dump_tasks(std::ostream &out, const std::vector<task> &tasks,
                const std::vector<virtual_resource_ptr> &temporaries,
                const std::vector<live_set> &            live_sets) {
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

  if (pending_tasks_.empty()) {
    return;
  }

  fmt::print("=== submitting batch #{} ===\n", current_batch_);

  // insert the dummy end task for liveness analysis
  /*task end_task;
  end_task.name = "END";
  for (std::size_t i = 0; i < temporaries_.size(); ++i) {
    if (!temporaries_[i]->discarded_) {
      // externally reachable
      end_task.add_read(i);
      // TODO preds?
      // end_task.preds.push_back();
    }
  }
  pending_tasks_.push_back(std::move(end_task));*/

  // we need to assign concrete resources to virtual resources
  /*const auto num_temporaries = temporaries_.size();
  temporaries.reserve(num_temporaries);
  for (auto &&tmp : pending_resources_) {
    temporaries.push_back(tmp.get());
  }*/

  fmt::print("Temporaries:\n");
  for (auto tmp : temporaries_) {
    if (tmp->name().empty()) {
      fmt::print(" - temporary <unnamed> (discarded={})\n", tmp->discarded_);
    } else {
      fmt::print(" - temporary `{}` (discarded={})\n", tmp->name(),
                 tmp->discarded_);
    }
  }

  fmt::print("Tasks:\n");
  int task_index = 0;
  for (auto &&t : pending_tasks_) {
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

  auto live_sets = compute_liveness(temporaries_, pending_tasks_);
  // auto interference = build_interference_graph(temporaries_, live_sets);
  // interference.dump();
  // allocate_resources(temporaries_, interference);

  // allocate resources for each color
  // register allocation

  /*fmt::print("Liveness:\n");
  const auto num_tasks = pending_tasks_.size();
  for (std::size_t i = 0; i < num_tasks; ++i) {
    const auto live_set = get_live_resources_at_step(pending_tasks_, i);

    fmt::print("  * {}: ", pending_tasks_[i].name);
    for (auto &&lv : live_set) {
      fmt::print("{},", lv->name());
    }
    fmt::print("\n");
  }*/

  {
    std::ofstream out_graphviz{fmt::format("graal_test_{}.dot", current_batch_),
                               std::ios::trunc};
    dump_tasks(out_graphviz, pending_tasks_, temporaries_, live_sets);
  }

  for (auto &&tmp : temporaries_) {
    tmp->tmp_index_ = invalid_temporary_index;
  }

  temporaries_.clear();
  pending_tasks_.clear();
  current_batch_++;
}

temporary_index queue_impl::add_temporary(virtual_resource_ptr resource) {
  if (resource->tmp_index_ != invalid_temporary_index) {
    // already added
    return resource->tmp_index_;
  }
  auto tmp_index = temporaries_.size();
  resource->tmp_index_ = tmp_index;
  temporaries_.push_back(resource);
  return tmp_index;
}

} // namespace detail

void scheduler::add_resource_access(
    detail::resource_tracker &tracker, access_mode mode,
    std::shared_ptr<detail::virtual_resource> virt_res) {
  if (tracker.batch == batch_index_ &&
      (mode == access_mode::read_only || mode == access_mode::read_write)) {
    // if the last write was in a different batch
    // we know that all commands have been submitted.

    auto &pred_succ = queue_.pending_tasks_[tracker.last_producer].succs;
    if (std::find(pred_succ.begin(), pred_succ.end(), task_index_) ==
        pred_succ.end()) {
      // add current task in the list of successors
      pred_succ.push_back(task_index_);
    }
    // add the producer to the current list of predecessors
    task_.preds.push_back(tracker.last_producer);
  }

  if (mode == access_mode::read_write || mode == access_mode::write_only) {
    tracker.batch = batch_index_;
    tracker.last_producer = task_index_;
  }

  // there's an associated virtual resource, track usage
  if (virt_res) {
    auto tmp_index = queue_.add_temporary(virt_res);

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
}

} // namespace graal