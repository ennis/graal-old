#include <algorithm>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <fmt/format.h>
#include <graal/queue.hpp>
#include <numeric>
#include <span>
#include <unordered_set>

namespace graal {
namespace detail {
namespace {

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

class interference_graph {
public:
  interference_graph(std::size_t num_temporaries) : n_{num_temporaries} {
    adj_matrix_.resize((n_ * (n_ + 1)) / 2, false);
  }

  void add_edge(std::size_t a, std::size_t b) { adj_matrix_[idx(a, b)] = true; }

  bool operator()(std::size_t a, std::size_t b) const {
    return adj_matrix_[idx(a, b)];
  }

  void dump() {
    for (std::size_t i = 0; i < n_; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        if (this->operator()(i, j)) {
          fmt::print(" - v{} -- v{}\n", i, j);
        }
      }
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
  std::vector<bool> adj_matrix_;
};

using live_set = std::vector<temporary_index>;

/// @brief
/// @param tasks
/// @param task_index
/// @return
std::vector<live_set>
compute_liveness(const std::vector<virtual_resource_ptr> &temporaries,
                 const std::vector<detail::task> &        tasks) {
  namespace chrono = std::chrono;
  auto start = chrono::high_resolution_clock::now();

  const size_t          num_tasks = tasks.size();
  std::vector<live_set> live_sets;
  live_sets.reserve(num_tasks);

  live_set cur_live;
  live_set cur_live_2;

  for (size_t t = 0; t < num_tasks; ++t) {

    // Compute the live set task index t, which is the union of the live-in set
    // and the live-out set. Note that most descriptions of liveness analysis
    // produce the live-out set, as the base from which to build the
    // interference graph. In our case, we must also consider "inputs" as live.
    // For instance, given the statement:
    //    a <- f(b)
    // if live_in = { b } and live_out = { a }, then a register allocation
    // algorithm could assign the same register for a and b:
    //    r <- f(r)
    // This does not work with GPU textures, because f could represent a draw
    // operation that samples a texture, and you'd be sampling and writing to
    // the same texture at the same time.

    // add defs to the live set
    cur_live_2.clear();
    std::set_union(cur_live.begin(), cur_live.end(), tasks[t].writes.begin(),
                   tasks[t].writes.end(), std::back_inserter(cur_live_2));

    // *also* add uses to the live set. see comment above
    cur_live.clear();
    std::set_union(cur_live_2.begin(), cur_live_2.end(), tasks[t].reads.begin(),
                   tasks[t].reads.end(), std::back_inserter(cur_live));

    // live_set(t) = live_set(t-1) U defs(t) U uses(t)

    // remove dead vars
    // TODO should only consider vars that are in the def/use set
    // otherwise we're searching for nothing
    auto it = std::remove_if(cur_live.begin(), cur_live.end(), [&](auto v) {
      // if var in current def/use set, skip
      if (std::binary_search(tasks[t].writes.begin(), tasks[t].writes.end(),
                             v) ||
          std::binary_search(tasks[t].reads.begin(), tasks[t].reads.end(), v)) {
        return false;
      }

      bool killed = true;
      for (size_t succ = t + 1; succ < num_tasks; ++succ) {
        if (std::binary_search(tasks[succ].writes.begin(),
                               tasks[succ].writes.end(), v)) {
          break; // there's a def, so it's dead
        }
        if (std::binary_search(tasks[succ].reads.begin(),
                               tasks[succ].reads.end(), v)) {
          killed = false; // there's a use without a def before
          break;
        }
      }
      // if we fall off the loop, this means that there were no
      // no defs, and no uses: assume it's dead
      return killed;
    });
    cur_live.erase(it, cur_live.end());

    // live_set(t) = (live_set(t-1) U defs(t) U uses(t)) - killed(t)

    live_sets.push_back(cur_live);
  }

  auto stop = chrono::high_resolution_clock::now();
  auto us = chrono::duration_cast<chrono::microseconds>(stop - start);

  fmt::print("liveness analysis took {}us\n", us.count());
  for (size_t t = 0; t < num_tasks; ++t) {
    fmt::print("live set for task #{}:", t);
    for (auto &&live : live_sets[t]) {
      fmt::print("{},", temporaries[live]->name());
    }
    fmt::print("\n");
  }

  return live_sets;
}

interference_graph
build_interference_graph(const std::vector<virtual_resource_ptr> &temporaries,
                         const std::vector<live_set> &            live_sets) {
  interference_graph g{temporaries.size()};

  for (auto &&lset : live_sets) {
    // for each task, add an edge between values that live at the same time
    for (auto a : lset) {
      for (auto b : lset) {
        if (a != b) {
          g.add_edge(a, b);
        }
      }
    }
  }

  return g;
}


void allocate_resources(const std::vector<virtual_resource_ptr> &temporaries,
                    const interference_graph &               interference)
{
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

  // allocate resources for each color

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

  // insert the dummy end task for liveness analysis
  task end_task;
  end_task.name = "END";
  for (std::size_t i = 0; i < temporaries_.size(); ++i) {
    if (!temporaries_[i]->discarded_) {
      // externally reachable
      end_task.add_read(i);
    }
  }
  pending_tasks_.push_back(std::move(end_task));

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
    task_index++;
  }

  auto live_sets = compute_liveness(temporaries_, pending_tasks_);
  auto interference = build_interference_graph(temporaries_, live_sets);
  interference.dump();
  allocate_resources(temporaries_, interference);

  // allocate resources for each color
  // register allocation

  //

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

  for (auto &&tmp : temporaries_) {
    tmp->tmp_index_ = invalid_temporary_index;
  }

  temporaries_.clear();
  pending_tasks_.clear();
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
    std::shared_ptr<detail::virtual_resource> virt_res, access_mode mode) {
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

} // namespace graal