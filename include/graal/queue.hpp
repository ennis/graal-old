#pragma once
#include <deque>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <graal/access_mode.hpp>
#include <graal/detail/image_resource.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/virtual_resource.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/range.hpp>
#include <graal/target.hpp>

namespace graal {

namespace detail {

class queue_impl;
using task_id = std::size_t;

enum class image_load_op { keep, discard, clear };

struct task {
  void add_read(temporary_index r);
  void add_write(temporary_index w);

  std::string                  name;
  std::vector<temporary_index> reads;
  std::vector<temporary_index> writes;
};

} // namespace detail

class scheduler {
  friend class detail::queue_impl;
  template <typename, image_type, target, access_mode, bool>
  friend class accessor;

public:
  /// @brief
  /// @param virt_res
  /// @param mode
  void add_resource_access(std::shared_ptr<detail::virtual_resource> virt_res,
                           access_mode                               mode);

private:
  scheduler(detail::queue_impl &queue, detail::task &t)
      : queue_{queue}, task_{t} {}

  detail::queue_impl &queue_;
  detail::task &      task_;
};

namespace detail {

class queue_impl {
  friend class ::graal::scheduler;

public:
  template <typename F> void schedule(std::string name, F f) {
    // create task
    task t;
    t.name = std::move(name);
    scheduler sched{*this, t};
    f(sched);
    pending_tasks_.push_back(std::move(t));
  }

  void enqueue_pending_tasks();

private:
  // when is it safe to remove the entry?
  // - when it is not in use anymore
  // when does it become unused?
  // - when the resource is marked as discarded, and
  // - when the last task that "uses" this resource has finished
  // what does it mean for a "task" to use a resource?
  // - a use is just referencing the resource within the task
  // how do we know the task has finished?
  // - each task is put in a "batch" before execution
  // - the batch is guarded by a fence; when this fence is signalled, the
  // virtual resource is not in use anymore
  //
  // What about the concrete resource?
  // - it depends on which queue the concrete resource is going to be used
  // - in OpenGL, there's only one queue, so it's OK to re-use the concrete
  // resource
  //		if there are no uses in unsubmitted tasks

  temporary_index add_temporary(virtual_resource_ptr resource);

  // set of resources visible in the current batch
  std::vector<virtual_resource_ptr> temporaries_;
  // pending tasks, a.k.a "batch"
  std::vector<task> pending_tasks_;
};

} // namespace detail

/// @brief
class queue {
public:
  queue() : impl_{std::make_shared<detail::queue_impl>()} {}

  /// @brief
  /// @tparam F
  /// @param f
  template <typename F> void schedule(F f) {
    impl_->schedule("", std::forward<F>(f));
  }

  /// @brief
  /// @tparam F
  /// @param name
  /// @param f
  template <typename F> void schedule(std::string name, F f) {
    impl_->schedule(std::move(name), std::forward<F>(f));
  }

  void enqueue_pending_tasks() { impl_->enqueue_pending_tasks(); }

private:
  std::shared_ptr<detail::queue_impl> impl_;
};

} // namespace graal