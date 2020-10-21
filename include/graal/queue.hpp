#pragma once
#include <deque>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <graal/access_mode.hpp>
#include <graal/buffer.hpp>
#include <graal/detail/image_resource.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/detail/task.hpp>
#include <graal/detail/virtual_resource.hpp>
#include <graal/image.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/range.hpp>
#include <graal/target.hpp>

namespace graal {

namespace detail {
class queue_impl;
} // namespace detail

/// @brief Command handler.
class handler {
  friend class detail::queue_impl;
  template <typename, image_type, target, access_mode, bool>
  friend class accessor;

public:
  /// @brief Adds a virtual resource access
  /// @param virt_res (optional)
  /// @param mode
  void add_resource_access(
      detail::resource_tracker &tracker, access_mode mode,
      std::shared_ptr<detail::virtual_resource> virt_res = nullptr);

  /// @brief Adds a callback function that is executed during submission.
  template <typename F> void add_command(F &&command_callback) {
    static_assert(std::is_invocable_v<F>,
                  "command callback has an invalid signature");
    task_.callbacks.push_back(std::move(command_callback));
  }

private:
  handler(detail::queue_impl &queue, detail::task &t,
          detail::task_index task_index, detail::batch_index batch_index)
      : queue_{queue}, task_{t}, task_index_{task_index}, batch_index_{
                                                              batch_index} {}

  detail::queue_impl &queue_;
  detail::task &      task_;
  detail::task_index  task_index_;
  detail::batch_index batch_index_;
};

namespace detail {

class queue_impl {
  friend class ::graal::handler;

public:
  template <typename F> void schedule(std::string name, F f) {
    // create task
    task t;
    t.name = std::move(name);
    task_index ti = tasks_.size();
    handler    sched{*this, t, ti, current_batch_};
    f(sched);
    tasks_.push_back(std::move(t));
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

  // temporary_index add_temporary(image);
  temporary_index add_temporary(virtual_resource_ptr resource);

  // set of all resources used in the batch (virtual or not)
  std::vector<virtual_resource_ptr> temporaries_;

  // resource
  // - virtual discard() (all resources discardable -> resource renaming)
  // - batch_resource_index
  //
  // virtual_resource : public resource
  // virtual_buffer_resource : public virtual_resource
  // virtual_image_resource: public virtual_resource
  // image_resource : public resource
  // image_base_impl<type> : public image_resource
  // buffer_base_impl : public resource
  //
  //

  // pending tasks, a.k.a "batch"
  std::vector<task> tasks_;
  batch_index       current_batch_ = 0;
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