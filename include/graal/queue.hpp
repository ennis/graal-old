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
#include <graal/detail/resource.hpp>
#include <graal/detail/task.hpp>
#include <graal/device.hpp>
#include <graal/image.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <graal/image_usage.hpp>
#include <graal/range.hpp>

namespace graal {

namespace detail {
class queue_impl;
} // namespace detail

/// @brief Command handler.
class handler {
  friend class detail::queue_impl;

  template <image_type, image_usage, access_mode, bool>
  friend class image_accessor;

  template <typename, buffer_usage, access_mode, bool>
  friend class buffer_accessor;

public:
  /// @brief Adds a callback function that is executed during submission.
  template <typename F> void add_command(F &&command_callback) {
    static_assert(std::is_invocable_v<F>,
                  "command callback has an invalid signature");
    task_.callbacks.push_back(std::move(command_callback));
  }


  // Called by buffer accessors to register an use of a buffer in a task
  template<typename T, bool HostVisible>
  void add_buffer_access(buffer<T, HostVisible>& buffer,
      access_mode mode, buffer_usage usage) 
  {
      add_buffer_access(buffer.impl_, mode, usage);
  }

  // Called by image accessors to register an use of an image in a task
  template<image_type Type, bool HostVisible>
  void add_image_access(image<Type, HostVisible> image,
      access_mode mode, image_usage usage) 
  {
      add_image_access(image.impl_, mode, usage);
  }

private:
  handler(detail::queue_impl &queue, detail::task &t,
          detail::task_index task_index, detail::batch_index batch_index)
      : queue_{queue}, task_{t}, task_index_{task_index}, batch_index_{
                                                              batch_index} {}

  // Called by buffer accessors to register an use of a buffer in a task
  void add_buffer_access(std::shared_ptr<detail::buffer_impl> buffer,
                         access_mode mode, buffer_usage usage);

  // Called by image accessors to register an use of an image in a task
  void add_image_access(std::shared_ptr<detail::image_impl> image,
                        access_mode mode, image_usage usage);

  void add_resource_access(std::shared_ptr<detail::resource> resource,
                           access_mode                       mode);

  detail::queue_impl &queue_;
  detail::task &      task_;
  detail::task_index  task_index_;
  detail::batch_index batch_index_;
};

namespace detail {

class queue_impl {
  friend class ::graal::handler;

public:
  queue_impl(device_impl_ptr device) : device_{std::move(device)} {}

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
  temporary_index add_temporary(resource_ptr resource);

  // set of all resources used in the batch (virtual or not)
  device_impl_ptr           device_;
  std::vector<resource_ptr> temporaries_;
  // pending tasks, a.k.a "batch"
  std::vector<task> tasks_;
  batch_index       current_batch_ = 0;
};

} // namespace detail

/// @brief
class queue {
public:
  queue(device &device)
      : impl_{std::make_shared<detail::queue_impl>(device.impl_)} {}

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