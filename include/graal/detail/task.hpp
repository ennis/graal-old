#pragma once
#include <graal/detail/virtual_resource.hpp>
#include <string>
#include <vector>

namespace graal::detail {

/// @brief Index of a task in a batch.
using task_index = std::size_t;
inline constexpr task_index invalid_task_index = (std::size_t)-1;

/// @brief Absolute index of a batch in a queue.
using batch_index = std::size_t;
inline constexpr batch_index invalid_batch_index = (std::size_t)-1;

struct task {
  void add_read(temporary_index r);
  void add_write(temporary_index w);

  std::string                  name;
  std::vector<temporary_index> reads;
  std::vector<temporary_index> writes;
  std::vector<task_index>      preds;
};

/// @brief Used to track the last producer when using a resource in a queue.
struct resource_tracker {
  // TODO mutex-protected
  batch_index batch = invalid_batch_index;
  task_index  last_producer = invalid_task_index;
};

} // namespace graal::detail