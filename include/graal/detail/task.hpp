#pragma once
#include <functional>
#include <string>
#include <vector>

namespace graal::detail {

using temporary_index = std::size_t;
constexpr temporary_index invalid_temporary_index = (temporary_index)-1;

/// @brief Index of a task in a batch.
using task_index = std::size_t;
inline constexpr task_index invalid_task_index = (std::size_t)-1;

/// @brief Absolute index of a batch in a queue.
using batch_index = std::size_t;
inline constexpr batch_index invalid_batch_index = (std::size_t)-1;

using task_callback_fn = void();

struct task {
  void add_read(temporary_index r);
  void add_write(temporary_index w);

  std::string                                  name;
  std::vector<temporary_index>                 reads;
  std::vector<temporary_index>                 writes;
  std::vector<task_index>                      preds;
  std::vector<task_index>                      succs;
  std::vector<std::function<task_callback_fn>> callbacks;
};

} // namespace graal::detail