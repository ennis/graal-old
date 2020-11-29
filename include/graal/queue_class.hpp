#pragma once
#include <cstddef>

namespace graal {
/// @brief Queue classes.
/// In graal, there are a number of queue classes, specialized for a particular type of workload.
/// Each queue class maps to a VkQueue (or more precisely, to a queue index).
/// However, they don't necessarily map to different VkQueues.
///
/// Example: A GPU with specialized queues for compute and transfer, but graphics and present on the same queue.
///     Queues: [GRAPHICS, COMPUTE, TRANSFER]
///     Queue class map:
///         graphics => 0,
///         compute => 1,
///         transfer => 2,
///         present => 0
///
/// Example: A GPU with only one queue:
///     Queues: [GENERAL]
///     Queue class map:
///         graphics => 0,
///         compute => 0,
///         transfer => 0,
///         present => 0
enum class queue_class : size_t { graphics, compute, transfer, present, max };

/// @brief Maximum number of queues for the application
constexpr inline size_t max_queues = static_cast<size_t>(queue_class::max);
}  // namespace graal