#pragma once
#include <cstdint>

namespace graal::detail {

/// @brief Sequence numbers uniquely identify a task.
using serial_number = uint64_t;

/// @brief A combination of a serial number and a queue, identifying a submission.
struct submission_number {
	uint64_t queue : 2 = 0;
	uint64_t serial : 62 = 0;
};

}  // namespace graal::detail