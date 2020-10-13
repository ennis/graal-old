#pragma once
#include <fmt/format.h>
#include <stdexcept>

namespace graal {

/// @brief
class unimplemented_error : public std::logic_error {
public:
    unimplemented_error() = default;

  unimplemented_error(const char *what)
      : std::logic_error{fmt::format("unimplemented: {}", what)} {}
};

} // namespace graal