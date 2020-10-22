#pragma once
#include <fmt/format.h>
#include <stdexcept>
#include <string_view>

namespace graal {

class shader_compilation_error : public std::exception {
public:
  shader_compilation_error(std::string log) : log_{std::move(log)} {}

  std::string_view log() const noexcept { return log_; }

private:
  std::string log_;
};

class program_link_error : public std::exception {
public:
  program_link_error(std::string log) : log_{std::move(log)} {}

  std::string_view log() const noexcept { return log_; }

private:
  std::string log_;
};

/// @brief
class unimplemented_error : public std::logic_error {
public:
  unimplemented_error() : std::logic_error{"unimplemented"} {}

  unimplemented_error(const char *what)
      : std::logic_error{fmt::format("unimplemented: {}", what)} {}
};

} // namespace graal