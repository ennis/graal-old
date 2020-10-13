#pragma once
#include <string>
#include <string_view>

namespace graal::detail {
class named_object {
public:
  explicit named_object() = default;

  std::string_view name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

private:
  std::string name_;
};
} // namespace graal::detail