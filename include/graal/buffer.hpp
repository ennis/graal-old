#pragma once
#include <graal/gl/glad.h>
#include <memory>
#include <stdexcept>

namespace graal {

namespace detail {
template <typename T> class buffer_impl {
public:
  buffer_impl(std::size_t size) {
    // TODO
  }

  GLuint get_gl_object() const {
    // TODO
  }

  void specify_size(std::size_t size) {
    if (has_size()) {
      throw std::logic_error{"size already specified"};
    }
    size_ = size;
  }

  std::size_t size() const {
    if (!has_size()) {
      throw std::logic_error{"size was unspecified"};
    }
    return size_;
  }

  bool has_size() const { return size_ != -1; }

private:
  std::size_t size_ = -1;
};
} // namespace detail

/// @brief 
/// @tparam T 
template <typename T> class buffer {
public:
  /// @brief
  buffer() : impl_{std::make_shared<detail::buffer_impl>()} {}

  /// @brief
  /// @param size
  buffer(std::size_t size)
      : impl_{std::make_shared<detail::buffer_impl>(size)} {}

  /// @brief
  /// @param size
  void specify_size(std::size_t size) { impl_->specify_size(size); }

  /// @brief
  /// @return
  std::size_t size() const { return impl_->size(); }

  /// @brief
  /// @return
  bool has_size() const { return impl_->has_size(); }

  /// @brief
  /// @return
  GLuint get_gl_object() const;

private:
  std::shared_ptr<detail::buffer_impl<T>> impl_;
};

} // namespace graal