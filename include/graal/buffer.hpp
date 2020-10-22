#pragma once
#include <graal/detail/virtual_resource.hpp>
#include <graal/raw_buffer.hpp>
#include <graal/glad.h>
#include <memory>
#include <stdexcept>

namespace graal {

namespace detail {

template <typename T> class buffer_impl_base {
public:
  void set_size(std::size_t size) {
    if (has_size() && size_ != size) {
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

template <typename T, bool ExternalAccess> class buffer_impl;

// buffer_impl<ExternalAccess=false>
template <typename T> class buffer_impl<T, false> : public buffer_impl_base<T> {
public:
  buffer_impl(std::size_t size) {
    // TODO
  }

  std::shared_ptr<virtual_buffer_resource> get_virtual_buffer() {
    if (!virt_buffer_) {
      // create a virtual image now
      auto s = buffer_impl_base<T>::size();
      virt_buffer_ = std::make_shared<virtual_buffer_resource>(s);
      virt_buffer_->set_name(std::string{named_object::name()});
    }
    return virt_buffer_;
  }

private:
  std::shared_ptr<virtual_buffer_resource> virt_buffer_;
};

// buffer_impl<ExternalAccess=true>
template <typename T> class buffer_impl<T, true> : public buffer_impl_base<T> {
public:
  buffer_impl(std::size_t size) {
    // TODO
  }

  GLuint get_gl_object() const {
    if (!buffer_.get()) {
      // no object allocated yet, allocate now
      auto s = buffer_impl_base<T>::size() * sizeof(T);
      // TODO infer flags from accesses, or specify dynamically
      buffer_ = create_buffer(s, nullptr,
                                  GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT |
                                      GL_MAP_WRITE_BIT);
    }

    /*if (sync) {
        //
    }
    */
    return buffer_.get();
  }

  /*  /// @brief Sets the queue that the buffer is currently used on
    /// @param queue
    void set_batch(queue queue, batch_index batch) {
        queue_ = std::move(queue);
    }*/

private:
  // queue queue_;
  buffer_handle buffer_;
};

} // namespace detail

/// @brief
/// @tparam T
template <typename T, bool ExternalAccess = true> class buffer {
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
  template <bool Ext = ExternalAccess>
  std::enable_if_t<Ext, GLuint> get_gl_object() const {
    return impl_->get_gl_object();
  }

private:
  std::shared_ptr<detail::buffer_impl<T, ExternalAccess>> impl_;
};

} // namespace graal