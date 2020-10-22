#pragma once
#include <graal/detail/virtual_resource.hpp>
#include <graal/glad.h>
#include <graal/raw_buffer.hpp>
#include <memory>
#include <span>
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
  // construct with unspecified size
  buffer_impl() {}

  // construct uninitialized with size
  buffer_impl(std::size_t size) { buffer_impl_base<T>::set_size(size); }

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
  // construct with unspecified size
  buffer_impl() {}

  // construct uninitialized from size
  buffer_impl(std::size_t size) { buffer_impl_base<T>::set_size(size); }

  // construct with initial data
  template <size_t Extent> buffer_impl(std::span<const T, Extent> data) {
    buffer_impl_base<T>::set_size(data.size());
    auto size_bytes = data.size_bytes();
    buffer_ = create_buffer(size_bytes, data.data(),
                            GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT |
                                GL_MAP_WRITE_BIT);
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
  using impl_t = detail::buffer_impl<T, ExternalAccess>;
  /// @brief
  buffer() : impl_{std::make_shared<impl_t>()} {}

  /// @brief
  /// @param size
  buffer(std::size_t size) : impl_{std::make_shared<impl_t>(size)} {}

  /// @brief
  /// @param data
  template <size_t Extent>
  buffer(std::span<const T, Extent> data)
      : impl_{std::make_shared<impl_t>(data)} {}

  /// @brief
  /// @param size
  void set_size(std::size_t size) { impl_->set_size(size); }

  /// @brief
  /// @return
  std::size_t size() const { return impl_->size(); }

  /// @brief
  /// @return
  bool has_size() const noexcept { return impl_->has_size(); }

  /// @brief
  /// @return
  template <bool Ext = ExternalAccess>
  std::enable_if_t<Ext, GLuint> get_gl_object() const {
    return impl_->get_gl_object();
  }

private:
  std::shared_ptr<impl_t> impl_;
};

template <typename T> using virtual_buffer = buffer<T, false>;

// deduction guides
template <typename T> buffer(std::span<const T>) -> buffer<T, true>;

} // namespace graal