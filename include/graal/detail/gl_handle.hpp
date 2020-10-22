#pragma once
#include <cstddef>
#include <graal/gl_types.hpp>
#include <utility>

namespace graal::detail {

/// Wrapper to use GLuint as a unique_ptr handle type
/// http://stackoverflow.com/questions/6265288/unique-ptr-custom-storage-type-example/6272139#6272139
/// This type follows the same semantics as unique_ptr.
/// Deleter is a functor type with a static operator()(GLuint) member, that is
/// in charge of deleting the OpenGL object by calling glDeleteX The deleter is
/// automatically called when obj != 0 and:
///		- the GLHandle goes out of scope
///		- the GLHandle is move-assigned another GLHandle
template <typename Deleter> class gl_handle {
public:
  /// @brief
  /// @param obj_
  gl_handle(GLuint obj_) : obj{obj_} {}

  /// @brief
  /// @param
  gl_handle(std::nullptr_t = nullptr) : obj{0} {}

  /// @brief
  /// @param rhs
  /// @return
  gl_handle(gl_handle &&rhs) noexcept : obj{rhs.obj} { rhs.obj = 0; }

  /// @brief Move-assignment operator: take ownership of the OpenGL resource
  /// @param rhs
  /// @return
  gl_handle &operator=(gl_handle &&rhs) noexcept {
    std::swap(obj, rhs.obj);
    return *this;
  }

  /// GL resources are not copyable, so delete the copy ctors
  gl_handle(const gl_handle &) = delete;
  gl_handle &operator=(const gl_handle &) = delete;

  ///
  ~gl_handle() {
    if (obj) {
      Deleter{}(obj);
      obj = 0;
    }
  }

  /// @brief
  /// @return
  unsigned int get() const { return obj; }

  /// @brief
  explicit operator bool() { return obj != 0; }

  /// @brief
  friend bool operator==(const gl_handle &l, const gl_handle &r) {
    return l.obj == r.obj;
  }

  /// @brief
  friend bool operator!=(const gl_handle &l, const gl_handle &r) {
    return !(l == r);
  }
  // default copy ctor and operator= are fine
  // explicit nullptr assignment and comparison unneeded
  // because of implicit nullptr constructor
  // swappable requirement fulfilled by std::swap

private:
  unsigned int obj;
};

} // namespace graal::detail
