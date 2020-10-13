#pragma once
#include "types.hpp"
#include <cstddef>
#include <utility>

namespace graal::gl {

/// Wrapper to use GLuint as a unique_ptr handle type
/// http://stackoverflow.com/questions/6265288/unique-ptr-custom-storage-type-example/6272139#6272139
/// This type follows the same semantics as unique_ptr.
/// Deleter is a functor type with a static operator()(GLuint) member, that is
/// in charge of deleting the OpenGL object by calling glDeleteX The deleter is
/// automatically called when obj != 0 and:
///		- the GLHandle goes out of scope
///		- the GLHandle is move-assigned another GLHandle
template <typename Deleter> class handle {
public:
  /// @brief
  /// @param obj_
  handle(GLuint obj_) : obj{obj_} {}

  /// @brief
  /// @param
  handle(std::nullptr_t = nullptr) : obj{0} {}

  /// @brief
  /// @param rhs
  /// @return
  handle(handle&&rhs) noexcept : obj{rhs.obj} { rhs.obj = 0; }

  /// @brief Move-assignment operator: take ownership of the OpenGL resource
  /// @param rhs
  /// @return
  handle &operator=(handle &&rhs) noexcept {
    std::swap(obj, rhs.obj);
    return *this;
  }

  /// GL resources are not copyable, so delete the copy ctors
  handle(const handle &) = delete;
  handle &operator=(const handle &) = delete;

  ///
  ~handle() {
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
  friend bool operator==(const handle &l, const handle &r) {
    return l.obj == r.obj;
  }

  /// @brief
  friend bool operator!=(const handle &l, const handle &r) { return !(l == r); }
  // default copy ctor and operator= are fine
  // explicit nullptr assignment and comparison unneeded
  // because of implicit nullptr constructor
  // swappable requirement fulfilled by std::swap

private:
  unsigned int obj;
};

} // namespace graal::gl
