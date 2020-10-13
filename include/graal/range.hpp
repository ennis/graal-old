#pragma once
#include <array>
#include <type_traits>

namespace graal {

template <int Dimensions> struct range {
  static_assert(Dimensions >= 1 || Dimensions <= 3,
                "size dimensions can be 1, 2 or 3");

  constexpr range() noexcept : arr{} {}

  template <int N = Dimensions, typename = std::enable_if_t<(N == 1), int>>
  constexpr range(size_t x) noexcept : arr{x} {}

  template <int N = Dimensions, typename = std::enable_if_t<(N == 2), int>>
  constexpr range(size_t x, size_t y) noexcept : arr{x, y} {}

  template <int N = Dimensions, typename = std::enable_if_t<(N == 3), int>>
  constexpr range(size_t x, size_t y, size_t z) noexcept : arr{x, y, z} {}

  template <int D> constexpr size_t get() const noexcept {
    static_assert(D < Dimensions);
    return arr[D];
  }

  constexpr size_t operator[](size_t index) const noexcept {
    return arr[index];
  }
  constexpr size_t &operator[](size_t index) noexcept { return arr[index]; }

  constexpr bool operator==(const range<Dimensions> &rhs) const {
    for (int i = 0; i < Dimensions; ++i) {
      if (arr[i] != rhs.arr[i]) {
        return false;
      }
    }
    return true;
  }

  constexpr bool operator!=(const range<Dimensions> &rhs) const {
    for (int i = 0; i < Dimensions; ++i) {
      if (arr[i] != rhs.arr[i]) {
        return true;
      }
    }
    return false;
  }

  std::array<size_t, Dimensions> arr;
};

// deduction guides
range(size_t)->range<1>;
range(size_t, size_t)->range<2>;
range(size_t, size_t, size_t)->range<3>;

using range_1d = range<1>;
using range_2d = range<2>;
using range_3d = range<3>;

} // namespace graal