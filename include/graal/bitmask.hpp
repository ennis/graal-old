#pragma once
#include <type_traits>

namespace graal {
#define GRAAL_BITMASK(T)                                                                      \
    [[nodiscard]] inline constexpr T operator&(T left, T right) noexcept {                    \
        using UT = std::underlying_type_t<T>;                                                 \
        return static_cast<T>(static_cast<UT>(left) & static_cast<UT>(right));                \
    }                                                                                         \
                                                                                              \
    [[nodiscard]] constexpr T operator|(T left, T right) noexcept {                           \
        using UT = std::underlying_type_t<T>;                                                 \
        return static_cast<T>(static_cast<UT>(left) | static_cast<UT>(right));                \
    }                                                                                         \
                                                                                              \
    [[nodiscard]] constexpr T operator^(T left, T right) noexcept { /* return left ^ right */ \
        using UT = std::underlying_type_t<T>;                                                 \
        return static_cast<T>(static_cast<UT>(left) ^ static_cast<UT>(right));                \
    }                                                                                         \
                                                                                              \
    inline constexpr T& operator&=(T& left, T right) noexcept { /* return left &= right */    \
        return left = left & right;                                                           \
    }                                                                                         \
                                                                                              \
    inline constexpr T& operator|=(T& left, T right) noexcept { /* return left |= right */    \
        return left = left | right;                                                           \
    }                                                                                         \
                                                                                              \
    inline constexpr T& operator^=(T& left, T right) noexcept { /* return left ^= right */    \
        return left = left ^ right;                                                           \
    }                                                                                         \
                                                                                              \
    [[nodiscard]] inline constexpr T operator~(T left) noexcept { /* return ~left */          \
        using UT = std::underlying_type_t<T>;                                                 \
        return static_cast<T>(~static_cast<UT>(left));                                        \
    }                                                                                         \
                                                                                              \
    [[nodiscard]] inline constexpr bool bitmask_includes(                                     \
            T left, T elem) noexcept { /* return (left & elem) != T{} */                      \
        return (left & elem) != T{};                                                          \
    }                                                                                         \
                                                                                              \
    inline constexpr bool operator!(T v) noexcept { /* return left ^= right */                \
        return v != T{};                                                                      \
    }                                                                                         \
                                                                                              \
    [[nodiscard]] inline constexpr bool bitmask_includes_all(                                 \
            T left, T elem) noexcept { /* return (left & elem) == elem */                     \
        return (left & elem) == elem;                                                         \
    }

}  // namespace graal