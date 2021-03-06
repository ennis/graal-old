#pragma once
#include <optional>
#include <vector>
#include <cassert>
#include <utility>

namespace graal::detail {

/// @brief Represents an "owned" handle to a vulkan object.
/// The object/scope that has a vk_handle owns the object and is responsible for its deletion.
/// The object is not automatically deleted, as it would need a backpointer to the device or instance.
/// Instead, the owner must call `release` and delete the object manually, or transfer ownership to another
/// object or function that takes charge of the deletion.
/// The destructor of `vk_handle` asserts if the handle is not nullptr.
template<typename T>
class handle {
public:
    constexpr handle() noexcept : handle_{nullptr} {
    }
    constexpr handle(std::nullptr_t) noexcept : handle_{nullptr} {
    }
    constexpr explicit handle(T handle) noexcept : handle_{handle} {
    }

    ~handle() noexcept {
        reset();
    }

    handle(const handle<T>&) = delete;
    handle<T>& operator=(const handle<T>&) = delete;

    handle(handle<T>&& other) noexcept : handle_{ std::exchange(other.handle_, nullptr) } {
    }

    handle<T>& operator=(handle<T>&& other) noexcept {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    handle<T>& operator=(std::nullptr_t) noexcept {
        reset();
        handle_ = nullptr;
        return *this;
    }

    [[nodiscard]] T release() noexcept {
        auto h = handle_;
        handle_ = nullptr;
        return h;
    }

    [[nodiscard]] T get() const noexcept {
        return handle_;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return handle_ != nullptr;
    }

    void reset() noexcept {
        assert(!handle_ && "vulkan handle was not null");
    }

    friend auto operator<=>(const handle<T>&, const handle<T>&) = default;

private:
    T handle_;
};

/// @brief
/// @tparam T
template<typename T>
class handle_vector {
public:
    using value_type = T;
    using const_iterator = typename std::vector<T>::const_iterator;
    using const_reverse_iterator = typename std::vector<T>::const_iterator;

    handle_vector() = default;

    ~handle_vector() {
        assert(empty() && "handle vector not empty");
    }

    handle_vector(const handle_vector&) = delete;
    handle_vector& operator=(const handle_vector&) = delete;

    handle_vector(handle_vector&&) = default;
    handle_vector& operator=(handle_vector&&) = default;

    /// @brief
    /// @return
    [[nodiscard]] bool empty() const noexcept {
        return handles_.empty();
    }

    /// @brief
    /// @param handle
    T push_back(handle<T> handle) {
        const auto h = handle.release();
        handles_.push_back(h);
        return h;
    }

    /// @brief
    /// @return
    handle<T> pop_back() {
        handle<T> top{handles_.back()};
        handles_.pop_back();
        return top;
    }

    /// @brief
    /// @param i
    /// @return
    [[nodiscard]] T operator[](size_t i) const noexcept {
        return handles_[i];
    }

    /// @brief
    /// @tparam Pred
    /// @param pred
    /// @return
    handle<T> swap_remove_at(size_t index) {
        if (index != handles_.size() - 1) { std::swap(handles_[index], handles_.back()); }
        return pop_back();
    }

    /// @brief
    /// @param index
    /// @return
    handle<T> remove_at(size_t index) {
        std::rotate(handles_.begin() + index, handles_.begin() + index + 1, handles_.end());
        return pop_back();
    }

    std::vector<T> release_all() noexcept {
        return std::exchange(handles_, std::vector<T>{});
    }

    [[nodiscard]] const_iterator begin() const noexcept {
        return handles_.begin();
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return handles_.end();
    }

    [[nodiscard]] const_iterator cbegin() const noexcept {
        return handles_.cbegin();
    }

    [[nodiscard]] const_iterator cend() const noexcept {
        return handles_.cend();
    }

    [[nodiscard]] const_reverse_iterator rbegin() const noexcept {
        return handles_.rbegin();
    }

    [[nodiscard]] const_reverse_iterator rend() const noexcept {
        return handles_.rend();
    }

private:
    std::vector<T> handles_;
};

}  // namespace graal::detail