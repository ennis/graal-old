#pragma once
#include <vector>
#include <iterator>
#include <span>
#include <algorithm>
#include <utility>

namespace graal::detail {

/// @brief Helper class to recycle objects (mostly vulkan handles) across
/// batches.
/// @tparam T
template<typename T>
class recycler {
public:
    // use it like this
    //   T out_obj = {};
    //   if (!recycler.fetch(out_obj)) {
    //      out_obj = create_object(...);
    //   }
    bool fetch(T& out_obj) {
        if (!free_list_.empty()) {
            out_obj = std::move(free_list_.back());
            free_list_.pop_back();
            return true;
        }
        return false;
    }

    template <typename Pred>
    bool fetch_if(T& out_obj, Pred pred) {
        for (size_t i = 0; i < free_list_.size(); ++i) {
            if (pred(free_list_[i])) {
                if (i != free_list_.size() - 1) {
                    std::swap(free_list_[i], free_list_.back());
                }
                out_obj = free_list_.back();
                free_list_.pop_back();
                return true;
            }
        }
        return false;
    }

    /// @brief Recycle one object
    /// @param obj
    void recycle(T&& obj) {
        free_list_.push_back(std::move(obj));
    }

    /// @brief Recycle a vector of objects
    /// @param vec
    void recycle_vector(std::vector<T>&& vec) {
        recycle(vec.begin(), vec.end());
        vec.clear();
    }

    /// @brief Recycle objects in the given iterator range
    /// @tparam InputIt
    /// @param first
    /// @param last
    template<typename InputIt>
    void recycle(InputIt first, InputIt last) {
        std::move(first, last, std::back_inserter(free_list_));
    }

    /// @brief Returns a reference to the list of free objects.
    /// @return
    std::span<const T> free_list() const {
        return std::span{free_list_};
    }

    /// @brief Clears the free list.
    void clear() {
        free_list_.clear();
    }

private:
    std::vector<T> free_list_;
};
}  // namespace graal::detail