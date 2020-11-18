#include <graal/detail/task.hpp>

namespace graal::detail {

namespace {

template<typename T>
void sorted_vector_insert(std::vector<T>& vec, T elem) {
    auto it = std::lower_bound(vec.begin(), vec.end(), elem);
    vec.insert(it, std::move(elem));
}

}  // namespace

void task::add_read(temporary_index r) {
    sorted_vector_insert(reads, std::move(r));
}

void task::add_write(temporary_index w) {
    sorted_vector_insert(writes, std::move(w));
}

}  // namespace graal::detail