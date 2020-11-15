#include <graal/detail/batch.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/queue.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <fstream>
#include <numeric>
#include <span>

namespace graal {
namespace detail {
namespace {

/// Converts a vector of intrusive_ptr to a vector of raw pointers.
template<typename T>
std::vector<T*> to_raw_ptr_vector(const std::vector<std::shared_ptr<T>>& v) {
    std::vector<T*> result;
    std::transform(
            v.begin(), v.end(), std::back_inserter(result), [](auto&& x) { return x.get(); });
    return result;
}

}  // namespace

//-----------------------------------------------------------------------------
class queue_impl {
public:
    queue_impl(device_impl_ptr device, const queue_properties& props);
    ~queue_impl();

    void enqueue_pending_tasks();
    void present(swapchain_image&& image);

    void wait_for_task(uint64_t sequence_number);
    const task* get_pending_task(uint64_t sequence_number) const noexcept;

    [[nodiscard]] handler begin_build_task(std::string_view name) {
        if (building_task_) { throw std::logic_error{"a task is already being built"}; }
        building_task_ = true;
        const auto task_sn = sequence_counter_ + 1;
        auto t = std::make_unique<detail::submit_task>();
        t->name = name;
        t->signal.sequence_number = task_sn;
        return handler{std::move(t)};
    }

    void end_build_task(handler&& h) {
        for (auto& a : h.accesses_) {
            register_resource_access(*h.task_, a.resource, a.mode);
        }
        current_batch_.add_task(std::move(h.task_));
        building_task_ = false;
        sequence_counter_++;
    }

    batch& current_batch() noexcept {
        return current_batch_;
    }

    const batch& current_batch() const noexcept {
        return current_batch_;
    }

private:
    void register_resource_access(
            detail::task& task, detail::resource_ptr resource, access_mode mode, recycler<vk::Semaphore>& semaphore_recycler);

    //command_buffer_pool get_command_buffer_pool();
    bool building_task_ = false;
    queue_properties props_;
    device_impl_ptr device_;
    batch current_batch_;
    uint64_t sequence_counter_ = 0;
    std::deque<batch> in_flight_batches_;
    vk::Semaphore timelines_[max_queues];
};

queue_impl::queue_impl(device_impl_ptr device, const queue_properties& props) :
    device_{std::move(device)}, props_{props}, current_batch_{device, 0} {
    auto vk_device = device_->get_vk_device();

    vk::SemaphoreTypeCreateInfo timeline_create_info{
            .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
    vk::SemaphoreCreateInfo semaphore_create_info{.pNext = &timeline_create_info};
    for (size_t i = 0; i < max_queues; ++i) {
        timelines_[i] = vk_device.createSemaphore(semaphore_create_info);
    }
}

queue_impl::~queue_impl() {
    auto vk_device = device_->get_vk_device();
    for (size_t i = 0; i < max_queues; ++i) {
        vk_device.destroySemaphore(timelines_[i]);
    }
}

const task* queue_impl::get_pending_task(uint64_t sequence_number) const noexcept {
    uint64_t start;
    uint64_t finish;

    for (auto&& b : in_flight_batches_) {
        start = b.start_sequence_number();
        finish = b.finish_sequence_number();
        if (sequence_number <= start) { return nullptr; }
        if (sequence_number > start && sequence_number <= finish) {
            return b.get_task(sequence_number);
        }
    }
    return nullptr;
}

void queue_impl::enqueue_pending_tasks() {
    //temporaries_.clear();
    //tasks_.clear();
    //sequence_counter_ = current_batch_.finish_sequence_number();
}

void queue_impl::present(swapchain_image&& image) {
    auto t = std::make_unique<present_task>();
    const auto sn = sequence_counter_++;
    t->name = "present";
    t->signal.sequence_number = sn;
    // access the image.
    // NOTE move out of the impl pointer: the image cannot be used after present
    // (conceptually, this is an ownership transfer to the presentation engine)
    register_resource_access(*t, std::move(image.impl_), access_mode::read_only);
}

void queue_impl::wait_for_task(uint64_t sequence_number) {
    /*assert(batch_to_wait <= current_batch_);
    const auto vk_device = device_->get_vk_device();
    const vk::Semaphore semaphores[] = {batch_index_semaphore_};
    const uint64_t values[] = {batch_to_wait};
    const vk::SemaphoreWaitInfo wait_info{
            .semaphoreCount = 1, .pSemaphores = semaphores, .pValues = values};
    vk_device.waitSemaphores(wait_info, 10000000000);  // 10sec batch timeout

    // reclaim resources of all batches with index <= batch_to_wait (we proved
    // that they are finished by waiting on the semaphore)
    while (!in_flight_batches_.empty() && in_flight_batches_.front().batch_index <= batch_to_wait) {
        auto& b = in_flight_batches_.front();

        // reclaim command buffer pools
        for (auto& t : b.threads) {
            t.cb_pool.reset(vk_device);
            free_cb_pools_.recycle(std::move(t.cb_pool));
        }

        // reclaim semaphores
        free_semaphores_.recycle_vector(std::move(b.semaphores));
        in_flight_batches_.pop_front();
    }*/
}

/// @brief Registers a resource access in a task
/// @param resource
/// @param mode
void queue_impl::register_resource_access(
     detail::task& task, detail::resource_ptr resource, access_mode mode, recycler<vk::Semaphore>& semaphore_recycler) 
{
    const auto vk_device = device_->get_vk_device();
    const auto last_write_sn = resource->last_write_sequence_number;
    const detail::task* last_write_task = get_pending_task(last_write_sn);
    const auto batch_start_sn = current_batch_.start_sequence_number();
    
    // last_write_sn == 0 means that the resource was never written by a task .
    // Note that it might still need synchronization with external processes (for an example with swapchain images, see below)
    if (last_write_sn) {
        if (last_write_task) {
            // Inter-batch synchronization

            // resource last write sequence number < start sequence number of current batch:
            // the last write to the resource was in a previous batch.
            // However, the task has to wait for the previous writer in a previous batch
            // to finish using the resource. The previous writer sequence number is stored in last_write,
            // and was set by the previous batch (if any).
            size_t q = last_write_task->signal.queue;
            task.waits[q] = std::max(task.waits[q], last_write_sn);
        }
        else {
            // Intra-batch synchronization

            // we can't do as above because we don't know yet on which queue the producer task will be scheduled
            if (mode == access_mode::read_only || mode == access_mode::read_write) {
                if (last_write_sn) {
                    auto predecessor = last_write_sn - batch_start_sn;  // SN relative to start of batch
                    task.preds.push_back(predecessor);
                }
            }
        }
    }

    if (mode == access_mode::read_write || mode == access_mode::write_only) {
        // we're writing to this resource: set last write SN
        // XXX if somehow there's an exception in the handler callback,
        // the resource's last_write_sequence_number will have been modified to an invalid task number.
        resource->last_write_sequence_number = task.signal.sequence_number;
    }

    // presentation doesn't support timelines (yet?), so handle synchronization for swapchain images separately
    // handle resources that need binary semaphore synchronization
    if (resource->type() == resource_type::swapchain_image) {
        swapchain_image_impl& swapchain_img = static_cast<swapchain_image_impl&>(*resource);
        if (auto sem = swapchain_img.consume_semaphore(nullptr)) {
            // syncing access on presentation engine
            task.wait_binary.push_back(sem);
        }
        else {
            if (task.type == task_type::present) {
                // rendering/presentation synchronization
                if (last_write_task) {
                    // this should not happen
                    assert(false);
                }
                auto prod_task = current_batch_.get_task(last_write_sn);
                vk::Semaphore semaphore;
                // TODO create semaphore
                prod_task->signal_binary.push_back(semaphore);
                task.wait_binary.push_back(semaphore);
            }
        }
    }

    // register the temporary on this batch
    const auto tmp_index = current_batch_.add_temporary(std::move(resource));

    switch (mode) {
        case access_mode::read_only: task.add_read(tmp_index); break;
        case access_mode::write_only: task.add_write(tmp_index); break;
        case access_mode::read_write:
            task.add_read(tmp_index);
            task.add_write(tmp_index);
            break;
    }
}

}  // namespace detail

void handler::add_buffer_access(
        std::shared_ptr<detail::buffer_impl> buffer, access_mode mode, buffer_usage usage) {
    add_resource_access(buffer, mode);
}

void handler::add_image_access(
        std::shared_ptr<detail::image_impl> image, access_mode mode, image_usage usage) {
    add_resource_access(image, mode);
    // TODO do something with usage? (layout transitions)
}

void handler::add_resource_access(
        std::shared_ptr<detail::virtual_resource> resource, access_mode mode) {
    accesses_.push_back(resource_access{.resource = std::move(resource), .mode = mode});
}

//-----------------------------------------------------------------------------
queue::queue(device& device, const queue_properties& props) :
    impl_{std::make_unique<detail::queue_impl>(device.impl_, props)} {
}

void queue::enqueue_pending_tasks() {
    impl_->enqueue_pending_tasks();
}

void queue::present(swapchain_image&& image) {
    impl_->present(std::move(image));
}

handler queue::begin_build_task(std::string_view name) const noexcept {
    // auto sequence_number = impl_->begin_build_task();
    // NOTE begin_build_task is not really necessary right now since the task object
    // is not created before the handler callback is executed,
    // but keep it anyway for future-proofing (we might want to know e.g. the task sequence number in the handler)
    return impl_->begin_build_task(name);
}

/// @brief Called to finish building a task. Adds the task to the current batch.
void queue::end_build_task(handler&& handler) {
    impl_->end_build_task(std::move(handler));
}

/*
size_t queue::next_task_index() const {
    return impl_->next_task_index();
}
size_t queue::current_batch_index() const {
    return impl_->current_batch_index();
}*/

}  // namespace graal