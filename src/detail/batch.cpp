#include <graal/detail/batch.hpp>
#include <graal/detail/swapchain_impl.hpp>


namespace graal::detail {
namespace {


}  // namespace


//-------------------------------------------------------------------------
batch::batch(device_impl_ptr device, serial_number start_sn) : device_{ std::move(device) }, start_sn{ start_sn }
{}


void batch::add_task(std::unique_ptr<task> task) noexcept {
    tasks.push_back(std::move(task));
}



void batch::submit(vk::Semaphore timeline_semaphore, recycled_batch_resources& recycled_resources)
{
  
}
}  // namespace graal::detail