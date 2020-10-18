#include <graal/detail/virtual_resource.hpp>

namespace graal::detail {

bool virtual_image_resource::is_aliasable_with(
    const virtual_resource &other) const {

  if (typeid(other) != typeid(virtual_image_resource)) {
    // can't share memory with something else than images
    return false;
  }

  const auto &img = static_cast<const virtual_image_resource &>(other);

  // for now, only alias memory between textures that have the exact same
  // const bool aliasable = format_

  // TODO
  return false;
}

void virtual_image_resource::allocate() {
  image_ = std::make_shared<image_resource>(desc_);
#ifdef GRAAL_TRACE_RESOURCES

#endif
}

void virtual_image_resource::alias_with(virtual_resource &other) {
  // precondition: is_aliasable_with(other) == true
  auto &other_img = static_cast<virtual_image_resource &>(other);
  image_ = other_img.image_;
}

bool virtual_buffer_resource::is_aliasable_with(
    const virtual_resource &other) const {
  // TODO
  return false;
}
} // namespace graal::detail