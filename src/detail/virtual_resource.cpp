#include <graal/detail/virtual_resource.hpp>

namespace graal::detail {

	bool virtual_image_resource::is_aliasable_with(const virtual_resource& other) {

		if (typeid(other) != typeid(virtual_image_resource)) {
			// can't share memory with something else than images
			return false;
		}

		const auto& img = static_cast<const virtual_image_resource&>(other);

		// for now, only alias memory between textures that have the exact same 
		//const bool aliasable = format_

	}
}