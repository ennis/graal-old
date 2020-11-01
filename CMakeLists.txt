# CMakeList.txt : CMake project for flair3, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("graal")

include(FindVulkan)

find_package(fmt REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(unofficial-vulkan-memory-allocator CONFIG REQUIRED)

set(HEADERS 
	"include/graal/detail/gl_handle.hpp" 
 "include/graal/detail/dependent_false.hpp" 
 "include/graal/detail/image_impl.hpp" 
 "include/graal/detail/named_object.hpp"
 "include/graal/detail/image_resource.hpp"
 "include/graal/detail/resource.hpp" 
 "include/graal/detail/task.hpp"
 "include/graal/queue.hpp" 
 "include/graal/image.hpp" 
	"include/graal/accessor.hpp" 
	"include/graal/glad.h" 
	"include/graal/khrplatform.h" 
	"include/graal/texture.hpp" 
	"include/graal/gl_types.hpp" 
	"include/graal/gl_format_info.hpp" 
	"include/graal/image_format.hpp" 
	"include/graal/errors.hpp"
 "include/graal/visibility.hpp" 
 "include/graal/range.hpp" 
 "include/graal/access_mode.hpp" 
 "include/graal/image_usage.hpp"
 "include/graal/image_type.hpp" 
 "include/graal/raw_buffer.hpp" 
 "include/graal/event.hpp" 
 "include/graal/program.hpp" 
 "include/graal/shader.hpp" 
 "include/graal/allow_flags.hpp" 
 "include/graal/flags.hpp" 
 "include/graal/flagsfwd.hpp"
 "include/graal/flags_iterator.hpp"
 "include/graal/vertex_array.hpp" 
 "include/graal/data_type.hpp" 
 "include/graal/sampler.hpp"
 # extensions
 "include/graal/ext/scene.hpp"
 "include/graal/ext/vertex_array_cache.hpp" 
 "include/graal/ext/vertex_traits.hpp" "include/graal/ext/geometry.hpp" "include/graal/device.hpp" "include/graal/detail/vk_image.hpp")

set(SOURCES
	"src/queue.cpp" 
	"src/buffer.cpp" 
	"src/image.cpp" 
	"src/image_format.cpp" 
	"src/gl_format_info.cpp"
	"src/texture.cpp"
	"src/glad.c"
	"src/image_format.cpp"  
	"src/errors.cpp"
	 "src/debug_output.cpp"
	 "src/shader.cpp"
	 "src/detail/image_resource.cpp"
	 "src/detail/resource.cpp" 
	 "src/raw_buffer.cpp" 
	 "src/program.cpp" "src/vertex_array.cpp" "src/ext/scene.cpp" "src/device.cpp")

# Add source to this project's executable.
add_library (graal STATIC ${SOURCES} ${HEADERS})

target_include_directories(graal PRIVATE "src")
target_include_directories(graal PUBLIC "include")
target_compile_features(graal PUBLIC cxx_std_20)
target_compile_definitions(graal PRIVATE 
		GRAAL_TRACE_RESOURCES 
		GRAAL_TRACE_BATCH_SUBMIT
	PUBLIC
		VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
	)
	
target_compile_options(graal PUBLIC /permissive- /std:c++latest)
target_link_libraries(graal 
	PUBLIC 
		Vulkan::Vulkan
		fmt::fmt
		OpenGL::GL
		# Extension-only
		glm
		tinyobjloader::tinyobjloader
	PRIVATE
		unofficial::vulkan-memory-allocator::vulkan-memory-allocator
	)


add_executable(graal-test 
	"tests/imgui_impl_opengl3.cpp"
	"tests/imgui_impl_glfw.cpp"
	"tests/main.cpp"  )

target_link_libraries(graal-test PRIVATE 
	graal
	Vulkan::Vulkan
	imgui::imgui
	glfw)

