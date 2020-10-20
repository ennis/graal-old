# CMakeList.txt : CMake project for flair3, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("graal")

find_package(fmt REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

set(HEADERS 
	"include/graal/queue.hpp" 
	"include/graal/image.hpp" 
	"include/graal/accessor.hpp" 
	"include/graal/buffer.hpp" 
	"include/graal/gl/handle.hpp" 
	"include/graal/gl/glad.h" 
	"include/graal/gl/khrplatform.h" 
	"include/graal/gl/handle.hpp" 
	"include/graal/gl/texture.hpp" 
	"include/graal/gl/types.hpp" 
	"include/graal/gl/format_info.hpp" 
	"include/graal/image_format.hpp" 
	"include/graal/errors.hpp"
 "include/graal/visibility.hpp" 
 "include/graal/range.hpp" 
 "include/graal/detail/dependent_false.hpp" 
 "include/graal/detail/image_impl.hpp" 
 "include/graal/access_mode.hpp" 
 "include/graal/target.hpp"
 "include/graal/detail/named_object.hpp"
 "include/graal/image_type.hpp" "include/graal/detail/image_resource.hpp" "include/graal/detail/virtual_resource.hpp" "include/graal/gl/buffer.hpp" "include/graal/event.hpp" "include/graal/detail/task.hpp")

set(SOURCES
	"src/queue.cpp" 
	"src/buffer.cpp" 
	"src/image.cpp" 
	"src/image_format.cpp" 
	"src/gl/format_info.cpp"
	"src/gl/texture.cpp"
	"src/gl/glad.c"
	"src/image_format.cpp"  
	"src/errors.cpp"
	 "src/gl/debug_output.cpp" "src/detail/image_resource.cpp" "src/detail/virtual_resource.cpp" "src/gl/buffer.cpp")

# Add source to this project's executable.
add_library (graal STATIC ${SOURCES} ${HEADERS})

target_include_directories(graal PRIVATE "src")
target_include_directories(graal PUBLIC "include")
target_compile_features(graal PUBLIC cxx_std_17)
target_compile_definitions(graal PRIVATE GRAAL_TRACE_RESOURCES)
#target_compile_options(graal PUBLIC /permissive- /std:c++latest)
target_link_libraries(graal 
	PUBLIC 
		fmt::fmt
		OpenGL::GL
	)


add_executable(graal-test 
	"tests/imgui_impl_opengl3.cpp"
	"tests/imgui_impl_glfw.cpp"
	"tests/main.cpp"
)

target_link_libraries(graal-test PRIVATE 
	graal
	imgui::imgui
	tinyobjloader::tinyobjloader
	glfw)

