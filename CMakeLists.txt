# CMakeList.txt : CMake project for flair3, include source and define project
# specific logic here.
cmake_minimum_required(VERSION 3.8)

project("graal")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FindVulkan)
include(FetchContent)

# ==========================================================
# Dependencies
include(ExternalDependencies)

set(HEADERS
    
    "include/graal/detail/dependent_false.hpp"
    "include/graal/detail/named_object.hpp"
    "include/graal/detail/resource.hpp"
    "include/graal/detail/task.hpp"
    "include/graal/queue.hpp"
    "include/graal/image.hpp"
    "include/graal/accessor.hpp"
    "include/graal/image_format.hpp"
    "include/graal/errors.hpp"
    "include/graal/visibility.hpp"
    "include/graal/range.hpp"
    "include/graal/access_mode.hpp"
    "include/graal/image_usage.hpp"
    "include/graal/image_type.hpp"
    "include/graal/event.hpp"
    "include/graal/shader.hpp"
    "include/graal/bitmask.hpp"
    # extensions
    "include/graal/ext/scene.hpp"
    "include/graal/ext/vertex_traits.hpp"
    "include/graal/ext/geometry.hpp"
    "include/graal/device.hpp"
    "include/graal/graphics_pipeline.hpp"
    "include/graal/instance.hpp"
    "include/graal/swapchain.hpp"
    "include/graal/detail/recycler.hpp"
    "include/graal/detail/swapchain_impl.hpp"
    "include/graal/detail/sequence_number.hpp"
    "include/graal/detail/command_buffer_pool.hpp" 
    "include/graal/queue_class.hpp" 
    "include/graal/render_pass.hpp" "include/graal/detail/staging_pool.hpp" "include/graal/detail/pipeline_stage_tracker.hpp")

set(SOURCES
    "src/queue.cpp"
    "src/buffer.cpp"
    "src/image.cpp"
    "src/image_format.cpp"
    "src/errors.cpp"
    "src/instance.cpp"
    "src/shader.cpp"
    "src/detail/staging_pool.cpp"
    "src/detail/resource.cpp"
    "src/detail/task.cpp"
    "src/device.cpp" 
    "src/swapchain.cpp" 
    "src/detail/swapchain_impl.cpp")

# Add source to this project's executable.
add_library(graal STATIC ${SOURCES} ${HEADERS})

target_include_directories(graal PRIVATE "src")
target_include_directories(graal PUBLIC "include")
target_compile_features(graal PUBLIC cxx_std_20)
target_compile_definitions(
  graal
  PRIVATE GRAAL_TRACE_RESOURCES GRAAL_TRACE_BATCH_SUBMIT
  PUBLIC VULKAN_HPP_NO_STRUCT_CONSTRUCTORS NOMINMAX)

if(WIN32)
    target_compile_definitions(graal PUBLIC VK_USE_PLATFORM_WIN32_KHR)
else()
    # TODO
endif()

target_compile_options(graal PUBLIC /permissive- /std:c++latest)
target_link_libraries(
  graal
  PUBLIC Vulkan::Vulkan
         fmt
         OpenGL::GL
         Boost::headers
         # Extension-only
         glm
         tinyobjloader
         vulkan_memory_allocator
  PRIVATE glslang SPIRV)

function(compile_shaders OUTPUT_SPV)
  foreach(SHADER ${ARGN})
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER})
    add_custom_command(
      OUTPUT ${SHADER}.spv
      COMMAND glslc ${SHADER} -g -o ${SHADER_OUTPUT}.spv
      DEPENDS ${SHADER}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Compiling shader ${SHADER_OUTPUT}.spv")
    list(APPEND ${OUTPUT_SPV} ${SHADER}.spv)
  endforeach()
  set(${OUTPUT_SPV}
      ${${OUTPUT_SPV}}
      PARENT_SCOPE)
endfunction()

set(SHADERS data/shaders/background.vert data/shaders/background.frag)

compile_shaders(COMPILED_SHADERS ${SHADERS})
message(STATUS "Compiled shaders: ${COMPILED_SHADERS}")

add_executable(
  graal-test  "tests/imgui_impl_glfw.cpp" "tests/imgui_impl_vulkan.cpp"
             "tests/main.cpp" ${COMPILED_SHADERS} "tests/load_image.cpp" "tests/load_image.hpp")

target_link_libraries(graal-test PRIVATE graal Vulkan::Vulkan imgui glfw OpenImageIO::OpenImageIO)
set_target_properties(
    graal-test PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(tools/shadergen)
