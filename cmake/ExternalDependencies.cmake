# This file contains the declaration of external dependencies.
# See README.md for more information.
# 
# Some dependencies must be installed beforehand and be visible to find_package.
# On Linux, if the library supports cmake, a simple installation through the package manager should be enough to make it visible to cmake.
# On Windows, it is recommended to use vcpkg, and use the CMAKE_TOOLCHAIN_FILE provided by vcpkg
# Special case: for Qt on windows, it is recommended to use the official installer and communicate the installation path to cmake through the Qt5_DIR variable.
#
# Other dependencies are fetched via cmake's FetchContent. 
# For those, you don't need to do anything as CMake will automatically download the source and declare the necessary targets.



#==============================================================================
# external dependencies

# --- OpenGL ------------------------------------
find_package(OpenGL REQUIRED)

# --- OpenImageIO ------------------------------------
find_package(OpenImageIO REQUIRED)
target_compile_definitions(OpenImageIO::OpenImageIO INTERFACE -DOIIO_USE_FMT=0)

# --- Boost ------------------------------------
find_package(Boost REQUIRED COMPONENTS program_options)



#==============================================================================
# FetchContent dependencies


# --- glslang ------------------------------------
FetchContent_Declare(glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG 10-11.0.0 
    )
FetchContent_GetProperties(glslang)
if(NOT glslang_POPULATED)
  set(BUILD_TESTING OFF CACHE INTERNAL "")
  set(ENABLE_GLSLANG_BINARIES OFF CACHE INTERNAL "")
  set(ENABLE_SPVREMAPPER OFF CACHE INTERNAL "")
  set(ENABLE_HLSL OFF CACHE INTERNAL "")
  set(ENABLE_AMD_EXTENSIONS OFF CACHE INTERNAL "")
  set(ENABLE_NV_EXTENSIONS OFF CACHE INTERNAL "")
  set(SKIP_GLSLANG_INSTALL ON CACHE INTERNAL "")
  set(ENABLE_OPT ON CACHE INTERNAL "")
  FetchContent_Populate(glslang)
  add_subdirectory(${glslang_SOURCE_DIR} ${glslang_BINARY_DIR})
endif()
set_target_properties(glslang PROPERTIES FOLDER External/glslang)
set_target_properties(OGLCompiler PROPERTIES FOLDER External/glslang)
set_target_properties(OSDependent PROPERTIES FOLDER External/glslang)
set_target_properties(SPIRV PROPERTIES FOLDER External/glslang)



# --- VMA ------------------------------------
FetchContent_Declare(
  # the name should preferably be lowercase because FetchContent_Populate() hilariously converts it to lowercase 
  # for the _SOURCE_DIR and _BINARY_DIR variables.
  # See http://cmake.3232098.n2.nabble.com/How-to-link-against-projects-added-through-FetchContent-tp7597224p7597227.html
  vulkan_memory_allocator  
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG 8cd86b6dd47220f481cd81dbfc79e0aaffac899f
)
FetchContent_GetProperties(vulkan_memory_allocator)
if(NOT vulkan_memory_allocator)
  FetchContent_Populate(vulkan_memory_allocator)
  add_library(vulkan_memory_allocator INTERFACE IMPORTED)
  target_include_directories(vulkan_memory_allocator INTERFACE "${vulkan_memory_allocator_SOURCE_DIR}/src")
endif()



# --- imgui ------------------------------------
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui
  GIT_TAG        v1.79
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)
  add_library(imgui STATIC "${imgui_SOURCE_DIR}/imgui.cpp"
                           "${imgui_SOURCE_DIR}/imgui_draw.cpp"
                           "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
                           "${imgui_SOURCE_DIR}/imgui_demo.cpp")
  target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}")
endif()
set_target_properties(imgui PROPERTIES FOLDER External)



# --- spdlog ------------------------------------
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.7.0
)
FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  # use our version of fmt
  #[CRAP] CMake madness to set a subproject option
  set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "")
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()
set_target_properties(spdlog PROPERTIES FOLDER External)



# --- Boost.JSON ------------------------------------
FetchContent_Declare(
  boost_json
  GIT_REPOSITORY https://github.com/boostorg/json.git
  GIT_TAG        ee8d72d8502b409b5561200299cad30ccdb91415
)
FetchContent_GetProperties(boost_json)
if(NOT boost_json_POPULATED)
  FetchContent_Populate(boost_json)
  set(BOOST_JSON_STANDALONE ON CACHE INTERNAL "")
  set(BOOST_JSON_BUILD_TESTS OFF CACHE INTERNAL "")
  set(BOOST_JSON_BUILD_FUZZERS OFF CACHE INTERNAL "")
  set(BOOST_JSON_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(BOOST_JSON_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
  add_subdirectory(${boost_json_SOURCE_DIR} ${boost_json_BINARY_DIR})
endif()
set_target_properties(boost_json PROPERTIES FOLDER External)



# --- glm ------------------------------------
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        0.9.9.8 
)
FetchContent_MakeAvailable(glm)



# --- fmt ------------------------------------
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        7.0.1
)
FetchContent_MakeAvailable(fmt)
set_target_properties(fmt PROPERTIES FOLDER External)




# --- GLFW3 (used for tests) ------------------------------------
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.2
)
FetchContent_GetProperties(glfw)
if(NOT glfw_POPULATED)
  FetchContent_Populate(glfw)
  set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
  set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
  set(GLFW_INSTALL OFF CACHE INTERNAL "")
  set(GLFW_VULKAN_STATIC OFF CACHE INTERNAL "")
  add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
endif()
set_target_properties(glfw PROPERTIES FOLDER External)



# --- tinyobjloader ------------------------------------
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG        v2.0.0rc7
)
FetchContent_MakeAvailable(tinyobjloader)
set_target_properties(tinyobjloader PROPERTIES FOLDER External)
