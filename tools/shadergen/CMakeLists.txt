add_executable(graal-shadergen "main.cpp")

target_link_libraries(graal-shadergen PRIVATE graal Vulkan::Vulkan glslang Boost::program_options
                                              SPIRV)
