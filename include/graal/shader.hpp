#pragma once
#include <graal/device.hpp>

#include <exception>
#include <ostream>
#include <string_view>
#include <vulkan/vulkan.hpp>

namespace graal {

/// @brief
enum class shader_stage {
  vertex = VK_SHADER_STAGE_VERTEX_BIT,
  fragment = VK_SHADER_STAGE_FRAGMENT_BIT,
  tess_control = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
  tess_evaluation = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
  geometry = VK_SHADER_STAGE_GEOMETRY_BIT,
  compute = VK_SHADER_STAGE_COMPUTE_BIT
};

inline constexpr vk::ShaderStageFlagBits shader_stage_to_vk_shader_stage_flag_bits(shader_stage stage) noexcept {
  return static_cast<vk::ShaderStageFlagBits>(stage);
}

/// @brief Thrown when a shader fails to compile.
class shader_compilation_error : public std::exception {
public:
  shader_compilation_error(shader_stage stage, std::string log)
      : stage_{stage}, log_{std::move(log)} {}

  /// @brief Compilation log
  std::string_view log() const noexcept { return log_; }

  /// @brief The stage of the shader that was being compiled.
  shader_stage stage() const noexcept { return stage_; }

private:
  shader_stage stage_;
  std::string  log_;
};

/// @brief Compiles GLSL source code into a VkShaderModule
/// @param stage
/// @param source
/// @param out_log
/// @return
[[nodiscard]] vk::ShaderModule compile_shader(device& device, shader_stage     stage,
                                           std::string_view source,
                                           std::ostream &   out_log);

} // namespace graal