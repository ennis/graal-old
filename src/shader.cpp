#include <graal/errors.hpp>
#include <graal/shader.hpp>

#include <fmt/ostream.h>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>

#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

namespace graal {

namespace {

// taken from "glslang/StandAlone/DirStackFileIncluder.h"
// Default include class for normal include convention of search backward
// through the stack of active include paths (for nested includes).
// Can be overridden to customize.
class dir_stack_file_includer : public glslang::TShader::Includer {
public:
  dir_stack_file_includer() : externalLocalDirectoryCount(0) {}

  virtual IncludeResult *includeLocal(const char *headerName,
                                      const char *includerName,
                                      size_t      inclusionDepth) override {
    return readLocalPath(headerName, includerName, (int)inclusionDepth);
  }

  virtual IncludeResult *includeSystem(const char *headerName,
                                       const char * /*includerName*/,
                                       size_t /*inclusionDepth*/) override {
    return readSystemPath(headerName);
  }

  // Externally set directories. E.g., from a command-line -I<dir>.
  //  - Most-recently pushed are checked first.
  //  - All these are checked after the parse-time stack of local directories
  //    is checked.
  //  - This only applies to the "local" form of #include.
  //  - Makes its own copy of the path.
  virtual void pushExternalLocalDirectory(const std::string &dir) {
    directoryStack.push_back(dir);
    externalLocalDirectoryCount = (int)directoryStack.size();
  }

  virtual void releaseInclude(IncludeResult *result) override {
    if (result != nullptr) {
      delete[] static_cast<char *>(result->userData);
      delete result;
    }
  }

  virtual ~dir_stack_file_includer() override {}

protected:
  std::vector<std::string> directoryStack;
  int                      externalLocalDirectoryCount;

  // Search for a valid "local" path based on combining the stack of include
  // directories and the nominal name of the header.
  virtual IncludeResult *readLocalPath(const char *headerName,
                                       const char *includerName, int depth) {
    // Discard popped include directories, and
    // initialize when at parse-time first level.
    directoryStack.resize(depth + externalLocalDirectoryCount);
    if (depth == 1)
      directoryStack.back() = getDirectory(includerName);

    // Find a directory that works, using a reverse search of the include stack.
    for (auto it = directoryStack.rbegin(); it != directoryStack.rend(); ++it) {
      std::string path = *it + '/' + headerName;
      std::replace(path.begin(), path.end(), '\\', '/');
      std::ifstream file(path, std::ios_base::binary | std::ios_base::ate);
      if (file) {
        directoryStack.push_back(getDirectory(path));
        return newIncludeResult(path, file, (int)file.tellg());
      }
    }

    return nullptr;
  }

  // Search for a valid <system> path.
  // Not implemented yet; returning nullptr signals failure to find.
  virtual IncludeResult *readSystemPath(const char * /*headerName*/) const {
    return nullptr;
  }

  // Do actual reading of the file, filling in a new include result.
  virtual IncludeResult *newIncludeResult(const std::string &path,
                                          std::ifstream &    file,
                                          int                length) const {
    char *content = new char[length];
    file.seekg(0, file.beg);
    file.read(content, length);
    return new IncludeResult(path, content, length, content);
  }

  // If no path markers, return current working directory.
  // Otherwise, strip file name and return path leading up to it.
  virtual std::string getDirectory(const std::string path) const {
    size_t last = path.find_last_of("/\\");
    return last == std::string::npos ? "." : path.substr(0, last);
  }
};

constexpr TBuiltInResource default_builtin_resources = {
    /* .MaxLights = */ 32,
    /* .MaxClipPlanes = */ 6,
    /* .MaxTextureUnits = */ 32,
    /* .MaxTextureCoords = */ 32,
    /* .MaxVertexAttribs = */ 64,
    /* .MaxVertexUniformComponents = */ 4096,
    /* .MaxVaryingFloats = */ 64,
    /* .MaxVertexTextureImageUnits = */ 32,
    /* .MaxCombinedTextureImageUnits = */ 80,
    /* .MaxTextureImageUnits = */ 32,
    /* .MaxFragmentUniformComponents = */ 4096,
    /* .MaxDrawBuffers = */ 32,
    /* .MaxVertexUniformVectors = */ 128,
    /* .MaxVaryingVectors = */ 8,
    /* .MaxFragmentUniformVectors = */ 16,
    /* .MaxVertexOutputVectors = */ 16,
    /* .MaxFragmentInputVectors = */ 15,
    /* .MinProgramTexelOffset = */ -8,
    /* .MaxProgramTexelOffset = */ 7,
    /* .MaxClipDistances = */ 8,
    /* .MaxComputeWorkGroupCountX = */ 65535,
    /* .MaxComputeWorkGroupCountY = */ 65535,
    /* .MaxComputeWorkGroupCountZ = */ 65535,
    /* .MaxComputeWorkGroupSizeX = */ 1024,
    /* .MaxComputeWorkGroupSizeY = */ 1024,
    /* .MaxComputeWorkGroupSizeZ = */ 64,
    /* .MaxComputeUniformComponents = */ 1024,
    /* .MaxComputeTextureImageUnits = */ 16,
    /* .MaxComputeImageUniforms = */ 8,
    /* .MaxComputeAtomicCounters = */ 8,
    /* .MaxComputeAtomicCounterBuffers = */ 1,
    /* .MaxVaryingComponents = */ 60,
    /* .MaxVertexOutputComponents = */ 64,
    /* .MaxGeometryInputComponents = */ 64,
    /* .MaxGeometryOutputComponents = */ 128,
    /* .MaxFragmentInputComponents = */ 128,
    /* .MaxImageUnits = */ 8,
    /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .MaxCombinedShaderOutputResources = */ 8,
    /* .MaxImageSamples = */ 0,
    /* .MaxVertexImageUniforms = */ 0,
    /* .MaxTessControlImageUniforms = */ 0,
    /* .MaxTessEvaluationImageUniforms = */ 0,
    /* .MaxGeometryImageUniforms = */ 0,
    /* .MaxFragmentImageUniforms = */ 8,
    /* .MaxCombinedImageUniforms = */ 8,
    /* .MaxGeometryTextureImageUnits = */ 16,
    /* .MaxGeometryOutputVertices = */ 256,
    /* .MaxGeometryTotalOutputComponents = */ 1024,
    /* .MaxGeometryUniformComponents = */ 1024,
    /* .MaxGeometryVaryingComponents = */ 64,
    /* .MaxTessControlInputComponents = */ 128,
    /* .MaxTessControlOutputComponents = */ 128,
    /* .MaxTessControlTextureImageUnits = */ 16,
    /* .MaxTessControlUniformComponents = */ 1024,
    /* .MaxTessControlTotalOutputComponents = */ 4096,
    /* .MaxTessEvaluationInputComponents = */ 128,
    /* .MaxTessEvaluationOutputComponents = */ 128,
    /* .MaxTessEvaluationTextureImageUnits = */ 16,
    /* .MaxTessEvaluationUniformComponents = */ 1024,
    /* .MaxTessPatchComponents = */ 120,
    /* .MaxPatchVertices = */ 32,
    /* .MaxTessGenLevel = */ 64,
    /* .MaxViewports = */ 16,
    /* .MaxVertexAtomicCounters = */ 0,
    /* .MaxTessControlAtomicCounters = */ 0,
    /* .MaxTessEvaluationAtomicCounters = */ 0,
    /* .MaxGeometryAtomicCounters = */ 0,
    /* .MaxFragmentAtomicCounters = */ 8,
    /* .MaxCombinedAtomicCounters = */ 8,
    /* .MaxAtomicCounterBindings = */ 1,
    /* .MaxVertexAtomicCounterBuffers = */ 0,
    /* .MaxTessControlAtomicCounterBuffers = */ 0,
    /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .MaxGeometryAtomicCounterBuffers = */ 0,
    /* .MaxFragmentAtomicCounterBuffers = */ 1,
    /* .MaxCombinedAtomicCounterBuffers = */ 1,
    /* .MaxAtomicCounterBufferSize = */ 16384,
    /* .MaxTransformFeedbackBuffers = */ 4,
    /* .MaxTransformFeedbackInterleavedComponents = */ 64,
    /* .MaxCullDistances = */ 8,
    /* .MaxCombinedClipAndCullDistances = */ 8,
    /* .MaxSamples = */ 4,
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
    /* .maxDualSourceDrawBuffersEXT = */ 1,

    /* .limits = */
    {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
    }};

void initialize_glslang() {
  static bool initialized = glslang::InitializeProcess();
  if (!initialized) {
    throw std::runtime_error{"could not initialize glslang"};
  }
}

EShLanguage shader_stage_to_sh_language(shader_stage stage) {
  switch (stage) {
  case graal::shader_stage::vertex:
    return EShLangVertex;
  case graal::shader_stage::fragment:
    return EShLangFragment;
  case graal::shader_stage::tess_control:
    return EShLangTessControl;
  case graal::shader_stage::tess_evaluation:
    return EShLangTessEvaluation;
  case graal::shader_stage::geometry:
    return EShLangGeometry;
  case graal::shader_stage::compute:
    return EShLangCompute;
  default:
    throw std::logic_error{"unsupported shader stage"};
  }
}

} // namespace

[[nodiscard]] vk::ShaderModule compile_shader(device &         device,
                                              shader_stage     stage,
                                              std::string_view source,
                                              std::ostream &   out_log) {
  initialize_glslang();

  // --- Compile and link program ---
  const char *const sources[] = {source.data()};
  int const         sourceLengths[] = {(int)source.size()};
  const auto        sh_lang = shader_stage_to_sh_language(stage);
  glslang::TShader  shader{sh_lang};
  shader.setStringsWithLengths(sources, sourceLengths, 1);
  shader.setEnvClient(glslang::EShClientVulkan,
                      glslang::EShTargetVulkan_1_1); // TODO customize?
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_1);

  dir_stack_file_includer includer;
  // TODO
  // includer.pushExternalLocalDirectory("path/to/include");
  TBuiltInResource Resources;

  bool success = shader.parse(&default_builtin_resources, 450, ECoreProfile,
                              true, true, EShMsgDefault, includer);

  if (!success) {
    fmt::print(stderr, "failed to compiler shader:\n{}\n", shader.getInfoLog());
    throw shader_compilation_error{stage, "failed to compile shader"};
  }

  glslang::TProgram prog;
  prog.addShader(&shader);
  success = prog.link(EShMsgDefault);

  if (!success) {
    fmt::print(stderr, "failed to link shader:\n{}\n", prog.getInfoLog());
    throw shader_compilation_error{stage, "failed to link shader"};
  }

  // --- generate SPIR-V ---
  spv::SpvBuildLogger   spv_build_logger;
  std::vector<uint32_t> spv;
  glslang::SpvOptions   spv_options; // use default options
  glslang::GlslangToSpv(*prog.getIntermediate(sh_lang), spv, &spv_build_logger,
                        &spv_options);
  fmt::print("SPIR-V generator log:\n{}\n", spv_build_logger.getAllMessages());

  fmt::print("SPIR-V dump:\n");
  for (auto w : spv) {
    fmt::print("{:08X}\n", w);
  }

  // save to file
  std::ofstream spv_file_out{"out.spv", std::ios::binary};
  spv_file_out.write((const char *)spv.data(), spv.size() * sizeof(uint32_t));

  // --- generate reflection data ---

  // --- create shader module ---
  auto                       vk_device = device.get_vk_device();
  vk::ShaderModuleCreateInfo smci{.codeSize = spv.size() * sizeof(uint32_t),
                                  .pCode = spv.data()};
  auto                       m = vk_device.createShaderModule(smci);
  return m;
}

} // namespace graal