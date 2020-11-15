#include <boost/program_options.hpp>

#include <fmt/ostream.h>
#include <fstream>
#include <ostream>
#include <iostream>
#include <vector>
#include <string_view>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Include/intermediate.h>

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
    } };


static void initialize_glslang() {
    static bool initialized = glslang::InitializeProcess();
    if (!initialized) {
        throw std::runtime_error{ "could not initialize glslang" };
    }
}

static std::string read_text_file(std::string_view path) {
    std::ifstream file_in{ std::string{path} };
    std::string   source;

    if (!file_in) {
        throw std::runtime_error{
            fmt::format("could not open file `{}` for reading") };
    }

    file_in.seekg(0, std::ios::end);
    source.reserve(file_in.tellg());
    file_in.seekg(0, std::ios::beg);

    source.assign(std::istreambuf_iterator<char>{file_in},
        std::istreambuf_iterator<char>{});
    return source;
}

static void parse_shader_interface_file(
    std::string_view source,
    std::ostream& out_log) 
{
    const char* const sources[] = { source.data() };
    int const         sourceLengths[] = { (int)source.size() };
    const auto        sh_lang = EShLangVertex;  // shouldn't matter
    glslang::TShader  shader{ sh_lang };
    shader.setStringsWithLengths(sources, sourceLengths, 1);
    shader.setEnvClient(glslang::EShClientVulkan,
        glslang::EShTargetVulkan_1_1); // TODO customize?
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_1);
    shader.parse(&default_builtin_resources, 450, 
        ECoreProfile,
        true, true, EShMsgDefault);

    // extract linker items
    auto interm = shader.getIntermediate();
    

}


int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description desc {"allowed options"};
	desc.add_options()("help", "print this help message")("o", po::value<std::string>(), "output header");

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(
		po::command_line_parser(argc, argv).
		options(desc).positional(p).run(), vm);
	po::notify(vm);

    if (!vm.count("input-file"))
    {
        fmt::print("graal-shadergen: no input file specified.\n");
        std::cerr << desc << "\n";
        desc.print(std::cerr);
        return 1;
    }

    auto input_files = vm["input-file"].as<std::vector<std::string>>();

    for (auto input : input_files) {

    }
	

    // 1. generate minimal valid shader by appending `void main() {}`
    // 2. generate spir-v
    // 3. generate JSON reflection DB
    // 4. apply JSON to C++ template
    //

}
