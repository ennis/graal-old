#pragma once
#include <graal/detail/sequence_number.hpp>
#include <graal/bitmask.hpp>

#include <vulkan/vulkan.hpp>

namespace graal::detail {

/// @brief Represents the known execution states of all pipeline stages
/// (i.e. for each stage, the last finished sequence number before a barrier).
struct pipeline_stage_tracker {
    enum stage_index : size_t {
        i_DI,
        i_VI,
        i_VS,
        i_TCS,
        i_TES,
        i_GS,
        i_TF,
        i_TS,
        i_MS,
        i_FSR,
        i_EFT,
        i_FS,
        i_LFT,
        i_CAO,
        i_FDP,
        i_CS,
        i_RTS,
        i_HST,
        i_CPR,
        i_ASB,
        i_TR,
        i_CR,
        max
    };

    static constexpr auto DI = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    static constexpr auto VI = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    static constexpr auto VS = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    static constexpr auto TCS = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
    static constexpr auto TES = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    static constexpr auto GS = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    static constexpr auto TF = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    static constexpr auto TS = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV;
    static constexpr auto MS = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV;
    static constexpr auto FSR = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV;
    static constexpr auto EFT = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    static constexpr auto FS = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    static constexpr auto LFT = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    static constexpr auto CAO = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    static constexpr auto FDP = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT;
    static constexpr auto CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    static constexpr auto RTS = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    static constexpr auto HST = VK_PIPELINE_STAGE_HOST_BIT;
    static constexpr auto CPR = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV;
    static constexpr auto ASB = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    static constexpr auto TR = VK_PIPELINE_STAGE_TRANSFER_BIT;
    static constexpr auto CR = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
    static constexpr auto ALL_COMMANDS = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    static constexpr auto ALL_GRAPHICS = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    static constexpr auto BOTTOM_OF_PIPE = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

    serial_number stages[stage_index::max] = {0};

    /// @brief Returns whether an barrier is needed for the specified execution dependency (sn+pipeline stages)
    /// @param sn
    /// @param source_flags
    /// @return
    bool needs_execution_barrier(
            serial_number sn, vk::PipelineStageFlags source_flags) const noexcept {
        const VkPipelineStageFlags src = (VkPipelineStageFlags) source_flags;
        bool needs_barrier = false;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | DI))
            needs_barrier |= stages[i_DI] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | VI))
            needs_barrier |= stages[i_VI] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | VS))
            needs_barrier |= stages[i_VS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TCS))
            needs_barrier |= stages[i_TCS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TES))
            needs_barrier |= stages[i_TES] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | GS))
            needs_barrier |= stages[i_GS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TF))
            needs_barrier |= stages[i_TF] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TS))
            needs_barrier |= stages[i_TS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | MS))
            needs_barrier |= stages[i_MS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | FSR))
            needs_barrier |= stages[i_FSR] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | EFT))
            needs_barrier |= stages[i_EFT] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | FS))
            needs_barrier |= stages[i_FS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | LFT))
            needs_barrier |= stages[i_LFT] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | CAO))
            needs_barrier |= stages[i_CAO] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | FDP))
            needs_barrier |= stages[i_FDP] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | RTS)) needs_barrier |= stages[i_RTS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | CS)) needs_barrier |= stages[i_CS] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ASB)) needs_barrier |= stages[i_ASB] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | TR)) needs_barrier |= stages[i_TR] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | CR)) needs_barrier |= stages[i_CR] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | HST)) needs_barrier |= stages[i_HST] < sn;
        if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | CPR)) needs_barrier |= stages[i_CPR] < sn;
        return needs_barrier;
    }

    /// @brief Applies an execution barrier on the specified source pipeline stages and updates the tracking information.
    /// @param sn
    /// @param source_flags
    void apply_execution_barrier(serial_number sn, vk::PipelineStageFlags source_flags) {
        const VkPipelineStageFlags src = (VkPipelineStageFlags) source_flags;
        /* DI  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | DI | VI | VS | TCS | TES
                              | GS | TF | TS | MS | FSR | EFT | FS | LFT | CAO | CS | RTS)) {
            if (stages[i_DI] < sn) { stages[i_DI] = sn; }
        }
        /* VI  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | VI | VS | TCS | TES | GS
                              | TF | FSR | EFT | FS | LFT | CAO)) {
            if (stages[i_VI] < sn) { stages[i_VI] = sn; }
        }
        /* VS  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | VS | TCS | TES | GS | TF
                              | FSR | EFT | FS | LFT | CAO)) {
            if (stages[i_VS] < sn) { stages[i_VS] = sn; }
        }
        /* TCS */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TCS | TES | GS | TF | FSR
                              | EFT | FS | LFT | CAO)) {
            if (stages[i_TCS] < sn) { stages[i_TCS] = sn; }
        }
        /* TES */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TES | GS | TF | FSR | EFT
                              | FS | LFT | CAO)) {
            if (stages[i_TES] < sn) { stages[i_TES] = sn; }
        }
        /* GS  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | GS | TF | FSR | EFT | FS
                              | LFT | CAO)) {
            if (stages[i_GS] < sn) { stages[i_GS] = sn; }
        }
        /* TF  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TF | FSR | EFT | FS | LFT
                              | CAO)) {
            if (stages[i_TF] < sn) { stages[i_TF] = sn; }
        }
        /* TS  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | TS | MS | FSR | EFT | FS
                              | LFT | CAO)) {
            if (stages[i_TS] < sn) { stages[i_TS] = sn; }
        }
        /* MS  */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | MS | FSR | EFT | FS | LFT
                              | CAO)) {
            if (stages[i_MS] < sn) { stages[i_MS] = sn; }
        }
        /* FSR */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | FSR | EFT | FS | LFT
                              | CAO)) {
            if (stages[i_FSR] < sn) { stages[i_FSR] = sn; }
        }
        /* EFT */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | EFT | FS | LFT | CAO)) {
            if (stages[i_EFT] < sn) { stages[i_EFT] = sn; }
        }
        /* FS  */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | FS | LFT | CAO)) {
            if (stages[i_FS] < sn) { stages[i_FS] = sn; }
        }
        /* LFT */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | LFT | CAO)) {
            if (stages[i_LFT] < sn) { stages[i_LFT] = sn; }
        }
        /* CAO */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | CAO)) {
            if (stages[i_CAO] < sn) { stages[i_CAO] = sn; }
        }
        /* FDP */ if (src
                      & (BOTTOM_OF_PIPE | ALL_COMMANDS | ALL_GRAPHICS | FDP | EFT | FS | LFT
                              | CAO)) {
            if (stages[i_FDP] < sn) { stages[i_FDP] = sn; }
        }
        /* CS  */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | CS)) {
            if (stages[i_CS] < sn) { stages[i_CS] = sn; }
        }
        /* RTS */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | RTS)) {
            if (stages[i_RTS] < sn) { stages[i_RTS] = sn; }
        }
        /* HST */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | HST)) {
            if (stages[i_HST] < sn) { stages[i_HST] = sn; }
        }
        /* CPR */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | CPR)) {
            if (stages[i_CPR] < sn) { stages[i_CPR] = sn; }
        }
        /* ASB */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | ASB)) {
            if (stages[i_ASB] < sn) { stages[i_ASB] = sn; }
        }
        /* TR  */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | TR)) {
            if (stages[i_TR] < sn) { stages[i_TR] = sn; }
        }
        /* CR  */ if (src & (BOTTOM_OF_PIPE | ALL_COMMANDS | CR)) {
            if (stages[i_CR] < sn) { stages[i_CR] = sn; }
        }
    }
};

/*enum class resource_state {
    indirect_command_read = (1<<0),
    vertex_buffer = (1 << 1),
    index_buffer = (1 << 2),
    color_attachment = (1 << 3),
    depth_attachment = (1 << 4),
    transfer_src = (1 << 5),
    transfer_dst = (1 << 6),
    shader_read = (1 << 7),
    shader_write = (1 << 8),
};

GRAAL_BITMASK(resource_state)*/

/*constexpr inline void get_resource_state_barriers(
    resource_state state,
    vk::AccessFlags access_mask,
    vk::PipelineStageFlags input_stage_mask,
    vk::PipelineStageFlags output_stage_mask)
{
    if (bitmask_includes(state, resource_state::vertex_buffer)) {
        access_mask |= vk::AccessFlagBits::eVertexAttributeRead;
        input_stage_mask |= vk::PipelineStageFlagBits::eVertexInput;
        output_stage_mask |= vk::PipelineStageFlagBits::eVertexInput;
    }

    if (bitmask_includes(state, resource_state::index_buffer)) {
        access_mask |= vk::AccessFlagBits::eIndexRead;
        input_stage_mask |= vk::PipelineStageFlagBits::eVertexInput;
        output_stage_mask |= vk::PipelineStageFlagBits::eVertexInput;
    }

    if (bitmask_includes(state, resource_state::color_attachment)) {
        access_mask |= vk::AccessFlagBits::eColorAttachmentWrite;
    }
}*/

/*
struct region_state {
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    std::array<serial_number, max_queues> access_sn;
    submission_number write_snn;
    vk::AccessFlags access_flags{};
    vk::PipelineStageFlags stages{};

    struct hash {
        // TODO
    };
};

class region_state_map {
public:
private:
    std::unordered_map<region, region_state> map;
};*/

}  // namespace graal::detail