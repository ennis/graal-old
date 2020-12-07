#pragma once
#include <cassert>
#include <graal/errors.hpp>
#include <string_view>
#include <vulkan/vulkan.hpp>

namespace graal {

/// @brief
// These formats correspond to vulkan formats
enum class image_format {
    undefined = VK_FORMAT_UNDEFINED,
    r4g4_unorm_pack8 = VK_FORMAT_R4G4_UNORM_PACK8,
    r4g4b4a4_unorm_pack16 = VK_FORMAT_R4G4B4A4_UNORM_PACK16,
    b4g4r4a4_unorm_pack16 = VK_FORMAT_B4G4R4A4_UNORM_PACK16,
    r5g6b5_unorm_pack16 = VK_FORMAT_R5G6B5_UNORM_PACK16,
    b5g6r5_unorm_pack16 = VK_FORMAT_B5G6R5_UNORM_PACK16,
    r5g5b5a1_unorm_pack16 = VK_FORMAT_R5G5B5A1_UNORM_PACK16,
    b5g5r5a1_unorm_pack16 = VK_FORMAT_B5G5R5A1_UNORM_PACK16,
    a1r5g5b5_unorm_pack16 = VK_FORMAT_A1R5G5B5_UNORM_PACK16,
    r8_unorm = VK_FORMAT_R8_UNORM,
    r8_snorm = VK_FORMAT_R8_SNORM,
    r8_uscaled = VK_FORMAT_R8_USCALED,
    r8_sscaled = VK_FORMAT_R8_SSCALED,
    r8_uint = VK_FORMAT_R8_UINT,
    r8_sint = VK_FORMAT_R8_SINT,
    r8_srgb = VK_FORMAT_R8_SRGB,
    r8g8_unorm = VK_FORMAT_R8G8_UNORM,
    r8g8_snorm = VK_FORMAT_R8G8_SNORM,
    r8g8_uscaled = VK_FORMAT_R8G8_USCALED,
    r8g8_sscaled = VK_FORMAT_R8G8_SSCALED,
    r8g8_uint = VK_FORMAT_R8G8_UINT,
    r8g8_sint = VK_FORMAT_R8G8_SINT,
    r8g8_srgb = VK_FORMAT_R8G8_SRGB,
    r8g8b8_unorm = VK_FORMAT_R8G8B8_UNORM,
    r8g8b8_snorm = VK_FORMAT_R8G8B8_SNORM,
    r8g8b8_uscaled = VK_FORMAT_R8G8B8_USCALED,
    r8g8b8_sscaled = VK_FORMAT_R8G8B8_SSCALED,
    r8g8b8_uint = VK_FORMAT_R8G8B8_UINT,
    r8g8b8_sint = VK_FORMAT_R8G8B8_SINT,
    r8g8b8_srgb = VK_FORMAT_R8G8B8_SRGB,
    b8g8r8_unorm = VK_FORMAT_B8G8R8_UNORM,
    b8g8r8_snorm = VK_FORMAT_B8G8R8_SNORM,
    b8g8r8_uscaled = VK_FORMAT_B8G8R8_USCALED,
    b8g8r8_sscaled = VK_FORMAT_B8G8R8_SSCALED,
    b8g8r8_uint = VK_FORMAT_B8G8R8_UINT,
    b8g8r8_sint = VK_FORMAT_B8G8R8_SINT,
    b8g8r8_srgb = VK_FORMAT_B8G8R8_SRGB,
    r8g8b8a8_unorm = VK_FORMAT_R8G8B8A8_UNORM,
    r8g8b8a8_snorm = VK_FORMAT_R8G8B8A8_SNORM,
    r8g8b8a8_uscaled = VK_FORMAT_R8G8B8A8_USCALED,
    r8g8b8a8_sscaled = VK_FORMAT_R8G8B8A8_SSCALED,
    r8g8b8a8_uint = VK_FORMAT_R8G8B8A8_UINT,
    r8g8b8a8_sint = VK_FORMAT_R8G8B8A8_SINT,
    r8g8b8a8_srgb = VK_FORMAT_R8G8B8A8_SRGB,
    b8g8r8a8_unorm = VK_FORMAT_B8G8R8A8_UNORM,
    b8g8r8a8_snorm = VK_FORMAT_B8G8R8A8_SNORM,
    b8g8r8a8_uscaled = VK_FORMAT_B8G8R8A8_USCALED,
    b8g8r8a8_sscaled = VK_FORMAT_B8G8R8A8_SSCALED,
    b8g8r8a8_uint = VK_FORMAT_B8G8R8A8_UINT,
    b8g8r8a8_sint = VK_FORMAT_B8G8R8A8_SINT,
    b8g8r8a8_srgb = VK_FORMAT_B8G8R8A8_SRGB,
    a8b8g8r8_unorm_pack32 = VK_FORMAT_A8B8G8R8_UNORM_PACK32,
    a8b8g8r8_snorm_pack32 = VK_FORMAT_A8B8G8R8_SNORM_PACK32,
    a8b8g8r8_uscaled_pack32 = VK_FORMAT_A8B8G8R8_USCALED_PACK32,
    a8b8g8r8_sscaled_pack32 = VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
    a8b8g8r8_uint_pack32 = VK_FORMAT_A8B8G8R8_UINT_PACK32,
    a8b8g8r8_sint_pack32 = VK_FORMAT_A8B8G8R8_SINT_PACK32,
    a8b8g8r8_srgb_pack32 = VK_FORMAT_A8B8G8R8_SRGB_PACK32,
    a2r10g10b10_unorm_pack32 = VK_FORMAT_A2R10G10B10_UNORM_PACK32,
    a2r10g10b10_snorm_pack32 = VK_FORMAT_A2R10G10B10_SNORM_PACK32,
    a2r10g10b10_uscaled_pack32 = VK_FORMAT_A2R10G10B10_USCALED_PACK32,
    a2r10g10b10_sscaled_pack32 = VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
    a2r10g10b10_uint_pack32 = VK_FORMAT_A2R10G10B10_UINT_PACK32,
    a2r10g10b10_sint_pack32 = VK_FORMAT_A2R10G10B10_SINT_PACK32,
    a2b10g10r10_unorm_pack32 = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    a2b10g10r10_snorm_pack32 = VK_FORMAT_A2B10G10R10_SNORM_PACK32,
    a2b10g10r10_uscaled_pack32 = VK_FORMAT_A2B10G10R10_USCALED_PACK32,
    a2b10g10r10_sscaled_pack32 = VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
    a2b10g10r10_uint_pack32 = VK_FORMAT_A2B10G10R10_UINT_PACK32,
    a2b10g10r10_sint_pack32 = VK_FORMAT_A2B10G10R10_SINT_PACK32,
    r16_unorm = VK_FORMAT_R16_UNORM,
    r16_snorm = VK_FORMAT_R16_SNORM,
    r16_uscaled = VK_FORMAT_R16_USCALED,
    r16_sscaled = VK_FORMAT_R16_SSCALED,
    r16_uint = VK_FORMAT_R16_UINT,
    r16_sint = VK_FORMAT_R16_SINT,
    r16_sfloat = VK_FORMAT_R16_SFLOAT,
    r16g16_unorm = VK_FORMAT_R16G16_UNORM,
    r16g16_snorm = VK_FORMAT_R16G16_SNORM,
    r16g16_uscaled = VK_FORMAT_R16G16_USCALED,
    r16g16_sscaled = VK_FORMAT_R16G16_SSCALED,
    r16g16_uint = VK_FORMAT_R16G16_UINT,
    r16g16_sint = VK_FORMAT_R16G16_SINT,
    r16g16_sfloat = VK_FORMAT_R16G16_SFLOAT,
    r16g16b16_unorm = VK_FORMAT_R16G16B16_UNORM,
    r16g16b16_snorm = VK_FORMAT_R16G16B16_SNORM,
    r16g16b16_uscaled = VK_FORMAT_R16G16B16_USCALED,
    r16g16b16_sscaled = VK_FORMAT_R16G16B16_SSCALED,
    r16g16b16_uint = VK_FORMAT_R16G16B16_UINT,
    r16g16b16_sint = VK_FORMAT_R16G16B16_SINT,
    r16g16b16_sfloat = VK_FORMAT_R16G16B16_SFLOAT,
    r16g16b16a16_unorm = VK_FORMAT_R16G16B16A16_UNORM,
    r16g16b16a16_snorm = VK_FORMAT_R16G16B16A16_SNORM,
    r16g16b16a16_uscaled = VK_FORMAT_R16G16B16A16_USCALED,
    r16g16b16a16_sscaled = VK_FORMAT_R16G16B16A16_SSCALED,
    r16g16b16a16_uint = VK_FORMAT_R16G16B16A16_UINT,
    r16g16b16a16_sint = VK_FORMAT_R16G16B16A16_SINT,
    r16g16b16a16_sfloat = VK_FORMAT_R16G16B16A16_SFLOAT,
    r32_uint = VK_FORMAT_R32_UINT,
    r32_sint = VK_FORMAT_R32_SINT,
    r32_sfloat = VK_FORMAT_R32_SFLOAT,
    r32g32_uint = VK_FORMAT_R32G32_UINT,
    r32g32_sint = VK_FORMAT_R32G32_SINT,
    r32g32_sfloat = VK_FORMAT_R32G32_SFLOAT,
    r32g32b32_uint = VK_FORMAT_R32G32B32_UINT,
    r32g32b32_sint = VK_FORMAT_R32G32B32_SINT,
    r32g32b32_sfloat = VK_FORMAT_R32G32B32_SFLOAT,
    r32g32b32a32_uint = VK_FORMAT_R32G32B32A32_UINT,
    r32g32b32a32_sint = VK_FORMAT_R32G32B32A32_SINT,
    r32g32b32a32_sfloat = VK_FORMAT_R32G32B32A32_SFLOAT,
    r64_uint = VK_FORMAT_R64_UINT,
    r64_sint = VK_FORMAT_R64_SINT,
    r64_sfloat = VK_FORMAT_R64_SFLOAT,
    r64g64_uint = VK_FORMAT_R64G64_UINT,
    r64g64_sint = VK_FORMAT_R64G64_SINT,
    r64g64_sfloat = VK_FORMAT_R64G64_SFLOAT,
    r64g64b64_uint = VK_FORMAT_R64G64B64_UINT,
    r64g64b64_sint = VK_FORMAT_R64G64B64_SINT,
    r64g64b64_sfloat = VK_FORMAT_R64G64B64_SFLOAT,
    r64g64b64a64_uint = VK_FORMAT_R64G64B64A64_UINT,
    r64g64b64a64_sint = VK_FORMAT_R64G64B64A64_SINT,
    r64g64b64a64_sfloat = VK_FORMAT_R64G64B64A64_SFLOAT,
    b10g11r11_ufloat_pack32 = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    e5b9g9r9_ufloat_pack32 = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
    d16_unorm = VK_FORMAT_D16_UNORM,
    x8_d24_unorm_pack32 = VK_FORMAT_X8_D24_UNORM_PACK32,
    d32_sfloat = VK_FORMAT_D32_SFLOAT,
    s8_uint = VK_FORMAT_S8_UINT,
    d16_unorm_s8_uint = VK_FORMAT_D16_UNORM_S8_UINT,
    d24_unorm_s8_uint = VK_FORMAT_D24_UNORM_S8_UINT,
    d32_sfloat_s8_uint = VK_FORMAT_D32_SFLOAT_S8_UINT,
    bc1_rgb_unorm_block = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
    bc1_rgb_srgb_block = VK_FORMAT_BC1_RGB_SRGB_BLOCK,
    bc1_rgba_unorm_block = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    bc1_rgba_srgb_block = VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
    bc2_unorm_block = VK_FORMAT_BC2_UNORM_BLOCK,
    bc2_srgb_block = VK_FORMAT_BC2_SRGB_BLOCK,
    bc3_unorm_block = VK_FORMAT_BC3_UNORM_BLOCK,
    bc3_srgb_block = VK_FORMAT_BC3_SRGB_BLOCK,
    bc4_unorm_block = VK_FORMAT_BC4_UNORM_BLOCK,
    bc4_snorm_block = VK_FORMAT_BC4_SNORM_BLOCK,
    bc5_unorm_block = VK_FORMAT_BC5_UNORM_BLOCK,
    bc5_snorm_block = VK_FORMAT_BC5_SNORM_BLOCK,
    bc6h_ufloat_block = VK_FORMAT_BC6H_UFLOAT_BLOCK,
    bc6h_sfloat_block = VK_FORMAT_BC6H_SFLOAT_BLOCK,
    bc7_unorm_block = VK_FORMAT_BC7_UNORM_BLOCK,
    bc7_srgb_block = VK_FORMAT_BC7_SRGB_BLOCK,
    etc2_r8g8b8_unorm_block = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
    etc2_r8g8b8_srgb_block = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
    etc2_r8g8b8a1_unorm_block = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
    etc2_r8g8b8a1_srgb_block = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
    etc2_r8g8b8a8_unorm_block = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
    etc2_r8g8b8a8_srgb_block = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
    eac_r11_unorm_block = VK_FORMAT_EAC_R11_UNORM_BLOCK,
    eac_r11_snorm_block = VK_FORMAT_EAC_R11_SNORM_BLOCK,
    eac_r11g11_unorm_block = VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
    eac_r11g11_snorm_block = VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
    astc_4x4_unorm_block = VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
    astc_4x4_srgb_block = VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
    astc_5x4_unorm_block = VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
    astc_5x4_srgb_block = VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
    astc_5x5_unorm_block = VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
    astc_5x5_srgb_block = VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
    astc_6x5_unorm_block = VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
    astc_6x5_srgb_block = VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
    astc_6x6_unorm_block = VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
    astc_6x6_srgb_block = VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
    astc_8x5_unorm_block = VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
    astc_8x5_srgb_block = VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
    astc_8x6_unorm_block = VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
    astc_8x6_srgb_block = VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
    astc_8x8_unorm_block = VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
    astc_8x8_srgb_block = VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
    astc_10x5_unorm_block = VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
    astc_10x5_srgb_block = VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
    astc_10x6_unorm_block = VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
    astc_10x6_srgb_block = VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
    astc_10x8_unorm_block = VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
    astc_10x8_srgb_block = VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
    astc_10x10_unorm_block = VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
    astc_10x10_srgb_block = VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
    astc_12x10_unorm_block = VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
    astc_12x10_srgb_block = VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
    astc_12x12_unorm_block = VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
    astc_12x12_srgb_block = VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
    g8b8g8r8_422_unorm = VK_FORMAT_G8B8G8R8_422_UNORM,
    b8g8r8g8_422_unorm = VK_FORMAT_B8G8R8G8_422_UNORM,
    g8_b8_r8_3plane_420_unorm = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
    g8_b8r8_2plane_420_unorm = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    g8_b8_r8_3plane_422_unorm = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
    g8_b8r8_2plane_422_unorm = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
    g8_b8_r8_3plane_444_unorm = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
    r10x6_unorm_pack16 = VK_FORMAT_R10X6_UNORM_PACK16,
    r10x6g10x6_unorm_2pack16 = VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
    r10x6g10x6b10x6a10x6_unorm_4pack16 = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
    g10x6b10x6g10x6r10x6_422_unorm_4pack16 = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
    b10x6g10x6r10x6g10x6_422_unorm_4pack16 = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
    g10x6_b10x6_r10x6_3plane_420_unorm_3pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
    g10x6_b10x6r10x6_2plane_420_unorm_3pack16 = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
    g10x6_b10x6_r10x6_3plane_422_unorm_3pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
    g10x6_b10x6r10x6_2plane_422_unorm_3pack16 = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
    g10x6_b10x6_r10x6_3plane_444_unorm_3pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
    r12x4_unorm_pack16 = VK_FORMAT_R12X4_UNORM_PACK16,
    r12x4g12x4_unorm_2pack16 = VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
    r12x4g12x4b12x4a12x4_unorm_4pack16 = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
    g12x4b12x4g12x4r12x4_422_unorm_4pack16 = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
    b12x4g12x4r12x4g12x4_422_unorm_4pack16 = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
    g12x4_b12x4_r12x4_3plane_420_unorm_3pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
    g12x4_b12x4r12x4_2plane_420_unorm_3pack16 = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
    g12x4_b12x4_r12x4_3plane_422_unorm_3pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
    g12x4_b12x4r12x4_2plane_422_unorm_3pack16 = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
    g12x4_b12x4_r12x4_3plane_444_unorm_3pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
    g16b16g16r16_422_unorm = VK_FORMAT_G16B16G16R16_422_UNORM,
    b16g16r16g16_422_unorm = VK_FORMAT_B16G16R16G16_422_UNORM,
    g16_b16_r16_3plane_420_unorm = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
    g16_b16r16_2plane_420_unorm = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
    g16_b16_r16_3plane_422_unorm = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
    g16_b16r16_2plane_422_unorm = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
    g16_b16_r16_3plane_444_unorm = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
    pvrtc1_2bpp_unorm_block_img = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
    pvrtc1_4bpp_unorm_block_img = VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
    pvrtc2_2bpp_unorm_block_img = VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
    pvrtc2_4bpp_unorm_block_img = VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
    pvrtc1_2bpp_srgb_block_img = VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
    pvrtc1_4bpp_srgb_block_img = VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
    pvrtc2_2bpp_srgb_block_img = VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
    pvrtc2_4bpp_srgb_block_img = VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
    astc_4x4_sfloat_block_ext = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
    astc_5x4_sfloat_block_ext = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
    astc_5x5_sfloat_block_ext = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
    astc_6x5_sfloat_block_ext = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
    astc_6x6_sfloat_block_ext = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
    astc_8x5_sfloat_block_ext = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
    astc_8x6_sfloat_block_ext = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
    astc_8x8_sfloat_block_ext = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
    astc_10x5_sfloat_block_ext = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
    astc_10x6_sfloat_block_ext = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
    astc_10x8_sfloat_block_ext = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
    astc_10x10_sfloat_block_ext = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
    astc_12x10_sfloat_block_ext = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
    astc_12x12_sfloat_block_ext = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
    a4r4g4b4_unorm_pack16_ext = VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
    a4b4g4r4_unorm_pack16_ext = VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
};

/// @brief
/// @param fmt
/// @return
inline constexpr std::string_view get_image_format_name(image_format fmt) {
  using namespace std::literals;
  switch (fmt) {
  case image_format::undefined: return "undefined"sv;
  case image_format::r4g4_unorm_pack8: return "r4g4_unorm_pack8"sv;
  case image_format::r4g4b4a4_unorm_pack16: return "r4g4b4a4_unorm_pack16"sv;
  case image_format::b4g4r4a4_unorm_pack16: return "b4g4r4a4_unorm_pack16"sv;
  case image_format::r5g6b5_unorm_pack16: return "r5g6b5_unorm_pack16"sv;
  case image_format::b5g6r5_unorm_pack16: return "b5g6r5_unorm_pack16"sv;
  case image_format::r5g5b5a1_unorm_pack16: return "r5g5b5a1_unorm_pack16"sv;
  case image_format::b5g5r5a1_unorm_pack16: return "b5g5r5a1_unorm_pack16"sv;
  case image_format::a1r5g5b5_unorm_pack16: return "a1r5g5b5_unorm_pack16"sv;
  case image_format::r8_unorm: return "r8_unorm"sv;
  case image_format::r8_snorm: return "r8_snorm"sv;
  case image_format::r8_uscaled: return "r8_uscaled"sv;
  case image_format::r8_sscaled: return "r8_sscaled"sv;
  case image_format::r8_uint: return "r8_uint"sv;
  case image_format::r8_sint: return "r8_sint"sv;
  case image_format::r8_srgb: return "r8_srgb"sv;
  case image_format::r8g8_unorm: return "r8g8_unorm"sv;
  case image_format::r8g8_snorm: return "r8g8_snorm"sv;
  case image_format::r8g8_uscaled: return "r8g8_uscaled"sv;
  case image_format::r8g8_sscaled: return "r8g8_sscaled"sv;
  case image_format::r8g8_uint: return "r8g8_uint"sv;
  case image_format::r8g8_sint: return "r8g8_sint"sv;
  case image_format::r8g8_srgb: return "r8g8_srgb"sv;
  case image_format::r8g8b8_unorm: return "r8g8b8_unorm"sv;
  case image_format::r8g8b8_snorm: return "r8g8b8_snorm"sv;
  case image_format::r8g8b8_uscaled: return "r8g8b8_uscaled"sv;
  case image_format::r8g8b8_sscaled: return "r8g8b8_sscaled"sv;
  case image_format::r8g8b8_uint: return "r8g8b8_uint"sv;
  case image_format::r8g8b8_sint: return "r8g8b8_sint"sv;
  case image_format::r8g8b8_srgb: return "r8g8b8_srgb"sv;
  case image_format::b8g8r8_unorm: return "b8g8r8_unorm"sv;
  case image_format::b8g8r8_snorm: return "b8g8r8_snorm"sv;
  case image_format::b8g8r8_uscaled: return "b8g8r8_uscaled"sv;
  case image_format::b8g8r8_sscaled: return "b8g8r8_sscaled"sv;
  case image_format::b8g8r8_uint: return "b8g8r8_uint"sv;
  case image_format::b8g8r8_sint: return "b8g8r8_sint"sv;
  case image_format::b8g8r8_srgb: return "b8g8r8_srgb"sv;
  case image_format::r8g8b8a8_unorm: return "r8g8b8a8_unorm"sv;
  case image_format::r8g8b8a8_snorm: return "r8g8b8a8_snorm"sv;
  case image_format::r8g8b8a8_uscaled: return "r8g8b8a8_uscaled"sv;
  case image_format::r8g8b8a8_sscaled: return "r8g8b8a8_sscaled"sv;
  case image_format::r8g8b8a8_uint: return "r8g8b8a8_uint"sv;
  case image_format::r8g8b8a8_sint: return "r8g8b8a8_sint"sv;
  case image_format::r8g8b8a8_srgb: return "r8g8b8a8_srgb"sv;
  case image_format::b8g8r8a8_unorm: return "b8g8r8a8_unorm"sv;
  case image_format::b8g8r8a8_snorm: return "b8g8r8a8_snorm"sv;
  case image_format::b8g8r8a8_uscaled: return "b8g8r8a8_uscaled"sv;
  case image_format::b8g8r8a8_sscaled: return "b8g8r8a8_sscaled"sv;
  case image_format::b8g8r8a8_uint: return "b8g8r8a8_uint"sv;
  case image_format::b8g8r8a8_sint: return "b8g8r8a8_sint"sv;
  case image_format::b8g8r8a8_srgb: return "b8g8r8a8_srgb"sv;
  case image_format::a8b8g8r8_unorm_pack32: return "a8b8g8r8_unorm_pack32"sv;
  case image_format::a8b8g8r8_snorm_pack32: return "a8b8g8r8_snorm_pack32"sv;
  case image_format::a8b8g8r8_uscaled_pack32: return "a8b8g8r8_uscaled_pack32"sv;
  case image_format::a8b8g8r8_sscaled_pack32: return "a8b8g8r8_sscaled_pack32"sv;
  case image_format::a8b8g8r8_uint_pack32: return "a8b8g8r8_uint_pack32"sv;
  case image_format::a8b8g8r8_sint_pack32: return "a8b8g8r8_sint_pack32"sv;
  case image_format::a8b8g8r8_srgb_pack32: return "a8b8g8r8_srgb_pack32"sv;
  case image_format::a2r10g10b10_unorm_pack32: return "a2r10g10b10_unorm_pack32"sv;
  case image_format::a2r10g10b10_snorm_pack32: return "a2r10g10b10_snorm_pack32"sv;
  case image_format::a2r10g10b10_uscaled_pack32: return "a2r10g10b10_uscaled_pack32"sv;
  case image_format::a2r10g10b10_sscaled_pack32: return "a2r10g10b10_sscaled_pack32"sv;
  case image_format::a2r10g10b10_uint_pack32: return "a2r10g10b10_uint_pack32"sv;
  case image_format::a2r10g10b10_sint_pack32: return "a2r10g10b10_sint_pack32"sv;
  case image_format::a2b10g10r10_unorm_pack32: return "a2b10g10r10_unorm_pack32"sv;
  case image_format::a2b10g10r10_snorm_pack32: return "a2b10g10r10_snorm_pack32"sv;
  case image_format::a2b10g10r10_uscaled_pack32: return "a2b10g10r10_uscaled_pack32"sv;
  case image_format::a2b10g10r10_sscaled_pack32: return "a2b10g10r10_sscaled_pack32"sv;
  case image_format::a2b10g10r10_uint_pack32: return "a2b10g10r10_uint_pack32"sv;
  case image_format::a2b10g10r10_sint_pack32: return "a2b10g10r10_sint_pack32"sv;
  case image_format::r16_unorm: return "r16_unorm"sv;
  case image_format::r16_snorm: return "r16_snorm"sv;
  case image_format::r16_uscaled: return "r16_uscaled"sv;
  case image_format::r16_sscaled: return "r16_sscaled"sv;
  case image_format::r16_uint: return "r16_uint"sv;
  case image_format::r16_sint: return "r16_sint"sv;
  case image_format::r16_sfloat: return "r16_sfloat"sv;
  case image_format::r16g16_unorm: return "r16g16_unorm"sv;
  case image_format::r16g16_snorm: return "r16g16_snorm"sv;
  case image_format::r16g16_uscaled: return "r16g16_uscaled"sv;
  case image_format::r16g16_sscaled: return "r16g16_sscaled"sv;
  case image_format::r16g16_uint: return "r16g16_uint"sv;
  case image_format::r16g16_sint: return "r16g16_sint"sv;
  case image_format::r16g16_sfloat: return "r16g16_sfloat"sv;
  case image_format::r16g16b16_unorm: return "r16g16b16_unorm"sv;
  case image_format::r16g16b16_snorm: return "r16g16b16_snorm"sv;
  case image_format::r16g16b16_uscaled: return "r16g16b16_uscaled"sv;
  case image_format::r16g16b16_sscaled: return "r16g16b16_sscaled"sv;
  case image_format::r16g16b16_uint: return "r16g16b16_uint"sv;
  case image_format::r16g16b16_sint: return "r16g16b16_sint"sv;
  case image_format::r16g16b16_sfloat: return "r16g16b16_sfloat"sv;
  case image_format::r16g16b16a16_unorm: return "r16g16b16a16_unorm"sv;
  case image_format::r16g16b16a16_snorm: return "r16g16b16a16_snorm"sv;
  case image_format::r16g16b16a16_uscaled: return "r16g16b16a16_uscaled"sv;
  case image_format::r16g16b16a16_sscaled: return "r16g16b16a16_sscaled"sv;
  case image_format::r16g16b16a16_uint: return "r16g16b16a16_uint"sv;
  case image_format::r16g16b16a16_sint: return "r16g16b16a16_sint"sv;
  case image_format::r16g16b16a16_sfloat: return "r16g16b16a16_sfloat"sv;
  case image_format::r32_uint: return "r32_uint"sv;
  case image_format::r32_sint: return "r32_sint"sv;
  case image_format::r32_sfloat: return "r32_sfloat"sv;
  case image_format::r32g32_uint: return "r32g32_uint"sv;
  case image_format::r32g32_sint: return "r32g32_sint"sv;
  case image_format::r32g32_sfloat: return "r32g32_sfloat"sv;
  case image_format::r32g32b32_uint: return "r32g32b32_uint"sv;
  case image_format::r32g32b32_sint: return "r32g32b32_sint"sv;
  case image_format::r32g32b32_sfloat: return "r32g32b32_sfloat"sv;
  case image_format::r32g32b32a32_uint: return "r32g32b32a32_uint"sv;
  case image_format::r32g32b32a32_sint: return "r32g32b32a32_sint"sv;
  case image_format::r32g32b32a32_sfloat: return "r32g32b32a32_sfloat"sv;
  case image_format::r64_uint: return "r64_uint"sv;
  case image_format::r64_sint: return "r64_sint"sv;
  case image_format::r64_sfloat: return "r64_sfloat"sv;
  case image_format::r64g64_uint: return "r64g64_uint"sv;
  case image_format::r64g64_sint: return "r64g64_sint"sv;
  case image_format::r64g64_sfloat: return "r64g64_sfloat"sv;
  case image_format::r64g64b64_uint: return "r64g64b64_uint"sv;
  case image_format::r64g64b64_sint: return "r64g64b64_sint"sv;
  case image_format::r64g64b64_sfloat: return "r64g64b64_sfloat"sv;
  case image_format::r64g64b64a64_uint: return "r64g64b64a64_uint"sv;
  case image_format::r64g64b64a64_sint: return "r64g64b64a64_sint"sv;
  case image_format::r64g64b64a64_sfloat: return "r64g64b64a64_sfloat"sv;
  case image_format::b10g11r11_ufloat_pack32: return "b10g11r11_ufloat_pack32"sv;
  case image_format::e5b9g9r9_ufloat_pack32: return "e5b9g9r9_ufloat_pack32"sv;
  case image_format::d16_unorm: return "d16_unorm"sv;
  case image_format::x8_d24_unorm_pack32: return "x8_d24_unorm_pack32"sv;
  case image_format::d32_sfloat: return "d32_sfloat"sv;
  case image_format::s8_uint: return "s8_uint"sv;
  case image_format::d16_unorm_s8_uint: return "d16_unorm_s8_uint"sv;
  case image_format::d24_unorm_s8_uint: return "d24_unorm_s8_uint"sv;
  case image_format::d32_sfloat_s8_uint: return "d32_sfloat_s8_uint"sv;
  case image_format::bc1_rgb_unorm_block: return "bc1_rgb_unorm_block"sv;
  case image_format::bc1_rgb_srgb_block: return "bc1_rgb_srgb_block"sv;
  case image_format::bc1_rgba_unorm_block: return "bc1_rgba_unorm_block"sv;
  case image_format::bc1_rgba_srgb_block: return "bc1_rgba_srgb_block"sv;
  case image_format::bc2_unorm_block: return "bc2_unorm_block"sv;
  case image_format::bc2_srgb_block: return "bc2_srgb_block"sv;
  case image_format::bc3_unorm_block: return "bc3_unorm_block"sv;
  case image_format::bc3_srgb_block: return "bc3_srgb_block"sv;
  case image_format::bc4_unorm_block: return "bc4_unorm_block"sv;
  case image_format::bc4_snorm_block: return "bc4_snorm_block"sv;
  case image_format::bc5_unorm_block: return "bc5_unorm_block"sv;
  case image_format::bc5_snorm_block: return "bc5_snorm_block"sv;
  case image_format::bc6h_ufloat_block: return "bc6h_ufloat_block"sv;
  case image_format::bc6h_sfloat_block: return "bc6h_sfloat_block"sv;
  case image_format::bc7_unorm_block: return "bc7_unorm_block"sv;
  case image_format::bc7_srgb_block: return "bc7_srgb_block"sv;
  case image_format::etc2_r8g8b8_unorm_block: return "etc2_r8g8b8_unorm_block"sv;
  case image_format::etc2_r8g8b8_srgb_block: return "etc2_r8g8b8_srgb_block"sv;
  case image_format::etc2_r8g8b8a1_unorm_block: return "etc2_r8g8b8a1_unorm_block"sv;
  case image_format::etc2_r8g8b8a1_srgb_block: return "etc2_r8g8b8a1_srgb_block"sv;
  case image_format::etc2_r8g8b8a8_unorm_block: return "etc2_r8g8b8a8_unorm_block"sv;
  case image_format::etc2_r8g8b8a8_srgb_block: return "etc2_r8g8b8a8_srgb_block"sv;
  case image_format::eac_r11_unorm_block: return "eac_r11_unorm_block"sv;
  case image_format::eac_r11_snorm_block: return "eac_r11_snorm_block"sv;
  case image_format::eac_r11g11_unorm_block: return "eac_r11g11_unorm_block"sv;
  case image_format::eac_r11g11_snorm_block: return "eac_r11g11_snorm_block"sv;
  case image_format::astc_4x4_unorm_block: return "astc_4x4_unorm_block"sv;
  case image_format::astc_4x4_srgb_block: return "astc_4x4_srgb_block"sv;
  case image_format::astc_5x4_unorm_block: return "astc_5x4_unorm_block"sv;
  case image_format::astc_5x4_srgb_block: return "astc_5x4_srgb_block"sv;
  case image_format::astc_5x5_unorm_block: return "astc_5x5_unorm_block"sv;
  case image_format::astc_5x5_srgb_block: return "astc_5x5_srgb_block"sv;
  case image_format::astc_6x5_unorm_block: return "astc_6x5_unorm_block"sv;
  case image_format::astc_6x5_srgb_block: return "astc_6x5_srgb_block"sv;
  case image_format::astc_6x6_unorm_block: return "astc_6x6_unorm_block"sv;
  case image_format::astc_6x6_srgb_block: return "astc_6x6_srgb_block"sv;
  case image_format::astc_8x5_unorm_block: return "astc_8x5_unorm_block"sv;
  case image_format::astc_8x5_srgb_block: return "astc_8x5_srgb_block"sv;
  case image_format::astc_8x6_unorm_block: return "astc_8x6_unorm_block"sv;
  case image_format::astc_8x6_srgb_block: return "astc_8x6_srgb_block"sv;
  case image_format::astc_8x8_unorm_block: return "astc_8x8_unorm_block"sv;
  case image_format::astc_8x8_srgb_block: return "astc_8x8_srgb_block"sv;
  case image_format::astc_10x5_unorm_block: return "astc_10x5_unorm_block"sv;
  case image_format::astc_10x5_srgb_block: return "astc_10x5_srgb_block"sv;
  case image_format::astc_10x6_unorm_block: return "astc_10x6_unorm_block"sv;
  case image_format::astc_10x6_srgb_block: return "astc_10x6_srgb_block"sv;
  case image_format::astc_10x8_unorm_block: return "astc_10x8_unorm_block"sv;
  case image_format::astc_10x8_srgb_block: return "astc_10x8_srgb_block"sv;
  case image_format::astc_10x10_unorm_block: return "astc_10x10_unorm_block"sv;
  case image_format::astc_10x10_srgb_block: return "astc_10x10_srgb_block"sv;
  case image_format::astc_12x10_unorm_block: return "astc_12x10_unorm_block"sv;
  case image_format::astc_12x10_srgb_block: return "astc_12x10_srgb_block"sv;
  case image_format::astc_12x12_unorm_block: return "astc_12x12_unorm_block"sv;
  case image_format::astc_12x12_srgb_block: return "astc_12x12_srgb_block"sv;
  case image_format::g8b8g8r8_422_unorm: return "g8b8g8r8_422_unorm"sv;
  case image_format::b8g8r8g8_422_unorm: return "b8g8r8g8_422_unorm"sv;
  case image_format::g8_b8_r8_3plane_420_unorm: return "g8_b8_r8_3plane_420_unorm"sv;
  case image_format::g8_b8r8_2plane_420_unorm: return "g8_b8r8_2plane_420_unorm"sv;
  case image_format::g8_b8_r8_3plane_422_unorm: return "g8_b8_r8_3plane_422_unorm"sv;
  case image_format::g8_b8r8_2plane_422_unorm: return "g8_b8r8_2plane_422_unorm"sv;
  case image_format::g8_b8_r8_3plane_444_unorm: return "g8_b8_r8_3plane_444_unorm"sv;
  case image_format::r10x6_unorm_pack16: return "r10x6_unorm_pack16"sv;
  case image_format::r10x6g10x6_unorm_2pack16: return "r10x6g10x6_unorm_2pack16"sv;
  case image_format::r10x6g10x6b10x6a10x6_unorm_4pack16: return "r10x6g10x6b10x6a10x6_unorm_4pack16"sv;
  case image_format::g10x6b10x6g10x6r10x6_422_unorm_4pack16: return "g10x6b10x6g10x6r10x6_422_unorm_4pack16"sv;
  case image_format::b10x6g10x6r10x6g10x6_422_unorm_4pack16: return "b10x6g10x6r10x6g10x6_422_unorm_4pack16"sv;
  case image_format::g10x6_b10x6_r10x6_3plane_420_unorm_3pack16: return "g10x6_b10x6_r10x6_3plane_420_unorm_3pack16"sv;
  case image_format::g10x6_b10x6r10x6_2plane_420_unorm_3pack16: return "g10x6_b10x6r10x6_2plane_420_unorm_3pack16"sv;
  case image_format::g10x6_b10x6_r10x6_3plane_422_unorm_3pack16: return "g10x6_b10x6_r10x6_3plane_422_unorm_3pack16"sv;
  case image_format::g10x6_b10x6r10x6_2plane_422_unorm_3pack16: return "g10x6_b10x6r10x6_2plane_422_unorm_3pack16"sv;
  case image_format::g10x6_b10x6_r10x6_3plane_444_unorm_3pack16: return "g10x6_b10x6_r10x6_3plane_444_unorm_3pack16"sv;
  case image_format::r12x4_unorm_pack16: return "r12x4_unorm_pack16"sv;
  case image_format::r12x4g12x4_unorm_2pack16: return "r12x4g12x4_unorm_2pack16"sv;
  case image_format::r12x4g12x4b12x4a12x4_unorm_4pack16: return "r12x4g12x4b12x4a12x4_unorm_4pack16"sv;
  case image_format::g12x4b12x4g12x4r12x4_422_unorm_4pack16: return "g12x4b12x4g12x4r12x4_422_unorm_4pack16"sv;
  case image_format::b12x4g12x4r12x4g12x4_422_unorm_4pack16: return "b12x4g12x4r12x4g12x4_422_unorm_4pack16"sv;
  case image_format::g12x4_b12x4_r12x4_3plane_420_unorm_3pack16: return "g12x4_b12x4_r12x4_3plane_420_unorm_3pack16"sv;
  case image_format::g12x4_b12x4r12x4_2plane_420_unorm_3pack16: return "g12x4_b12x4r12x4_2plane_420_unorm_3pack16"sv;
  case image_format::g12x4_b12x4_r12x4_3plane_422_unorm_3pack16: return "g12x4_b12x4_r12x4_3plane_422_unorm_3pack16"sv;
  case image_format::g12x4_b12x4r12x4_2plane_422_unorm_3pack16: return "g12x4_b12x4r12x4_2plane_422_unorm_3pack16"sv;
  case image_format::g12x4_b12x4_r12x4_3plane_444_unorm_3pack16: return "g12x4_b12x4_r12x4_3plane_444_unorm_3pack16"sv;
  case image_format::g16b16g16r16_422_unorm: return "g16b16g16r16_422_unorm"sv;
  case image_format::b16g16r16g16_422_unorm: return "b16g16r16g16_422_unorm"sv;
  case image_format::g16_b16_r16_3plane_420_unorm: return "g16_b16_r16_3plane_420_unorm"sv;
  case image_format::g16_b16r16_2plane_420_unorm: return "g16_b16r16_2plane_420_unorm"sv;
  case image_format::g16_b16_r16_3plane_422_unorm: return "g16_b16_r16_3plane_422_unorm"sv;
  case image_format::g16_b16r16_2plane_422_unorm: return "g16_b16r16_2plane_422_unorm"sv;
  case image_format::g16_b16_r16_3plane_444_unorm: return "g16_b16_r16_3plane_444_unorm"sv;
  case image_format::pvrtc1_2bpp_unorm_block_img: return "pvrtc1_2bpp_unorm_block_img"sv;
  case image_format::pvrtc1_4bpp_unorm_block_img: return "pvrtc1_4bpp_unorm_block_img"sv;
  case image_format::pvrtc2_2bpp_unorm_block_img: return "pvrtc2_2bpp_unorm_block_img"sv;
  case image_format::pvrtc2_4bpp_unorm_block_img: return "pvrtc2_4bpp_unorm_block_img"sv;
  case image_format::pvrtc1_2bpp_srgb_block_img: return "pvrtc1_2bpp_srgb_block_img"sv;
  case image_format::pvrtc1_4bpp_srgb_block_img: return "pvrtc1_4bpp_srgb_block_img"sv;
  case image_format::pvrtc2_2bpp_srgb_block_img: return "pvrtc2_2bpp_srgb_block_img"sv;
  case image_format::pvrtc2_4bpp_srgb_block_img: return "pvrtc2_4bpp_srgb_block_img"sv;
  case image_format::astc_4x4_sfloat_block_ext: return "astc_4x4_sfloat_block_ext"sv;
  case image_format::astc_5x4_sfloat_block_ext: return "astc_5x4_sfloat_block_ext"sv;
  case image_format::astc_5x5_sfloat_block_ext: return "astc_5x5_sfloat_block_ext"sv;
  case image_format::astc_6x5_sfloat_block_ext: return "astc_6x5_sfloat_block_ext"sv;
  case image_format::astc_6x6_sfloat_block_ext: return "astc_6x6_sfloat_block_ext"sv;
  case image_format::astc_8x5_sfloat_block_ext: return "astc_8x5_sfloat_block_ext"sv;
  case image_format::astc_8x6_sfloat_block_ext: return "astc_8x6_sfloat_block_ext"sv;
  case image_format::astc_8x8_sfloat_block_ext: return "astc_8x8_sfloat_block_ext"sv;
  case image_format::astc_10x5_sfloat_block_ext: return "astc_10x5_sfloat_block_ext"sv;
  case image_format::astc_10x6_sfloat_block_ext: return "astc_10x6_sfloat_block_ext"sv;
  case image_format::astc_10x8_sfloat_block_ext: return "astc_10x8_sfloat_block_ext"sv;
  case image_format::astc_10x10_sfloat_block_ext: return "astc_10x10_sfloat_block_ext"sv;
  case image_format::astc_12x10_sfloat_block_ext: return "astc_12x10_sfloat_block_ext"sv;
  case image_format::astc_12x12_sfloat_block_ext: return "astc_12x12_sfloat_block_ext"sv;
  case image_format::a4r4g4b4_unorm_pack16_ext: return "a4r4g4b4_unorm_pack16_ext"sv;
  case image_format::a4b4g4r4_unorm_pack16_ext: return "a4b4g4r4_unorm_pack16_ext"sv;
  default:
    // TODO
    assert(false);
  }
}

[[nodiscard]] inline constexpr bool is_depth_and_stencil_format(image_format fmt) noexcept
{
    switch (fmt) {
    case image_format::d16_unorm_s8_uint: return true;
    case image_format::d24_unorm_s8_uint: return true;
    case image_format::d32_sfloat_s8_uint: return true;
    default: return false;
    }
}


[[nodiscard]] inline constexpr bool is_stencil_only_format(image_format fmt) noexcept
{
    switch (fmt) {
    case image_format::s8_uint: return true;
    default: return false;
    }
}

[[nodiscard]] inline constexpr bool is_depth_only_format(image_format fmt) noexcept
{
    switch (fmt) {
    case image_format::d16_unorm: return true;
    case image_format::x8_d24_unorm_pack32: return true;
    case image_format::d32_sfloat: return true;
    default: return false;
    }
}

[[nodiscard]] inline constexpr vk::Format get_vk_format(image_format fmt) noexcept {
  return static_cast<vk::Format>(fmt);
}

} // namespace graal