#include "load_image.hpp"

#include <OpenImageIO/imageio.h>

graal::image_2d load_texture(graal::queue& queue, const std::filesystem::path& path, graal::image_usage usage, bool mipmaps) {
    const auto image_input = OIIO::ImageInput::open(path.string());
    if (!image_input) { throw std::runtime_error{"could not load image"}; }

    const auto& spec = image_input->spec();

    // determine the format
    if (spec.nchannels > 4) { throw std::runtime_error{"too many channels in image file"}; }

    graal::image_format fmt;
    OIIO::TypeDesc fmt_tydesc;
    uint32_t bpp;
    if (spec.format == OIIO::TypeUInt8) {
        // U8
        switch (spec.nchannels) {
            case 1:
                bpp = 1;
                fmt = graal::image_format::r8_unorm;
                fmt_tydesc = OIIO::TypeUInt8;
                break;
            case 2:
                bpp = 2;
                fmt = graal::image_format::r8g8_unorm;
                fmt_tydesc = OIIO::TypeUInt8;
                break;
            case 3:
                bpp = 4;
                fmt = graal::image_format::r8g8b8a8_unorm;
                fmt_tydesc = OIIO::TypeUInt8;
                break;
            case 4:
            default:
                bpp = 4;
                fmt = graal::image_format::r8g8b8a8_unorm;
                fmt_tydesc = OIIO::TypeUInt8;
                break;
        }
    } else {
        throw std::runtime_error{"unsupported channel type"};
    }

    const auto mip_levels = mipmaps ? graal::get_mip_level_count(spec.width, spec.height) : 1;

    // create the texture object
    graal::image_2d texture{queue.get_device(), usage | graal::image_usage::transfer_dst, fmt,
            graal::range{(size_t) spec.width, (size_t) spec.height},
            graal::image_properties{.mip_levels = mip_levels}};
    texture.set_name(path.string());

    const uint32_t buffer_px_stride = bpp;
    const uint32_t buffer_row_length = spec.width;
    const uint32_t image_width = spec.width;
    const uint32_t image_height = spec.height;

    // get a staging buffer
    vk::Buffer staging_buffer;
    vk::DeviceSize staging_buffer_offset;
    void* staging_mem =
            queue.get_staging_buffer(16, buffer_row_length * image_height * buffer_px_stride, staging_buffer, staging_buffer_offset);

    // read into the staging buffer
    image_input->read_image(0, 0, 0, spec.nchannels, fmt_tydesc, staging_mem, buffer_px_stride);

    queue.compute_pass("load_image", [=](graal::handler& h) {

        // XXX 
        h.add_image_access(texture, vk::AccessFlagBits::eTransferWrite,
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                vk::ImageLayout::eTransferDstOptimal);

        return [=](vk::CommandBuffer cb) {
            vk::BufferImageCopy region{
                .bufferOffset = 0,
                    .bufferRowLength = buffer_row_length,
                    .bufferImageHeight = image_height,
                    .imageSubresource =
                            vk::ImageSubresourceLayers{
                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                                    .mipLevel = 0,
                                    .baseArrayLayer = 0,
                                    .layerCount = 1},
                    .imageOffset = vk::Offset3D{.x = 0, .y = 0, .z = 0},
                    .imageExtent = vk::Extent3D{.width = image_width,
                            .height = image_height,
                            .depth = 1}};

            cb.copyBufferToImage(staging_buffer, texture.vk_image(),
                    vk::ImageLayout::eTransferDstOptimal, 1, &region);

            // generate mipmaps (TODO)
            if (mip_levels > 1) {
                // barrier:
                for (size_t i = 1; i < mip_levels; ++i) {
                    //cb.blitImage();
                }
            }
        };
    });

    return texture;
}