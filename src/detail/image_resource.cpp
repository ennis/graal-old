#include <graal/glad.h>
#include <graal/image.hpp>

//---------------------------------------------------------------------------------
namespace graal {
namespace detail {

/*image_resource::image_resource(const image_desc &desc) : desc_{desc} {
  switch (desc_.type) {
  case image_type::image_1d:
    tex_ = create_texture_1d(GL_TEXTURE_1D, desc_.format, desc_.size[0], desc_.num_mipmaps);
    break;
  case image_type::image_1d_array:
    tex_ =
        create_texture_2d(GL_TEXTURE_1D_ARRAY, desc_.format, desc_.size[0],
                              desc_.size[1], desc_.num_mipmaps);
    break;
  case image_type::image_2d:
    tex_ = create_texture_2d(GL_TEXTURE_2D, desc_.format, desc_.size[0],
                                 desc_.size[1], desc_.num_mipmaps);
    break;
  case image_type::image_2d_array:
    tex_ =
        create_texture_3d(GL_TEXTURE_2D_ARRAY, desc_.format, desc_.size[0],
                              desc_.size[1], desc_.size[2], desc_.num_mipmaps);
    break;
  case image_type::image_3d:
    tex_ =
        create_texture_3d(GL_TEXTURE_3D, desc_.format, desc_.size[0],
                              desc_.size[1], desc_.size[2], desc_.num_mipmaps);
    break;
  case image_type::image_cube_map:
    tex_ =
        create_texture_2d(GL_TEXTURE_CUBE_MAP, desc_.format, desc_.size[0],
                              desc_.size[1], desc_.num_mipmaps);
    break;
  case image_type::image_2d_multisample:
    tex_ = create_texture_2d_multisample(GL_TEXTURE_2D_MULTISAMPLE,
                                             desc_.format, desc_.size[0],
                                             desc_.size[1], desc_.num_samples);
    break;
  case image_type::image_2d_multisample_array:
    tex_ = create_texture_3d_multisample(
        GL_TEXTURE_2D_MULTISAMPLE_ARRAY, desc_.format, desc_.size[0],
        desc_.size[1], desc_.size[3], desc_.num_samples);
    break;
  }
}

image_resource::~image_resource() {}*/

} // namespace detail
} // namespace graal