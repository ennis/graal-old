#pragma once
#include <filesystem>
#include <graal/image.hpp>
#include <graal/queue.hpp>

graal::image_2d load_texture(graal::queue& queue, const std::filesystem::path& path,
        graal::image_usage usage = graal::image_usage::sampled | graal::image_usage::storage,
        bool mipmaps = false);
