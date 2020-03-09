//
// Created by muyezhu on 12/2/18.
//
#include "mcp3d_image_common.hpp"
#include "mcp3d_tiff_tile_format.hpp"

using namespace std;

mcp3d::MImageInfo mcp3d::MTiffStackFormat::ReadImageInfo(const std::vector<std::string> &img_paths,
                                                         bool is_root_image, bool is_full_path,
                                                         const std::string &common_dir)
{
    mcp3d::TiffInfo tiff_info(img_paths[0]);
    int stack_xdim = tiff_info.image_width,
        stack_ydim = tiff_info.image_height,
        stack_zdim = tiff_info.n_directory;
}

