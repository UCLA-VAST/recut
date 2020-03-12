//
// Created by mzhu on 4/4/17.
//

#ifndef MCP3D_LARGE_IMAGE_IO_HPP
#define MCP3D_LARGE_IMAGE_IO_HPP

#include <cstdlib>
#include <iostream>
#include <tiff.h>
#include <tiffio.h>
#include "mcp3d_image_common.hpp"

namespace mcp3d
{
void ResizeLargeTiff(const std::string &tiff_path);

/// crop very large tiff images into square tiles of images.
/// image can be gray, rgb or rgba, with uint8, uint16 or uint32 data type
/// assuming tif image contain single directory.
/// only planar_config = 1 is handled
/// generated image subimgs have full path:
/// tif_subimg_dir/[tif_name]_x[i]_y[j]_z[k].tif
/// return true on successful operation
/// (0 returned from validate_large_tif_subimgs(tiff_path)) or small image
bool LargeTiffToSubimgs(const std::string &tiff_path,
                        uint32_t subimg_height = mcp3d::TIFFTILE_YDIM,
                        uint32_t subimg_width = mcp3d::TIFFTILE_XDIM);

/// copy sub image data from tiled tiff image into subimg_buf
/// the subimg's top left pixel has coordinate (subimg_origin_x, subimg_origin_y)
/// the subimg has height subimg_height and witdh subimg_width
/// if portion of sub image is outside of tiff image, black pixels are padded
/// called by large_image_io::large_tiff_to_subimgs internally
/// if not given or 0, subimg_width is same as subimg_height
/// subimg_src_origin_x: x coordinate of top left pixel of subimage
///                      of original tiff image
/// background: color of background. default 'b': background is black
void TransferTileToSubimg(const TIFF *tif, tdata_t subimg_buf,
                          int64_t subimg_height, int64_t subimg_width,
                          int64_t subimg_src_origin_x,
                          int64_t subimg_src_origin_y);

void TransferTileToSubimg(const std::string& tif_path, tdata_t subimg_buf,
                          int64_t subimg_height, int64_t subimg_width,
                          int64_t subimg_src_origin_x,
                          int64_t subimg_src_origin_y);

void TransferScanlineToSubimg(const std::string& tif_path, tdata_t subimg_buf,
                              int64_t subimg_height, int64_t subimg_width,
                              int64_t subimg_src_origin_x,
                              int64_t subimg_src_origin_yund);

void TransferScanlineToSubimg(const TIFF *tif, tdata_t subimg_buf,
                              int64_t subimg_height, int64_t subimg_width,
                              int64_t subimg_src_origin_x,
                              int64_t subimg_src_origin_y);

void TransferToSubimg(const TIFF *tif, tdata_t subimg_buf,
                      int64_t subimg_height, int64_t subimg_width,
                      int64_t subimg_src_origin_x,
                      int64_t subimg_src_origin_y);

void TransferToSubimg(const std::string& tif_path, tdata_t subimg_buf,
                      int64_t subimg_height, int64_t subimg_width,
                      int64_t subimg_src_origin_x,
                      int64_t subimg_src_origin_y);


/// copy scanlines of data from origin tif image to sub image scanlines.
/// the number of scanlines copied is equal to sub image height
/// called by large_image_io::large_tiff_to_subimgs internally
void TransferScanlineToSubimgRow(::TIFF *tif,
                                 ::TIFF **tif_subimgs,
                                 tdata_t *subimg_scanline_bufs,
                                 tdata_t scanline_buf,
                                 uint32_t tif_row,
                                 int64_t subimg_height,
                                 int64_t subimg_width,
                                 int num_subimgs_in_row);

void WriteTilesToLargeTiff(const std::vector<std::string> &tile_img_paths,
                           const std::string &dst_path,
                           int64_t img_height, int64_t img_width,
                           VoxelType datatype, bool skip_validation);

bool ValidateLargeTiffSubimgs(const std::string &tiff_path);

}

#endif //MCP3D_LARGE_IMAGE_IO_HPP
