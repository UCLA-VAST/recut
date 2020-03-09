//
// Created by muyezhu on 9/19/17.
//
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mcp3d_image.hpp"
#include "mcp3d_tiff_partial_io.hpp"
#include "readwrite.hpp"

using namespace std;

void mcp3d::ImgRead(const string &img_path, cv::Mat &m,
                    VoxelType datatype, int downsize, bool invert)
{
    if (!mcp3d::IsFile(img_path))
        MCP3D_INVALID_ARGUMENT(img_path + " does not exist");
    mcp3d::TiffInfo tiff_info(img_path);
    if (tiff_info.samples_per_pixel > 3)
        MCP3D_DOMAIN_ERROR("does not support alpha channel")
    if (tiff_info.bits_per_sample != 8 &&
        tiff_info.bits_per_sample != 16 &&
        tiff_info.bits_per_sample != 32)
        MCP3D_DOMAIN_ERROR("only support 8 bit, 16 bit or 32 bit images")
    m = cv::imread(img_path, cv::IMREAD_ANYDEPTH + cv::IMREAD_GRAYSCALE);
    ImgConvert(m, datatype, downsize, invert);
}

void mcp3d::ImgConvert(cv::Mat &m, VoxelType datatype, int downsize, bool invert)
{
    MCP3D_ASSERT (!m.empty())
    if ((datatype != VoxelType::M8U &&
         datatype != VoxelType::M16U &&
         datatype != VoxelType::M32S)) MCP3D_DOMAIN_ERROR(
            "only support 8 bit unsigned, 16 bit unsigned or "
                    "32 bit signed gray scale images");
    while (downsize > 1)
    {
        cv::pyrDown(m, m);
        --downsize;
    }
    if (m.channels() > 1)
        // tiff images with chunky planar configuration pack in rgbrgbrgb order
        #if CV_MAJOR_VERSION < 4
            cv::cvtColor(m, m, CV_RGB2GRAY);
        #else
            cv::cvtColor(m, m, cv::COLOR_RGB2GRAY);
        #endif
    if (datatype == VoxelType::M8U)
        m.convertTo(m, CV_8U);
    else if (datatype == VoxelType::M16U)
        m.convertTo(m, CV_16U);
    else
        m.convertTo(m, CV_32S);
    if (invert)
    {
        if (datatype == VoxelType::M8U)
            m = UINT8_MAX - m;
        else if (datatype == VoxelType::M16U)
            m = UINT16_MAX - m;
        else
            m = INT32_MAX - m;
    }
}

void mcp3d::SubImgRead(const string &img_path, cv::Mat &m,
                       int64_t x_len, int64_t y_len,
                       int64_t x_start, int64_t y_start,
                       VoxelType datatype, int downsize, bool invert)
{
    mcp3d::TiffInfo tiff_info(img_path);
    short samples_per_pixel = tiff_info.samples_per_pixel,
          bits_per_sample = tiff_info.bits_per_sample;
    int64_t subimg_size = bits_per_sample / 8 *
                        samples_per_pixel * y_len * x_len;
    if (subimg_size <= 0) MCP3D_INVALID_ARGUMENT("invalid subimage size")
    tdata_t subimg_buf = _TIFFmalloc(subimg_size);
    ::TIFF* tif = TIFFOpen(img_path.c_str(), "r");
    MCP3D_ASSERT(tif)

    if (tiff_info.is_tiled)
        mcp3d::TransferTileToSubimg(tif, subimg_buf, y_len, x_len, x_start, y_start);
    else
        mcp3d::TransferScanlineToSubimg(tif, subimg_buf, y_len, x_len, x_start, y_start);
    int cv_type = VoxelTypeToCVTypes(datatype, samples_per_pixel);
    m.create(y_len, x_len, cv_type);
    // copy into a cv::Mat for easier memory management
    memcpy(m.ptr(), subimg_buf, (uint64_t)subimg_size);
    _TIFFfree(subimg_buf);
    TIFFClose(tif);
    mcp3d::ImgConvert(m, datatype, downsize, invert);
}

void mcp3d::WriteAllTiles(std::vector<std::string> &tile_img_paths,
                          const std::string &dst_path,
                          int64_t img_height, int64_t img_width,
                          int64_t tile_x_len, int64_t tile_y_len,
                          VoxelType datatype, bool skip_validation)
{
    if (!SupportedTiffVoxeltype(datatype)) MCP3D_INVALID_ARGUMENT(
            "only supporting 8 bit unsigned, 16 bit unsigned "
                    "or signed 32 bit pixel data type for large image io")
    if (!skip_validation)
        mcp3d::SortTiffTilePaths(tile_img_paths);
    if (img_height < tile_y_len)
        img_height = tile_y_len;
    if (img_width < tile_x_len)
        img_width = tile_x_len;
    mcp3d::WriteTilesToLargeTiff(tile_img_paths, dst_path,
                                 img_height, img_width,
                                 datatype, skip_validation);
}