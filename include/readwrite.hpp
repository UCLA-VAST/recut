//
// Created by muyezhu on 9/19/17.
//

#ifndef MCP3D_READWRITE_HPP
#define MCP3D_READWRITE_HPP

#include <new>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "mcp3d_image_common.hpp"

namespace mcp3d
{
void ImgRead(const std::string &img_path, cv::Mat &m,
             VoxelType datatype, int downsize = 1, bool invert = false);

void ImgConvert(cv::Mat &m, VoxelType datatype, int downsize, bool invert = false);
    
void SubImgRead(const std::string &img_path, cv::Mat &m,
                int64_t x_len, int64_t y_len,
                int64_t x_start, int64_t y_start,
                VoxelType datatype = VoxelType::M8U,
                int downsize = -1, bool invert = false);

void WriteAllTiles(std::vector<std::string> &tile_img_paths,
                   const std::string &dst_path,
                   int64_t img_height, int64_t img_width,
                   int64_t tile_x_len, int64_t tile_y_len, VoxelType datatype,
                   bool skip_validation);

template<typename T>
void TensorPlaneWrite(const MCPTensor3D <T> &tensor, int z,
                      const std::string &img_path, bool invert = false)
{
    typename mcp3d::MCPTensor3D<T>::Dimensions d = tensor.dimensions();
    if (z < 0 || z >= d[2]) MCP3D_INVALID_ARGUMENT("invalid z value")
    int type;
    if (sizeof(T) == 1)
        type = CV_8U;
    else if (sizeof(T) == 2)
        type = CV_16U;
    else MCP3D_DOMAIN_ERROR("only supporting 8 bit or 16 bit images")
    T *mat_ptr = new(std::nothrow) T[d[0] * d[1]];
    if (mat_ptr == nullptr) MCP3D_BAD_ALLOC("failed to allocate memory")
    for (int i = 0; i < d[0]; ++i)
        for (int j = 0; j < d[1]; ++j)
            mat_ptr[i * d[1] + j] =
                    tensor.data()[i * d[1] * d[2] + j * d[2] + z];
    cv::Mat m(d[0], d[1], type, mat_ptr);
    if (invert) {
        if (type == CV_8U)
            m = 255 - m;
        else
            m = 65535 - m;
    }
    cv::imwrite(img_path, m);
    delete[] mat_ptr;
}

}
#endif //MCP3D_READWRITE_HPP
