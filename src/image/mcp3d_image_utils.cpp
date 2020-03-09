//
// Created by muyezhu on 10/13/17.
//
#include <regex>
#include <iostream>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include "common/mcp3d_utility.hpp"
#include "mcp3d_image_utils.hpp"

using namespace std;

bool mcp3d::CompTiffTileId(const string &tif_path1, const string &tif_path2)
{
    string tif_name1 = mcp3d::Basename(tif_path1),
           tif_name2 = mcp3d::Basename(tif_path2);
    regex tilestr_pattern("(.*)(_x[0-9]+)(_y[0-9]+)(_z[0-9]+).tif");
    smatch m1xyz, m2xyz;
    string s1name, s1x, s1y, s1z, s2name, s2x, s2y, s2z;
    regex_match(tif_name1, m1xyz, tilestr_pattern);
    regex_match(tif_name2, m2xyz, tilestr_pattern);
    if (m1xyz.empty() || m2xyz.empty()) MCP3D_RUNTIME_ERROR(
            "string does not contain valid tile format _x[0-9]+_y[0-9]+_z[0-9]+.tif")
    s1name = m1xyz.str(1);
    s2name = m2xyz.str(1);
    s1z = m1xyz.str(4);
    s2z = m2xyz.str(4);
    if (s1name != s2name || s1z != s2z) MCP3D_RUNTIME_ERROR(
            "the two tiff tile path does not correspond to the same tiff image")
    s1x = m1xyz.str(2);
    s1y = m1xyz.str(3);
    s2x = m2xyz.str(2);
    s2y = m2xyz.str(3);
    if (s1x.size() != s2x.size() || s1y.size() != s2y.size()) MCP3D_RUNTIME_ERROR(
            "the two tiff tile path have different tile xy pattern string length")
    if (s1y < s2y)
        return true;
    else if (s1y > s2y)
        return false;
    else
        return s1x <= s2x;
}

bool mcp3d::MatIsEqual(cv::Mat m1, cv::Mat m2)
{
    if (m1.rows != m2.rows || m1.cols != m2.cols)
        return false;
    if (m1.type() != m2.type())
        return false;
    size_t n_bytes;
    if (m1.isContinuous() && m2.isContinuous())
    {
        n_bytes = m1.elemSize1() * m1.rows * m1.cols;
        return memcmp(m1.ptr(), m2.ptr(), n_bytes) == 0;
    }
    else
    {
        n_bytes = m1.elemSize1() * m1.cols;
        for (int r = 0; r < m1.rows; ++r)
            if (memcmp(m1.ptr(r), m2.ptr(r), n_bytes) != 0)
                return false;
        return true;
    }
}

long mcp3d::LinearAddressRobust(const std::vector<int> &dims_, int d0, int d1,
                                int d2)
{
    MCP3D_ASSERT(dims_.size() >= 2 && dims_.size() <= 3)
    vector<int> dims(dims_);
    if (dims.size() == 2)
        dims.push_back(1);
    MCP3D_ASSERT(d0 >= 0 && d0 < dims[0] &&
                 d1 >= 0 && d1 < dims[1] &&
                 d2 >= 0 && d2 < dims[2])
    std::vector<long> strides(3, 0);
    for (int i = 2; i>= 0; --i)
    {
        // if dims[i] == 0, dimension stride is 0
        if (dims[i] > 1)
        {
            long s = 1;
            for (int j = 2; j > i; --j)
                s *= (long)dims[j];
            strides[i] = s;
        }
    }
    return (long)d0 * strides[0] + (long)d1 * strides[1] + (long)d2 * strides[2];
}

long mcp3d::LinearAddress(const std::vector<int> &dims_, int d0, int d1, int d2)
{
    MCP3D_ASSERT(dims_.size() >= 2 && dims_.size() <= 3)
    vector<int> dims(dims_);
    if (dims.size() == 2)
        dims.push_back(1);
    std::vector<long> strides(3, 0);
    for (int i = 2; i>= 0; --i)
    {
        if (dims[i] > 1)
        {
            long s = 1;
            for (int j = 2; j > i; --j)
                s *= (long)dims[j];
            strides[i] = s;
        }
    }
    return (long)d0 * strides[0] + (long)d1 * strides[1] + (long)d2 * strides[2];
}

bool mcp3d::DataVolumeEqual(const void *ptr1, const void *ptr2,
                            const vector<int> &dims, int element_size)
{
    MCP3D_ASSERT(element_size > 0)
    MCP3D_ASSERT(mcp3d::AllPositive(dims))
    size_t n_bytes = mcp3d::ReduceProdSeq<size_t>(dims) * element_size;
    return memcmp(ptr1, ptr2, n_bytes) == 0;
}

