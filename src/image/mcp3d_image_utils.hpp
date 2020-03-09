//
// Created by muyezhu on 10/13/17.
//

#ifndef MCP3D_IMAGE_UTILS_HPP
#define MCP3D_IMAGE_UTILS_HPP

#include <type_traits>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include "common/mcp3d_common.hpp"
#include "mcp3d_voxel_types.hpp"
#include "mcp3d_tiff_utils.hpp"
#include "mcp3d_image_view.hpp"

namespace mcp3d
{

inline bool TiffTilePathsIsSorted(const std::vector<std::string> &tile_paths)
{
    return std::is_sorted(tile_paths.begin(), tile_paths.end(), mcp3d::CompTiffTileId);
}

inline void SortTiffTilePaths(std::vector<std::string> &tile_paths)
{
    std::sort(tile_paths.begin(), tile_paths.end(), CompTiffTileId);
}

bool MatIsEqual(cv::Mat m1, cv::Mat m2);

/// d0 is the slowest varying dimension, d2 is the fastest. the *Robust version
/// performs more range checks. d0 - d2 are coordinates along each dimension.
/// dims_ contain size of each dimensions
long LinearAddressRobust(const std::vector<int> &dims_, int d0, int d1, int d2 = 0);

long LinearAddress(const std::vector<int> &dims_, int d0, int d1, int d2 = 0);

template <typename T1 = int, typename T2 = int>
std::vector<int> XyzDimsWithStrides(const std::vector<T1> &extents_,
                                    const std::vector<T2> &strides_ = std::vector<T2>{});

template <typename VType>
void SetConstant(void *data, const std::vector<int> &dims, VType value = VType{0});

template <typename VType>
void SetRandom(VType *data, const std::vector<int> &dims);

template <typename VType>
void SetConstantBlock(void *data, const std::vector<int> &dims,
                      const MImageBlock &image_block_ = mcp3d::MImageBlock{},
                      VType value = VType{0});

template <typename VType>
void SetConstantMultiBlocks(void *data, const std::vector<int> &dims,
                            const std::vector<mcp3d::MImageBlock> &image_blocks,
                            VType value = VType{0});

/// copy volumes of data from src to dst. the desired behavior is to copy
/// dst[x_offset: x_offset + x_extent: x_stride,
///     y_offset: y_offset + y_extent: y_stride,
///     z_offset: z_offset + z_extent: z_stride] to
/// src[x_offset: x_offset + x_extent: x_stride,
///     y_offset: y_offset + y_extent: y_stride,
///     z_offset: z_offset + z_extent: z_stride]
/// if VType_src is not given or if VType_src = VType_dst,
/// interpret src as having same data type as destination
/// if the src and dst data type is not the same,
/// src data is first converted to dst data type and then copied
template<typename VType_dst, typename VType_src = VType_dst>
void CopyDataVolume(void *dst, const std::vector<int> &dst_dims,
                    void *src, const std::vector<int> &src_dims,
                    const mcp3d::MImageBlock& dst_block = mcp3d::MImageBlock{},
                    const mcp3d::MImageBlock& src_block = mcp3d::MImageBlock{});

/// the two pointers must have the same type and same underlying array dimensions
bool DataVolumeEqual(const void *ptr1, const void *ptr2,
                     const std::vector<int> &dims, int element_size);

template<typename VType>
VType BackgroundValue(bool black_background = true);

template <typename VType>
void QuickSort(void* data, const std::vector<int> &dims);

// the q-th percentile of a vector of sorted values (min to max) is the q/100 value
// from back of the vector
template<typename VType>
VType TopPercentile(void *data, const std::vector<int> &dims, double q);


}

template <typename T1, typename T2>
std::vector<int> mcp3d::XyzDimsWithStrides(const std::vector<T1> &extents_,
                                           const std::vector<T2> &strides_)
{
    MCP3D_ASSERT(!extents_.empty())
    MCP3D_ASSERT(mcp3d::AllPositive(extents_))
    std::vector<T1> extents = extents_;
    std::vector<T2> strides = strides_;
    if (strides.empty())
        strides.insert(strides.end(), extents.size(), T2{1});
    MCP3D_ASSERT(mcp3d::AllPositive(strides))
    MCP3D_ASSERT(extents.size() == strides.size())
    std::vector<int> strided_extents;
    for (size_t i = 0; i < extents.size(); ++i)
        strided_extents.push_back(extents[i] / strides[i] + (int)(extents[i] % strides[i] > 0));
    return strided_extents;
};

template <typename VType>
void mcp3d::SetConstant(void *data, const std::vector<int> &dims, VType value)
{
    MCP3D_ASSERT(data)
    static_assert(std::is_arithmetic<VType>(),
                  "must have arithmetic element types");
    MCP3D_ASSERT(!dims.empty())
    MCP3D_ASSERT(mcp3d::AllPositive(dims))
    if (value == (VType)0)
        memset(data, value, mcp3d::ReduceProdSeq<size_t>(dims) * sizeof(VType));
    else
    {
        long addr_end = mcp3d::ReduceProdSeq<long>(dims);
        for (long i = 0; i < addr_end; ++i)
            ((VType*)data)[i] = value;
    }
}

template <typename VType>
void mcp3d::SetRandom(VType *data, const std::vector<int> &dims)
{
    MCP3D_ASSERT(data)
    static_assert(std::is_arithmetic<VType>(),
                  "must have arithmetic element types");
    MCP3D_ASSERT(dims.size() == 2 || dims.size() == 3)
    MCP3D_ASSERT(mcp3d::AllPositive(dims))
    if (dims.size() == 2)
    {
        MCPArray2DMap<VType> map(data, dims[0], dims[1]);
        map.setRandom();
    }
    else
    {
        MCPTensor3DMap<VType> map(data, dims[0], dims[1], dims[2]);
        map.setRandom();
    }
}

template <typename VType>
void mcp3d::SetConstantBlock(void *data, const std::vector<int> &dims,
                             const mcp3d::MImageBlock &image_block_,
                             VType value)
{
    MCP3D_ASSERT(data)
    static_assert(std::is_arithmetic<VType>(),
                  "must have arithmetic element types");
    MCP3D_ASSERT(dims.size() == 3)
    if (image_block_.empty())
    {
        mcp3d::SetConstant<VType>(data, dims, value);
        return;
    }
    std::vector<int> offsets(move(image_block_.offsets())),
                     extents(move(image_block_.extents())),
                     strides(move(image_block_.strides()));
    for (int i = 0; i < 3; ++i)
        if (extents[i] == 0)
            extents[i] = dims[i];

    MCP3D_ASSERT(AllNonNegative(offsets) && AllNonNegative(extents))
    MCP3D_ASSERT(AllPositive(strides))
    int z_end = std::min(dims[0], offsets[0] + extents[0]);
    int y_end = std::min(dims[1], offsets[1] + extents[1]);
    int x_end = std::min(dims[2], offsets[2] + extents[2]);
    if (offsets == std::vector<int>({0, 0, 0}) &&
        std::vector<int>({y_end, y_end, x_end}) == dims)
        mcp3d::SetConstant(data, dims, value);
    for (int z = offsets[0]; z < z_end; z += strides[0])
    {
        long addr;
        if (value == 0 && mcp3d::ReduceProdSeq<int>(strides) == 1)
        {
            size_t n_bytes = sizeof(VType) * extents[2];
            for (int y = offsets[1]; y < y_end; y += strides[1])
            {
                addr = mcp3d::LinearAddress(dims, z, offsets[1], offsets[0]);
                memset(((VType*)data) + addr, 0, n_bytes);
            }
        }
        else
        {
            for (int y = offsets[1]; y < y_end; y += strides[1])
            {
                int x = offsets[2];
                addr = mcp3d::LinearAddress(dims, z, y, x);
                for (; x < x_end; x += strides[2])
                    ((VType*)data)[addr++] = value;
            }
        }
    }
}

template <typename VType>
void mcp3d::SetConstantMultiBlocks(void *data, const std::vector<int> &dims,
                                   const std::vector<mcp3d::MImageBlock> &image_blocks,
                                   VType value)
{
    MCP3D_ASSERT(data)
    static_assert(std::is_arithmetic<VType>(),
                  "must have arithmetic element types");
    if (image_blocks.empty())
        SetConstant<VType>(data, dims, value);
    for (const mcp3d::MImageBlock& block: image_blocks)
        if (block.empty())
        {
            SetConstant<VType>(data, dims, value);
            return;
        }
    for (const mcp3d::MImageBlock& block: image_blocks)
        SetConstantBlock<VType>(data, dims, block, value);
}


template<typename VType_dst, typename VType_src>
void mcp3d::CopyDataVolume(void *dst, const std::vector<int> &dst_dims,
                           void *src, const std::vector<int> &src_dims,
                           const mcp3d::MImageBlock& dst_block,
                           const mcp3d::MImageBlock& src_block)
{
    MCP3D_ASSERT(dst && src)
    static_assert(std::is_arithmetic<VType_dst>() && std::is_arithmetic<VType_src>(),
                  "must have arithmetic element types");
    MCP3D_ASSERT(dst_dims.size() == 3 && src_dims.size() == 3)
    MCP3D_ASSERT(AllPositive(dst_dims) && AllPositive(src_dims))
    std::vector<int> src_offsets(move(src_block.offsets())),
                     src_extents(move(src_block.extents())),
                     src_strides(move(src_block.strides())),
                     dst_offsets(move(dst_block.offsets())),
                     dst_strides(move(dst_block.strides()));
    // for any missing extents value (0 value), fill it in as if
    // offsets: end of dimension was given
    for (int i = 0; i < 3; ++i)
        if (src_extents[i] == 0)
            src_extents[i] = src_dims[i] - src_offsets[i];

    MCP3D_ASSERT(AllNonNegative(src_offsets) && AllNonNegative(src_extents))
    MCP3D_ASSERT(AllPositive(src_strides))
    MCP3D_ASSERT(AllNonNegative(SubtractSeq<int>(src_dims, AddSeq<int>(src_offsets, src_extents))))
    if (ReduceProdSeq<int>(src_extents) == 0)
        return;

    MCP3D_ASSERT(AllNonNegative(dst_offsets))
    MCP3D_ASSERT(AllPositive(dst_strides))
    std::vector<int> src_extents_strided =
            mcp3d::XyzDimsWithStrides(src_extents, src_strides);
    std::vector<int> dst_extents = mcp3d::MultiplySeq<int>(dst_strides,
                                                           src_extents_strided);
    std::vector<int> dst_remain_extents = SubtractSeq<int>(dst_dims, AddSeq<int>(dst_offsets, dst_extents));
    MCP3D_ASSERT(AllNonNegative(dst_remain_extents))

    // copy data
    for (int z_dst = dst_offsets[0], z_src = src_offsets[0];
         z_src < src_offsets[0] + src_extents[0];
         z_dst += dst_strides[0], z_src += src_strides[0])
    {
        // memcpy for simple case
        if (mcp3d::ReduceProdSeq<int>(src_strides) == 1 &&
            mcp3d::ReduceProdSeq<int>(dst_strides) == 1 &&
            std::is_same<VType_src, VType_dst>::value)
        {
            if (mcp3d::AllZeros(src_offsets) && mcp3d::AllZeros(dst_offsets) &&
                src_extents == src_dims && dst_extents == dst_dims && src_dims == dst_dims)
            {
                #ifdef VERBOSE
                if (z_dst == dst_offsets[2])
                    MCP3D_MESSAGE("copy continuous block to same type")
                #endif
                size_t n_bytes = sizeof(VType_dst) * mcp3d::ReduceProdSeq<int>(src_dims);
                memcpy(dst, src, n_bytes);
            }
            else
            {
                #ifdef VERBOSE
                if (z_dst == dst_offsets[2])
                    MCP3D_MESSAGE("copy discontinuous unit stride block to sampe type");
                #endif
                long addr_dst = mcp3d::LinearAddress(dst_dims, z_dst, dst_offsets[1], dst_offsets[2]),
                     addr_src = mcp3d::LinearAddress(src_dims, z_src, src_offsets[1], src_offsets[2]);
                for(int y_src = src_offsets[1]; y_src < src_offsets[1] + src_extents[1]; ++y_src)
                {
                    size_t n_bytes = src_extents[2] * sizeof(VType_src);
                    memcpy(((VType_dst *) dst) + addr_dst,
                           ((VType_src *) src) + addr_src, n_bytes);
                    addr_dst += dst_dims[2];
                    addr_src += src_dims[2];
                }
            }
        }
        else
        {
            #ifdef VERBOSE
            if (z_dst == dst_offsets[2])
                MCP3D_MESSAGE("general copy")
            #endif
            for (int y_dst = dst_offsets[1], y_src = src_offsets[1];
                 y_src < src_offsets[1] + src_extents[1];
                 y_dst += dst_strides[1], y_src += src_strides[1])
            {
                int x_dst = dst_offsets[2], x_src = src_offsets[2];
                long addr_src = mcp3d::LinearAddress(src_dims, z_src, y_src, x_src),
                     addr_dst = mcp3d::LinearAddress(dst_dims, z_dst, y_dst, x_dst);
                if (std::is_same<VType_dst, VType_src>())
                    for (; x_src < src_offsets[2] + src_extents[2]; x_src += src_strides[2])
                    {
                        (reinterpret_cast<VType_dst*>(dst))[addr_dst] = (reinterpret_cast<VType_dst*>(src))[addr_src];
                        addr_dst += dst_strides[2];
                        addr_src += src_strides[2];
                    }
                else
                    for (; x_src < src_offsets[2] + src_extents[2]; x_src += src_strides[2])
                    {
                        (reinterpret_cast<VType_dst*>(dst))[addr_dst] = static_cast<VType_dst>((reinterpret_cast<VType_src*>(src))[addr_src]);
                        addr_dst += dst_strides[2];
                        addr_src += src_strides[2];
                    }
            }
        }
    }
}

template <typename VType>
VType mcp3d::BackgroundValue(bool black_background)
{
    static_assert(std::is_arithmetic<VType>::value, "VType must be arithmetic type");
    if (std::is_same<VType, uint8_t>::value)
    {
        if (black_background)
            return 0;
        else
            return UINT8_MAX;
    }
    if (std::is_same<VType, int8_t>::value)
    {
        if (black_background)
            return 0;
        else
            return INT8_MAX;
    }
    if (std::is_same<VType, uint16_t>::value)
    {
        if (black_background)
            return 0;
        else
            return UINT16_MAX;
    }
    if (std::is_same<VType, int16_t>::value)
    {
        if (black_background)
            return 0;
        else
            return INT16_MAX;
    }
    if (std::is_same<VType, uint32_t>::value)
    {
        if (black_background)
            return 0;
        else
            return UINT32_MAX;
    }
    if (std::is_same<VType, int32_t>::value)
    {
        if (black_background)
            return 0;
        else
            return INT32_MAX;
    }
    if (std::is_same<VType, uint64_t>::value)
    {
        if (black_background)
            return 0;
        else
            return UINT64_MAX;
    }
    if (std::is_same<VType, int64_t>::value)
    {
        if (black_background)
            return 0;
        else
            return INT64_MAX;
    }
    if (std::is_same<VType, float_t>::value)
    {
        if (black_background)
            return 0;
        else
            return FLT_MAX;
    }
    if (std::is_same<VType, double_t>::value)
    {
        if (black_background)
            return 0;
        else
            return DBL_MAX;
    }
    MCP3D_RUNTIME_ERROR("unknown voxel type")
}

template <typename VType>
VType mcp3d::TopPercentile(void *data, const std::vector<int> &dims, double q)
{
    MCP3D_ASSERT(data)
    MCP3D_ASSERT(mcp3d::AllPositive(dims))
    long n = mcp3d::ReduceProdSeq<long>(dims);
    MCP3D_ASSERT(q >= 0 && q <= 1)
    std::vector<VType> values;
    std::vector<int64_t> counts;
    typename std::vector<VType>::iterator value_iter;
    VType* data_vtype = (VType *)data;
    for (long i = 0; i < n; ++i)
    {
        value_iter = std::lower_bound(values.begin(), values.end(), data_vtype[i]);
        auto distance = std::distance(values.begin(), value_iter);
        if (distance < values.size() && *value_iter == data_vtype[i])
            ++counts[distance];
        else
        {
            values.insert(value_iter, data_vtype[i]);
            counts.insert(counts.begin() + distance, 0);
        }
    }
    auto n_q = (long)(std::floor(q * n));
    long n_counted = 0;
    for(int64_t i = (int64_t)(counts.size()) - 1; i >= 0; --i)
    {
        n_counted += counts[i];
        if (n_counted >= n_q)
            return values[i];
    }
    MCP3D_RUNTIME_ERROR("requested percentile value not found")
}

#endif //MCP3D_IMAGE_UTILS_HPP
