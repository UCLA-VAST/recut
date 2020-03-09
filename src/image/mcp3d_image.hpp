//
// Created by muyezhu on 9/15/17.
// currently only plan to support tif
// if rgb channels encountered will convert to gray
//

#ifndef MCP3D_MCP3D_IMAGE_HPP
#define MCP3D_MCP3D_IMAGE_HPP

#include <iostream>
#include <memory>
#include <new>
#include <cstring>
#include <vector>
#include <omp.h>
#include <tiff.h>
#include <tiffio.h>
#include "common/mcp3d_common.hpp"
#include "project_structure/mcp3d_project_structure.hpp"
#include "mcp3d_image_common.hpp"
#include "mcp3d_image_utils.hpp"
#include "mcp3d_image_view.hpp"
#include "mcp3d_image_base.hpp"


namespace mcp3d
{
class MImage: public MImageBase
{
public:
    /// default constructor, assuming reading from disk
    MImage(): MImageBase() {};
    /// used by files on divice and wrappers
    explicit MImage(bool from_disk): MImageBase(from_disk) {}

    /// exactly copy other. if other has loaded data, copy data as well
    /// delegate to MImage(other, true, true)
    MImage(const MImage& other): MImage(other, true, true) {}

    /// copy MImage to different stages, image info and foler slices will
    /// always be copied
    MImage(const MImage& other, bool copy_selection, bool copy_data):
            MImageBase(other, copy_selection, copy_data) {}

    /// image data created from memory with only dimensions and voxel_type
    /// information (no parent image), cannot read any device stored data
    /// create image info accordingly. MImage created from memory has a single
    /// level 0, image_selected_view_ and image_loaded_view_ are always the
    /// entire image, and can move image_selected_local_view_ within level 0
    explicit MImage(const std::vector<int>& dimensions,
                    VoxelType voxel_type = VoxelType::M16U);

    /// img_root_dir must exist, and is assumed to be the root of the volume
    /// pyramids or one of the level in the volume pyramids
    /// if the latter is true, MImageIO will retrieve the root instead
    void ReadImageInfo(const std::string &img_root_dir);

    /// used to add newly written images (after last ReadImageInfo) to MImageInfo
    /// hiarachy. requires image_info_ to be non empty
    void RefreshImageInfo();

    /// for device storage based MImage, this function calls SetImageView method
    /// of an MImageIO instance, which calls SetImageViewImpl with format
    /// specific parameters. voxel type here is the type to read into memory as,
    /// may not be same type as voxel type on device. if unknown, device stored
    /// voxel type is used.
    /// image view selection applies to next image reading operation, the selection
    /// does not reflect data held by data_. the image view selected has global
    /// context
    /// do nothing if in memory or is wrapper
    /// the function gaurantees the offsets of the global image block is within
    /// level 0 image boundary. at other image pyramid level, the offsets are
    /// allowed to run out of boundary, in which case reading image will return
    /// all background voxels. this is due to an level 0 with for example width
    /// 11, where its level 0 image will have width 5. global view offset = 10
    /// along width is valid, but will be out of bounds (10 / 2 = 5) at level 1
    void SelectView(const MImageBlock &view_block, int pyr_level = 0,
                    bool interpret_block_as_local = false,
                    VoxelType voxel_type = VoxelType::UNKNOWN);

    template <typename VType>
    bool LoadedDataIsType();

    /// views partially outside of global image volume is valid. the out of volume
    /// voxels will be filled with background. views have no overlap with image
    /// volume are invalid
    /// if reading operation fails, the data vector is cleared since its now in
    /// unknown state. image loaded view is cleared. caught error is rethrown
    void ReadData(bool black_background = true,
                  const std::string &mode = "verbose");

    /// does not take owernership of data
    /// if src_dims has 5 elements, its interpreted as tczyx dimensions
    /// if src_dims has 3 elements, its interpreted as zyx dimensions
    template <typename VType = uint8_t>
    void WrapData(std::vector<VType*> src_ptrs,
                  const std::vector<int>& src_dims);

    /// should only be called by rank 0 process or thread. save MImagePyrInfo
    /// in its pyr_level_dir as __pyr_level_xxx_info__.json. reconstruct
    /// pyr_level_dir using image root dir to form full path before writing
    /// to ensure consistent image info read outcome from stored image files or
    /// json files. if a pyr level dir is composite: pyr_level_xxx+, the range
    /// of pyr info objects for all levels are written separately to the pyr
    /// level directory. e.g. pyr_level_2+ continas two pyramid levels 2 and 3,
    /// __pyr_level_2_info__.json and __pyr_level_3_info__.json are both written
    /// to directory pyr_level_2+
    void SaveImageInfo();

    /// write output with opencv
    void WriteViewXYPlane(const std::string &img_path, int t, int c, int z);

    /// write loaded data. if no data loaded, load selected data first. if no
    /// selection is made either, do nothing
    /// output name should contain x, y, z start position of the view, with xyz
    /// values relative to the dimensions at the view's pyr_level. additionally
    /// stride, channel, time should also be in name
    void WriteViewVolume(const std::string &out_dir, const std::string& img_name_prefix,
                         FileFormats volume_format = FileFormats::UNKNOWN);

    /// write a whole pyramid level. active writers will write. for images constructed from memory
    void WriteAssembledImageLevel(const std::string& out_dir);

    int MaxPyramidLevel() const;

    int MaxNumPyramidLevels() const;

    /// if this function called on local machine with remote host option, validate
    /// call image states, and qsub an MPI executable on remote. local
    /// execution will do multi threading unless instructed not to
    /// writing image pyramid successively, using start_parent_level as initial
    /// image volume to downsize from, while the new pyramid level to be written
    /// is less than end_level. if all is well the pyramid levels should be
    /// generated with [start_parent_level, end_parent_level) as parent image,
    /// leaving the image with pyramid levels
    /// [0, ..., start_parent_level, start_parent_level + 1, ..., end_parent_level + 1)
    /// if end_parent_level argument not given, all possible levels are written
    /// rgb tiff image's pyramids will be gray
    void WriteImagePyramids(int start_parent_level, int end_parent_level = 999,
                            bool multi_threading = true, bool save_image_info = true,
                            FileFormats write_format = FileFormats::UNKNOWN);

    void WriteImagePyramid(int parent_level = 0, bool multi_threading = true,
                           bool save_image_info = true, 
                           FileFormats write_format = FileFormats::UNKNOWN);

    /// if not a wrapper, return true if data_ vector not empty
    /// if is a wrapper, return true if wrapped_data_ vector not empty
    bool DataExists() const;

    // acquire ownership of data in other_data and release current existing data
    // use with extreme caution: consistency between other_data and image_info_,
    // image view objects are not validated. mostly used for performance
    // considerations in certain operations
    template <typename VType = uint8_t>
    void AcquireData(std::vector<std::unique_ptr<VType[]>> &other_data);

    // false if neither has loaded data
    bool HasEqualData(const MImage &other);

    /// set selected image view data type. does not change selection dimensions
    void set_voxel_type(VoxelType voxel_type);

    friend class MImageIO;
private:
    void SetUpFromDimensionsAndVoxelType(const std::vector<int> &dimensions,
                                         VoxelType voxel_type);

    /// clear data_ vector for non wrapper instances. clear wrapped_data_ for
    /// wrapper instances
    void ClearData();

    void ValidateVolumeData(int c, int t, bool auto_allocate);

    void ValidateVolumeData(int volume_index, bool auto_allocate);

};

}

template <typename  VType>
bool mcp3d::MImage::LoadedDataIsType()
{
    static_assert(std::is_arithmetic<VType>(),
                  "can only pass arithmetic type template parameter");;
    return mcp3d::TypeToVoxelType<VType>() == voxel_type();
}

template <typename VType>
void mcp3d::MImage::AcquireData(std::vector<std::unique_ptr<VType[]>> &other_data)
{
    if (other_data.empty())
        return;
    data_.clear();
    if (!LoadedDataIsType<VType>())
        std::cout << "warning: currently loaded voxel type "
                  << mcp3d::VoxelTypeEnumToStr(voxel_type())
                  << ", acquiring voxel type " << mcp3d::VoxelTypeEnumToStr(mcp3d::TypeToVoxelType<VType>()) << std::endl;
    for (size_t i = 0; i < other_data.size(); ++i)
    {
        std::unique_ptr<uint8_t[]> other_ptr(reinterpret_cast<uint8*>(other_data[i].release()));
        data_.push_back(std::move(other_ptr));
    }
}

template <typename VType>
void mcp3d::MImage::WrapData(std::vector<VType*> src_ptrs,
                             const std::vector<int>& src_dims)
{
    MCP3D_ASSERT(!can_read_storage_)
    MCP3D_ASSERT(src_dims.size() == 5 || src_dims.size() == 3)
    ClearData();
    is_wrapper_ = true;
    SetUpFromDimensionsAndVoxelType(src_dims, mcp3d::TypeToVoxelType<VType>());
    for (const auto& src_ptr: src_ptrs)
        wrapped_data_.push_back(reinterpret_cast<uint8_t*>(src_ptr));
}

#endif //MCP3D_MCP3D_IMAGE_HPP
