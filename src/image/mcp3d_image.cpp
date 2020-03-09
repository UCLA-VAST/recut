//
// Created by muyezhu on 9/17/17.
//
#include <cstring>
#include <algorithm>
#include "common/mcp3d_utility.hpp"
#include "mcp3d_image_io.hpp"

using namespace std;

mcp3d::MImage::MImage(const vector<int>& dimensions, VoxelType voxel_type):
                                                    MImageBase(false)
{
    MCP3D_ASSERT(voxel_type != mcp3d::VoxelType::UNKNOWN)
    SetUpFromDimensionsAndVoxelType(dimensions, voxel_type);
}

void mcp3d::MImage::SetUpFromDimensionsAndVoxelType(const vector<int> &dimensions,
                                                    VoxelType voxel_type)
{
    image_info_ = mcp3d::MImageInfo(dimensions, voxel_type);
    selected_view_.set_image_info(image_info_);
    // default arguments
    selected_view_.SelectView(mcp3d::MImageBlock({0, 0, 0},
                              {image_info_.xyz_dims()}),
                              0, false, mcp3d::VoxelType::M8U);
    selected_view_.set_voxel_type(voxel_type);
    // allocates memory
    AllocateSelection();
    // copy selected_view_ to loaded_view_
    loaded_view_.CopyView(selected_view_);
}

void mcp3d::MImage::ReadImageInfo(const std::string &img_root_dir)
{
    MCP3D_ASSERT(can_read_storage_)
    std::unique_ptr<MImageIO> io = make_unique<MImageIO>();
    string img_root_dir_(img_root_dir);
    if (mcp3d::IsFile(img_root_dir_))
        img_root_dir_ = mcp3d::ParentDir(img_root_dir_);
    MCP3D_ASSERT(mcp3d::IsDir(img_root_dir_))
    try
    {
        image_info_ = io->ReadImageInfo(img_root_dir);
    }
    catch (...)
    {
        MCP3D_PRINT_NESTED_EXCEPTION
        MCP3D_RETHROW(current_exception())
    }
    selected_view_.set_image_info(image_info_);
    loaded_view_.set_image_info(image_info_);
}

void mcp3d::MImage::RefreshImageInfo()
{
    if (image_info_.empty())
        MCP3D_RUNTIME_ERROR("can not refresh empty image info. "
                            "use ReadImageInfo instead")
    std::unique_ptr<MImageIO> io = make_unique<MImageIO>();
    io->RefreshImageInfo(*this);
    selected_view_.set_image_info(image_info_);
    loaded_view_.set_image_info(image_info_);
}

void mcp3d::MImage::set_voxel_type(VoxelType voxel_type)
{
    if (!can_read_storage_ && voxel_type == mcp3d::VoxelType::UNKNOWN)
    {
        cout << "can not set in memory MImage view voxel type to VoxelType::UNKNOWN" << endl;
        return;
    }
    selected_view_.set_voxel_type(voxel_type);
}

void mcp3d::MImage::SelectView(const MImageBlock &view_block,
                               int pyr_level,
                               bool interpret_block_as_local,
                               VoxelType voxel_type)
{
    if (!can_read_storage_ || is_wrapper_)
    {
        cout << "image created in memory or is wrapper, do nothing" << endl;
        return;
    }
    MCP3D_ASSERT(pyr_level >= 0)
    if (pyr_level >= n_pyr_levels())
        MCP3D_OUT_OF_RANGE("pyr_level exceeds maximum available value")
    if(selected_view_.image_info_.empty())
        selected_view_.set_image_info(image_info_);
    std::unique_ptr<MImageIO> io = make_unique<MImageIO>();
    try
    {
        selected_view_.SelectView(view_block, pyr_level,
                               interpret_block_as_local, voxel_type);
    }
    catch (...)
    {
        MCP3D_PRINT_NESTED_EXCEPTION
        MCP3D_RETHROW(current_exception())
    }
}

void mcp3d::MImage::ClearData()
{
    if (!is_wrapper_)
        data_.clear();
    else
        wrapped_data_.clear();
}

void mcp3d::MImage::ValidateVolumeData(int t, int c, bool auto_allocate)
{
    if (can_read_storage_)
    {
        if (!DataExists())
        {
            if (auto_allocate)
            {
                MCP3D_ASSERT(!selected_view_.empty())
                MCP3D_ASSERT(c >= 0 && c < selected_view_.n_channels())
                MCP3D_ASSERT(t >= 0 && t < selected_view_.n_times())
                AllocateSelection();
            }
            MCP3D_ASSERT(DataExists())
        }
        else
        {
            if (!loaded_view_.empty())
            {
                MCP3D_ASSERT(c >= 0 && c < loaded_view_.n_channels())
                MCP3D_ASSERT(t >= 0 && t < loaded_view_.n_times())
            }
            else
            {
                MCP3D_ASSERT(!selected_view_.empty())
                MCP3D_ASSERT(c >= 0 && c < selected_view_.n_channels())
                MCP3D_ASSERT(t >= 0 && t < selected_view_.n_times())
            }

        }
    }
    else
    {
        MCP3D_ASSERT(DataExists())
        MCP3D_ASSERT(c >= 0 && c < loaded_view_.n_channels())
        MCP3D_ASSERT(t >= 0 && t < loaded_view_.n_times())
    }

}

void mcp3d::MImage::ValidateVolumeData(int volume_index, bool auto_allocate)
{
    if (can_read_storage_)
    {
        if (!DataExists())
        {
            if (auto_allocate)
            {
                MCP3D_ASSERT(!selected_view_.empty())
                MCP3D_ASSERT(volume_index >= 0 &&
                             volume_index < selected_view_.n_volumes())
                AllocateSelection();
            }
            MCP3D_ASSERT(DataExists())
        }
        else
        {
            if (!loaded_view_.empty())
                MCP3D_ASSERT(volume_index >= 0 &&
                             volume_index < loaded_view_.n_volumes())
            else
            {
                MCP3D_ASSERT(!selected_view_.empty())
                // case where volume is requested before image read completes
                // mostly internal library operations rather than public API
                MCP3D_ASSERT(volume_index >= 0 &&
                             volume_index < selected_view_.n_volumes())
            }
        }
    }
    else
    {
        MCP3D_ASSERT(DataExists())
        MCP3D_ASSERT(volume_index >= 0 && volume_index < loaded_view_.n_volumes())
    }
}

bool mcp3d::MImage::DataExists() const
{
    if (!is_wrapper_)
        return !data_.empty();
    else
        return !wrapped_data_.empty();
}

void mcp3d::MImage::ReadData(bool black_background, const string &mode)
{
    if (!can_read_storage_)
    {
        if (mode == "verbose")
            MCP3D_MESSAGE("in memory data or wrapper, do nothing")
        return;
    }
    if (image_info_.empty())
        MCP3D_RUNTIME_ERROR("image info is not read, can not read stored data")
    if (selected_view_.empty())
        MCP3D_RUNTIME_ERROR("no image view selected")
    if (selected_view_.SameView(loaded_view_))
    {
        if (mode == "verbose")
            MCP3D_MESSAGE("selected image data is already loaded, do nothing")
        return;
    }
    if (mode == "verbose")
        selected_view_.PrintView();
    AllocateSelection();
    std::unique_ptr<MImageIO> io = make_unique<MImageIO>();
    try
    {
        io->ReadData(*this, black_background);
        loaded_view_.CopyView(selected_view_);
    }
    catch (...)
    {
        ClearData();
        loaded_view_.Clear();
        MCP3D_PRINT_NESTED_EXCEPTION
        MCP3D_RETHROW(current_exception())
    }
}

void mcp3d::MImage::SaveImageInfo()
{
    if (!mcp3d::MPIInitialized())
        // lambda function for macro expansion
        RANK0_CALL_SYNC([&](){image_info_.Save();})
    // in MPI execution consistent save should be controlled at higher level
    else
        image_info_.Save();
}

void mcp3d::MImage::WriteViewXYPlane(const std::string &img_path, int z,
                                     int c, int t)
{
    MCP3D_ASSERT(!data_.empty() || !wrapped_data_.empty())
    MCP3D_ASSERT(z >= 0 && z < loaded_view_.view_zdim())
    MCP3D_ASSERT(c >= 0 && c < loaded_view_.n_channels())
    MCP3D_ASSERT(t >= 0 && t < loaded_view_.n_times())
    int cv_type = mcp3d::VoxelTypeToCVTypes(loaded_view_.voxel_type(), 1);
    uint8_t* ptr = Plane(z, c, t);
    cv::Mat m(loaded_view_.view_ydim(), loaded_view_.view_xdim(),
              cv_type, ptr);
    cv::imwrite(img_path, m);
}

void mcp3d::MImage::WriteViewVolume(const string &out_dir,
                                    const string& img_name_prefix,
                                    FileFormats volume_format)
{
    if (volume_format == FileFormats::UNKNOWN)
        MCP3D_DOMAIN_ERROR("view volume format can not be unknown")
    mcp3d::MakeDirectories(out_dir);
    if (loaded_view_.empty() || selected_view_.empty())
        cout << "no view is loaded or selected, do nothing" << endl;
    else if (loaded_view().empty())
        ReadData();

    unique_ptr<MImageIO> image_io = make_unique<MImageIO>();
    MCP3D_TRY(image_io->WriteViewVolume(*this, out_dir, img_name_prefix,
                                        volume_format);)
}


int mcp3d::MImage::MaxPyramidLevel() const
{
    MCP3D_ASSERT(!image_info_.empty())
    if (!can_read_storage_)
        return 0;
    int max_level = 0;
    int parent_xdim = image_pyr_infos()[0].xdim(),
        parent_ydim = image_pyr_infos()[0].ydim();
    while (true)
    {
        int child_xdim = parent_xdim / 2,
            child_ydim = parent_ydim / 2;
        int pyr_ratio = (int)(IntPow(2, max_level + 1));
        if (child_xdim < pyr_ratio || child_ydim < pyr_ratio)
            break;
        parent_xdim = child_xdim;
        parent_ydim = child_ydim;
        ++max_level;
    }
    return max_level;
}

int mcp3d::MImage::MaxNumPyramidLevels() const
{
    if (!can_read_storage_)
        return 1;
    return MaxPyramidLevel() + 1;
}

void mcp3d::MImage::WriteImagePyramids(int start_parent_level,
                                       int end_parent_level,
                                       bool multi_threading,
                                       bool save_image_info,
                                       FileFormats write_format)
{
    if (start_parent_level >= n_pyr_levels())
        MCP3D_INVALID_ARGUMENT("the requested parent level does not exist within the image hierarchy")
    for (int i = start_parent_level; i < end_parent_level; ++i)
    {
        if (i == MaxPyramidLevel())
        {
            int child_xdim = image_pyr_infos()[i].xdim() / 2,
                    child_ydim = image_pyr_infos()[i].ydim() / 2;
            int pyr_ratio = (int)(IntPow(2, i + 1));
            cout << "xy dimensions at level " << i + 1
                 << " would be: " << mcp3d::JoinVector(vector<int>({child_xdim, child_ydim}), ", ")
                 << ", less than the pyramid ratio " << pyr_ratio
                 << ", not producing its pyramid and any later pyramids" << endl;
            break;
        }
        MCP3D_TRY(WriteImagePyramid(i, multi_threading, save_image_info, write_format);)
    }
}

void mcp3d::MImage::WriteImagePyramid(int parent_level, bool multi_threading,
                                      bool save_image_info,
                                      FileFormats write_format)
{
    try
    {
        unique_ptr<MImageIO> image_io = make_unique<MImageIO>();
        image_io->WriteImagePyramid(*this, parent_level, multi_threading,
                                    write_format);
        RefreshImageInfo();
        if (save_image_info)
            SaveImageInfo();
    }
    catch (...)
    {
        MCP3D_RETHROW(current_exception())
    }
}

bool mcp3d::MImage::HasEqualData(const MImage &other)
{
    if (loaded_view_.empty() || other.loaded_view_.empty())
        return false;
    if (loaded_view_.view_dims() != other.loaded_view_.view_dims())
        return false;
    if (is_wrapper_ && other.is_wrapper_)
        if (wrapped_data_ == other.wrapped_data_)
            return true;
    for (int i = 0; i < loaded_view_.n_volumes(); ++i)
        if (!mcp3d::DataVolumeEqual(Volume(i), other.ConstVolume(i),
                                    loaded_view_.view_xyz_dims(),
                                    loaded_view_.BytesPerVoxel()))
            return false;
    return true;
}


