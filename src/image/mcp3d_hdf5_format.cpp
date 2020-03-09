//
// Created by muyezhu on 1/30/19.
//
#include <cstring>
#include <boost/algorithm/string/predicate.hpp>
#include <nlohmann/json.hpp>
#include "common/mcp3d_utility.hpp"
#include "mcp3d_hdf5_utils.hpp"
#include "mcp3d_imaris_io.hpp"
#include "mcp3d_hdf5_format.hpp"

using namespace std;
using json = nlohmann::json;

mcp3d::MImageInfo mcp3d::MHdf5Format::ReadImageInfo(const vector<string> &image_paths,
                                                    bool is_level0_image,
                                                    bool is_full_path,
                                                    const string &common_dir)
{
    MCP3D_ASSERT(image_paths.size() == 1)
    string hdf_path = is_full_path ?
                      image_paths[0] :
                      mcp3d::JoinPath({common_dir, image_paths[0]});
    MCP3D_ASSERT(mcp3d::FileIsFormat(hdf_path, mcp3d::FileFormats::HDF5))
    if (mcp3d::FileIsImaris(hdf_path))
        MCP3D_TRY(return ReadImageInfoImaris(hdf_path, is_level0_image);)
    else
        MCP3D_TRY(return ReadImageInfoVirtualHdf5(hdf_path, is_level0_image);)
}

void mcp3d::MHdf5Format::ReadData(mcp3d::MImage &image)
{
    MCP3D_ASSERT(!image.image_info().empty())
    MCP3D_ASSERT(!image.selected_view().empty())
    if (!image.DataExists())
        image.AllocateSelection();
    string image_path(image.image_info().image_path(0, 0));
    int pyr_level = image.selected_pyr_level();
    hid_t hdf5_id = mcp3d::Hdf5Handle(image_path);
    for (int i = 0; i < image.selected_view().n_channels(); ++i)
    {
        // assuming single time point
        int volume_id = i;
        if (mcp3d::FileIsImaris(image_path))
        {
            hid_t dataset_id = mcp3d::ImarisDatasetHandle(hdf5_id, pyr_level,
                                                          image.selected_view().view_channels()[i]);
            ReadDataset(dataset_id, image, volume_id);
            H5Dclose(dataset_id);
        }
        else // placeholder for virtual hdf5
        {

        }
    }
    H5Fclose(hdf5_id);
}

void mcp3d::MHdf5Format::ReadDataset(hid_t dataset_id, MImage &image, int volume_id)
{
    MCP3D_ASSERT(mcp3d::IsHdf5Dataset(dataset_id))
    MCP3D_ASSERT(volume_id >= 0 && volume_id < image.n_volumes())
    hid_t mem_type_id = mcp3d::Hdf5DataType(image.selected_view().voxel_type());
    hid_t mem_dataspace_id = MemoryDataSpace(image);
    hid_t file_dataspace_id = FileDataSpace(dataset_id, image);
    herr_t success = H5Dread(dataset_id, mem_type_id,
                             mem_dataspace_id, file_dataspace_id,
                             H5P_DEFAULT, image.Volume(volume_id));
    if (success < 0)
        MCP3D_RUNTIME_ERROR("failure to read hdf5 dataset")
}

hid_t mcp3d::MHdf5Format::MemoryDataSpace(const MImage &image)
{
    hsize_t mem_dims[3];
    for (int i = 0; i < 3; ++i)
        mem_dims[i] = (hsize_t)image.selected_view().view_xyz_dims()[i];
    hid_t mem_dataspace_id = H5Screate_simple(3, mem_dims, mem_dims);
    MCP3D_ASSERT(mem_dataspace_id >= 0)

    hsize_t mem_dataspace_start[3];
    for (int i = 0; i < 3; ++i)
        mem_dataspace_start[i] = 0;
    hsize_t mem_dataspace_stride[3];
    for (int i = 0; i < 3; ++i)
        mem_dataspace_stride[i] = 1;
    hsize_t mem_dataspace_count[3];
    vector<int> view_end = image.selected_view().view_level_offsets() +
                           image.selected_view().view_level_extents();
    vector<int> view_valid_end = mcp3d::Minimum(view_end, image.selected_view().view_level_image_xyz_dims());
    vector<int> view_in_bound_extents =
            mcp3d::XyzDimsWithStrides(
                    mcp3d::SubtractSeq<int>(view_valid_end,
                                            image.selected_view().view_level_offsets()),
                    image.selected_view().view_level_strides());
    for (int i = 0; i < 3; ++i)
        mem_dataspace_count[i] = (hsize_t)view_in_bound_extents[i];
    H5Sselect_hyperslab(mem_dataspace_id, H5S_SELECT_SET, mem_dataspace_start,
                        mem_dataspace_stride, mem_dataspace_count, NULL);
    return mem_dataspace_id;
}

hid_t mcp3d::MHdf5Format::FileDataSpace(hid_t dataset_id, const MImage &image)
{
    hid_t file_dataspace_id = H5Dget_space(dataset_id);
    MCP3D_ASSERT(file_dataspace_id >= 0)

    hsize_t file_dataspace_start[3];
    for (int i = 0; i < 3; ++i)
        file_dataspace_start[i] = (hsize_t)image.selected_view().view_level_offsets()[i];
    hsize_t file_dataspace_stride[3];
    for (int i = 0; i < 3; ++i)
        file_dataspace_stride[i] = (hsize_t)image.selected_view().view_level_strides()[i];
    hsize_t file_dataspace_count[3];
    vector<int> view_end = image.selected_view().view_level_offsets() +
                           image.selected_view().view_level_extents();
    vector<int> view_valid_end = mcp3d::Minimum(view_end, image.selected_view().view_level_image_xyz_dims());
    vector<int> view_in_bound_extents =
            mcp3d::XyzDimsWithStrides(
                    mcp3d::SubtractSeq<int>(view_valid_end,
                                            image.selected_view().view_level_offsets()),
                    image.selected_view().view_level_strides());
    for (int i = 0; i < 3; ++i)
        file_dataspace_count[i] = (hsize_t)view_in_bound_extents[i];
    H5Sselect_hyperslab(file_dataspace_id, H5S_SELECT_SET, file_dataspace_start,
                        file_dataspace_stride, file_dataspace_count, NULL);
    return file_dataspace_id;
}

mcp3d::MImageInfo mcp3d::MHdf5Format::ReadImageInfoImaris(const string &image_path,
                                                          bool is_level0_image)
{
    MCP3D_ASSERT(mcp3d::IsFile(image_path))
    vector<string> resolutions = mcp3d::ImarisResolutions(image_path);
    if (!mcp3d::ImarisResolutionsCorrect(resolutions))
        MCP3D_RUNTIME_ERROR("incorrect resolutions levels in .ims file")
    int n_levels = (int)resolutions.size();
    mcp3d::MImageInfo imaris_info{};
    // imaris may generate resolutions with xyz dimensions less than pyramid
    // ratio. if next level will have this occur, do not read
    for (int level = 0; level < n_levels; ++level)
    {
        mcp3d::MImageInfo level_info =
                ReadImagePyrInfoImaris(image_path, level,
                                       is_level0_image && level == 0);
        imaris_info += level_info;

        if (level < n_levels - 1)
        {
            long next_xy_ratio = mcp3d::IntPow(2, level + 1);
            int next_xdim = level_info.xyz_dims()[2] / 2,
                next_ydim = level_info.xyz_dims()[1] / 2;
            if (next_xy_ratio > next_xdim || next_xy_ratio > next_ydim)
            {
                MCP3D_MESSAGE("discarding level " + to_string(level + 1) +
                              " and later due to volume being under-sized in "
                              "xy dimensions")
                break;
            }
            int z_scale_start_level = imaris_info.z_scale_start_level();
            if (z_scale_start_level == mcp3d::ZSCALE_NONE)
                continue;
            long next_z_ratio = mcp3d::IntPow(2, level - z_scale_start_level + 2);
            int next_zdim = level_info.xyz_dims()[0] / 2;
            if (next_z_ratio > next_zdim)
            {
                MCP3D_MESSAGE("discarding level " + to_string(level + 1) +
                              " and later due to volume being under-sized in "
                              "z dimensions")
                break;
            }
        }
    }
    return imaris_info;
}

mcp3d::MImageInfo mcp3d::MHdf5Format::ReadImagePyrInfoImaris(const string &image_path,
                                                             int resolution_level,
                                                             bool is_level0_image)
{
    json img_pyr_info;
    img_pyr_info["format"] = "ims";
    img_pyr_info["level 0 image"] = is_level0_image ? "true" : "false";
    img_pyr_info["z dimension scaled"] = "unknown";
    img_pyr_info["image pyramid directory"] = mcp3d::ParentDir(image_path);
    mcp3d::ImarisResolutionInfo info =
            mcp3d::ReadImarisResolutionInfo(image_path, resolution_level, 0);
    img_pyr_info["x dimension"] = info.image_xyz_sizes[2];
    img_pyr_info["y dimension"] = info.image_xyz_sizes[1];
    img_pyr_info["z dimension"] = info.image_xyz_sizes[0];
    img_pyr_info["channels"] = info.n_channels;
    img_pyr_info["time points"] = 1;
    // single .imaris file
    img_pyr_info["xchunk dimension"] = info.image_xyz_sizes[2];
    img_pyr_info["ychunk dimension"] = info.image_xyz_sizes[1];
    img_pyr_info["zchunk dimension"] = info.image_xyz_sizes[0];
    img_pyr_info["dimensions order"] = "tczyx";
    img_pyr_info["image sequence"] = vector<string>({mcp3d::Basename(image_path)});
    img_pyr_info["voxel type"] = info.voxel_type_string;
    mcp3d::MPyrImageInfo pyr_info(img_pyr_info, mcp3d::FileFormats::HDF5);
    return mcp3d::MImageInfo(pyr_info);
}

mcp3d::MImageInfo mcp3d::MHdf5Format::ReadImageInfoVirtualHdf5(const string &image_path,
                                                               bool is_level0_image)
{
    // placeholder implementation
    return mcp3d::MImageInfo {};
}

