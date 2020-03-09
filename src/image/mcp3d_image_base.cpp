//
// Created by muyezhu on 4/5/19.
//

#include "mcp3d_image_utils.hpp"
#include "mcp3d_image_base.hpp"

using namespace std;

mcp3d::MImageBase::MImageBase(const MImageBase &other, bool copy_selection,
                              bool copy_data):
                                    can_read_storage_(other.can_read_storage_),
                                    is_wrapper_(other.is_wrapper_),
                                    image_info_(other.image_info())
{
    image_info_ = other.image_info();
    selected_view_.set_image_info(image_info_);
    loaded_view_.set_image_info(image_info_);
    if (!is_wrapper_)
    {
        if (copy_selection)
            selected_view_.CopyView(other.selected_view());
        if (copy_data)
        {
            loaded_view_.CopyView(other.loaded_view());
            CopyData(other);
        }
    }
    else
    {
        selected_view_.CopyView(other.selected_view());
        loaded_view_.CopyView(other.loaded_view());
        wrapped_data_ = other.wrapped_data_;
    }
}

void mcp3d::MImageBase::CopyData(const mcp3d::MImageBase &other, const string& mode)
{
    if (other.loaded_view().empty())
    {
        if (mode == "verbose")
        MCP3D_MESSAGE("no data to copy from input image");
        return;
    }
    // copy loaded view of other image into selected view of this image
    selected_view_.CopyView(other.loaded_view());
    AllocateSelection();
    long nbytes = selected_view_.BytesPerVolume();
    for (int i = 0; i < selected_view_.n_volumes(); ++i)
    {
        std::memcpy(Volume<uint8_t>(i), other.ConstVolume<uint8_t>(i), (size_t)nbytes);
    }
}

void mcp3d::MImageBase::AllocateSelection(const string &output)
{
    if (is_wrapper_)
    {
        if (output == "verbose")
        MCP3D_MESSAGE("MImage instance is data wrapper, not allocating memory");
        return;
    }
    if (selected_view_.empty())
    MCP3D_RUNTIME_ERROR("can only allocate memory for non empty image view selection")
    if (selected_view_.view_dims() == loaded_view_.view_dims() &&
        selected_view_.voxel_type() == loaded_view_.voxel_type())
    {
        if (output == "verbose")
        MCP3D_MESSAGE("loaded image view have identical dimensions and voxel "
                              "type as selected view, no need to allocate memory")
        return;
    }
    ClearData();
    loaded_view_.Clear();
    for (int i = 0; i < selected_view_.n_volumes(); ++i)
    {
        data_.push_back(make_unique<uint8_t[]>((size_t)selected_view_.BytesPerVolume()));
        MCP3D_ASSERT(data_[i])
    }
}

long mcp3d::MImageBase::LoadedVoxelAddress(int z, int y, int x) const
{
    if (loaded_view_.empty())
        return -1;
    return mcp3d::LinearAddressRobust(loaded_view_.view_xyz_dims(), z, y, x);
}

const mcp3d::MPyrImageInfo& mcp3d::MImageBase::image_pyr_info(int level) const
{
    MCP3D_ASSERT(!image_info_.empty())
    MCP3D_ASSERT(level >= 0 && level < image_info().n_pyr_levels())
    return image_info().pyr_infos()[level];
}
