//
// Created by muyezhu on 4/5/19.
//

#ifndef MCP3D_MCP3D_IMAGE_BASE_HPP
#define MCP3D_MCP3D_IMAGE_BASE_HPP

#include "common/mcp3d_common.hpp"
#include "project_structure/mcp3d_folder_slices.hpp"
#include "mcp3d_image_utils.hpp"
#include "mcp3d_image_info.hpp"
#include "mcp3d_image_view.hpp"

namespace mcp3d
{

class MImageBase
{
public:
    MImageBase(): MImageBase(true) {};

    explicit MImageBase(bool can_read_storage): can_read_storage_(can_read_storage),
                                                is_wrapper_(false),
                                                data_(std::vector<std::unique_ptr<uint8_t []>>{}),
                                                wrapped_data_(std::vector<uint8_t*> {}),
                                                older_data_(std::vector<std::vector<std::unique_ptr<uint8_t []>>>{}),
                                                image_folder_slices_(MFolderSlices {}),
                                                image_info_(MImageInfo {}),
                                                selected_view_(MImageView {}),
                                                loaded_view_(MImageView{}){};

    MImageBase(const MImageBase& other, bool copy_selection, bool copy_data);

    virtual ~MImageBase()= default;

    /// allocate memory for selected view and clears loaded_image_view_ since
    /// this function is called to prepare for loading new data.
    /// the function does nothing if MImage is a wrapper
    /// if the selected and loaded views have identical dimensions and voxel
    /// type, memory is not reallocated but loaded_image_view_ is still cleared
    void AllocateSelection(const std::string &output = "");

    virtual void ClearData() = 0;

    virtual bool DataExists() const = 0;

    template <typename VType = uint8_t>
    VType* Volume(int t, int c, bool auto_allocate = false);

    /// (1) can read storage: if auto_allocate is false, assert data_ vector non
    ///     empty, check volume index against image_loaded_view_
    ///     (or image_selected_view_ if image_loaded_view_ empty).
    ///     otherwise, call AllocateSelection and check volume index against
    ///     image_selected_view_
    /// (2) in memory or wrapper: assert wrapped_data_ or data_non empty, volume
    ///     index in range
    template <typename VType = uint8_t>
    VType* Volume(int volume_index = 0, bool auto_allocate = false);

    // assert wrapped_data_ or data_ non empty, loaded_view_ non empty,
    /// volume index in range of image_loaded_view_. will not allocate memory
    template <typename VType = uint8_t>
    const VType* ConstVolume(int volume_index = 0) const;

    /// assert wrapped_data_ or data_ non empty, use image_loaded_view_
    /// as target view if it is non empty, othewise use image_selected_view_
    /// target view must be non empty
    /// check z, c, t in range of target view
    template <typename VType = uint8_t>
    VType* Plane(int t, int c, int z, bool auto_allocate = false);

    /// the coordinates here are local to voxels in selected image view.
    /// first element in first array of the data_ vector is indexed (0, 0, 0, 0, 0)
    template <typename VType = uint8_t>
    void SetVoxel(int t, int c, int z, int y, int x, VType val);

    template <typename VType = uint8_t>
    VType& operator() (int t, int c, int z, int y, int x);

    template <typename VType = uint8_t>
    const VType& At(int t, int c, int z, int y, int x);

    // instance getters
    bool can_read_storage() const  { return can_read_storage_; }

    bool is_wrapper() const  { return is_wrapper_; }

    /// depends on image_info_
    const MImageInfo& image_info() const { return image_info_; }

    const std::vector<MPyrImageInfo>& image_pyr_infos() const  { return image_info_.pyr_infos(); }

    const MPyrImageInfo& image_pyr_info(int level = 0) const;

    int SelectedViewBytesPerVoxel() const { return selected_view_.BytesPerVoxel(); }

    int LoadedViewBytesPerVoxel() const { return loaded_view_.BytesPerVoxel(); }

    int StorageDataBytesPerVoxel() const  { return image_info().BytesPerVoxel(); }

    std::string image_root_dir() const { return image_info_.image_root_dir(); }

    const MImageView& selected_view() const { return selected_view_; }

    const MImageView& loaded_view() const { return loaded_view_; }

    int xdim(int pyr_level = 0) const { return image_info().xyz_dims(pyr_level)[2]; }

    int ydim(int pyr_level = 0) const { return image_info().xyz_dims(pyr_level)[1]; }

    int zdim(int pyr_level = 0) const { return image_info().xyz_dims(pyr_level)[0]; }

    std::vector<int> xyz_dims(int pyr_level = 0) const  { return image_info().xyz_dims(pyr_level); }

    int n_channels() const { return image_info().dims()[1]; }

    int n_times() const { return image_info().dims()[0]; }

    int n_volumes() const  { return n_channels() * n_times(); }

    std::vector<int32_t> dims(int pyr_level = 0) const { return image_info().dims(pyr_level); }

    int selected_pyr_level() const { return selected_view_.pyr_level(); }

    int loaded_pyr_level() const { return loaded_view_.pyr_level(); }

    int n_pyr_levels() const { return image_info_.n_pyr_levels(); }

    VoxelType voxel_type() const  { return loaded_view_.voxel_type(); }

protected:
    const bool can_read_storage_;  // can not modify
    bool is_wrapper_;
    std::vector<std::unique_ptr<uint8_t []>> data_;
    std::vector<uint8_t*> wrapped_data_;
    std::vector<std::vector<std::unique_ptr<uint8_t []>>> older_data_;
    MFolderSlices image_folder_slices_;
    MImageInfo image_info_;
    /// these views are in the global context
    /// for in memory MImage instances, the selected and loaded global views
    /// should always be the entire memory content
    MImageView selected_view_, loaded_view_;

private:
    /// copy data in other's view
    void CopyData(const MImageBase &other, const std::string& mode = std::string());

    virtual void ValidateVolumeData(int c, int t, bool auto_allocate) = 0;

    virtual void ValidateVolumeData(int volume_index, bool auto_allocate) = 0;

    long LoadedVoxelAddress(int z, int y, int x) const;
};

}

template <typename VType>
VType* mcp3d::MImageBase::Volume(int t, int c, bool auto_allocate)
{
    ValidateVolumeData(t, c, auto_allocate);
    long index = mcp3d::LinearAddressRobust(
            {loaded_view().n_times(), loaded_view().n_channels()}, t, c);
    if (!is_wrapper_)
        return reinterpret_cast<VType*>(data_[index].get());
    else
        return reinterpret_cast<VType*>(wrapped_data_[index]);
}

template <typename VType>
VType* mcp3d::MImageBase::Volume(int volume_index, bool auto_allocate)
{
    ValidateVolumeData(volume_index, auto_allocate);
    if (!is_wrapper_)
        return reinterpret_cast<VType*>(data_[volume_index].get());
    else
        return reinterpret_cast<VType*>(wrapped_data_[volume_index]);
}

template <typename VType>
const VType* mcp3d::MImageBase::ConstVolume(int volume_index) const
{
    MCP3D_ASSERT(DataExists())
    MCP3D_ASSERT(!loaded_view_.empty())
    MCP3D_ASSERT(volume_index >= 0 && volume_index < loaded_view_.n_volumes())
    if (!is_wrapper_)
        return reinterpret_cast<const VType*>(data_[volume_index].get());
    else
        return reinterpret_cast<const VType*>(wrapped_data_[volume_index]);
}

template <typename VType>
VType* mcp3d::MImageBase::Plane(int t, int c, int z, bool auto_allocate)
{
    const mcp3d::MImageView& view = loaded_view_.empty() ?
                                    selected_view_ : loaded_view_;
    MCP3D_ASSERT(!view.empty())
    MCP3D_ASSERT(z >= 0 && z < view.view_zdim() &&
                 c >= 0 && c < view.n_channels() &&
                 t >= 0 && t < view.n_times())
    int n_loaded_volumes = is_wrapper_ ? (int)wrapped_data_.size() : (int)data_.size();
    MCP3D_ASSERT(n_loaded_volumes == view.n_channels() * view.n_times())
    long volume_index = mcp3d::LinearAddressRobust({view.n_times(), view.n_channels()}, t, c);
    return reinterpret_cast<VType*>(Volume((int) volume_index, auto_allocate) +
                                    view.BytesPerPlane() * (long)z);
}

template <typename VType>
void mcp3d::MImageBase::SetVoxel(int t, int c, int z, int y, int x, VType val)
{
    Volume<VType>(t, c)[LoadedVoxelAddress(z, y, x)] = val;
}

template <typename VType>
VType& mcp3d::MImageBase::operator() (int t, int c, int z, int y, int x)
{
    return Volume<VType>(t, c)[LoadedVoxelAddress(z, y, x)];
}

template <typename VType>
const VType& mcp3d::MImageBase::At(int t, int c, int z, int y, int x)
{
    return Volume<VType>(t, c)[LoadedVoxelAddress(z, y, x)];
}

#endif //MCP3D_MCP3D_IMAGE_BASE_HPP
