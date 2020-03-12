//
// Created by muyezhu on 2/26/18.
//

#ifndef MCP3D_MCP3D_IMAGE_REGION_HPP
#define MCP3D_MCP3D_IMAGE_REGION_HPP

#include "mcp3d_image_info.hpp"

namespace mcp3d
{

class MImageBlock
{
public:
    /// offsets_, extents_ and strides_ contain 3 dimensions, in order of zyx
    /// if values missing from constructor parameter, offsets_ and extents_
    /// prepend 0, strides_ prepend 1.
    /// channel_ is same length as input when input
    /// is not empty. otherwise, if any dimension of extents equal to 0,
    /// n_channel_ is set to zero, else n_channel_ = 1 and channels[0] = 0
    /// times_ is a single 0 at the moment if n_channels_ > 0, else n_times_ = 0
    /// channels_ will always hold max_channel_number_ integers for correct code
    /// behavior if introducing more channels from copied instance or within
    /// view selection. similarly times_ will always hold max_time_numer_
    explicit MImageBlock(const std::vector<int>& offsets = std::vector<int>(),
                         const std::vector<int>& extents = std::vector<int>(),
                         const std::vector<int>& strides = std::vector<int>(),
                         const std::vector<int>& channels = std::vector<int>(),
                         const std::vector<int>& times = std::vector<int>());

    MImageBlock(const MImageBlock& other);

    MImageBlock& operator=(const MImageBlock& other);

    bool operator== (const MImageBlock& other) const;

    bool empty() const  { return AnyZeros(extents()); }

    void Clear();

    std::vector<int> offsets() const  { return std::vector<int>({offsets_[0], offsets_[1], offsets_[2]}); }

    std::vector<int> extents() const  { return std::vector<int>({extents_[0], extents_[1], extents_[2]}); }

    std::vector<int> strides() const  { return std::vector<int>({strides_[0], strides_[1], strides_[2]}); }

    std::vector<int> channels() const;

    std::vector<int> times() const;

    int* offsets_ptr()  { return offsets_.get(); }

    int* extents_ptr()  { return extents_.get(); }

    int* strides_ptr()  { return strides_.get(); }

    int* channels_ptr()  { return channels_.get(); }

    int* times_ptr()  { return times_.get(); }

    void PrintView() const;

    friend class MImageView;

private:
    void CopyOther(const mcp3d::MImageBlock& other);
    std::unique_ptr<int[]> offsets_, extents_, strides_, channels_, times_;
    int n_channels_, n_times_, max_channel_number_, max_time_number_;
};

class MImageView
{
    /// manages the view into an image volume through the image pyramids
    /// offsets, extents and strides apply to zyx dimensions
    /// channels and times are vectors of selected channels and time points,
    /// does not need to be continuous in value
public:
    explicit MImageView(const MImageInfo& img_info = MImageInfo{}):
                                            image_info_(img_info),
                                            global_image_block_{},
                                            voxel_type_(img_info.voxel_type(0)),
                                            pyr_level_(-1),
                                            interpret_block_as_local_(false),
                                            view_level_offsets_(3, 0),
                                            view_level_extents_(3, 0),
                                            view_level_strides_(3, 1) {}


    bool operator== (const MImageView& other) const;

    bool operator!= (const MImageView& other) const { return ! ((*this) == other); }

    /// const MImageBlock& image_block provides offsets, extents, strides in
    /// the global or local image coordinate system, depending on value of
    /// interpret_view_as_local. the offsets and extents define a rectangular
    /// volume within which voxels wll be retrieved. the strides parameter further
    /// determines if the voxels will be retrieved in strided manner
    /// if the view is global, MImageRegion fills in
    /// default values for image_block if needed, and copy construct
    /// global_image_view_ data member from image_block.
    /// MImageRegion's own view_level_offsets_, view_level_extents_, view_level_strides_
    /// are then calculated from the global coordinate system at requested pyramid
    /// level by scaling xyz according to pyramid ratio of the pyramid level and
    /// format of the image.
    /// therefore retrieve image at given pyramid level with view_level_offsets_,
    /// view_level_extents_, view_level_strides_ refers to the same level 0 image
    /// area identified by global_image_view.
    /// if image_block has local pyramid level interpretation, global view is produced
    /// from image_block in similar but reverse process
    /// in both global view and pyramid level view,
    /// default values for all offsets along all axes is 0
    /// default values for all extents along all axes is from offset till
    /// last element along the given axis. offsets + extent can exceed axis
    /// length. extents with zero value will be changed to default since empty
    /// view selection is not meaningful. the out of range portion of data will
    /// be padded with background pixels. but value of offset in global image block
    /// must be within axis range (at other view level the offsets are allowed to
    /// run out of bounds due to earlier level boundary voxel may not have correspondence
    /// in later levels)
    /// default values for all strides along all axes is 1
    void SelectView(const MImageBlock &image_block, int pyr_level = 0,
                    bool interpret_block_as_local = false,
                    VoxelType voxel_type = VoxelType::UNKNOWN);

    bool OutOfPyrImageBoundary() const;

    /// if ViewEntirelyOutOfPyrImageBoundary() is true, return false
    bool PartiallyOutOfPyrImageBoundary() const;

    void Clear();

    // this function returns true if the views will retrieve identical data
    // provided they are within the same global volume. does not assert equality
    // on _image_info
    bool SameView(const MImageView& other);

    /// if image_info is currently empty, copy image info as well
    /// if image_info is not empty, it must be equal in dimensions to
    /// other.image_info
    void CopyView(const MImageView &other);

    void PrintView() const;

    double ViewMemorySize(const std::string &unit) const;

    const MImageBlock& global_image_block() const { return global_image_block_; }

    const MImageInfo& image_info() const  { return image_info_; }

    int pyr_level() const { return pyr_level_; }

    bool interpret_block_as_local() const  { return interpret_block_as_local_; }

    const std::vector<int>& view_level_offsets() const  { return view_level_offsets_; }

    /// the extents covered by the selected block, may not be the dimension of
    /// retrieved data due to striding, equal to view_xyz_dims() if strides
    /// are all 1
    const std::vector<int>& view_level_extents() const  { return view_level_extents_; }

    const std::vector<int>& view_level_strides() const  { return view_level_strides_; }

    const std::vector<int> view_channels() const { return global_image_block_.channels(); }

    const std::vector<int> view_times() const { return global_image_block_.times(); }

    bool empty() const { return AnyZeros(view_level_extents_); }

    bool view_is_unit_strided(const std::string& axes = "xyz") const;

    int view_xdim() const;

    int view_ydim() const;

    int view_zdim() const;

    int n_channels() const;

    int n_times() const;

    /// number of xyz volumes in view
    int n_volumes() const;

    /// xyz dimensions of data in the selected block, accounting for striding
    std::vector<int> view_xyz_dims() const;

    /// voxel dimensions of selected or loaded image block.
    std::vector<int> view_dims() const;

    /// voxel dimensions of global image volume, returned from image_info_
    std::vector<int> global_image_dims() const;

    std::vector<int> view_level_image_dims() const;

    std::vector<int> view_level_image_xyz_dims() const;

    /// number of voxels in an xyz volume
    long VoxelsPerVolume() const;

    /// number of voxels in an xy plane
    long VoxelsPerPlane() const;

    int BytesPerVoxel() const;

    long BytesPerVolume() const;

    long BytesPerPlane() const;

    VoxelType voxel_type() const   { return voxel_type_; }

    friend class MImageBase;
    friend class MImage;
    friend class MImageIO;
private:
    void set_voxel_type(VoxelType voxel_type)   { voxel_type_ = voxel_type; }

    void set_image_info(const MImageInfo& image_info);

    MImageInfo image_info_;
    MImageBlock global_image_block_;
    VoxelType voxel_type_;
    int pyr_level_;
    bool interpret_block_as_local_;
    std::vector<int> view_level_offsets_, view_level_extents_, view_level_strides_;
};

}

#endif //MCP3D_MCP3D_IMAGE_REGION_HPP
