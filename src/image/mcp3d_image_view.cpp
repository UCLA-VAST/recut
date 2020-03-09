//
// Created by muyezhu on 2/26/18.
//
#include <algorithm>
#include "common/mcp3d_common.hpp"
#include "mcp3d_image_view.hpp"
#include "mcp3d_image_common.hpp"

using namespace std;

mcp3d::MImageBlock::MImageBlock(const vector<int> &offsets,
                                const vector<int> &extents,
                                const vector<int> &strides,
                                const vector<int> &channels,
                                const vector<int> &times):
                                max_channel_number_(mcp3d::MAX_CHANNEL_NUMBER),
                                max_time_number_(mcp3d::MAX_TIME_POINT_NUMBER)
{
    MCP3D_ASSERT(offsets.size() <= (size_t)3 &&
                 extents.size() <= (size_t)3 &&
                 strides.size() <= (size_t)3)
    if (! offsets.empty() && ! AllNonNegative(offsets))
        MCP3D_OUT_OF_RANGE("negative offsets encountered")
    if (! extents.empty() && ! AllNonNegative(extents))
        MCP3D_OUT_OF_RANGE("negative extents encountered")
    if (! strides.empty() && !AllPositive(strides))
        MCP3D_OUT_OF_RANGE("non positive strides encountered")
    if (! channels.empty() && ! AllNonNegative(channels))
        MCP3D_OUT_OF_RANGE("negative channels encountered")
    if (! times.empty() && ! AllNonNegative(times))
        MCP3D_OUT_OF_RANGE("negative time points encountered")

    offsets_ = make_unique<int[]>(3);
    for (size_t i = 0; i < 3; ++i)
    {
        size_t offset_pos = i + offsets.size() - 3;
        if (offset_pos >= 0 and offset_pos < offsets.size())
            offsets_[i] = offsets[offset_pos];
        else
            offsets_[i] = 0;
    }
    extents_ = make_unique<int[]>(3);
    for (size_t i = 0; i < 3; ++i)
    {
        size_t extent_pos = i + extents.size() - 3;
        if (extent_pos >= 0 and extent_pos < extents.size())
            extents_[i] = extents[extent_pos];
        else
            extents_[i] = 0;
    }
    strides_ = make_unique<int[]>(3);
    for (size_t i = 0; i < 3; ++i)
    {
        size_t stride_pos = i + strides.size() - 3;
        if (stride_pos >= 0 and stride_pos < strides.size())
            strides_[i] = strides[stride_pos];
        else
            strides_[i] = 1;
    }

    channels_ = make_unique<int[]>((size_t)max_channel_number_);
    if (!channels.empty())
    {
        for (size_t i = 0; i < channels.size(); ++i)
        {
            if (channels[i] >= max_channel_number_)
                MCP3D_OUT_OF_RANGE("channel number exceeds maximum possible value")
            channels_[i] = channels[i];

        }
        n_channels_ = (int)(channels.size());
    }
    else
    {
        if (extents_[0] == 0 || extents_[1] == 0 || extents_[2] == 0)
            n_channels_ = 0;
        else
        {
            n_channels_ = 1;
            channels_[0] = 0;
        }
    }
    times_ = make_unique<int[]>((size_t)max_time_number_);
    if (n_channels_ == 0)
        n_times_ = 0;
    else
    {
        n_times_ = 1;
        times_[0] = 0;
    }
}

mcp3d::MImageBlock::MImageBlock(const mcp3d::MImageBlock& other)
{
    CopyOther(other);
}

mcp3d::MImageBlock& mcp3d::MImageBlock::operator=(const mcp3d::MImageBlock& other)
{
    if (this != &other)
        CopyOther(other);
    return *this;
}

void mcp3d::MImageBlock::CopyOther(const mcp3d::MImageBlock &other)
{
    max_channel_number_ = other.max_channel_number_;
    max_time_number_ = other.max_time_number_;
    offsets_ = make_unique<int[]>(3);
    for (int i = 0; i < 3; ++i)
        offsets_[i] = other.offsets_[i];
    extents_ = make_unique<int[]>(3);
    for (int i = 0; i < 3; ++i)
        extents_[i] = other.extents_[i];
    strides_ = make_unique<int[]>(3);
    for (int i = 0; i < 3; ++i)
        strides_[i] = other.strides_[i];
    n_channels_ = other.n_channels_;
    channels_ = make_unique<int[]>((size_t)max_channel_number_);
    for (int i = 0; i < n_channels_; ++i)
        channels_[i] = other.channels_[i];
    n_times_ = other.n_times_;
    times_ = make_unique<int[]>((size_t)max_time_number_);
    for (int i = 0; i < n_times_; ++i)
        times_[i] = other.times_[i];
}

vector<int> mcp3d::MImageBlock::channels() const
{
    vector<int> channels;
    for (int i = 0; i < n_channels_; ++i)
        channels.push_back(channels_[i]);
    return channels;
}

vector<int> mcp3d::MImageBlock::times() const
{
    vector<int> times;
    if (n_times_ == 1)
        times.push_back(0);
    return times;
}

bool mcp3d::MImageBlock::operator== (const mcp3d::MImageBlock& other) const
{
    for (int i = 0; i < 3; ++i)
        if (offsets_[i] != other.offsets_[i])
            return false;
    for (int i = 0; i < 3; ++i)
        if (extents_[i] != other.extents_[i])
            return false;
    for (int i = 0; i < 3; ++i)
        if (strides_[i] != other.strides_[i])
            return false;
    if (max_channel_number_ != other.max_channel_number_)
        return false;
    if (n_channels_ != other.n_channels_)
        return false;
    for (int i = 0; i < n_channels_; ++i)
        if (channels_[i] != other.channels_[i])
            return false;
    if (max_time_number_ != other.max_time_number_)
        return false;
    if (n_times_ != other.n_times_)
        return false;
    return times_[0] == other.times_[0];
}

void mcp3d::MImageBlock::Clear()
{
    for (int i = 0; i < 3; ++i)
    {
        offsets_[i] = 0;
        extents_[i] = 0;
        strides_[i] = 1;
    }
    n_channels_ = 0;
    n_times_ = 0;
}

void mcp3d::MImageBlock::PrintView() const
{
    cout << "z axis range: [" << offsets_[0] << ", " << offsets_[0] + extents_[0] << "], striding = " << strides_[0] << endl;
    cout << "y axis range: [" << offsets_[1] << ", " << offsets_[1] + extents_[1] << "], striding = " << strides_[1]  << endl;
    cout << "x axis range: [" << offsets_[2] << ", " << offsets_[2] + extents_[2] << "], striding = " << strides_[2]  << endl;
    cout << "channels selected: " << mcp3d::JoinArray<int>(channels_, n_channels_, ", ", true) << endl;
    cout << "time points selected: [0]" << endl;
}

bool mcp3d::MImageView::operator==(const mcp3d::MImageView &other) const
{
    return view_level_offsets_ == other.view_level_offsets() &&
           view_level_extents_ == other.view_level_extents() &&
           view_level_strides_ == other.view_level_strides() &&
           view_channels() == other.view_channels() &&
           view_times() == other.view_times() &&
           pyr_level_ == other.pyr_level() &&
           voxel_type_ == other.voxel_type() &&
           interpret_block_as_local_ == other.interpret_block_as_local_ &&
           image_info_ == other.image_info_;
}

void mcp3d::MImageView::SelectView(const MImageBlock &image_block,
                                   int pyr_level,
                                   bool interpret_block_as_local,
                                   VoxelType voxel_type)
{
    MCP3D_ASSERT(!image_info_.empty())
    MCP3D_ASSERT(pyr_level >= 0)
    pyr_level_ = pyr_level;
    if (pyr_level_ >= image_info_.n_pyr_levels())
        MCP3D_OUT_OF_RANGE("pyramid level exceeds maximum available")
    interpret_block_as_local_ = interpret_block_as_local;
    if (voxel_type == mcp3d::VoxelType::UNKNOWN)
        set_voxel_type(image_info_.pyr_infos()[pyr_level].voxel_type());
    else
        set_voxel_type(voxel_type);

    global_image_block_.n_channels_ = image_block.n_channels_;
    global_image_block_.n_times_ = image_block.n_times_;
    if (global_image_block_.n_channels_ == 0)
    {
        global_image_block_.n_channels_ = 1;
        global_image_block_.channels_[0] = 0;
    }
    else
        for (int i = 0; i < image_block.n_channels_; ++i)
            global_image_block_.channels_[i] = image_block.channels_[i];
    if (global_image_block_.n_times_ == 0)
    {
        global_image_block_.n_times_ = 1;
        global_image_block_.times_[0] = 0;
    }

    if (interpret_block_as_local)  // image block is at pyr_level
    {
        for (int i = 0; i < 3; ++i)
        {
            view_level_offsets_[i] = image_block.offsets_[i];
            if (image_block.extents_[i] == 0)
                view_level_extents_[i] = image_info_.xyz_dims(pyr_level)[i] -
                                         view_level_offsets_[i];
            else
                view_level_extents_[i] = image_block.extents_[i];
            view_level_strides_[i] = image_block.strides_[i];
        }
        for (int i = 0; i < 3; ++i)
        {
            int upscale_factor = i == 0 ?
                                 image_info_.pyr_z_ratios()[pyr_level] :
                                 image_info_.pyr_xy_ratios()[pyr_level];
            global_image_block_.offsets_[i] = view_level_offsets_[i] * upscale_factor;
            global_image_block_.extents_[i] = view_level_extents_[i] * upscale_factor;
            // not scaling up strides unless strides is greater than one.
            // there's multiple possible level 0 strides that can result in
            // current level stride equal to 1
            global_image_block_.strides_[i] = view_level_strides_[i] == 1 ?
                                              1 : view_level_strides_[i] * upscale_factor;
        }
    }
    else // image block is at level 0
    {
        for (int i = 0; i < 3; ++i)
        {
            global_image_block_.offsets_[i] = image_block.offsets_[i];
            if (image_block.extents_[i] == 0)
                global_image_block_.extents_[i] = image_info_.pyr_infos()[0].xyz_dims()[i] -
                                                  global_image_block_.offsets_[i];
            else
                global_image_block_.extents_[i] = image_block.extents_[i];
            global_image_block_.strides_[i] = image_block.strides_[i];
        }

        for (int i = 0; i < 3; ++i)
        {
            int downscale_factor = i == 0 ?
                                   image_info_.pyr_z_ratios()[pyr_level] :
                                   image_info_.pyr_xy_ratios()[pyr_level];
            view_level_offsets_[i] = global_image_block_.offsets_[i] / downscale_factor;
            view_level_extents_[i] = max(1, global_image_block_.extents_[i] / downscale_factor);
            view_level_strides_[i] = max(1, global_image_block_.strides_[i] / downscale_factor);
        }
    }

    // range validation for global block for xyz
    for (int i = 0; i < 3; ++i)
    {
        if (global_image_block_.offsets_ptr()[i] >= image_info_.xyz_dims()[i])
            MCP3D_OUT_OF_RANGE("global image block: index " +
                               to_string(global_image_block_.offsets_ptr()[i]) +
                               " is out of bounds for axis " + to_string(i) +
                               " with size " + to_string(image_info_.xyz_dims()[i]))
    }
    MCP3D_ASSERT(view_level_offsets_.size() == 3 && view_level_extents_.size() == 3 && view_level_strides_.size() == 3)
    if (! AllPositive(view_level_strides_))
        MCP3D_OUT_OF_RANGE("non positive strides encountered: " + JoinVector(view_level_strides_, ", "))
}

bool mcp3d::MImageView::OutOfPyrImageBoundary() const
{
    // if any of the offsets values are out of level image bound, the view
    // is out of boundary
    return !mcp3d::AllGreater(view_level_image_xyz_dims(),
                              view_level_offsets_);
}

bool mcp3d::MImageView::PartiallyOutOfPyrImageBoundary() const
{
    // if view entirely out of boundary, return false
    if (OutOfPyrImageBoundary())
        return false;
    vector<int> n_in_boundary_voxels = mcp3d::XyzDimsWithStrides(
            mcp3d::SubtractSeq<int>(view_level_image_xyz_dims(),
                                    view_level_offsets_),
            view_level_strides_);
    vector<int> in_boundary_full_stride_extents = (n_in_boundary_voxels - 1) *
                                                   view_level_strides_ + 1;
    vector<int> remaining_extents = view_level_extents_ -
                                    in_boundary_full_stride_extents;
    return !mcp3d::AllGreater(view_level_strides_, remaining_extents);
}

void mcp3d::MImageView::Clear()
{
    global_image_block_.Clear();
    for (int i = 0; i < 3; ++i)
    {
        view_level_offsets_[i] = 0;
        view_level_extents_[i] = 0;
        view_level_strides_[i] = 1;
    }
    pyr_level_ = 0;
    voxel_type_ = mcp3d::VoxelType::UNKNOWN;
}

bool mcp3d::MImageView::SameView(const MImageView &other)
{
    return view_level_offsets_ == other.view_level_offsets() &&
           view_xyz_dims() == other.view_xyz_dims() &&
           view_level_strides_ == other.view_level_strides() &&
           view_channels() == other.view_channels() &&
           view_times() == other.view_times() &&
           pyr_level_ == other.pyr_level() &&
           voxel_type_ == other.voxel_type();
}

void mcp3d::MImageView::CopyView(const MImageView &other)
{
    if (other.empty())
        MCP3D_INVALID_ARGUMENT("can not copy view selection from empty view selection")
    if (image_info_.empty())
        image_info_ = other.image_info_;
    else
        MCP3D_ASSERT(global_image_dims() == other.global_image_dims())
    for (int i = 0; i < 3; ++i)
    {
        view_level_offsets_[i] = other.view_level_offsets()[i];
        view_level_extents_[i] = other.view_level_extents()[i];
        view_level_strides_[i] = other.view_level_strides()[i];
    }
    global_image_block_ = other.global_image_block_;
    pyr_level_ = other.pyr_level();
    voxel_type_ = other.voxel_type();
}

void mcp3d::MImageView::PrintView() const
{
    if (global_image_block_.extents_[0] == 0)
    {
        cout << "global image view uninitialized" << endl;
        return;
    }
    cout << "global image view: pyramid level = 0" << endl;
    global_image_block_.PrintView();
    cout << "pyramid image view: pyramid level = " << pyr_level_ << endl;
    cout << "z axis range: [" << view_level_offsets_[0] << ", " << view_level_offsets_[0] + view_level_extents_[0] << "], striding = " << view_level_strides_[0] << endl;
    cout << "y axis range: [" << view_level_offsets_[1] << ", " << view_level_offsets_[1] + view_level_extents_[1] << "], striding = " << view_level_strides_[1]  << endl;
    cout << "x axis range: [" << view_level_offsets_[2] << ", " << view_level_offsets_[2] + view_level_extents_[2] << "], striding = " << view_level_strides_[2]  << endl;
    cout << "channels selected: " << mcp3d::JoinVector<int>(view_channels(), ", ") << endl;
    cout << "time points selected: " << mcp3d::JoinVector<int>(view_times(), ", ") << endl;
    cout << "equal to " << ViewMemorySize("GB") << " GB of memory" << endl;
}

double mcp3d::MImageView::ViewMemorySize(const std::string &unit) const
{
    return mcp3d::MemorySize(BytesPerVolume(), unit);
}

int mcp3d::MImageView::view_xdim() const
{
    if (empty())
        return 0;
    return view_xyz_dims()[2];
}

int mcp3d::MImageView::view_ydim() const
{
    if (empty())
        return 0;
    return view_xyz_dims()[1];
}

int mcp3d::MImageView::view_zdim() const
{
    if (empty())
        return 0;
    return view_xyz_dims()[0];
}

bool mcp3d::MImageView::view_is_unit_strided(const string& axes) const
{
    int strides = 1;
    string axes_ = mcp3d::StringLower(axes);
    for (const auto& axis: axes_)
    {
        if (axis == 'x')
            strides *= view_level_strides_[2];
        else if (axis == 'y')
            strides *= view_level_strides_[1];
        else if (axis == 'z')
            strides *= view_level_strides_[0];
    }
    return strides == 1;
}

int mcp3d::MImageView::n_channels() const
{
    if (empty())
        return 0;
    return (int) view_channels().size();
}

int mcp3d::MImageView::n_times() const
{
    if (empty())
        return 0;
    return (int) view_times().size();
}

vector<int> mcp3d::MImageView::view_xyz_dims() const
{
    if (empty())
        return vector<int>(3, 0);
    else
        return mcp3d::XyzDimsWithStrides(view_level_extents_, view_level_strides_);
}

vector<int> mcp3d::MImageView::view_dims() const
{
    return vector<int>({n_times(), n_channels(), view_zdim(), view_ydim(), view_xdim()});
}

vector<int> mcp3d::MImageView::global_image_dims() const
{
    return image_info_.dims(0);
}

vector<int> mcp3d::MImageView::view_level_image_dims() const
{
    if (empty())
        return vector<int>(5, 0);
    return image_info_.dims(pyr_level_);
}

vector<int> mcp3d::MImageView::view_level_image_xyz_dims() const
{
    if (empty())
        return vector<int>(3, 0);
    return image_info_.xyz_dims(pyr_level_);
}

int mcp3d::MImageView::n_volumes() const
{
    if (empty())
        return 0;
    return global_image_block_.n_channels_ * global_image_block_.n_times_;
}

long mcp3d::MImageView::VoxelsPerVolume() const
{
    if (empty())
        return 0;
    return (long) view_xdim() * (long) view_ydim() * (long) view_zdim();
}

long mcp3d::MImageView::VoxelsPerPlane() const
{
    if (empty())
        return 0;
    return (long) view_xdim() * (long) view_ydim();
}

int mcp3d::MImageView::BytesPerVoxel() const
{
    if (empty())
        return 0;
    return mcp3d::BytesPerVoxelElement(voxel_type());
}

long mcp3d::MImageView::BytesPerVolume() const
{
    return VoxelsPerVolume() * (long)BytesPerVoxel();
}

long mcp3d::MImageView::BytesPerPlane() const
{
    return VoxelsPerPlane() * (long)BytesPerVoxel();
}

void mcp3d::MImageView::set_image_info(const MImageInfo &image_info)
{
    image_info_= image_info;
    Clear();
}


