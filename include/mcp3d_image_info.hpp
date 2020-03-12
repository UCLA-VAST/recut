//
// Created by muyezhu on 2/11/18.
//

#ifndef MCP3D_MCP3D_IMAGE_CONFIG_HPP
#define MCP3D_MCP3D_IMAGE_CONFIG_HPP

#include <unordered_set>
#include <nlohmann/json.hpp>
#include "common/mcp3d_common.hpp"
#include "mcp3d_voxel_types.hpp"
#include "mcp3d_file_formats.hpp"

/* Given a directory containing a image volume and its downsampled pyramids,
 * organized as following:
 * image_root_dir:
 *    (level 0 images / pyr_level_0): level 0 images can be under pyr_level_0
 *                                    or scattered
 *    pyr_level_1
 *    pyr_level_2
 *    ...
 *    pyr_level_n:
 * MImageInfo describe all image data under image_root_dir
 * MPyrImageInfo describe all image data at a given pyramid level
 *
 * MPyrImageInfo required json entries:
 *    format: FileFormats enum string
 *    x dimension: total size of image along x dimension across the entire image
 *                 volume at current pyramid level
 *    channels: number of channels
 *    time points: number of time points
 *    xchunk dimension: a chunk is the unit of storage for image data.
 *                      ome-tif and hdf5 may save volume tiles.
 *                      tif image sequence's unit of storage is a single tiff
 *                      image plane
 *                      within this storage unit, the size along x dimension is
 *                      xchunk dimension
 *    dimensions order: order of image dimensions on disk storage. default tczyx
 *    level 0 image: whether the current pyramid level is at level 0 of the entire
 *                   image volume. this is simply the image with greatest xyz
 *                   dimensions under image root directory. level 0 images in
 *                   different processing stage of the project_structure may have different
 *                   xyz dimensions, as some algorithms may operate on different
 *                   resolutions
 *    image pyramid directory: directory where current image pyramid files are
 *                             located. if not collected under a MImageInfo
 *                             instance, this directory is a full path
 *    image sequences: base name of image chunks sorted by xyzct order.
 *                     to recove full path of an image in standalone MPyrImageInfo:
 *                     image pyramid directory + image basename
 *    voxel type: VoxelType enum string
 *    pyramid ratio: factor of downsample from level 0 image. between level i
 *                   and level i + 1, the downsample factor is strictly 2. other
 *                   factors are invalid
 *                   f_i = x_i / x_0 = y_i / y_0
 *    description: optional string, can contain arbitary content
 *
 * MImageInfo: collection of MPyrImageInfo, sorted by ascending pyramid level
 *             find the nearest common directory from image pyramid directories,
 *             assign it as image root dir.
 */
namespace mcp3d
{
class MImagePyrInfoEntries
{
public:
    MImagePyrInfoEntries(): required_entries_({}), optional_entries_({})
    {
        required_entries_.insert(
                {"format", "x dimension", "y dimension", "z dimension",
                 "channels", "time points", "level 0 image",
                 "xchunk dimension", "ychunk dimension", "zchunk dimension",
                 "dimensions order", "image pyramid directory",
                 "image sequence", "voxel type"});
        optional_entries_.insert({"z dimension scaled", "description"});
    }

    const std::unordered_set<std::string>& required_entries() const { return required_entries_; }
private:
    std::unordered_set<std::string> required_entries_;
    std::unordered_set<std::string> optional_entries_;
};

/// entire volume at a given pyramid level
class MPyrImageInfo
{
public:
    using json = nlohmann::json;

    MPyrImageInfo()   { SetEmptyPyrInfo(); }

    explicit MPyrImageInfo(const std::string &json_path,
                           FileFormats expected_format = FileFormats::UNKNOWN);

    explicit MPyrImageInfo(const json &image_pyr_info,
                           FileFormats expected_format = FileFormats::UNKNOWN);

    MPyrImageInfo(const MPyrImageInfo& other);

    /// does not compare the description_ string
    bool operator==(const MPyrImageInfo& other) const;

    bool operator!=(const MPyrImageInfo& other) const  { return !(*this == other); }

    FileFormats pyr_image_format() const { return pyr_file_format_; }

    bool is_level0_image() const  { return is_level0_image_; }

    bool zdim_scaled() const  { return zdim_scaled_; }

    /// # of dimensions > 1
    int rank() const  { return rank_; }

    int xdim() const  { return xdim_; }

    int ydim() const  { return ydim_; }

    int zdim() const  { return zdim_; }

    int n_channels() const { return n_channels_; }

    int n_times() const { return n_times_; }

    std::vector<int> xyz_dims() const  { return std::vector<int>({zdim_, ydim_, xdim_}); }

    std::vector<int> dims() const  { return std::vector<int>({n_times_, n_channels_, zdim_, ydim_, xdim_}); }

    VoxelType voxel_type() const  { return pyr_voxel_type_; }

    int xchunk_dim() const  { return xchunk_dim_; }

    int ychunk_dim() const  { return ychunk_dim_; }

    int zchunk_dim() const  { return zchunk_dim_; }

    std::vector<int> chunk_xyz_dims() const

    { return std::vector<int>({zchunk_dim_, ychunk_dim_, xchunk_dim_}); }

    int n_xchunks() const  { return n_xchunks_; }

    int n_ychunks() const  { return n_ychunks_; }

    int n_zchunks() const  { return n_zchunks_; }

    int n_chunks()  const  { return n_chunks_; }

    std::string pyr_level_dir() const  { return pyr_level_dir_; };

    const std::string &dims_order() const { return dims_order_; }

    const std::vector<std::string>& image_sequence() const { return image_names_; }

    const json& pyr_image_info() const { return pyr_image_info_; }

    json& pyr_image_info() { return pyr_image_info_; }

    static const std::string in_memory_str()  { return "__in_memory__"; }

    friend class MImageInfo;
private:
    /// build MImagePyrInfo instance for MImage instances that do not have
    /// storage data (aka wrapper MImage or simple data container created in memory)
    /// this constructor should only be called through MImageInfo
    explicit MPyrImageInfo(const std::vector<int>& dimensions,
                           VoxelType voxel_type);

    void SetEmptyPyrInfo();

    void RequiredEntriesComplete();

    void LoadJson(const std::string &json_path);

    void ParseJson(FileFormats expected_format = FileFormats::UNKNOWN);

    json pyr_image_info_;
    FileFormats pyr_file_format_;
    bool is_level0_image_, zdim_scaled_;
    int rank_;
    int xdim_, ydim_, zdim_, n_channels_, n_times_;
    int xchunk_dim_, ychunk_dim_, zchunk_dim_;
    int n_xchunks_, n_ychunks_, n_zchunks_, n_chunks_;
    //xyzct or xyczt, refer to the dimension of image files on disk
    std::string dims_order_;
    std::string pyr_level_dir_;
    std::vector<std::string> image_names_;
    VoxelType pyr_voxel_type_;
    std::string description_;
};

class MImageInfo
{
public:
    MImageInfo() = default;

    explicit MImageInfo(const std::string& image_info_path);

    /// rejects __from_memory__ pyr_info
    explicit MImageInfo(MPyrImageInfo& pyr_info);

    /// rejects __from_memory__ pyr_info
    explicit MImageInfo(std::vector<MPyrImageInfo>& pyr_infos);

    /// calls MImagePyrInfo(const std::vector<int>& dimensions, VoxelType voxel_type)
    MImageInfo(const std::vector<int>& dimensions, VoxelType voxel_type);

    /// rejects __from_memory__ pyr_info
    void AddPyrInfos(std::vector<MPyrImageInfo>& pyr_infos);

    /// rejects __from_memory__ pyr_info
    void AddPyrInfo(MPyrImageInfo& pyr_info);

    bool operator==(const MImageInfo& other) const;

    MImageInfo& operator+= (MImageInfo& rhs);

    MImageInfo& operator+= (MImageInfo&& rhs);

    /// asserts not __from_memory__
    bool AllPathsExist() const;

    /// asserts not __from_memory__
    bool LevelPathsExist(int pyr_level) const;

    /// if image info is empty, return immediately
    /// if image_roof_dir is equal to MImagePyrInfo::from_memory_str, assert
    /// one and only one level exist, and return. otherwise:
    /// (1) image root dir must exist
    /// (2) total levels must equal to total number of MImagePyrInfo objects.
    ///     the first MImagePyrInfo in the pyr_infos_ vector must be level 0 image
    /// (3) number of level 0 pyramid must equal to 1.
    /// (4) pyramid ratios must be geometric series of 2^i, i = {0, 1, 2...}
    /// (5) number of composite pyramid directory must be at most one. if there
    ///     is a composite directory, it must not be any more pyramid level
    ///     directories after it
    /// (6) pyramid level dir name must be consistent with its pyramid level.
    ///     if a MImagePyrInfo object is at position 1 of the pyr_infos_ vector,
    ///     its pyramid level directory should be pyr_level_01[-nn]. if a composite
    ///     pyramid level encountered, check no later MImagePyrInfo object
    void ValidatePyramidsStructure() const;

    /// high level sequence of writing operations should be such that images
    /// are written first, then image info. therefore no matter if MImage was
    /// constructed from storage, at image info saving all image paths must
    /// exist. image writing operation should update image info. image info
    /// writing is synchronized across threads and done only from rank 0 thread
    void Save();

    void Clear();

    int BytesPerVoxel() const;

    /// construct full image path to image at given pyramid level, and position
    /// in the image sequence vector of that level
    std::string image_path(int pyr_level, int pos) const;

    bool empty() const { return pyr_infos_.empty(); }

    int n_pyr_levels() const { return (int)pyr_infos_.size(); }

    int z_scale_start_level() const  { return z_scale_start_level_; }

    std::vector<int> dims(int pyr_level = 0) const;

    std::vector<int> xyz_dims(int pyr_level = 0) const;

    FileFormats file_format(int pyr_level = 0) const;

    VoxelType voxel_type(int pyr_level = 0) const;

    std::string image_root_dir() const  { return image_root_dir_; }

    std::string pyr_level_dir(int pyr_level = 0) const;

    // pyr_level_dir_name should be full path
    bool has_pyr_level_dir(const std::string& pyr_level_dir_path);

    const std::vector<MPyrImageInfo>& pyr_infos() const { return pyr_infos_; }

    std::vector<MPyrImageInfo>& pyr_infos() { return pyr_infos_; }

    const std::vector<int> pyr_xy_ratios() const { return pyr_xy_ratios_; }

    const std::vector<int> pyr_z_ratios() const { return pyr_z_ratios_; }

private:
    void ReOrganizePyInfos();

    void CalculatePyrRatios();

    /// if image root dir is empty, and a level0 image pyramid info encountered,
    /// use the pyr_level_dir of the level0 image pyramid info as image root.
    /// image root dir will only be updated once. assertion is made on if no more
    /// than one image pyramid info is level0.
    void UpdateImageRootDir();

    static bool HigherResolution(const MPyrImageInfo& pyr1, const MPyrImageInfo& pyr2);

    void SaveImpl();
    // starting pyramid level to have z dimension scaled from parent level
    int z_scale_start_level_;
    std::string image_root_dir_;
    std::vector<int> pyr_xy_ratios_, pyr_z_ratios_;
    std::vector<MPyrImageInfo> pyr_infos_;
};
}

#endif //MCP3D_MCP3D_IMAGE_CONFIG_HPP
