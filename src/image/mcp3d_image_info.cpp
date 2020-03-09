//
// Created by muyezhu on 2/11/18.
//
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include "common/mcp3d_common.hpp"
#include "project_structure/mcp3d_project_structure.hpp"
#include "mcp3d_image_constants.hpp"
#include "mcp3d_image_info.hpp"

using namespace std;
using json = nlohmann::json;

mcp3d::MPyrImageInfo::MPyrImageInfo(const string &json_path,
                                    FileFormats expected_format)
{
    if (! mcp3d::IsFile(json_path) || !boost::algorithm::ends_with(json_path, ".json"))
        MCP3D_OS_ERROR(json_path + " does not exist or is not a json file")
    LoadJson(json_path);
    if (!mcp3d::IsDir(pyr_image_info_["image pyramid directory"]) ||
           pyr_image_info_["image pyramid directory"] == ".")
        pyr_image_info_["image pyramid directory"] = mcp3d::ParentDir(json_path);
    if (!mcp3d::IsDir(pyr_image_info_["image pyramid directory"]))
        MCP3D_RUNTIME_ERROR("img pyramid directory from json file: " +
                            pyr_image_info_["image pyramid directory"].get<string>() +
                            " is not a valid directory")
    ParseJson(expected_format);
}

mcp3d::MPyrImageInfo::MPyrImageInfo(const json& image_pyr_info,
                                    FileFormats expected_format):
        pyr_image_info_(image_pyr_info)
{
    if (!mcp3d::IsDir(pyr_image_info_["image pyramid directory"]))
        MCP3D_RUNTIME_ERROR("img pyramid directory from json file: " +
                            pyr_image_info_["image pyramid directory"].get<string>() +
                            " is not a valid directory")
    ParseJson(expected_format);
}

mcp3d::MPyrImageInfo::MPyrImageInfo(const std::vector<int> &dimensions,
                                    VoxelType voxel_type)
{
    MCP3D_ASSERT(mcp3d::KnownVoxelType(voxel_type))
    std::vector<int> dims = mcp3d::To5D(dimensions);
    pyr_image_info_["format"] = "unknown";
    pyr_image_info_["x dimension"] = dims[4];
    pyr_image_info_["y dimension"] = dims[3];
    pyr_image_info_["z dimension"] = dims[2];
    pyr_image_info_["channels"] = dims[1];
    pyr_image_info_["time points"] = dims[0];
    pyr_image_info_["level 0 image"] = "true";
    pyr_image_info_["z dimension scaled"] = "false";
    pyr_image_info_["image pyramid directory"] = in_memory_str();
    pyr_image_info_["image sequence"] = vector<string>({in_memory_str()});
    pyr_image_info_["xchunk dimension"] = dims[4];
    pyr_image_info_["ychunk dimension"] = dims[3];
    pyr_image_info_["zchunk dimension"] = dims[2];
    pyr_image_info_["dimensions order"] = "tczyx";
    pyr_image_info_["voxel type"] = mcp3d::VoxelTypeEnumToStr(voxel_type);
    pyr_image_info_["description"] = mcp3d::MPyrImageInfo::in_memory_str();
    ParseJson();
}

mcp3d::MPyrImageInfo::MPyrImageInfo(const mcp3d::MPyrImageInfo& other):
                              pyr_image_info_(other.pyr_image_info_)
{
    ParseJson();
}

bool mcp3d::MPyrImageInfo::operator==(const mcp3d::MPyrImageInfo &other) const
{
    return (pyr_file_format_ == other.pyr_file_format_ &&
            is_level0_image_ == other.is_level0_image_ &&
            zdim_scaled_ == other.zdim_scaled_ &&
            rank_ == other.rank_ &&
            xdim_ == other.xdim_ &&
            ydim_ == other.ydim_ &&
            zdim_ == other.zdim_ &&
            n_channels_ == other.n_channels_ &&
            n_times_ == other.n_times_ &&
            xchunk_dim_ == other.xchunk_dim_ &&
            ychunk_dim_ == other.ychunk_dim_ &&
            zchunk_dim_ == other.zchunk_dim_ &&
            n_xchunks_ == other.n_xchunks_ &&
            n_ychunks_ == other.n_ychunks_ &&
            n_zchunks_ == other.n_zchunks_ &&
            n_chunks_ == other.n_chunks_ &&
            dims_order_ == other.dims_order_ &&
            pyr_level_dir_ == other.pyr_level_dir_ &&
            image_names_ == other.image_names_ &&
            pyr_voxel_type_ == other.pyr_voxel_type_);
}

void mcp3d::MPyrImageInfo::SetEmptyPyrInfo()
{
    pyr_file_format_ = mcp3d::FileFormats::UNKNOWN;
    pyr_voxel_type_ = mcp3d::VoxelType::UNKNOWN;
    is_level0_image_ = false;
    zdim_scaled_ = false;
    pyr_level_dir_ = "";
    image_names_ = {};
    dims_order_ = "";
    rank_ = -1;
    xdim_ = 0;
    ydim_ = 0;
    zdim_ = 0;
    n_channels_ = 0;
    n_times_ = 0;
    xchunk_dim_ = 0;
    ychunk_dim_ = 0;
    zchunk_dim_ = 0;
    n_xchunks_ = 0;
    n_ychunks_ = 0;
    n_zchunks_ = 0;
    n_chunks_ = 0;
}

void mcp3d::MPyrImageInfo::LoadJson(const string &json_path)
{
    ifstream ifs(json_path);
    ifs >> pyr_image_info_;
    ifs.close();
}

void mcp3d::MPyrImageInfo::RequiredEntriesComplete()
{
    // ensure all required entries present
    for (const string &entry: mcp3d::MImagePyrInfoEntries{}.required_entries())
        if (pyr_image_info_.find(entry) == pyr_image_info_.end())
        MCP3D_INVALID_ARGUMENT("required image config entry " + entry + " not found")
}

void mcp3d::MPyrImageInfo::ParseJson(FileFormats expected_format)
{
    RequiredEntriesComplete();
    // image format
    pyr_file_format_ = mcp3d::FileFormatExtToEnum(pyr_image_info_["format"]);
    if (expected_format != FileFormats::UNKNOWN)
        if (expected_format != pyr_file_format_)
            MCP3D_RUNTIME_ERROR(mcp3d::FileFormatEnumToExt(expected_format) +
                                " format expected, got " +
                                mcp3d::FileFormatEnumToExt(pyr_file_format_))

    string is_level0_str = mcp3d::StringLower(pyr_image_info_["level 0 image"]);
    if (is_level0_str != "true" && is_level0_str != "false")
        MCP3D_INVALID_ARGUMENT("level 0 image entry should equal to either true or false")
    is_level0_image_ = is_level0_str == "true";

    if (pyr_image_info_.find("z dimension scaled") == pyr_image_info_.end())
        pyr_image_info_["z dimension scaled"] = "unknown";
    zdim_scaled_ = pyr_image_info_["z dimension scaled"] == "true";

    pyr_level_dir_ = pyr_image_info_["image pyramid directory"].get<string>();

    // image global rank and dimensions
    try
    {
        xdim_ = pyr_image_info_["x dimension"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("x dimension parsing failure\n" + string(e.what()))
    }
    try
    {
        ydim_ = pyr_image_info_["y dimension"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("y dimension parsing failure\n" + string(e.what()))
    }
    try
    {
        zdim_ = pyr_image_info_["z dimension"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("z dimension parsing failure\n" + string(e.what()))
    }
    try
    {
        n_channels_ = pyr_image_info_["channels"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("channel number parsing failure\n" + string(e.what()))
    }
    try
    {
        n_times_ = pyr_image_info_["time points"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("time point number parsing failure\n" + string(e.what()))
    }
    MCP3D_ASSERT(xdim_ > 1 && ydim_ > 1 && zdim_ > 0 && n_channels_ > 0 && n_times_ > 0)

    rank_ = 2 + (int)(zdim_ > 1) + (int)(n_channels_ > 1) + (int)(n_times_ > 1);
    // chunk dimensions
    try
    {
        xchunk_dim_ = pyr_image_info_["xchunk dimension"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("xchunk dimension parsing failure\n" + string(e.what()))
    }
    try
    {
        ychunk_dim_ = pyr_image_info_["ychunk dimension"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("ychunk dimension parsing failure\n" + string(e.what()))
    }
    try
    {
        zchunk_dim_ = pyr_image_info_["zchunk dimension"];
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("zchunk dimension parsing failure\n" + string(e.what()))
    }
    MCP3D_ASSERT(xchunk_dim_ <= xdim_ && ychunk_dim_ <= ydim_ && zchunk_dim_ <= zdim_)

    // chunk numbers
    n_xchunks_ = xdim_ / xchunk_dim_;
    n_ychunks_ = ydim_ / ychunk_dim_;
    n_zchunks_ = zdim_ / zchunk_dim_;
    n_chunks_ = n_xchunks_ * n_ychunks_ * n_zchunks_ * n_channels_ * n_times_;

    // dimension ordering
    dims_order_ = mcp3d::StringLower(pyr_image_info_["dimensions order"]);
    MCP3D_ASSERT(dims_order_ == "tczyx" || dims_order_ == "ctzyx")
    // pixel data type
    string pixel_type_str = pyr_image_info_["voxel type"];
    pyr_voxel_type_ = mcp3d::VoxelTypeStrToEnum(pixel_type_str);
    MCP3D_ASSERT(pyr_voxel_type_ != VoxelType::UNKNOWN)

    try
    {
        image_names_ = pyr_image_info_["image sequence"].get<vector<string>>();
    }
    catch (const exception& e)
    {
        MCP3D_RUNTIME_ERROR("image sequence parsing failure\n" + string(e.what()))
    }
    if (pyr_file_format_ == mcp3d::FileFormats::TIFF)
        MCP3D_ASSERT(image_names_.size() == size_t(n_chunks_) ||
                     (image_names_.size() == 1 &&
                      image_names_[0] == MPyrImageInfo::in_memory_str()))
    else if (pyr_file_format_ == mcp3d::FileFormats::HDF5)
        MCP3D_ASSERT(image_names_.size() == 1)

    if (pyr_image_info_.find("description") != pyr_image_info_.end())
        description_ = pyr_image_info_["description"].get<string>();
}

mcp3d::MImageInfo::MImageInfo(const std::string &image_info_path)
{
    MCP3D_ASSERT(mcp3d::IsFile(image_info_path));
    json pyr_info_jsons;
    ifstream ifs(image_info_path);
    ifs >> pyr_info_jsons;
    ifs.close();
    vector<MPyrImageInfo> pyr_infos;
    json pyr_info_json;
    for (const auto& item: pyr_info_jsons.items())
    {
        pyr_info_json = item.value();
        pyr_infos.emplace_back(pyr_info_json);
    }
    AddPyrInfos(pyr_infos);
}

mcp3d::MImageInfo::MImageInfo(MPyrImageInfo &pyr_info)
{
    AddPyrInfo(pyr_info);
}

mcp3d::MImageInfo::MImageInfo(vector<MPyrImageInfo> &pyr_infos)
{
    AddPyrInfos(pyr_infos);
}

mcp3d::MImageInfo::MImageInfo(const vector<int>& dimensions,
                              mcp3d::VoxelType voxel_type)
{
    mcp3d::MPyrImageInfo pyr_info(dimensions, voxel_type);
    pyr_infos_.push_back(move(pyr_info));
    pyr_xy_ratios_.push_back(1);
    pyr_z_ratios_.push_back(1);
    z_scale_start_level_ = mcp3d::ZSCALE_NONE;
    image_root_dir_ = mcp3d::MPyrImageInfo::in_memory_str();
}

void mcp3d::MImageInfo::AddPyrInfo(MPyrImageInfo &pyr_info)
{
    vector<mcp3d::MPyrImageInfo> pyr_infos;
    pyr_infos.push_back(move(pyr_info));
    AddPyrInfos(pyr_infos);
}

void mcp3d::MImageInfo::AddPyrInfos(vector<MPyrImageInfo> &pyr_infos)
{
    if (pyr_infos.empty())
        return;
    for (auto& pyr_info:pyr_infos)
    {
        MCP3D_ASSERT(pyr_info.pyr_level_dir() !=
                             mcp3d::MPyrImageInfo::in_memory_str())
        pyr_infos_.push_back(move(pyr_info));
    }
    ReOrganizePyInfos();
}

void mcp3d::MImageInfo::ReOrganizePyInfos()
{
    sort(pyr_infos_.begin(), pyr_infos_.end(), mcp3d::MImageInfo::HigherResolution);
    CalculatePyrRatios();
    UpdateImageRootDir();
}

void mcp3d::MImageInfo::CalculatePyrRatios()
{
    pyr_xy_ratios_.clear();
    pyr_z_ratios_.clear();
    // full size image has pyramid ratio = 1
    for (int i = 0; i < (int)pyr_infos_.size(); ++i)
    {
        pyr_xy_ratios_.push_back(pyr_infos_[0].xdim() / pyr_infos_[i].xdim());
        pyr_z_ratios_.push_back(pyr_infos_[0].zdim() / pyr_infos_[i].zdim());
    }
    z_scale_start_level_ = mcp3d::ZSCALE_NONE;
    for (int i = 0; i < (int)pyr_infos_.size(); ++i)
        if (pyr_z_ratios_[i] > 1)
        {
            z_scale_start_level_ = i;
            for (int j = i; j < (int)pyr_infos().size(); ++j)
            {
                pyr_infos_[j].zdim_scaled_ = true;
                pyr_infos_[j].pyr_image_info_["z dimension scaled"] = "true";
            }
            break;
        }
}

void mcp3d::MImageInfo::UpdateImageRootDir()
{
    if (pyr_infos_.empty())
        return;
    int n_level0_pyr = 0;
    vector<string> pyr_level_dirs;
    for (auto& pyr_info: pyr_infos_)
    {
        n_level0_pyr += (int)pyr_info.is_level0_image();
        if (pyr_info.is_level0_image() && image_root_dir_.empty())
            image_root_dir_ = pyr_info.pyr_level_dir_;
    }
    MCP3D_ASSERT(n_level0_pyr == 0 || n_level0_pyr == 1)
}

bool mcp3d::MImageInfo::HigherResolution(const mcp3d::MPyrImageInfo& pyr1,
                                         const mcp3d::MPyrImageInfo& pyr2)
{
    return pyr1.xdim() > pyr2.xdim();
}

bool mcp3d::MImageInfo::operator==(const mcp3d::MImageInfo& other) const
{
    if (n_pyr_levels() != other.n_pyr_levels())
        return false;
    if (image_root_dir_ != other.image_root_dir_)
        return false;
    if (pyr_xy_ratios_ != other.pyr_xy_ratios_)
        return false;
    for (int i = 0; i < n_pyr_levels(); ++i)
        if (pyr_infos()[i] != other.pyr_infos()[i])
            return false;
    return true;
}

mcp3d::MImageInfo& mcp3d::MImageInfo::operator+= (mcp3d::MImageInfo& rhs)
{
    if (!image_root_dir_.empty() && !rhs.image_root_dir_.empty() &&
            image_root_dir_ != rhs.image_root_dir_)
        MCP3D_RUNTIME_ERROR("can not apply += to MImageInfo objects with different image root directory")
    if (image_root_dir_.empty() && !rhs.image_root_dir_.empty())
        image_root_dir_ = rhs.image_root_dir_;
    AddPyrInfos(rhs.pyr_infos());
    return *this;
}

mcp3d::MImageInfo& mcp3d::MImageInfo::operator+= (mcp3d::MImageInfo&& rhs)
{
    if (!image_root_dir_.empty() && !rhs.image_root_dir_.empty() &&
        image_root_dir_ != rhs.image_root_dir_)
    MCP3D_RUNTIME_ERROR("can not apply += to MImageInfo objects with different image root directory")
    if (image_root_dir_.empty() && !rhs.image_root_dir_.empty())
        image_root_dir_ = rhs.image_root_dir_;
    AddPyrInfos(rhs.pyr_infos());
    return *this;
}

bool mcp3d::MImageInfo::AllPathsExist() const
{
    MCP3D_ASSERT(image_root_dir_ != mcp3d::MPyrImageInfo::in_memory_str())
    for (int i = 0; i < n_pyr_levels(); ++i)
        if (!LevelPathsExist(i))
            return false;
    return true;
}

bool mcp3d::MImageInfo::LevelPathsExist(int pyr_level) const
{
    MCP3D_ASSERT(image_root_dir_ != mcp3d::MPyrImageInfo::in_memory_str())
    MCP3D_ASSERT(pyr_level >= 0 && pyr_level < n_pyr_levels())
    bool is_mpi = mcp3d::MPIInitialized();
    int n_threads = is_mpi? 1 : mcp3d::DefaultNumThreads();
    const MPyrImageInfo& pyr_info = pyr_infos()[pyr_level];
    int n_imgs = (int)(pyr_info.image_sequence().size());
    int n_imgs_per_thread = n_imgs / n_threads;
    bool exist = true;
    #pragma omp parallel num_threads(n_threads)
    {
        try
        {
            CHECK_PARALLEL_MODEL
            int thread_id = omp_get_thread_num();
            int img_id_begin = thread_id * n_imgs_per_thread,
                img_id_end = min(img_id_begin + n_imgs_per_thread, n_imgs);
            for (int j = img_id_begin; j < img_id_end; ++j)
                if (!mcp3d::IsFile(
                        mcp3d::JoinPath(pyr_info.pyr_level_dir_, pyr_info.image_sequence()[j])))
                    MCP3D_RUNTIME_ERROR(
                        mcp3d::JoinPath(pyr_info.pyr_level_dir_, pyr_info.image_sequence()[j]) + " is not a file")
        }
        catch (const mcp3d::MCPRuntimeError& e)
        {
            mcp3d::PrintNested(e);
            exist = false;
        }
    }
    return exist;
}

void mcp3d::MImageInfo::ValidatePyramidsStructure() const
{
    if (empty())
        return;
    if (image_root_dir_ == mcp3d::MPyrImageInfo::in_memory_str())
    {
        MCP3D_ASSERT(n_pyr_levels() == 1)
        MCP3D_ASSERT(pyr_infos()[0].is_level0_image())
        return;
    }
    MCP3D_ASSERT(mcp3d::IsDir(image_root_dir_))
    MCP3D_ASSERT(pyr_infos_.size() == pyr_xy_ratios_.size())
    MCP3D_ASSERT(pyr_infos_.size() == pyr_z_ratios_.size())
    MCP3D_ASSERT(pyr_infos_[0].is_level0_image())
    // one and only one level 0 image
    for (int i = 1; i < n_pyr_levels(); ++i)
    {
        const MPyrImageInfo& pyr_info = pyr_infos_[i];
        MCP3D_ASSERT(!pyr_info.is_level0_image())
    }

    // pyramid xy ratios correct
    for (int i = 0; i < n_pyr_levels(); ++i)
        MCP3D_ASSERT(pyr_xy_ratios_[i] == mcp3d::IntPow(2, i))
    // pyramid z ratios correct
    for (int i = 0; i < n_pyr_levels(); ++i)
    {
        if (z_scale_start_level_ == mcp3d::ZSCALE_NONE ||
            i < z_scale_start_level_)
            MCP3D_ASSERT(pyr_z_ratios_[i] == 1)
        else
            MCP3D_ASSERT(pyr_z_ratios_[i] == mcp3d::IntPow(2, i - z_scale_start_level_ + 1))
    }
    // if composite pyramid level directory exists, it must contains all this and
    // later resolutions
    bool image_root_composite = false;
    for (int i = 0; i < (int)pyr_infos_.size(); ++i)
    {
        if (mcp3d::MPyramidsLayout::IsCompositePyrLevelDir(pyr_infos()[i].pyr_level_dir_))
        {
            for (int j = i; j < (int)pyr_infos_.size(); ++j)
                MCP3D_ASSERT(mcp3d::MPyramidsLayout::IsCompositePyrLevelDir(pyr_infos()[j].pyr_level_dir_))
            break;
        }
        // if image root dir is a composite level, e.g. contains a FusionStitcher.ims
        else if (i > 0 && pyr_infos()[i].pyr_level_dir_ == image_root_dir_)
        {
            image_root_composite = true;
            for (int j = i; j < (int)pyr_infos_.size(); ++j)
                MCP3D_ASSERT(pyr_infos()[j].pyr_level_dir_ == image_root_dir_)
            break;
        }
    }
    // pyramid level directory name and level value consistent
    for (int i = 0; i < n_pyr_levels(); ++i)
    {
        if (i == 0)
            MCP3D_ASSERT(pyr_infos_[i].pyr_level_dir_ == image_root_dir_ ||
                         mcp3d::MPyramidsLayout::IsPyrLevel0Dir(pyr_infos_[i].pyr_level_dir_))
        else if (!mcp3d::MPyramidsLayout::IsCompositePyrLevelDir(pyr_infos_[i].pyr_level_dir_))
        {
            if (!image_root_composite)
                MCP3D_ASSERT(i == mcp3d::MPyramidsLayout::DirPyrLevel(pyr_infos_[i].pyr_level_dir_))
        }
        else
            MCP3D_ASSERT(i >= mcp3d::MPyramidsLayout::CompositeDirStartPyrLevel(pyr_infos_[i].pyr_level_dir_) &&
                         i <= mcp3d::MPyramidsLayout::CompositeDirEndPyrLevel(pyr_infos_[i].pyr_level_dir_))
    }
}

string mcp3d::MImageInfo::image_path(int pyr_level, int pos) const
{
    MCP3D_ASSERT(pyr_level >= 0 && pos >= 0)
    if (pyr_level >= n_pyr_levels())
        MCP3D_OUT_OF_RANGE("pyr_level exceeds maximum available")
    if (pos >= (int)(pyr_infos_[pyr_level].image_names_.size()))
        MCP3D_OUT_OF_RANGE("pos exceeds maximum available")
    if (image_root_dir_ == mcp3d::MPyrImageInfo::in_memory_str())
        return mcp3d::MPyrImageInfo::in_memory_str();
    return mcp3d::JoinPath({pyr_infos_[pyr_level].pyr_level_dir_,
                            pyr_infos_[pyr_level].image_names_[pos]});
}

vector<int> mcp3d::MImageInfo::dims(int pyr_level) const
{
    if (empty())
        return {0, 0, 0, 0, 0};
    MCP3D_ASSERT(pyr_level >= 0 && pyr_level < n_pyr_levels())
    return pyr_infos_[pyr_level].dims();
}

vector<int> mcp3d::MImageInfo::xyz_dims(int pyr_level) const
{
    if (empty())
        return {0, 0, 0};
    MCP3D_ASSERT(pyr_level >= 0 && pyr_level < n_pyr_levels())
    return pyr_infos_[pyr_level].xyz_dims();
}

mcp3d::FileFormats mcp3d::MImageInfo::file_format(int pyr_level) const
{
    if (empty())
        return mcp3d::FileFormats::UNKNOWN;
    MCP3D_ASSERT(pyr_level >= 0 && pyr_level < n_pyr_levels())
    return pyr_infos_[pyr_level].pyr_file_format_;
}

mcp3d::VoxelType mcp3d::MImageInfo::voxel_type(int pyr_level) const
{
    if (empty())
        return mcp3d::VoxelType::UNKNOWN;
    MCP3D_ASSERT(pyr_level >= 0 && pyr_level < n_pyr_levels())
    return pyr_infos_[pyr_level].pyr_voxel_type_;
}

string mcp3d::MImageInfo::pyr_level_dir(int pyr_level) const
{
    if (empty())
        return string();
    MCP3D_ASSERT(pyr_level >= 0 && pyr_level < n_pyr_levels())
    return pyr_infos_[pyr_level].pyr_level_dir_;
}

bool mcp3d::MImageInfo::has_pyr_level_dir(const string &pyr_level_dir_path)
{
    if (empty())
        return false;
    for (const auto& pyr_info: pyr_infos_)
        if (pyr_info.pyr_level_dir_ == pyr_level_dir_path)
            return true;
    return false;
}

void mcp3d::MImageInfo::Save()
{
    SaveImpl();
}

void mcp3d::MImageInfo::Clear()
{
    image_root_dir_ = string();
    pyr_xy_ratios_.clear();
    pyr_infos_.clear();
}

int mcp3d::MImageInfo::BytesPerVoxel() const
{
    if (empty())
        return 0;
    return mcp3d::BytesPerVoxelElement(voxel_type(0));
}

void mcp3d::MImageInfo::SaveImpl()
{
    if (image_root_dir_ == mcp3d::MPyrImageInfo::in_memory_str())
    {
        MCP3D_MESSAGE("in memory image info, not saving anything")
        return;
    }
    if (pyr_infos_.empty())
    {
        MCP3D_MESSAGE("image info is empty, not saving anything")
        return;
    }
    MCP3D_ASSERT(AllPathsExist())
    json pyr_infos;
    for (int i = 0; i < n_pyr_levels(); ++i)
        pyr_infos[mcp3d::MPyramidsLayout::PyrLevelDirName(i)] =
                pyr_infos_[i].pyr_image_info_;
    ofstream out_file(mcp3d::MPyramidsLayout::ImageInfoPath(image_root_dir_),
                      fstream::out);
    out_file << pyr_infos.dump(4) << endl;
    out_file.close();
}

