//
// Created by muyezhu on 3/17/19.
//

#ifndef MCP3D_MCP3D_FOLDER_SLICES_HPP
#define MCP3D_MCP3D_FOLDER_SLICES_HPP

#include <unordered_map>
#include <map>
#include <nlohmann/json.hpp>

namespace mcp3d
{
/// one of the 5 dimensions that a given folder slice into
enum class FolderDimensions
{
    X, Y, Z, CHANNEL, TIME, UNKNOWN = -1
};

inline std::string FolderDimensionStr(FolderDimensions dim)
{
    if (dim == FolderDimensions::X)
        return "x";
    else if (dim == FolderDimensions::Y)
        return "y";
    else if (dim == FolderDimensions::Z)
        return "z";
    else if (dim == FolderDimensions::CHANNEL)
        return "ch";
    else if (dim == FolderDimensions::TIME)
        return "t";
    else
        return "unknown";
}

class MPyrFolderSlices
{
public:
    MPyrFolderSlices(): MPyrFolderSlices(0) {};
    explicit MPyrFolderSlices(int pyr_level,
                              const std::map<FolderDimensions, int>&
                                 folder_slice_numbers =
                                     {{FolderDimensions::X, 1},
                                      {FolderDimensions::Y, 1},
                                      {FolderDimensions::Z, 1}});

private:
    int pyr_level_;
    std::map<FolderDimensions, int> folder_slice_numbers_;
};

/// MFolderSlices has channel and time slices and MPyrFOlderSlices instances
/// MPyrFolderSlices has zyx slices
class MFolderSlices
{
public:
    using json = nlohmann::json;

    MFolderSlices(): MFolderSlices(std::string()) {};

    explicit MFolderSlices(const std::string& image_root_dir,
                           const std::vector<std::string>& channel_slice_names =
                                std::vector<std::string>{},
                           const std::unordered_map<int, MPyrFolderSlices>&
                                pyr_folder_slices =
                                std::unordered_map<int, MPyrFolderSlices>{}):
                                 image_root_dir_(image_root_dir),
                                 channel_slice_names_(channel_slice_names),
                                 time_slice_names_({""}),
                                 pyr_folder_slices_(pyr_folder_slices) {};

    /// prefix slice index with "fslice_" for parsing
    static std::vector<std::string> DefaultFolderSliceNames(FolderDimensions dim,
                                                            int n_slices);

    /// return true if dir has a flat structure, aka unsliced by sub folder
    /// hierarchies. if dir is not a key in folder_structures_, or if
    /// all folder slice vectors in folder_structures_[dir] are empty, dir is
    /// considered flat.
    /// root
    ///    - ch0
    ///        - img files
    /// is not flat. the channel folder slice has dimension 1
    /// if image_root_dir_ is empty, or if dir is not a subdirectory of root
    /// directory, throw error
    bool IsFlat(const std::string& dir);

    const std::string& image_root_dir()  { return image_root_dir_; }

    friend class MImage;

private:
    void SetImageRoot(const std::string& img_root_dir);

    void SetFolderSlices(const std::string &sub_dir,
                         const std::map<FolderDimensions, std::vector<std::string>> &kvs =
                         std::map<FolderDimensions, std::vector<std::string>>({}));

    std::string image_root_dir_;
    // time dimension is assumed to be length 1 always
    std::vector<std::string> channel_slice_names_, time_slice_names_;
    // number of folder slices along each folder dimension, defaults to 1
    std::unordered_map<int, MPyrFolderSlices> pyr_folder_slices_;
};

}

#endif //MCP3D_MCP3D_FOLDER_SLICES_HPP
