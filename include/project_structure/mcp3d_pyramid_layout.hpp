//
// Created by muyezhu on 3/23/18.
//

#ifndef MCP3D_MCP3D_PYRAMID_LAYOUT_HPP
#define MCP3D_MCP3D_PYRAMID_LAYOUT_HPP

#include <unordered_set>
#include <map>
#include <nlohmann/json.hpp>

namespace mcp3d
{

class MPyramidsLayout
{
public:
    MPyramidsLayout() = default;

    /// does not assert dir exists
    static bool IsMProjectImageComponentDir(const std::string &dir);

    static std::string
    PyrLevelDirName(int pyr_level, int composite_pyr_level_end = -1);

    /// does not assert img_root_dir exists
    static std::string
    PyrLevelDirPath(const std::string &img_root_dir, int pyr_level,
                    int composite_pyr_level_end = -1);

    /// does not assert dir exists
    static bool IsPyrLevelDir(const std::string &dir);

    /// does not assert dir exists
    static bool IsPyrLevel0Dir(const std::string &dir);

    /// does not assert dir exists
    static bool IsCompositePyrLevelDir(const std::string &dir);

    /// return - 1 if dir is not a pyr level dir, otherwise return the xx in
    /// pyr_level_xx[-yy]. does not assert dir exists
    static int DirPyrLevel(const std::string &dir);

    /// return -1 if dir is not a composite pyr level dir, other wise return
    /// the nn in pyr_level_nn-mm. does not assert dir exists
    static int CompositeDirStartPyrLevel(const std::string &dir);

    /// return -1 if dir is not a composite pyr level dir, other wise return
    /// the mm in pyr_level_nn-mm. does not assert dir exists
    static int CompositeDirEndPyrLevel(const std::string &dir);

    /// asserts dir exists and parsed image root dir exists
    static std::string ParsePyramidRootDir(const std::string &dir);

    /// asserts image_root_dir exists. return an additional string() if
    /// pyr_level_00 is not found, aka image_root_dir acts as pyr_level_00
    /// folder. sort output, place string() in front
    static std::vector<std::string>
    PyrLevelDirs(const std::string &image_root_dir,
                 bool full_path = true,
                 bool exclude_empty_dirs = true);

    static std::vector<std::string>
    NonZeroPyrLevelDirs(const std::string &image_root_dir,
                        bool full_path = true,
                        bool exclude_empty_dirs = true);

    /// does not assert dir exists or returned path exists
    static std::string ImageInfoPath(const std::string &image_root_dir);

    static std::string pyr_level_dir_prefix()
    {
        return "pyr_level_";
    }

    static std::string image_info_json_name()
    {
        return "__image_info__.json";
    };

    static const std::unordered_set<std::string> project_image_components()
    {
        return std::unordered_set<std::string>(
                {"raw", "stitch", "segmentation"});
    }
};

}

#endif //MCP3D_MCP3D_PYRAMID_LAYOUT_HPP
