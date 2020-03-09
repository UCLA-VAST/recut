//
// Created by muyezhu on 12/2/18.
//
#include <boost/algorithm/string/predicate.hpp>
#include "mcp3d_image_common.hpp"
#include "mcp3d_image_formats.hpp"

using namespace std;



mcp3d::MImageInfo mcp3d::MImageFormats::ReadImageInfo(const string &pyr_level_dir,
                                                      bool is_root_image,
                                                      const string &format_ext)
{
    if (!mcp3d::IsDir(pyr_level_dir))
        MCP3D_OS_ERROR(pyr_level_dir + " is not a directory")
    if (!CorrectFileExtension(format_ext))
        MCP3D_RUNTIME_ERROR("incorrect extension given to " +
                                    mcp3d::FileFormatEnumToExt(format_)+
                            " format class")
    vector<string> img_paths = mcp3d::FilesEndsWithInDir(pyr_level_dir, format_ext,
                                                         false, true, -1);
    if (img_paths.empty())
        return mcp3d::MImageInfo{};
    if (format_ext == mcp3d::ImarisExtension())
    {
        vector<string> stitched_ims_paths;
        for (const auto& img_path: img_paths)
            if (img_path.find("FusionStitcher") != string::npos)
                stitched_ims_paths.push_back(img_path);
        if (stitched_ims_paths.empty())
            return mcp3d::MImageInfo{};
        cout << "stitched imaris image found. reading image info..." << endl;
        return ReadImageInfo(stitched_ims_paths, is_root_image, false, pyr_level_dir);
    }
    cout << format_ext << " images found. reading image info..." << endl;
    MCP3D_TRY(return ReadImageInfo(img_paths, is_root_image, false, pyr_level_dir);)
}

