//
// Created by muyezhu on 3/17/19.
//
#include <string>
#include <unordered_set>
#include <boost/algorithm/string/predicate.hpp>
#include "common/mcp3d_utility.hpp"
#include "mcp3d_file_formats.hpp"

using namespace std;


unordered_set<string> mcp3d::FileFormatExtensions(FileFormats format)

{
    if (format == mcp3d::FileFormats::TIFF)
        return unordered_set<string>({"tiff", "tif"});
    else if (format == mcp3d::FileFormats::OMETIFF)
        return unordered_set<string>({"ome-tiff", "ome-tif",
                                      "ometiff", "ometif", "ome.tiff", "ome.tif"});
    else if (format == mcp3d::FileFormats::HDF5)
        return unordered_set<string>({mcp3d::ImarisExtension(),
                                      mcp3d::VirtualHdf5Extension()});
        // mcp3d::FileFormats::UNKNOWN
    else
        return unordered_set<string>({"tiff", "tif", "ome-tiff", "ome-tif",
                                      "ometiff", "ometif", "ome.tiff", "ome.tif",
                                      mcp3d::ImarisExtension(),
                                      mcp3d::VirtualHdf5Extension()});
}

mcp3d::FileFormats mcp3d::FileFormatExtToEnum(const std::string &format_ext)
{
    if (format_ext.empty())
        return mcp3d::FileFormats::UNKNOWN;
    string format_extension(format_ext);
    if (format_extension[0] == '.')
        format_extension = format_extension.substr(1, format_extension.size() - 1);

    if (format_extension == "tiff" || format_extension == "tif")
        return mcp3d::FileFormats::TIFF;
    else if (format_extension == "ome-tiff" ||
             format_extension == "ometiff" ||
             format_extension == "ome.tiff" ||
             format_extension == "ome-tif" ||
             format_extension == "ometif" ||
             format_extension == "ome.tif")
        return mcp3d::FileFormats::OMETIFF;
    else if (format_extension == mcp3d::ImarisExtension() ||
             format_extension == mcp3d::VirtualHdf5Extension())
        return mcp3d::FileFormats::HDF5;
    else if (format_extension == "stack")
        return mcp3d::FileFormats::STACK;
    else
        return mcp3d::FileFormats::UNKNOWN;
}

string mcp3d::FileFormatEnumToExt(FileFormats format)
{
    if (format == mcp3d::FileFormats::TIFF)
        return "tiff";
    else if (format == mcp3d::FileFormats::OMETIFF)
        return "ome-tiff";
    else if (format == mcp3d::FileFormats::HDF5)
        return "hdf5";
    else if (format == mcp3d::FileFormats::STACK)
        return "stack";
    else
        return "unknown";
}

pair<mcp3d::FileFormats, string> mcp3d::FindFileFormatFromFile(const string &file_path,
                                                               FileFormats format)
{
    if (!mcp3d::IsFile(file_path))
        MCP3D_OS_ERROR(file_path + " is not a file")
    const unordered_set<string>& format_exts = mcp3d::FileFormatExtensions(format);
    for (const auto& format_ext: format_exts)
        if (boost::algorithm::ends_with(file_path, format_ext))
        {
            // for .ims files further check for FusionStitcher string
            if (format_ext == mcp3d::ImarisExtension())
            {
                if (file_path.find("FusionStitcher") == string::npos)
                    continue;
            }
            return make_pair(mcp3d::FileFormatExtToEnum(format_ext), format_ext);
        }
    return make_pair(mcp3d::FileFormats::UNKNOWN, "");
}

std::pair<mcp3d::FileFormats, string> mcp3d::FindFileFormatInDir(const std::string &dir)
{
    if (!mcp3d::IsDir(dir))
        MCP3D_OS_ERROR(dir + " is not a directory")
    boost::filesystem::path p(dir);
    boost::filesystem::directory_iterator it_end;
    const unordered_set<string>& format_exts = mcp3d::FileFormatExtensions();
    for(auto it = boost::filesystem::directory_iterator(p); it != it_end; ++it)
    {
        if (boost::filesystem::is_regular_file(it->status()))
        {
            string name = it->path().filename().string();
            for (const auto& format_ext: format_exts)
                if (boost::algorithm::ends_with(name, format_ext))
                {
                    // for .ims files further check for FusionStitcher string
                    if (format_ext == mcp3d::ImarisExtension())
                    {
                        if (name.find("FusionStitcher") == string::npos)
                            continue;
                    }
                    return make_pair(mcp3d::FileFormatExtToEnum(format_ext), format_ext);
                }
        }
    }
    return make_pair(mcp3d::FileFormats::UNKNOWN, "");
}

bool mcp3d::FileIsFormat(const std::string &file_path, FileFormats format)
{
    unordered_set<string> format_strs;
    if (format == FileFormats::UNKNOWN)
    {
        format_strs = mcp3d::FileFormatExtensions();
        for (const auto& format_str: format_strs)
            if (boost::algorithm::ends_with(file_path, format_str))
                return false;
        return true;
    }
    else
    {
        format_strs = mcp3d::FileFormatExtensions(format);
        for (const auto& format_str: format_strs)
            if (boost::algorithm::ends_with(file_path, format_str))
                return true;
        return false;
    }
}

bool mcp3d::FileIsImaris(const std::string &file_path)
{
    return boost::algorithm::ends_with(file_path, mcp3d::ImarisExtension());
}

bool mcp3d::FileIsVirtualHdf5(const std::string &file_path)
{
    return boost::algorithm::ends_with(file_path, mcp3d::VirtualHdf5Extension());
}