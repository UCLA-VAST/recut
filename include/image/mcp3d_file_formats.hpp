//
// Created by muyezhu on 3/17/19.
//

#ifndef MCP3D_MCP3D_FORMATS_HPP
#define MCP3D_MCP3D_FORMATS_HPP

namespace mcp3d
{

enum class FileFormats
{
    TIFF, OMETIFF, HDF5, STACK, UNKNOWN = -1
};

inline std::string ImarisExtension() { return "ims"; }

inline std::string VirtualHdf5Extension() { return "virtual.hdf5"; }

std::unordered_set<std::string> FileFormatExtensions(FileFormats format =
FileFormats::UNKNOWN);

FileFormats FileFormatExtToEnum(const std::string &format_ext);

std::string FileFormatEnumToExt(FileFormats format);

std::pair<FileFormats, std::string>
FindFileFormatFromFile(const std::string &file_path,
                       FileFormats format = FileFormats::UNKNOWN);

/// find a ImageFormat from files in dir.
/// uses a directory iterator and returns on first hit.
/// return ImageFormat::UNKNOWN if no supported format found
std::pair<FileFormats, std::string> FindFileFormatInDir(const std::string &dir);

/// does not assert file_path exists
/// if format is UNKNOWN, return true if file_path contains no known
/// extension. otherwise, return true if file_path contains extension
/// consistent with the format
bool FileIsFormat(const std::string& file_path, FileFormats format);

/// does not assert file_path exists
bool FileIsImaris(const std::string& file_path);

/// does not assert file_path exists
bool FileIsVirtualHdf5(const std::string& file_path);

}


#endif //MCP3D_MCP3D_FORMATS_HPP
