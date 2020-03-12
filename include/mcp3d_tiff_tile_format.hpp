//
// Created by muyezhu on 12/1/18.
//

#ifndef MCP3D_MCP_TIFF_TILE_FORMAT_HPP
#define MCP3D_MCP_TIFF_TILE_FORMAT_HPP

#include "common/mcp3d_utility.hpp"
#include "image/mcp3d_image.hpp"
#include "image/mcp3d_image_formats.hpp"

namespace mcp3d
{
/// this class supports uniform tiff image stack chunks that represent the
/// volume such that at least one of the xyz dimensions of an individual chunk
/// is smaller than the that dimension of the represented volume. multiple z
/// planes can be packed into directories. does not enforce the xml tags of
/// ometiff, but will name images as .ome.tif to differentiate from output from
/// MTiffFormat
/// layout: at each pyramid level, use x_[0-9]_y[0-9]_z_[0-9] to indicate the
/// global coordinate of top left voxel in first directory. each pyramid level
/// will maintain identical chunk xyz dimensions, where each voxel represnets
/// increasingly coarse information, therefore the x_[0-9]_y[0-9]_z_[0-9] string
/// will differ. if no earlier pyramid level uses tiff chunk yet, the chunks
/// created chunks will have xyz dimensions (mcp3d::TIFFCHUNK_XDIM,
/// mcp3d::TIFFCHUNK_YDIM, mcp3d::TIFFCHUNK_ZDIM)
class MTiffStackFormat: public MImageFormats
{
public:
    using json = nlohmann::json;

    MTiffStackFormat(): MImageFormats(FileFormats::OMETIFF) {};

    bool CanRead() override { return true; }

    bool CanWrite() override { return true; }

    bool CanReadPartial() override { return true; }

    bool CanWriteInChunk() override { return false; }

    bool CanWriteInChunkParallel() override { return false; }

    /// will sort img_paths
    MImageInfo ReadImageInfo(const std::vector<std::string> &img_paths, bool is_root_image,
                             bool is_full_path, const std::string &common_dir) override;

    using MImageFormats::ReadImageInfo;

};
}

#endif //MCP3D_MCP_TIFF_TILE_FORMAT_HPP
