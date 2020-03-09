//
// Created by muyezhu on 1/30/19.
//

#ifndef MCP3D_IMS_IO_HPP_HPP
#define MCP3D_IMS_IO_HPP_HPP

#include <tuple>
#include <vector>
#include <hdf5.h>
#include "mcp3d_voxel_types.hpp"

namespace mcp3d
{

struct ImarisResolutionInfo
{
    std::string imaris_path;
    std::vector<std::string> channel_names;
    int resolution_level, n_channels;
    // the chunk here is the hdf5 internal chunk, not file chunk size
    // in MImageInfo
    std::vector<int> image_xyz_sizes, chunk_xyz_dims;
    std::string voxel_type_string;
    ImarisResolutionInfo(): imaris_path(std::string{}),
                            channel_names(std::vector<std::string>{}),
                            resolution_level(-1), n_channels(0),
                            image_xyz_sizes(std::vector<int>{}),
                            chunk_xyz_dims(std::vector<int>{}),
                            voxel_type_string(std::string{}) {}
};

ImarisResolutionInfo ReadImarisResolutionInfo(const std::string &imaris_path,
                                              int resolution_level, int time = 0);

// will close file
std::vector<std::string> ImarisResolutions(const std::string &imaris_path);

// empty vector returns true
bool ImarisResolutionsCorrect(const std::vector<std::string>& resolutions);

// will close file
std::vector<std::string> ImarisResolutionChannels(hid_t imaris_id,
                                                  int resolution_level,
                                                  int time = 0);

// zyx dimensions of channel dataset without chunk padding
// will close file. the ImageSizeZ, ImageSizeY, ImageSizeX are represented
// as character arrays: e.g. '2', '0', '4', '8'
std::vector<int> ImarisChannelImageXyzSizes(hid_t imaris_id,
                                            int resolution_level,
                                            int channel, int time = 0);

// asserts image xyz sizes from all channels under the resolution level are equal
std::vector<int> ImarisResolutionImageXyzSizes(hid_t imaris_id,
                                               int resolution_level,
                                               int time = 0);

std::vector<int> ImarisChannelChunkXyzDims(hid_t imaris_id,
                                           int resolution_level,
                                           int channel, int time = 0);

std::vector<int> ImarisResolutionChunkXyzDims(hid_t imaris_id,
                                              int resolution_level, int time = 0);


VoxelType ImarisChannelVoxelType(hid_t imaris_id, int resolution_level,
                                 int channel, int time = 0);

VoxelType ImarisResolutionVoxelType(hid_t imaris_id, int resolution_level,
                                    int time = 0);



// handle to /DataSet/ResolutionLevel i
// will not close file
hid_t ImarisResolutionHandle(hid_t imaris_id, int resolution_level);

hid_t ImarisChannelHandle(hid_t imaris_id, int resolution_level,
                          int channel, int time = 0);

hid_t ImarisTimePointHandle(hid_t imaris_id, int resolution_level, int time = 0);

// handle to /DataSet/ResolutionLevel i/TimePoint j/Channel k/Data
// will not close file
hid_t ImarisDatasetHandle(hid_t imaris_id, int resolution_level,
                          int channel, int time = 0);

}




#endif //MCP3D_IMS_IO_HPP_HPP
