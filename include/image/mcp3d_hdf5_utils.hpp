//
// Created by muyezhu on 3/13/19.
//

#ifndef MCP3D_MCP3D_HDF5_UTIL_HPP
#define MCP3D_MCP3D_HDF5_UTIL_HPP

#include <memory>
#include <hdf5.h>
#include <image/mcp3d_voxel_types.hpp>

/// will close hid_t handles created by the function, but will not close
/// handles received as argument
namespace mcp3d
{
std::vector<std::string> ObjectNamesInGroup(hid_t group_id);

// retrieve attribute value into buffer. returns number of elements in
// attribute value. caller should know the attribute value element datatype
// in order to interpret content in buffer
int Hdf5AttributeValue(hid_t object_id, const char *attribute_name,
                       std::unique_ptr<uint8_t[]> &buffer);

VoxelType Hdf5DatasetVoxelType(hid_t dataset_id);

hid_t Hdf5DataType(VoxelType voxel_type);

std::string Hdf5DatasetVoxelTypeString(hid_t dataset_id);

// zyx dimensions of dataset chunk. if dataset is not chunked, return 0, 0, 0
std::vector<int> Hdf5DatasetChunkDimensions(hid_t dataset_id, int ndims = 3);

void SetHdf5DatasetZlibDeflate(hid_t dataset_id, int deflation);

bool Hdf5ZlibFilterAvailable();

void CloseHdfObject(hid_t object_id);

H5O_type_t HdfObjectType(hid_t object_id);

bool IsHdf5Dataset(hid_t dataset_id);

bool IsChunkedHdf5Dataset(hid_t dataset_id);

hid_t Hdf5Handle(const std::string &hdf5_path);

hid_t DatasetCreationPropertyHandle(hid_t dataset_id);
}

#endif //MCP3D_MCP3D_HDF5_UTIL_HPP
