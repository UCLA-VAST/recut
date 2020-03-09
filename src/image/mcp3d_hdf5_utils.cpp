//
// Created by muyezhu on 3/13/19.
//
#include <3rd_party/hdf5/include/hdf5.h>
#include "common/mcp3d_common.hpp"
#include "mcp3d_hdf5_utils.hpp"

using namespace std;

// herr_t (*H5L_iterate_t)( hid_t g_id, const char *name,
//                          const H5L_info_t *info, void *op_data)
// callback function type defined by hdf5 library
herr_t GetGroupObjectNames(hid_t g_id, const char *name,
                           const H5L_info_t *info, void* op_data)
{
    ((unordered_set<string>*)op_data)->insert(string(name));
    return 0;
}

vector<string> mcp3d::ObjectNamesInGroup(hid_t group_id)
{
    MCP3D_ASSERT(group_id >= 0)
    // though highly unlikely, guard against multi-paths reachable groups
    unordered_set<string> unique_group_names;
    H5G_info_t group_info;
    H5Gget_info(group_id, &group_info);
    herr_t success;
    hsize_t idx = 0;
    while (idx < group_info.nlinks)
    {
        success = H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                             &idx, GetGroupObjectNames, &unique_group_names);
        if (success < 0)
        MCP3D_RUNTIME_ERROR("error occured iterating through Imaris group")
    }
    vector<string> group_names(unique_group_names.begin(),
                               unique_group_names.end());
    sort(group_names.begin(), group_names.end());
    return group_names;
}

int mcp3d::Hdf5AttributeValue(hid_t object_id, const char *attribute_name,
                              std::unique_ptr<uint8_t[]> &buffer)
{
    MCP3D_ASSERT(object_id >= 0)
    hid_t attribute_id = H5Aopen(object_id, attribute_name, H5P_DEFAULT);
    if (attribute_id < 0)
        MCP3D_RUNTIME_ERROR("can not open attribute " + string(attribute_name))
    H5A_info_t attribute_info;
    herr_t success = H5Aget_info(attribute_id, &attribute_info);
    MCP3D_ASSERT(success >= 0)
    hsize_t n_bytes = attribute_info.data_size;
    buffer = make_unique<uint8_t[]>(n_bytes);
    hid_t type_id = H5Aget_type(attribute_id);
    MCP3D_ASSERT(type_id >= 0)
    success = H5Aread(attribute_id, type_id, buffer.get());
    if (success < 0)
        MCP3D_RUNTIME_ERROR("failed to read attribute " + string(attribute_name))
    size_t n_type_bytes = H5Tget_size(type_id);
    int n_elements = (int)(n_bytes / n_type_bytes);
    H5Tclose(type_id);
    return n_elements;
}

mcp3d::VoxelType mcp3d::Hdf5DatasetVoxelType(hid_t dataset_id)
{
    MCP3D_ASSERT(mcp3d::IsHdf5Dataset(dataset_id))
    hid_t data_type_id = H5Dget_type(dataset_id);
    MCP3D_ASSERT(data_type_id >= 0)
    H5T_sign_t data_type_sign = H5Tget_sign(data_type_id);
    H5T_class_t data_type_class = H5Tget_class(data_type_id);
    size_t data_type_size = H5Tget_size(data_type_id);
    if (data_type_class == H5T_INTEGER)
    {
        if (data_type_size == 1)
        {
            if (data_type_sign == H5T_SGN_NONE)
                return mcp3d::VoxelType::M8U;
            else
                return mcp3d::VoxelType::M8S;
        }
        else if (data_type_size == 2)
        {
            if (data_type_sign == H5T_SGN_NONE)
                return mcp3d::VoxelType::M16U;
            else
                return mcp3d::VoxelType::M16S;
        }
        else if (data_type_size == 4)
        {
            if (data_type_sign == H5T_SGN_NONE)
                return mcp3d::VoxelType::M32U;
            else
                return mcp3d::VoxelType::M32S;
        }
        else if (data_type_size == 8)
        {
            if (data_type_sign == H5T_SGN_NONE)
                return mcp3d::VoxelType::M64U;
            else
                return mcp3d::VoxelType::M64S;
        }
        else
            MCP3D_RUNTIME_ERROR("unsupported hdf5 dataset datatype")
    }
    else if (data_type_class == H5T_FLOAT)
    {
        if (data_type_size == 4)
            return mcp3d::VoxelType::M32F;
        else if (data_type_size == 8)
            return mcp3d::VoxelType::M64F;
        else
            MCP3D_RUNTIME_ERROR("unsupported hdf5 dataset datatype")
    }
    else
        MCP3D_RUNTIME_ERROR("unsupported hdf5 dataset datatype")
}

hid_t mcp3d::Hdf5DataType(mcp3d::VoxelType voxel_type)
{
    if (voxel_type == mcp3d::VoxelType::M8U)
        return H5T_NATIVE_UCHAR;
    else if (voxel_type == mcp3d::VoxelType::M8S)
        return H5T_NATIVE_CHAR;
    else if (voxel_type == mcp3d::VoxelType::M16U)
        return H5T_NATIVE_USHORT;
    else if (voxel_type == mcp3d::VoxelType::M16S)
        return H5T_NATIVE_SHORT;
    else if (voxel_type == mcp3d::VoxelType::M32U)
        return H5T_NATIVE_UINT32;
    else if (voxel_type == mcp3d::VoxelType::M32S)
        return H5T_NATIVE_INT32;
    else if (voxel_type == mcp3d::VoxelType::M64U)
        return H5T_NATIVE_UINT64;
    else if (voxel_type == mcp3d::VoxelType::M64S)
        return H5T_NATIVE_INT64;
    else if (voxel_type == mcp3d::VoxelType::M32F)
        return H5T_NATIVE_FLOAT;
    else if (voxel_type == mcp3d::VoxelType::M64F)
        return H5T_NATIVE_DOUBLE;
    else
        MCP3D_RUNTIME_ERROR("voxel type can not be unknown")
}

string mcp3d::Hdf5DatasetVoxelTypeString(hid_t dataset_id)
{
    mcp3d::VoxelType voxel_type = mcp3d::Hdf5DatasetVoxelType(dataset_id);
    string result = mcp3d::VoxelTypeEnumToStr(voxel_type);
    return result;
}

vector<int> mcp3d::Hdf5DatasetChunkDimensions(hid_t dataset_id, int ndims)
{
    if (!mcp3d::IsChunkedHdf5Dataset(dataset_id))
        return vector<int>({0, 0, 0});
    hid_t property_id = mcp3d::DatasetCreationPropertyHandle(dataset_id);
    unique_ptr<hsize_t []> dims(new hsize_t [ndims]);
    H5Pget_chunk(property_id, ndims, dims.get());
    vector<int> chunk_dims;
    for (int i = 0; i < ndims; ++i)
        chunk_dims.push_back((int)dims[i]);
    H5Pclose(property_id);
    return chunk_dims;
}

bool mcp3d::IsHdf5Dataset(hid_t dataset_id)
{
    return mcp3d::HdfObjectType(dataset_id) == H5O_TYPE_DATASET;
}

bool mcp3d::IsChunkedHdf5Dataset(hid_t dataset_id)
{
    if (!mcp3d::IsHdf5Dataset(dataset_id))
        return false;
    hid_t property_id = mcp3d::DatasetCreationPropertyHandle(dataset_id);
    bool is_chunked_dataset = H5Pget_layout(property_id) == H5D_CHUNKED;
    H5Pclose(property_id);
    return is_chunked_dataset;
}

void mcp3d::SetHdf5DatasetZlibDeflate(hid_t dataset_id, int deflation)
{
    if (!mcp3d::IsHdf5Dataset(dataset_id))
    {
        MCP3D_MESSAGE("identifier is not a dataset identifier. do nothing")
        return;
    }
    if (!mcp3d::Hdf5ZlibFilterAvailable())
    {
        MCP3D_MESSAGE("zlib filter not available. do nothing")
        return;
    }
    if (deflation >= 10 || deflation < 0)
    {
        MCP3D_MESSAGE("invalid deflate value. do nothing")
        return;
    }

    hid_t property_id = mcp3d::DatasetCreationPropertyHandle(dataset_id);
    H5Pset_deflate(property_id, (uint32_t)deflation);
    MCP3D_ASSERT(H5Pall_filters_avail(property_id))
    H5Pclose(property_id);
}

bool mcp3d::Hdf5ZlibFilterAvailable()
{
    if (!H5Zfilter_avail(H5Z_FILTER_DEFLATE))
        return false;
    uint32_t filter_config;
    H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_config);
    return (H5Z_FILTER_CONFIG_DECODE_ENABLED & filter_config) &&
           (H5Z_FILTER_CONFIG_ENCODE_ENABLED & filter_config);
}

void mcp3d::CloseHdfObject(hid_t object_id)
{
    MCP3D_ASSERT(object_id >= 0)
    H5O_info_t object_info;
    herr_t success = H5Oget_info(object_id, &object_info);
    MCP3D_ASSERT(success >= 0)
    H5O_type_t type = object_info.type;
    if (type == H5O_TYPE_DATASET)
        H5Dclose(object_id);
    else if (type == H5O_TYPE_GROUP)
        H5Gclose(object_id);
    else if (type == H5O_TYPE_NAMED_DATATYPE)
        H5Tclose(object_id);
    else
        MCP3D_MESSAGE("warning: does not know how to close object, "
                      "leaving handle open")
}

H5O_type_t mcp3d::HdfObjectType(hid_t object_id)
{
    MCP3D_ASSERT(object_id >= 0)
    H5O_info_t object_info;
    herr_t success = H5Oget_info(object_id, &object_info);
    MCP3D_ASSERT(success >= 0)
    return object_info.type;
}

hid_t mcp3d::Hdf5Handle(const string &hdf5_path)
{
    MCP3D_ASSERT(mcp3d::IsFile(hdf5_path))
    hid_t handle = H5Fopen(hdf5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    MCP3D_ASSERT(handle >= 0)
    return handle;
}

hid_t mcp3d::DatasetCreationPropertyHandle(hid_t dataset_id)
{
    MCP3D_ASSERT(mcp3d::IsHdf5Dataset(dataset_id))
    hid_t property_id = H5Dget_create_plist(dataset_id);
    MCP3D_ASSERT(property_id >= 0)
    return property_id;
}

