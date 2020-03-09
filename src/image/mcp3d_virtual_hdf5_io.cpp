//
// Created by muyezhu on 1/30/19.
//
#include <boost/algorithm/string/predicate.hpp>
#include "common/mcp3d_common.hpp"
#include "mcp3d_virtual_hdf5_io.hpp"

using namespace std;

bool mcp3d::IsHdf5File(const std::string &file_path)
{
    if (!mcp3d::IsFile(file_path))
        return false;
    hid_t handle = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    H5Fclose(handle);
    return handle >= 0;
}
