//
// Created by muyezhu on 10/4/18.
//

#ifndef MCP3D_HDF5_IO_HPP
#define MCP3D_HDF5_IO_HPP

#include <cstring>
#include <hdf5.h>

namespace mcp3d
{

bool IsHdf5File(const std::string& file_path);

/// read data with
template <typename VType>
void ReadHdf5Data(VType* view_ptr, const std::string& hdf5_file_path,
                  int level, int x_start, int x_end, int y_start, int y_end,
                  int z_start, int z_end);
}

#endif //MCP3D_HDF5_IO_HPP
