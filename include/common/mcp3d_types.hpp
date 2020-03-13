//
// Created by muyezhu on 9/15/17.
//

#ifndef MCP3D_MCP3D_TYPES_HPP
#define MCP3D_MCP3D_TYPES_HPP

#include <cstdint>
#include <type_traits>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace mcp3d
{
template<typename T>
using MCPArray2DMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Aligned>;

template<typename T>
using MCPArray2D = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor + Eigen::AutoAlign>;

template<typename T>
using MCPTensor3DMap = Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>, Eigen::Aligned>;

template<typename T>
using MCPTensor3D = Eigen::Tensor<T, 3, Eigen::RowMajor + Eigen::AutoAlign>;

}
#endif //MCP3D_MCP3D_TYPES_HPP
