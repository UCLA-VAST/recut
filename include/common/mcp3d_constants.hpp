//
// Created by muyezhu on 10/21/17.
//

#ifndef MCP3D_MCP3D_CONSTANTS_HPP
#define MCP3D_MCP3D_CONSTANTS_HPP

#include <string>
#include <unordered_map>
#include <boost/filesystem.hpp>

namespace mcp3d
{
enum SWCCol
{
    id, node_type, x, y, z, radius, parent
};

static const int N_THREADS = 8;
static const int ALIGN_SIZE = 64;
}
#endif //MCP3D_MCP3D_CONSTANTS_HPP
