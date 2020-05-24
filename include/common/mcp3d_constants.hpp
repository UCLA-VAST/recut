//
// Created by muyezhu on 10/21/17.
//

#ifndef MCP3D_MCP3D_CONSTANTS_HPP
#define MCP3D_MCP3D_CONSTANTS_HPP

#include <filesystem>
#include <string>
#include <unordered_map>

namespace mcp3d {
enum SWCCol { id, node_type, x, y, z, radius, parent };

static const int N_THREADS = 8;
static const int ALIGN_SIZE = 64;
} // namespace mcp3d
#endif // MCP3D_MCP3D_CONSTANTS_HPP
