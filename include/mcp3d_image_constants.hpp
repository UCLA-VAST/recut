//
// Created by muyezhu on 3/2/18.
//

#ifndef MCP3D_MCP3D_IMAGE_CONSTANTS_HPP
#define MCP3D_MCP3D_IMAGE_CONSTANTS_HPP

#include <vector>

namespace mcp3d
{
static const int ZPAD_AUTO_VAL = 3;
/// tile width and height in tiffs written as tiled
static const int TIFFTILE_XDIM = 128;
static const int TIFFTILE_YDIM = 128;
static const int MAX_TIFFTILE_XDIM = 4096;
static const int MAX_TIFFTILE_YDIM = 4096;
static const int MAX_CHANNEL_NUMBER = 5;
static const int MAX_TIME_POINT_NUMBER = 1;
/// width and height of ome-tif
static const int TIFFCHUNK_XDIM = 2048;
static const int TIFFCHUNK_YDIM = 2048;
static const int TIFFCHUNK_ZDIM = 100;
static const int TIFFTILE_XPAD = 3;
static const int TIFFTILE_YPAD = 3;
static const int DEFAULT_SOMA_K = 21;
static const std::vector<double> DEFAULT_SOMA_SIGMAS = {10, 15, 25};
static const int SOMA_SEG_INTENSITY = 254;
static const int NEURITE_SEG_INTENSITY = 127;
static const int ZSCALE_AUTO = 0;
static const int ZSCALE_NONE = -1;


}

#endif //MCP3D_MCP3D_IMAGE_CONSTANTS_HPP
