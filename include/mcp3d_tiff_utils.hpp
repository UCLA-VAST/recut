//
// Created by muyezhu on 3/1/19.
//

#ifndef MCP3D_MCP3D_TIFF_UTILS_HPP
#define MCP3D_MCP3D_TIFF_UTILS_HPP

#include <tiff.h>
#include <tiffio.h>

namespace mcp3d
{

struct TiffInfo
{
    TiffInfo() : image_width(0), image_height(0),
                 tile_width(0), tile_height(0),
                 rows_in_strip(0),
                 bits_per_sample(0), samples_per_pixel(0),
                 sample_format(-1),
                 planar_config(-1), n_directory(0), is_tiled(true) {};
    /// obtain MCPTIFFInfo fields.
    /// if multiple directory exists for the tif image,
    /// n_directory will be updated accordingly,
    /// but other fields will use values of first directory
    TiffInfo(const std::string& tif_path);

    TiffInfo(const ::TIFF* tif);

    void ShowInfo();

    void GetTiffInfo(const TIFF* tif);

    // return true if image_height, image_width, bits_per_sample,
    // samples_per_pixel, n_directory and planar_config are identical
    // between two TiffInfo structs
    bool operator==(const TiffInfo &rhs) const;

    bool operator!=(const TiffInfo &rhs) const;
    /// see documentation on sample format above
    bool UnknownSampleFormat() const
    {return sample_format > 6 || sample_format < 1;}

    uint32_t image_width, image_height;
    uint32_t tile_width, tile_height;
    uint32_t rows_in_strip;
    short bits_per_sample, samples_per_pixel, sample_format;
    short planar_config;
    short n_directory;
    bool is_tiled;
    std::string tif_path;
};

/// set TIFFTAG_PLANARCONFIG = 1, TIFFTAG_ORIENTATION = 1,
/// TIFFTAG_PHOTOMETRIC = 1. these tags must be set for opencv to correctly
/// read an image
void SetCommonTags(::TIFF *tif);

void SetTiledTiffTags(::TIFF *tif, uint32_t img_height, uint32_t img_width,
                      uint32_t tile_height, uint32_t tile_width,
                      short bits_per_sample = 8, short samples_per_pixel = 1);

void SetStripTiffTags(::TIFF *tif, uint32_t img_height, uint32_t img_width,
                      uint32_t rows_in_strip, short bits_per_sample,
                      short samples_per_pixel);

/*
 * see ImageJ hyperstack plugin for some details:
 * https://imagej.nih.gov/ij/developer/api/ij/plugin/HyperStackConverter.html
 * most relevant: The default "xyczt" order is used if 'order' is null
 * DimensionOrder attribute that specifies the rasterization order of the
 * image planes. For example, XYZTC means that there is a series of image
 * planes with the Z axis varying fastest, followed by T, followed by C
 * (e.g. if a XYZTC dataset contains two focal planes, three time points and
 * two channels, the order would be: Z0-T0-C0, Z1-T0-C0, Z0-T1-C0, Z1-T1-C0,
 * Z0-T2-C0, Z1-T2-C0, Z0-T0-C1, Z1-T0-C1, Z0-T1-C1, Z1-T1-C1, Z0-T2-C1,
 * Z1-T2-C1).
 * example hyperstack tiff_info output:
 * TIFF Directory at offset 0xe233bb25 (3795041061)
 * Subfile Type: (0 = 0x0)
 * Image Width: 1400 Image Length: 1400
 * Resolution: 3.55544, 3.55544 (unitless)
 * Bits/Sample: 16
 * Compression Scheme: None
 * Photometric Interpretation: min-is-black
 * Samples/Pixel: 1
 * Rows/Strip: 1400
 * Planar Configuration: single image plane
 * ImageDescription: ImageJ=1.50e
 * images=968  // this is channels * slices * time points
 * channels=4
 * slices=242  // this is z level
 * hyperstack=true
 * mode=color
 * unit=micron
 * spacing=3.0
 * loop=false
 * min=0.0
 * max=65535.0
 * The whole ImageDescription string is the OME-XML portion of OME-TIF
*/
void SetTiffHyperStackTags(::TIFF *tif, int n_channels, int n_planes,
                           int n_times = 1, short samples_per_pixel = 1,
                           short bits_per_sample = 8,
                           float resolution = -1.0f,
                           float min_val = 0.0f,
                           float max_val = -1.0f);

std::string GetTiffPath(const std::vector<std::string> &img_paths, int pos,
                        bool is_full_path, const std::string& common_dir);

void VerifyTiffSequence(const std::vector<std::string> &img_paths,
                        bool is_full_path, const std::string& common_dir);

void CheckTiffSupported(const TiffInfo &tiff_info);

void CheckTiffSupported(const std::string &tif_path);

bool CompTiffTileId(const std::string &tif_path1, const std::string &tif_path2);

}


#endif //MCP3D_MCP3D_TIFF_UTILS_HPP
