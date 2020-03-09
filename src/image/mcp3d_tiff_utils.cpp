//
// Created by muyezhu on 3/1/19.
//
#include <omp.h>
#include "common/mcp3d_common.hpp"
#include "mcp3d_tiff_utils.hpp"

using namespace std;

mcp3d::TiffInfo::TiffInfo(const TIFF* tif)
{
    GetTiffInfo(tif);
}

mcp3d::TiffInfo::TiffInfo(const string& tif_path_): tif_path(tif_path_)
{
    if (!mcp3d::IsFile(tif_path))
        MCP3D_INVALID_ARGUMENT(tif_path + " is not a file");

    ::TIFF* tif = TIFFOpen(tif_path.c_str(), "r");
    if (!tif)
        MCP3D_RUNTIME_ERROR(tif_path + " can not be opened as tiff image")
    GetTiffInfo(tif);
    TIFFClose(tif);
}

void mcp3d::TiffInfo::GetTiffInfo(const TIFF *tif)
{
    if (! tif) MCP3D_RUNTIME_ERROR("bad pointer to tif image");
    ::TIFF* tif_ = const_cast<TIFF*>(tif);
    TIFFGetField(tif_, TIFFTAG_IMAGEWIDTH, &image_width);
    TIFFGetField(tif_, TIFFTAG_IMAGELENGTH, &image_height);
    if (TIFFIsTiled(tif_))
    {
        is_tiled = true;
        TIFFGetField(tif_, TIFFTAG_TILEWIDTH, &tile_width);
        TIFFGetField(tif_, TIFFTAG_TILELENGTH, &tile_height);
        rows_in_strip = 0;
    }
    else
    {
        is_tiled = false;
        TIFFGetField(tif_, TIFFTAG_ROWSPERSTRIP, &rows_in_strip);
        tile_height = 0;
        tile_width = 0;
    }
    TIFFGetField(tif_, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFGetField(tif_, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    // guarding against incorrectly written samples_per_pixel fields
    if (samples_per_pixel != 1 && samples_per_pixel != 3 && samples_per_pixel != 4)
    {
        cout << "warning: invalid samples_per_pixel = " << samples_per_pixel
             << " encountered. calculating from tile / strip size. ";
        if (is_tiled)
        {
            long n_bytes = TIFFTileRowSize(const_cast<TIFF*>(tif));
            samples_per_pixel = (short)(n_bytes / tile_width / (bits_per_sample / 8));
        }
        else
        {
            long n_bytes = TIFFStripSize(const_cast<TIFF*>(tif));
            samples_per_pixel = (short)(n_bytes / rows_in_strip / image_width / (bits_per_sample / 8));
        }
        cout << "samples_per_pixel = " << samples_per_pixel << endl;
    }

    TIFFGetField(tif_, TIFFTAG_PLANARCONFIG, &planar_config);

    n_directory = 1;
    while (TIFFReadDirectory(tif_))
        ++n_directory;
}

void mcp3d::TiffInfo::ShowInfo()
{
    cout <<
         "image width: " + to_string(image_width) + "\n" +
         "image length: " + to_string(image_height) + "\n" +
         "tile width: " + to_string(tile_width) + "\n" +
         "tile width: " + to_string(tile_height) + "\n" +
         "rows in strip: " + to_string(rows_in_strip) + "\n" +
         "bits per sample:" + to_string(bits_per_sample) + "\n" +
         "samples per pixel: " + to_string(samples_per_pixel) + "\n" +
         "planar config: " + to_string(planar_config) + "\n" +
         "directory number: " + to_string(n_directory) + "\n" +
         "is tiled: " + to_string(is_tiled) << endl;
}

bool mcp3d::TiffInfo::operator==(const mcp3d::TiffInfo& rhs) const
{
    return image_width == rhs.image_width &&
           image_height == rhs.image_height &&
           bits_per_sample == rhs.bits_per_sample &&
           samples_per_pixel == rhs.samples_per_pixel &&
           n_directory == rhs.n_directory &&
           planar_config == rhs.planar_config;
}

bool mcp3d::TiffInfo::operator!=(const mcp3d::TiffInfo& rhs) const
{
    return !((*this) == rhs);
}

void mcp3d::SetCommonTags(::TIFF *tif)
{
    MCP3D_ASSERT(tif)
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, 1);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, 1);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
}

void mcp3d::SetStripTiffTags(::TIFF *tif, uint32_t img_height, uint32_t img_width,
                             uint32_t rows_in_strip, short bits_per_sample,
                             short samples_per_pixel)
{
    MCP3D_ASSERT(tif)
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img_width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img_height);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rows_in_strip);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    mcp3d::SetCommonTags(tif);
}

void mcp3d::SetTiledTiffTags(::TIFF *tif, uint32_t img_height, uint32_t img_width,
                             uint32_t tile_height, uint32_t tile_width,
                             short bits_per_sample, short samples_per_pixel)
{
    MCP3D_ASSERT(tif)
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img_width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img_height);
    TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width);
    TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    mcp3d::SetCommonTags(tif);
}

void mcp3d::SetTiffHyperStackTags(::TIFF *tif, int n_channels, int n_planes,
                                  int n_times, short samples_per_pixel,
                                  short bits_per_sample,
                                  float resolution,
                                  float min_val, float max_val)
{
    MCP3D_ASSERT(tif);
    if (samples_per_pixel != 1) MCP3D_DOMAIN_ERROR(
            "tiff hyper stack expect samples per pixel = 1")
    MCP3D_ASSERT(n_channels > 0 && n_planes > 0 && n_times > 0)
    string metadata = "ImageJ=1.50e\n";
    int n_imgs = n_channels * n_planes * n_times;
    metadata.append("images=").append(to_string(n_imgs)).append("\n");
    metadata.append("channels=").append(to_string(n_channels)).append("\n");
    metadata.append("slices=").append(to_string(n_planes)).append("\n");
    metadata.append("frames=").append(to_string(n_times)).append("\n");
    metadata.append("hyperstack=true\nmode=color\nunit=micron\nloop=false");
    if (resolution > 0)
        metadata.append("spacing=").append(to_string(resolution)).append("\n");
    if (max_val < 0)
    {
        if (bits_per_sample == 8)
            max_val = float(UINT8_MAX);
        else
            max_val = float(UINT16_MAX);
    }
    metadata.append("min=").append(to_string(min_val)).append("\n");
    metadata.append("max=").append(to_string(max_val)).append("\n");
    TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, metadata);
}

string mcp3d::GetTiffPath(const vector<string> &img_paths, int pos,
                          bool is_full_path, const string &common_dir)
{
    if ((size_t)pos >= img_paths.size()|| pos < 0)
    MCP3D_OUT_OF_RANGE("vector index out of range")
    string img_path(img_paths[pos]);
    if (!is_full_path)
        img_path = mcp3d::JoinPath({common_dir, img_path});
    return img_path;
}

void mcp3d::VerifyTiffSequence(const vector<string> &img_paths, bool is_full_path,
                               const string& common_dir)
{
    if (!mcp3d::IsFile(GetTiffPath(img_paths, 0, is_full_path, common_dir)))
        MCP3D_OS_ERROR(GetTiffPath(img_paths, 0, is_full_path, common_dir) + " is not a file")
    size_t path_len = img_paths[0].size();
    mcp3d::TiffInfo tiff_info_first(GetTiffPath(img_paths, 0, is_full_path, common_dir));
    bool is_mpi = mcp3d::MPIInitialized();
    int n_threads = is_mpi ? 1 : min(mcp3d::DefaultNumThreads(), (int)img_paths.size());
    int n_imgs_per_thread = (int)img_paths.size() / n_threads;
    mcp3d::MultiThreadExceptions me;
    #pragma omp parallel num_threads(n_threads)
    {
        me.RunAndCaptureException([] {CHECK_PARALLEL_MODEL});
        if (me.HasCapturedException())
        {
        #pragma omp cancel parallel
        }
        me.RunAndCaptureException([&]
        {
            int thread_id = omp_get_thread_num();
            int z_begin = thread_id * n_imgs_per_thread,
                z_end = min(z_begin + n_imgs_per_thread, (int)img_paths.size());
            for (int z = z_begin; z < z_end; ++z)
            {
                if (!mcp3d::IsFile(GetTiffPath(img_paths, z, is_full_path, common_dir)))
                    MCP3D_OS_ERROR(img_paths[z] + ": not a valid image path");
                //if (img_paths[z].size() != path_len)
                    //MCP3D_INVALID_ARGUMENT(img_paths[z] +
                                           //": inconsistent image name length in sequence")
                mcp3d::TiffInfo tiff_info_next(GetTiffPath(img_paths, z, is_full_path, common_dir));
                if (tiff_info_first != tiff_info_next)
                  MCP3D_INVALID_ARGUMENT(img_paths[0] + " and " + img_paths[z] +
                                         " have different dimensions, "
                                         "pixel data type or samples per pixel");
          }
      });
        if (me.HasCapturedException())
        {
        #pragma omp cancel parallel
        }
    }
    if (me.HasCapturedException())
        MCP3D_RETHROW(me.e_ptr())
    mcp3d::CheckTiffSupported(tiff_info_first);
}

void mcp3d::CheckTiffSupported(const mcp3d::TiffInfo &tiff_info)
{
    if (tiff_info.bits_per_sample > 32)
    MCP3D_DOMAIN_ERROR("currently only support 8 bit, 16 bit or 32 bit images");
    if (tiff_info.samples_per_pixel > 1)
        cout << "currently only support gray images, rgb iamges will be read as gray" << endl;
}

void mcp3d::CheckTiffSupported(const std::string &tif_path)
{
    mcp3d::TiffInfo tiff_info(tif_path);
    mcp3d::CheckTiffSupported(tiff_info);
}
