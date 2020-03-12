//
// Created by muyezhu on 2/12/18.
//
#include <iostream>
#include <memory>
#include <type_traits>
#include <map>
#include "mcp3d_image_common.hpp"
#include "mcp3d_image_view.hpp"
#include "mcp3d_image.hpp"
#include "mcp3d_image_formats.hpp"
#include "mcp3d_tiff_format.hpp"

#ifndef MCP3D_MCP3D_IMAGE_IO_HPP
#define MCP3D_MCP3D_IMAGE_IO_HPP

namespace mcp3d
{
class MImageFormats;

class MImageIO
{
public:
    MImageIO();

    /// parse image root directory from given directory, read image info from
    /// image root directory
    MImageInfo ReadImageInfo(const std::string &img_root_dir);

    void RefreshImageInfo(MImage& image);

    /// (1) assert image_info.PyramidsStructureValid
    /// (2) assert image_info.AllPathsExist
    /// (3) calls ValidateImageInfo with the MImageFormats class of the level 0 image
    void ValidateImageInfo(const MImageInfo& image_info);

    template<typename VType>
    static void FillViewWithBackground(MImage& image, const MImageBlock& view_block,
                                       bool black_background = true);

    void ReadData(MImage &image, bool black_background = true);

    void WriteViewVolume(const MImage &image, const std::string &out_dir,
                         const std::string &img_name_prefix,
                         FileFormats write_format);

    int ImagePyramidChunkNumber(const MImage &img, FileFormats output_format,
                                int parent_level);

    /// ui should ask user if over writing existing pyramid levels.
    /// this function will assume it should delete existing levels and make anew
    /// parent level and child level need not have the same ImageFormat
    /// at parent level the ImageFormats class must support reading, at child
    /// level the ImageFormats class must support writing
    void WriteImagePyramid(MImage& image, int parent_level, bool multi_threading,
                           FileFormats write_format);

    bool ReadableFormat(FileFormats format);

    bool WritableFormat(FileFormats format);

    #if MCP3D_MPI_BUILD
    void WriteImagePyramidMPI(const std::string &img_root_dir, int parent_level,
                              FileFormats write_format, bool abort_all_on_fail,
                              MPI_Comm writer_comm = MPI_COMM_WORLD);

    #endif

private:
    /// use img_root_dir as is (no image root directory parsing performed).
    MImageInfo ReadImageInfoImpl(const std::string &img_root_dir);

    // for pyr_level_dirs under img_root_dir, if no MPyrImageInfo exists in
    /// image_info, read into pyr_level_dir. calls ValidateImageInfo
    void ReadImageInfoFromPyrDirs(const std::string &img_root_dir,
                                  MImageInfo &image_info);

    /// get a list of image paths under pyr_level_dir, and read
    /// image pyramid info from them.
    /// assumes the pyramid level has a single file type
    MImageInfo ReadImageInfoFromPyrDirFiles(const std::string &pyr_level_dir,
                                            bool is_level0_image);

    std::map<FileFormats, std::unique_ptr<MImageFormats>> io_formats_;

    // scale_z options specific to each format for MImage view setting
    std::map<FileFormats, int> formats_z_scale_start_level_;
};

}

template<typename VType>
void mcp3d::MImageIO::FillViewWithBackground(MImage &image,
                                             const MImageBlock& view_block,
                                             bool black_background)
{
    VType background_value = mcp3d::BackgroundValue<VType>(black_background);
    for (int i = 0; i < image.n_volumes(); ++i)
        mcp3d::SetConstantBlock<VType>(image.Volume<VType>(i),
                                       image.selected_view().view_xyz_dims(),
                                       view_block, background_value);
}


#endif //MCP3D_MCP3D_IMAGE_IO_HPP
