//
// Created by muyezhu on 2/12/18.
//

#ifndef MCP3D_MCP3D_IMAGE_FORMATS_HPP
#define MCP3D_MCP3D_IMAGE_FORMATS_HPP

#include "mcp3d_voxel_types.hpp"
#include "mcp3d_image_info.hpp"
#include "mcp3d_image_view.hpp"
#include "mcp3d_image.hpp"
#include "project_structure/mcp3d_pyramid_layout.hpp"


namespace mcp3d
{

class MImage;

class MImageFormats
{
public:
    explicit MImageFormats(FileFormats format): format_(format) {}

    virtual bool CanRead() = 0;

    virtual bool CanWrite() = 0;

    virtual bool CanReadPartial() = 0;

    /// can modify data chunk without modifying data outside of the chunk
    virtual bool CanWriteInChunk() = 0;

    virtual bool CanWriteInChunkParallel() = 0;

    /// this functions do not attempt to read __image_info__.json, but
    /// directly query images on disk. json reading is handled by MImageIO
    /// the directories also need not conforming to project structure. the
    /// project structure is also handled by MImageIO. images under the given
    /// directory with extension consistent with ext_str are passed to
    /// ReadImageInfo(const std::vector<std::string> &, bool), which perform
    /// format specific task. empty MImageInfo is returned directly if no image
    /// with ext_str is found in pyr_level_dir
    MImageInfo ReadImageInfo(const std::string &pyr_level_dir,
                             bool is_root_image,
                             const std::string &format_ext);

    /// img_paths should already be sorted
    virtual MImageInfo ReadImageInfo(const std::vector<std::string> &img_paths,
                                     bool is_root_image, bool is_full_path,
                                     const std::string &common_dir) = 0;

    /// validations common to all formats are performed by MImageIO.
    /// the validations performed by derived classes of MImageFormats are format
    /// specific
    virtual void ValidateImageInfo(const mcp3d::MImageInfo& image_info) = 0;

    virtual void ReadData(MImage &img) = 0;

    virtual void WriteViewVolume(const MImage &img, const std::string &out_dir,
                                 const std::string &img_name_prefix) = 0;

    /// number of individual chunks output by write pyramid operation
    /// does not check if child level exceeds maximum allowed image pyramid level
    /// does not check if parent level exists (MImageIO checks this)
    virtual int ImagePyramidChunkNumber(const MImage &img, int parent_level) = 0;

    virtual void WriteImagePyramidOMP(MImage& image, int parent_level,
                                      bool multi_threading) = 0;

    bool CorrectFileExtension(const std::string& ext)
        { return format_ == mcp3d::FileFormatExtToEnum(ext); }

    #if MCP3D_MPI_BUILD

    virtual void WriteImagePyramidMPI(MImage &image, int parent_level,
                                      bool abort_all_on_fail,
                                      const std::string &err_log_path,
                                      const std::string &out_log_path,
                                      MPI_Comm comm_writer) = 0;

    #endif

protected:
    FileFormats format_;
};
}

#if MCP3D_MPI_BUILD

#include <mpi.h>

#endif

#endif //MCP3D_MCP3D_IMAGE_FORMATS_HPP
