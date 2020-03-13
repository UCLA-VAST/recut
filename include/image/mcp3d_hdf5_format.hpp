//
// Created by muyezhu on 1/30/19.
//

#ifndef MCP3D_MCP3D_HDF5_FORMAT_HPP_HPP
#define MCP3D_MCP3D_HDF5_FORMAT_HPP_HPP

#include <hdf5.h>
#include "mcp3d_image_info.hpp"
#include "mcp3d_image_formats.hpp"

/// the hdf5 format will support reading .ims, reading and writing .hdf5 files,
/// accessed by a single virtual hdf5 file. this allows parallel processing
/// and easier syntax. the virtual file should have extension .virtual.hdf5
/// note that the .ims files are often assumed to have a single time point 0
namespace mcp3d
{

class MHdf5Format: public MImageFormats
{
public:
    using json = nlohmann::json;

    MHdf5Format(): MImageFormats(FileFormats::HDF5)  {};

    bool CanRead() override { return true; }

    bool CanWrite() override { return true; }

    bool CanReadPartial() override { return true; }

    bool CanWriteInChunk() override { return true; }

    bool CanWriteInChunkParallel() override { return false; }

    // image_paths can be a single .ims / .hdf5 file or multiple .hdf5 file
    MImageInfo ReadImageInfo(const std::vector<std::string> &image_paths,
                             bool is_level0_image,
                             bool is_full_path,
                             const std::string &common_dir) override;

    void ValidateImageInfo(const mcp3d::MImageInfo& image_info) override{};

    void ReadData(MImage &image) override;

    void WriteViewVolume(const MImage &img, const std::string &out_dir,
                         const std::string &img_name_prefix) override{};

    int ImagePyramidChunkNumber(const MImage &img, int parent_level) override{};

    void WriteImagePyramidOMP(MImage& image, int parent_level, bool multi_threading) override {};

#if MCP3D_MPI_BUILD

    virtual void WriteImagePyramidMPI(MImage &image, int parent_level,
                                      bool abort_all_on_fail,
                                      const std::string &err_log_path,
                                      const std::string &out_log_path,
                                      MPI_Comm comm_writer) override {};

#endif

private:
    // image_path is full path
    MImageInfo ReadImageInfoImaris(const std::string &image_path,
                                   bool is_level0_image);

    MImageInfo ReadImagePyrInfoImaris(const std::string &image_path,
                                      int resolution_level,
                                      bool is_level0_image);

    MImageInfo ReadImageInfoVirtualHdf5(const std::string &image_path,
                                        bool is_level0_image);

    void ReadDataset(hid_t dataset_id, MImage &image, int volume_id = 0);

    hid_t MemoryDataSpace(const MImage& image);

    hid_t FileDataSpace(hid_t dataset_id, const MImage& image);

};

}

#endif //MCP3D_MCP3D_HDF5_FORMAT_HPP_HPP
