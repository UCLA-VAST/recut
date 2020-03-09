//
// Created by muyezhu on 2/12/18.
//

#ifndef MCP3D_MCP3D_TIFF_FORMAT_HPP
#define MCP3D_MCP3D_TIFF_FORMAT_HPP

#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mcp3d_image_common.hpp"
#include "mcp3d_image.hpp"
#include "mcp3d_image_formats.hpp"
#include "mcp3d_tiff_partial_io.hpp"

namespace mcp3d
{
/// this class supports sequence of tiff images of single channel and time point
/// aka each tiff image in the sequence is presumed to have a single directory
/// containing all data of one z plane
/// single image file that contain multiple z level, channels or time points
/// should be organized as tif stack or hdf
class MTiffFormat: public MImageFormats
{
public:
    using json = nlohmann::json;

    MTiffFormat(): MImageFormats(FileFormats::TIFF)  {};

    bool CanRead() override { return true; }

    bool CanWrite() override { return true; }

    bool CanReadPartial() override { return true; }

    bool CanWriteInChunk() override { return false; }

    bool CanWriteInChunkParallel() override { return false; }

    MImageInfo ReadImageInfo(const std::vector<std::string> &image_paths,
                             bool is_level0_image,
                             bool is_full_path,
                             const std::string &common_dir) override;

    // needed to find ReadImageInfo overload in MImageFormats
    using MImageFormats::ReadImageInfo;

    void ValidateImageInfo(const MImageInfo& image_info) override;

    // rgb image or views with non unit stridesalong xy axes will only return
    // black background, due to a intermediate pointer being created, which
    // MImageIO can not prefill with indicated background
    void ReadData(MImage &image) override;

    void ReadTiffImage(const std::string &img_path, cv::Mat &m);

    // does not allocation memory
    template <typename VType>
    void ReadPartialTiff(const std::string &img_path, VType *buffer,
                         const MImage &image);

    void WriteViewVolume(const MImage &img, const std::string &out_dir,
                         const std::string &img_name_prefix) override {};

    int ImagePyramidChunkNumber(const MImage &img, int parent_level) override;

    // divide image chunks to write evenly among threads. output is tiled tiff
    // maintain in memory tile strips of paren data
    // even if on disk storage of tiff image has rgb pixel types the pyramid will
    // be generated in grey
    void WriteImagePyramidOMP(MImage& image, int parent_level, bool multi_threading) override;

    #if MCP3D_MPI_BUILD

    void WriteImagePyramidMPI(MImage &image, int parent_level, bool abort_all_on_fail,
                              const std::string &err_log_path,
                              const std::string &out_log_path,
                              MPI_Comm comm_writer) override;

    #endif

private:
    // do not call alone. does not ensure memory allocation for image data
    template <typename VType>
    void ReadDataImpl(MImage &image);

    void WriteImagePyramidImpl(const MImage& image, int parent_level, int z_level);
};

// this method does not manage memory of MImage
template <typename VType>
void MTiffFormat::ReadDataImpl(MImage &image)
{
    const MImageInfo& image_info = image.image_info();
    const MImageView& view = image.selected_view();

    // dimension of selected image view at current pyr_level, accounting for strides
    int view_xdim = view.view_xdim(),
        view_ydim = view.view_ydim();
    // dimension of selected image view at current pyr_level, assuming unit strides
    int view_xextent = view.view_level_extents()[2],
        view_yextent = view.view_level_extents()[1],
        view_zextent = view.view_level_extents()[0];
    int view_xstride = view.view_level_strides()[2],
        view_ystride = view.view_level_strides()[1],
        view_zstride = view.view_level_strides()[0];
    // the image portion covered by view assuming unit strides will be read first
    // if strides are not all unit, a transfer will happen according to the strides
    long n_plane_voxels = view.VoxelsPerPlane();
    std::unique_ptr<VType[]> ptr_temp;
    bool rgb_unit_stride = false;
    for (int z_image = view.view_level_offsets()[0], z_view = 0;
         z_image < view.view_level_offsets()[0] + view_zextent;
         z_image += view_zstride, ++z_view)
    {
        // if z value out of boundary, exist loop. background pixels are filled
        // in MImageIO
        if (z_image >= image_info.xyz_dims(image.selected_pyr_level())[0])
            break;
        VType* ptr_img = image.Plane<VType>(0, 0, z_view);
        std::string img_path(image_info.image_path(image.selected_pyr_level(), z_image));
        mcp3d::TiffInfo tiff_info(img_path);
        // if unit stride and 1 sample per pixel in tiff image, read into ptr_img
        if (view.view_is_unit_strided("xy") && tiff_info.samples_per_pixel == 1)
        {
            ReadPartialTiff<VType>(img_path, ptr_img, image);
        }
        // otherwise read to an intermediate buffer, then transfer
        else
        {
            bool fill_background = image.selected_view().OutOfPyrImageBoundary() ||
                                   image.selected_view().PartiallyOutOfPyrImageBoundary();
            // check if rgb, if so convert to gray before copy
            if (tiff_info.samples_per_pixel > 1)
            {
                long ptr_temp_addr = z_view * n_plane_voxels;
                if (z_image == view.view_level_offsets()[0])
                    ptr_temp = std::make_unique<VType[]>((size_t)view.VoxelsPerVolume());
                if (fill_background)
                    memset(ptr_temp.get(), 0, (size_t)view.VoxelsPerVolume() *
                                              (size_t)view.BytesPerVoxel());
                MCP3D_ASSERT(ptr_temp.get())
                std::unique_ptr<VType[]> ptr_cv = std::make_unique<VType[]>((size_t)view_xextent *
                                                                   (size_t)view_yextent *
                                                                   (size_t)tiff_info.samples_per_pixel);
                MCP3D_ASSERT(ptr_cv.get())
                ReadPartialTiff<VType>(img_path, ptr_cv.get(), image);
                cv::Mat m_in(view_yextent, view_xextent,
                             mcp3d::TypeToCVTypes<VType>(tiff_info.samples_per_pixel),
                             ptr_cv.get());
                cv::Mat m_out(view.view_ydim(), view.view_xdim(),
                              mcp3d::TypeToCVTypes<VType>(1),
                              ptr_temp.get() + ptr_temp_addr);
                SERIALIZE_OPENCV_MPI
                #if CV_MAJOR_VERSION < 4
                    cv::cvtColor(m_in, m_out, CV_RGB2GRAY);
                #else
                    cv::cvtColor(m_in, m_out, cv::COLOR_RGB2GRAY);
                #endif
                // if unit stride, do unique pointer ownership transfer
                if (view.view_is_unit_strided("xy"))
                    rgb_unit_stride = true;
            }
            else
            {
                size_t n_elements = (size_t)view_xextent *
                                    (size_t)view_yextent *
                                    (size_t)tiff_info.samples_per_pixel;
                ptr_temp = std::make_unique<VType[]>(n_elements);
                MCP3D_ASSERT(ptr_temp.get())
                if (fill_background)
                    memset(ptr_temp.get(), 0, n_elements * view.BytesPerVoxel());
                ReadPartialTiff<VType>(img_path, ptr_temp.get(), image);
                mcp3d::CopyDataVolume<VType>(ptr_img,
                                             {1, view_ydim, view_xdim},
                                             ptr_temp.get(),
                                             {1, view_yextent, view_xextent},
                                             mcp3d::MImageBlock{},
                                             mcp3d::MImageBlock({}, {}, {1, view_ystride, view_xstride}));
            }
        }
    }
    // unique pointer ownership transfer
    if (rgb_unit_stride)
    {
        std::vector<std::unique_ptr<VType[]>> ptr_temp_vec;
        ptr_temp_vec.push_back(std::move(ptr_temp));
        image.AcquireData<VType>(ptr_temp_vec);
    }
}

template <typename VType>
void MTiffFormat::ReadPartialTiff(const std::string &img_path, VType *buffer,
                                  const MImage &image)
{
    MCP3D_ASSERT(buffer)
    if (!mcp3d::IsFile(img_path))
        MCP3D_OS_ERROR(img_path + " is not a file")
    const MImageView& view = image.selected_view();
    mcp3d::TransferToSubimg(img_path, buffer,
                            view.view_level_extents()[1],
                            view.view_level_extents()[2],
                            view.view_level_offsets()[2],
                            view.view_level_offsets()[1]);
}

}

#endif //MCP3D_MCP3D_TIFF_FORMAT_HPP
