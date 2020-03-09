//
// Created by muyezhu on 2/12/18.
//
#include <algorithm>
#include <thread>
#include <tiff.h>
#include <tiffio.h>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common/mcp3d_common.hpp"
#include "mcp3d_tiff_format.hpp"

using namespace std;
using namespace boost;
using json = nlohmann::json;

mcp3d::MImageInfo mcp3d::MTiffFormat::ReadImageInfo(const vector<string> &image_paths,
                                                    bool is_level0_image,
                                                    bool is_full_path,
                                                    const string &common_dir)
{
    MCP3D_ASSERT(!image_paths.empty())
    MCP3D_TRY(mcp3d::VerifyTiffSequence(image_paths, is_full_path, common_dir);)
    vector<string> image_basenames;
    for (const auto& image_path: image_paths)
    {
        if (is_full_path)
            image_basenames.push_back(mcp3d::Basename(image_path));
        else
            image_basenames.push_back(image_path);
    }

    const string& tiff0_path = mcp3d::GetTiffPath(image_paths, 0,
                                                  is_full_path, common_dir);
    TiffInfo tif_info(tiff0_path);
    json img_pyr_info;
    img_pyr_info["format"] = "tif";
    img_pyr_info["level 0 image"] = is_level0_image ? "true" : "false";
    img_pyr_info["z dimension scaled"] = "unknown";
    string parent_dir = is_full_path ? mcp3d::ParentDir(tiff0_path) : common_dir;
    img_pyr_info["image pyramid directory"] = parent_dir;
    img_pyr_info["x dimension"] = tif_info.image_width;
    img_pyr_info["y dimension"] = tif_info.image_height;
    img_pyr_info["z dimension"] = image_paths.size();
    img_pyr_info["channels"] = 1;
    img_pyr_info["time points"] = 1;
    img_pyr_info["xchunk dimension"] = tif_info.image_width;
    img_pyr_info["ychunk dimension"] = tif_info.image_height;
    img_pyr_info["zchunk dimension"] = 1;
    img_pyr_info["dimensions order"] = "tczyx";
    img_pyr_info["image sequence"] = image_basenames;
    if (tif_info.bits_per_sample == 8)
        img_pyr_info["voxel type"] = "M8U";
    else if (tif_info.bits_per_sample == 16)
        img_pyr_info["voxel type"] = "M16U";
    else if (tif_info.bits_per_sample == 32)
        img_pyr_info["voxel type"] = "M32S";
    else
        MCP3D_DOMAIN_ERROR("only supporting uint8, uint16 and int32 tiff image sequence")
    mcp3d::MPyrImageInfo pyr_info(img_pyr_info, mcp3d::FileFormats::TIFF);
    return mcp3d::MImageInfo(pyr_info);
}

void mcp3d::MTiffFormat::ValidateImageInfo(const MImageInfo &image_info)
{
    for (int i = 0; i < image_info.n_pyr_levels(); ++i)
        if (image_info.file_format(i) == mcp3d::FileFormats::TIFF)
        {
            MCP3D_ASSERT(image_info.pyr_infos()[i].zdim() ==
                         (int)image_info.pyr_infos()[i].image_sequence().size())
            MCP3D_ASSERT(image_info.pyr_infos()[i].xchunk_dim() ==
                         image_info.pyr_infos()[i].xdim())
            MCP3D_ASSERT(image_info.pyr_infos()[i].ychunk_dim() ==
                         image_info.pyr_infos()[i].ydim())
            MCP3D_ASSERT(image_info.pyr_infos()[i].zchunk_dim() == 1)
            MCP3D_ASSERT(image_info.pyr_infos()[i].n_times() == 1)
        }
}

void mcp3d::MTiffFormat::ReadData(mcp3d::MImage &image)
{
    MCP3D_ASSERT(!image.image_info().empty())
    MCP3D_ASSERT(!image.selected_view().empty())
    if (!image.DataExists())
        image.AllocateSelection();
    MCP3D_TRY(IMAGE_SELECTION_TYPED_CALL(ReadDataImpl, image, image))
}

void mcp3d::MTiffFormat::ReadTiffImage(const string &img_path, cv::Mat &m)
{
    if (mcp3d::IsFile(img_path))
        MCP3D_OS_ERROR(img_path + " is not a file")
    mcp3d::TiffInfo tiff_info(img_path);
    if (tiff_info.n_directory > 1)
        MCP3D_DOMAIN_ERROR("does not support multi directory tiff image")
    if (tiff_info.samples_per_pixel > 3)
        MCP3D_DOMAIN_ERROR("does not support alpha channel")
    if (tiff_info.bits_per_sample != 8 &&
        tiff_info.bits_per_sample != 16 &&
        tiff_info.bits_per_sample != 32)
        MCP3D_DOMAIN_ERROR("only support 8 bit, 16 bit or 32 bit images")
    m = cv::imread(img_path, cv::IMREAD_ANYDEPTH + cv::IMREAD_GRAYSCALE);
}

int mcp3d::MTiffFormat::ImagePyramidChunkNumber(const mcp3d::MImage &img,
                                                int parent_level)
{
    return img.image_pyr_infos()[parent_level].zdim();
}

void mcp3d::MTiffFormat::WriteImagePyramidOMP(MImage &image, int parent_level,
                                              bool multi_threading)
{
    MCP3D_ASSERT(parent_level >= 0 && parent_level < image.n_pyr_levels())
    int parent_xdim = image.image_pyr_info(parent_level).xdim(),
        parent_ydim = image.image_pyr_info(parent_level).ydim(),
        parent_zdim = image.image_pyr_info(parent_level).zdim();
    int child_xdim = parent_xdim / 2,
        child_ydim = parent_ydim / 2;
    if (child_xdim * child_ydim == 0)
    {
        cout << "child level has diminished xy dimensions "
             << mcp3d::JoinVector(vector<int>({child_xdim, child_ydim}), ",")
             << ", do nothing" << endl;
        return;
    }
    bool is_mpi = mcp3d::MPIInitialized();
    if (is_mpi)
        multi_threading = false;
    if (multi_threading)
    {
        int n_threads = min(mcp3d::DefaultNumThreads(), parent_zdim),
            n_imgs_per_thread = max(1, image.zdim() / n_threads + (int)(image.zdim() % n_threads > 0));
        mcp3d::MultiThreadExceptions me;
        #pragma omp parallel num_threads(n_threads)
        {
            me.RunAndCaptureException([] {CHECK_PARALLEL_MODEL});
            if (me.HasCapturedException())
            {
                #pragma omp cancel parallel
            }
            me.RunAndCaptureException([&]{
                int thread_id = omp_get_thread_num();
                int z_begin = thread_id * n_imgs_per_thread,
                    z_end = min(z_begin + n_imgs_per_thread, parent_zdim);
                for (int z = z_begin; z < z_end; ++z)
                    WriteImagePyramidImpl(image, parent_level, z);
            });
            if (me.HasCapturedException())
            {
                #pragma omp cancel parallel
            }
        }
        if (me.HasCapturedException())
        MCP3D_RETHROW(me.e_ptr())
    }
    else
    {
        for(int z = 0; z < parent_zdim; ++z)
            WriteImagePyramidImpl(image, parent_level, z);
    }

}

void FillParentTileRow(vector<unique_ptr<mcp3d::MImage>>& parent_tile_rows_readers,
                       mcp3d::MImageBlock& parent_tile_row_view,
                       const mcp3d::MImage &image,
                       int parent_xdim, int parent_ydim,
                       int parent_level, int z_level,
                       int child_tile_row, int child_tile_ydim)
{
    uint8_t* volume_ptr;
    if (child_tile_row == 0)
    {
        // 4 rows of parent tiles
        // parent row zero is gauranteed to exist. when child tile row is zero,
        // read parent tile rows 0 -2 into reader vector positions 1 - 3
        for (int i = 0; i < 4; ++i)
        {
            parent_tile_rows_readers.push_back(make_unique<mcp3d::MImage>(image, false, false));
            if (! (i > 0 && parent_ydim > child_tile_ydim * (i - 1)))
            {
                // note: volume_ptr is created because compiler evaluates loaded_view()
                // first, Volume(0) second. in the code logic Volume(0)
                // is meant to update loaded_view() prior to the view object
                // being queried
                parent_tile_row_view.offsets_ptr()[0] = 0;
                parent_tile_row_view.offsets_ptr()[1] = 0;
                parent_tile_row_view.offsets_ptr()[2] = 0;
                parent_tile_row_view.extents_ptr()[0] = 1;
                parent_tile_row_view.extents_ptr()[1] = child_tile_ydim;
                parent_tile_row_view.extents_ptr()[2] = parent_xdim;
                parent_tile_rows_readers[i]->SelectView(parent_tile_row_view);
                parent_tile_rows_readers[i]->AllocateSelection();
                volume_ptr = parent_tile_rows_readers[i]->Volume(0);
                IMAGE_SELECTION_TYPED_CALL(mcp3d::SetConstant,
                                           (*(parent_tile_rows_readers[i])),
                                           volume_ptr,
                                           parent_tile_rows_readers[i]->selected_view().view_xyz_dims(),
                                           0)
            }
        }
        for (int i = 0; i < 3; ++i )
        {
            if (parent_ydim > child_tile_ydim * i)
            {
                /*
                parent_tile_row_view = mcp3d::MImageBlock(
                        {0, child_tile_ydim * i, z_level},
                        {parent_xdim, child_tile_ydim, 1}); */
                parent_tile_row_view.offsets_ptr()[0] = z_level;
                parent_tile_row_view.offsets_ptr()[1] = child_tile_ydim * i;
                parent_tile_row_view.offsets_ptr()[2] = 0;
                parent_tile_row_view.extents_ptr()[0] = 1;
                parent_tile_row_view.extents_ptr()[1] = child_tile_ydim;
                parent_tile_row_view.extents_ptr()[2] = parent_xdim;
                parent_tile_rows_readers[i + 1]->SelectView(
                        parent_tile_row_view, parent_level, true);
                parent_tile_rows_readers[i + 1]->ReadData(true, "quiet");
            }
        }
    }
    // non zero child tile row i: pop two MImage pointers from front, push two back,
    // read parent tile rows 2i + 1 and 2i + 2 into them
    else
    {
        rotate(parent_tile_rows_readers.begin(),
               parent_tile_rows_readers.begin() + 2,
               parent_tile_rows_readers.end());

        for (int i = 1; i < 3; ++i)
        {
            if (parent_ydim > child_tile_ydim * (2 * child_tile_row + i))
            {
                /*
                parent_tile_row_view = mcp3d::MImageBlock({0, child_tile_ydim * (2 * child_tile_row + i), z_level},
                                                          {parent_xdim, child_tile_ydim, 1}); */
                parent_tile_row_view.offsets_ptr()[0] = z_level;
                parent_tile_row_view.offsets_ptr()[1] = child_tile_ydim * (2 * child_tile_row + i);
                parent_tile_row_view.offsets_ptr()[2] = 0;
                parent_tile_row_view.extents_ptr()[0] = 1;
                parent_tile_row_view.extents_ptr()[1] = child_tile_ydim;
                parent_tile_row_view.extents_ptr()[2] = parent_xdim;
                parent_tile_rows_readers[i + 1]->SelectView(
                        parent_tile_row_view, parent_level, true);
                parent_tile_rows_readers[i + 1]->ReadData(true, "quiet");
            }
            else
            {
                parent_tile_row_view.offsets_ptr()[0] = 0;
                parent_tile_row_view.offsets_ptr()[1] = 0;
                parent_tile_row_view.offsets_ptr()[2] = 0;
                parent_tile_row_view.extents_ptr()[0] = 1;
                parent_tile_row_view.extents_ptr()[1] = child_tile_ydim;
                parent_tile_row_view.extents_ptr()[2] = parent_xdim;
                parent_tile_rows_readers[i + 1]->SelectView(
                        parent_tile_row_view);
                parent_tile_rows_readers[i + 1]->AllocateSelection();
                volume_ptr = parent_tile_rows_readers[i + 1]->Volume(0);
                IMAGE_SELECTION_TYPED_CALL(mcp3d::SetConstant,
                                           (*(parent_tile_rows_readers[i + 1])),
                                           volume_ptr,
                                           parent_tile_rows_readers[i +1]
                                           ->selected_view().view_xyz_dims(),
                                           0)
            }
        }
    }
    MCP3D_ASSERT(parent_tile_rows_readers.size() == 4)
}

void mcp3d::MTiffFormat::WriteImagePyramidImpl(const mcp3d::MImage &image,
                                               int parent_level, int z_level)
{
    string pyr_level_dir(mcp3d::JoinPath(image.image_root_dir(),
                                         mcp3d::MPyramidsLayout::PyrLevelDirName(parent_level + 1)));
    if (!mcp3d::IsDir(pyr_level_dir))
        mcp3d::MakeDirectories(pyr_level_dir);
    const mcp3d::MPyrImageInfo& parent_pyr_info = image.image_pyr_info(parent_level);
    MCP3D_ASSERT(z_level >= 0 && z_level < parent_pyr_info.zdim())
    string parent_image_path(image.image_info().image_path(parent_level, z_level));
    string child_image_path(mcp3d::JoinPath(pyr_level_dir,
                                      parent_pyr_info.image_sequence()[z_level]));

    int parent_xdim = parent_pyr_info.xdim(),
        parent_ydim = parent_pyr_info.ydim();
    int child_xdim = parent_xdim / 2,
        child_ydim = parent_ydim / 2;
    #ifdef VERBOSE
    cout << parent_image_path << endl;
    cout << "generating pyramid level " << parent_level + 1
         << " from pyramid level " << parent_level << " image: "
         << "parent_xdim = " << parent_xdim << endl;
    cout << "parent_ydim = " << parent_ydim << endl;
    cout << "child_xdim = " << child_xdim <<endl;
    cout << "child_ydim = " << child_ydim << endl;
    #endif
    // note output dimension for pyrDown is (parent_xdim + 1) / 2, (parent_ydim + 1) / 2
    // for parent image with odd number height or width, using original parent
    // image violates the parent_xdim / child_xdim = 2 pyramid ratio requirement.
    // truncate the parent image to be the largest even dimension size
    if (parent_xdim <= 8192 && parent_ydim <= 8192)
    {
        SERIALIZE_OPENCV_MPI
        cv::Mat parent = cv::imread(parent_image_path,
                                    cv::IMREAD_ANYDEPTH + cv::IMREAD_GRAYSCALE);
        cv::Rect roi(0, 0, parent_xdim - (int)(parent_xdim % 2 > 0),
                 parent_ydim - (int)(parent_ydim> 0));
        parent = parent(roi);
        cv::Mat child;
        cv::pyrDown(parent, child);
        cv::imwrite(child_image_path, child);
        return;
    }

    int child_tile_xdim = mcp3d::TIFFTILE_XDIM,
        child_tile_ydim = mcp3d::TIFFTILE_YDIM;
    int n_tiles_x = child_xdim / child_tile_xdim + (int)(child_xdim % child_tile_xdim > 0),
        n_tiles_y = child_ydim / child_tile_ydim + (int)(child_ydim % child_tile_ydim > 0);

    ::TIFF* child_tif = TIFFOpen(child_image_path.c_str(), "w");
    MCP3D_ASSERT(child_tif)
    int parent_bytes_per_sample = image.StorageDataBytesPerVoxel();
    mcp3d::SetTiledTiffTags(child_tif, (uint32_t)child_ydim, (uint32_t)child_xdim,
                            (uint32_t)child_tile_ydim, (uint32_t)child_tile_xdim,
                            (short)parent_bytes_per_sample * (short)8, 1 );
    // reading parent image in full rows of tiles for less io cost in scanline
    // or strip based images. maintain 3 such rows for boundary pixel extraction
    mcp3d::MImageBlock parent_tile_row_view(vector<int>({0, 0, 0}),
                                            vector<int>({0, 0, 0}),
                                            vector<int>({1, 1, 1}),
                                            vector<int>({0}),
                                            vector<int>{0});

    // allocate memory for 4 parent tile rows, a child tile and a parent tile row
    // padded for a pyr down kernel
    long n_tile_bytes = (long)child_tile_xdim * (long)child_tile_ydim *
                        (long)parent_bytes_per_sample;
    // the matrix padded_parent_quad contains 4 child tiles, and border width 2
    // in all NSWE directions. since the height width of the matrix is even,
    // pyrDown gives exact half height and width output
    int padded_parent_quad_xdim = 2 * child_tile_xdim + 4,
        padded_parent_quad_ydim = 2 * child_tile_ydim + 4;
    vector<int> padded_parent_quad_dims = {1, padded_parent_quad_ydim,
                                           padded_parent_quad_xdim};
    long n_padded_parent_tile_bytes = (long)padded_parent_quad_xdim *
                                      (long)padded_parent_quad_ydim *
                                      (long)parent_bytes_per_sample;
    unique_ptr<uint8_t[]> child_tile((new (nothrow) uint8_t[n_tile_bytes]));
    unique_ptr<uint8_t[]> padded_parent_quad((new (nothrow) uint8_t[n_padded_parent_tile_bytes]));
    MCP3D_ASSERT(child_tile && padded_parent_quad)

    // child tile (i, j) need parent tiles (2i, 2j), (2i, 2j + 1),
    // (2i + 1, 2j), (2i + 1, 2j + 1) and padding.
    // at child tile row 0, puts an empty unique ptr at front of vector, load and push back
    // parent rows 0, 1, 2. parent rows (0, 1) will become child row 0, parent
    // row 2 serves padding pixels. at each new child tile row, discard front two
    // unique ptr, and load 2 new parent rows if available, otherwise put
    // empty unique ptr (actual code does rotation).
    // for each child tile row i, the parent row 2i, 2i + 1 is
    // in vector positions 1 and 2, while vector positions 0 and 4 gives padding
    // if boundary check passes
    // each unique_ptr<mcp3d::MImage> will read a parent tile row
    mcp3d::MImageBlock dst_block, src_block;
    for (int i = 0; i < 3; ++i)
    {
        dst_block.offsets_ptr()[i] = 0;
        src_block.offsets_ptr()[i] = 0;
        src_block.extents_ptr()[i] = 0;
    }

    vector<unique_ptr<mcp3d::MImage>> parent_tile_rows_readers;
    for (int i = 0; i < n_tiles_y; ++i)
    {
        FillParentTileRow(parent_tile_rows_readers, parent_tile_row_view, image,
                          parent_xdim, parent_ydim, parent_level, z_level,
                          i, child_tile_ydim);
        // retrieve padded parent tile quad
        for (int j = 0; j < n_tiles_x; ++j)
        {
            int y_up_pad = i > 0 ? 2 : 0,
                y_down_pad = i < n_tiles_y - 1 ? min(2, parent_ydim - 2 * (i + 1) * child_tile_ydim) : 0,
                x_left_pad = j > 0 ? 2 : 0,
                x_right_pad = j < n_tiles_x - 1 ? min(2, parent_xdim - 2 * (j + 1) * child_tile_xdim) : 0;

            // provide black padding if needed
            if (y_up_pad == 0 || y_down_pad == 0 || x_left_pad == 0 || x_right_pad == 0)
                IMAGE_SELECTION_TYPED_CALL(mcp3d::SetConstant, (*(parent_tile_rows_readers[1])),
                                           padded_parent_quad.get(),
                                           padded_parent_quad_dims)
            uint8_t* parent_tile_row;
            vector<int> parent_tile_row_dims;
            // copy data
            for (int k = 1; k < 3; ++k)
            {
                parent_tile_row = parent_tile_rows_readers[k]->Volume<uint8_t>(0);
                parent_tile_row_dims = parent_tile_rows_readers[k]->selected_view().view_xyz_dims();
                /*
                dst_offsets = {2 - x_left_pad, 2 + (int)(k > 1) * child_tile_ydim, 0};
                src_offsets = {j * 2 * child_tile_xdim - x_left_pad, 0, 0};
                src_extents = {min(2 * child_tile_xdim + x_left_pad + x_right_pad,
                                   parent_tile_row_dims[0] - src_offsets[0]),
                               min(child_tile_ydim, parent_tile_row_dims[1]),
                               1}; */
                dst_block.offsets_ptr()[0] = 0;
                dst_block.offsets_ptr()[1] = 2 + (int)(k > 1) * child_tile_ydim;
                dst_block.offsets_ptr()[2] = 2 - x_left_pad;
                src_block.offsets_ptr()[0] = 0;
                src_block.offsets_ptr()[1] = 0;
                src_block.offsets_ptr()[2] = j * 2 * child_tile_xdim - x_left_pad;
                src_block.extents_ptr()[0] = 1;
                src_block.extents_ptr()[1] = min(child_tile_ydim, parent_tile_row_dims[1]);
                src_block.extents_ptr()[2] = min(2 * child_tile_xdim + x_left_pad + x_right_pad,
                                                 parent_tile_row_dims[2] -
                                                 src_block.offsets_ptr()[2]);
                IMAGE_SELECTION_TYPED_CALL(mcp3d::CopyDataVolume, (*(parent_tile_rows_readers[k])),
                                           padded_parent_quad.get(),
                                           padded_parent_quad_dims,
                                           parent_tile_row,
                                           parent_tile_row_dims,
                                           dst_block, src_block)
            }
            if (y_up_pad > 0)
            {
                parent_tile_row = parent_tile_rows_readers[0]->Volume<uint8_t>(0);
                parent_tile_row_dims = parent_tile_rows_readers[0]->selected_view().view_xyz_dims();
                /*
                dst_offsets = {2 - x_left_pad, 0, 0};
                src_offsets = {j * 2 * child_tile_xdim - x_left_pad, child_tile_ydim - 2, 0};
                src_extents = {min(2 * child_tile_xdim + x_left_pad + x_right_pad,
                                   parent_tile_row_dims[0] - src_offsets[0]),
                               2, 1}; */
                dst_block.offsets_ptr()[0] = 0;
                dst_block.offsets_ptr()[1] = 0;
                dst_block.offsets_ptr()[2] = 2 - x_left_pad;
                src_block.offsets_ptr()[0] = 0;
                src_block.offsets_ptr()[1] = child_tile_ydim - 2;
                src_block.offsets_ptr()[2] = j * 2 * child_tile_xdim - x_left_pad;
                src_block.extents_ptr()[0] = 1;
                src_block.extents_ptr()[1] = 2;
                src_block.extents_ptr()[2] = min(2 * child_tile_xdim + x_left_pad + x_right_pad,
                                                 parent_tile_row_dims[2] -
                                                 src_block.offsets_ptr()[2]);
                IMAGE_SELECTION_TYPED_CALL(mcp3d::CopyDataVolume, (*(parent_tile_rows_readers[0])),
                                           padded_parent_quad.get(),
                                           padded_parent_quad_dims,
                                           parent_tile_row,
                                           parent_tile_row_dims,
                                           dst_block, src_block);
            }
            if (y_down_pad > 0)
            {
                parent_tile_row = parent_tile_rows_readers[3]->Volume<uint8_t>(0);
                parent_tile_row_dims = parent_tile_rows_readers[3]->selected_view().view_xyz_dims();
                /*
                dst_offsets = {2 - x_left_pad, padded_parent_quad_ydim - 2, 0};
                src_offsets = {j * 2 * child_tile_xdim - x_left_pad, 0, 0};
                src_extents = {min(2 * child_tile_xdim + x_left_pad + x_right_pad,
                                   parent_tile_row_dims[0] - src_offsets[0]),
                               min(2, parent_ydim - (i + 1) * 2 * child_tile_ydim),
                               1}; */
                dst_block.offsets_ptr()[0] = 0;
                dst_block.offsets_ptr()[1] = padded_parent_quad_ydim - 2;
                dst_block.offsets_ptr()[2] = 2 - x_left_pad;
                src_block.offsets_ptr()[0] = 0;
                src_block.offsets_ptr()[1] = 0;
                src_block.offsets_ptr()[2] = j * 2 * child_tile_xdim - x_left_pad;
                src_block.extents_ptr()[0] = 1;
                src_block.extents_ptr()[1] = min(2, parent_ydim - (i + 1) * 2 * child_tile_ydim);
                src_block.extents_ptr()[2] = min(2 * child_tile_xdim + x_left_pad + x_right_pad,
                                                 parent_tile_row_dims[2] -
                                                 src_block.offsets_ptr()[2]);
                IMAGE_SELECTION_TYPED_CALL(mcp3d::CopyDataVolume, (*(parent_tile_rows_readers[3])),
                                           padded_parent_quad.get(),
                                           padded_parent_quad_dims,
                                           parent_tile_row,
                                           parent_tile_row_dims,
                                           dst_block, src_block)
            }
            // pyramid down operation
            cv::Mat padded_parent_quad_mat(padded_parent_quad_ydim,
                                           padded_parent_quad_xdim,
                                           mcp3d::VoxelTypeToCVTypes((*(parent_tile_rows_readers[1])).voxel_type(), 1),
                                           padded_parent_quad.get());
            cv::Mat padded_parent_quad_pyrdown_mat;
            // disable opencv threading
            SERIALIZE_OPENCV_MPI
            cv::pyrDown(padded_parent_quad_mat, padded_parent_quad_pyrdown_mat);
            dst_block.offsets_ptr()[0] = 0;
            dst_block.offsets_ptr()[1] = 0;
            dst_block.offsets_ptr()[2] = 0;
            src_block.offsets_ptr()[0] = 0;
            src_block.offsets_ptr()[1] = 1;
            src_block.offsets_ptr()[2] = 1;
            src_block.extents_ptr()[0] = 1;
            src_block.extents_ptr()[1] = child_tile_ydim;
            src_block.extents_ptr()[2] = child_tile_xdim;
            IMAGE_SELECTION_TYPED_CALL(mcp3d::CopyDataVolume, (*(parent_tile_rows_readers[1])),
                                       child_tile.get(),
                                       {1, child_tile_ydim, child_tile_xdim},
                                       padded_parent_quad_pyrdown_mat.ptr(),
                                       {1, child_tile_ydim + 2, child_tile_xdim + 2},
                                       dst_block, src_block)
            // write the child tile to child tif
            ::TIFFWriteEncodedTile(child_tif, (uint32_t)(i * n_tiles_x + j),
                                   child_tile.get(), n_tile_bytes);
        }
    }
    ::TIFFClose(child_tif);
}

#if MCP3D_MPI_BUILD

void
mcp3d::MTiffFormat::WriteImagePyramidMPI(MImage &image, int parent_level,
                                         bool abort_all_on_fail,
                                         const std::string &err_log_path,
                                         const std::string &out_log_path,
                                         MPI_Comm comm_writer)
{
    CHECK_PARALLEL_MODEL
    int rank, size;
    MPI_Comm_rank(comm_writer, &rank);
    MPI_Comm_size(comm_writer, &size);

    ofstream ofs;
    int parent_xdim = image.image_pyr_info(parent_level).xdim(),
        parent_ydim = image.image_pyr_info(parent_level).ydim(),
        parent_zdim = image.image_pyr_info(parent_level).zdim();
    int child_xdim = parent_xdim / 2,
        child_ydim = parent_ydim / 2;
    if (child_xdim * child_ydim == 0)
    {
        if (rank == 0)
        {
            ofs.open(err_log_path, ofstream::out | ofstream::app);
            ofs << "child level has diminished xy dimensions "
                << mcp3d::JoinVector(vector<int>({child_xdim, child_ydim}), ",")
                << ", do nothing" << endl;
            ofs.close();
        }
        return;
    }
    string pyr_level_dir(mcp3d::JoinPath(image.image_root_dir(),
                                         mcp3d::MPyramidsLayout::PyrLevelDirName(parent_level + 1)));

    int n_workers = min(size - 1, parent_zdim);
    MPI_Status status;
    if (rank > 0 && rank <= n_workers)
    {
        int z = rank - 1;
        int success = -1;
        while (true)
        {
            try
            {
                WriteImagePyramidImpl(image, parent_level, z);
            }
            catch (...)
            {
                MCP3D_PRINT_NESTED_EXCEPTION
                MPI_Send(&z, 1, MPI_INT, 0, 0, comm_writer);
            }
            // -1 for success
            MPI_Send(&success, 1, MPI_INT, 0, 0, comm_writer);
            MPI_Recv(&z, 1, MPI_INT, 0, 0, comm_writer, &status);
            // if all images dispensed and completed, exit
            if (z == success)
                break;
        }
    }
    if (rank == 0)
    {
        ofs.open(out_log_path, ofstream::out | ofstream::app);
        ofs << "pyramid down operations start" << endl;
        ofs.close();

        const vector<string>& sequence = image.image_pyr_infos()[parent_level].image_sequence();
        int buffer, success = -1;
        int next_z = n_workers;
        for (int i = 0; i < parent_zdim; ++i)
        {
            MPI_Recv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm_writer, &status);
            if (buffer != success)
            {
                ofs.open(err_log_path, ofstream::out | ofstream::app);
                ofs << "failure on image: " << sequence[buffer] << endl;
                if (abort_all_on_fail)
                {
                    ofs << "aborting all" << endl;
                    ofs.close();
                    mcp3d::RemovePath(pyr_level_dir);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                ofs.close();
            }
            if (next_z < parent_zdim)
            {
                MPI_Send(&next_z, 1, MPI_INT, status.MPI_SOURCE, 0, comm_writer);
                ++next_z;
            }
            else
                MPI_Send(&success, 1, MPI_INT, status.MPI_SOURCE, 0, comm_writer);
        }
    }
}

#endif