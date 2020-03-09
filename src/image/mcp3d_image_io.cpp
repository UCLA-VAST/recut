//
// Created by muyezhu on 2/12/18.
//
#include <memory>
#include <algorithm>
#include <unordered_set>
#include "common/mcp3d_utility.hpp"
#include "mcp3d_file_formats.hpp"
#include "mcp3d_image_formats.hpp"
#include "mcp3d_tiff_format.hpp"
#include "mcp3d_hdf5_format.hpp"
#include "mcp3d_virtual_hdf5_io.hpp"
#include "mcp3d_image_io.hpp"

using namespace std;

mcp3d::MImageIO::MImageIO()
{
    io_formats_[FileFormats::TIFF] = make_unique<MTiffFormat>();
    io_formats_[FileFormats::HDF5] = make_unique<MHdf5Format>();
}

mcp3d::MImageInfo mcp3d::MImageIO::ReadImageInfo(const string &img_root_dir)
{
    string image_root_dir = mcp3d::MPyramidsLayout::ParsePyramidRootDir(
                                                      img_root_dir);
    MCP3D_TRY(return ReadImageInfoImpl(image_root_dir);)
}

mcp3d::MImageInfo mcp3d::MImageIO::ReadImageInfoImpl(const string &img_root_dir)
{
    cout << "Reading image info from image root directory " << img_root_dir << endl;
    mcp3d::MImageInfo image_info;
    // try to read __image_info__.json first
    cout << "Looking for " << mcp3d::MPyramidsLayout::image_info_json_name() << endl;
    string image_info_json_path(mcp3d::MPyramidsLayout::ImageInfoPath(img_root_dir));
    if (mcp3d::IsFile(image_info_json_path))
    {
        cout << "found..." << endl;
        image_info = MImageInfo(image_info_json_path);
    }
    MCP3D_TRY(ReadImageInfoFromPyrDirs(img_root_dir, image_info);)
    if (image_info.empty())
        MCP3D_MESSAGE("found no supported images under " + img_root_dir)
    return image_info;
}

void mcp3d::MImageIO::ReadImageInfoFromPyrDirs(const string &img_root_dir,
                                               MImageInfo &image_info)
{
    // level 0 images are allowed to be directly under img_root_dir, or under
    // img_root_dir/pyr_level_0
    // read pyr image info from directory images
    vector<string> pyr_level_dirs =
            mcp3d::MPyramidsLayout::PyrLevelDirs(img_root_dir, true, true);
    for(const auto& pyr_level_dir: pyr_level_dirs)
    {
        if (image_info.has_pyr_level_dir(pyr_level_dir))
            continue;
        bool is_root_image = pyr_level_dir == img_root_dir;
        try
        {
            MImageInfo pyr_image_info(
                    ReadImageInfoFromPyrDirFiles(pyr_level_dir,is_root_image));
            if (pyr_image_info.empty())
                continue;
            image_info += pyr_image_info;
        }
        catch(...)
        {
            MCP3D_RETHROW(current_exception())
        }
    }
    MCP3D_TRY(ValidateImageInfo(image_info);)
}

mcp3d::MImageInfo mcp3d::MImageIO::ReadImageInfoFromPyrDirFiles(const string &pyr_level_dir,
                                                                bool is_level0_image)
{
    if (!mcp3d::IsDir(pyr_level_dir))
        MCP3D_OS_ERROR(pyr_level_dir + " is not a directory")

    // read image info from images
    pair<mcp3d::FileFormats, string> img_format =
            mcp3d::FindFileFormatInDir(pyr_level_dir);
    mcp3d::FileFormats format = img_format.first;
    const string& read_format_str = img_format.second;
    // if no image files found under pyr_level_dir, return an empty MImageInfo
    if (format == mcp3d::FileFormats::UNKNOWN)
    {
        MCP3D_MESSAGE("no images with known format found in " + pyr_level_dir)
        return MImageInfo{};
    }
    if (!io_formats_[format]->CanRead())
        MCP3D_IMAGE_FORMAT_UNSUPPORTED("image format not readable: " +
                                        mcp3d::FileFormatEnumToExt(format))

    try
    {
        cout << "Looking for " << mcp3d::FileFormatEnumToExt(format)
             << " image files under " << pyr_level_dir << endl;
        mcp3d::MImageInfo info = io_formats_[format]->ReadImageInfo(pyr_level_dir,
                                                                    is_level0_image,
                                                                    read_format_str);
        return info;
    }
    catch (...)
    {
        MCP3D_RETHROW(current_exception())
    }
}

void mcp3d::MImageIO::RefreshImageInfo(MImage &image)
{
    if (image.image_info().empty())
        MCP3D_RUNTIME_ERROR("can not refresh empty image info. use ReadImageInfo instead")
    ReadImageInfoFromPyrDirs(image.image_root_dir(), image.image_info_);
}

void mcp3d::MImageIO::ValidateImageInfo(const mcp3d::MImageInfo& image_info)
{
    try
    {
        // validations common to all formats
        image_info.ValidatePyramidsStructure();
        MCP3D_ASSERT(image_info.AllPathsExist())
        // validate image info with format specific function
        for(int i = 0; i < image_info.n_pyr_levels(); ++i)
            io_formats_[image_info.pyr_infos()[i].pyr_image_format()]
                    ->ValidateImageInfo(image_info);
    }
    catch (const mcp3d::MCPAssertionError& e)
    {
        MCP3D_RETHROW(make_exception_ptr(e))
    }
}

// image formats should be known at every level given image info is correctly read
void mcp3d::MImageIO::ReadData(mcp3d::MImage &image, bool black_background)
{
    if (image.image_info().empty())
        MCP3D_RUNTIME_ERROR("image data can only be read after image info is read")
    if (image.selected_view().empty())
        MCP3D_RUNTIME_ERROR("no image view selected for reading")
    mcp3d::FileFormats format = image.image_pyr_info(
            image.selected_view().pyr_level()).pyr_image_format();
    if (!ReadableFormat(format))
        MCP3D_IMAGE_FORMAT_UNSUPPORTED("reading " + mcp3d::FileFormatEnumToExt(
                format) + " format not supported")
    bool out_of_boundary = image.selected_view().OutOfPyrImageBoundary();
    bool partially_out_of_boundary = image.selected_view().PartiallyOutOfPyrImageBoundary();
    if (out_of_boundary || partially_out_of_boundary)
    {
        IMAGE_SELECTION_TYPED_CALL(mcp3d::MImageIO::FillViewWithBackground,
                         image, image, mcp3d::MImageBlock{},
                         black_background);
        if (out_of_boundary)
        {
            #ifdef VERBOSE
            MCP3D_MESSAGE("view is entirely out of view level image bounds, "
                          "fill with background voxels");
            #endif
            return;
        }
        else
        {
            #ifdef VERBOSE
            MCP3D_MESSAGE("view is partially out of view level image bounds, "
                          "fill with background voxels");
            #endif
        }
    }
    MCP3D_TRY(io_formats_[format]->ReadData(image);)
}

void mcp3d::MImageIO::WriteViewVolume(const MImage &image,
                                      const string &out_dir,
                                      const string &img_name_prefix,
                                      FileFormats write_format)
{
    MCP3D_ASSERT(!image.loaded_view().empty())
    int rl = image.loaded_view().pyr_level();
    mcp3d::FileFormats level_format = image.image_pyr_info(
            rl).pyr_image_format();
    if (write_format == mcp3d::FileFormats::UNKNOWN)
        write_format = level_format;
    MCP3D_ASSERT(WritableFormat(write_format))
    mcp3d::MakeDirectories(out_dir);
    io_formats_[write_format]->WriteViewVolume(image, out_dir, img_name_prefix);
}

int mcp3d::MImageIO::ImagePyramidChunkNumber(const MImage &img,
                                             FileFormats output_format,
                                             int parent_level)
{
    if (output_format == mcp3d::FileFormats::UNKNOWN ||
            io_formats_.find(output_format) == io_formats_.end())
        MCP3D_RUNTIME_ERROR("output format not known")
    if (!io_formats_[output_format]->CanWrite())
        MCP3D_RUNTIME_ERROR("write operation not supported for " +
                                    mcp3d::FileFormatEnumToExt(output_format))
    if (img.n_pyr_levels() <= parent_level)
        MCP3D_RUNTIME_ERROR("requested parent level does not exist in image")
    return io_formats_[output_format]->ImagePyramidChunkNumber(img,
                                                               parent_level);
}

void mcp3d::MImageIO::WriteImagePyramid(mcp3d::MImage &image, int parent_level,
                                        bool multi_threading,
                                        mcp3d::FileFormats write_format)
{
    string pyr_level_dir(mcp3d::JoinPath(image.image_root_dir(),
                                         mcp3d::MPyramidsLayout::PyrLevelDirName(parent_level + 1)));
    mcp3d::RemovePath(pyr_level_dir);
    mcp3d::MakeDirectories(pyr_level_dir);

    try
    {
        MCP3D_ASSERT(parent_level >= 0 && parent_level < image.n_pyr_levels())
        image.image_info().LevelPathsExist(parent_level);
    }
    catch (const mcp3d::MCPAssertionError& e)
    {
        MCP3D_RETHROW(make_exception_ptr(e))
    }
    mcp3d::FileFormats read_format = image.image_info().file_format(
            parent_level);
    if (write_format == mcp3d::FileFormats::UNKNOWN)
        write_format = read_format;
    MCP3D_ASSERT(ReadableFormat(read_format))
    MCP3D_ASSERT(WritableFormat(write_format))

    try
    {
        io_formats_[write_format]->WriteImagePyramidOMP(image, parent_level, multi_threading);
    }
    catch (...)
    {
        mcp3d::RemovePath(pyr_level_dir);
        MCP3D_RETHROW(current_exception());
    }
}

bool mcp3d::MImageIO::ReadableFormat(FileFormats format)
{
    if (io_formats_.find(format) == io_formats_.end())
        return false;
    if (format == FileFormats::UNKNOWN)
        return false;
    return io_formats_[format]->CanRead();
}

bool mcp3d::MImageIO::WritableFormat(FileFormats format)
{
    if (io_formats_.find(format) == io_formats_.end())
        return false;
    if (format == FileFormats::UNKNOWN)
        return false;
    return io_formats_[format]->CanWrite();
}

#if MCP3D_MPI_BUILD

void mcp3d::MImageIO::WriteImagePyramidMPI(const std::string &img_root_dir, int parent_level,
                                           FileFormats write_format, bool abort_all_on_fail,
                                           MPI_Comm writer_comm)
{
    MCP3D_ASSERT(mcp3d::MPIInitialized())
    int rank, size;
    MPI_Comm_rank(writer_comm, &rank);
    MPI_Comm_size(writer_comm, &size);

    // set up logs. rank0 will only maintain log and make file system operations
    string log_dir(mcp3d::JoinPath({img_root_dir, "log"}));
    RANK0_COMM_CALL_SYNC(mcp3d::MakeDirectories, writer_comm, log_dir)
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char time_cstr[13];
    strftime(time_cstr, 13, "%Y%m%d%H%M", localtime(&now));
    string err_log_path = mcp3d::JoinPath({log_dir, string("write_image_pyramid_err") + time_cstr}),
           out_log_path = mcp3d::JoinPath({log_dir, string("write_image_pyramid_out") + time_cstr});

    ofstream ofs;
    if (rank == 0)
    {
        ofs.open(err_log_path, ofstream::out);
        ofs.close();
        ofs.open(out_log_path, ofstream::out);
        ofs.close();
    }

    INIT_TIMER(0)
    TIC_COMM(writer_comm, 0)
    mcp3d::MImage image(true);
    if (rank == 0)
    {
        image.ReadImageInfo(img_root_dir);
        image.SaveImageInfo();
    }
    MPI_Barrier(writer_comm);
    if (rank != 0)
        image.ReadImageInfo(img_root_dir);
    TOC_COMM(writer_comm, 0)

    if (rank == 0)
    {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ofs.open(out_log_path, ofstream::out | ofstream::app);
        ofs << "discarded " << world_size - size << " slow processes" << endl;
        ofs.close();

        if (image.image_info().empty())
        {
            ofs.open(err_log_path, ofstream::out | ofstream::app);
            ofs << "no known image data found under " << image.image_root_dir() << endl;
            ofs.close();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (parent_level < 0 || parent_level >= image.n_pyr_levels())
        {
            ofs.open(err_log_path, ofstream::out | ofstream::app);
            ofs << "requested parent image level does not exist" << endl;
            ofs.close();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(writer_comm);
    mcp3d::FileFormats read_format = image.image_pyr_info(parent_level).pyr_image_format();
    if (write_format == mcp3d::FileFormats::UNKNOWN)
        write_format = read_format;
    if (rank == 0)
    {
        ofs.open(out_log_path, ofstream::out | ofstream::app);
        ofs << "image root directory: " << img_root_dir << endl;
        ofs << "create one level of pyramid image from level " << parent_level << endl;
        ofs << "    input dimensions " << mcp3d::JoinVector<int>(image.image_pyr_info(parent_level).dims(), ", ") << endl;
        ofs << "    voxel type: " << mcp3d::VoxelTypeEnumToStr(image.voxel_type()) << endl;
        ofs << "reading image info: " << ELAPSE(0) << " seconds"<< endl;
        ofs.close();

        if(!ReadableFormat(read_format))
        {
            ofs.open(err_log_path, ofstream::out | ofstream::app);
            ofs << "image format unreadable" << endl;
            ofs.close();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!WritableFormat(write_format))
        {
            ofs.open(err_log_path, ofstream::out | ofstream::app);
            ofs << "requested write format not supported" << endl;
            ofs.close();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(writer_comm);

    // remove older output, create directory
    string pyr_level_dir(mcp3d::JoinPath({image.image_root_dir(),
                                         mcp3d::MPyramidsLayout::PyrLevelDirName(parent_level + 1)}));
    RANK0_COMM_CALL_SYNC(mcp3d::RemovePath, writer_comm, pyr_level_dir)
    RANK0_COMM_CALL_SYNC(mcp3d::MakeDirectories, writer_comm, pyr_level_dir)

    // check input paths integrity
    const vector<string>& sequence = image.image_pyr_infos()[parent_level].image_sequence();
    int n_parent_imgs = (int)(sequence.size());
    string parent_dir(mcp3d::JoinPath({image.image_root_dir(),
                                       image.image_pyr_infos()[parent_level].pyr_level_dir()}));
    int n_workers = min(size - 1, n_parent_imgs);
    MPI_Status status;

    if (rank > 0 && rank <= n_workers)
    {
        string img_path;
        int32_t img_id = 0;
        for (int i = (rank - 1) * n_parent_imgs / n_workers;
             i < min(rank * n_parent_imgs / n_workers, n_parent_imgs); ++i)
        {
            img_path = mcp3d::JoinPath({parent_dir, sequence[i]});
            img_id = (rank - 1) * n_parent_imgs / n_workers + i;
            // send failures
            if (!mcp3d::IsFile(img_path))
                MPI_Send(&img_id, 1, MPI_INT32_T, 0, 0, writer_comm);
        }
        // send -1 to mark task completion
        img_id = -1;
        MPI_Send(&img_id, 1, MPI_INT32_T, 0, 0, writer_comm);
    }
    if (rank == 0)
    {
        unordered_set<int32_t> bad_paths_id;
        int finished = 0;
        int32_t buffer;
        while (finished < n_workers)
        {
            MPI_Recv(&buffer, 1, MPI_INT32_T, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     writer_comm, &status);
            if (buffer >= 0)
                bad_paths_id.insert(buffer);
            else
                ++finished;
        }
        if (!bad_paths_id.empty())
        {
            ofs.open(err_log_path, ofstream::out | ofstream::app);
            ofs << "bad image paths:" << endl;
            for (const auto& bad_path_id: bad_paths_id)
                ofs << sequence[bad_path_id] << endl;
            ofs <<"aborting" << endl;
            ofs.close();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    TIC_COMM(writer_comm, 0)
    io_formats_[write_format]->WriteImagePyramidMPI(image, parent_level,
                                                    abort_all_on_fail,
                                                    err_log_path,
                                                    out_log_path, writer_comm);
    TOC_COMM(writer_comm, 0)
    if (rank == 0)
    {
        ofs.open(out_log_path, ofstream::out | ofstream::app);
        double n_min = ELAPSE(0) / 60;
        ofs << "image pyramid generation complete in " << n_min
            << " minutes with " << size - 1 << " writers" << endl;
        ofs.close();
    }
}

#endif



