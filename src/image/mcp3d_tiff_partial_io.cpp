//
// Created by mzhu on 4/4/17.
//
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "mcp3d_image_utils.hpp"
#include "mcp3d_tiff_partial_io.hpp"

using namespace std;

void mcp3d::ResizeLargeTiff(const std::string &tiff_path)
{
    if (!mcp3d::IsFile(tiff_path))
        MCP3D_INVALID_ARGUMENT(tiff_path + " is not a file");

    ::TIFF* tif = TIFFOpen(tiff_path.c_str(), "r");
    ::TIFF* tiff_resize = TIFFOpen((mcp3d::RemoveFileExt(tiff_path) + "_resize.tif").c_str(), "w");

    if (tif)
    {
        uint32_t image_width, image_length;
        uint32_t tile_width, tile_length;
        short bits_per_pixel;
        uint32_t x, y;
        tdata_t buf, buf_resize;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &image_width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &image_length);
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_length);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_pixel);

        uint32_t img_width_resize = (image_width - image_width % tile_width) / 2;
        uint32_t img_length_resize = (image_length - image_length % tile_length) / 2;
        uint32_t tile_width_resize = tile_width / 2;
        uint32_t tile_length_resize = tile_length / 2;
        mcp3d::SetTiledTiffTags(tiff_resize, img_length_resize, img_width_resize,
                                tile_length_resize, tile_width_resize,
                                bits_per_pixel, 1);

        buf = _TIFFmalloc(TIFFTileSize(tif));
        buf_resize = _TIFFmalloc(TIFFTileSize(tif) / 4);
        ttile_t tile = 0;
        tsize_t size = TIFFTileSize(tif) / 4;

        for (y = 0; y < image_length - tile_length; y += tile_length)
            for (x = 0; x < image_width - tile_width; x += tile_width)
            {
                TIFFReadTile(tif, buf, x, y, 0, 1);
                for (uint32_t i = 0; i < tile_length; ++i)
                    for (uint32_t j = 0; j < tile_width; ++j)
                        if (i % 2 == 0 && j % 2 == 0)
                        {
                            ((uchar*)buf_resize)[i / 2 * tile_width_resize + j / 2] = ((uchar*)buf)[i * tile_width + j];
                        }


                if (TIFFWriteRawTile(tiff_resize, tile, buf_resize, size) > 0)
                    ++tile;
            }

        _TIFFfree(buf);
        TIFFClose(tif);
        TIFFClose(tiff_resize);
        _TIFFfree(buf_resize);
    }
}

void ValidateTiffSubImageRequest(const mcp3d::TiffInfo& tiff_info,
                                 const int64_t subimg_height,
                                 const int64_t subimg_width,
                                 const int64_t subimg_origin_x = 0,
                                 const int64_t subimg_origin_y = 0)
{
    if (tiff_info.n_directory > 1)
        MCP3D_DOMAIN_ERROR("does not support multi directory tiff image")
    if (tiff_info.planar_config != 1)
        // https://www.awaresystems.be/imaging/tiff/tifftags/planarconfiguration.html
        MCP3D_DOMAIN_ERROR("only supporting chunky planar config")
    if (tiff_info.samples_per_pixel > 3)
        MCP3D_DOMAIN_ERROR("does not support alpha channel")
    if (tiff_info.bits_per_sample != 8 && tiff_info.bits_per_sample != 16 && tiff_info.bits_per_sample != 32)
        MCP3D_DOMAIN_ERROR("only supporting 8 bits, 16 bits or 32 bits per voxel")
    if (subimg_origin_x < 0 || subimg_origin_x >= tiff_info.image_width ||
        subimg_origin_y < 0 || subimg_origin_y >= tiff_info.image_height)
        MCP3D_INVALID_ARGUMENT("invalid subimage starting coordinates")
}

void mcp3d::TransferTileToSubimg(const string& tif_path, tdata_t subimg_buf,
                                 int64_t subimg_height, int64_t subimg_width,
                                 int64_t subimg_src_origin_x,
                                 int64_t subimg_src_origin_y)
{
    if (!mcp3d::IsFile(tif_path))
        MCP3D_OS_ERROR(tif_path + " is not a file")
    ::TIFF* tif = TIFFOpen(tif_path.c_str(), "r");
    mcp3d::TransferTileToSubimg(tif, subimg_buf, subimg_height, subimg_width,
                                subimg_src_origin_x, subimg_src_origin_y);
    ::TIFFClose(tif);
}

void mcp3d::TransferTileToSubimg(const TIFF *tif, tdata_t subimg_buf,
                                 int64_t subimg_height, int64_t subimg_width,
                                 int64_t subimg_src_origin_x,
                                 int64_t subimg_src_origin_y)
{
    MCP3D_ASSERT(subimg_buf)
    mcp3d::TiffInfo tiff_info(tif);
    if (! tiff_info.is_tiled) MCP3D_DOMAIN_ERROR(
            "wrong function call: tiff image is stripped")
    if (subimg_width == 0)
        subimg_width = subimg_height;
    uint32_t img_height = tiff_info.image_height,
             img_width = tiff_info.image_width,
             tile_height = tiff_info.tile_height,
             tile_width = tiff_info.tile_width;
    short bits_per_sample = tiff_info.bits_per_sample,
          samples_per_pixel = tiff_info.samples_per_pixel,
          planar_config = tiff_info.planar_config;
    ValidateTiffSubImageRequest(tiff_info,
                                subimg_height, subimg_width,
                                subimg_src_origin_x, subimg_src_origin_y);
    tdata_t tile_buf;
    tile_buf = _TIFFmalloc(TIFFTileSize(const_cast<TIFF*>(tif)));
    if (!tile_buf)
        MCP3D_BAD_ALLOC("can not allocate memory for image tile")

    // subimg_focus_x, subimg_focus_y:
    // relative to the top left pixel of the original tif image.
    // coordinates of the upper left pixel of rectangular region in the subimg
    // to be copied. does not have to be aligned with
    // top left pixel of the next tile in original tif to be read
    // tile_start_x, tile_start_y:
    // relative to the top left pixel of the next tile to be copied.
    // relative coordinates of subimg_focus point in the tile.
    // needed when subimg dimension do not evenly divide tile dimension
    int64_t tile_start_x, tile_start_y,
            subimg_focus_x, subimg_focus_y;
    subimg_focus_x = subimg_src_origin_x;
    subimg_focus_y = subimg_src_origin_y;
    while (true)
    {
        // fill subimg with current tile.
        // if subimg is partially out of image area, background padding is already
        // done in MImageIO, just make sure to read within bounds
        if (subimg_focus_x < img_width && subimg_focus_y < img_height)
            TIFFReadTile(const_cast<TIFF*>(tif), tile_buf,
                         (uint32_t)subimg_focus_x, (uint32_t)subimg_focus_y,
                         0, (uint16_t)planar_config);

        // subimg_x_offset, subimg_y_offset:
        // relative to subimg_focus_x and subimg_focus_y,
        // used to correctly index into the rectangular area in the subimg
        // currently being filled
        int64_t subimg_x_offset = subimg_focus_x - subimg_src_origin_x,
                subimg_y_offset = subimg_focus_y - subimg_src_origin_y;
        int64_t subimg_sample_offset =
                samples_per_pixel * (subimg_y_offset * subimg_width + subimg_x_offset);

        tile_start_x = subimg_focus_x % tile_width;
        tile_start_y = subimg_focus_y % tile_height;

        int64_t delta_y = subimg_focus_y % tile_height == 0 ?
                          min((int64)tile_height, (int64)subimg_height - subimg_y_offset) :
                          min((int64)tile_height - subimg_focus_y % tile_height, (int64)subimg_height - subimg_y_offset),
                delta_x = subimg_focus_x % tile_width == 0 ?
                          min((int64)tile_width, (int64)subimg_width - subimg_x_offset) :
                          min((int64)tile_width - subimg_focus_x % tile_width, (int64)subimg_width - subimg_x_offset);
        int64_t tile_end_y = tile_start_y + delta_y,
                tile_end_x = tile_start_x + delta_x;

        if (subimg_focus_x < img_width && subimg_focus_y < img_height)
        {
            int64_t subimg_sample_index = subimg_sample_offset;
            int64_t tile_sample_index = samples_per_pixel * (tile_width * tile_start_y + tile_start_x);
            size_t n_bytes = (size_t)(tile_end_x - tile_start_x) * (bits_per_sample / 8) * samples_per_pixel;
            for (int64_t i = tile_start_y; i < tile_end_y; ++i)
            {
                if (bits_per_sample == 8)
                    memcpy(((uint8_t*)subimg_buf) + subimg_sample_index,
                           ((uint8_t*)tile_buf) + tile_sample_index, n_bytes);

                else if (bits_per_sample == 16)
                    memcpy(((uint16_t*)subimg_buf) + subimg_sample_index,
                           ((uint16_t*)tile_buf) + tile_sample_index, n_bytes);
                else
                    memcpy(((int32_t*)subimg_buf) + subimg_sample_index,
                           ((int32_t*)tile_buf) + tile_sample_index, n_bytes);

                subimg_sample_index += samples_per_pixel * subimg_width;
                tile_sample_index += samples_per_pixel * tile_width;
            }
        }
        // update subimg_focus_x and subimg_focus_y
        subimg_focus_x += delta_x;
        if (subimg_focus_x - subimg_src_origin_x >= subimg_width)
        {
            subimg_focus_x = subimg_src_origin_x;
            subimg_focus_y += delta_y;
            if (subimg_focus_y - subimg_src_origin_y >= subimg_height)
                break;
        }
    }
    _TIFFfree(tile_buf);
}

void mcp3d::TransferScanlineToSubimg(const string& tif_path, tdata_t subimg_buf,
                                     int64_t subimg_height, int64_t subimg_width,
                                     int64_t subimg_src_origin_x,
                                     int64_t subimg_src_origin_y)
{
    if (!mcp3d::IsFile(tif_path))
        MCP3D_OS_ERROR(tif_path + " is not a file")
    ::TIFF* tif = ::TIFFOpen(tif_path.c_str(), "r");
    mcp3d::TransferScanlineToSubimg(tif, subimg_buf, subimg_height, subimg_width,
                                    subimg_src_origin_x, subimg_src_origin_y);
    ::TIFFClose(tif);
}

void mcp3d::TransferScanlineToSubimg(const TIFF *tif, tdata_t subimg_buf,
                                     int64_t subimg_height,
                                     int64_t subimg_width,
                                     int64_t subimg_src_origin_x,
                                     int64_t subimg_src_origin_y)
{
    MCP3D_ASSERT(subimg_buf)
    mcp3d::TiffInfo tiff_info(tif);
    if (tiff_info.is_tiled)
        MCP3D_DOMAIN_ERROR("wrong function call: tiff image is tiled")
    if (subimg_width == 0)
        subimg_width = subimg_height;
    uint32_t img_height = tiff_info.image_height,
             img_width = tiff_info.image_width,
             rows_in_strip = tiff_info.rows_in_strip;
    short bits_per_sample = tiff_info.bits_per_sample,
          samples_per_pixel = tiff_info.samples_per_pixel;
    ValidateTiffSubImageRequest(tiff_info,
                                subimg_height, subimg_width,
                                subimg_src_origin_x, subimg_src_origin_y);
    // tsize_t is return type of TIFFScanlineSize on cluster, instead
    // of tmsize_t
    tsize_t strip_size = TIFFStripSize(const_cast<TIFF*>(tif));
    tdata_t strip_buf = _TIFFmalloc(strip_size);
    if (!strip_buf) MCP3D_BAD_ALLOC("can not allocate memory for tiff strip")
    // scanline does not support random access and require sequentially read
    // through unneeded data. use strips instead
    int64_t subimg_yend = subimg_src_origin_y + subimg_height,
            subimg_xend = subimg_src_origin_x + subimg_width;
    int64_t subimg_sample_offset = 0, strip_sample_offset;
    int64_t copy_y_width = min(subimg_yend, (int64_t)img_height) - subimg_src_origin_y;
    int64_t copy_x_width = min(subimg_xend, (int64_t)img_width) - subimg_src_origin_x;

    // if subimg is partially out of image area, background padding is already
    // done in MImageIO, just make sure to read within bounds
    for (int64_t i = subimg_src_origin_y;
         i < subimg_src_origin_y + copy_y_width; ++i)
    {
        // offset of the strip within src img
        int64_t strip_offset = i / rows_in_strip;
        // offset of current line offset within strip
        int64_t scanline_offset = i % rows_in_strip;
        // read new strip when previous strip exhausted, or when first entering
        // the loop
        if (scanline_offset == 0 || i == subimg_src_origin_y)
            TIFFReadEncodedStrip(const_cast<TIFF*>(tif),
                                 (uint32_t)strip_offset, strip_buf, strip_size);

        strip_sample_offset = samples_per_pixel * (img_width * scanline_offset + subimg_src_origin_x);
        size_t n_bytes = (size_t)samples_per_pixel * copy_x_width * (bits_per_sample / 8);

        if (bits_per_sample == 8)
            memcpy(((uint8_t*)subimg_buf) + subimg_sample_offset,
                   ((uint8_t*)strip_buf) + strip_sample_offset, n_bytes);
        else if (bits_per_sample == 16)
            memcpy(((uint16_t*)subimg_buf) + subimg_sample_offset,
                   ((uint16_t*)strip_buf) + strip_sample_offset, n_bytes);
        else
            memcpy(((int32_t*)subimg_buf) + subimg_sample_offset,
                   ((int32_t*)strip_buf) + strip_sample_offset, n_bytes);

        subimg_sample_offset += samples_per_pixel * subimg_width;
    }
    _TIFFfree(strip_buf);
}

void mcp3d::TransferToSubimg(const ::TIFF *tif, tdata_t subimg_buf,
                             int64_t subimg_height, int64_t subimg_width,
                             int64_t subimg_src_origin_x,
                             int64_t subimg_src_origin_y)
{
    mcp3d::TiffInfo tiff_info(tif);
    if (tiff_info.is_tiled)
        mcp3d::TransferTileToSubimg(tif, subimg_buf, subimg_height, subimg_width,
                                    subimg_src_origin_x, subimg_src_origin_y);
    else
        mcp3d::TransferScanlineToSubimg(tif, subimg_buf, subimg_height, subimg_width,
                                        subimg_src_origin_x, subimg_src_origin_y);
}

void mcp3d::TransferToSubimg(const string& tif_path, tdata_t subimg_buf,
                             int64_t subimg_height, int64_t subimg_width,
                             int64_t subimg_src_origin_x,
                             int64_t subimg_src_origin_y)
{
    mcp3d::TiffInfo tiff_info(tif_path);
    if (tiff_info.is_tiled)
        mcp3d::TransferTileToSubimg(tif_path, subimg_buf, subimg_height, subimg_width,
                                    subimg_src_origin_x, subimg_src_origin_y);
    else
        mcp3d::TransferScanlineToSubimg(tif_path, subimg_buf, subimg_height, subimg_width,
                                        subimg_src_origin_x, subimg_src_origin_y);
}

void mcp3d::TransferScanlineToSubimgRow(::TIFF *tif,
                                        ::TIFF **tif_subimgs,
                                        tdata_t *subimg_scanline_bufs,
                                        tdata_t scanline_buf,
                                        uint32_t tif_row,
                                        int64_t subimg_height,
                                        int64_t subimg_width,
                                        int num_subimgs_in_row)
{
    mcp3d::TiffInfo tiff_info(tif);
    if (tiff_info.is_tiled) MCP3D_DOMAIN_ERROR(
            "wrong function call: tiff image is tiled")
    if (subimg_width == 0)
        subimg_width = subimg_height;
    uint32_t img_height = tiff_info.image_height,
             img_width = tiff_info.image_width;
    short bits_per_sample = tiff_info.bits_per_sample,
          samples_per_pixel = tiff_info.samples_per_pixel;
    if (samples_per_pixel > 3) MCP3D_DOMAIN_ERROR(
            "does not support alpha channel")
    ValidateTiffSubImageRequest(tiff_info, subimg_height, subimg_width);
    // fill a row of image subimgs by writing scan lines
    for (uint32_t i = 0; i < subimg_height; ++i)
    {
        // if subimg is partially out of image area, background padding is already
        // done in MImageIO, just make sure to read within bounds
        if (tif_row < img_height)
            TIFFReadScanline(tif, scanline_buf, tif_row++);
        int64_t subimg_y_offset = i % subimg_height;
        for (uint32_t j = 0; j < num_subimgs_in_row * subimg_width; ++j)
        {
            int64_t subimg_index = j / subimg_width;
            int64_t subimg_x_offset = j % subimg_width;
            for (int k = 0; k < samples_per_pixel; ++k)
            {
                int64_t index = samples_per_pixel * subimg_x_offset + k;
                // copy origin tif image scan line data to
                // appropriate subimg scan lines
                if (j < img_width)
                {
                    if (bits_per_sample == 8)
                        ((uint8_t*)subimg_scanline_bufs[subimg_index])[index] =
                            ((uint8_t*)scanline_buf)[samples_per_pixel * j + k];
                    else if (bits_per_sample == 16)
                        ((uint16_t*)subimg_scanline_bufs[subimg_index])[index] =
                            ((uint16_t*)scanline_buf)[samples_per_pixel * j + k];
                    else
                        ((int32_t*)subimg_scanline_bufs[subimg_index])[index] =
                                ((int32_t*)scanline_buf)[samples_per_pixel * j + k];
                }
                // pad black pixels if subimg partially falls out of
                // original image column boundary
                else
                {
                    if (bits_per_sample == 8)
                        ((uint8_t*)subimg_scanline_bufs[subimg_index])[index] = 0;
                    else if (bits_per_sample == 16)
                        ((uint16_t*)subimg_scanline_bufs[subimg_index])[index] = 0;
                    else
                        ((int32_t*)subimg_scanline_bufs[subimg_index])[index] = 0;
                }
            }
        }
        for (int j = 0; j < num_subimgs_in_row; ++j)
        {
            int write_line = TIFFWriteScanline(tif_subimgs[j],
                                               subimg_scanline_bufs[j],
                                               (uint32_t)subimg_y_offset, 0);
            if (write_line < 0)
                std::cerr << "failed to write scan line" << std::endl;
        }
    }
}

void large_tiled_tiff_to_subimgs(TIFF* tif,
                                 const uint32_t& img_height,
                                 const uint32_t& img_width,
                                 const string& tif_subimg_dir,
                                 const string& img_name,
                                 const uint32_t& subimg_height,
                                 const uint32_t& subimg_width,
                                 const short& bits_per_sample,
                                 const short& samples_per_pixel,
                                 const short& planar_config)
{
    // number of image subimgs in row and column direction
    uint32_t num_subimgs_in_row = (img_height % subimg_height == 0) ?
                                  img_height / subimg_height :
                                  (img_height / subimg_height + 1);
    uint32_t num_subimgs_in_col = (img_width % subimg_width == 0) ?
                                  img_width / subimg_width :
                                  (img_width / subimg_width + 1);
    // subimg_origin_x:
    // x coordinate of upper left corner of the square image subimg
    uint32_t subimg_origin_x, subimg_origin_y;
    // allocate memory for tiles and subimgs
    tdata_t subimg_buf;
    size_t subimg_size = bits_per_sample / 8 *
                         samples_per_pixel * subimg_height * subimg_height;
    subimg_buf = _TIFFmalloc(subimg_size);
    if (!subimg_buf) MCP3D_BAD_ALLOC("can not allocate memory for sub image")
    // produce image subimgs
    for (uint32_t i = 0; i < num_subimgs_in_col; ++i)
        for (uint32_t j = 0; j < num_subimgs_in_row; ++j)
        {
            // image subimg full path:
            // tif_subimg_dir/[tif_name]_tiley[i]_tilex[j].tif
            TIFF* tif_subimg = TIFFOpen(
                    mcp3d::JoinPath(
                            {tif_subimg_dir, img_name +
                                             "_x" +
                                    mcp3d::PadNumStr(
                                            i * subimg_width,
                                            img_width) +
                                             "_y" +
                                    mcp3d::PadNumStr(
                                            j * subimg_height,
                                            img_height) +
                                             "_z0.tif"}).c_str(), "w");
            // set tags for tiff subimg
            mcp3d::SetTiledTiffTags(tif_subimg, subimg_height, subimg_width,
                                    subimg_height, subimg_width,
                                    bits_per_sample, samples_per_pixel);

            subimg_origin_x = i * subimg_width;
            subimg_origin_y = j * subimg_height;

            mcp3d::TransferTileToSubimg(tif, subimg_buf,
                                        subimg_height, subimg_width,
                                        subimg_origin_x, subimg_origin_y);

            TIFFWriteRawTile(tif_subimg, 0, subimg_buf, subimg_size);
            TIFFClose(tif_subimg);
        }
    _TIFFfree(subimg_buf);
}

void large_stripped_tiff_to_subimgs(TIFF* tif,
                                    const uint32_t& img_height,
                                    const uint32_t& img_width,
                                    const string& tif_subimg_dir,
                                    const string& img_name,
                                    const uint32_t& subimg_height,
                                    const uint32_t& subimg_width,
                                    const short& bits_per_sample,
                                    const short& samples_per_pixel,
                                    const short& planar_config)
{
    // number of image subimgs in row and column direction
    uint32_t num_subimgs_in_row = (img_width % subimg_height == 0) ?
                                  img_width / subimg_height :
                                  (img_width / subimg_height + 1);
    uint32_t num_subimgs_in_col = (img_height % subimg_height == 0) ?
                                  img_height / subimg_height :
                                  (img_height / subimg_height + 1);
    // array of TIFF* pointers to a row of image subimgs
    TIFF* tif_subimgs[num_subimgs_in_row];
    // scanline buffer of original tif image
    tdata_t scanline_buf = _TIFFmalloc(TIFFScanlineSize(tif));
    // array of scanline buffers of a row of image subimgs
    tdata_t subimg_scanline_bufs[num_subimgs_in_row];
    uint32_t tif_row = 0;
    for (uint32_t i = 0; i < num_subimgs_in_col; ++i)
    {
        for (uint32_t j = 0; j < num_subimgs_in_row; ++j)
        {
             tif_subimgs[j]= TIFFOpen(
                     mcp3d::JoinPath(
                             {tif_subimg_dir, img_name +
                                              "_x" +
                                     mcp3d::PadNumStr(
                                             i * subimg_width,
                                             img_width) +
                                              "_y" +
                                     mcp3d::PadNumStr(
                                             j * subimg_height,
                                             img_height) +
                                              "_z0.tif"}).c_str(), "w");
            // set tags for tiff subimg
            uint32_t rows_in_strip = TIFFDefaultStripSize(tif_subimgs[j], 0);
            mcp3d::SetStripTiffTags(tif_subimgs[j], subimg_height, subimg_width,
                                    rows_in_strip, bits_per_sample,
                                    samples_per_pixel);
            if (i == 0)
            {
                tsize_t subimg_scanline_size = TIFFScanlineSize(tif_subimgs[i]);
                subimg_scanline_bufs[j] = _TIFFmalloc(subimg_scanline_size);
                // allocate memory for subimg scanline
                if (!subimg_scanline_bufs[j])
                {
                    _TIFFfree(scanline_buf);
                    for (uint32_t k = 0; k < j; ++k)
                    {
                        _TIFFfree(subimg_scanline_bufs[k]);
                        TIFFClose(tif_subimgs[k]);
                    }
                    TIFFClose(tif_subimgs[j]);
                    MCP3D_BAD_ALLOC(
                            "can not allocate memory for image subimg scanlines")
                }
            }

        }

        mcp3d::TransferScanlineToSubimgRow(tif, tif_subimgs,
                                           subimg_scanline_bufs,
                                           scanline_buf,
                                           tif_row,
                                           subimg_height,
                                           subimg_width,
                                           num_subimgs_in_row);
        // close a row of image subimg tifs
        for (uint32_t j = 0; j < num_subimgs_in_row; ++j)
            TIFFClose(tif_subimgs[j]);
    }
    // free memory for original image and subimg images scanline buffers
    _TIFFfree(scanline_buf);
    for (uint32_t j = 0; j < num_subimgs_in_row; ++j)
        _TIFFfree(subimg_scanline_bufs[j]);

}

bool mcp3d::LargeTiffToSubimgs(const std::string &tiff_path,
                               uint32_t subimg_height,
                               uint32_t subimg_width)
{
    if (!mcp3d::IsFile(tiff_path))
        throw invalid_argument(tiff_path + " is not a file");

    pair<string, string> tiff_path_split = mcp3d::SplitBaseName(tiff_path);
    string img_name = mcp3d::RemoveFileExt(tiff_path_split.second);
    string tif_subimg_dir = mcp3d::JoinPath(
            {tiff_path_split.first, img_name + "_tiles"});

    ::TIFF* tif = TIFFOpen(tiff_path.c_str(), "r");
    mcp3d::TiffInfo tiff_info(tiff_path);

    if (tiff_info.image_height <= mcp3d::TIFFTILE_YDIM &&
        tiff_info.image_width <= mcp3d::TIFFTILE_YDIM)
    {
        cout << "small image. do nothing." << endl;
        return true;
    }

    if (subimg_height > mcp3d::MAX_TIFFTILE_YDIM)
    {
        cout << "requesting image subimg height greater than MAX_TILE_Y_LEN. "
                "resetting image subimg height to"
                " MAX_TILE_Y_LEN = " + to_string(mcp3d::MAX_TIFFTILE_YDIM) << endl;
        subimg_height = mcp3d::MAX_TIFFTILE_YDIM;
    }

    // create [tiff_name]_tiles folder in same directory as the tiff image
    mcp3d::MakeDirectories(tif_subimg_dir);
    mcp3d::CheckTiffSupported(tiff_info);
    if (!tiff_info.is_tiled)
        large_stripped_tiff_to_subimgs(tif, tiff_info.image_height,
                                       tiff_info.image_width,
                                       tif_subimg_dir, img_name,
                                       subimg_height, subimg_width,
                                       tiff_info.bits_per_sample,
                                       tiff_info.samples_per_pixel,
                                       tiff_info.planar_config);
    else
        large_tiled_tiff_to_subimgs(tif, tiff_info.image_height,
                                    tiff_info.image_width,
                                    tif_subimg_dir, img_name,
                                    subimg_height, subimg_width,
                                    tiff_info.bits_per_sample,
                                    tiff_info.samples_per_pixel,
                                    tiff_info.planar_config);

    TIFFClose(tif);
    return mcp3d::ValidateLargeTiffSubimgs(tiff_path) == 0;
}

void mcp3d::WriteTilesToLargeTiff(const vector<string> &tile_img_paths,
                                  const string &dst_path,
                                  int64_t img_height, int64_t img_width,
                                  VoxelType datatype, bool skip_validation)
{
    if (img_height <= 0 || img_width <= 0)
        MCP3D_INVALID_ARGUMENT("invalid image dimensions")
    if (!mcp3d::SupportedTiffVoxeltype(datatype))
        MCP3D_INVALID_ARGUMENT("only supporting 8 bit unsigned, 16 bit unsigned or signed 32 bit int images")
    string dst_dir = mcp3d::ParentDir(dst_path);
    mcp3d::MakeDirectories(dst_dir);
    if (!skip_validation)
    {
        mcp3d::VerifyTiffSequence(tile_img_paths, true, string());
        if (!mcp3d::TiffTilePathsIsSorted(tile_img_paths))
            MCP3D_INVALID_ARGUMENT("expecting sorted tile tiff paths")
    }
    mcp3d::TiffInfo tiff_info(tile_img_paths[0]);
    uint32_t tile_height = tiff_info.image_height,
           tile_width = tiff_info.image_width;
    short bits_per_pixel;
    if (datatype == VoxelType::M8U)
        bits_per_pixel = 8;
    else if (datatype == VoxelType::M16U)
        bits_per_pixel = 16;
    else
        bits_per_pixel = 32;
    ::TIFF *tif = TIFFOpen(dst_path.c_str(), "w");
    mcp3d::SetTiledTiffTags(tif, (uint32) img_height, (uint32) img_width,
                            tile_height, tile_width, bits_per_pixel);
    tsize_t nbytes = tile_height * tile_width * bits_per_pixel / 8;
    int cv_type;
    if (datatype == VoxelType::M8U)
        cv_type = CV_8U;
    else if (datatype == VoxelType::M16U)
        cv_type = CV_16U;
    else
        cv_type = CV_32S;
    for (uint32_t i = 0; i < tile_img_paths.size(); ++i)
    {
        cout << tile_img_paths[i] << endl;
        cv::Mat m = cv::imread(tile_img_paths[i], cv::IMREAD_ANYDEPTH + cv::IMREAD_GRAYSCALE);
        m.convertTo(m, cv_type);
        TIFFWriteRawTile(tif, i, m.ptr(), nbytes);
    }
    TIFFClose(tif);
}

bool mcp3d::ValidateLargeTiffSubimgs(const string &tiff_path)
{
    if (!mcp3d::IsFile(tiff_path))
        MCP3D_INVALID_ARGUMENT(tiff_path + " is not a file");
    ::TIFF* tif = TIFFOpen(tiff_path.c_str(), "r");
    if (! tif)
        MCP3D_RUNTIME_ERROR("bad pointer to tif")
    pair<string, string> path_split = mcp3d::SplitBaseName(tiff_path);
    string img_name = mcp3d::RemoveFileExt(path_split.second);
    string img_tiles_dir = mcp3d::JoinPath(
            {path_split.first, img_name + "_tiles"});
    if (!mcp3d::IsDir(img_tiles_dir))
    {
        cout << img_tiles_dir << " does not exist" << endl;
        return false;
    }
    vector<string> tile_img_paths {mcp3d::FilesInDir(
            img_tiles_dir, true, false,
            {"_x", "_y", "_z", ".tif"})};
    if (tile_img_paths.empty())
    {
        cout << "no tiles generated for image: " << tiff_path << endl;
        return false;
    }
    uint32_t tile_rows = 0, tile_cols = 0;
    uint32_t img_rows, img_cols;
    uint32_t total_rows = 0, total_cols = 0;
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &img_rows);
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &img_cols);
    size_t tile_img_clength = tile_img_paths[0].size();
    for (const string& tile_img_path: tile_img_paths)
    {
        if (tile_img_path.size() != tile_img_clength)
        {
            cout << "tile image names have inconsistent length, "
                    "will not be sortable" << endl;
            return false;
        }
        uint32_t t_rows, t_cols;
        ::TIFF* tif_tile = TIFFOpen(tile_img_path.c_str(), "r");
        if (! tif_tile)
        {
            cout << "unable to open image tile: " << tile_img_path << endl;
            return false;
        }

        TIFFGetField(tif_tile, TIFFTAG_IMAGELENGTH, &t_rows);
        TIFFGetField(tif_tile, TIFFTAG_IMAGEWIDTH, &t_cols);
        if (tile_rows == 0)
            tile_rows = t_rows;
        else if (tile_rows != t_rows)
        {
            cout << "image tiles have inconsistent row counts: "
                 << tile_img_path << endl;
            return false;
        }
        if (tile_cols == 0)
            tile_cols = t_cols;
        else if (tile_cols != t_cols)
        {
            cout << "image tiles have inconsistent row counts: "
                 << tile_img_path << endl;
            return false;
        }
        total_rows += tile_rows;
        total_cols += tile_cols;
    }
    if (total_rows < img_rows || total_cols < img_cols)
    {
        cout << "sum of all image tile row/column counts less than "
                "image row/column count: " << tiff_path << endl;
        return false;
    }
    return true;
}