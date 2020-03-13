set(MCP3D_IMAGE_SRC
    mcp3d_image_common.hpp
    mcp3d_image_macros.hpp
    mcp3d_image_constants.hpp
    mcp3d_image_utils.hpp
    mcp3d_tiff_utils.hpp
    mcp3d_hdf5_utils.hpp
    mcp3d_voxel_types.hpp
    mcp3d_file_formats.hpp
    mcp3d_image_info.hpp
    mcp3d_image_view.hpp
    mcp3d_image_formats.hpp
    mcp3d_tiff_format.hpp
    mcp3d_tiff_tile_format.hpp
    mcp3d_tiff_partial_io.hpp
    mcp3d_virtual_hdf5_io.hpp
    mcp3d_imaris_io.hpp
    mcp3d_image_io.hpp
    readwrite.hpp
    mcp3d_image_base.hpp
    mcp3d_image.hpp
    mcp3d_virtual_image.hpp
    mcp3d_image_voxel_iterator.hpp
    mcp3d_image_utils.cpp
    mcp3d_tiff_utils.cpp
    mcp3d_hdf5_utils.cpp
    mcp3d_voxel_types.cpp
    mcp3d_file_formats.cpp
    mcp3d_image_info.cpp
    mcp3d_image_view.cpp
    mcp3d_image_formats.cpp
    mcp3d_tiff_format.cpp
    mcp3d_tiff_tile_format.cpp
    mcp3d_hdf5_format.hpp
    mcp3d_hdf5_format.cpp
    mcp3d_tiff_partial_io.cpp
    mcp3d_virtual_hdf5_io.cpp
    mcp3d_imaris_io.cpp
    mcp3d_image_io.cpp
    mcp3d_image_base.cpp
    mcp3d_image.cpp
    readwrite.cpp
        )

add_library(mcp3d_image SHARED ${MCP3D_IMAGE_SRC})
target_link_libraries(mcp3d_image
                      ${EXT_LINK_LIBS}
                      mcp3d_common
                      mcp3d_project_structure)

install(TARGETS mcp3d_image LIBRARY DESTINATION ${MCP3D_INSTALL_DIR}/lib)
set_target_properties(mcp3d_image PROPERTIES INSTALL_RPATH \$ORIGIN)
