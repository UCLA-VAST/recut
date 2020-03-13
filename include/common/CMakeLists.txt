set(MCP3D_COMMON_SRC
    mcp3d_common.hpp
    mcp3d_constants.hpp
    mcp3d_exceptions.hpp
    mcp3d_macros.hpp
    mcp3d_macros_local.hpp
    mcp3d_macros_mpi.hpp
    mcp3d_paths.hpp
    mcp3d_types.hpp
    mcp3d_geometry.hpp
    mcp3d_utility.hpp
    mcp3d_exceptions.cpp
    mcp3d_paths.cpp
    mcp3d_utility.cpp)
add_library(mcp3d_common SHARED ${MCP3D_COMMON_SRC})
target_link_libraries(mcp3d_common ${EXT_LINK_LIBS})

install(TARGETS mcp3d_common LIBRARY DESTINATION ${MCP3D_INSTALL_DIR}/lib)
set_target_properties(mcp3d_common PROPERTIES INSTALL_RPATH \$ORIGIN)
