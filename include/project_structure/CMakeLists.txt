set(MCP3D_PROJECT_STRUCTURE_SRC
    mcp3d_project_structure.hpp
    mcp3d_pyramid_layout.hpp
    mcp3d_folder_slices.hpp
    mcp3d_project_layout.hpp
    mcp3d_pyramid_layout.cpp
    mcp3d_folder_slices.cpp
    mcp3d_project_layout.cpp)

add_library(mcp3d_project_structure SHARED ${MCP3D_PROJECT_STRUCTURE_SRC})

install(TARGETS mcp3d_project_structure LIBRARY DESTINATION ${MCP3D_INSTALL_DIR}/lib)
set_target_properties(mcp3d_project_structure PROPERTIES INSTALL_RPATH \$ORIGIN)