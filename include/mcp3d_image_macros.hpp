//
// Created by muyezhu on 3/3/18.
//

#ifndef MCP3D_MCP3D_IMAGE_MACROS_HPP
#define MCP3D_MCP3D_IMAGE_MACROS_HPP

#define IMAGE_SELECTION_TYPED_CALL(function, image, ...) {             \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M8U)   \
        function<uint8_t>(__VA_ARGS__);                                \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M8S)   \
        function<int8_t>(__VA_ARGS__);                                 \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M16U)  \
        function<uint16_t>(__VA_ARGS__);                               \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M16S)  \
        function<int16_t>(__VA_ARGS__);                                \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M32U)  \
        function<uint32_t>(__VA_ARGS__);                               \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M32S)  \
        function<int32_t>(__VA_ARGS__);                                \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M64U)  \
        function<uint64_t>(__VA_ARGS__);                               \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M64S)  \
        function<int64_t>(__VA_ARGS__);                                \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M32F)  \
        function<float>(__VA_ARGS__);                                  \
    if (image.selected_view().voxel_type() == mcp3d::VoxelType::M64F)  \
        function<double>(__VA_ARGS__);                                 \
}

#endif //MCP3D_MCP3D_IMAGE_MACROS_HPP
