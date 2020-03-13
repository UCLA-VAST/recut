//
// Created by muyezhu on 3/15/18.
//

#ifndef MCP3D_MCP3D_IMAGE_VOXEL_ITERATOR_HPP
#define MCP3D_MCP3D_IMAGE_VOXEL_ITERATOR_HPP

#include <boost/iterator/iterator_facade.hpp>
#include "mcp3d_image.hpp"

namespace mcp3d
{
template <typename IType, typename VType>
class ImageVoxelIterator:
        public boost::iterator_facade<ImageVoxelIterator<IType, VType>,
                VType, boost::random_access_traversal_tag>
{
public:
    ImageVoxelIterator() = default;
    explicit ImageVoxelIterator(IType& image);
private:
    IType& image_;
};

}

template <typename IType, typename VType>
mcp3d::ImageVoxelIterator<IType, VType>::ImageVoxelIterator(IType& image): image_(image)
{
    static_assert(std::is_arithmetic<VType>(), "VType must be arithmetic type");
    if (mcp3d::TypeToVoxelType<VType>() != image_.voxel_type())
        MCP3D_INVALID_ARGUMENT("wrong voxel element type")
}


#endif //MCP3D_MCP3D_IMAGE_VOXEL_ITERATOR_HPP
