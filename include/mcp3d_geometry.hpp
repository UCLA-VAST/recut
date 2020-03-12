//
// Created by muyezhu on 3/13/18.
//

#ifndef MCP3D_MCP3D_GEOMETRY_HPP
#define MCP3D_MCP3D_GEOMETRY_HPP

#include <cstdint>
#include <type_traits>
#include <limits>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_member.hpp>

namespace mcp3d
{
template <typename IndexType = int32_t>
class Position3D
{
public:
    typedef IndexType index_type;
    explicit Position3D(std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> x_ = (IndexType)0,
                        std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> y_ = (IndexType)0,
                        std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> z_ = (IndexType)0);
    IndexType x, y, z;
    bool operator== (const Position3D &other)
    { return x == other.x and y == other.y and z == other.z; }
    bool operator!= (const Position3D &other)
    { return !((*this) == other); }
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void save(Archive &ar, const uint32_t version) const;
    template <typename Archive>
    void load(Archive &ar, const uint32_t version);
    BOOST_SERIALIZATION_SPLIT_MEMBER();
    bool is_integer_, is_signed_, is_float_;
    int width_;
};

template <typename IndexType = int32_t, typename ValueType = int32_t>
class Point3D
{
public:
    explicit Point3D(std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> x_ = (IndexType)0,
                     std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> y_ = (IndexType)0,
                     std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> z_ = (IndexType)0,
                     std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> v_ = std::numeric_limits<ValueType>::max()):
                                                x(x_), y(y_), z(z_), voxel_val_(v_) {};
    ValueType voxel_val() const { return voxel_val_; }
    IndexType x, y, z;
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, const uint32_t version);
    ValueType voxel_val_;

};

}

template <typename IndexType>
mcp3d::Position3D<IndexType>::Position3D(std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> x_,
                                         std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> y_,
                                         std::enable_if_t<std::is_arithmetic<IndexType>::value, IndexType> z_):
                                                           x(x_), y(y_), z(z_)
{
    is_integer_ = std::is_integral<IndexType>();
    is_float_ = !is_integer_;
    is_signed_ = std::is_signed<IndexType>();
    width_ = sizeof(IndexType);
}

template <typename IndexType>
template <typename Archive>
void mcp3d::Position3D<IndexType>::save(Archive &ar, const uint32_t version) const
{
    ar & is_integer_;
    ar & is_signed_;
    ar & is_float_;
    ar & width_;
    ar & x;
    ar & y;
    ar & z;
}

template <typename IndexType>
template <typename Archive>
void mcp3d::Position3D<IndexType>::load(Archive &ar, const uint32_t version)
{
    ar & is_integer_;
    ar & is_signed_;
    ar & is_float_;
    ar & width_;
    // xor
    if (is_integer_ ^ std::is_integral<IndexType>() ||
        is_float_ ^ std::is_floating_point<IndexType>() ||
        is_signed_ ^ std::is_signed<IndexType>())
        MCP3D_INVALID_ARGUMENT("archived object and this object index type mismatch")
    ar & x;
    ar & y;
    ar & z;
}

#endif //MCP3D_MCP3D_GEOMETRY_HPP
