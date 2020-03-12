//
// Created by muyezhu on 2/11/18.
//
#ifndef MCP3D_MCP3D_SUPPORTED_FORMATS_HPP
#define MCP3D_MCP3D_SUPPORTED_FORMATS_HPP

#include <cstdint>
#include <unordered_set>
#include <map>

namespace mcp3d
{
enum class VoxelType
{
    M8U, M8S, M16U, M16S, M32U, M32S, M64U, M64S, M32F, M64F, UNKNOWN = -1
};

VoxelType VoxelTypeStrToEnum(const std::string &type_str);

std::string VoxelTypeEnumToStr(VoxelType voxel_type);

template <typename VType>
VoxelType TypeToVoxelType()  { return VoxelType::UNKNOWN; }

template<>
inline VoxelType TypeToVoxelType<uint8_t>() { return VoxelType::M8U; }

template<>
inline VoxelType TypeToVoxelType<int8_t>() { return VoxelType::M8S; }

template<>
inline VoxelType TypeToVoxelType<uint16_t>() { return VoxelType::M16U; }

template<>
inline VoxelType TypeToVoxelType<int16_t>() { return VoxelType::M16S; }

template<>
inline VoxelType TypeToVoxelType<uint32_t>() { return VoxelType::M32U; }

template<>
inline VoxelType TypeToVoxelType<int32_t>() { return VoxelType::M32S; }

template<>
inline VoxelType TypeToVoxelType<uint64_t>() { return VoxelType::M64U; }

template<>
inline VoxelType TypeToVoxelType<int64_t>() { return VoxelType::M64S; }

template<>
inline VoxelType TypeToVoxelType<float>() { return VoxelType::M32F; }

template<>
inline VoxelType TypeToVoxelType<double>() { return VoxelType::M64F; }

inline bool KnownVoxelType(VoxelType d)
{ return d != VoxelType::UNKNOWN ; }

inline bool SupportedTiffVoxeltype(VoxelType d)
{ return d == VoxelType::M8U || d == VoxelType::M16U || d == VoxelType:: M32S; }

inline bool UnsignedVoxeltype(VoxelType d)
{ return d == VoxelType::M8U || d == VoxelType::M16U || d == VoxelType::M32U || d == VoxelType::M64U ; }

inline bool SignedVoxeltype(VoxelType d)
{ return d == VoxelType::M8S || d == VoxelType::M16S || d == VoxelType::M32S || d == VoxelType::M64S ; }

inline bool FloatingVoxeltype(VoxelType d)
{ return d == VoxelType::M32F || d == VoxelType::M64F; }

int BytesPerVoxelElement(VoxelType sample_type);

inline int BytesPerVoxelElement(const std::string &type_str)
{ return BytesPerVoxelElement(VoxelTypeStrToEnum(type_str)); }

int VoxelTypeToCVTypes(VoxelType vt, int n_channels = 1);

template <typename VType>
int TypeToCVTypes(int n_channels = 1);

}

template <typename VType>
int mcp3d::TypeToCVTypes(int n_channels)
{
    return mcp3d::VoxelTypeToCVTypes(mcp3d::TypeToVoxelType<VType>(), n_channels);
}

#endif //MCP3D_MCP3D_SUPPORTED_FORMATS_HPP
