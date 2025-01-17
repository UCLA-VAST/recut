#pragma once
#include <cstdint>
#include <utility>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <numeric>

// Define preprocessor macros, templates and types to be used for
// configuration and compile time behavior
// Note it is preferred to pass this macros to `cmake ..
// -DMACRO=ON` and add them to the root CMakelists.txt

// uint32_t overflows at ~2e9 nodes, ~ 1024 * 1024 * 1024 * 2
// typedef uint32_t VID_t; // overflows after ~2e9 ~= 2 1024^3 tiles
typedef uint64_t VID_t; // for multi-interval runs

using OffsetCoord = openvdb::Coord; // = Int32 = int32_t
using GridCoord = openvdb::Coord;
// position must be a openvdb::math:Vec?? type not a Coord
// integer positions get cast to floats, half and doubles positions is
// acceptable
using PositionT = openvdb::Vec3f; // equiv. to Vec3s, both are <float>
using FPCodec = openvdb::points::FixedPointCodec</*1-byte=*/true,
                                                 openvdb::points::UnitRange>;
using openvdb::math::CoordBBox;

namespace vb = openvdb::OPENVDB_VERSION_NAME;
namespace vt = openvdb::tree;
namespace vto = openvdb::tools;
namespace vp = vb::points;

#define VOXEL_POINTS 1
#define VOXEL_SIZE 1
#define LEAF_LOG2DIM 3
#define INTER1_LOG2DIM 4
#define INTER2_LOG2DIM 5
#define INDEX_TYPE vb::PointDataIndex32
// Length of a bound box edge in one dimension in image index space / world
// space units
constexpr int LEAF_LENGTH = VOXEL_SIZE * (1 << LEAF_LOG2DIM);
constexpr int INTER1_LENGTH = LEAF_LENGTH * (1 << INTER1_LOG2DIM);
// equivalent:
// EnlargedPointDataGrid::TreeType::LeafNodeType::DIM == LEAF_LENGTH

// WARNING: all custom grid must be registered before use both for
// this application and any downstream application
// for example a custom type will crash vdb_view or houdini unless you added it
// to the software as a native primitive and re-compiled
// also, most OpenVDB methods are grid type specific, so you would have to
// modify methods to suit your needs. The specific default layout configuration
// has been extensively tested and small changes are unlikely to help also you
// are generally better off creating multiple grids of different type than a
// grid for a specific data type
using PointLeaf = typename vp::PointDataLeafNode<INDEX_TYPE, LEAF_LOG2DIM>;
using PointInternalNode1 = typename vt::InternalNode<PointLeaf, INTER1_LOG2DIM>;
using PointTree = typename vt::Tree<
    vt::RootNode<vt::InternalNode<PointInternalNode1, INTER2_LOG2DIM>>>;
using EnlargedPointDataGrid = typename openvdb::Grid<PointTree>;
// using EnlargedPointDataGrid = openvdb::Grid<vp::PointDataTree>;
// using EnlargedPointDataGrid = vp::PointDataGrid;
using UpdateGrid = openvdb::BoolGrid;
using UpdateLeaf = UpdateGrid::TreeType::LeafNodeType;
using UInt8Tree = vt::Tree4<uint32_t, 5, 4, 3>::Type;
// custom and needs registering
using ImgGrid = openvdb::Grid<UInt8Tree>;

// using EnlargedPointIndexGrid = typename openvdb::Grid<
// openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<
// openvdb::tree::InternalNode<openvdb::tools::PointIndexLeafNode<
// openvdb::PointIndex32, LEAF_LOG2DIM>,
// INTER1_LOG2DIM>,
// INTER2_LOG2DIM>>>>;
// using EnlargedPointIndexGrid = openvdb::Grid<vp::PointIndexTree>;
using EnlargedPointIndexGrid = vto::PointIndexGrid;

#define MAX_EDGE_LENGTH_UM 2
#define SWC_MIN_LINE 110
#define COARSEN_FACTOR .75
//#define COARSEN_STEPS 0
// force used to be set to 3 for raw human proofread seeds multiplied by voxel_size[0]
#define FORCE_SOMA_DILATION 1
#define FIND_SOMA_DILATION 1
// no filtering by default below
#define MIN_SOMA_RADIUS_UM 0 
#define MAX_SOMA_RADIUS_UM 100
// these are more specific to experimental conditions in 
// original datasets Recut was tested with: 
//#define MIN_SOMA_RADIUS_UM 3.0
//#define MAX_SOMA_RADIUS_UM 18.0
#define MIN_Z_DEPTH 30
// path distance in um of branches to prune
#define MIN_BRANCH_LENGTH 60
#define MIN_WINDOW_UM 150.0
#define EXPAND_WINDOW_UM 30.0
#define SWC_PRECISION 3
// determined empirically from comparing z radii vs. xy radii for 6x data
//#define ANISOTROPIC_FACTOR (1 / (.16))
#define ANISOTROPIC_FACTOR (1. / 2.5)

// Below are deprecated or are not (yet) affecting behavior:
#define MIN_RADII 2
// Set the pruning / coverage semantics by defining what adjacent hop count
// qualifies as covering its neighbor.
// Must be 1 or greater
#define DILATION_FACTOR 1
#define RECUT_LEVEL_SET_HALF_WIDTH 2
#define SOMA_LABEL_LENGTH 128
// 1 hour in seconds
//#define TIMEOUT 1h
#define GROW_THRESHOLD 128
#define SKELETON_GRAIN .09
#define CLASSIC_PRUNE 0
#define TOPOLOGY_CLOSE_FACTOR 1.5
#define SOMA_CLOSE_FACTOR 7
#define OPEN_FACTOR 5
#define FG_PCT .8
#define MATCH_DISTANCE 8.
