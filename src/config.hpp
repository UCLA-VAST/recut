#pragma once
#include <cstdint>
#include <utility>

//// compile time error printing
//#define strcat_(x, y) x ## y
//#define strcat(x, y) strcat_(x, y)
//   #define PRINT_ERROR(x) \
    //template <int> \
    //struct strcat(strcat(value_of_, x), _is); \
    //static_assert(strcat(strcat(value_of_, x), _is)<x>::x, "");

// c++17 unused printing utility
template <auto val> constexpr void static_print() {
#if !defined(__GNUC__) || defined(__clang__)
  int static_print_is_implemented_only_for_gcc = 0;
#else
  int unused = 0;
#endif
};

/// ex. power<int, 4, 2>::value
template <typename T, T V, T N, typename I = std::make_integer_sequence<T, N>>
struct power;
template <typename T, T V, T N, T... Is>
struct power<T, V, N, std::integer_sequence<T, Is...>> {
  static constexpr T value =
      (static_cast<T>(1) * ... * (V * static_cast<bool>(Is + 1)));
};

// Define preprocessor macros, templates and types to be used for
// configuration and compile time behavior
// Note it is preferred to pass this macros to `cmake ..
// -DMACRO=ON` and add them to the root CMakelists.txt

// uint32_t overflows at ~2e9 nodes, ~ 1024 * 1024 * 1024 * 2
// typedef uint32_t VID_t; // overflows after ~2e9 ~= 2 1024^3 tiles
typedef uint64_t VID_t; // for multi-interval runs

#ifdef USE_VDB
#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

// using OffsetCoord = openvdb::Vec3<int8_t>;
using OffsetCoord = openvdb::Coord; // = Int32 = int32_t
using GridCoord = openvdb::Coord;
// position must be a openvdb::math:Vec?? type not a Coord
// integer positions get cast to floats, half and doubles positions is
// acceptable
using PositionT = openvdb::Vec3f; // equiv. to Vec3s, both are <float>

namespace vb = openvdb::v8_1;
namespace vt = openvdb::tree;
namespace vp = vb::points;

#define VOXEL_SIZE 1.f
#define LEAF_LOG2DIM 3
#define INTER1_LOG2DIM 4
#define INTER2_LOG2DIM 5

using Leaf = typename vp::PointDataLeafNode<vb::PointDataIndex32, LEAF_LOG2DIM>;
using InternalNode1 = typename vt::InternalNode<Leaf, INTER1_LOG2DIM>;
using EnlargedPointDataTree = typename
    vt::Tree<vt::RootNode<vt::InternalNode<InternalNode1, INTER2_LOG2DIM>>>;
using EnlargedPointDataGrid = typename openvdb::Grid<EnlargedPointDataTree>;

// Length of a bound box edge in one dimension in image index space / world
// space units
constexpr int LEAF_LENGTH = VOXEL_SIZE * power<int, 2, LEAF_LOG2DIM>::value;
constexpr int INTER1_LENGTH =
    LEAF_LENGTH * power<int, 2, INTER1_LOG2DIM>::value;
// equivalent:
// EnlargedPointDataGrid::TreeType::LeafNodeType::DIM == LEAF_LENGTH

// PRINT_ERROR(LEAF_LENGTH);
// PRINT_ERROR(INTER1_LENGTH);

#else // not VDB
using OffsetCoord = std::vector<int8_t>;
using GridCoord = std::vector<int32_t>;
#endif

// pre-generated array of vertices initialized wth desired default values,
// useful for mmap
// must match the VID_t bit length type
// on remote servers with mounted filesystems it is
// recommended to place the interval in /tmp/ to avoid mmap issues
//#define INTERVAL_BASE "/tmp/interval_base_32.bin";
//#define INTERVAL_BASE "../data/interval_base_64bit.bin"

// equivalent max allows up to interval actual size shown with
// block_size 4 including padding (ghost cells) WARNING: if you change this
// number you need to rerun CreateIntervalBase function in recut_tests.cpp to
// save an interval at least that size at /tmp/
// MAX_INTERVAL_VERTICES needs to be larger than just the processing region to
// account for ghost regions see utils.hpp:get_used_vertex_size() or
// recut_test.cpp:PrintDefaultInfo() Vertices needed for a 1024^3 interval block
// size 4 : 3623878656 Vertices needed for a 2048^3 interval block size 4 :
// 28991029248 Vertices needed for a 8^3 interval block size 2 : 4096
#ifdef TEST_ALL_BENCHMARKS
// const VID_t MAX_INTERVAL_VERTICES =  3623878656;
const VID_t MAX_INTERVAL_VERTICES = 4096;
#else
const VID_t MAX_INTERVAL_VERTICES = 4096;
#endif

// equivalent max allows up to interval actual size of 256, 256, 256 with
// block_size 4 including padding (ghost cells) Note this is ~786 MB, for
// VertexAttr of size 24 bytes WARNING: if you change this number you need to
// rerun CreateIntervalBase function in recut_tests.cpp to save an interval at
// least that size at /tmp/
// const VID_t MAX_INTERVAL_VERTICES = 32768000;
//#define INTERVAL_BASE "/mnt/huge/interval_base_64bit.bin" // must match the
// VID_t bit length type

// Parallel strategies other than OMP defined here
//#define ASYNC // run without TF macro to use the std::async instead of TF
// thread pool, warning much much slower not recommended #define TF // if
// defined, use CPP taskflow to use a workstealing thread pool for new blocks
// TF must be used in conjucntion with ASYNC macro to allow asynchronous starts
// of new blocks during the marching step
#ifdef TF
#define ASYNC
#endif

// depending on your prune semantics a radius of 1 may cover its neighbor
// therefore account for this by subtracting 1 from all radii values
// open accumulate_prune function, if a neighbor has a radius of 1 or greater
// is the current considered pruned
// if only radii 2 or greater cover a neighbor then the erosion factor would
// be 1. While a high erosion factor creates prevents redundant coverage of
// radii, it means the effective coverage may be lower
// keep in mind this is merely a heuristic statistic to judge pruning methods
// you should refer to the actual pruning method semantics
// when checking 1 hop away (adjacent) from current,
// all radii greater than 1 imply some redundancy in coverage
// but this may be desired with DILATION_FACTORS higher than 1
#define DILATION_FACTOR 1
