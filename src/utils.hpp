#pragma once

#include "markers.h"
#include "range/v3/all.hpp"
#include "v3d_image_io/basic_4dimage.h"
#include "recut_parameters.hpp"
#include "seed.hpp"
#include "vertex_attr.hpp"
#include <algorithm> //min, clamp
#include <atomic>
#include <chrono>
#include <cstdlib> //rand srand
#include <ctime>   // for srand
#include <filesystem>
#include <hdf5.h>
#include <iomanip> //std::setw()
#include <math.h>
#include <openvdb/tools/Clip.h>
#include <GEL/Geometry/Graph.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Dense.h>          // copyToDense
#include <openvdb/tools/LevelSetFilter.h> // LevelSetFilter
#include <openvdb/tools/LevelSetSphere.h> // createLevelSetSphere
#include <openvdb/tools/LevelSetUtil.h> // sdfSegment, sdfToFogVolume, extractEnclosedRegion
#include <openvdb/tools/VolumeToSpheres.h> // fillWithSpheres
#include <queue>
#include <ranges>
#include <sstream>
#include <stdlib.h> // ultoa
#include <string>
#include <tiffio.h>
#include <variant>

namespace rng = ranges;
namespace rv = ranges::views;

using image_width = std::variant<uint8_t, uint16_t>;

#define PI 3.14159265
// be able to change pp values into std::string
#define XSTR(x) STR(x)
#define STR(x) #x

// A helper to create overloaded function objects and type pattern match
// Taken from Functional C++ - Cukic
template <typename... Fs> struct overloaded : Fs... {
  using Fs::operator()...;
};

template <typename... Fs> overloaded(Fs...) -> overloaded<Fs...>;

auto print_iter = [](auto iterable) {
  rng::for_each(iterable, [](auto i) { std::cout << i << ", "; });
  std::cout << '\n';
};

auto print_markers = [](auto tree) {
  for (auto m : tree)
    std::cout << *m << '\n';
};

auto print_iter_name = [](auto iterable, std::string name) {
  std::cout << name << ": ";
  print_iter(iterable);
};

auto coord_to_id = [](auto xyz, auto lengths) {
  return static_cast<VID_t>(xyz[2]) * (lengths[0] * lengths[1]) +
    xyz[1] * lengths[0] + xyz[0];
};

auto new_grid_coord = [](auto x, auto y, auto z) -> GridCoord {
  return GridCoord(x, y, z);
};

auto new_offset_coord = [](auto x, auto y, auto z) -> OffsetCoord {
  OffsetCoord offset{x, y, z};
  return offset;
};

auto zeros = []() { return new_grid_coord(0, 0, 0); };

auto ones = []() { return new_grid_coord(1, 1, 1); };

auto zeros_off = []() { return new_offset_coord(0, 0, 0); };

auto min_max = [](std::array<double, 3> arr) -> std::pair<double, float> {
  float maxx = arr[0];
  float minx = arr[0];
  for (int i = 1; i < 3; ++i) {
    auto current = arr[i];
    if (maxx < current)
      maxx = current;
    if (current < minx)
      minx = current;
  }
  // return std::make_pair<float, float>(minx, maxx>);
  return {minx, maxx};
};

auto coord_invert = [](auto coord) {
  return new_offset_coord(-coord[0], -coord[1], -coord[2]);
};

auto id_to_coord = [](auto id, auto lengths) {
  GridCoord coords(3);
  coords[0] = id % lengths[0];
  coords[1] = (id / lengths[0]) % lengths[1];
  coords[2] = (id / (lengths[0] * lengths[1])) % lengths[2];
  return coords;
};

// auto id_to_string = [](VID_t id) {
// char buffer [sizeof(VID_t)*8+1];
// ultoa (id,buffer,DECIMAL);
// return buffer;
//}

auto id_to_off_coord = [](auto id, auto lengths) {
  auto coord = id_to_coord(id, lengths);
  return new_offset_coord(coord[0], coord[1], coord[2]);
};

auto coord_to_str = [](auto coords) {
  std::ostringstream coord_str;
  coord_str << '[' << coords[0] << ", " << coords[1] << ", " << coords[2]
    << "]";
  return coord_str.str();
};

auto tree_to_str = [](auto id1, auto id2) {
  std::ostringstream coord_str;
  coord_str << '{' << id1 << ',' << id2 << "}";
  return coord_str.str();
};

const auto print_coord = [](auto coords, std::string name = "") {
  if (!name.empty()) {
    std::cout << name << ": ";
  }
  std::cout << coord_to_str(coords) << '\n';
};

auto coord_add = [](auto x, auto y) {
  return GridCoord(x[0] + y[0], x[1] + y[1], x[2] + y[2]);
};

auto coord_sub = [](auto x, auto y) {
  return GridCoord(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
};

auto coord_div = [](const auto &x, const auto &y) {
  return GridCoord(x[0] / y[0], x[1] / y[1], x[2] / y[2]);
};

auto coord_prod = [](const auto &x, const auto &y) {
  return GridCoord(x[0] * y[0], x[1] * y[1], x[2] * y[2]);
};

auto coord_prod_accum = [](const auto coord) -> VID_t {
  return static_cast<VID_t>(coord[0]) * coord[1] * coord[2];
};

auto coord_mod = [](auto x, auto y) {
  return GridCoord(x[0] % y[0], x[1] % y[1], x[2] % y[2]);
};

auto coord_to_vec = [](auto coord) {
  return std::vector<VID_t>{static_cast<VID_t>(coord[0]),
    static_cast<VID_t>(coord[1]),
    static_cast<VID_t>(coord[2])};
};

const auto coord_all_eq = [](auto x, auto y) {
  if (x[0] != y[0])
    return false;
  if (x[1] != y[1])
    return false;
  if (x[2] != y[2])
    return false;
  return true;
};

const auto coord_all_lt = [](auto x, auto y) {
  if (x[0] >= y[0])
    return false;
  if (x[1] >= y[1])
    return false;
  if (x[2] >= y[2])
    return false;
  return true;
};

const auto coord_reverse = [](auto &coord) {
  auto z = coord[0];
  coord[0] = coord[2];
  coord[2] = z;
};

const auto coord_to_vdb = [](auto coord) {
  return new openvdb::Coord(coord[0], coord[1], coord[2]);
};

// the MyMarker operator< provided by library doesn't work
static const auto lt = [](const MyMarker *lhs, const MyMarker *rhs) {
  return std::tie(lhs->z, lhs->y, lhs->x) < std::tie(rhs->z, rhs->y, rhs->x);
};

static const auto eq = [](const MyMarker *lhs, const MyMarker *rhs) {
  // return *lhs == *rhs;
  // std::cout << "eq lhs " << lhs->description(2,2) << " rhs " <<
  // rhs->description(2,2) << '\n';
  return lhs->x == rhs->x && lhs->y == rhs->y && lhs->z == rhs->z;
};

// composing with pipe '|' is possible with actions and views
// passing a container to an action must be done by using std::move
const auto unique_count = [](std::vector<MyMarker *> v) {
  return rng::distance(std::move(v) | rng::actions::sort(lt) |
      rng::actions::unique(eq));
};

auto marker_dist = [](MyMarker *a, MyMarker *b) {
  return sqrt((a->x - b->x) * (a->x - b->x) + (a->y - b->y) * (a->y - b->y) +
      (a->z - b->z) * (a->z - b->z));
};

// taken from Bryce Adelstein Lelbach's Benchmarking C++ Code talk:
struct high_resolution_timer {
  high_resolution_timer() : start_time_(take_time_stamp()) {}

  void restart() { start_time_ = take_time_stamp(); }

  double elapsed() const // return elapsed time in seconds
  {
    return double(take_time_stamp() - start_time_) * 1e-9;
  }
  auto elapsed_formatted() const // return elapsed time in d:h:m:s
  {
    uint64_t t = int(elapsed());
    unsigned int days = t / 24 / 3600;
    unsigned short hours = t % (24 * 3600) / 3600;
    unsigned short minutes = t % 3600 / 60;
    unsigned short seconds = t % 60;
    std::stringstream s;
    s << std::setfill('0') << std::setw(2) << days << ':' << std::setw(2)
      << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2)
      << seconds; // << " d:h:m:s";
    return s.str();
  }
  std::uint64_t elapsed_nanoseconds() const {
    return take_time_stamp() - start_time_;
  }

  protected:
  static std::uint64_t take_time_stamp() {
    return std::uint64_t(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
  }

  private:
  std::uint64_t start_time_;
};

// Note: this is linux specific, other cross-platform solutions available:
//     https://stackoverflow.com/questions/1528298/get-path-of-executable
/*
   auto get_parent_dir() {
// fs::path full_path(fs::current_path());
fs::path full_path(fs::canonical("/proc/self/exe"));
return fs::canonical(full_path.parent_path().parent_path());
}
*/

auto get_data_dir() {
  fs::path a = CMAKE_INSTALL_DATADIR;
  return a;
}

VID_t get_central_coord(int grid_size) {
  return grid_size / 2 - 1; // place at center
}

VID_t get_central_vid(int grid_size) {
  auto coord = get_central_coord(grid_size);
  auto vid = (VID_t)coord * grid_size * grid_size + coord * grid_size + coord;
  return vid; // place at center
}

VID_t get_central_diag_vid(int grid_size) {
  auto coord = get_central_coord(grid_size);
  coord++; // add 1 to all x, y, z
  auto vid = (VID_t)coord * grid_size * grid_size + coord * grid_size + coord;
  return vid; // place at center diag
}

MyMarker *get_central_root(int grid_size) {
  VID_t x, y, z;
  x = y = z = get_central_coord(grid_size); // place at center
  auto root = new MyMarker();
  root->x = x;
  root->y = y;
  root->z = z;
  return root;
}

void get_img_coord(VID_t id, VID_t &i, VID_t &j, VID_t &k, VID_t grid_size) {
  i = id % grid_size;
  j = (id / grid_size) % grid_size;
  k = (id / (grid_size * grid_size)) % grid_size;
}

std::vector<MyMarker *> coords_to_markers(std::vector<GridCoord> coords) {
  return coords | rv::transform([](GridCoord coord) {
      return new MyMarker(static_cast<double>(coord.x()),
          static_cast<double>(coord.y()),
          static_cast<double>(coord.z()));
      }) |
  rng::to_vector;
}

/* tile_length parameter is actually irrelevant due to
 * copy on write, the chunk requested during reading
 * or mmapping is
 */
VID_t get_used_vertex_size(VID_t grid_size, VID_t block_size) {
  auto len = grid_size / block_size;
  auto total_blocks = len * len * len;
  auto pad_block_size = block_size + 2;
  auto pad_block_num = pad_block_size * pad_block_size * pad_block_size;
  // this is the total vertices that will be used including ghost cells
  auto tile_vert_num = pad_block_num * total_blocks;
  return tile_vert_num;
}

auto print_marker_3D = [](auto markers, auto tile_lengths, std::string stage) {
  for (int zi = 0; zi < tile_lengths[2]; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * tile_lengths[0] + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < tile_lengths[1]; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < tile_lengths[0]; xi++) {
        VID_t index = ((VID_t)xi) + yi * tile_lengths[0] +
          zi * tile_lengths[1] * tile_lengths[0];
        auto value = std::string{"-"};
        for (const auto &m : markers) {
          if (m->vid(tile_lengths[0], tile_lengths[1]) == index) {
            if (stage == "label") {
              value = "B";
            } else if (stage == "radius") {
              value = std::to_string(m->radius);
            } else {
              assertm(false, "stage not recognized");
            }
          }
        }
        cout << value << " ";
      }
      cout << '\n';
    }
    cout << '\n';
  }
  cout << '\n';
};

// only values strictly greater than bkg_thresh are valid
auto copy_vdb_to_dense_buffer(openvdb::FloatGrid::Ptr grid,
    const openvdb::math::CoordBBox &bbox) {
  // cout << "copy_vdb_to_dense_buffer of size: ";
  // bbox's by default are inclusive of start and end indices
  // which violates recut's model of indepedent bounding boxes
  auto dim = bbox.dim().offsetBy(-1);
  // cout << dim << '\n';
  // cout << "buffer size " << coord_prod_accum(dim) << '\n';
  auto buffer = std::make_unique<uint16_t[]>(coord_prod_accum(dim));
  for (int z = bbox.min()[2]; z < bbox.max()[2]; z++) {
    for (int y = bbox.min()[1]; y < bbox.max()[1]; y++) {
      for (int x = bbox.min()[0]; x < bbox.max()[0]; x++) {
        openvdb::Coord xyz(x, y, z);
        auto val = grid->tree().getValue(xyz);
        auto buffer_coord = xyz - bbox.min();
        auto id = coord_to_id(buffer_coord, dim);
        buffer[id] = val;
      }
    }
  }
  return buffer;
}

// only values strictly greater than bkg_thresh are valid
auto create_vdb_mask(EnlargedPointDataGrid::Ptr grid,
    const openvdb::math::CoordBBox &bbox) {
  cout << "create_vdb_mask(): \n";
  auto inclusive_dim = bbox.dim().offsetBy(-1);
  cout << inclusive_dim << '\n';
  auto mask = std::make_unique<uint16_t[]>(coord_prod_accum(inclusive_dim));
  for (int z = bbox.min()[2]; z < bbox.max()[2]; z++) {
    for (int y = bbox.min()[1]; y < bbox.max()[1]; y++) {
      for (int x = bbox.min()[0]; x < bbox.max()[0]; x++) {
        openvdb::Coord xyz(x, y, z);
        auto leaf_iter = grid->tree().probeConstLeaf(xyz);
        auto buffer_coord = xyz - bbox.min();
        auto id = coord_to_id(buffer_coord, inclusive_dim);
        if (leaf_iter) {
          auto ind = leaf_iter->beginIndexVoxel(xyz);
          if (ind) {
            mask[id] = 1;
          } else {
            mask[id] = 0;
          }
        } else {
          mask[id] = 0;
        }
      }
    }
  }
  return mask;
}

// only values strictly greater than bkg_thresh are valid
template <typename T>
void print_sdf(T vdb_accessor, const GridCoord &lengths,
    const GridCoord &offsets = {0, 0, 0}) {
  std::cout << std::fixed << std::setprecision(0);
  for (int z = offsets.z(); z < offsets.z() + lengths[2]; z++) {
    cout << "y | Z=" << z << '\n';
    for (int x = 0; x < 2 * lengths[0] + 4; x++) {
      cout << "-";
    }
    cout << '\n';
    for (int y = offsets.y(); y < offsets.y() + lengths[1]; y++) {
      cout << y << " | ";
      for (int x = offsets.x(); x < offsets.x() + lengths[0]; x++) {
        openvdb::Coord xyz(x, y, z);
        auto val = vdb_accessor.getValue(xyz);
        if (val == std::numeric_limits<float>::max()) {
          cout << "- ";
        } else {
          cout << val << " ";
        }
      }
      cout << '\n';
    }
    cout << '\n';
  }
}

// only values strictly greater than bkg_thresh are valid
template <typename T>
void print_vdb_mask(T vdb_accessor, const GridCoord &lengths,
    const GridCoord &offsets = {0, 0, 0}) {
  for (int z = offsets.z(); z < offsets.z() + lengths[2]; z++) {
    cout << "y | Z=" << z << '\n';
    for (int x = 0; x < 2 * lengths[0] + 4; x++) {
      cout << "-";
    }
    cout << '\n';
    for (int y = offsets.y(); y < offsets.y() + lengths[1]; y++) {
      cout << y << " | ";
      for (int x = offsets.x(); x < offsets.x() + lengths[0]; x++) {
        openvdb::Coord xyz(x, y, z);
        auto val = vdb_accessor.isValueOn(xyz);
        if (val) {
          cout << val << " ";
          //}
      } else {
        cout << "- ";
      }
    }
    cout << '\n';
  }
  cout << '\n';
}
}

auto ids_to_coords = [](auto ids, auto lengths) {
  return ids | rv::transform([&](auto v) { return id_to_coord(v, lengths); }) |
    rng::to_vector;
};

auto set_if_active = [](auto leaf, GridCoord coord) {
  if (leaf->isValueOn(coord))
    leaf->setValue(coord, true);
};

auto testbit = [](auto field_, auto bit_offset) {
  return static_cast<bool>(field_ & (1 << bit_offset));
};

auto setbit = [](auto field_, auto bit_offset) {
  return field_ | 1 << bit_offset;
};

auto unsetbit = [](auto field_, auto bit_offset) {
  return field_ &= ~(1 << bit_offset);
};

auto set_root = [](auto handle, auto ind) {
  // ???? 1???
  handle.set(*ind, setbit(handle.get(*ind), 3));
};

auto is_root = [](auto handle, auto ind) {
  return testbit(handle.get(*ind), 3);
};

auto set_selected = [](auto handle, auto ind) {
  // ???? ???1
  handle.set(*ind, setbit(handle.get(*ind), 0));
};

auto is_selected = [](auto handle, auto ind) {
  return testbit(handle.get(*ind), 0);
};

auto set_surface = [](auto handle, auto ind) {
  // ???? ??1?
  handle.set(*ind, setbit(handle.get(*ind), 1));
};

auto set_prune_visited = [](auto handle, auto ind) {
  // ???1 ????
  handle.set(*ind, setbit(handle.get(*ind), 4));
};

auto is_prune_visited = [](auto handle, auto ind) {
  // XXX? XXXX
  return testbit(handle.get(*ind), 4);
};

auto unset_tombstone = [](auto handle, auto ind) {
  // ??0? ????
  handle.set(*ind, unsetbit(handle.get(*ind), 5));
};

auto set_tombstone = [](auto handle, auto ind) {
  // ??1? ????
  handle.set(*ind, setbit(handle.get(*ind), 5));
};

auto is_tombstone = [](auto handle, auto ind) {
  // XX?X XXXX
  return testbit(handle.get(*ind), 5);
};

auto is_surface = [](auto handle, auto ind) {
  // XXXX XX?X
  return testbit(handle.get(*ind), 1);
};

auto valid_radius = [](auto handle, auto ind) {
  // XXXX XX?X
  return handle.get(*ind) != 0;
};

auto valid_parent = [](auto handle, auto ind) {
  auto parent = handle.get(*ind);
  return parent[0] || parent[1] || parent[2];
};

auto label = [](auto handle, auto ind) -> char {
  if (is_root(handle, ind)) {
    return 'R';
  }
  if (is_selected(handle, ind)) {
    return 'V';
  }
  return '?';
};

auto is_valid = [](auto flags_handle, auto ind, bool accept_tombstone = false) {
  if (is_selected(flags_handle, ind)) {
    if (accept_tombstone) {
      return true;
    } else {
      if (is_tombstone(flags_handle, ind)) {
        return false;
      } else {
        return true;
      }
    }
  }
  return false;
};

auto keep_root = [](const auto &flags_handle, const auto &parents_handle,
    const auto &radius_handle, auto ind) {
  return is_valid(flags_handle, ind) && is_root(flags_handle, ind);
};

auto not_root = [](const auto &flags_handle, const auto &parents_handle,
    const auto &radius_handle, auto ind) {
  return is_valid(flags_handle, ind) && !is_root(flags_handle, ind);
};

auto inactivates_visited = [](auto &flags_handle, const auto &parents_handle,
    const auto &radius_handle, const auto &ind,
    auto leaf) { leaf->setActiveState(ind, false); };

auto prunes_visited = [](auto &flags_handle, const auto &parents_handle,
    const auto &radius_handle, const auto &ind,
    auto leaf) { set_tombstone(flags_handle, ind); };

auto print_point_count = [](auto grid) {
  openvdb::Index64 count = openvdb::points::pointCount(grid->tree());
  std::cout << "Point count: " << count << '\n';
};

auto get_transform = []() {
  // grid_transform must use the same voxel size for all tiles
  // and be identical
  auto grid_transform =
    openvdb::math::Transform::createLinearTransform(VOXEL_SIZE);
  // The offset to cell-center point data grids
  //const openvdb::math::Vec3d offset(VOXEL_SIZE / 2., VOXEL_SIZE / 2.,
      //VOXEL_SIZE / 2.);
  //grid_transform->postTranslate(offset);
  return grid_transform;
};

auto copy_selected =
[](EnlargedPointDataGrid::Ptr grid) -> openvdb::FloatGrid::Ptr {
  auto float_grid = openvdb::FloatGrid::create();
  float_grid->setTransform(get_transform());

  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    auto float_leaf = new openvdb::tree::LeafNode<float, LEAF_LOG2DIM>(
        leaf_iter->origin(), 0.);

    // note: some attributes need mutability
    openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
        leaf_iter->attributeArray("flags"));

    uint32_t leaf_count = 0;
    for (auto ind = leaf_iter->beginIndexOn(); ind; ++ind) {
      if (is_selected(flags_handle, ind)) {
        float_leaf->setValue(ind.getCoord(), 1.);
        ++leaf_count;
      }
    }

    if (leaf_count) {
      float_grid->tree().addLeaf(float_leaf);
    }
  }

  return float_grid;
};

auto print_positions = [](auto grid) {
  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    auto bbox = leaf_iter->getNodeBoundingBox();
    std::cout << bbox << std::endl;

    // Extract the position attribute from the leaf by name (P is position).
    const openvdb::points::AttributeArray &array =
      leaf_iter->constAttributeArray("P");
    // Create a read-only AttributeHandle. Position always uses Vec3f.
    openvdb::points::AttributeHandle<PositionT> positionHandle(array);

    openvdb::points::AttributeHandle<float> radius_handle(
        leaf_iter->constAttributeArray("pscale"));

    openvdb::points::AttributeHandle<uint8_t> flags_handle(
        leaf_iter->constAttributeArray("flags"));

    openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
        leaf_iter->constAttributeArray("parents"));

    // Iterate over the point indices in the leaf.
    for (auto indexIter = leaf_iter->beginIndexOn(); indexIter; ++indexIter) {
      // Extract the voxel-space position of the point.
      openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);

      auto radius = radius_handle.get(*indexIter);

      auto recv_flags = flags_handle.get(*indexIter);

      auto recv_parent = parents_handle.get(*indexIter);

      // Extract the world-space position of the voxel.
      const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
      // Compute the world-space position of the point.
      openvdb::Vec3f worldPosition =
        grid->transform().indexToWorld(voxelPosition + xyz);
      // Verify the index and world-space position of the point
      std::cout << "ind:" << *indexIter << " ";
      std::cout << "xyz: " << xyz << ' ';
      std::cout << "WorldPosition=" << worldPosition << ' ';
      std::cout << coord_to_str(xyz) << " -> " << coord_to_str(recv_parent)
        << ' ';
      std::cout << +(recv_flags) << ' ';
      std::cout << +(radius) << '\n';
    }
  }
};

auto print_all_points = [](const EnlargedPointDataGrid::Ptr grid,
    openvdb::math::CoordBBox bbox,
    std::string stage = "label") {
  std::cout << "Print all points " << stage << "\n";
  // 3D visual
  for (int z = bbox.min()[2]; z < bbox.max()[2]; z++) {
    cout << "y | Z=" << z << '\n';
    for (int x = 0; x < 2 * bbox.extents()[0] + 4; x++) {
      cout << "-";
    }
    cout << '\n';
    for (int y = bbox.min()[1]; y < bbox.max()[1]; y++) {
      cout << y << " | ";
      for (int x = bbox.min()[0]; x < bbox.max()[0]; x++) {
        openvdb::Coord xyz(x, y, z);
        auto leaf_iter = grid->tree().probeConstLeaf(xyz);
        if (!leaf_iter) {
          cout << "- ";
          continue;
        }
        auto ind = leaf_iter->beginIndexVoxel(xyz);
        if (!ind) {
          cout << "- ";
          continue;
        }

        openvdb::points::AttributeHandle<float> radius_handle(
            leaf_iter->constAttributeArray("pscale"));

        openvdb::points::AttributeHandle<uint8_t> flags_handle(
            leaf_iter->constAttributeArray("flags"));

        openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
            leaf_iter->constAttributeArray("parents"));

        if (ind) {
          if (stage == "radius") {
            cout << +(radius_handle.get(*ind)) << " ";
          } else if (stage == "valid") {
            if (is_valid(flags_handle, ind)) {
              cout << "V ";
            } else {
              cout << "- ";
            }
          } else if (stage == "parent") {
            auto recv_parent = parents_handle.get(*ind);
            std::cout << coord_to_str(recv_parent);
            // if (valid_parent(parents_handle, ind)) {
            // cout << parents_handle.get(*ind) << " ";
            //} else {
            // cout << "- ";
            //}
          } else if (stage == "surface") {
            if (testbit(flags_handle.get(*ind), 1)) {
              // if (is_surface(flags_handle, ind)) {
              cout << "L "; // L for leaf and because disambiguates selected
                            // S
            } else {
              cout << "- ";
            }
            } else if (stage == "label" || stage == "connected") {
              // cout << label(flags_handle, ind) << ' ';
              cout << +(flags_handle.get(*ind)) << ' ';
            }
          } else {
            cout << "- ";
          }
        }
        cout << '\n';
      }
      cout << '\n';
    }
  };

  template <typename image_t>
    void print_image_3D(const image_t *inimg1d, const GridCoord lengths,
        const image_t bkg_thresh = 0) {
      auto total_count = coord_prod_accum(lengths);
      cout << "Print image 3D:\n";
      for (int zi = 0; zi < lengths[2]; zi++) {
        cout << "y | Z=" << zi << '\n';
        for (int xi = 0; xi < 2 * lengths[0] + 4; xi++) {
          cout << "-";
        }
        cout << '\n';
        for (int yi = 0; yi < lengths[1]; yi++) {
          cout << yi << " | ";
          for (int xi = 0; xi < lengths[0]; xi++) {
            auto index = coord_to_id(GridCoord(xi, yi, zi), lengths);
            assertm(index < total_count, "requested index is out of bounds");
            auto val = inimg1d[index];
            if (val > bkg_thresh) {
              cout << val << " ";
            } else {
              cout << "- ";
            }
          }
          cout << '\n';
        }
        cout << '\n';
      }
    }

  template <typename image_t> void print_image(image_t *inimg1d, VID_t size) {
    cout << "print image " << '\n';
    for (VID_t i = 0; i < size; i++) {
      cout << i << " " << +inimg1d[i] << '\n';
    }
  }

  // Note this test is on a single pixel width path through
  // the domain, thus it's an extremely hard test to pass
  // not even original fastmarching can
  // recover all the original pixels
  template <typename image_t>
    VID_t trace_mesh_image(VID_t id, image_t *inimg1d, const VID_t desired_selected,
        int grid_size) {
      VID_t i, j, k, ic, jc, kc;
      i = j = k = ic = kc = jc = 0;
      // set root to 1
      inimg1d[id] = 1;
      VID_t actual = 1; // count root
      srand(time(NULL));
      while (actual < desired_selected) {
        // calc i, j, k coords
        i = id % grid_size;
        j = (id / grid_size) % grid_size;
        k = (id / (grid_size * grid_size)) % grid_size;
        // std::cout << "previous " << id << " i " << i << " j " << j << " k " << k
        // << '\n';

        // try to find a suitable next location in one of the
        // six directions, if a proper direction was found
        // update the id and the coords to reflect
        // if this iterations direction is invalid skip to next
        long dir = rand() % 6;
        if (dir == 4) {
          if (k == 0) {
            continue;
          }
          k -= 1;
          id = id - grid_size * grid_size;
        } else if (dir == 2) {
          if (j == 0) {
            continue;
          }
          j -= 1;
          id = id - grid_size;
        } else if (dir == 0) {
          if (i == 0) {
            continue;
          }
          i -= 1;
          id = id - 1;
        } else if (dir == 1) {
          if (i == grid_size - 1) {
            continue;
          }
          i += 1;
          id = id + 1;
        } else if (dir == 3) {
          if (j == grid_size - 1) {
            continue;
          }
          j += 1;
          id = id + grid_size;
        } else if (dir == 5) {
          if (k == grid_size - 1) {
            continue;
          }
          k += 1;
          id = id + grid_size * grid_size;
        }
        // std::cout << "node id" << id << " i " << i << " j " << j << " k " << k <<
        // '\n';
        // Note id is allowed to slide along
        // previously visited until it finds a new selection
        // this prevents it from every getting guaranteed stuck
        // surrounded on all 6 sides, in practice however, this can
        // make high slt_pct very inefficient since, it's hard to find
        // last unselected pixels randomly
        get_img_coord(id, ic, jc, kc, grid_size);
        assert(ic == i);
        assert(jc == j);
        assert(kc == k);
        // already selected doesn't count
        if (inimg1d[id] != 1) {
          inimg1d[id] = 1;
          actual++;
        }
      }
      return actual;
    }

  bool is_covered_by_parent(VID_t index, VID_t root_vid, int radius,
      VID_t grid_size) {
    VID_t i, j, k, pi, pj, pk;
    get_img_coord(index, i, j, k, grid_size);
    get_img_coord(root_vid, pi, pj, pk, grid_size);
    auto x = static_cast<double>(i) - pi;
    auto y = static_cast<double>(j) - pj;
    auto z = static_cast<double>(k) - pk;
    auto vdistance = sqrt(x * x + y * y + z * z);

    if (static_cast<double>(radius) >= vdistance) {
      return true;
    }
    return false;
  }

  auto print_descriptor = [](auto grid_base) {
    // if (grid_base->isType<EnlargedPointDataGrid>()) {
    // Warning custom types must be registered before writing or reading them
    auto gridPtr = openvdb::gridPtrCast<EnlargedPointDataGrid>(grid_base);
    auto leafIter = gridPtr->tree().cbeginLeaf();
    if (leafIter) {
      const openvdb::points::AttributeSet::Descriptor &descriptor =
        leafIter->attributeSet().descriptor();

      std::cout << "Number of Attributes: " << descriptor.size() << std::endl;

      for (auto it : descriptor.map()) {
        int index = it.second;
        std::cout << "Attribute[" << it.second << "]" << std::endl;

        std::string name = it.first;
        std::cout << "\tName = " << it.first << std::endl;

        const openvdb::NamePair &type = descriptor.type(index);
        std::cout << "\tValueType = " << type.first << std::endl;
        std::cout << "\tCodec = " << type.second << std::endl;
      }
    }
    //}
  };

  template <typename T> void print_grid_metadata(T vdb_grid) {
    // stats need to be refreshed manually after any changes with addStatsMetadata
    vdb_grid->addStatsMetadata();

    if (std::is_same<T, EnlargedPointDataGrid::Ptr>::value)
      print_descriptor(vdb_grid);

    auto mem_usage_bytes = vdb_grid->memUsage();
    auto active_voxel_dim = vdb_grid->evalActiveVoxelDim();
    VID_t hypo_bound_vol_count = static_cast<VID_t>(active_voxel_dim[0]) *
      active_voxel_dim[1] * active_voxel_dim[2];
    auto active_voxel_count = vdb_grid->activeVoxelCount();
    auto fg_pct =
      (static_cast<double>(100) * active_voxel_count) / hypo_bound_vol_count;

    cout << "Metadata for vdb grid name: " << vdb_grid->getName()
      << " creator: " << vdb_grid->getCreator() << '\n';
    cout << "Grid class: "
      << vdb_grid->gridClassToString(vdb_grid->getGridClass()) << '\n';
    for (openvdb::MetaMap::MetaIterator iter = vdb_grid->beginMeta();
        iter != vdb_grid->endMeta(); ++iter) {
      const std::string &name = iter->first;
      openvdb::Metadata::Ptr value = iter->second;
      std::string valueAsString = value->str();
      std::cout << name << " = " << valueAsString << '\n';
    }

    // cout << "Tree type: "
    // cout << "Value type: "
    // cout << "Leaf count: " << vdb_grid->tree().print() << '\n';
    // cout << "Leaf count: " << vdb_grid->tree().leafCount() << '\n';
    cout << "Active voxel_dim: " << active_voxel_dim << '\n';
    cout << "Mem usage GB: " << static_cast<double>(mem_usage_bytes) / (1 << 30)
      << '\n';
    cout << "Active voxel count: " << active_voxel_count << '\n';
    cout << "Bytes per active voxel count: "
      << mem_usage_bytes / static_cast<double>(active_voxel_count) << '\n';
    cout << "Foreground (%): " << fg_pct << '\n';
    cout << "Hypothetical bounding volume voxel count: " << hypo_bound_vol_count
      << '\n';
    cout << "Compression factor over 2 byte hypothetical bounding volume: "
      << static_cast<double>(hypo_bound_vol_count * 2) / mem_usage_bytes
      << '\n';
    cout << "Usage bytes per hypothetical bounding voxel count (multiplier): "
      << static_cast<double>(mem_usage_bytes) / hypo_bound_vol_count << '\n';
    cout << '\n';
  }

  auto set_grid_meta = [](auto grid, auto lengths, float requested_fg_pct = -1,
      int channel = 0, int resolution_level = 0,
      std::string name = "topology", int upsample_z = 1) {
    grid->setName(name);
    grid->setCreator("recut");
    grid->setIsInWorldSpace(true);
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    grid->insertMeta("channel", openvdb::Int32Metadata(channel));
    grid->insertMeta("resolution_level",
        openvdb::Int32Metadata(resolution_level));
    grid->insertMeta("original_bounding_extent_x",
        openvdb::FloatMetadata(static_cast<float>(lengths[0])));
    grid->insertMeta("original_bounding_extent_y",
        openvdb::FloatMetadata(static_cast<float>(lengths[1])));
    grid->insertMeta("original_bounding_extent_z",
        openvdb::FloatMetadata(static_cast<float>(lengths[2])));
    grid->insertMeta("requested_fg_pct",
        openvdb::FloatMetadata(requested_fg_pct));
    grid->insertMeta("upsample_z_factor", openvdb::Int32Metadata(upsample_z));
  };

  auto copy_to_point_grid = [](openvdb::FloatGrid::Ptr other, auto lengths,
      float requested_fg_pct = -1) {
    throw std::runtime_error("incomplete implementation");

    // Use the topology to create a PointDataTree
    vp::PointDataTree::Ptr pointTree(
        new vp::PointDataTree(other->tree(), 0, openvdb::TopologyCopy()));

    // Ensure all tiles have been voxelized
    pointTree->voxelizeActiveTiles();

    using PositionAttribute =
      openvdb::points::TypedAttributeArray<PositionT, FPCodec>;

    openvdb::NamePair positionType = PositionAttribute::attributeType();
    // Create a new Attribute Descriptor with position only
    openvdb::points::AttributeSet::Descriptor::Ptr descriptor(
        openvdb::points::AttributeSet::Descriptor::create(positionType));

    // Determine the number of points / voxel and points / leaf.
    openvdb::Index leaf_voxels =
      EnlargedPointDataGrid::TreeType::LeafNodeType::SIZE;
    openvdb::Index leaf_points = VOXEL_POINTS * leaf_voxels;

    // Iterate over the leaf nodes in the point tree.
    for (auto leafIter = pointTree->beginLeaf(); leafIter; ++leafIter) {
      // Initialize the attributes using the descriptor and point count.
      leafIter->initializeAttributes(descriptor, leaf_points);

      //// Initialize the voxel offsets
      // openvdb::Index offset(0);
      // for (openvdb::Index index = 0; index < leaf_voxels; ++index) {
      // offset += VOXEL_POINTS;
      // leafIter->setOffsetOn(index, offset);
      //}
    }

    auto grid = EnlargedPointDataGrid::create(pointTree);

    cout << "initial point count\n";
    print_point_count(grid);

    grid->tree().prune();

    set_grid_meta(grid, lengths, requested_fg_pct);

    return grid;
  };

  auto create_point_grid = [](auto &positions, auto lengths, auto transform_ptr,
      float requested_fg_pct = -1) {
    // The VDB Point-Partioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    vp::PointAttributeVector<PositionT> wrapper(positions);

    auto point_index_grid = vto::createPointIndexGrid<EnlargedPointIndexGrid>(
        wrapper, *transform_ptr);

    auto grid =
      openvdb::points::createPointDataGrid<FPCodec, EnlargedPointDataGrid>(
          *point_index_grid, wrapper, *transform_ptr);

    grid->tree().prune();

    set_grid_meta(grid, lengths, requested_fg_pct);

    return grid;
  };

  auto create_vdb_grid = [](auto lengths, float requested_fg_pct = -1) {
    auto topology_grid = EnlargedPointDataGrid::create();
    set_grid_meta(topology_grid, lengths, requested_fg_pct);

    return topology_grid;
  };

  // was this grid previously run through the connected components stage?
  // std::any_of has confusing bugs so keep classic implementation below:
  auto is_connected = [](auto vdb_grid) {
    for (openvdb::MetaMap::MetaIterator iter = vdb_grid->beginMeta();
        iter != vdb_grid->endMeta(); ++iter) {
      if (iter->first == "connected") {
        return true;
      }
    }
    return false;
  };

  auto get_metadata = [](auto vdb_grid) -> std::pair<GridCoord, float> {
    GridCoord image_lengths;
    float requested_fg_pct = -1; // default value if not found acceptable
    for (openvdb::MetaMap::MetaIterator iter = vdb_grid->beginMeta();
        iter != vdb_grid->endMeta(); ++iter) {
      // name and val
      const std::string &name = iter->first;
      openvdb::Metadata::Ptr value = iter->second;

      if (name == "file_bbox_max") {
        openvdb::Vec3I v = static_cast<openvdb::Vec3IMetadata &>(*value).value();
        image_lengths = GridCoord(v);
      } else if (name == "requested_fg_pct") {
        requested_fg_pct = static_cast<openvdb::FloatMetadata &>(*value).value();
      }
    }
    return std::pair(image_lengths, requested_fg_pct);
  };

  auto append_attributes = [](auto grid) {
    // Append a "radius" attribute to the grid to hold the radius.
    // Note that this attribute type is not registered by default so needs to be
    // explicitly registered.
    using Codec = openvdb::points::NullCodec;
    openvdb::points::TypedAttributeArray<float, Codec>::registerType();
    openvdb::NamePair radiusAttribute =
      openvdb::points::TypedAttributeArray<float, Codec>::attributeType();
    openvdb::points::appendAttribute(grid->tree(), "pscale", radiusAttribute);

    // append a state flag attribute
    openvdb::points::TypedAttributeArray<uint8_t, Codec>::registerType();
    openvdb::NamePair flagsAttribute =
      openvdb::points::TypedAttributeArray<uint8_t, Codec>::attributeType();
    openvdb::points::appendAttribute(grid->tree(), "flags", flagsAttribute);

    // append a parent offset attribute
    openvdb::points::TypedAttributeArray<OffsetCoord, Codec>::registerType();
    openvdb::NamePair parentAttribute =
      openvdb::points::TypedAttributeArray<OffsetCoord, Codec>::attributeType();
    openvdb::points::appendAttribute(grid->tree(), "parents", parentAttribute);

    // append a value attribute for fastmarching
    openvdb::NamePair valueAttribute =
      openvdb::points::TypedAttributeArray<float, Codec>::attributeType();
    openvdb::points::appendAttribute(grid->tree(), "value", valueAttribute);
  };

  auto read_vdb_file(fs::path fn, std::string grid_name = "") {
#ifdef LOG
    // cout << "Reading vdb file: " << fn << " grid: " << grid_name << " ...\n";
#endif
    if (!fs::exists(fn)) {
      cout << "Input image file does not exist or not found, exiting...\n";
      exit(1);
    }
    openvdb::io::File file(fn.string());
    file.open();
    if (grid_name.empty()) {
      grid_name = file.beginName().gridName(); // get the 1st grid
    }
    openvdb::GridBase::Ptr base_grid = file.readGrid(grid_name);
    file.close();

#ifdef LOG_FULL
    print_grid_metadata(base_grid);
#endif
    return base_grid;
  }

  // Create a VDB file object and write out a vector of grids.
  // Add the grid pointer to a container.
  // openvdb::GridPtrVec grids;
  // grids.push_back(grid);
  void write_vdb_file(openvdb::GridPtrVec vdb_grids, fs::path fp = "") {
    // safety checks
    auto default_fn = "topology.vdb";
    if (fp == "") {
      fp = get_data_dir() / default_fn;
    } else {
      auto dir = fp.parent_path();
      if (dir == "" || fs::exists(dir)) {
        if (fs::exists(fp)) {
#ifdef LOG
          std::cout << "Warning: " << fp << " already exists, overwriting...\n";
#endif
        }
      } else {
#ifdef LOG
        std::cout << "Directory: " << dir << " does not exist, creating...\n";
#endif
        fs::create_directories(dir);
      }
    }

    auto timer = new high_resolution_timer();
    openvdb::io::File vdb_file(fp.string());
    vdb_file.write(vdb_grids);
    vdb_file.close();
  }

  /*
   * sets all to 1 for tcase 0
   * tcase4 : trace_mesh_image
   * tcase5 : sphere grid
   * tcase6 : reserved for real images throws error if passed otherwise
   * tcase7 : cube of selected centered at root, side length grid_size /2
   * takes an empty binarized inimg1d (all zeros)
   * and creates a central sphere of specified
   * radius directly in the center of the grid
   */
  template <typename image_t>
    VID_t create_image(int tcase, image_t *inimg1d, int grid_size,
        const VID_t desired_selected, VID_t root_vid) {

      // need to count total selected for tcase 3 and 5
      VID_t count_selected_pixels = 0;
      assertm(desired_selected > 0, "must select at least 1 pixel: the root");

      // for tcase 5 sphere grid
      auto radius = grid_size / 4;
      radius = radius > 1 ? radius : 1; // clamp to 1
      double tcase1_factor = PI;
      double tcase2_factor = 2 * PI;

      assertm(grid_size / 2 >= radius,
          "Can't specify a radius larger than grid_size / 2");
      auto root_x = static_cast<int>(get_central_coord(grid_size));
      auto root_y = static_cast<int>(get_central_coord(grid_size));
      auto root_z = static_cast<int>(get_central_coord(grid_size));
      auto xmin = std::clamp(root_x - radius, 0, grid_size - 1);
      auto xmax = std::clamp(root_x + radius, 0, grid_size - 1);
      auto ymin = std::clamp(root_y - radius, 0, grid_size - 1);
      auto ymax = std::clamp(root_y + radius, 0, grid_size - 1);
      auto zmin = std::clamp(root_z - radius, 0, grid_size - 1);
      auto zmax = std::clamp(root_z + radius, 0, grid_size - 1);

      double dh = 1 / grid_size;
      double x, y, z;
      double w = 1 / 24;
      bool found_barrier_region;
      for (int xi = 0; xi < grid_size; xi++) {
        for (int yi = 0; yi < grid_size; yi++) {
          for (int zi = 0; zi < grid_size; zi++) {
            VID_t index = ((VID_t)xi) + yi * grid_size + zi * grid_size * grid_size;
            x = xi * dh;
            y = yi * dh;
            z = zi * dh;
            if (tcase == 0) {
              inimg1d[index] = 1;
            } else if (tcase == 1) {
              inimg1d[index] = (uint16_t)1 + .5 * sin(tcase1_factor * x) *
                sin(tcase1_factor * y) *
                sin(tcase1_factor * z);
            } else if (tcase == 2) {
              inimg1d[index] = (uint16_t)1 - .99 * sin(tcase2_factor * x) *
                sin(tcase2_factor * y) *
                sin(tcase2_factor * z);
            } else if (tcase == 3) {
              double r = sqrt(x * x + y * y);
              double R = sqrt(x * x + y * y + z * z);
              found_barrier_region = false;
              std::vector<double> Rvecs = {.15, .25, .35, .45};
              for (std::vector<double>::size_type ri = 0; ri < Rvecs.size(); ri++) {
                double Rvec = Rvecs[ri];
                bool condition0, condition1;
                condition0 = condition1 = false;
                if (Rvec < R && R < Rvec + w) {
                  condition0 = true;
                }
                if (ri == 0) {
                  if (r < .05) {
                    if (z < 0) {
                      condition1 = true;
                    }
                  }
                } else if (r < .1) {
                  if (ri % 2 == 0) {
                    if (z < 0) {
                      condition1 = true;
                    }
                  } else {
                    if (z > 0) {
                      condition1 = true;
                    }
                  }
                }
                if (!condition0 != !condition1) { // xor / set difference
                  found_barrier_region = true;
                  break;
                }
              }
              if (found_barrier_region) {
                inimg1d[index] = 0;
              } else {
                inimg1d[index] = 1;
                count_selected_pixels++;
              }
            } else if (tcase == 4) { // 4 start with zero grid, and select in
                                     // function trace_mesh_image
              inimg1d[index] = 0;
            } else if (tcase == 5) {
              // make an accurate radius sphee centered around the root
              if (is_covered_by_parent(index, root_vid, radius, grid_size)) {
                inimg1d[index] = 1;
                count_selected_pixels++;
              } else {
                inimg1d[index] = 0;
              }
            } else if (tcase == 7) {
              // make a square centered around root
              inimg1d[index] = 0;
              if ((xi >= xmin) && (xi <= xmax)) {
                if ((yi >= ymin) && (yi <= ymax)) {
                  if ((zi >= zmin) && (zi <= zmax)) {
                    inimg1d[index] = 1;
                    count_selected_pixels++;
                  }
                }
              }
            }
          }
        }
      }

      // root will always be selected
      // therefore there will always be at least 1 selected
      // tcase 4 will also select root in a special way
      // in the trace_mesh_image method
      // note root is always marked as selected at run time by recut
      // since it is the seed location
      if (tcase != 4) {
        if (inimg1d[root_vid] == 0) {
          inimg1d[root_vid] = 1;
          count_selected_pixels++;
        }
      }

      // return number of pixels selected
      // must count total selected for all tcase 4 and above since it may not match
      // desired
      if (tcase < 3) {
        return grid_size * grid_size * grid_size;
      } else if ((tcase == 3) || (tcase == 5)) {
        return count_selected_pixels;
      } else if (tcase == 4) {
        return trace_mesh_image(root_vid, inimg1d, desired_selected, grid_size);
      } else if (tcase == 7) {
        return count_selected_pixels;
      }

      // tcase 6 means real image so it's not valid
      assertm(false, "tcase not recognized: tcase 6 is reserved for reading real "
          "images\n tcase higher than 7 not specified");
      return 0; // never reached
    }

  /**
   * DEPRECATED
   * create a set of lines in each dimension for the domain
   * @param start the root linear index line paths must connect to
   * this
   * @param grid_size the dimension along each dimension for testing
   * this is always the same
   * @param inim1d the simple array image
   * @param line_per_dim the number of lines per
   */
  VID_t lattice_grid(VID_t start, uint16_t *inimg1d, int line_per_dim,
      int grid_size) {
    int tile = grid_size / line_per_dim; // roughly equiv
    std::vector<VID_t> x(line_per_dim + 1);
    std::vector<VID_t> y(line_per_dim + 1);
    std::vector<VID_t> z(line_per_dim + 1);
    VID_t i, j, k, count;
    i = j = k = 0;
    count = 0;
    VID_t selected = 0;
    for (int count = 0; count < grid_size; count += tile) {
      x.push_back(count);
      y.push_back(count);
      z.push_back(count);
      std::cout << "Count " << count << '\n';
    }
    get_img_coord(start, i, j, k, grid_size);
    x.push_back(i);
    y.push_back(j);
    z.push_back(k);

    for (auto &xi : x) {
      for (auto &yi : y) {
        for (int zi = 0; zi < grid_size; zi++) {
          int index = int(xi + yi * grid_size + zi * grid_size * grid_size);
          if (inimg1d[index] != 1) {
            inimg1d[index] = 1; // set to max
            selected++;
          }
        }
      }
    }

    for (auto &xi : x) {
      for (auto &zi : z) {
        for (int yi = 0; yi < grid_size; yi++) {
          int index = int(xi + yi * grid_size + zi * grid_size * grid_size);
          if (inimg1d[index] != 1) {
            inimg1d[index] = 1; // set to max
            selected++;
          }
        }
      }
    }

    for (auto &yi : y) {
      for (auto &zi : z) {
        for (int xi = 0; xi < grid_size; xi++) {
          int index = int(xi + yi * grid_size + zi * grid_size * grid_size);
          if (inimg1d[index] != 1) {
            inimg1d[index] = 1; // set to max
            selected++;
          }
        }
      }
    }
    return selected;
  }

  RecutCommandLineArgs get_args(int grid_size, int tile_length, int block_size,
      int slt_pct, int tcase, bool input_is_vdb = false,
      std::string input_type = "point",
      std::string output_type = "point",
      int downsample_factor = 1) {

    bool print = false;

    RecutCommandLineArgs args;
    args.input_type = input_type;
    args.output_type = output_type;
    args.convert_only = false; // default is true from CL
    fs::path data_dir_path = get_data_dir();
    args.seed_path = data_dir_path / "test_markers" / std::to_string(grid_size) /
      ("tcase" + std::to_string(tcase)) /
      ("slt_pct" + std::to_string(slt_pct));
    auto lengths = GridCoord(grid_size);
    args.image_lengths = lengths;
    args.image_offsets = zeros();

    // tcase 6 means use real data, in which case we need to either
    // set max and min explicitly (to save time) or recompute what the
    // actual values are
    if (tcase == 6) {
      // selected percent is only use for tcase 6 and 4
      // otherwise it is ignored for other tcases so that
      // nothing is recalculated
      // note: a background_thresh of 0 would simply take all pixels within the
      // domain and check that all were used

      args.image_offsets = {1123 / downsample_factor, 12947 / downsample_factor, 342};
      args.image_lengths = {grid_size, grid_size, grid_size};

      if (const char *env_p = std::getenv("TEST_IMAGE")) {
        std::cout << "Using $TEST_IMAGE environment variable: " << env_p << '\n';
        args.input_path = std::string(env_p);
      } else {
        std::cout << "Warning likely fatal: must run: export "
          "TEST_IMAGE=\"abs/path/to/image\" to set the environment "
          "variable\n\n";
      }

      if (const char *env_p = std::getenv("TEST_MARKER")) {
        std::cout << "Using $TEST_MARKER environment variable: " << env_p << '\n';
        args.seed_path = std::filesystem::path{env_p};
      } else {
        std::cout << "Warning likely fatal must run: export "
          "TEST_MARKER=\"abs/path/to/marker\" to set the environment "
          "variable\n\n";
      }

      // foreground_percent is always double between .0 and 1.
      args.foreground_percent = static_cast<double>(slt_pct) / 100.;
      // pre-determined and hardcoded thresholds for the file above
      // to save time recomputing is disabled
    } else {
      // by setting the max intensities you do not need to recompute them
      // in the update function, this is critical for benchmarking
      args.max_intensity = 2;
      args.min_intensity = 0;
      fs::path input_path = data_dir_path / "test_images" /
        std::to_string(grid_size) /
        ("tcase" + std::to_string(tcase)) /
        ("slt_pct" + std::to_string(slt_pct));
      if (input_is_vdb) {
        if (args.input_type == "point") {
          args.input_path = input_path / "point.vdb";
        } else if (args.input_type == "float") {
          args.input_path = input_path / "float.vdb";
        }
      } else {
        args.input_path = input_path / "ch0";
      }
    }

    // the total number of blocks allows more parallelism
    VID_t img_vox_num = grid_size * grid_size * grid_size;
    args.tcase = tcase;
    args.slt_pct = slt_pct;
    args.selected = img_vox_num * (slt_pct / (float)100);
    args.root_vid = get_central_vid(grid_size);

    if (print)
      args.PrintParameters();

    return args;
  }

  void write_marker(VID_t x, VID_t y, VID_t z, unsigned int radius, fs::path fn,
      GridCoord voxel_size) {
    auto print = false;
#ifdef LOG
    print = true;
#endif

    bool rerun = false;
    auto volume = ((4 * PI) / 3.) * pow(radius, 3);
    if (!fs::exists(fn) || rerun) {
      fs::remove_all(fn); // make sure it's an overwrite
      if (print)
        cout << "      Delete old: " << fn << '\n';
      fs::create_directories(fn);
      fn /= ("marker_" + std::to_string(static_cast<int>(x * voxel_size[0])) +
          "_" + std::to_string(static_cast<int>(y * voxel_size[1])) + "_" +
          std::to_string(static_cast<int>(z * voxel_size[2])) + "_" +
          std::to_string(static_cast<int>(volume)));
      std::ofstream mf;
      mf.open(fn.string());
      mf << "# x,y,z,radius\n";
      mf << x << ',' << y << ',' << z << ',' << radius;
      mf.close();
      if (print)
        cout << "      Wrote marker: " << fn << '\n';
    }
  }

  template <typename image_t> struct Histogram {
    std::map<image_t, uint64_t> bin_counts;
    image_t granularity;

    // granularity : the range of pixel values for each bin
    Histogram(image_t granularity = 8) : granularity(granularity) {}

    void operator()(image_t val) {
      auto i = val / granularity;
      if (bin_counts.count(i)) {
        bin_counts[i] = bin_counts[i] + 1;
      } else {
        bin_counts[i] = 1;
      }
    }

    int size() const { return bin_counts.size(); }

    // print to csv
    friend std::ostream &operator<<(std::ostream &os,
        const Histogram<image_t> &hist) {

      uint64_t cumulative_count = 0;
      rng::for_each(hist.bin_counts, [&cumulative_count](const auto kvalpair) {
          cumulative_count += kvalpair.second;
          });
      if (cumulative_count == 0) {
        throw std::domain_error("S-curve histogram has cumulative count of 0");
      }

      // metadata
      os << "# granularity " << hist.granularity << '\n';
      os << "# total " << cumulative_count << '\n';
      os << "range,count,%,cumulative %\n";

      // relies on map being forward ordered for the clean cumulative impl
      double cumulative_pct = 0.;
      for (const auto [key, value] : hist.bin_counts) {
        auto pct_double = (100 * static_cast<double>(value)) / cumulative_count;
        cumulative_pct += pct_double;
        os << hist.granularity * key << ',' << value << ',' << pct_double << ','
          << cumulative_pct << '\n';
      }

      return os;
    }

    uint64_t operator[](const image_t key) const {
      return this->bin_counts.at(key);
    }

    // FIXME check template types match
    Histogram<image_t> operator+(Histogram<image_t> const &rhistogram) {
      if (rhistogram.granularity != this->granularity) {
        throw std::runtime_error("Granularities mistmatch");
      }

      auto merged_histogram = Histogram<image_t>(this->granularity);

      auto these_keys = this->bin_counts | rv::keys | rng::to_vector;
      auto rhs_keys = rhistogram.bin_counts | rv::keys | rng::to_vector;

      auto matches = rv::set_intersection(these_keys, rhs_keys) | rng::to_vector;

      // overwrite all shared keys with summed values
      for (auto key : matches) {
        if (rhistogram.bin_counts.count(key)) {
          const auto rhist_value = rhistogram[key];
          merged_histogram.bin_counts[key] =
            this->bin_counts.at(key) + rhistogram.bin_counts.at(key);
        }
      }

      auto add_difference = [&matches, &merged_histogram](
          const auto reference_histogram, const auto keys) {
        for (const auto key : rv::set_difference(keys, matches)) {
          merged_histogram.bin_counts[key] =
            reference_histogram.bin_counts.at(key);
        }
      };

      // add all potential unique keys from this
      add_difference(*this, these_keys);
      // add all potential unique keys from rhs
      add_difference(rhistogram, rhs_keys);
      return merged_histogram;
    }

    Histogram<image_t> &operator+=(Histogram<image_t> const &rhistogram) {
      *this = *this + rhistogram;
      return *this;
    }
  };

  template <typename image_t>
    Histogram<image_t> hist(image_t *buffer, GridCoord buffer_lengths,
        GridCoord buffer_offsets, int granularity = 2) {
      auto histogram = Histogram<image_t>(granularity);
      for (auto z : rv::iota(0, buffer_lengths[2])) {
        for (auto y : rv::iota(0, buffer_lengths[1])) {
          for (auto x : rv::iota(0, buffer_lengths[0])) {
            GridCoord xyz(x, y, z);
            GridCoord buffer_xyz = coord_add(xyz, buffer_offsets);
            auto val = buffer[coord_to_id(buffer_xyz, buffer_lengths)];
            histogram(val);
          }
        }
      }
      return histogram;
    }

  // keep only voxels strictly greater than bkg_thresh
  auto convert_buffer_to_vdb_acc =
    [](auto buffer, GridCoord buffer_lengths, GridCoord buffer_offsets,
        GridCoord image_offsets, auto accessor, std::string grid_type,
        auto bkg_thresh = 0, int upsample_z = 1) {
      for (auto z : rv::iota(0, buffer_lengths[2])) {
        for (auto y : rv::iota(0, buffer_lengths[1])) {
          for (auto x : rv::iota(0, buffer_lengths[0])) {
            GridCoord xyz(x, y, z);
            GridCoord buffer_xyz = coord_add(xyz, buffer_offsets);
            GridCoord grid_xyz = coord_add(xyz, image_offsets);
            auto val = buffer[coord_to_id(buffer_xyz, buffer_lengths)];
            // auto val = std::get<T>(buffer[coord_to_id(buffer_xyz,
            // buffer_lengths)]);
            //  voxels equal to bkg_thresh are always discarded
            if (val > bkg_thresh) {
              for (auto upsample_z_idx : rv::iota(0, upsample_z)) {
                auto upsample_grid_xyz =
                  GridCoord(grid_xyz[0], grid_xyz[1],
                      (upsample_z * grid_xyz[2]) + upsample_z_idx);
                if (grid_type == "mask") {
                  accessor.setValueOn(upsample_grid_xyz);
                } else if (grid_type == "uint8") {
                  accessor.setValue(upsample_grid_xyz,
                      std::clamp(static_cast<uint8_t>(val),
                        static_cast<uint8_t>(0),
                        static_cast<uint8_t>(255)));
                } else if (grid_type == "float") {
                  accessor.setValue(upsample_grid_xyz, static_cast<float>(val));
                } else {
                  throw std::runtime_error("Unknown grid type");
                }
              }
            }
          }
        }
      }
    };

  // keep only voxels strictly greater than bkg_thresh
  auto convert_buffer_to_vdb = [](auto buffer, GridCoord buffer_lengths,
      GridCoord buffer_offsets,
      GridCoord image_offsets, auto &positions,
      auto bkg_thresh = 0, int upsample_z = 1) {
    // print_coord(buffer_lengths, "buffer_lengths");
    // print_coord(buffer_offsets, "buffer_offsets");
    // print_coord(image_offsets, "image_offsets");
    for (auto z : rv::iota(0, buffer_lengths[2])) {
      for (auto y : rv::iota(0, buffer_lengths[1])) {
        for (auto x : rv::iota(0, buffer_lengths[0])) {
          GridCoord xyz(x, y, z);
          GridCoord buffer_xyz = coord_add(xyz, buffer_offsets);
          GridCoord grid_xyz = coord_add(xyz, image_offsets);
          auto val = buffer[coord_to_id(buffer_xyz, buffer_lengths)];
          // voxels equal to bkg_thresh are always discarded
          if (val > bkg_thresh) {
            for (auto upsample_z_idx : rv::iota(0, upsample_z)) {
              positions.push_back(
                  PositionT(grid_xyz[0], grid_xyz[1],
                    (upsample_z * grid_xyz[2]) + upsample_z_idx));
            }
          }
        }
      }
    }
  };

  //// keep only voxels strictly greater than bkg_thresh
  // auto convert_buffer_leaf = [](auto buffer, GridCoord buffer_lengths,
  // GridCoord buffer_offsets, GridCoord image_offsets,
  // auto grid,
  // auto bkg_thresh = 0) {
  // auto leaf_iter = grid->tree().probeLeaf(coord);
  // auto ind = leaf_iter->beginIndexVoxel(coord);
  // auto leaf_iter = grid->
  // auto bbox = leaf_iter->getNodeBoundingBox();
  // std::cout << "Leaf BBox: " << bbox << '\n';
  // for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
  // auto bbox = leaf_iter->getNodeBoundingBox();
  //// print_coord(buffer_lengths, "buffer_lengths");
  //// print_coord(buffer_offsets, "buffer_offsets");
  //// print_coord(image_offsets, "image_offsets");
  // for (auto z : rv::iota(0, buffer_lengths[2])) {
  // for (auto y : rv::iota(0, buffer_lengths[1])) {
  // for (auto x : rv::iota(0, buffer_lengths[0])) {
  // GridCoord xyz(x, y, z);
  // GridCoord buffer_xyz = coord_add(xyz, buffer_offsets);
  // GridCoord grid_xyz = coord_add(xyz, image_offsets);
  // auto val = buffer[coord_to_id(buffer_xyz, buffer_lengths)];
  //// voxels equal to bkg_thresh are always discarded
  // if (val > bkg_thresh) {
  // position_handle.set(*ind, xyz);
  //}
  //}
  //}
  //}
  //}
  //} ;

  // passing a page number >= 0 means write a multipage tiff file
  template <typename image_t>
    void write_tiff_page(image_t *inimg1d, TIFF *tiff, const GridCoord dims,
        uint32_t page_number) {

      TIFFSetField(tiff, TIFFTAG_PAGENUMBER, page_number, page_number);
      // TIFFSetField(tiff, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
      encoded_tiff_write(inimg1d, tiff, dims);

      // for (unsigned int y = 0; y < dims[1]; ++y) {
      // TIFFWriteScanline(tiff, inimg1d + y * dims[0], y, 0);
      //}

      TIFFWriteDirectory(tiff);
    }

  // passing a page number >= 0 means write a multipage tiff file
  template <typename image_t>
    void write_single_z_plane(image_t *inimg1d, fs::path fn, const GridCoord dims) {

      TIFF *tiff = TIFFOpen(fn.string().c_str(), "w");
      if (!tiff) {
        throw std::runtime_error(
            "ERROR reading (not existent, not accessible or no TIFF file)");
      }

      encoded_tiff_write(inimg1d, tiff, dims);
      // for (unsigned int y = 0; y < dims[1]; ++y) {
      // TIFFWriteScanline(tiff, inimg1d + y * dims[0], y, 0);
      //}

      TIFFClose(tiff);
    }

  template <typename image_t = uint16_t>
    void write_tiff(image_t *inimg1d, fs::path base, const GridCoord dims,
        bool rerun = false) {
      auto print = false;
#ifdef LOG
      print = true;
#endif

      base /= "ch0";
      if (!fs::exists(base) || rerun) {
        fs::remove_all(base); // make sure it's an overwrite
        if (print)
          cout << "      Delete old: " << base << '\n';
        fs::create_directories(base);
        for (int z = 0; z < dims[2]; ++z) {
          std::ostringstream fn;
          fn << "img_" << std::setfill('0') << std::setw(6) << z << ".tif";
          VID_t start = z * dims[0] * dims[1];
          write_single_z_plane(&(inimg1d[start]), base / fn.str(), dims);
        }
        if (print)
          cout << "      Wrote test images in: " << base << '\n';
      }
    }

  auto open_tiff_file = [](std::string fn) {
    // try reading file
    if (!fs::exists(fn)) {
      throw std::runtime_error("ERROR non existent TIFF file " + fn);
    }
    TIFF *tiff;
    try {
      tiff = TIFFOpen(&fn[0], "r");
    } catch (...) {
      throw std::runtime_error("libtiff threw while during TIFFOpen " + fn);
    }
    if (!tiff) {
      throw std::runtime_error("ERROR could not open TIFF file " + fn);
    }

    // not supported currently
    if (TIFFIsTiled(tiff)) {
      throw std::runtime_error(
          "ERROR TIFF file must be striped, instead found tiled file");
    }

    short samples_per_pixel;
    TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    if (samples_per_pixel > 1) {
      throw std::runtime_error(
          "Recut does not support TIFFs with samples per pixel > 1 yet");
    }

    return tiff;
  };

  auto read_tiff_bit_width = [](auto fn) {
    auto tiff = open_tiff_file(fn);
    short bits_per_sample;
    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFClose(tiff);
    return bits_per_sample;
  };

  auto read_tiff_dims = [](auto fn) {
    auto tiff = open_tiff_file(fn);
    uint32_t image_width, image_height;
    uint32_t dircount = 0;
    {
      TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &image_width);
      TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &image_height);
      do {
        ++dircount;
      } while (TIFFReadDirectory(tiff));
    }
    TIFFClose(tiff);

    return GridCoord(image_width, image_height, dircount);
  };

  auto read_tiff_file = [](auto fn, auto z, auto dense, bool is_multi = false) {
    auto tiff = open_tiff_file(fn);

    // advance to correct directory (z)
    if (is_multi) {
      rng::for_each(rv::indices(z), [tiff](auto i) { TIFFReadDirectory(tiff); });
      auto dir = TIFFCurrentDirectory(tiff);
      if (TIFFCurrentDirectory(tiff) != z) {
        throw std::runtime_error("Directory mismatch at: " + std::to_string(dir) +
            " requested: " + std::to_string(z));
      }
    }

    short bits_per_sample;
    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);

    auto dims = dense->bbox().dim();

    // if (dims[0] != dense->bbox.dim()[0] ||
    // dims[1] != dense->bbox.dim()[1]) {
    // std::ostringstream os;
    // os << "mismatch among tif file contents in width or height\nexpected: "
    //<< dense->bbox.dim() << "\ngot " << dims[0] << " by " << dims[1]
    //<< '\n';
    // throw std::runtime_error(os.str());
    //}

    unsigned int samples_per_pixel = 1; // grayscale=1 ; RGB=3
    size_t bytes_per_pixel = (size_t)bits_per_sample / 8 * samples_per_pixel;
    size_t n_image_bytes = (size_t)bytes_per_pixel * dims[1] * dims[0];

    auto data_ptr = dense->data();
    uint64_t z_offset = static_cast<uint64_t>(z) * dense->bbox().dim()[0] *
      dense->bbox().dim()[1];
    tstrip_t n_strips = TIFFNumberOfStrips(tiff);
    tsize_t strip_size_bytes = TIFFStripSize(tiff);
    int64_t subimg_sample_offset = 0, strip_sample_offset;

    uint64_t strip_offset = 0;
    for (tstrip_t strip_idx = 0; (tstrip_t)strip_idx < n_strips; ++strip_idx) {
      // decode and place 1 strip
      auto bytes_read = TIFFReadEncodedStrip(
          tiff, strip_idx, &data_ptr[z_offset + strip_offset], strip_size_bytes);
      strip_offset += (strip_size_bytes / (bits_per_sample / 8));
    }
    TIFFClose(tiff);
  };

  template <typename image_t = uint16_t>
    auto read_tiff_planes = [](const std::vector<std::string> &fns,
        const CoordBBox &bbox) {
      auto dense =
        std::make_unique<vto::Dense<image_t, vto::LayoutXYZ>>(bbox, /*fill*/ 0.);

      rng::for_each(fns | rv::enumerate, [densep = dense.get()](auto fn_z) {
          auto [z, fn] = fn_z;
          read_tiff_file(fn, z, densep);
          });

      return dense;
    };

  // std::unique_ptr<vto::Dense<image_width, vto::LayoutXYZ>>
  std::unique_ptr<vto::Dense<uint8_t, vto::LayoutXYZ>>
    read_tiff_paged(const std::string &fn) {
      auto dims = read_tiff_dims(fn);
      auto dense = std::make_unique<vto::Dense<uint8_t, vto::LayoutXYZ>>(
          CoordBBox(zeros(), dims), /*fill*/ 0.);
      rng::for_each(rv::indices(dims.z()), [densep = dense.get(), fn](auto z) {
          read_tiff_file(fn, z, densep, true);
          });
      return dense;

      // auto bits_per_sample = read_tiff_bit_width(fn);
      // if (bits_per_sample == 8) {
      // auto dense = std::make_unique<vto::Dense<uint8_t, vto::LayoutXYZ>>(
      // CoordBBox(zeros(), dims), [>fill<] 0.);
      // read_tiff_file(fn, 0, dense.get());
      // return dense;
      //} else if (bits_per_sample == 16) {
      // auto dense = std::make_unique<vto::Dense<uint16_t, vto::LayoutXYZ>>(
      // CoordBBox(zeros(), dims), [>fill<] 0.);
      // read_tiff_file(fn, 0, dense.get());
      // return dense;
      //}

      // throw std::runtime_error(
      //"Recut only supports unsigned (grayscale) 8 or 16-bit tiffs");
      // return nullptr;
    }

  auto get_dir_files = [](const fs::path &dir, const std::string &ext) {
    std::ostringstream os;
    os << "Passed : " << dir;
    if (!(fs::exists(dir) && fs::is_directory(dir))) {
      os << " get_dir_files() must be passed a path to an existing directory";
      throw std::runtime_error(os.str());
    }

    // not a safe range, so convert to object before passing to range-v3
    auto iter = fs::directory_iterator(dir);
    auto fn_pairs =
      iter |
      rv::filter([](auto const &entry) { return fs::is_regular_file(entry); }) |
      rv::filter([&ext](auto const &entry) {
          return entry.path().extension() == ext;
          }) |
    rv::transform([&dir](auto const &entry) {
        auto fn = (dir / entry.path().filename()).string();
        auto tokens = fn | rv::split('_') | rng::to<std::vector<std::string>>();
        if (tokens.empty()) {
        throw std::runtime_error("input images must be have their z-plane "
            "specified after _ like img_000000.tif");
        }
        auto str_index = tokens.back();

        // remove non digit characters
        auto clean_index = str_index |
        rv::filter([](char c) { return isdigit(c); }) |
        rng::to<std::string>();

        int index = std::stoi(clean_index);
        return std::make_pair(index, fn);
        }) |
    rv::filter([](auto const &entry) { return fs::exists(entry.second); }) |
      rng::to_vector;

    if (fn_pairs.empty()) {
      os << " directory must contain at least one file with extension " << ext;
      throw std::runtime_error(os.str());
    }

    std::sort(fn_pairs.begin(), fn_pairs.end());

    auto tif_filenames = fn_pairs |
      rv::transform([](auto fpair) { return fpair.second; }) |
      rng::to_vector;

    return tif_filenames;
  };

  auto get_tif_bit_width = [](const std::string &tif_dir) {
    const auto tif_filenames = get_dir_files(tif_dir, ".tif");
    // take the first file, conver to char*
    TIFF *tiff = TIFFOpen(&tif_filenames[0][0], "r");
    if (!tiff) {
      throw std::runtime_error(
          "ERROR reading (not existent, not accessible or no TIFF file)");
    }
    short bits_per_sample, samples_per_pixel;
    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFClose(tiff);
    return bits_per_sample;
  };

  auto get_tif_dims = [](const std::vector<std::string> &tif_filenames) {
    // take the first file, conver to char*
    TIFF *tiff = TIFFOpen(&tif_filenames[0][0], "r");
    if (!tiff) {
      throw std::runtime_error(
          "ERROR reading (not existent, not accessible or no TIFF file)");
    }
    uint32_t image_width, image_height;
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &image_width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &image_height);
    TIFFClose(tiff);
    return GridCoord(image_width, image_height, tif_filenames.size());
  };

  template <typename image_t = uint16_t>
    auto read_tiff_dir(const std::string &dir) {
      const auto tif_filenames = get_dir_files(dir, ".tif");
      const auto dims = get_tif_dims(tif_filenames);
      // bbox is inclusive
      const auto bbox = CoordBBox(zeros(), dims.offsetBy(-1));
      assertm(dims == bbox.dim(), "dims and bbox dims must match");
      return read_tiff_planes<image_t>(tif_filenames, bbox);
    }

  // this is not thread safe if concurrent threads
  // add or subtract elements from vector.
  // added such that vectors of atomics can
  // be created more cleanly while retaining atomicity
  template <typename T> struct atomwrapper {
    atomic<T> _a;

    T load() { return _a.load(); }

    void store(T val, memory_order order = memory_order_release) {
      _a.store(val, order);
    }

    bool compare_exchange_strong(bool val, bool val2) {
      return _a.compare_exchange_strong(val, val2);
    }

    atomwrapper() : _a() {}

    atomwrapper(const atomic<T> &a) : _a(a.load()) {}

    atomwrapper(const atomwrapper &other) : _a(other._a.load()) {}

    atomwrapper &operator=(const atomwrapper &other) {
      _a.store(other._a.load());
      return *this;
    }
  };

  template <typename T> struct CompareResults {
    T false_negatives;
    T false_positives;
    VID_t duplicate_count;
    VID_t match_count;

    CompareResults(T false_negatives, T false_positives, VID_t duplicate_count,
        VID_t match_count)
      : false_negatives(false_negatives), false_positives(false_positives),
      duplicate_count(duplicate_count), match_count(match_count) {}
  };

  auto get_vids = [](auto tree, auto lengths) {
    return tree |
      rv::transform([&](auto v) { return v->vid(lengths[0], lengths[1]); }) |
      rng::to_vector;
  };

  auto get_vids_sorted = [](auto tree, auto lengths) {
    return get_vids(tree, lengths) | rng::action::sort;
  };

  // prints mismatches between two trees in uid sorted order no assertions
  auto compare_tree = [](auto truth_tree, auto check_tree, auto lengths) {
    bool print = false;

    if (print)
      std::cout << "compare tree\n";
    // duplicate_count will be asserted to == 0 at caller
    VID_t duplicate_count = 0;
    duplicate_count += truth_tree.size() - unique_count(truth_tree);
    duplicate_count += check_tree.size() - unique_count(check_tree);

    auto truth_vids = get_vids_sorted(truth_tree, lengths);
    auto check_vids = get_vids_sorted(check_tree, lengths);

    // std::cout << "truth_vids\n";
    // print_iter(truth_vids);
    // std::cout << "check_vids\n";
    // print_iter(check_vids);

    auto matches = rv::set_intersection(truth_vids, check_vids);

    VID_t match_count = rng::distance(matches);
    if (print)
      std::cout << "match count: " << match_count << '\n';

    auto get_negatives = [&](auto &tree, auto &matches, auto specifier) {
      return rv::set_difference(tree, matches) | rv::transform([&](auto vid) {
          if (print)
          std::cout << "false " << specifier
          << " at: " << coord_to_str(id_to_coord(vid, lengths))
          << '\n';
          return std::make_pair(vid, std::string(specifier));
          }) |
      rng::to_vector;
    };

    auto result = new CompareResults<std::vector<std::pair<VID_t, std::string>>>(
        get_negatives(truth_vids, matches, "negative"),
        get_negatives(check_vids, matches, "positive"), duplicate_count,
        match_count);
    return result;
  };

  /* returns available memory to system in bytes
  */
  inline size_t GetAvailMem() {
#if defined(_SC_AVPHYS_PAGES)
    map<string, size_t> mem_info;
    std::ifstream read_mem("/proc/meminfo");
    string read_buf, last_buf;
    for (; !read_mem.eof();) {
      read_mem >> read_buf;
      if (read_buf[read_buf.size() - 1] == ':') {
        last_buf = read_buf.substr(0, read_buf.size() - 1);
      } else {
        if (!last_buf.empty()) {
          mem_info[last_buf] = atoll(read_buf.c_str());
        }
        last_buf.clear();
      }
    }
    return (mem_info["MemFree"] + mem_info["Cached"]) * 1024;
#else
    return 0L;
#endif
  }

  // avoid a dependency on boost.accumulators
  // return mean of iterable and sample standard deviation
  template <typename T> std::tuple<double, double, double> iter_stats(T v) {

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double stdev;

    if (v.size() == 1) {
      stdev = 0.;
    } else {
      std::vector<double> diff(v.size());
      std::transform(v.begin(), v.end(), diff.begin(),
          [mean](double x) { return x - mean; });
      double sq_sum =
        std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

      // the mean is calculated from this data set the population mean
      // is unknown, therefore use the sample stdev (n - 1)
      stdev = std::sqrt(sq_sum / (v.size() - 1));
    }

    return {mean, sum, stdev};
  }

  auto get_img_vid = [](const VID_t i, const VID_t j, const VID_t k,
      const VID_t image_length_x,
      const VID_t image_length_y) -> VID_t {
    auto image_length_xy = image_length_x * image_length_y;
    return k * image_length_xy + j * image_length_x + i;
  };

  // see DILATION_FACTOR definition or accumulate_prune() implementation
  auto create_coverage_mask_accurate = [](std::vector<MyMarker *> &markers,
      auto mask, auto lengths) {
    auto sz0 = lengths[0];
    auto sz1 = lengths[1];
    auto sz2 = lengths[2];
    assertm(sz0 == sz1, "must matching sizes for now");
    assertm(sz1 == sz2, "must matching sizes for now");
    auto sz01 = static_cast<VID_t>(sz0 * sz1);
    auto tol_sz = sz01 * sz2;
    VID_t count_selected_pixels = 0;
    for (VID_t index = 0; index < tol_sz; ++index) {
      mask[index] = 0;
      // check all marker to see if their radius covers it
      for (const auto &marker : markers) {
        assertm(marker->radius > 0, "Markers must have a radius > 0");
        if (is_covered_by_parent(index, marker->vid(sz0, sz1),
              marker->radius - (DILATION_FACTOR - 1), sz0)) {
          mask[index] = 1;
          count_selected_pixels++;
          break;
        }
      }
    }
  };

  auto create_coverage_mask = [](std::vector<MyMarker *> &markers, auto mask,
      auto lengths) {
    auto sz0 = lengths[0];
    auto sz1 = lengths[1];
    auto sz2 = lengths[2];
    auto sz01 = static_cast<VID_t>(sz0 * sz1);
    for (const auto &marker : markers) {
      int32_t r = marker->radius;
      assertm(marker->radius > 0, "Markers must have a radius > 0");
      r -= (DILATION_FACTOR - 1);
      auto x = static_cast<int32_t>(marker->x);
      auto y = static_cast<int32_t>(marker->y);
      auto z = static_cast<int32_t>(marker->z);
      for (int32_t kk = -r; kk <= r; kk++) {
        int32_t z2 = z + kk;
        if (z2 < 0 || z2 >= sz2)
          continue;
        for (int32_t jj = -r; jj <= r; jj++) {
          int32_t y2 = y + jj;
          if (y2 < 0 || y2 >= sz1)
            continue;
          for (int32_t ii = -r; ii <= r; ii++) {
            int32_t x2 = x + ii;
            if (x2 < 0 || x2 >= sz0)
              continue;
            auto dst = abs(ii) + abs(jj) + abs(kk);
            if (dst > r)
              continue;
            int32_t ind = z2 * sz01 + y2 * sz0 + x2;
            // if (mask[ind] > 0) {
            // std::cout << "Warning: marker " << marker->description(sz0, sz1) <<
            //" is over covering at pixel " << x2 << " " << y2 << " " << z2 <<
            //'\n';
            //}
            mask[ind]++;
          }
        }
      }
    }
  };

  template <typename T, typename T2, typename T3>
    auto check_coverage(const T mask, const T2 inimg1d, const VID_t tol_sz,
        T3 bkg_thresh) {
      VID_t match_count = 0;
      VID_t over_coverage = 0;
      std::vector<VID_t> false_negatives;
      std::vector<VID_t> false_positives;

      for (VID_t i = 0; i < tol_sz; i++) {
        auto check = mask[i];
        assertm(check <= 1,
            "this function only works on binarized mask as first input");
        // ground represents the original pixel value wheras
        // check merely indicates how many times a pixel was
        // covered in a pruning pass
        auto ground = inimg1d[i];

        if (ground > bkg_thresh) {
          if (check) {
            match_count++;
            // over_coverage is a measure of redundancy in the
            // pruning method, it signifies that multiple
            // markers covered the same pixel more than once
            over_coverage += check - 1;
          } else {
            false_negatives.push_back(i);
          }
        } else {
          if (check) {
            false_positives.push_back(i);
            // keep over_coverage independent of false_positive measure
            // by subtracting 1 still
            over_coverage += check - 1;
          }
        }
      }

      // duplicate count gets set to over coverage
      return new CompareResults<std::vector<VID_t>>(
          false_negatives, false_positives, over_coverage, match_count);
    }

  auto covered_by_bboxs = [](const auto coord, const auto bboxs) {
    for (const auto bbox : bboxs) {
      if (bbox.isInside(coord))
        return true;
    }
    return false;
  };

  auto find_or_assign = [](std::array<double, 3> swc_coord,
      auto &coord_to_swc_id) -> uint32_t {
    auto val = coord_to_swc_id.find(swc_coord);
    if (val == coord_to_swc_id.end()) {
      auto new_val = coord_to_swc_id.size();
      if (new_val >= std::numeric_limits<int32_t>::max())
        throw std::runtime_error("Total swc line count overflows 32-bit "
            "integer used in some swc programs");
      coord_to_swc_id[swc_coord] = new_val;
      assertm(new_val == (coord_to_swc_id.size() - 1),
          "map must now be 1 size larger");
      return new_val;
    }
    return coord_to_swc_id[swc_coord];
  };

  // throws if any leaf does not have monotonically increasing offsets or
  // within bounds of attribute arrays
  auto validate_grid = [](EnlargedPointDataGrid::Ptr grid) {
    // TODO use leafManager and tbb::parallel_for
    for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
      leaf_iter->validateOffsets();
    }
  };

  // return sorted origins of all active leafs
  template <typename GridTypePtr>
    std::vector<GridCoord> get_origins(GridTypePtr grid) {
      std::vector<GridCoord> origins;
      // TODO use leafManager and tbb::parallel_for
      for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
        origins.push_back(leaf_iter->origin());
      }
      std::sort(origins.begin(), origins.end());
      return origins;
    }

  template <typename GridTypePtr>
    bool leaves_intersect(GridTypePtr grid, GridTypePtr other) {
      std::vector<GridCoord> out;
      // inputs must be sorted
      auto origins = get_origins(grid);
      auto other_origins = get_origins(other);
      std::set_intersection(origins.begin(), origins.end(), other_origins.begin(),
          other_origins.end(), std::back_inserter(out));
      return !out.empty();
    }

  auto read_vdb_float = [](std::string fn) {
    auto base_grid = read_vdb_file(fn);
    auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
#ifdef LOG_FULL
    print_grid_metadata(float_grid);
#endif
    return float_grid;
  };

  auto combine_grids = [](std::string lhs, std::string rhs, std::string out) {
    auto first_grid = read_vdb_float(lhs);
    {
      auto second_grid = read_vdb_float(rhs);

      // arithmetic sums in-place to first_grid and empties second_grid
      vb::tools::compSum(*first_grid, *second_grid);

#ifdef LOG_FULL
      print_grid_metadata(first_grid);
#endif
    }

    openvdb::GridPtrVec grids;
    grids.push_back(first_grid);
    write_vdb_file(grids, out);
  };

  auto upsample_idx = [](int original_idx, int upsample_factor) -> int {
    /* scale the z, */
    return upsample_factor * original_idx;
    // return upsample_factor * original_idx +
    /* then offset it into the center of the upsample*/
    //(upsample_factor / 2);
  };

  /* typically called with the topology_grid as the point_grid, and a
   * connected component as the float grid
   */
  auto collect_all_points = [](EnlargedPointDataGrid::Ptr point_grid,
      openvdb::FloatGrid::Ptr float_grid) {
    auto spheres = std::vector<openvdb::Vec4s>();
    // define local fn to add a sphere for a coord that is valid in
    // topology_grid Note this can be accelerated by going in leaf order
    auto emplace_coord = [&point_grid, &spheres](auto coord) {
      auto leaf_iter = point_grid->tree().probeLeaf(coord);
      if (leaf_iter) {
        // assertm(leaf_iter, "leaf must be on, since the float_grid is derived
        // from the " "active topology of it");

        openvdb::points::AttributeWriteHandle<float> radius_handle(
            leaf_iter->attributeArray("pscale"));

        auto ind = leaf_iter->beginIndexVoxel(coord);
        // assertm(ind, "ind must be on, since the float_grid is derived from the
        // "
        //"active topology of it");

        if (ind) {
          auto radius = radius_handle.get(*ind);
          spheres.emplace_back(coord[0], coord[1], coord[2], radius);
        }
      }
    };

    auto timer = high_resolution_timer();
    // construct spheres from underlying topology of the float_grid
    // get on coords of current the float_grid
    for (openvdb::FloatGrid::ValueOnCIter iter = float_grid->cbeginValueOn();
        iter.test(); ++iter) {

      if (iter.isVoxelValue()) {
        emplace_coord(iter.getCoord());
      } else {

        openvdb::CoordBBox bbox;
        iter.getBoundingBox(bbox);

        for (auto bbox_iter = bbox.begin(); bbox_iter; ++bbox_iter) {
          // only adds if topology grid leaf and ind are also on
          emplace_coord(*bbox_iter);
        }
      }
    }
#ifdef LOG
    cout << "Collect float grid points in " << timer.elapsed() << '\n';
#endif
    return spheres;
  };

  // modifies the contents of the passed marker vector nX
  // ensuring that linkings are bidirectional
  // that there are no self-links or repeats in the
  // neighbor list
  void check_nbr(vector<MyMarker *> &nX) {

    for (VID_t i = 0; i < nX.size(); ++i) {
      // remove repeats
      sort(nX[i]->nbr.begin(), nX[i]->nbr.end());
      nX[i]->nbr.erase(unique(nX[i]->nbr.begin(), nX[i]->nbr.end()),
          nX[i]->nbr.end());

      // remove self linkages
      long int pos =
        find(nX[i]->nbr.begin(), nX[i]->nbr.end(), i) - nX[i]->nbr.begin();
      if (pos >= 0 && pos < nX[i]->nbr.size())
        nX[i]->nbr.erase(nX[i]->nbr.begin() + pos); // remove at pos
    }

    //  ensure linkings are bidirectional, add if not
    //  for all markers
    for (VID_t i = 0; i < nX.size(); ++i) {
      // for all neighbors of this marker
      for (VID_t j = 0; j < nX[i]->nbr.size(); ++j) {
        if (i != j) {
          bool fnd = false;
          for (VID_t k = 0; k < nX[nX[i]->nbr[j]]->nbr.size(); ++k) {
            if (nX[nX[i]->nbr[j]]->nbr[k] == i) {
              fnd = true;
              break;
            }
          }

          if (!fnd) {
            // enforce link
            nX[nX[i]->nbr[j]]->nbr.push_back(i);
          }
        }
      }
    }
  };

  auto coord_dist = [](const GridCoord &a, const GridCoord &b) -> float {
    std::array<float, 3> diff = {
      static_cast<float>(a[0]) - static_cast<float>(b[0]),
      static_cast<float>(a[1]) - static_cast<float>(b[1]),
      static_cast<float>(a[2]) - static_cast<float>(b[2])};
    return std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
  };

  // accept_tombstone is a way to see pruned vertices still in active_vertex
  std::vector<MyMarker *> convert_to_markers(EnlargedPointDataGrid::Ptr grid,
      bool accept_tombstone) {

    std::vector<MyMarker *> outtree;
#ifdef FULL_PRINT
    cout << "Generating results." << '\n';
#endif
    auto timer = high_resolution_timer();

    // get a mapping to stable address pointers in outtree such that a markers
    // parent is valid pointer when returning just outtree
    std::unordered_map<GridCoord, MyMarker *> coord_to_marker_ptr;
    std::unordered_map<GridCoord, VID_t> coord_to_idx;

    // iterate all active vertices ahead of time so each marker
    // can have a pointer to it's parent marker
    for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
      // cout << leaf_iter->getNodeBoundingBox() << '\n';

      openvdb::points::AttributeHandle<uint8_t> flags_handle(
          leaf_iter->constAttributeArray("flags"));

      openvdb::points::AttributeHandle<float> radius_handle(
          leaf_iter->constAttributeArray("pscale"));

      openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
          leaf_iter->constAttributeArray("parents"));

      // openvdb::points::AttributeHandle<OffsetCoord> position_handle(
      // leaf_iter->constAttributeArray("P"));

      for (auto ind = leaf_iter->beginIndexOn(); ind; ++ind) {
        // get coord
        auto coord = ind.getCoord();
        // create all valid new marker objects
        if (is_valid(flags_handle, ind, accept_tombstone)) {
          // std::cout << "\t " << coord<< '\n';
          assertm(coord_to_marker_ptr.count(coord) == 0,
              "Can't have two matching vids");
          // get original i, j, k
          auto marker = new MyMarker(coord[0], coord[1], coord[2]);
          if (is_root(flags_handle, ind)) {
            // a marker with a type of 0, must be a root
            marker->type = 0;
          }
          marker->radius = radius_handle.get(*ind);
          // save this marker ptr to a map
          coord_to_marker_ptr[coord] = marker;
          coord_to_idx[coord] = outtree.size();
          // std::cout << "\t " << coord_to_str(coord) << " -> "
          //<< (coord + parents_handle.get(*ind)) << " " << marker->radius
          //<< '\n';
          assertm(marker->radius, "can't have 0 radius");
          outtree.push_back(marker);
        }
      }
    }

    // now that a pointer to all desired markers is known
    // iterate and complete the marker definition
    for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {

      openvdb::points::AttributeHandle<uint8_t> flags_handle(
          leaf_iter->constAttributeArray("flags"));

      openvdb::points::AttributeHandle<float> radius_handle(
          leaf_iter->constAttributeArray("pscale"));

      openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
          leaf_iter->constAttributeArray("parents"));

      for (auto ind = leaf_iter->beginIndexOn(); ind; ++ind) {
        // create all valid new marker objects
        if (is_valid(flags_handle, ind, accept_tombstone)) {
          // get coord
          auto coord = ind.getCoord();
          assertm(coord_to_marker_ptr.count(coord),
              "did not find vertex in marker map");
          auto marker = coord_to_marker_ptr[coord]; // get the ptr
          if (is_root(flags_handle, ind)) {
            // a marker with a parent of 0, must be a root
            marker->parent = 0;
          } else {
            auto parent_coord = parents_handle.get(*ind) + coord;

            // find parent
            assertm(coord_to_marker_ptr.count(parent_coord),
                "did not find parent in marker map");

            auto parent =
              coord_to_marker_ptr[parent_coord]; // adjust marker->parent =
                                                 // parent;
            marker->parent = parent;
            marker->nbr.push_back(coord_to_idx[parent_coord]);
          }
        }
      }
    }

#ifdef LOG
    cout << "Total marker size: " << outtree.size() << " nodes" << '\n';
#endif

#ifdef FULL_PRINT
    cout << "Finished generating results within " << timer.elapsed() << " sec."
      << '\n';
#endif

    return outtree;
  }

  // modify the radius value within the point grid to reflect a predetermined
  // radius mutates the value that was previously calculated usually somas have a
  // more accurate radius value determined before Recut processes so it is
  // appropriate to rewrite with these more accurate radii values before a prune
  // step
  auto adjust_soma_radii =
    [](const std::vector<Seed> &seeds,
        EnlargedPointDataGrid::Ptr grid) -> EnlargedPointDataGrid::Ptr {
      assertm(seeds.size() > 0, "passed seeds is empty");
      rng::for_each(seeds, [grid](const auto &seed) {
          const auto leaf = grid->tree().probeLeaf(seed.coord);

          // sanity checks
          assertm(leaf, "corresponding leaf of passed seed must be active");
          auto ind = leaf->beginIndexVoxel(seed.coord);
          assertm(ind, "corresponding voxel of passed seed must be active");
          assertm(seed.radius, "passed radii value of 0 is invalid");

          // modify the radius value
          openvdb::points::AttributeWriteHandle<float> radius_handle(
              leaf->attributeArray("pscale"));

          auto previous_radius = radius_handle.get(*ind);
          radius_handle.set(*ind, seed.radius);

#ifdef FULL_PRINT
          cout << "Adjusted " << seed.coord << " radius " << previous_radius << " -> "
          << radius_handle.get(*ind) << '\n';
#endif
          });

      return grid;
    };

  template <typename FilterP, typename Pred>
    void visit_float(openvdb::FloatGrid::Ptr float_grid,
        EnlargedPointDataGrid::Ptr point_grid, FilterP keep_if,
        Pred predicate) {
      for (auto float_leaf = float_grid->tree().beginLeaf(); float_leaf;
          ++float_leaf) {

        auto point_leaf = point_grid->tree().probeLeaf(float_leaf->origin());
        assertm(point_leaf, "leaf must be on, since component is derived from the "
            "active topology of it");

        // note: some attributes need mutability
        openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
            point_leaf->attributeArray("flags"));

        openvdb::points::AttributeHandle<float> radius_handle(
            point_leaf->constAttributeArray("pscale"));

        // Extract the position attribute from the leaf by name (P is position).
        const openvdb::points::AttributeArray &arr =
          point_leaf->constAttributeArray("P");
        // Create a read-only AttributeHandle. Position always uses Vec3f.
        openvdb::points::AttributeHandle<PositionT> position_handle(arr);

        openvdb::points::AttributeWriteHandle<OffsetCoord> parents_handle(
            point_leaf->attributeArray("parents"));

        for (auto float_ind = float_leaf->beginValueOn(); float_ind; ++float_ind) {
          const auto coord = float_ind.getCoord();
          auto ind = point_leaf->beginIndexVoxel(coord);
          if (keep_if(coord, float_leaf)) {
            predicate(flags_handle, parents_handle, radius_handle, ind, coord);
          }
        }
      }
    }

  template <typename FilterP, typename Pred>
    void visit(EnlargedPointDataGrid::Ptr grid, FilterP keep_if, Pred predicate) {
      for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {

        // note: some attributes need mutability
        openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
            leaf_iter->attributeArray("flags"));

        openvdb::points::AttributeHandle<float> radius_handle(
            leaf_iter->constAttributeArray("pscale"));

        openvdb::points::AttributeWriteHandle<OffsetCoord> parents_handle(
            leaf_iter->attributeArray("parents"));

        for (auto ind = leaf_iter->beginIndexOn(); ind; ++ind) {
          if (keep_if(flags_handle, parents_handle, radius_handle, ind)) {
            predicate(flags_handle, parents_handle, radius_handle, ind, leaf_iter);
          }
        }
      }
    }

  vector<PositionT>
    convert_float_to_positions(openvdb::FloatGrid::Ptr float_grid) {
      std::vector<PositionT> positions;

      for (auto iter = float_grid->cbeginValueOn(); iter; ++iter) {
        auto coord = iter.getCoord();
        positions.emplace_back(coord[0], coord[1], coord[2]);
      }
      return positions;
    };

  EnlargedPointDataGrid::Ptr
    convert_float_to_point(openvdb::FloatGrid::Ptr float_grid) {
      auto [lengths, requested_fg_pct] = get_metadata(float_grid);
      auto positions = convert_float_to_positions(float_grid);
      auto point_grid =
        create_point_grid(positions, lengths, get_transform(), requested_fg_pct);
      return point_grid;
    };

  std::pair<vector<MyMarker *>, std::unordered_map<GridCoord, VID_t>>
    convert_float_to_markers(openvdb::FloatGrid::Ptr component,
        EnlargedPointDataGrid::Ptr point_grid,
        uint16_t prune_radius_factor) {
#ifdef FULL_PRINT
      std::cout << "Convert\n";
#endif

      auto timer = high_resolution_timer();
      std::vector<MyMarker *> outtree;

      // get a mapping to stable address pointers in outtree such that a markers
      // parent is valid pointer when returning just outtree
      std::unordered_map<GridCoord, MyMarker *> coord_to_marker_ptr;

      auto keep_if = [](const auto coord, const auto float_leaf) {
        return float_leaf->isValueOn(coord);
      };

      auto establish_marker_set = [&coord_to_marker_ptr, &outtree,
           prune_radius_factor](const auto &flags_handle,
               auto &parents_handle,
               const auto &radius_handle,
               const auto &ind,
               const auto &coord) {
             assertm(coord_to_marker_ptr.count(coord) == 0,
                 "Can't have two matching vids");
             // get original i, j, k
             auto marker = new MyMarker(coord[0], coord[1], coord[2]);
             assertm(marker != nullptr, "is a nullptr");

             marker->radius = radius_handle.get(*ind);
             if (marker->radius == 0) {
               throw std::runtime_error("Note: active vertex can not have radius 0\n");
             }
             if (is_root(flags_handle, ind)) {
               // a marker with a type of 0, must be a root
               marker->type = 0;
               // increase reported radius slightly to remove nodes on edge
               // and decrease proofreading efforts
               // otherwise any concavities outside of a sphere get marked as
               // branches which can heavily impact morphological metrics round to
               // nearest int
               marker->radius = static_cast<uint16_t>(
                   (static_cast<double>(marker->radius) * FORCE_SOMA_DILATION) + .5);
             } else {
               // upsample by factor to account for anisotropic images
               // neurites may appear thinner due to anisotropic imaging
               // this mitigates this effect
               // however, dilation trick like below causes large nodules probably
               // because previously inflated get maxed and creates compounding cycle
               // round to nearest int
               marker->radius =
                 static_cast<uint16_t>((static_cast<double>(marker->radius) *
                       static_cast<double>(prune_radius_factor)) +
                     .5);
             }
             if (marker->radius == 0) {
               throw std::runtime_error("Note: active marker can not have radius 0\n");
             }

             // save this marker ptr to a map
             coord_to_marker_ptr.emplace(coord, marker);

             outtree.push_back(marker);
           };

      // iterate all active vertices ahead of time so each marker
      // can have a pointer to its parent marker.
      // iterate by leaf markers since attributes are stored in chunks of leaf size.
      visit_float(component, point_grid, keep_if, establish_marker_set);

      // sorting improves pruning by favoring higher relevance/radii
      // sort by markers by decreasing radii (~relevance)
      std::sort(outtree.begin(), outtree.end(),
          [](const MyMarker *l, const MyMarker *r) {
          return l->radius > r->radius;
          });

      // place all somas (type 0 first) while preserving large radii precedence
      std::stable_partition(outtree.begin(), outtree.end(),
          [](const MyMarker *l) { return l->type == 0; });

      // for advantra prune method and assigning correct nbr index
      std::unordered_map<GridCoord, VID_t> coord_to_idx;

      // establish coord to idx
      rng::for_each(outtree | rv::enumerate, [&coord_to_idx](auto markerp) {
          auto [i, marker] = markerp;
          auto coord = GridCoord(marker->x, marker->y, marker->z);
          coord_to_idx[coord] = i;
          });

      // now that a pointer to all desired markers is known
      // iterate and complete the marker definition
      auto assign_parent = [&coord_to_marker_ptr, &coord_to_idx, &outtree](
          const auto &flags_handle, auto &parents_handle,
          const auto &radius_handle, const auto &ind,
          const auto &coord) {
        auto marker = coord_to_marker_ptr[coord]; // get the ptr
        if (marker == nullptr) {
          cout << "could not find " << coord << '\n';
          auto idx = coord_to_idx.at(coord);
          cout << idx << '\n';
          throw std::runtime_error("marker not valid");
        }

        if (is_root(flags_handle, ind)) {
          // a marker with a parent of 0, must be a root
          marker->parent = 0;
        } else {
          auto parent_coord = parents_handle.get(*ind) + coord;

          if (coord_to_marker_ptr.count(parent_coord) < 1) {
            throw std::runtime_error(
                "did not find parent in marker map during assign");
          }

          // find parent
          auto parent = coord_to_marker_ptr[parent_coord]; // adjust
          marker->parent = parent;

          marker->nbr.push_back(coord_to_idx[parent_coord]);
        }
      };

      visit_float(component, point_grid, keep_if, assign_parent);

#ifdef FULL_PRINT
      cout << "Finished generating results within " << timer.elapsed() << " sec."
        << '\n';
#endif
      return {outtree, coord_to_idx};
    }

  auto all_invalid = [](const auto &flags_handle, const auto &parents_handle,
      const auto &radius_handle, const auto &ind) {
    return !is_selected(flags_handle, ind);
  };

  auto convert_vdb_to_dense = [](auto grid) {
    // inclusive of both ends of bounding box
    vto::Dense<uint16_t, vto::LayoutXYZ> dense(grid->evalActiveVoxelBoundingBox(),
        /*fill*/ 0.);
    vto::copyToDense(*grid, dense);
    return dense;
  };

  // Write a multi-page (pyramidal) tiff file using the libtiff library
  // join conversion and writing by z plane for performance, note that for large
  // components, create a full dense buffer will fault with bad_alloc due to size
  // z-plane by z-plane helps prevents this
  template <typename GridT>
    std::string write_vdb_to_tiff_page(GridT grid, std::string base,
        CoordBBox bbox = {}) {

      if (bbox.empty())
        bbox = grid->evalActiveVoxelBoundingBox(); // inclusive both ends

      auto fn = base + ".tif";
      TIFF *tiff = TIFFOpen(fn.c_str(), "w");
      if (!tiff) {
        throw std::runtime_error(
            "ERROR reading (not existent, not accessible or no TIFF file)");
      }
      auto minz = bbox.min()[2];
      auto maxz = bbox.max()[2];

      // inclusive range with index
      auto zrng = rng::closed_iota_view(minz, maxz) | rv::enumerate;

      // output each plane to separate page within the same file
      rng::for_each(zrng, [grid, tiff, &bbox](const auto zpair) {
          auto [page_number, z] = zpair;
          auto min = GridCoord(bbox.min()[0], bbox.min()[1], z);
          auto max = GridCoord(bbox.max()[0], bbox.max()[1], z); // inclusive
          auto plane_bbox = CoordBBox(min, max);

          // inclusive of both ends of bounding box
          vto::Dense<uint8_t, vto::LayoutXYZ> dense(plane_bbox, /*fill*/ 0.);
          vto::copyToDense(*grid, dense);

          write_tiff_page(dense.data(), tiff, plane_bbox.dim(), page_number);
          });

      TIFFClose(tiff);
      return fn;
    }

  template <typename image_t>
    void encoded_tiff_write(image_t *inimg1d, TIFF *tiff, const GridCoord dims) {
      unsigned int samples_per_pixel = 1; // grayscale=1 ; RGB=3
      unsigned int bits_per_sample = 8 * sizeof(image_t);
      TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
      TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
      TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
      TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, dims[0]);
      TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, dims[1]);
      TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
      TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
      TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, dims[1]);
      TIFFSetField(tiff, TIFFTAG_XRESOLUTION, 1);
      TIFFSetField(tiff, TIFFTAG_YRESOLUTION, 1);
      TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE);
      TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE);
      TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

      auto length_of_strip_in_bytes = dims[0] * dims[1] * sizeof(image_t);
      auto err_code =
        TIFFWriteEncodedStrip(tiff, 0, inimg1d, length_of_strip_in_bytes);
      if (err_code == -1) {
        throw std::runtime_error("ERROR write encoded strip of TIFF file)");
      }
    }

  // join conversion and writing by z plane for performance, note that for large
  // components, create a full dense buffer will fault with bad_alloc due to size
  // z-plane by z-plane like below prevents this
  template <typename GridT>
    std::string write_vdb_to_tiff_planes(GridT grid, fs::path base,
        CoordBBox bbox = {}, int channel = 0,
        int component_index = 0) {
      if (bbox.empty())
        bbox = grid->evalActiveVoxelBoundingBox(); // inclusive both ends

      base /= ("ch" + std::to_string(channel));
      fs::remove_all(base); // make sure it's an overwrite
      fs::create_directories(base);

      auto zrng = rng::closed_iota_view(bbox.min()[2], bbox.max()[2]) |
        rv::enumerate; // inclusive range

      // output each plane to separate file
      rng::for_each(zrng, [&](const auto zpair) {
          auto [index, z] = zpair;
          auto min = GridCoord(bbox.min()[0], bbox.min()[1], z);
          auto max = GridCoord(bbox.max()[0], bbox.max()[1], z); // inclusive
          auto plane_bbox = CoordBBox(min, max);

          // inclusive of both ends of bounding box
          vto::Dense<uint16_t, vto::LayoutXYZ> dense(plane_bbox, /*fill*/ 0.);
          vto::copyToDense(*grid, dense);

          // overflows at 1 million z planes
          std::ostringstream fn;
          fn << "component_" << component_index << "_img_" << std::setfill('0')
          << std::setw(6) << index << ".tif";

          // cout << '\n' << fn.str() << '\n';
          // print_image_3D(dense.data(), plane_bbox.dim());

          write_single_z_plane(dense.data(), base / fn.str(), plane_bbox.dim());
          });
      return base.string();
    }

  // for all active values of the output grid copy the value at that coordinate
  // from the inputs grid this could be replaced by openvdb's provided CSG/copying
  // functions
  template <typename ValueGridT, typename OutputGridT>
    void copy_values(ValueGridT img_grid, OutputGridT output_grid) {
      auto accessor = img_grid->getConstAccessor();
      auto output_accessor = output_grid->getAccessor();
      for (auto iter = output_grid->cbeginValueOn(); iter.test(); ++iter) {
        auto val = accessor.getValue(iter.getCoord());
        output_accessor.setValue(iter.getCoord(), val);
      }
    }

  template <typename GridT>
    void add_mask_to_image_grid(ImgGrid::Ptr image_grid, GridT mask_grid) {
      // iterate mask grid
    }

  // valued_grid : holds the pixel intensity values
  // topology_grid : holds the topology of the neuron cluster in question
  // values copied in topology and written to tiff
  template <typename GridT>
    std::pair<ImgGrid::Ptr, CoordBBox>
    create_window_grid(ImgGrid::Ptr valued_grid, GridT component_grid,
        std::ofstream &component_log,
        std::array<double, 3> voxel_size,
        std::vector<Seed> component_seeds, int min_window_um,
        bool labels, float expand_window_um = 0) {

      if (!valued_grid)
        throw std::runtime_error("The first grid passed to '--output-windows' must be of type uint8 VDB");
      // if an expanded crop is requested, the actual image values outside of the
      // component bounding volume are needed therefore clip the original image
      // to a bounding volume
      auto bbox = component_grid->evalActiveVoxelBoundingBox();
      //std::cout << "  bbox " << bbox << '\n';

      rng::for_each(component_seeds, [&](const auto &seed) {
          //std::cout << "  coord " << seed.coord << '\n';
          auto extent_um = min_window_um + seed.radius_um + expand_window_um;
          rng::for_each(rv::iota(0, 3), [&](auto i) {
              auto extent_voxels = extent_um / voxel_size[i];

              // find a possible min/max in coordinate space
              auto old_min = bbox.min()[i];
              auto new_min =
              static_cast<int>(seed.coord[i] - extent_voxels);

              auto old_max = bbox.max()[i];
              auto new_max =
              static_cast<int>(seed.coord[i] + extent_voxels);

              // if the new_min or max would expand the bbox then keep it
              bbox.min()[i] = std::min(old_min, new_min);
              bbox.max()[i] = std::max(old_max, new_max);
              });
          });
      //std::cout << "  bbox " << bbox << '\n';

      if (labels) {
        // choose the first seed if there are multiple
        auto seed = component_seeds[0];

        bbox.min()[2] = seed.coord.z() - seed.radius;
        bbox.max()[2] = seed.coord.z() + seed.radius;
      }

      vb::BBoxd clipBox(bbox.min().asVec3d(), bbox.max().asVec3d());
      const auto output_grid = vto::clip(*valued_grid, clipBox);

      if (output_grid->activeVoxelCount()) {
        bbox = output_grid->evalActiveVoxelBoundingBox();
      }

      //std::cout << "  bbox " << bbox << '\n';
      // alternatively... for simply carrying values across:
      // copy_values(valued_grid, component_grid);
      // or you can use the component_grid to mask the valued_grid
      // to isolate window pixels to those covered by the component like:
      // output_grid = vto::tools::clip(output_grid, component_grid);

      return {output_grid, bbox};
    }

  // valued_grid : holds the pixel intensity values
  // topology_grid : holds the topology of the neuron cluster in question
  // values copied in topology and written z-plane by z-plane to individual tiff
  // files tiff component also saved
  template <typename GridT>
    std::string write_output_windows(GridT output_grid, fs::path dir,
        std::ofstream &runtime, int index = 0,
        bool output_vdb = false, bool paged = true,
        CoordBBox bbox = {}, int channel = 0) {

      auto base = dir / ("img-component-" + std::to_string(index) + "-ch" +
          std::to_string(channel));

      std::string output_fn;
      if (output_grid->activeVoxelCount()) {
        auto timer = high_resolution_timer();
        if (paged) // all to one file
          output_fn = write_vdb_to_tiff_page(output_grid, base, bbox);
        else
          output_fn =
            write_vdb_to_tiff_planes(output_grid, dir, bbox, channel, index);

        runtime << "Write tiff, " << timer.elapsed() << '\n';

        if (output_vdb) {
          timer.restart();
          openvdb::GridPtrVec component_grids;
          component_grids.push_back(output_grid);
          write_vdb_file(component_grids, base.string() + ".vdb");
#ifdef LOG
          // cout << "Wrote window of component to vdb in " << timer.elapsed() <<
          // " s\n";
#endif
        }
      } else {
        cout << "Warning: component " << index
          << " had an empty window for the --output-windows grid\n";
      }
      return output_fn;
    }

  auto adjust_marker = [](MyMarker *marker, GridCoord offsets) {
    marker->x += offsets[0];
    marker->y += offsets[1];
    marker->z += offsets[2];
  };

  template <typename VType>
    long quick_sort_partition(void *data, long low, long high) {
      static_assert(std::is_arithmetic<VType>(),
          "must have arithmetic element types");
      auto vtype_data = (VType *)data;
      VType pivot = vtype_data[low + (high - low) / 2];
      while (true) {
        while (vtype_data[low] < pivot)
          ++low;
        while (vtype_data[high] > pivot)
          --high;
        if (low >= high)
          return high;
        std::swap<VType>(vtype_data[low], vtype_data[high]);
        ++low;
        --high;
      }
    }

  template <typename VType> void quick_sort(void *data, long low, long high) {
    std::stack<long> index_stack;
    index_stack.push(high);
    index_stack.push(low);
    long low_index, high_index, pivot_index;
    while (!index_stack.empty()) {
      low_index = index_stack.top();
      index_stack.pop();
      high_index = index_stack.top();
      index_stack.pop();
      if (low_index < high_index) {
        pivot_index = ::quick_sort_partition<VType>(data, low_index, high_index);
        index_stack.push(pivot_index);
        index_stack.push(low_index);
        index_stack.push(high_index);
        index_stack.push(pivot_index + 1);
      }
    }
  }

  template <typename VType> void quick_sort(void *data, long n_elements) {
    if (n_elements <= 1)
      return;
    ::quick_sort<VType>(data, 0, n_elements - 1);
  }

  template <typename VType>
    VType bkg_threshold(VType *data, const VID_t &n, double q) {
      static_assert(std::is_arithmetic<VType>::value,
          "VType must be arithmetic type");
      assert(data);
      assertm(q >= 0 && q <= 1, "desired background pct must be between 0 and 1");
      std::unique_ptr<VType[]> data_copy(new (std::nothrow) VType[n]);
      assert(data_copy);
      memcpy(data_copy.get(), data, sizeof(VType) * n);
      quick_sort<VType>(data_copy.get(), n);
      double index = (1 - q) * n;
      if (index < 0)
        return data_copy[0];
      if (index > n - 1)
        return data_copy[n - 1];
      auto low_index = (long)std::floor(index), high_index = (long)std::ceil(index);
      // nearest interpolation
      if (index - low_index <= high_index - index)
        return data_copy[low_index];
      else
        return data_copy[high_index];
    }

  template <typename GridTypePtr>
    GridTypePtr merge_grids(std::vector<GridTypePtr> grids) {
      if (grids.size() < 1)
        throw std::runtime_error("Can't merge empty grids");

      for (VID_t i = 0; i < (grids.size() - 1); ++i) {
        grids[i + 1]->tree().merge(grids[i]->tree(),
            vb::MERGE_ACTIVE_STATES_AND_NODES);
        // alternate method:
        // vb::tools::compActiveLeafVoxels(grids[i]->tree(), grids[i +
        // 1]->tree());
        // leaves grids[i] empty, copies all to grids[i+1]
      }
      auto final_grid = grids[grids.size() - 1];
      final_grid->tree().prune(); // collapse uniform values
      return final_grid;
    }

  // This is extremely slow don't use this
  // it's much faster to find all possible seeds and filter them
  // as is done in create_seed_pairs() anyway
  // Warning this can not clip mask.vdb files due to bug in openvdb::tools::clip
  // which is why SDFs are used
  // when attempting to parallelized this implementation had seg fault for unknown
  // reason
  template <typename GridTypePtr>
    GridTypePtr clip_by_seed(GridTypePtr grid, std::vector<Seed> seeds) {
#ifdef LOG
      std::cout << "\tClipping image by user passed seeds and +-max radius of "
        "each seed\n";
#endif
      auto timer = high_resolution_timer();

      auto component_grids = seeds | rv::transform([grid](const Seed &seed) {
          auto offset =
          GridCoord(seed.radius, seed.radius, seed.radius);
          vb::BBoxd clipBox((seed.coord - offset).asVec3d(),
              (seed.coord + offset).asVec3d());
          return vto::clip(*grid, clipBox);
          }) |
      rng::to_vector;
#ifdef LOG
      std::cout << "\tFinished seed clip in " << timer.elapsed() << '\n';
#endif
      timer.restart();

      auto merged = merge_grids(component_grids);
#ifdef LOG
      std::cout << "\tFinished seed merge in " << timer.elapsed() << '\n';
#endif
      return merged;
    }

  void write_marker_files(std::vector<MyMarker *> component_markers,
      fs::path component_dir_fn) {
    rng::for_each(component_markers, [&component_dir_fn](const auto marker) {
        // write marker file
        std::ofstream marker_file;
        auto mass = ((4 * PI) / 3.) * pow(marker->radius, 3);
        marker_file.open(component_dir_fn /
            ("marker_" + std::to_string(static_cast<int>(marker->x)) +
             "_" + std::to_string(static_cast<int>(marker->y)) + "_" +
             std::to_string(static_cast<int>(marker->z)) + "_" +
             std::to_string(int(mass))));

        marker_file << "# soma/seed x,y,z in original image\n";
        marker_file << marker->x << ',' << marker->y << ',' << marker->z << '\n';
        });
  }

#ifdef USE_HDF5

  int hdf5_attr(hid_t object_id, const char *attribute_name,
      std::unique_ptr<uint8_t[]> &buffer) {
    assertm(object_id >= 0, "object_id must be > 0");
    hid_t attribute_id = H5Aopen(object_id, attribute_name, H5P_DEFAULT);
    if (attribute_id < 0)
      std::runtime_error("can not open attribute " + string(attribute_name));
    H5A_info_t attribute_info;
    herr_t success = H5Aget_info(attribute_id, &attribute_info);
    assertm(success >= 0, "H5Aget_info failed");
    hsize_t n_bytes = attribute_info.data_size;
    buffer = make_unique<uint8_t[]>(n_bytes);
    hid_t type_id = H5Aget_type(attribute_id);
    assert(type_id >= 0);
    success = H5Aread(attribute_id, type_id, buffer.get());
    if (success < 0)
      std::runtime_error("failed to read attribute " + string(attribute_name));
    size_t n_type_bytes = H5Tget_size(type_id);
    int n_elements = (int)(n_bytes / n_type_bytes);
    H5Tclose(type_id);
    return n_elements;
  }

  auto imaris_image_bbox = [](std::string file_name, int resolution = 0,
      int channel = 0) {
    auto max_coord = zeros();
    auto file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    {
      hid_t dataset_id = H5Gopen(file_id, "DataSet", H5P_DEFAULT);
      {
        auto rname = "ResolutionLevel " + std::to_string(resolution);
        hid_t res_id = H5Gopen(dataset_id, rname.c_str(), H5P_DEFAULT);
        {
          auto tname = "TimePoint 0";
          hid_t time_id = H5Gopen(res_id, tname, H5P_DEFAULT);
          {
            auto cname = "Channel " + std::to_string(channel);
            hid_t channel_id = H5Gopen(time_id, cname.c_str(), H5P_DEFAULT);
            {
              int zdim, ydim, xdim;
              unique_ptr<uint8_t[]> buffer;
              int len = hdf5_attr(channel_id, "ImageSizeZ", buffer);
              // FIXME change this to static cast
              zdim = stoi(string((char *)buffer.get(), len));
              len = hdf5_attr(channel_id, "ImageSizeY", buffer);
              ydim = stoi(string((char *)buffer.get(), len));
              len = hdf5_attr(channel_id, "ImageSizeX", buffer);
              xdim = stoi(string((char *)buffer.get(), len));
              max_coord = GridCoord(xdim, ydim, zdim);
            }
            H5Gclose(channel_id);
          }
          H5Gclose(time_id);
        }
        H5Gclose(res_id);
      }
      H5Gclose(dataset_id);
    }
    H5Fclose(file_id);
    return CoordBBox(zeros(), max_coord);
  };

  hid_t memory_dataspace(const CoordBBox &bbox) {
    // hdf5 is all in z,y,x order but returns buffers in c-order
    hsize_t mem_dims[3] = {static_cast<hsize_t>(bbox.dim().z()),
      static_cast<hsize_t>(bbox.dim().y()),
      static_cast<hsize_t>(bbox.dim().x())};

    hid_t mem_dataspace_id = H5Screate_simple(3, mem_dims, mem_dims);
    if (mem_dataspace_id < 0)
      std::runtime_error("Error reading mem_dataspace_id");

    hsize_t mem_dataspace_start[3] = {0, 0, 0};
    hsize_t mem_dataspace_stride[3] = {1, 1, 1};
    hsize_t mem_dataspace_count[3] = {static_cast<hsize_t>(bbox.dim().z()),
      static_cast<hsize_t>(bbox.dim().y()),
      static_cast<hsize_t>(bbox.dim().x())};
    H5Sselect_hyperslab(mem_dataspace_id, H5S_SELECT_SET, mem_dataspace_start,
        mem_dataspace_stride, mem_dataspace_count, NULL);
    return mem_dataspace_id;
  }

  hid_t file_dataspace(hid_t data_id, const CoordBBox &bbox) {
    hid_t file_dataspace_id = H5Dget_space(data_id);
    if (file_dataspace_id < 0)
      std::runtime_error("Invalid H5Dget_space read");

    // offsets into the image
    hsize_t file_dataspace_start[3] = {static_cast<hsize_t>(bbox.min().z()),
      static_cast<hsize_t>(bbox.min().y()),
      static_cast<hsize_t>(bbox.min().x())};
    hsize_t file_dataspace_stride[3] = {1, 1, 1};
    hsize_t file_dataspace_count[3] = {static_cast<hsize_t>(bbox.dim().z()),
      static_cast<hsize_t>(bbox.dim().y()),
      static_cast<hsize_t>(bbox.dim().x())};
    H5Sselect_hyperslab(file_dataspace_id, H5S_SELECT_SET, file_dataspace_start,
        file_dataspace_stride, file_dataspace_count, NULL);
    return file_dataspace_id;
  }

  // all HDF5 calls are not thread safe
  auto load_imaris_tile = [](std::string file_name, const CoordBBox &bbox,
      int resolution = 0, int channel = 0) {
    // check inputs
    auto imaris_bbox = imaris_image_bbox(file_name, resolution, channel);
    // FIXME get voxel type
    // auto dense = std::make_unique<vto::Dense<uint16_t,
    // vto::LayoutXYZ>>(bbox);
    auto dense = std::make_unique<vto::Dense<uint8_t, vto::LayoutXYZ>>(bbox);

    if (!imaris_bbox.isInside(bbox)) {
      std::ostringstream os;
      os << "Requested bbox: " << bbox
        << " is not inside image bbox: " << imaris_bbox << '\n';
      std::runtime_error(os.str());
    }

    auto file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    {
      hid_t dataset_id = H5Gopen(file_id, "DataSet", H5P_DEFAULT);
      {
        auto rname = "ResolutionLevel " + std::to_string(resolution);
        hid_t res_id = H5Gopen(dataset_id, rname.c_str(), H5P_DEFAULT);
        {
          auto tname = "TimePoint 0";
          hid_t time_id = H5Gopen(res_id, tname, H5P_DEFAULT);
          {
            auto cname = "Channel " + std::to_string(channel);
            hid_t channel_id = H5Gopen(time_id, cname.c_str(), H5P_DEFAULT);
            {
              hid_t data_id = H5Dopen(channel_id, "Data", H5P_DEFAULT);
              if (data_id < 0) {
                std::runtime_error("Can not open Data at " + rname + tname +
                    cname);
              }
              hid_t mem_dataspace_id = memory_dataspace(bbox);
              // hid_t mem_type_id = H5T_NATIVE_USHORT;
              hid_t mem_type_id = H5T_NATIVE_UCHAR;
              hid_t file_dataspace_id = file_dataspace(data_id, bbox);
              herr_t success =
                H5Dread(data_id, mem_type_id, mem_dataspace_id,
                    file_dataspace_id, H5P_DEFAULT, dense->data());

              if (success < 0)
                std::runtime_error("Failed to read imaris hdf5 dataset");

              H5Dclose(data_id);
            }
            H5Gclose(channel_id);
          }
          H5Gclose(time_id);
        }
        H5Gclose(res_id);
      }
      H5Gclose(dataset_id);
    }
    H5Fclose(file_id);

    return dense;
  };

  /*
     auto get_hdf5_bbox = [](std::string file_name, int channel = 0) {
     auto file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  // open DataSetInfo group
  hid_t data_set_info = H5Gopen(file_id, "DataSetInfo");
  // open image group
  hid_t image = H5Gopen(data_set_info, "Image");
  // open attribute LSMEmissionWavelength
  hid_t vAttributeId = H5Aopen_name(image, "X");
  // get data space
  hid_t vAttributeSpaceId = H5Aget_space(vAttributeId);
  // get attribute value size
  hsize_t vAttributeSize = 0;
  H5Sget_simple_extent_dims(vAttributeSpaceId, &vAttributeSize, NULL);
  // create buffer
  char *vBuffer = new char[(bpSize)vAttributeSize + 1];
  vBuffer[vAttributeSize] = '\0';
  // read attribute value
  H5Aread(vAttributeId, H5T_C_S1, vBuffer);
  };

  auto get_hdf5_bbox = [](std::string file_name, int channel = 0) {
  auto mFileId = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  // open DataSetInfo group
  hid_t vDataSetInfoId = H5Gopen(mFileId, “DataSetInfo”);
  // open channel group
  hid_t vChannel3Id = H5Gopen(vDataSetInfoId, “Channel ” +
  std::to_string(channel));
  // open attribute LSMEmissionWavelength
  hid_t vAttributeId = H5Aopen_name(vChannelId, “LSMEmissionWavelength”);
  // get data space
  hid_t vAttributeSpaceId = H5Aget_space(vAttributeId);
  // get attribute value size
  hsize_t vAttributeSize = 0;
  H5Sget_simple_extent_dims(vAttributeSpaceId, &vAttributeSize, NULL);
  // create buffer
  char *vBuffer = new char[(bpSize)vAttributeSize + 1];
  vBuffer[vAttributeSize] = '\0';
  // read attribute value
  H5Aread(vAttributeId, H5T_C_S1, vBuffer);
  };

  auto get_hdf5_data = [](std::string file_name, int channel = 0) {
  mFileId = H5Fopen(mFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t vDataSetId = H5Gopen(mFileId, “DataSet”);
  hid_t vLevelId = H5Gopen(vDataSetId, “Resolution Level 0”);
  hid_t vTimePointId = H5Gopen(vLevelId, "TimePoint 0");
  hid_t vChannelId = H5Gopen(vTimePointId, "Channel 0");
  hid_t vDataId = H5Dopen(vChannelId, “Data”);
  // read the attributes ImageSizeX,Y,Z
  hsize_t vFileDim[3];
  hid_t vFileSpaceId = H5Screate_simple(3, vFileDim, NULL);
  char *vBuffer = new vBuffer[ImageSizeZ * ImageSizeY * ImageSizeX];
  H5Dread(vDataId, H5T_NATIVE_CHAR, H5S_ALL, vFileSpaceId, H5P_DEFAULT,
  vBuffer);
  };
  */

#endif // USE_HDF5

  /*
   * The tile size and shape define the requested "view" of the image
   * an image view is referred to as a tile
   * that we will load at one time. There is a one to one mapping
   * of an image view and a tile. There is also a one to
   * one mapping between each voxel of the image view and the
   * vertex of the tile. Note the tile is an array of
   * initialized unvisited structs, so they start off at arbitrary
   * location but are defined as they are visited.
   */
  template <typename image_t = uint16_t>
    std::unique_ptr<vto::Dense<image_t, vto::LayoutXYZ>>
    load_tile(const CoordBBox &bbox, const std::string &dir) {
      auto timer = high_resolution_timer();

      const auto tif_filenames = get_dir_files(dir, ".tif"); // sorted by z
      if (tif_filenames.empty()) {
        throw std::runtime_error("No .ims or .tiff images found in directory: " +
            dir);
      }

      // bbox defines an inclusive range
      auto tile_filenames =
        tif_filenames | rv::slice(bbox.min()[2], bbox.max()[2] + 1) |
        rv::remove_if([](auto const &fn) { return fn.empty(); }) | rng::to_vector;

      auto dense = read_tiff_planes<image_t>(tile_filenames, bbox);
#ifdef LOG
      // cout << "Load image " << bbox << " in " << timer.elapsed() << " sec." <<
      // '\n';
#endif
      return dense;
    }

  auto get_unique_fn = [](std::string probe_name) {
    // make sure it's a clean write
    while (fs::exists(probe_name)) {
      auto l = probe_name | rv::split('-') | rng::to<std::vector<std::string>>();
      l.back() = std::to_string(std::stoi(l.back()) + 1);
      probe_name = l | rv::join('-') | rng::to<std::string>();
    }
    return probe_name;
  };

  auto convert_fn_vdb = [](const fs::path &file_path, auto split_char,
      auto args) -> std::string {
    auto parent = file_path.parent_path();
    auto file_name = file_path.filename().string();

    std::string stripped = file_name | rv::split(split_char) | rv::drop_last(1) |
      rv::join(split_char) | rng::to<std::string>();
    if (!stripped.empty())
      stripped += "-";
    stripped += args->output_type;

    if (args->foreground_percent >= 0) {
      std::ostringstream out;
      out.precision(3);
      out << std::fixed << args->foreground_percent;
      stripped += "-fgpct-" + out.str();
    }
    if (args->image_offsets.z())
      stripped += "-zoff" + std::to_string(args->image_offsets.z());
    stripped += ".vdb";
    return (parent / stripped).string();
  };

  auto get_output_name = [](RecutCommandLineArgs *args) -> std::string {
    if (args->input_type == "ims") {
#ifndef USE_HDF5
      throw std::runtime_error("HDF5 dependency required for input type ims");
#endif
      return convert_fn_vdb(args->input_path, '.', args);
    }

    if (args->input_type == "tiff") {
      const auto tif_filenames = get_dir_files(args->input_path, ".tif");
      return convert_fn_vdb(tif_filenames[0], '_', args);
    }
    std::string default_output_name = "out.vdb";
    return default_output_name;
  };

  auto convert_sdf_to_points = [](auto sdf, auto image_lengths,
      auto foreground_percent) {
    // write_vdb_file({sdf}, "fog.vdb");
    std::vector<PositionT> positions;
    sdf->tree().isValueOn(GridCoord(0, 0, 0));
    for (auto iter = sdf->cbeginValueOn(); iter.test(); ++iter) {
      auto coord = iter.getCoord();
      positions.push_back(PositionT(coord.x(), coord.y(), coord.z()));
    }

    auto topology_grid = create_point_grid(positions, image_lengths,
        get_transform(), foreground_percent);
    append_attributes(topology_grid);
    return topology_grid;
  };

  auto anisotropic_factor = [](std::array<double, 3> voxel_size) {
    auto min_max_pair = min_max(voxel_size);
    // round to the nearest int
    return static_cast<uint16_t>((min_max_pair.second / min_max_pair.first) + .5);
  };

  // write seed/somas to disk
  // Converts the seeds from voxel space to um space
  auto write_seeds = [](fs::path run_dir, std::vector<Seed> seeds,
      std::array<double, 3> voxel_size) {
    // start seeds directory
    fs::path seed_dir = run_dir / "seeds";
    fs::create_directories(seed_dir);

    rng::for_each(seeds, [&](const auto &seed) {
        std::ofstream seed_file;
        seed_file.open(
            seed_dir /
            ("marker_" +
             std::to_string(static_cast<int>(seed.coord.x() * voxel_size[0])) +
             "_" +
             std::to_string(static_cast<int>(seed.coord.y() * voxel_size[1])) +
             "_" +
             std::to_string(static_cast<int>(seed.coord.z() * voxel_size[2])) +
             "_" + std::to_string(static_cast<int>(seed.volume))),
            std::ios::app);
        seed_file << std::fixed << std::setprecision(SWC_PRECISION);
        seed_file << "#x,y,z,radius in um based of voxel size: [" << voxel_size[0]
        << ',' << voxel_size[1] << ',' << voxel_size[2] << "]\n";
        seed_file << voxel_size[0] * seed.coord.x() << ','
        << voxel_size[1] * seed.coord.y() << ','
        << voxel_size[2] * seed.coord.z() << ',' << seed.radius_um
        << '\n';
        });
  };

  // center by bbox
  auto get_center_of_grid = [](auto component) -> GridCoord {
    auto bbox = component->evalActiveVoxelBoundingBox();
    auto center = bbox.getCenter();
    return new_grid_coord(center.x(), center.y(), center.z());
  };

  auto is_coordinate_active = [](EnlargedPointDataGrid::Ptr topology_grid,
      GridCoord coord) {
    auto leaf = topology_grid->tree().probeLeaf(coord);
    try {
      if (leaf) {
        return leaf->beginIndexVoxel(coord)
          ? true
          : false; // remove if outside the surface
      } else {
        return false;
      }
    } catch (...) {
      return false;
    }
  };

  // take a volume in image space (pixel) units
  // and adjust it to an approximate coverage in
  // world space (um^3) units
  // this works for iso or aniso voxel sizes
  // the adjustment would be exact for cubes
  // for spheres its probably close enough
  auto adjust_volume_by_voxel_size =
    [](uint64_t volume, std::array<double, 3> voxel_size) -> uint64_t {
      return static_cast<double>(volume) * voxel_size[0] * voxel_size[1] *
        voxel_size[2];
    };

  std::vector<Seed>
    filter_seeds_by_points(EnlargedPointDataGrid::Ptr topology_grid,
        std::vector<Seed> seeds) {

      // auto local_bbox =
      // openvdb::math::CoordBBox(grid_offsets, grid_offsets + grid_lengths);

      auto filtered_seeds =
        seeds | rv::remove_if([topology_grid](auto seed) {
            // if (local_bbox.isInside(seed.coord)) {
            if (topology_grid->tree().isValueOn(seed.coord)) {
            return false;
            } else {
#ifdef FULL_PRINT
            cout << "Warning: seed at " << seed.coord
            << " is not selected in the segmentation so it is "
            "ignored. "
            "May indicate the image and marker directories are  "
            "mismatched or major inaccuracies in segmentation\n ";
#endif
            }
            //} else {
            // #ifdef FULL_PRINT
            // cout << "Warning: seed at " << seed.coord << " in image bbox "
            //<< local_bbox
            //<< " is not within the images bounding box so it is "
            //"ignored\n";
            // #endif
            //}
            return true; // remove it
        }) |
      rng::to_vector;

#ifdef LOG
      cout << "Only " << filtered_seeds.size() << " of " << seeds.size()
        << " seeds in directory have an active voxel and are connected to a "
        "component in the provided image\n";
#endif
      return filtered_seeds;
    }

  std::unique_ptr<vto::LevelSetFilter<openvdb::FloatGrid>>
    create_morph_filter(openvdb::FloatGrid::Ptr sdf) {
      // establish the filter for opening
      auto filter = std::make_unique<vto::LevelSetFilter<openvdb::FloatGrid>>(*sdf);
      filter->setSpatialScheme(openvdb::math::FIRST_BIAS);
      filter->setTemporalScheme(openvdb::math::TVD_RK1);
      return filter;
    }

  std::optional<std::pair<GridCoord, float>>
    sdf_to_seed(const openvdb::FloatGrid::Ptr sdf_component) {
      // check input
      assertm(sdf_component->getGridClass() == openvdb::GRID_LEVEL_SET,
          "sdf_to_seed() only accepts grids of type level set");
      assertm(
          sdf_component->activeVoxelCount(),
          "sdf_to_seed() can only accept components with at least 1 active voxel");

      // for (auto voxelIter = sdf_component->cbeginValueOn(); voxelIter;
      // ++voxelIter) { auto coord = voxelIter.getCoord(); std::cout << coord <<
      // '\n';
      //}
      // std::cout << "finished printing sdf components\n";

      auto volume_voxels = sdf_component->activeVoxelCount();
      // estimate the radii from the volume of active voxels
      float radius_voxels = std::cbrtf((volume_voxels * 3) / (4 * PI));
      auto current_sdf = sdf_component->deepCopy();
      auto next_sdf = sdf_component->deepCopy();

      // for (auto voxelIter = current_sdf->cbeginValueOn(); voxelIter; ++voxelIter)
      // { auto coord = voxelIter.getCoord();
      //// auto val = sdf_component->getValue(voxelIter.getCoord());
      // auto leaf = sdf_component->tree().probeLeaf(coord);
      //// try {
      // if (leaf) {
      // auto val = leaf->getValue(coord);
      // std::cout << coord << ' ' << " val " << val << '\n';
      //}
      //}

      // establish the filter for opening
      auto filter = create_morph_filter(next_sdf);

      // saves the grid before erasure in current_sdf
      while (next_sdf->activeVoxelCount()) {
        // std::cout << "voxel count " << next_sdf->activeVoxelCount() << '\n';
        //  since we know next_sdf still hasn't been completely erased
        //  by opening yet, we should preserve a copy of it in case
        //  its about to be erased
        current_sdf = next_sdf->deepCopy();

        // erode
        // contracts next_sdf towards its center
        // the smaller this number is the finer the estimate of the
        // center however the final contraction point can be outside
        // of the original component which is not biologically
        // plausible and has to be unnecessarily discarded for lack
        // of a better method
        // note that vto::fillWithSpheres also can have a similar problem
        filter->offset(1);
      }

      assertm(current_sdf->activeVoxelCount(),
          "current_sdf must have at least 1 active voxel");
      std::cout << "Final voxel count " << current_sdf->activeVoxelCount() << '\n';
      // current_sdf is now approximately a tiny spherical level set which
      // approximates the center of mass of the component morphologically

      // Need to guarantee your on an active voxel
      // Iterate through all the active voxels and choose the one with the maximum
      // radius in the OG SDF
      auto min_location =
        std::make_pair(std::numeric_limits<float>::max(), GridCoord{0, 0, 0});
      float val;
      for (auto voxelIter = current_sdf->cbeginValueOn(); voxelIter; ++voxelIter) {
        auto coord = voxelIter.getCoord();
        // std::cout << coord << '\n';
        //  auto val = sdf_component->getValue(voxelIter.getCoord());
        auto leaf = sdf_component->tree().probeLeaf(coord);
        // try {
        if (leaf) {
          val = leaf->getValue(coord);
          // std::cout << ' ' << " val " << val << '\n';
          if (val < min_location.first)
            min_location = std::make_pair(val, coord);
        }
      }

      // this should be impossible to trigger since current_sdf always
      // has at least 1 voxel to iterate and current sdf should be a erosion
      // (a subset) of sdf_component
      if (min_location.first == std::numeric_limits<float>::max()) {
        // throw std::runtime_error("sdf_to_seed() found no valid location");
        std::cout << "  discarded\n";
        return {};
      }

      return std::make_pair(min_location.second, radius_voxels);
      }

      auto binarize_uint8_grid = [](auto image_grid) {
        auto accessor = image_grid->getAccessor();
        for (auto iter = image_grid->beginValueOn(); iter; ++iter) {
          auto coord = iter.getCoord();
          auto val = accessor.getValue(coord);
          if (val > 0) {
            accessor.setValue(coord, 255);
          }
        }
      };
