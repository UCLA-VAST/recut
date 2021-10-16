#pragma once

#include "config.hpp"
#include "markers.h"
#include "range/v3/all.hpp"
#include "recut_parameters.hpp"
#include "vertex_attr.hpp"
#include <algorithm> //min, clamp
#include <atomic>
#include <chrono>
#include <cstdlib> //rand srand
#include <ctime>   // for srand
#include <filesystem>
#include <math.h>
#include <numeric>
#include <openvdb/tools/Composite.h>
#include <stdlib.h> // ultoa

namespace fs = std::filesystem;
namespace rng = ranges;

#ifdef USE_MCP3D
#include <common/mcp3d_common.hpp>
#include <common/mcp3d_utility.hpp> // PadNumStr
#include <image/mcp3d_image.hpp>
#include <image/mcp3d_voxel_types.hpp> // convert to CV type
#include <image/mcp3d_voxel_types.hpp>
#include <opencv2/opencv.hpp> // imwrite
#endif

#define PI 3.14159265
// be able to change pp values into std::string
#define XSTR(x) STR(x)
#define STR(x) #x

auto print_iter = [](auto iterable) {
  rng::for_each(iterable, [](auto i) { std::cout << i << ", "; });
  std::cout << '\n';
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

// taken from Bryce Adelstein Lelbach's Benchmarking C++ Code talk:
struct high_resolution_timer {
  high_resolution_timer() : start_time_(take_time_stamp()) {}

  void restart() { start_time_ = take_time_stamp(); }

  double elapsed() const // return elapsed time in seconds
  {
    return double(take_time_stamp() - start_time_) * 1e-9;
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

// Note: this is linux specific, other cross platform solutions available:
//     https://stackoverflow.com/questions/1528298/get-path-of-executable
std::string get_parent_dir() {
  // fs::path full_path(fs::current_path());
  fs::path full_path(fs::canonical("/proc/self/exe"));
  return fs::canonical(full_path.parent_path().parent_path()).string();
}

std::string get_data_dir() { return CMAKE_INSTALL_DATADIR; }

VID_t get_central_coord(int grid_size) {
  return grid_size / 2 - 1; // place at center
}

VID_t get_central_vid(int grid_size) {
  auto coord = get_central_coord(grid_size);
  auto root_vid =
      (VID_t)coord * grid_size * grid_size + coord * grid_size + coord;
  return root_vid; // place at center
}

VID_t get_central_diag_vid(int grid_size) {
  auto coord = get_central_coord(grid_size);
  coord++; // add 1 to all x, y, z
  auto root_vid =
      (VID_t)coord * grid_size * grid_size + coord * grid_size + coord;
  return root_vid; // place at center diag
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
  return coords | rng::views::transform([](GridCoord coord) {
           return new MyMarker(static_cast<double>(coord.x()),
                               static_cast<double>(coord.y()),
                               static_cast<double>(coord.z()));
         }) |
         rng::to_vector;
}

/* interval_length parameter is actually irrelevant due to
 * copy on write, the chunk requested during reading
 * or mmapping is
 */
VID_t get_used_vertex_size(VID_t grid_size, VID_t block_size) {
  auto len = grid_size / block_size;
  auto total_blocks = len * len * len;
  auto pad_block_size = block_size + 2;
  auto pad_block_num = pad_block_size * pad_block_size * pad_block_size;
  // this is the total vertices that will be used including ghost cells
  auto interval_vert_num = pad_block_num * total_blocks;
  return interval_vert_num;
}

auto print_marker_3D = [](auto markers, auto interval_lengths,
                          std::string stage) {
  for (int zi = 0; zi < interval_lengths[2]; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * interval_lengths[0] + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < interval_lengths[1]; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < interval_lengths[0]; xi++) {
        VID_t index = ((VID_t)xi) + yi * interval_lengths[0] +
                      zi * interval_lengths[1] * interval_lengths[0];
        auto value = std::string{"-"};
        for (const auto &m : markers) {
          if (m->vid(interval_lengths[0], interval_lengths[1]) == index) {
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
void print_vdb_mask(T vdb_accessor, const GridCoord &lengths,
                    const int bkg_thresh = -1) {
  cout << "print_vdb_mask(): \n";
  for (int z = 0; z < lengths[2]; z++) {
    cout << "y | Z=" << z << '\n';
    for (int x = 0; x < 2 * lengths[0] + 4; x++) {
      cout << "-";
    }
    cout << '\n';
    for (int y = 0; y < lengths[1]; y++) {
      cout << y << " | ";
      for (int x = 0; x < lengths[0]; x++) {
        openvdb::Coord xyz(x, y, z);
        auto val = vdb_accessor.isValueOn(xyz);
        // if ((bkg_thresh > -1) && (val <= bkg_thresh)) {
        if (val) {
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

auto ids_to_coords = [](auto ids, auto lengths) {
  return ids | rng::views::transform([&](auto v) {
           return id_to_coord(v, lengths);
         }) |
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

auto copy_selected =
    [](EnlargedPointDataGrid::Ptr grid) -> openvdb::FloatGrid::Ptr {
  auto float_grid = openvdb::FloatGrid::create();
  VID_t vcount = 0;

  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    auto origin = leaf_iter->getNodeBoundingBox().min();
    auto float_leaf =
        new openvdb::tree::LeafNode<float, LEAF_LOG2DIM>(origin, 0.);

    // note: some attributes need mutability
    openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
        leaf_iter->attributeArray("flags"));

    for (auto ind = leaf_iter->beginIndexOn(); ind; ++ind) {
      if (is_selected(flags_handle, ind)) {
        float_leaf->setValue(ind.getCoord(), 1.);
        ++vcount;
      }
    }
    float_grid->tree().addLeaf(float_leaf);
  }

  float_grid->tree().prune();
  cout << "total active: " << vcount << '\n';
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

template <typename T>
void print_image_3D(const T *inimg1d, const GridCoord interval_lengths,
                    const T bkg_thresh = 0) {
  cout << "Print image 3D:\n";
  for (int zi = 0; zi < interval_lengths[2]; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * interval_lengths[0] + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < interval_lengths[1]; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < interval_lengths[0]; xi++) {
        VID_t index = ((VID_t)xi) + yi * interval_lengths[0] +
                      zi * interval_lengths[0] * interval_lengths[1];
        auto val = inimg1d[index];
        if (val > bkg_thresh) {
          cout << +(val) << " ";
        } else {
          cout << "- ";
        }
      }
      cout << '\n';
    }
    cout << '\n';
  }
}

template <typename T> void print_image(T *inimg1d, VID_t size) {
  cout << "print image " << '\n';
  for (VID_t i = 0; i < size; i++) {
    cout << i << " " << +inimg1d[i] << '\n';
  }
}

// Note this test is on a single pixel width path through
// the domain, thus it's an extremely hard test to pass
// not even original fastmarching can
// recover all of the original pixels
VID_t trace_mesh_image(VID_t id, uint16_t *inimg1d,
                       const VID_t desired_selected, int grid_size) {
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

auto set_grid_meta = [](auto grid, auto lengths, auto bkg_thresh) {
  grid->setName("topology");
  grid->setCreator("recut");
  grid->setIsInWorldSpace(true);
  grid->setGridClass(openvdb::GRID_FOG_VOLUME);
  grid->insertMeta("original_bounding_extent_x",
                   openvdb::FloatMetadata(static_cast<float>(lengths[0])));
  grid->insertMeta("original_bounding_extent_y",
                   openvdb::FloatMetadata(static_cast<float>(lengths[1])));
  grid->insertMeta("original_bounding_extent_z",
                   openvdb::FloatMetadata(static_cast<float>(lengths[2])));

  grid->insertMeta("bkg_thresh", openvdb::FloatMetadata(bkg_thresh));
};

auto copy_to_point_grid = [](openvdb::FloatGrid::Ptr other, auto lengths,
                             float bkg_thresh = 0.) {
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

  grid->tree().prune();

  set_grid_meta(grid, lengths, bkg_thresh);

  return grid;
};

auto create_point_grid = [](auto &positions, auto lengths, auto transform_ptr,
                            float bkg_thresh = 0.) {
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

  set_grid_meta(grid, lengths, bkg_thresh);

  return grid;
};

auto create_vdb_grid = [](auto lengths, float bkg_thresh = 0.) {
  auto topology_grid = EnlargedPointDataGrid::create();
  set_grid_meta(topology_grid, lengths, bkg_thresh);

  return topology_grid;
};

auto get_metadata = [](auto vdb_grid) -> std::pair<GridCoord, float> {
  GridCoord image_lengths(0, 0, 0);
  float bkg_thresh = 0; // default value if not found acceptable
  for (openvdb::MetaMap::MetaIterator iter = vdb_grid->beginMeta();
       iter != vdb_grid->endMeta(); ++iter) {
    // name and val
    const std::string &name = iter->first;
    openvdb::Metadata::Ptr value = iter->second;

    if (name == "original_bounding_extent_x") {
      image_lengths[0] = static_cast<openvdb::FloatMetadata &>(*value).value();
    } else if (name == "original_bounding_extent_y") {
      image_lengths[1] = static_cast<openvdb::FloatMetadata &>(*value).value();
    } else if (name == "original_bounding_extent_z") {
      image_lengths[2] = static_cast<openvdb::FloatMetadata &>(*value).value();
    } else if (name == "bkg_thresh") {
      bkg_thresh = static_cast<openvdb::FloatMetadata &>(*value).value();
    }
  }
  return std::pair(image_lengths, bkg_thresh);
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

#ifdef LOG
  cout << "appended all attributes\n";
#endif
};

auto read_vdb_file(std::string fn, std::string grid_name = "topology") {
#ifdef LOG
  cout << "Reading vdb file: " << fn << " grid: " << grid_name << " ...\n";
#endif
  if (!fs::exists(fn)) {
    cout << "Input image file does not exist or not found, exiting...\n";
    exit(1);
  }
  openvdb::io::File file(fn);
  file.open();
  openvdb::GridBase::Ptr base_grid = file.readGrid(grid_name);
  // EnlargedPointDataGrid grid =
  // openvdb::gridPtrCast<EnlargedPointDataGrid>(file.readGrid(grid_name));
  file.close();

#ifdef LOG
  print_grid_metadata(base_grid);
#endif
  return base_grid;
}

// Create a VDB file object and write out a vector of grids.
// Add the grid pointer to a container.
// openvdb::GridPtrVec grids;
// grids.push_back(grid);
void write_vdb_file(openvdb::GridPtrVec vdb_grids, std::string fp = "") {

  // safety checks
  auto default_fn = "topology.vdb";
  if (fp.empty()) {
    fp = get_data_dir() + '/' + default_fn;
  } else {
    auto dir = fs::path(fp).remove_filename();
    if (dir.empty() || fs::exists(dir)) {
      if (fs::exists(fp)) {
#ifdef LOG
        cout << "Warning: " << fp << " already exists, overwriting...\n";
#endif
      }
    } else {
#ifdef LOG
      cout << "Directory: " << dir << " does not exist, creating...\n";
#endif
      fs::create_directories(dir);
    }
  }

  auto timer = new high_resolution_timer();
  openvdb::io::File vdb_file(fp);
  cout << "start write of vdb\n";
  vdb_file.write(vdb_grids);
  vdb_file.close();

#ifdef LOG
  cout << "Finished write whole grid in: " << timer->elapsed() << " sec\n";
  cout << "Wrote output to " << fp << '\n';
#endif
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
VID_t create_image(int tcase, uint16_t *inimg1d, int grid_size,
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
          for (int ri = 0; ri < Rvecs.size(); ri++) {
            double Rvec = Rvecs[ri];
            bool condition0, condition1;
            condition0 = condition1 = false;
            if (Rvec < R < Rvec + w) {
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
  int interval = grid_size / line_per_dim; // roughly equiv
  std::vector<VID_t> x(line_per_dim + 1);
  std::vector<VID_t> y(line_per_dim + 1);
  std::vector<VID_t> z(line_per_dim + 1);
  VID_t i, j, k, count;
  i = j = k = 0;
  count = 0;
  VID_t selected = 0;
  for (int count = 0; count < grid_size; count += interval) {
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
      for (int zi; zi < grid_size; zi++) {
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
      for (int yi; yi < grid_size; yi++) {
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
      for (int xi; xi < grid_size; xi++) {
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

RecutCommandLineArgs get_args(int grid_size, int interval_length,
                              int block_size, int slt_pct, int tcase,
                              bool force_regenerate_image = false,
                              bool input_is_vdb = false,
                              std::string type = "point",
                              int downsample_factor = 1) {

  bool print = false;

  RecutCommandLineArgs args;
  args.type_ = type;
  auto params = args.recut_parameters();
  auto str_path = get_data_dir();
  params.set_marker_file_path(
      str_path + "/test_markers/" + std::to_string(grid_size) + "/tcase" +
      std::to_string(tcase) + "/slt_pct" + std::to_string(slt_pct) + "/");
  auto lengths = GridCoord(grid_size);
  args.set_image_lengths(lengths);
  args.set_image_offsets(zeros());

  // tcase 6 means use real data, in which case we need to either
  // set max and min explicitly (to save time) or recompute what the
  // actual values are
  if (tcase == 6) {
    // selected percent is only use for tcase 6 and 4
    // otherwise it is ignored for other tcases so that
    // nothing is recalculated
    // note: a background_thresh of 0 would simply take all pixels within the
    // domain and check that all were used

    // first marker is at 58, 230, 111 : 7333434
    // args.set_image_offsets({57, 228, 110});
    // root at {1125, 12949, 344}
    args.set_image_offsets(
        {1123 / downsample_factor, 12947 / downsample_factor, 342});
    args.set_image_lengths({grid_size, grid_size, grid_size});

    if (const char *env_p = std::getenv("TEST_IMAGE")) {
      std::cout << "Using $TEST_IMAGE environment variable: " << env_p << '\n';
      args.set_image_root_dir(std::string(env_p));
    } else {
      std::cout << "Warning likely fatal: must run: export "
                   "TEST_IMAGE=\"abs/path/to/image\" to set the environment "
                   "variable\n\n";
    }

    if (const char *env_p = std::getenv("TEST_MARKER")) {
      std::cout << "Using $TEST_MARKER environment variable: " << env_p << '\n';
      params.set_marker_file_path(std::string(env_p));
    } else {
      std::cout << "Warning likely fatal must run: export "
                   "TEST_MARKER=\"abs/path/to/marker\" to set the environment "
                   "variable\n\n";
    }

    // foreground_percent is always double between .0 and 1.
    params.set_foreground_percent(static_cast<double>(slt_pct) / 100.);
    // pre-determined and hardcoded thresholds for the file above
    // to save time recomputing is disabled
  } else {
    // by setting the max intensities you do not need to recompute them
    // in the update function, this is critical for benchmarking
    params.set_max_intensity(2);
    params.set_min_intensity(0);
    params.force_regenerate_image = force_regenerate_image;
    auto image_root_dir =
        str_path + "/test_images/" + std::to_string(grid_size) + "/tcase" +
        std::to_string(tcase) + "/slt_pct" + std::to_string(slt_pct);
    if (input_is_vdb) {
      if (args.type_ == "point") {
        args.set_image_root_dir(image_root_dir + "/point.vdb");
      } else if (args.type_ == "float") {
        args.set_image_root_dir(image_root_dir + "/float.vdb");
      }
    } else {
      args.set_image_root_dir(image_root_dir);
    }
  }

  // the total number of blocks allows more parallelism
  // ideally intervals >> thread count
  params.interval_length = interval_length;
  VID_t img_vox_num = grid_size * grid_size * grid_size;
  params.tcase = tcase;
  params.slt_pct = slt_pct;
  params.selected = img_vox_num * (slt_pct / (float)100);
  params.root_vid = get_central_vid(grid_size);

  // For now, params are only saved if this
  // function is called, in the future
  // args and params should be combined to be
  // flat
  args.set_recut_parameters(params);
  if (print)
    args.PrintParameters();

  return args;
}

void write_marker(VID_t x, VID_t y, VID_t z, std::string fn) {
  auto print = false;
#ifdef LOG
  print = true;
#endif

  bool rerun = false;
  if (!fs::exists(fn) || rerun) {
    fs::remove_all(fn); // make sure it's an overwrite
    if (print)
      cout << "      Delete old: " << fn << '\n';
    fs::create_directories(fn);
    fn = fn + "/marker";
    std::ofstream mf;
    mf.open(fn);
    mf << "# x,y,z\n";
    mf << x << "," << y << "," << z;
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
  template <typename T>
  friend std::ostream &operator<<(std::ostream &os, const Histogram<T> &hist) {

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

    auto these_keys = this->bin_counts | rng::views::keys | rng::to_vector;
    auto rhs_keys = rhistogram.bin_counts | rng::views::keys | rng::to_vector;

    auto matches =
        rng::views::set_intersection(these_keys, rhs_keys) | rng::to_vector;

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
      for (const auto key : rng::views::set_difference(keys, matches)) {
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
                        GridCoord buffer_offsets, int granularity = 8) {
  auto histogram = Histogram<image_t>(granularity);
  for (auto z : rng::views::iota(0, buffer_lengths[2])) {
    for (auto y : rng::views::iota(0, buffer_lengths[1])) {
      for (auto x : rng::views::iota(0, buffer_lengths[0])) {
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
auto convert_buffer_to_vdb_acc = [](auto buffer, GridCoord buffer_lengths,
                                    GridCoord buffer_offsets,
                                    GridCoord image_offsets, auto accessor,
                                    auto bkg_thresh = 0, int upsample_z = 1) {
  // half-range of uint8_t, recorded max values of 8k / 64 -> ~128
  auto val_transform = [](auto val) { return std::clamp(val / 64, 0, 127); };

  for (auto z : rng::views::iota(0, buffer_lengths[2])) {
    for (auto y : rng::views::iota(0, buffer_lengths[1])) {
      for (auto x : rng::views::iota(0, buffer_lengths[0])) {
        GridCoord xyz(x, y, z);
        GridCoord buffer_xyz = coord_add(xyz, buffer_offsets);
        GridCoord grid_xyz = coord_add(xyz, image_offsets);
        auto val = buffer[coord_to_id(buffer_xyz, buffer_lengths)];
        // voxels equal to bkg_thresh are always discarded
        if (val > bkg_thresh) {
          for (auto upsample_z_idx : rng::views::iota(0, upsample_z)) {
            auto upsample_grid_xyz =
                GridCoord(grid_xyz[0], grid_xyz[1],
                          (upsample_z * grid_xyz[2]) + upsample_z_idx);
            accessor.setValue(upsample_grid_xyz, val_transform(val));
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
  print_coord(buffer_lengths, "buffer_lengths");
  print_coord(buffer_offsets, "buffer_offsets");
  print_coord(image_offsets, "image_offsets");
  for (auto z : rng::views::iota(0, buffer_lengths[2])) {
    for (auto y : rng::views::iota(0, buffer_lengths[1])) {
      for (auto x : rng::views::iota(0, buffer_lengths[0])) {
        GridCoord xyz(x, y, z);
        GridCoord buffer_xyz = coord_add(xyz, buffer_offsets);
        GridCoord grid_xyz = coord_add(xyz, image_offsets);
        auto val = buffer[coord_to_id(buffer_xyz, buffer_lengths)];
        // voxels equal to bkg_thresh are always discarded
        if (val > bkg_thresh) {
          for (auto upsample_z_idx : rng::views::iota(0, upsample_z)) {
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
// for (auto z : rng::views::iota(0, buffer_lengths[2])) {
// for (auto y : rng::views::iota(0, buffer_lengths[1])) {
// for (auto x : rng::views::iota(0, buffer_lengths[0])) {
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

#ifdef USE_MCP3D
void write_tiff(uint16_t *inimg1d, std::string base, int grid_size,
                bool rerun = false) {
  auto print = false;
#ifdef LOG
  print = true;
#endif

  base = base + "/ch0";
  std::vector<VID_t> plane_extents{static_cast<VID_t>(grid_size),
                                   static_cast<VID_t>(grid_size), 1};
  if (!fs::exists(base) || rerun) {
    fs::remove_all(base); // make sure it's an overwrite
    if (print)
      cout << "      Delete old: " << base << '\n';
    fs::create_directories(base);
    // print_image(inimg1d, grid_size * grid_size * grid_size);
    for (int zi = 0; zi < grid_size; zi++) {
      std::string fn = base;
      fn = fn + "/img_";
      // fn = fn + mcp3d::PadNumStr(zi, 9999); // pad to 4 digits
      fn = fn + std::to_string(zi); // pad to 4 digits
      std::string suff = ".tif";
      fn = fn + suff;
      VID_t start = zi * grid_size * grid_size;
      // cout << "fn: " << fn << " start: " << start << '\n';
      // print_image(&(inimg1d[start]), grid_size * grid_size);
      // cout << fn << '\n';
      // print_image_3D(&(inimg1d[start]), plane_extents);

      { // cv write
        int cv_type = mcp3d::VoxelTypeToCVType(mcp3d::VoxelType::M16U, 1);
        cv::Mat m(grid_size, grid_size, cv_type, &(inimg1d[start]));
        cv::imwrite(fn, m);
      }

      // uint8_t* ptr = Plane(z, c, t);
      // std::vector<int> dims = {grid_size, grid_size, grid_size};
      // mcp3d::image::MImage mimg(dims); // defaults to uint16 format
    }
    if (print)
      cout << "      Wrote test images in: " << base << '\n';
  }
}

auto read_tiff = [](std::string fn, auto image_offsets, auto image_lengths,
                    mcp3d::MImage &image) {
  cout << "Read: " << fn << '\n';
  // read data from channel 0
  image.ReadImageInfo({0}, true);
  try {
    // use unit strides only
    mcp3d::MImageBlock block(
        {image_offsets[0], image_offsets[1], image_offsets[2]},
        {image_lengths[0], image_lengths[1], image_lengths[2]});
    image.SelectView(block, 0);
    image.ReadData(true, "quiet");
  } catch (...) {
    assertm(false, "error in image io. neuron tracing not performed");
  }
};
#endif

// stamp the compile time config
// so that past logs are explicit about
// their flags
void print_macros() {

#ifdef RV
  cout << "RV" << '\n';
#endif

#ifdef NO_RV
  cout << "NO_RV" << '\n';
#endif

#ifdef USE_OMP_BLOCK
  cout << "USE_OMP_BLOCK" << '\n';
#endif

#ifdef USE_OMP_INTERVAL
  cout << "USE_OMP_INTERVAL" << '\n';
#endif
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
  return tree | rng::views::transform([&](auto v) {
           return v->vid(lengths[0], lengths[1]);
         }) |
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

  auto matches = rng::views::set_intersection(truth_vids, check_vids);

  VID_t match_count = rng::distance(matches);
  if (print)
    std::cout << "match count: " << match_count << '\n';

  auto get_negatives = [&](auto &tree, auto &matches, auto specifier) {
    return rng::views::set_difference(tree, matches) |
           rng::views::transform([&](auto vid) {
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

template <typename I> std::ostream open_swc_outputs(I root_vids) {
  std::ofstream out("out.swc");
  return out;
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

// n,type,x,y,z,radius,parent
// for more info see:
// http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
// https://github.com/HumanBrainProject/swcPlus/blob/master/SWCplus_specification.html
auto print_swc_line = [](GridCoord swc_coord, bool is_root, uint8_t radius,
                         const OffsetCoord parent_offset_coord,
                         const CoordBBox &bbox, std::ofstream &out,
                         std::map<GridCoord, uint32_t> &coord_to_swc_id,
                         bool bbox_adjust = true) {
  std::ostringstream line;

  GridCoord swc_lengths = bbox.extents();
  if (bbox_adjust) {
    swc_coord = swc_coord - bbox.min();
    // CoordBBox uses extents inclusively, but we want exclusive bbox
    swc_lengths = bbox.extents().offsetBy(-1);
  }

  auto find_or_assign = [&coord_to_swc_id](GridCoord swc_coord) -> uint32_t {
    auto val = coord_to_swc_id.find(swc_coord);
    if (val == coord_to_swc_id.end()) {
      auto new_val = coord_to_swc_id.size();
      coord_to_swc_id[swc_coord] = new_val;
      assertm(new_val == (coord_to_swc_id.size() - 1),
              "map must now be 1 size larger");
      return new_val;
    }
    return coord_to_swc_id[swc_coord];
  };

  // n
  uint32_t id;
  if (coord_to_swc_id.empty()) {
    id = coord_to_id(swc_coord, swc_lengths);
  } else {
    id = find_or_assign(swc_coord);
  }
  assertm(id < std::numeric_limits<int32_t>::max(),
          "id overflows int32_t limit");
  line << id << ' ';

  // type_id
  if (is_root) {
    line << "1" << ' ';
  } else {
    line << '3' << ' ';
  }

  // coordinates
  line << swc_coord[0] << ' ' << swc_coord[1] << ' ' << swc_coord[2] << ' ';

  // radius
  line << +(radius) << ' ';

  // parent
  if (is_root) {
    line << "-1";
  } else {
    auto parent_coord = coord_add(swc_coord, parent_offset_coord);
    uint32_t parent_vid;
    if (coord_to_swc_id.empty()) {
      parent_vid = coord_to_id(parent_coord, swc_lengths);
    } else {
      parent_vid = find_or_assign(parent_coord);
    }
    assertm(parent_vid < std::numeric_limits<int32_t>::max(),
            "id overflows int32_t limit");
    line << parent_vid;
  }

  line << '\n';

  if (out.is_open()) {
    out << line.str();
  } else {
    std::cout << line.str();
  }
};

auto get_transform = []() {
  // grid_transform must use the same voxel size for all intervals
  // and be identical
  auto grid_transform =
      openvdb::math::Transform::createLinearTransform(VOXEL_SIZE);
  // The offset to cell-center points
  const openvdb::math::Vec3d offset(VOXEL_SIZE / 2., VOXEL_SIZE / 2.,
                                    VOXEL_SIZE / 2.);
  grid_transform->postTranslate(offset);
  return grid_transform;
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
auto get_origins =
    [](EnlargedPointDataGrid::Ptr grid) -> std::vector<GridCoord> {
  std::vector<GridCoord> origins;
  // TODO use leafManager and tbb::parallel_for
  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    origins.push_back(leaf_iter->origin());
  }
  std::sort(origins.begin(), origins.end());
  return origins;
};

auto leaves_intersect = [](EnlargedPointDataGrid::Ptr grid,
                           EnlargedPointDataGrid::Ptr other) {
  std::vector<GridCoord> out;
  // inputs must be sorted
  auto origins = get_origins(grid);
  auto other_origins = get_origins(other);
  std::set_intersection(origins.begin(), origins.end(), other_origins.begin(),
                        other_origins.end(), std::back_inserter(out));
  return !out.empty();
};

auto read_vdb_float = [](std::string fn) {
  auto base_grid = read_vdb_file(fn);
  auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
#ifdef LOG
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

#ifdef LOG
    print_grid_metadata(first_grid);
#endif
  }

  openvdb::GridPtrVec grids;
  grids.push_back(first_grid);
  write_vdb_file(grids, out);
};

auto get_id_map = []() {
  std::map<GridCoord, uint32> coord_to_swc_id;
  // add a dummy value that will never be on to the map so that real indices
  // start at 1
  coord_to_swc_id[GridCoord(INT_MIN, INT_MIN, INT_MIN)] = 0;
  return coord_to_swc_id;
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
    int pos =
        find(nX[i]->nbr.begin(), nX[i]->nbr.end(), i) - nX[i]->nbr.begin();
    if (pos >= 0 && pos < nX[i]->nbr.size())
      nX[i]->nbr.erase(nX[i]->nbr.begin() + pos); // remove at pos
  }

  // ensure linkings are bidirectional, add if not
  for (VID_t i = 0; i < nX.size(); ++i) {
    for (VID_t j = 0; j < nX[i]->nbr.size(); ++j) {
      if (i != j) {
        bool fnd = false;
        for (int k = 0; k < nX[nX[i]->nbr[j]]->nbr.size(); ++k) {
          if (nX[nX[i]->nbr[j]]->nbr[k] == i) {
            fnd = true;
            break;
          }
        }

        if (!fnd) {
          // enforce link
          nX[nX[i]->nbr[j]]->nbr.push_back(i);
          cout << "enforced bidirectional link: " << nX[i]->nbr[j] << " -- "
               << i << '\n';
        }
      }
    }
  }
};

// sphere grouping Advantra prune strategy
std::vector<MyMarker *> advantra_prune(vector<MyMarker *> nX) {

  std::vector<MyMarker *> nY;
  auto no_neighbor_count = 0;

  // nX[0].corr = FLT_MAX; // so that the dummy node gets index 0 again, larges
  // correlation
  vector<long> indices(nX.size());
  for (long i = 0; i < indices.size(); ++i)
    indices[i] = i;
  // TODO sort by float value if possible
  // sort(indices.begin(), indices.end(), CompareIndicesByNodeCorrVal(&nX));

  // translate a dense linear idx of X to the sparse linear idx of y
  vector<long> X2Y(nX.size(), -1);
  X2Y[0] = 0; // first one is with max. correlation

  nY.push_back(nX[0]);

  // add soma nodes as independent groups at the beginning
  for (long i = 0; i < nX.size(); ++i) {
    // all somas are automatically kept
    if (nX[i]->type == 0) {
      X2Y[i] = nY.size();
      auto nYi = new MyMarker(*nX[i]);
      nY.push_back(nYi);
    }
  }

  for (long i = 0; i < indices.size(); ++i) { // add the rest of the nodes

    long ci = indices[i];

    if (X2Y[ci] != -1)
      continue; // skip if it was added to a group already

    X2Y[ci] = nY.size();
    // create a new marker in the sparse set starting from an existing one
    // that has not been pruned yet
    auto nYi = new MyMarker(*(nX[ci]));
    float grp_size = 1;

    float r2 = GROUP_RADIUS * GROUP_RADIUS; // sig2rad * nX[ci].sig;
    // auto node_radius = nYi->radius;
    // auto node_radius_upsampled =  node_radius * 5; // for anisotropic images
    // float r2 = node_radius_upsampled * node_radius_upsampled;
    float d2;
    // TODO optimize this for closest point
    for (long j = 0; j < nX.size();
         ++j) { // check the rest that was not grouped
      if (j != ci && X2Y[j] == -1) {
        d2 = pow(nX[j]->x - nX[ci]->x, 2);
        if (d2 <= r2) {
          d2 += pow(nX[j]->y - nX[ci]->y, 2);
          if (d2 <= r2) {
            d2 += pow(nX[j]->z - nX[ci]->z, 2);
            if (d2 <= r2) {

              // mark the idx, since nY is being accumulated to
              X2Y[j] = nY.size();

              // TODO modify marker to have a set of markers
              for (int k = 0; k < nX[j]->nbr.size(); ++k) {
                nYi->nbr.push_back(
                    nX[j]
                        ->nbr[k]); // append the neighbours of the group members
              }

              // update local average with x,y,z,sig elements from nX[j]
              ++grp_size;
              float a = (grp_size - 1) / grp_size;
              float b = (1.0 / grp_size);
              nYi->x = a * nYi->x + b * nX[j]->x;
              nYi->y = a * nYi->y + b * nX[j]->y;
              nYi->z = a * nYi->z + b * nX[j]->z;
            }
          }
        }
      }
    }

    if (nYi->nbr.size() == 0) {
      ++no_neighbor_count;
      if (nYi->parent == nullptr)
        throw std::runtime_error("parent is also invalid");
      // cout << "nXi coord " << nX[ci]->x << ',' << nX[ci]->y << ',' <<
      // nX[ci]->z << '\n'; cout << "nYi coord " << nYi->x << ',' << nYi->y <<
      // ',' << nYi->z << '\n'; cout << "  parent coord " << nYi->parent->x <<
      // ',' << nYi->parent->y << ',' << nYi->parent->z << '\n';
    }

    // nYi.type = Node::AXON; // enforce type
    nY.push_back(nYi);
  }

  // once complete mapping is established, update the indices from
  // the original linear index to the new sparse group index according
  // to the X2Y idx map vector
  for (int i = 1; i < nY.size(); ++i) {
    for (int j = 0; j < nY[i]->nbr.size(); ++j) {
      nY[i]->nbr[j] = X2Y[nY[i]->nbr[j]];
    }
  }

  check_nbr(nY); // remove doubles and self-linkages after grouping

  cout << nY.size() << '\n';
  cout << no_neighbor_count << '\n';

  return nY;
}

template <typename T> class BfsQueue {
public:
  std::queue<T> kk;
  BfsQueue() {}
  void enqueue(T item) { this->kk.push(item); }
  T dequeue() {
    T output = kk.front();
    kk.pop();
    return output;
  }
  int size() { return kk.size(); }
  bool hasItems() { return !kk.empty(); }
};

// advantra based re-extraction of tree based on bfs
std::vector<MyMarker *>
advantra_extract_trees(std::vector<MyMarker *> nlist,
                       bool remove_isolated_tree_with_one_node = false) {

  BfsQueue<int> q;
  std::vector<MyMarker *> tree;

  vector<int> dist(nlist.size());
  vector<int> nmap(nlist.size());
  vector<int> parent(nlist.size());

  for (int i = 0; i < nlist.size(); ++i) {
    dist[i] = INT_MAX;
    nmap[i] = -1;   // indexing in output tree
    parent[i] = -1; // parent index in current tree
  }

  dist[0] = -1;

  // Node tree0(nlist[0]); // first element of the nodelist is dummy both in
  // input and output tree.clear(); tree.push_back(tree0);
  int treecnt = 0; // independent tree counter, will be obsolete

  int seed;

  auto get_undiscovered2 = [](std::vector<int> dist) -> int {
    for (int i = 1; i < dist.size(); i++) {
      if (dist[i] == INT_MAX) {
        return i;
      }
    }
    return -1;
  };

  while ((seed = get_undiscovered2(dist)) > 0) {

    treecnt++;

    dist[seed] = 0;
    nmap[seed] = -1;
    parent[seed] = -1;
    q.enqueue(seed);

    int nodesInTree = 0;

    while (q.hasItems()) {

      // dequeue(), take from FIFO structure,
      // http://en.wikipedia.org/wiki/Queue_%28abstract_data_type%29
      int curr = q.dequeue();

      auto n = new MyMarker(*nlist[curr]);
      n->nbr.clear();
      if (n->type != 0)
        n->type = treecnt + 2; // vaa3d viz

      // choose the best single parent of the possible neighbors
      if (parent[curr] > 0) {
        n->nbr.push_back(nmap[parent[curr]]);
        // get the ptr to the marker from the id of the parent of the current
        n->parent = nlist[parent[curr]];
      } else if (nlist[curr]->nbr.size() != 0) {
        // get the ptr to the marker from the id of the min element of nbr
        // the smaller the id of an element, the higher precedence it has
        auto nbrs = nlist[curr]->nbr;
        auto min_idx = 0;
        auto min_element = nbrs[min_idx];
        for (int i = 0; i < nbrs.size(); ++i) {
          if (nbrs[i] < min_element) {
            min_element = nbrs[i];
          }
        }
        n->parent = nlist[min_element];
      } else {
        throw std::runtime_error("node can't have 0 nbrs");
      }

      nmap[curr] = tree.size();
      tree.push_back(n);
      ++nodesInTree;

      // for each node adjacent to current
      for (int j = 0; j < nlist[curr]->nbr.size(); j++) {

        int adj = nlist[curr]->nbr[j];

        if (dist[adj] == INT_MAX) {
          dist[adj] = dist[curr] + 1;
          parent[adj] = curr;
          // enqueue(), add to FIFO structure,
          // http://en.wikipedia.org/wiki/Queue_%28abstract_data_type%29
          q.enqueue(adj);
        }
      }

      // check if there were any neighbours
      if (nodesInTree == 1 && !q.hasItems() &&
          remove_isolated_tree_with_one_node) {
        tree.pop_back(); // remove the one that was just added
        nmap[curr] = -1; // cancel the last entry
      }
    }
  }

  cout << treecnt << " trees\n";
  return tree;
}

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
  std::map<GridCoord, MyMarker *> coord_to_marker_ptr;
  std::map<GridCoord, VID_t> coord_to_idx;

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
    [](const std::vector<std::pair<GridCoord, uint8_t>> &roots,
       EnlargedPointDataGrid::Ptr grid) -> EnlargedPointDataGrid::Ptr {
  // Iterate over leaf nodes that contain topology (active)
  // checking for roots within them
  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    auto leaf_bbox = leaf_iter->getNodeBoundingBox();

    // FILTER for those in this leaf
    auto leaf_roots =
        roots | rng::views::remove_if([&leaf_bbox](const auto &coord_radius) {
          const auto [coord, radius] = coord_radius;
          return !leaf_bbox.isInside(coord);
        }) |
        rng::to_vector;

    if (leaf_roots.empty())
      continue;

   std::cout << "Leaf BBox: " << leaf_bbox << '\n';

    rng::for_each(leaf_roots, [&leaf_iter, &grid](const auto &coord_radius) {
      const auto [coord, radius] = coord_radius;
      assertm(leaf_iter, "corresponding leaf of passed root must be active");
      auto ind = leaf_iter->beginIndexVoxel(coord);
      assertm(ind, "corresponding voxel of passed root must be active");

      // modify the radius value
      openvdb::points::AttributeWriteHandle<float> radius_handle(
          leaf_iter->attributeArray("pscale"));

      auto previous_radius = radius_handle.get(*ind);
      radius_handle.set(*ind, radius);

#ifdef FULL_PRINT
      cout << "Adjusted " << coord << " radius " << previous_radius << " -> "
           << radius_handle.get(*ind) << '\n';
#endif
    });
  }

  return grid;
};

// modify the radius value within the point grid to reflect a predetermined
// radius mutates the value that was previously calculated usually somas have a
// more accurate radius value determined before Recut processes so it is
// appropriate to rewrite with these more accurate radii values before a prune
// step
auto adjust_soma_radii_deprecated =
    [](const std::vector<std::pair<GridCoord, uint8_t>> &roots,
       EnlargedPointDataGrid::Ptr grid) -> EnlargedPointDataGrid::Ptr {
  rng::for_each(roots, [grid](const auto &coord_radius) {
    const auto [coord, radius] = coord_radius;
    const auto leaf = grid->tree().probeLeaf(coord);
    assertm(leaf, "corresponding leaf of passed root must be active");
    auto ind = leaf->beginIndexVoxel(coord);
    assertm(ind, "corresponding voxel of passed root must be active");

    // modify the radius value
    openvdb::points::AttributeWriteHandle<float> radius_handle(
        leaf->attributeArray("pscale"));

    auto previous_radius = radius_handle.get(*ind);
    radius_handle.set(*ind, radius);

#ifdef FULL_PRINT
    cout << "Adjusted " << coord << " radius " << previous_radius << " -> "
         << radius_handle.get(*ind) << '\n';
#endif
  });

  return grid;
};
