#pragma once

#include "app2_helpers.hpp"
#include "mesh_reconstruction.hpp"
#include "morphological_soma_segmentation.hpp"
#include "recut_parameters.hpp"
#include "tile_thresholds.hpp"
#include "tree_ops.hpp"
#include "utils.hpp"
#include <algorithm>
#include <bits/stdc++.h>
#include <bitset>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <execution>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <openvdb/tools/LevelSetRebuild.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tbb/global_control.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_set>

template <typename... Ts> struct Overload : Ts... { using Ts::operator()...; };
template <class... Ts> Overload(Ts...) -> Overload<Ts...>;

using ThreshV =
    std::variant<TileThresholds<uint8_t> *, TileThresholds<uint16_t> *>;
using HistV = std::variant<Histogram<uint8_t>, Histogram<uint16_t>>;
using TileV =
    std::variant<std::unique_ptr<vto::Dense<uint16_t, vto::LayoutXYZ>>,
                 std::unique_ptr<vto::Dense<uint8_t, vto::LayoutXYZ>>>;

struct InstrumentedUpdateStatistics {
  int iterations;
  double total_time;
  double computation_time;
  double io_time;
  std::vector<uint16_t> tile_open_counts;

  // These refer to the FIFO sizes
  std::vector<uint64_t> max_sizes;
  std::vector<uint64_t> mean_sizes;

  InstrumentedUpdateStatistics(int iterations, double total_time,
                               double computation_time, double io_time,
                               std::vector<uint16_t> tile_open_counts)
      : iterations(iterations), total_time(total_time),
        computation_time(computation_time), io_time(io_time),
        tile_open_counts(tile_open_counts) {}
};

template <class image_t> class Recut {
public:
  // Member Naming Conventions
  // length: number of items typically along a 3D dimension or linear
  //    data-structure such as a list or array
  // count: total instances of that item; can also be dynamic or unpredictable
  //    total occurences; also useful for number of items in unordered data
  //    structures
  // size: number of total items within that unit, typically this number
  //    is fixed / const and represents the product of various length variables
  //    and is used as short hand to prevent recomputation of same values
  // coordinates: come at the end of variable name to aid discoverability and
  //    auto-complete
  // hierarchical: if the variable represents the number of item2 per item1
  //    then the naming follows item1_item2_... descending in precedence
  //    since the number is more of a property of item1, since item2 may
  //    be unaware of such grouping
  // n, num, number: these terms are avoided due to ambiguity
  // iterables: are given plural names where possible to aid "for each" clarity
  // vid, id: unique identifier of an instance
  // idx: index for example in looping variables
  //
  // Example:
  // A 3D image has a dimension of this->image_lengths[0],
  // this->image_lengths[1], this->image_lengths[2]. Therefore image_size =
  // this->image_lengths[0] * this->image_lengths[1] * this->image_lengths[2].
  // If the program were keeping track of multiple images then the variable
  // image_count would record that number
  VID_t image_size, grid_tile_size, tile_block_size;
  openvdb::math::CoordBBox image_bbox;

  GridCoord image_lengths;
  GridCoord image_offsets;
  GridCoord tile_lengths;
  GridCoord block_lengths;
  GridCoord grid_tile_lengths;
  GridCoord tile_block_lengths;

  bool input_is_vdb;
  EnlargedPointDataGrid::Ptr topology_grid;
  openvdb::FloatGrid::Ptr input_grid;
  openvdb::BoolGrid::Ptr update_grid;
  openvdb::MaskGrid::Ptr mask_grid;
  ImgGrid::Ptr img_grid;

  std::vector<OffsetCoord> const lower_stencil{new_offset_coord(0, 0, -1),
                                               new_offset_coord(0, -1, 0),
                                               new_offset_coord(-1, 0, 0)};

  std::vector<OffsetCoord> const higher_stencil{new_offset_coord(0, 0, 1),
                                                new_offset_coord(0, 1, 0),
                                                new_offset_coord(1, 0, 0)};

  std::vector<OffsetCoord> const stencil{
      new_offset_coord(0, 0, -1), new_offset_coord(0, 0, 1),
      new_offset_coord(0, -1, 0), new_offset_coord(0, 1, 0),
      new_offset_coord(-1, 0, 0), new_offset_coord(1, 0, 0)};

  std::ofstream out;
  RecutCommandLineArgs *args;
  std::map<GridCoord, std::deque<VertexAttr>> map_fifo;
  std::map<GridCoord, local_heap> heap_map;
  std::map<GridCoord, std::deque<VertexAttr>> connected_map;

  // tile specific global data structures
  vector<bool> active_tiles;
  std::filesystem::path run_dir;
  std::filesystem::path log_fn;

  Recut(RecutCommandLineArgs &args) : args(&args) {}

  void operator()();
  void start_run_dir_and_logs();
  // void print_to_swc(std::string swc_path);
  void adjust_parent();
  void prune_radii();
  void prune_branch();
  void convert_topology();

  void partition_components(std::vector<Seed> seeds, bool prune);

  void initialize_globals(const VID_t &grid_tile_size,
                          const VID_t &tile_block_size);

  bool filter_by_label(VertexAttr *v, bool accept_tombstone);

  image_t get_img_val(const image_t *tile, GridCoord coord);
  inline VID_t rotate_index(VID_t img_coord, const VID_t current,
                            const VID_t neighbor, const VID_t tile_block_size,
                            const VID_t pad_block_size);
  template <typename T>
  void place_vertex(const VID_t nb_tile_id, VID_t block_id, VID_t nb,
                    struct VertexAttr *dst, GridCoord dst_coord,
                    OffsetCoord msg_offsets, std::string stage, T vdb_accessor);
  bool any_fifo_active(std::map<GridCoord, std::deque<VertexAttr>> &check_fifo);
  bool are_tiles_finished();
  void activate_all_tiles();

  // indexing helpers that use specific global lengths from recut
  GridCoord id_tile_to_img_offsets(const VID_t tile_id);
  GridCoord id_block_to_tile_offsets(const VID_t block_id);
  inline VID_t id_img_to_block_id(const VID_t id);
  VID_t id_img_to_tile_id(const VID_t id);
  template <typename T> VID_t coord_img_to_block_id(T coord);
  template <typename T> VID_t coord_img_to_tile_id(T coord);
  GridCoord id_tile_block_to_img_offsets(VID_t tile_id, VID_t block_id);
  VID_t v_to_img_id(VID_t tile_id, VID_t block_id, VertexAttr *v);
  GridCoord v_to_img_coord(VID_t tile_id, VID_t block_id, VertexAttr *v);
  OffsetCoord v_to_off(VID_t tile_id, VID_t block_id, VertexAttr *v);

  template <typename T>
  void check_ghost_update(VID_t tile_id, VID_t block_id, GridCoord dst_coord,
                          VertexAttr *dst, std::string stage, T vdb_accessor);
  template <typename IndT, typename FlagsT, typename ParentsT, typename ValueT,
            typename PointIter, typename UpdateIter>
  bool accumulate_value(const image_t *tile, VID_t tile_id, VID_t block_id,
                        GridCoord dst_coord, IndT dst_ind,
                        OffsetCoord offset_to_current, VID_t &revisits,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_adjacent_invalid, PointIter point_leaf,
                        UpdateIter update_leaf, FlagsT flags_handle,
                        ParentsT parents_handle, ValueT value_handle,
                        GridCoord current_coord, IndT current_ind,
                        float current_vox);
  template <typename T2, typename FlagsT, typename ParentsT, typename PointIter,
            typename UpdateIter>
  bool accumulate_connected(const image_t *tile, VID_t tile_id, VID_t block_id,
                            GridCoord dst_coord, T2 ind,
                            OffsetCoord offset_to_current, VID_t &revisits,
                            const TileThresholds<image_t> *tile_thresholds,
                            bool &found_adjacent_invalid, PointIter point_leaf,
                            UpdateIter update_leaf, FlagsT flags_handle,
                            ParentsT parents_handle,
                            std::deque<VertexAttr> &connected_fifo);
  bool is_covered_by_parent(VID_t tile_id, VID_t block_id, VertexAttr *current);
  template <class Container, typename IndT, typename RadiusT, typename FlagsT,
            typename UpdateIter>
  void accumulate_prune(VID_t tile_id, VID_t block_id, GridCoord dst_coord,
                        IndT ind, const uint8_t current_radius,
                        bool current_unvisited, Container &fifo,
                        RadiusT radius_handle, FlagsT flags_handle,
                        UpdateIter update_leaf);
  template <class Container, typename T, typename FlagsT, typename RadiusT,
            typename UpdateIter>
  void accumulate_radius(VID_t tile_id, VID_t block_id, GridCoord dst_coord,
                         T ind, const uint8_t current_radius, Container &fifo,
                         FlagsT flags_handle, RadiusT radius_handle,
                         UpdateIter update_leaf);
  void integrate_update_grid(
      EnlargedPointDataGrid::Ptr grid,
      vt::LeafManager<PointTree> grid_leaf_manager, std::string stage,
      std::map<GridCoord, std::deque<VertexAttr>> &fifo,
      std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo,
      openvdb::BoolGrid::Ptr update_grid, VID_t tile_id);
  template <class Container> void dump_buffer(Container buffer);
  template <typename T2>
  void march_narrow_band(const image_t *tile, VID_t tile_id, VID_t block_id,
                         std::string stage,
                         const TileThresholds<image_t> *tile_thresholds,
                         std::deque<VertexAttr> &connected_fifo,
                         std::deque<VertexAttr> &fifo, T2 leaf_iter);
  template <class Container, typename T2>
  void value_tile(const image_t *tile, VID_t tile_id, VID_t block_id,
                  std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T2 leaf_iter);
  template <class Container, typename T2>
  void connected_tile(const image_t *tile, VID_t tile_id, VID_t block_id,
                      std::string stage,
                      const TileThresholds<image_t> *tile_thresholds,
                      Container &connected_fifo, Container &fifo,
                      VID_t revisits, T2 leaf_iter);
  template <class Container, typename T2>
  void radius_tile(const image_t *tile, VID_t tile_id, VID_t block_id,
                   std::string stage,
                   const TileThresholds<image_t> *tile_thresholds,
                   Container &fifo, VID_t revisits, T2 leaf_iter);
  template <class Container, typename T2>
  void prune_tile(const image_t *tile, VID_t tile_id, VID_t block_id,
                  std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T2 leaf_iter);
  void create_march_thread(VID_t tile_id, VID_t block_id);
  template <typename local_image_t>
  TileThresholds<local_image_t> *get_tile_thresholds(local_image_t *tile,
                                                     int tile_vertex_size);
  template <typename T2>
  std::atomic<double>
  process_tile(VID_t tile_id, const image_t *tile, std::string stage,
               const TileThresholds<image_t> *tile_thresholds, T2 vdb_accessor);
  template <typename T1, typename T2, typename T3, typename T4>
  void io_tile(int tile_id, T1 &grids, T2 &uint8_grids, T3 &float_grids,
               T4 &mask_grids, std::string stage, HistV &histogram);
  template <class Container>
  void update(std::string stage, Container &fifo = nullptr);
  GridCoord get_input_image_lengths(RecutCommandLineArgs *args);
  void initialize();
  void update_hierarchical_dims(const GridCoord &tile_lengths);
  inline VID_t sub_block_to_block_id(VID_t iblock, VID_t jblock, VID_t kblock);
  template <class Container> void setup_radius(Container &fifo);
  void
  activate_vids(EnlargedPointDataGrid::Ptr grid, const std::vector<Seed> &roots,
                const std::string stage,
                std::map<GridCoord, std::deque<VertexAttr>> &fifo,
                std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo);
  void set_parent_non_branch(const VID_t tile_id, const VID_t block_id,
                             VertexAttr *dst, VertexAttr *potential_new_parent);
};

template <class image_t>
GridCoord Recut<image_t>::id_tile_to_img_offsets(const VID_t tile_id) {
  return coord_add(this->image_offsets,
                   coord_prod(id_to_coord(tile_id, this->grid_tile_lengths),
                              this->tile_lengths));
}

template <class image_t>
GridCoord Recut<image_t>::id_block_to_tile_offsets(const VID_t block_id) {
  // return coord_add(
  // this->image_offsets,
  return coord_prod(id_to_coord(block_id, this->tile_block_lengths),
                    this->block_lengths);
}

/**
 * returns tile_id, the tile domain this vertex belongs to
 * does not consider overlap of ghost regions
 */
template <class image_t>
template <typename T>
VID_t Recut<image_t>::coord_img_to_tile_id(T coord) {
  return coord_to_id(coord_div(coord, this->tile_lengths),
                     this->grid_tile_lengths);
}

/**
 * id : linear idx relative to unpadded image
 * returns tile_id, the tile domain this vertex belongs to
 * with respect to the original unpadded image
 * does not consider overlap of ghost regions
 */
template <class image_t>
VID_t Recut<image_t>::id_img_to_tile_id(const VID_t id) {
  return coord_img_to_tile_id(id_to_coord(id, this->image_lengths));
}

template <class image_t>
template <typename T>
VID_t Recut<image_t>::coord_img_to_block_id(T coord) {
  return coord_to_id(
      coord_div(coord_mod(coord, this->tile_lengths), this->block_lengths),
      this->tile_block_lengths);
}

/**
 * all block_nums are a linear row-wise idx, relative to current tile
 * vid : linear idx into the full domain inimg1d
 * the tile contributions are modded away
 * such that all block_nums are relative to a single
 * tile
 * returns block_id, the block domain this vertex belongs
 * in one of the tiles
 * Note: block_nums are renumbered within each tile
 * does not consider overlap of ghost regions
 */
template <class image_t>
VID_t Recut<image_t>::id_img_to_block_id(const VID_t vid) {
  return coord_img_to_block_id(id_to_coord(vid, this->image_lengths));
}

// first coord of block with respect to whole image
template <typename image_t>
GridCoord Recut<image_t>::id_tile_block_to_img_offsets(VID_t tile_id,
                                                       VID_t block_id) {
  return coord_add(id_tile_to_img_offsets(tile_id),
                   id_block_to_tile_offsets(block_id));
}

// recompute the vertex vid
template <typename image_t>
GridCoord Recut<image_t>::v_to_img_coord(VID_t tile_id, VID_t block_id,
                                         VertexAttr *v) {
  return coord_add(v->offsets, id_tile_block_to_img_offsets(tile_id, block_id));
}

// recompute the vertex vid
template <typename image_t>
VID_t Recut<image_t>::v_to_img_id(VID_t tile_id, VID_t block_id,
                                  VertexAttr *v) {
  return coord_to_id(v_to_img_coord(tile_id, block_id, v));
}

// recompute the vertex vid
template <typename image_t>
OffsetCoord Recut<image_t>::v_to_off(VID_t tile_id, VID_t block_id,
                                     VertexAttr *v) {
  return coord_mod(v_to_img_coord(tile_id, block_id, v), this->block_lengths);
}

// activates
// the tiles of the leaf and reads
// them to the respective heaps
template <class image_t>
template <class Container>
void Recut<image_t>::setup_radius(Container &fifo) {
  for (size_t tile_id = 0; tile_id < grid_tile_size; ++tile_id) {
    active_tiles[tile_id] = true;
#ifdef FULL_PRINT
    cout << "Set tile " << tile_id << " to active\n";
#endif
  }
}

template <class image_t>
void Recut<image_t>::activate_vids(
    EnlargedPointDataGrid::Ptr grid, const std::vector<Seed> &seeds,
    const std::string stage, std::map<GridCoord, std::deque<VertexAttr>> &fifo,
    std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo) {

  assertm(!(seeds.empty()), "Must have at least one seed");

  // Iterate over leaf nodes that contain topology (active)
  // checking for seed within them
  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    auto leaf_bbox = leaf_iter->getNodeBoundingBox();
    // std::cout << "Leaf BBox: " << leaf_bbox << '\n';

    // FILTER for those in this leaf
    auto leaf_seed_coords = seeds | rv::transform(&Seed::coord) |
                            rv::remove_if([&leaf_bbox](GridCoord coord) {
                              return !leaf_bbox.isInside(coord);
                            }) |
                            rng::to_vector;

    if (leaf_seed_coords.empty())
      continue;

    this->active_tiles[0] = true;

    // Set Values
    auto update_leaf = this->update_grid->tree().probeLeaf(leaf_bbox.min());
    assertm(update_leaf, "Update must have a corresponding leaf");

    rng::for_each(leaf_seed_coords, [&update_leaf](auto coord) {
      // this only adds to update_grid if the root happens
      // to be on a boundary
      set_if_active(update_leaf, coord);
    });

    // Extract the position attribute from the leaf by name (P is position).
    const openvdb::points::AttributeArray &array =
        leaf_iter->constAttributeArray("P");
    // Create a read-only AttributeHandle. Position always uses Vec3f.
    openvdb::points::AttributeHandle<PositionT> position_handle(array);

    openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
        leaf_iter->attributeArray("flags"));

    openvdb::points::AttributeWriteHandle<OffsetCoord> parents_handle(
        leaf_iter->attributeArray("parents"));

    openvdb::points::AttributeWriteHandle<float> radius_handle(
        leaf_iter->attributeArray("pscale"));

    auto temp_coord = new_grid_coord(LEAF_LENGTH, LEAF_LENGTH, LEAF_LENGTH);

    if (stage == "connected" || stage == "value") {

      rng::for_each(leaf_seed_coords, [&](auto coord) {
        auto ind = leaf_iter->beginIndexVoxel(coord);
        if (this->args->input_type == "float") {
          assertm(this->input_grid->tree().isValueOn(coord),
                  "All root coords must be filtered with respect to topology");
        }
        assertm(ind,
                "All root coords must be filtered with respect to topology");

        // place a root with proper vid and parent of itself
        // set flags as root
        set_selected(flags_handle, ind);
        set_root(flags_handle, ind);
        parents_handle.set(*ind, zeros_off());

        auto offsets = coord_mod(coord, temp_coord);
        if (stage == "connected") {
          connected_fifo[leaf_iter->origin()].emplace_back(
              /*edge_state*/ flags_handle.get(*ind), offsets, zeros_off());
        } else if (stage == "value") {
          auto block_id = coord_img_to_block_id(leaf_iter->origin());
          // ignore the input radius size, use the root flags set above
          auto rootv =
              new VertexAttr(flags_handle.get(*ind),
                             /* offsets*/ coord_sub(coord, leaf_bbox.min()), 0);
          heap_map[leaf_iter->origin()].push(rootv, block_id, stage);
        }
      });

    } else if (stage == "prune") {

      rng::for_each(leaf_seed_coords, [&](auto coord) {
        auto ind = leaf_iter->beginIndexVoxel(coord);
        assertm(ind,
                "All seed coords must be filtered with respect to topology");
        auto offsets = coord_mod(coord, temp_coord);
        set_prune_visited(flags_handle, ind);
        fifo[leaf_iter->origin()].emplace_back(
            /*edge_state*/ flags_handle.get(*ind), offsets, zeros_off());
      });
    }
  }
}

template <typename T, typename T2> T absdiff(const T &lhs, const T2 &rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

/*
 * Takes a grid coord
 * and converts it to linear index of current
 * tile buffer of tile currently processed
 * returning the value
 */
template <class image_t>
image_t Recut<image_t>::get_img_val(const image_t *tile, GridCoord coord) {
  auto buffer_coord =
      coord_to_id(coord_mod(coord, this->tile_lengths), this->tile_lengths);
  return tile[buffer_coord];
}

/*
template <typename image_t>
bool Recut<image_t>::is_covered_by_parent(VID_t tile_id, VID_t block_id,
                                          VertexAttr *current) {
  VID_t i, j, k, pi, pj, pk;
  get_img_coord(current->vid, i, j, k);
  get_img_coord(current->parent, pi, pj, pk);
  auto x = static_cast<double>(i) - pi;
  auto y = static_cast<double>(j) - pj;
  auto z = static_cast<double>(k) - pk;
  auto vdistance = sqrt(x * x + y * y + z * z);

  auto parent_tile_id = id_img_to_tile_id(current->parent);
  auto parent_block_id = id_img_to_block_id(current->parent);

  auto parent =
      get_active_vertex(parent_tile_id, parent_block_id, current->parent);

  if (static_cast<double>(parent->radius) >= vdistance) {
    return true;
  }
  return false;
}

template <typename image_t>
void Recut<image_t>::set_parent_non_branch(const VID_t tile_id,
                                           const VID_t block_id,
                                           VertexAttr *dst,
                                           VertexAttr *potential_new_parent) {
  // roots never need to have their parents set
  // it will always be to themsevles
  if (!(dst->root())) {
    while (potential_new_parent->is_branch_point()) {
      // move up the tree until you find a non
      // branch point or the root itself
      if (potential_new_parent->root()) {
        break;
      }
      assertm(potential_new_parent->valid_parent(),
              "potential_new_parent does not have valid parent");
      potential_new_parent = get_active_vertex(tile_id, block_id,
                                               potential_new_parent->parent);
    }
    assertm(!(potential_new_parent->is_branch_point()),
            "can not assign child to a vertex with already 2 children unless "
            "it is a root");

    auto parent_coord = id_to_coord(potential_new_parent->vid, image_lengths);
    // new parent is guaranteed to not be a branch point now
    if (potential_new_parent->root()) {
      dst->set_parent(potential_new_parent->vid);
    } else if (potential_new_parent->has_single_child()) {
      potential_new_parent->mark_branch_point();
      dst->set_parent(potential_new_parent->vid);
    } else {
      potential_new_parent->mark_has_single_child();
      dst->set_parent(potential_new_parent->vid);
    }
  }
}
*/

template <class image_t>
template <class Container, typename IndT, typename RadiusT, typename FlagsT,
          typename UpdateIter>
void Recut<image_t>::accumulate_prune(
    VID_t tile_id, VID_t block_id, GridCoord dst_coord, IndT ind,
    const uint8_t current_radius, bool current_unvisited, Container &fifo,
    RadiusT radius_handle, FlagsT flags_handle, UpdateIter update_leaf) {
  if (ind && is_selected(flags_handle, ind)) {
#ifdef FULL_PRINT
    std::cout << "\tcheck foreground dst: " << coord_to_str(dst_coord) << '\n';
#endif
    auto offset = coord_mod(dst_coord, this->block_lengths);

    auto add_prune_dst = [&]() {
      set_prune_visited(flags_handle, ind);
      fifo.emplace_back(flags_handle.get(*ind), offset,
                        radius_handle.get(*ind));
      set_if_active(update_leaf, dst_coord);
#ifdef FULL_PRINT
      std::cout << "  added dst " << dst_coord << " rad "
                << +(radius_handle.get(*ind)) << '\n';
#endif
    };

    // check if dst is covered by current
    // dst can only be 1 hop away (adjacent) from current, therefore
    // all radii greater than 1 imply some redundancy in coverage
    // but this may be desired with DILATION_FACTOR higher than 1
    if (current_radius >= DILATION_FACTOR) {
      auto dst_was_updated = false;
      // dst itself can be used to pass messages
      // like modified radius and prune status
      // to other blocks / tiles
      uint8_t update_radius;
      if (current_radius) // protect underflow
        update_radius = current_radius - 1;
      else
        update_radius = 0;
      if ((!is_prune_visited(flags_handle, ind)) &&
          (update_radius < radius_handle.get(*ind))) {
        // previously pruned vertex can transmits transitive
        // coverage info
        radius_handle.set(*ind, update_radius);
        assertm(radius_handle.get(*ind) == update_radius,
                "radii doesn't match");
        dst_was_updated = true;
      }

      // dst should be covered by current
      // if it hasn't already by pruned
      if (!(is_root(flags_handle, ind) || is_tombstone(flags_handle, ind))) {
        set_tombstone(flags_handle, ind);
        dst_was_updated = true;
      }

      if (dst_was_updated) {
#ifdef FULL_PRINT
        std::cout << "\tcurrent covers radius of: "
                  << +(radius_handle.get(*ind)) << " at dst " << dst_coord
                  << " " << +(flags_handle.get(*ind));
#endif
        add_prune_dst();
      }
    } else {

      // even if dst is not covered if it's already been
      // pruned or visited there's no more work to do
      if (!(is_tombstone(flags_handle, ind) ||
            is_prune_visited(flags_handle, ind))) {
        add_prune_dst();
      }
    }
  }
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current tile_id and block_id, since
 * it is potentially adding these vertices to the unique heap of tile_id
 * and block_id. only one parent when selected. If one of these vertexes on
 * the edge but still within tile_id and block_id domain is updated it
 * is the responsibility of check_ghost_update to take note of the update such
 * that this update is propagated to the relevant tile and block see
 * integrate_update_grid(). dst_coord : continuous vertex id VID_t of the dst
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <class Container, typename T, typename FlagsT, typename RadiusT,
          typename UpdateIter>
void Recut<image_t>::accumulate_radius(VID_t tile_id, VID_t block_id,
                                       GridCoord dst_coord, T ind,
                                       const uint8_t current_radius,
                                       Container &fifo, FlagsT flags_handle,
                                       RadiusT radius_handle,
                                       UpdateIter update_leaf) {
  // although current vertex can belong in the boundary
  // region of a separate block /tile it must be only
  // 1 voxel away (within this block /tile's ghost region)
  // therefore all neighbors / destinations of current
  // must be checked to make sure they protude into
  // the actual current block / tile region
  if (ind && is_selected(flags_handle, ind)) {
#ifdef FULL_PRINT
    std::cout << "\tcheck foreground dst: " << coord_to_str(dst_coord) << '\n';
#endif

    const uint8_t updated_radius = 1 + current_radius;
    auto dst_radius = radius_handle.get(*ind);

    // if radius not set yet it necessitates it is 1 higher than current OR an
    // update from another block / tile creates new lower updates
    if ((dst_radius == 0) || (dst_radius > updated_radius)) {
#ifdef FULL_PRINT
      cout << "\tAdd dst " << coord_to_str(dst_coord) << " label "
           << +(flags_handle.get(*ind)) << " radius " << +(dst_radius)
           << " from current radius " << +(current_radius) << '\n';
#endif
      radius_handle.set(*ind, updated_radius);

      // construct a dst message
      fifo.emplace_back(flags_handle.get(*ind),
                        coord_mod(dst_coord, this->block_lengths),
                        updated_radius);
      set_if_active(update_leaf, dst_coord);
    }
  }
}

/**
 * accumulate is the smallest scope function of fast marching, it can only
 * operate on a VertexAttr (voxel) that are within the current tile_id and
 * block_id, since it is potentially adding these vertices to the unique heap of
 * tile_id and block_id. only one parent when selected. If one of these
 * vertexes on the edge but still within tile_id and block_id domain is
 * updated it is the responsibility of check_ghost_update to take note of the
 * update such that this update is propagated to the relevant tile and block
 * see vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <typename IndT, typename FlagsT, typename ParentsT, typename ValueT,
          typename PointIter, typename UpdateIter>
bool Recut<image_t>::accumulate_value(
    const image_t *tile, VID_t tile_id, VID_t block_id, GridCoord dst_coord,
    IndT dst_ind, OffsetCoord offset_to_current, VID_t &revisits,
    const TileThresholds<image_t> *tile_thresholds,
    bool &found_adjacent_invalid, PointIter point_leaf, UpdateIter update_leaf,
    FlagsT flags_handle, ParentsT parents_handle, ValueT value_handle,
    GridCoord current_coord, IndT current_ind, float current_vox) {

#ifdef FULL_PRINT
  cout << "\tcheck dst: " << coord_to_str(dst_coord);
  cout << " bkg_thresh " << +(tile_thresholds->bkg_thresh) << '\n';
#endif

  // skip backgrounds
  // the image voxel of this dst vertex is the primary method to exclude this
  // pixel/vertex for the remainder of all processing
  auto found_background = false;
  image_t dst_vox;
  if (this->input_is_vdb) {
    // input grid and point grid are synonymous in terms of active topology
    // on active point was already found to be foreground
    if (point_leaf->isValueOn(dst_coord)) {
      dst_vox = this->input_grid->tree().getValue(dst_coord);
    } else {
      found_background = true;
    }
  } else {
    dst_vox = get_img_val(tile, dst_coord);
    if (dst_vox <= tile_thresholds->bkg_thresh) {
      found_background = true;
    }
  }

  if (found_background) {
    found_adjacent_invalid = true;
#ifdef FULL_PRINT
    cout << "\t\tfailed tile_thresholds->bkg_thresh" << '\n';
#endif
    return false;
  }

  // current march values
  auto current_value = value_handle.get(*current_ind);
  auto dst_value = value_handle.get(*dst_ind);

  // solve for update value
  // dst_id linear idx relative to full image domain
  float updated_val_attr = static_cast<float>(
      current_value + (tile_thresholds->calc_weight(current_vox) +
                       tile_thresholds->calc_weight(dst_vox)) *
                          0.5);

  // cout << "updated value " << updated_val_attr << '\n';

  // check for updates according to criterion
  // starting background values are 0
  // traditionally background values should be INF or FLOAT_MAX
  // but 0s are more compressible and simpler
  // must check all are not root though since root has distance 0 by definition
  if (((dst_value == 0) && !is_root(flags_handle, dst_ind)) ||
      (dst_value > updated_val_attr)) {
    // all dsts are guaranteed within this domain
    // skip already selected vertices too
    if (is_selected(flags_handle, dst_ind)) {
      revisits += 1;
      return false;
    }

    // set parents, mark selected in topology
    set_selected(flags_handle, dst_ind);
    set_if_active(update_leaf, dst_coord);
    parents_handle.set(*dst_ind, offset_to_current);
    auto offset = coord_mod(dst_coord, this->block_lengths);

    // ensure traces a path back to root
    // TODO optimize away copy construction of the vert
    auto vert = new VertexAttr(Bitfield(flags_handle.get(*dst_ind)), offset,
                               /*parent*/ offset_to_current, updated_val_attr);
    heap_map[point_leaf->origin()].push(vert, block_id, "value");

#ifdef FULL_PRINT
    cout << "\tadded new dst to active set, vid: " << dst_coord << '\n';
#endif
    return true;
  }
  return false;
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current tile_id and block_id, since
 * it is potentially adding these vertices to the unique heap of tile_id
 * and block_id. only one parent when selected. If one of these vertexes on
 * the edge but still within tile_id and block_id domain is updated it
 * is the responsibility of check_ghost_update to take note of the update such
 * that this update is propagated to the relevant tile and block see
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <typename T2, typename FlagsT, typename ParentsT, typename PointIter,
          typename UpdateIter>
bool Recut<image_t>::accumulate_connected(
    const image_t *tile, VID_t tile_id, VID_t block_id, GridCoord dst_coord,
    T2 ind, OffsetCoord offset_to_current, VID_t &revisits,
    const TileThresholds<image_t> *tile_thresholds,
    bool &found_adjacent_invalid, PointIter point_leaf, UpdateIter update_leaf,
    FlagsT flags_handle, ParentsT parents_handle,
    std::deque<VertexAttr> &connected_fifo) {

#ifdef FULL_PRINT
  cout << "\tcheck dst: " << coord_to_str(dst_coord);
  cout << " bkg_thresh " << +(tile_thresholds->bkg_thresh) << '\n';
#endif

  // skip backgrounds
  // the image voxel of this dst vertex is the primary method to exclude this
  // pixel/vertex for the remainder of all processing
  auto found_background = false;
  if (this->input_is_vdb) {
    if (!point_leaf->isValueOn(dst_coord))
      found_background = true;
  } else {
    auto dst_vox = get_img_val(tile, dst_coord);
    if (dst_vox <= tile_thresholds->bkg_thresh) {
      found_background = true;
    }
  }

  if (found_background) {
    found_adjacent_invalid = true;
#ifdef FULL_PRINT
    cout << "\t\tfailed tile_thresholds->bkg_thresh" << '\n';
#endif
    return false;
  }

  // all dsts are guaranteed within this domain
  // skip already selected vertices too
  if (is_selected(flags_handle, ind)) {
    revisits += 1;
    return false;
  }

  // set parents, mark selected in topology
  set_selected(flags_handle, ind);
  set_if_active(update_leaf, dst_coord);
  parents_handle.set(*ind, offset_to_current);
  auto offset = coord_mod(dst_coord, this->block_lengths);

  // ensure traces a path back to root
  connected_fifo.emplace_back(Bitfield(flags_handle.get(*ind)), offset,
                              /*parent*/ offset_to_current);

#ifdef FULL_PRINT
  cout << "\tadded new dst to active set, vid: " << dst_coord << '\n';
#endif
  return true;
}

/*
 * this will place necessary updates towards regions in outside blocks
 * or tiles safely by leveraging update_grid
 */
template <class image_t>
template <typename T>
void Recut<image_t>::place_vertex(const VID_t nb_tile_id, const VID_t block_id,
                                  const VID_t nb_block_id,
                                  struct VertexAttr *dst, GridCoord dst_coord,
                                  OffsetCoord msg_offsets, std::string stage,
                                  T vdb_accessor) {
  active_tiles[nb_tile_id] = true;
  vdb_accessor.setValueOn(dst_coord);

#ifdef FULL_PRINT
  auto nb_block_coord = id_to_coord(nb_block_id, this->tile_block_lengths);
  cout << "\t\t\tplace_vertex(): tile " << nb_tile_id << " nb block "
       << coord_to_str(nb_block_coord) << " msg offsets "
       << coord_to_str(msg_offsets) << '\n';
#endif
}

/*
 * This function holds all the logic of whether the update of a vertex within
 * one tiles and blocks domain is adjacent to another tile and block.
 * If the vertex is covered by an adjacent region then it passes the vertex to
 * place_vertex for potential updating or saving. Assumes star stencil, no
 * diagonal connection in 3D this yields 6 possible block and or tile
 * connection corners.  block_id and tile_id are in linearly addressed
 * row-order. dst is always guaranteed to be within block_id and tile_id
 * region. dst has already been protected by global padding out of bounds from
 * guard in accumulate. This function determines if dst is in a border region
 * and which neighbor block / tile should be notified of adjacent change
 */
template <class image_t>
template <typename T>
void Recut<image_t>::check_ghost_update(VID_t tile_id, VID_t block_id,
                                        GridCoord dst_coord, VertexAttr *dst,
                                        std::string stage, T vdb_accessor) {
  auto dst_offsets = coord_mod(dst_coord, this->block_lengths);
  auto dst_tile_offsets = coord_mod(dst_coord, this->tile_lengths);
  auto block_coord = id_to_coord(block_id, this->tile_block_lengths);

#ifdef FULL_PRINT
  cout << "\t\tcheck_ghost_update(): on " << coord_to_str(dst_coord)
       << ", block " << coord_to_str(block_coord) << '\n';
#endif

  // check all 6 directions for possible ghost updates
  if (dst_offsets[0] == 0) {
    if (dst_coord[0] > 0) { // protect from image out of bounds
      VID_t nb = block_id - 1;
      VID_t nb_tile_id = tile_id; // defaults to current tile
      if (dst_tile_offsets[0] == 0) {
        nb_tile_id = tile_id - 1;
        // Convert block coordinates into linear index row-ordered
        nb = sub_block_to_block_id(this->tile_block_lengths[0] - 1,
                                   block_coord[1], block_coord[2]);
      }
      if ((nb >= 0) && (nb < tile_block_size)) // within valid block bounds
        place_vertex(nb_tile_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(this->block_lengths[0], dst_offsets[1],
                                      dst_offsets[2]),
                     stage, vdb_accessor);
    }
  }
  if (dst_offsets[1] == 0) {
    if (dst_coord[1] > 0) { // protect from image out of bounds
      VID_t nb = block_id - this->tile_block_lengths[0];
      VID_t nb_tile_id = tile_id; // defaults to current tile
      if (dst_tile_offsets[1] == 0) {
        nb_tile_id = tile_id - this->grid_tile_lengths[0];
        nb = sub_block_to_block_id(
            block_coord[0], this->tile_block_lengths[1] - 1, block_coord[2]);
      }
      if ((nb >= 0) && (nb < tile_block_size)) // within valid block bounds
        place_vertex(nb_tile_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], this->block_lengths[1],
                                      dst_offsets[2]),
                     stage, vdb_accessor);
    }
  }
  if (dst_offsets[2] == 0) {
    if (dst_coord[2] > 0) { // protect from image out of bounds
      VID_t nb =
          block_id - this->tile_block_lengths[0] * this->tile_block_lengths[1];
      VID_t nb_tile_id = tile_id; // defaults to current tile
      if (dst_tile_offsets[2] == 0) {
        nb_tile_id =
            tile_id - this->grid_tile_lengths[0] * this->grid_tile_lengths[1];
        nb = sub_block_to_block_id(block_coord[0], block_coord[1],
                                   this->tile_block_lengths[2] - 1);
      }
      if ((nb >= 0) && (nb < tile_block_size)) // within valid block bounds
        place_vertex(nb_tile_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], dst_offsets[1],
                                      this->block_lengths[2]),
                     stage, vdb_accessor);
    }
  }

  if (dst_offsets[2] == this->block_lengths[0] - 1) {
    if (dst_coord[2] <
        this->image_lengths[2] - 1) { // protect from image out of bounds
      VID_t nb =
          block_id + this->tile_block_lengths[0] * this->tile_block_lengths[1];
      VID_t nb_tile_id = tile_id; // defaults to current tile
      if (dst_tile_offsets[2] == this->tile_lengths[2] - 1) {
        nb_tile_id =
            tile_id + this->grid_tile_lengths[0] * this->grid_tile_lengths[1];
        nb = sub_block_to_block_id(block_coord[0], block_coord[1], 0);
      }
      if ((nb >= 0) && (nb < tile_block_size)) // within valid block bounds
        place_vertex(nb_tile_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], dst_offsets[1], -1),
                     stage, vdb_accessor);
    }
  }
  if (dst_offsets[1] == this->block_lengths[1] - 1) {
    if (dst_coord[1] <
        this->image_lengths[1] - 1) { // protect from image out of bounds
      VID_t nb = block_id + this->tile_block_lengths[0];
      VID_t nb_tile_id = tile_id; // defaults to current tile
      if (dst_tile_offsets[1] == this->tile_lengths[1] - 1) {
        nb_tile_id = tile_id + this->grid_tile_lengths[0];
        nb = sub_block_to_block_id(block_coord[0], 0, block_coord[2]);
      }
      if ((nb >= 0) && (nb < tile_block_size)) // within valid block bounds
        place_vertex(nb_tile_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], -1, dst_offsets[2]),
                     stage, vdb_accessor);
    }
  }
  if (dst_offsets[0] == this->block_lengths[2] - 1) {
    if (dst_coord[0] <
        this->image_lengths[0] - 1) { // protect from image out of bounds
      VID_t nb = block_id + 1;
      VID_t nb_tile_id = tile_id; // defaults to current tile
      if (dst_tile_offsets[0] == this->tile_lengths[0] - 1) {
        nb_tile_id = tile_id + 1;
        nb = sub_block_to_block_id(0, block_coord[1], block_coord[2]);
      }
      if ((nb >= 0) && (nb < tile_block_size)) // within valid block bounds
        place_vertex(nb_tile_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(-1, dst_offsets[1], dst_offsets[2]),
                     stage, vdb_accessor);
    }
  }
}

// adds to iterable but not to active vertices since its from outside domain
template <typename Container, typename T>
void integrate_point(std::string stage, Container &fifo, T &connected_fifo,
                     local_heap heap, GridCoord adj_coord,
                     EnlargedPointDataGrid::Ptr grid, OffsetCoord adj_offsets,
                     GridCoord potential_update) {
  // FIXME this might be slow to lookup every time
  // auto leaf_iter = grid->tree().probeConstLeaf(potential_update);
  auto adj_leaf_iter = grid->tree().probeConstLeaf(adj_coord);
  auto ind = adj_leaf_iter->beginIndexVoxel(adj_coord);

  openvdb::points::AttributeHandle<uint8_t> flags_handle(
      adj_leaf_iter->constAttributeArray("flags"));

  Bitfield bf{flags_handle.get(*ind)};
  // indicate that this is just a message
  bf.unset(0);

#ifdef FULL_PRINT
  // std::cout << "\tintegrate_point(): " << adj_coord << '\n';
  // std::cout << "\t\tpotential update: " << potential_update << ' '
  //<< adj_leaf_iter << '\n';
#endif

  if (stage == "connected") {
    // this doesn't speed up the runtime at all
    // openvdb::points::AttributeHandle<uint8_t> current_flags_handle(
    // leaf_iter->constAttributeArray("flags"));
    // auto potential_update_ind =
    // leaf_iter->beginIndexVoxel(potential_update); if
    // (!potential_update_ind) return; if (is_selected(current_flags_handle,
    // potential_update_ind)) return; // no work to do at
    //// this point

    openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
        adj_leaf_iter->constAttributeArray("parents"));
    connected_fifo.emplace_back(bf, adj_offsets, parents_handle.get(*ind));

  } else if (stage == "value") {
    // TODO pass value of vertex for shortest path
    openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
        adj_leaf_iter->constAttributeArray("parents"));
    auto point = new VertexAttr(
        flags_handle.get(*ind),
        /* offsets*/ coord_sub(adj_coord, adj_leaf_iter->origin()), 0);
    // auto block_id = coord_img_to_block_id(adj_leaf_iter->origin());
    heap.push(point, 0, stage);

  } else if (stage == "radius") {
    openvdb::points::AttributeHandle<float> radius_handle(
        adj_leaf_iter->constAttributeArray("pscale"));
    fifo.emplace_back(bf, adj_offsets, zeros(), radius_handle.get(*ind));

  } else if (stage == "prune") {
    openvdb::points::AttributeHandle<float> radius_handle(
        adj_leaf_iter->constAttributeArray("pscale"));
    fifo.emplace_back(bf, adj_offsets, zeros(), radius_handle.get(*ind));
  }
}

void integrate_adj_leafs(GridCoord start_coord,
                         std::vector<OffsetCoord> stencil_offsets,
                         openvdb::BoolGrid::Ptr update_grid,
                         std::deque<VertexAttr> &fifo,
                         std::deque<VertexAttr> &connected_fifo,
                         local_heap heap, std::string stage,
                         EnlargedPointDataGrid::Ptr grid, int offset_value) {
  // force evaluation by saving to vector to get desired side effects
  // from integrate_point
  auto _ = // from one corner find 3 adj leafs via 1 vox offset
      stencil_offsets | rv::transform([&start_coord](auto stencil_offset) {
        return std::pair{/*rel. offset*/ stencil_offset,
                         coord_add(start_coord, stencil_offset)};
      }) |
      // get the corresponding leaf from update grid
      rv::transform([update_grid](auto coord_pair) {
        return std::pair{/*rel. offset*/ coord_pair.first,
                         update_grid->tree().probeConstLeaf(coord_pair.second)};
      }) |
      // does adj leaf have any border topology?
      rv::remove_if([](auto leaf_pair) {
        if (leaf_pair.second) {
          return leaf_pair.second->isEmpty(); // any of values true
        } else {
          return true;
        }
      }) |
      // for each adjacent leaf with values on
      rv::transform([&](auto leaf_pair) {
        // cout << leaf_pair.first << '\n';
        // which dim is this leaf offset in, can only be in 1
        auto dim = -1;
        for (int i = 0; i < 3; ++i) {
          // the stencil offset at .first has only 1 non-zero
          if (leaf_pair.first[i]) {
            dim = i;
          }
        }

        // iterate all active topology in the adj leaf
        for (auto value_iter = leaf_pair.second->cbeginValueOn(); value_iter;
             ++value_iter) {
          // PERF this might not be most efficient way to get index
          auto adj_coord = value_iter.getCoord();
          // filter to voxels active *and* true
          // true means they were updated in this iteration
          if (leaf_pair.second->getValue(adj_coord)) {
            // actual offset within real adjacent leaf
            auto adj_offsets = coord_mod(adj_coord, GridCoord(LEAF_LENGTH));
            // offset with respect to current leaf
            // the offset dim gets a special value according to whether
            // it is a positive or negative offset
            adj_offsets[dim] = offset_value;

            // only use coords that are in the surface facing the current
            // block remove if it doesn't match
            if ((leaf_pair.first[dim] + start_coord[dim]) == adj_coord[dim]) {
              // find the adjacent vox back in the current leaf which touches
              // adj_coord
              auto potential_update = adj_coord;
              potential_update[dim] = start_coord[dim];
              integrate_point(stage, fifo, connected_fifo, heap, adj_coord,
                              grid, adj_offsets, potential_update);
            }
          }
        }
        return leaf_pair;
      }) |
      rng::to_vector; // force eval
}

/* Core exchange step of the fastmarching algorithm, this processes and
 * sets update_grid values to false while leaving bit mask active set untouched.
 * tiles and blocks can receive all of their updates from the current
 * iterations run of march_narrow_band safely to complete the iteration
 */
template <class image_t>
void Recut<image_t>::integrate_update_grid(
    EnlargedPointDataGrid::Ptr grid,
    vt::LeafManager<PointTree> grid_leaf_manager, std::string stage,
    std::map<GridCoord, std::deque<VertexAttr>> &fifo,
    std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo,
    openvdb::BoolGrid::Ptr update_grid, VID_t tile_id) {
  // gather all changes on 6 boundary leafs
  {
    auto integrate_range =
        [&, this](const openvdb::tree::LeafManager<
                  openvdb::points::PointDataTree>::LeafRange &range) {
          // for each leaf with active voxels i.e. containing topology
          for (auto leaf_iter = range.begin(); leaf_iter; ++leaf_iter) {
            // integrate_leaf(leaf_iter);
            auto bbox = leaf_iter->getNodeBoundingBox();

            // lower corner adjacents, have an offset at that dim of -1
            integrate_adj_leafs(bbox.min(), this->lower_stencil, update_grid,
                                fifo[leaf_iter->origin()],
                                connected_fifo[leaf_iter->origin()],
                                heap_map[leaf_iter->origin()], stage, grid, -1);
            // upper corner adjacents, have an offset at that dim equal to
            // leaf log2 dim
            integrate_adj_leafs(
                bbox.max(), this->higher_stencil, update_grid,
                fifo[leaf_iter->origin()], connected_fifo[leaf_iter->origin()],
                heap_map[leaf_iter->origin()], stage, grid, LEAF_LENGTH);
          }
        };

    tbb::parallel_for(grid_leaf_manager.leafRange(), integrate_range);
  }

  // set update_grid false, keeping active values intact on boundary for
  // the lifetime of the program for sparse checks
  {
    auto fill_range =
        [](const openvdb::tree::LeafManager<vb::BoolTree>::LeafRange &range) {
          // for each leaf with active voxels i.e. containing topology
          for (auto leaf_iter = range.begin(); leaf_iter; ++leaf_iter) {
            // FIXME probably a more efficient way (hierarchically?) to set
            // all to false
            leaf_iter->fill(false);
          }
        };

    auto timer = high_resolution_timer();
    openvdb::tree::LeafManager<vb::BoolTree> update_grid_leaf_manager(
        update_grid->tree());
    tbb::parallel_for(update_grid_leaf_manager.leafRange(), fill_range);
#ifdef LOG_FULL
    cout << "Fill update_grid to false in " << timer.elapsed() << " sec.\n";
#endif
  }
}

template <class image_t> void Recut<image_t>::activate_all_tiles() {
  for (VID_t tile_id = 0; tile_id < grid_tile_size; ++tile_id) {
    active_tiles[tile_id] = true;
  }
}

/*
 * If any tile is active return false, a tile is active if
 * any of its blocks are still active
 */
template <class image_t> bool Recut<image_t>::are_tiles_finished() {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Tiles active: ";
#endif
  for (VID_t tile_id = 0; tile_id < grid_tile_size; ++tile_id) {
    if (active_tiles[tile_id]) {
      tot_active++;
#ifdef LOG_FULL
      cout << tile_id << ", ";
#endif
    }
  }
  cout << '\n';
  if (tot_active == 0) {
    return true;
  } else {
#ifdef LOG_FULL
    cout << tot_active << " total tiles active" << '\n';
#endif
    return false;
  }
}

/*
 * If any block is active return false, a block is active if its
 * corresponding heap is not empty
 */
template <class image_t>
bool Recut<image_t>::any_fifo_active(
    std::map<GridCoord, std::deque<VertexAttr>> &check_fifo) {
  // std::experimental::parallel::any_of(check_fifo.begin(), check_fifo.end(),
  // [](const auto& pair) { return !pair.second.empty(); };

  auto timer = high_resolution_timer();
  auto found = std::any_of(
      std::execution::par_unseq, check_fifo.begin(), check_fifo.end(),
      [](const auto &pair) { return !pair.second.empty(); });

#ifdef LOG_FULL
  cout << "Check fifos in " << timer.elapsed() << " sec.\n";
#endif

  return found;

  // VID_t tot_active = 0;
  ////#ifdef LOG_FULL
  //// cout << "Blocks active: ";
  ////#endif
  // for (const auto &pair : check_fifo) {
  // if (!pair.second.empty()) {
  // tot_active++;
  ////#ifdef LOG_FULL
  //// cout << pair.first << ", ";
  //// print_iter_name(pair.second, "deque");
  ////#endif
  //}
  //}
  // if (tot_active == 0) {
  // return true;
  //} else {
  // #ifdef LOG_FULL
  // cout << '\n' << tot_active << " total blocks active" << '\n';
  // #endif
  // return false;
  //}
}

// if the parent has been pruned then set the current
// parent further upstream
// parameters are intentionally pass by value (copied)
OffsetCoord adjust_vertex_parent(EnlargedPointDataGrid::Ptr grid,
                                 OffsetCoord parent_offset,
                                 const GridCoord &original_coord,
                                 std::vector<GridCoord> valid_list = {}) {
  GridCoord parent_coord;
  GridCoord current_coord = original_coord;
  while (true) {
    // find parent
    parent_coord = coord_add(current_coord, parent_offset);
    auto parent_leaf = grid->tree().probeConstLeaf(parent_coord);
    assertm(parent_leaf, "must have parent leaf");
    auto parent_ind = parent_leaf->beginIndexVoxel(parent_coord);
    assertm(parent_ind, "inactive parents must be unreachable");

    // check state
    openvdb::points::AttributeHandle<uint8_t> flags_handle(
        parent_leaf->constAttributeArray("flags"));

    auto valid_parent = false;

    // filter by set of known active coords, only using the topology to
    // find the next in the valid list
    if (valid_list.empty()) {
      if (is_valid(flags_handle, parent_ind)) {
        valid_parent = true;
      }
    } else {
      for (const auto &coord : valid_list) {
        if (coord_all_eq(parent_coord, coord)) {
          valid_parent = true;
          break;
        }
      }
    }

    if (valid_parent) {
      break;
    } else {
      // prep for next iteration
      current_coord = parent_coord;

      // get new offset
      openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
          parent_leaf->constAttributeArray("parents"));
      parent_offset = parents_handle.get(*parent_ind);
    }
  }

  // parent guaranteed unpruned and upward traceable from current
  return coord_sub(parent_coord, original_coord);
}

template <class image_t>
template <class Container>
void Recut<image_t>::dump_buffer(Container buffer) {
  std::cout << "\n\nDump buffer\n";
  for (auto tile_id = 0; tile_id < grid_tile_size; ++tile_id) {
    for (auto block_id = 0; block_id < tile_block_size; ++block_id) {
      for (auto &v : buffer[tile_id][block_id]) {
        std::cout << v.description();
      }
    }
  }
  std::cout << "\n\nFinished buffer dump\n";
}

template <class image_t>
template <class Container, typename T2>
void Recut<image_t>::value_tile(const image_t *tile, VID_t tile_id,
                                VID_t block_id, std::string stage,
                                const TileThresholds<image_t> *tile_thresholds,
                                Container &fifo, VID_t revisits, T2 leaf_iter) {
  if (heap_map[leaf_iter->origin()].empty())
    return;

  auto update_leaf = this->update_grid->tree().probeLeaf(leaf_iter->origin());
  auto bbox = leaf_iter->getNodeBoundingBox();
  assertm(update_leaf, "corresponding leaf does not exist");

#ifdef FULL_PRINT
  cout << "\nMarching " << bbox << '\n';
#endif

  // load flags
  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle =
      leaf_iter->attributeArray("flags");
  openvdb::points::AttributeWriteHandle<OffsetCoord> parents_handle =
      leaf_iter->attributeArray("parents");
  openvdb::points::AttributeWriteHandle<float> value_handle =
      leaf_iter->attributeArray("value");

  VertexAttr *msg_vertex;
  VID_t visited = 0;
  while (!(heap_map[leaf_iter->origin()].empty())) {

#ifdef FULL_PRINT
    visited += 1;
#endif

    // msg_vertex might become undefined during scatter
    // or if popping from the fifo, take needed info
    msg_vertex = heap_map[leaf_iter->origin()].pop(block_id, stage);

    const bool in_domain = msg_vertex->selected();
    auto surface = msg_vertex->surface();
    auto msg_coord = coord_add(msg_vertex->offsets, leaf_iter->origin());
    auto msg_off = coord_mod(msg_coord, this->block_lengths);
    auto msg_ind = leaf_iter->beginIndexVoxel(msg_coord);
    auto msg_vox = this->input_grid->tree().getValue(msg_coord);

#ifdef FULL_PRINT
    cout << "check current " << msg_coord << ' ' << in_domain << '\n';
#endif

    // invalid can either be out of range of the entire global image or it
    // can be a background vertex which occurs due to pixel value below the
    // threshold, previously selected vertices are considered valid
    auto found_adjacent_invalid = false;
    auto valids =
        // star stencil offsets to img coords
        this->stencil | rv::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rv::remove_if([this, &found_adjacent_invalid](auto coord_img) {
          if (this->image_bbox.isInside(coord_img))
            return false;
          found_adjacent_invalid = true;
          return true;
        }) |
        // within leaf?
        rv::remove_if([&](auto coord_img) {
          auto mismatch = !bbox.isInside(coord_img);
          if (mismatch) {
            if (this->input_is_vdb) {
              if (!this->topology_grid->tree().isValueOn(coord_img)) {
                found_adjacent_invalid = true;
              }
            } else {
              auto dst_vox = get_img_val(tile, coord_img);
              if (dst_vox <= tile_thresholds->bkg_thresh) {
                found_adjacent_invalid = true;
              }
            }
          }
          return mismatch;
        }) |
        // visit valid voxels
        rv::remove_if([&](auto coord_img) {
          auto offset_to_current = coord_sub(msg_coord, coord_img);
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // is background?  ...has side-effects
          return !accumulate_value(tile, tile_id, block_id, coord_img, ind,
                                   offset_to_current, revisits, tile_thresholds,
                                   found_adjacent_invalid, leaf_iter,
                                   update_leaf, flags_handle, parents_handle,
                                   value_handle, coord_img, msg_ind, msg_vox);
        }) |
        rng::to_vector; // force full evaluation via vector

    // ignore if already designated as surface
    // also prevents adding to fifo twice
    if (in_domain) {
      if (surface ||
          (found_adjacent_invalid && !(is_surface(flags_handle, msg_ind)))) {
#ifdef FULL_PRINT
        std::cout << "\tfound surface vertex in " << bbox.min() << " "
                  << coord_to_str(msg_coord) << ' ' << msg_off << '\n';
#endif

        // save all surface vertices for the radius stage
        // each fifo corresponds to a specific tile_id and block_id
        // so there are no race conditions
        // save all surface vertices for the radius stage
        set_surface(flags_handle, msg_ind);
        fifo.emplace_back(Bitfield(flags_handle.get(*msg_ind)), msg_off,
                          parents_handle.get(*msg_ind));
      }
    }
  }

  assertm(heap_map[leaf_iter->origin()].empty(), "not empty");
#ifdef FULL_PRINT
  cout << "visited vertices: " << visited << '\n';
#endif
}

template <class image_t>
template <class Container, typename T2>
void Recut<image_t>::connected_tile(
    const image_t *tile, VID_t tile_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds, Container &connected_fifo,
    Container &fifo, VID_t revisits, T2 leaf_iter) {
  if (connected_fifo.empty())
    return;

  auto update_leaf = this->update_grid->tree().probeLeaf(leaf_iter->origin());
  auto bbox = leaf_iter->getNodeBoundingBox();
  assertm(update_leaf, "corresponding leaf does not exist");

#ifdef FULL_PRINT
  cout << "\nMarching " << bbox << '\n';
#endif

  // load flags
  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle =
      leaf_iter->attributeArray("flags");
  openvdb::points::AttributeWriteHandle<OffsetCoord> parents_handle =
      leaf_iter->attributeArray("parents");

  VertexAttr *msg_vertex;
  VID_t visited = 0;
  while (!(connected_fifo.empty())) {

#ifdef FULL_PRINT
    visited += 1;
#endif

    // msg_vertex might become undefined during scatter
    // or if popping from the fifo, take needed info
    msg_vertex = &(connected_fifo.front());
    const bool in_domain = msg_vertex->selected();
    auto surface = msg_vertex->surface();
    auto msg_coord = coord_add(msg_vertex->offsets, leaf_iter->origin());
    auto msg_off = coord_mod(msg_coord, this->block_lengths);
    auto msg_ind = leaf_iter->beginIndexVoxel(msg_coord);

#ifdef FULL_PRINT
    cout << "check current " << msg_coord << ' ' << in_domain << '\n';
#endif

    // invalid can either be out of range of the entire global image or it
    // can be a background vertex which occurs due to pixel value below the
    // threshold, previously selected vertices are considered valid
    auto found_adjacent_invalid = false;
    auto valids =
        // star stencil offsets to img coords
        this->stencil | rv::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rv::remove_if([this, &found_adjacent_invalid](auto coord_img) {
          if (this->image_bbox.isInside(coord_img))
            return false;
          found_adjacent_invalid = true;
          return true;
        }) |
        // within leaf?
        rv::remove_if([&](auto coord_img) {
          auto mismatch = !bbox.isInside(coord_img);
          if (mismatch) {
            if (this->input_is_vdb) {
              if (!this->topology_grid->tree().isValueOn(coord_img)) {
                found_adjacent_invalid = true;
              }
            } else {
              auto dst_vox = get_img_val(tile, coord_img);
              if (dst_vox <= tile_thresholds->bkg_thresh) {
                found_adjacent_invalid = true;
              }
            }
          }
          return mismatch;
        }) |
        // visit valid voxels
        rv::remove_if([&](auto coord_img) {
          auto offset_to_current = coord_sub(msg_coord, coord_img);
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // is background?  ...has side-effects
          return !accumulate_connected(
              tile, tile_id, block_id, coord_img, ind, offset_to_current,
              revisits, tile_thresholds, found_adjacent_invalid, leaf_iter,
              update_leaf, flags_handle, parents_handle, connected_fifo);
        }) |
        rng::to_vector; // force full evaluation via vector

    // ignore if already designated as surface
    // also prevents adding to fifo twice
    if (in_domain) {
      if (surface ||
          (found_adjacent_invalid && !(is_surface(flags_handle, msg_ind)))) {
#ifdef FULL_PRINT
        std::cout << "\tfound surface vertex in " << bbox.min() << " "
                  << coord_to_str(msg_coord) << ' ' << msg_off << '\n';
#endif

        // save all surface vertices for the radius stage
        // each fifo corresponds to a specific tile_id and block_id
        // so there are no race conditions
        // save all surface vertices for the radius stage
        set_surface(flags_handle, msg_ind);
        fifo.emplace_back(Bitfield(flags_handle.get(*msg_ind)), msg_off,
                          parents_handle.get(*msg_ind));
      }
    }
    // safe to remove msg now with no issue of invalidations
    connected_fifo.pop_front(); // remove it
  }

#ifdef FULL_PRINT
  cout << "visited vertices: " << visited << '\n';
#endif
}

template <class image_t>
template <class Container, typename T2>
void Recut<image_t>::radius_tile(const image_t *tile, VID_t tile_id,
                                 VID_t block_id, std::string stage,
                                 const TileThresholds<image_t> *tile_thresholds,
                                 Container &fifo, VID_t revisits,
                                 T2 leaf_iter) {
  if (fifo.empty())
    return;

  auto update_leaf = this->update_grid->tree().probeLeaf(leaf_iter->origin());
  auto bbox = leaf_iter->getNodeBoundingBox();
  // load read-only flags
  openvdb::points::AttributeHandle<uint8_t> flags_handle =
      leaf_iter->constAttributeArray("flags");
  // read-write radius
  openvdb::points::AttributeWriteHandle<float> radius_handle =
      leaf_iter->attributeArray("pscale");

  VID_t visited = 0;
  while (!(fifo.empty())) {
    // msg_vertex will be invalidated during scatter
    // or any other insertion or deltion to fifo
    auto msg_vertex = &(fifo.front());

    auto msg_coord = coord_add(msg_vertex->offsets, leaf_iter->origin());
    auto msg_ind = leaf_iter->beginIndexVoxel(msg_coord);

    // radius field can now be be mutated
    // set any vertex that shares a border with background
    // to the known radius of 1
    if (msg_vertex->selected()) {
      if (msg_vertex->surface()) {
        msg_vertex->radius = 1;
        radius_handle.set(*msg_ind, 1);
        // if in domain notify potential outside domains of the change
        set_if_active(update_leaf, msg_coord);
      }
    }

    if (msg_vertex->radius < 1) {
      std::ostringstream err;
      err << msg_coord << " has radii " << +(msg_vertex->radius);
      err << " handle radius has " << +(radius_handle.get(*msg_ind)) << " msg? "
          << msg_vertex->unselected() << "surf? " << msg_vertex->surface()
          << ' ' << msg_vertex->offsets << '\n';
      cout << err.str() << '\n';
      throw std::runtime_error(err.str());
    }

#ifdef LOG_FULL
    visited += 1;
#endif

    auto updated_inds =
        // star stencil offsets to img coords
        this->stencil | rv::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rv::remove_if([this](auto coord_img) {
          return !this->image_bbox.isInside(coord_img);
        }) |
        // within leaf?
        rv::remove_if(
            [&](auto coord_img) { return !bbox.isInside(coord_img); }) |
        rv::transform([&](auto coord_img) { return coord_img; }) |
        // visit valid voxels
        rv::transform([&](auto coord_img) {
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // ...has side-effects
          accumulate_radius(tile_id, block_id, coord_img, ind,
                            msg_vertex->radius, fifo, flags_handle,
                            radius_handle, update_leaf);
          return ind;
        }) |
        rng::to_vector; // force evaluation

    // now remove the front
    fifo.pop_front();
  }
}

template <class image_t>
template <class Container, typename T2>
void Recut<image_t>::prune_tile(const image_t *tile, VID_t tile_id,
                                VID_t block_id, std::string stage,
                                const TileThresholds<image_t> *tile_thresholds,
                                Container &fifo, VID_t revisits, T2 leaf_iter) {
  if (fifo.empty())
    return;

  auto update_leaf = this->update_grid->tree().probeLeaf(leaf_iter->origin());
  auto bbox = leaf_iter->getNodeBoundingBox();

  openvdb::points::AttributeWriteHandle<float> radius_handle =
      leaf_iter->attributeArray("pscale");

  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle =
      leaf_iter->attributeArray("flags");

  VID_t visited = 0;
  while (!(fifo.empty())) {
    // fifo starts with only roots
    auto current = &(fifo.front());

#ifdef LOG_FULL
    visited += 1;
#endif

    auto msg_coord = coord_add(current->offsets, leaf_iter->origin());

    if (current->selected()) {
      auto msg_ind = leaf_iter->beginIndexVoxel(msg_coord);
      set_prune_visited(flags_handle, msg_ind);
    }

#ifdef FULL_PRINT
    // all block ids are a linear row-wise idx, relative to current tile
    cout << '\n'
         << coord_to_str(msg_coord) << " tile " << tile_id << " block "
         << bbox.min() << " label " << current->label() << " radius "
         << +(current->radius) << '\n';
#endif

    // force full evaluation by saving to vector
    auto updated_inds =
        // star stencil offsets to img coords
        this->stencil | rv::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rv::remove_if([this](auto coord_img) {
          return !this->image_bbox.isInside(coord_img);
        }) |
        // within leaf?
        rv::remove_if(
            [&](auto coord_img) { return !bbox.isInside(coord_img); }) |
        // visit valid voxels
        rv::transform([&](auto coord_img) {
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // ...has side-effects
          accumulate_prune(tile_id, block_id, coord_img, ind, current->radius,
                           current->tombstone(), fifo, radius_handle,
                           flags_handle, update_leaf);
          return ind;
        }) |
        rng::to_vector;

    // now remove the front
    fifo.pop_front();
  } // end while over fifo
} // end prune_tile

template <class image_t>
template <typename T2>
void Recut<image_t>::march_narrow_band(
    const image_t *tile, VID_t tile_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds,
    std::deque<VertexAttr> &connected_fifo, std::deque<VertexAttr> &fifo,
    T2 leaf_iter) {
#ifdef FULL_PRINT
  auto timer = high_resolution_timer();
  auto loc = tree_to_str(tile_id, block_id);
  // cout << "\nMarching " << loc << ' ' << leaf_iter->origin() << '\n';
#endif

  VID_t revisits = 0;

  if (stage == "value") {
    value_tile(tile, tile_id, block_id, stage, tile_thresholds, fifo, revisits,
               leaf_iter);
  } else if (stage == "connected") {
    connected_tile(tile, tile_id, block_id, stage, tile_thresholds,
                   connected_fifo, fifo, revisits, leaf_iter);
  } else if (stage == "radius") {
    radius_tile(tile, tile_id, block_id, stage, tile_thresholds, fifo, revisits,
                leaf_iter);
  } else if (stage == "prune") {
    prune_tile(tile, tile_id, block_id, stage, tile_thresholds, fifo, revisits,
               leaf_iter);
  } else {
    assertm(false, "Stage name not recognized");
  }

#ifdef FULL_PRINT
  // cout << "Marched " << loc << " in " << timer.elapsed() << " s" << '\n';
#endif

} // end march_narrow_band

template <class image_t>
template <typename T2>
std::atomic<double> Recut<image_t>::process_tile(
    VID_t tile_id, const image_t *tile, std::string stage,
    const TileThresholds<image_t> *tile_thresholds, T2 vdb_accessor) {
  auto timer = high_resolution_timer();

  vt::LeafManager<PointTree> grid_leaf_manager(this->topology_grid->tree());

  integrate_update_grid(this->topology_grid, grid_leaf_manager, stage,
                        this->map_fifo, this->connected_map, this->update_grid,
                        tile_id);

  auto march_range = [&,
                      this](const openvdb::tree::LeafManager<
                            openvdb::points::PointDataTree>::LeafRange &range) {
    // for each leaf with active voxels i.e. containing topology
    for (auto leaf_iter = range.begin(); leaf_iter; ++leaf_iter) {
      auto block_id = coord_img_to_block_id(leaf_iter->origin());
      march_narrow_band(tile, tile_id, block_id, stage, tile_thresholds,
                        this->connected_map[leaf_iter->origin()],
                        this->map_fifo[leaf_iter->origin()], leaf_iter);
    }
  };

  auto is_active = [&, this]() {
    // connected or value stages are determined finished by the status of their
    // fifo or heaps respectively
    if (stage == "connected") {
      return any_fifo_active(this->connected_map);
    } else if (stage == "value") {
      return std::any_of(
          std::execution::par_unseq, heap_map.begin(), heap_map.end(),
          [](const auto &kv_pair) { return !kv_pair.second.empty(); });
    }
    return any_fifo_active(this->map_fifo);
  };

  // if there is a single block per tile than this while
  // loop will exit after one iteration
  VID_t inner_iteration_idx = 0;
  for (; is_active(); ++inner_iteration_idx) {

    { // march
      auto iter_start = timer.elapsed();
      tbb::parallel_for(grid_leaf_manager.leafRange(), march_range);
#ifdef LOG_FULL
      cout << "\nMarched " << stage << " in " << timer.elapsed() - iter_start
           << " sec.\n";
#endif
    }

    { // integrate
      auto integrate_start = timer.elapsed();
      integrate_update_grid(this->topology_grid, grid_leaf_manager, stage,
                            this->map_fifo, this->connected_map,
                            this->update_grid, tile_id);
#ifdef LOG_FULL
      cout << "Integrated " << stage << " in "
           << timer.elapsed() - integrate_start << " sec.\n";
#endif
    }

  } // iterations per tile

  active_tiles[tile_id] = false;

#ifdef LOG_FULL
  cout << "Interval: " << tile_id << " in " << inner_iteration_idx
       << " iterations, total " << timer.elapsed() << " sec." << '\n';
#endif
  return timer.elapsed();
}

// Calculate new tile thresholds or use input thresholds according
// to args this function has no sideffects outside
// of the returned tile_thresholds struct
template <class image_t>
template <typename local_image_t>
TileThresholds<local_image_t> *
Recut<image_t>::get_tile_thresholds(local_image_t *buffer,
                                    int tile_vertex_size) {

  auto tile_thresholds = new TileThresholds<local_image_t>();

  // assign thresholding value
  // foreground parameter takes priority
  // Note if either foreground or background percent is equal to or greater
  // than 0 than it was changed by a user so it takes precedence over the
  // defaults
  if (this->args->foreground_percent >= 0) {
    auto timer = high_resolution_timer();
    tile_thresholds->bkg_thresh = bkg_threshold<local_image_t>(
        buffer, tile_vertex_size, (this->args->foreground_percent) / 100);
#ifdef LOG
    // std::cout << "bkg_thresh in " << timer.elapsed() << " s\n";
#endif
  } else { // if bkg set explicitly and foreground wasn't
    if (this->args->background_thresh >= 0) {
      tile_thresholds->bkg_thresh = this->args->background_thresh;
    } else {
      // otherwise: tile_thresholds->bkg_thresh default inits to 0
      tile_thresholds->bkg_thresh = 0;
    }
  }

  if (this->args->convert_only) {
    tile_thresholds->max_int = std::numeric_limits<local_image_t>::max();
    tile_thresholds->min_int = std::numeric_limits<local_image_t>::min();
    return tile_thresholds;
  }

  if (this->args->max_intensity < 0) {
    tile_thresholds->get_max_min(buffer, tile_vertex_size);
  } else if (this->args->min_intensity < 0) {
    // if max intensity was set but not a min, just use the bkg_thresh value
    auto already_set = [tile_thresholds](const auto bkg_thresh) {
      if (bkg_thresh >= 0) {
        tile_thresholds->min_int = bkg_thresh;
        return true;
      }
      return false;
    };

    if (!already_set(tile_thresholds->bkg_thresh)) {
      // max and min members will be set
      tile_thresholds->get_max_min(buffer, tile_vertex_size);
      tile_thresholds->bkg_thresh = tile_thresholds->min_int;
    }
  } else { // both values were set
    // either of these values are signed and default inited -1, casting
    // them to unsigned image_t would lead to hard to find errors
    assertm(this->args->max_intensity >= 0, "invalid user max");
    assertm(this->args->min_intensity >= 0, "invalid user min");
    // otherwise set global max min from recut_parameters
    tile_thresholds->max_int = (local_image_t)this->args->max_intensity;
    tile_thresholds->min_int = (local_image_t)this->args->min_intensity;
  }

#ifdef LOG_FULL
  cout << "max_int: " << +(tile_thresholds->max_int)
       << " min_int: " << +(tile_thresholds->min_int) << '\n';
  cout << "bkg_thresh value = " << +(tile_thresholds->bkg_thresh) << '\n';
#endif
  return tile_thresholds;
}

template <class image_t>
template <typename T1, typename T2, typename T3, typename T4>
void Recut<image_t>::io_tile(int tile_id, T1 &grids, T2 &uint8_grids,
                             T3 &float_grids, T4 &mask_grids, std::string stage,
                             HistV &histogram) {

  // only start with tiles that have active processing to do
  if (!active_tiles[tile_id]) {
    return;
  }

  auto tile_timer = high_resolution_timer();

  if (stage == "convert") {
    auto tile_offsets = id_tile_to_img_offsets(tile_id);
    // bbox is inclusive, therefore substract 1
    auto tile_max = (tile_offsets + this->tile_lengths).offsetBy(-1);
    GridCoord buffer_offsets = zeros();
    // protect out of bounds of whole image by cropping z only to extent
    auto z_extent = this->image_offsets[2] + this->image_lengths[2];

    if (tile_max.z() >= z_extent) {
      // bbox defines an inclusive range at both ends
      tile_max[2] = z_extent - 1;
    }
    const auto tile_bbox = CoordBBox(tile_offsets, tile_max);

    TileV dense_tile;

    if (args->input_type == "ims") {
#ifdef USE_HDF5
      dense_tile =
          load_imaris_tile(args->input_path.generic_string(), tile_bbox,
                           args->resolution_level, args->channel);
#else
      throw std::runtime_error("HDF5 dependency required for input type ims");
#endif
    }

    if (args->input_type == "tiff") {
      auto bits_per_sample = get_tif_bit_width(args->input_path.string());
      if (bits_per_sample == 8)
        dense_tile =
            load_tile<uint8_t>(tile_bbox, args->input_path.generic_string());
      else if (bits_per_sample == 16)
        dense_tile =
            load_tile<uint16_t>(tile_bbox, args->input_path.generic_string());
      else
        throw std::runtime_error("Only 8-bits and 16-bits TIFF are supported");
    }

    ThreshV tile_thresholds;

    std::visit(
        [this, &tile_thresholds](auto &tile) {
          auto tile_dims = tile->bbox().dim();
          auto tile_vertex_size = coord_prod_accum(tile_dims);
          tile_thresholds = get_tile_thresholds(tile->data(), tile_vertex_size);
        },
        dense_tile);

    assertm(!this->input_is_vdb, "input can't be vdb during convert stage");

    auto convert_start = tile_timer.elapsed();

    // #ifdef FULL_PRINT
    //  cout << "print_image\n";
    //  print_image_3D(dense_tile->data(), tile_bbox.dim());
    // #endif

    // visit type of buffer uint8/uint16
    // TODO pass to lambda by ref where appropriate
    std::visit(
        [&, this](const auto &tile, const auto &tile_thresholds,
                  const auto &histogram) mutable {
          auto buffer = tile->data();
          // FIXME collapse these blocks into an output std::variant
          if (this->args->output_type == "uint8") {
            uint8_grids[tile_id] = ImgGrid::create();
            convert_buffer_to_vdb_acc(
                buffer, tile_bbox.dim(),
                /*buffer_offsets=*/buffer_offsets,
                /*image_offsets=*/tile_bbox.min(),
                uint8_grids[tile_id]->getAccessor(), this->args->output_type,
                tile_thresholds->bkg_thresh, this->args->upsample_z);
            if (args->histogram) {
              // histogram += hist(buffer, tile_bbox.dim(), buffer_offsets);
            }
          } else if (this->args->output_type == "float") {
            float_grids[tile_id] = openvdb::FloatGrid::create();
            convert_buffer_to_vdb_acc(
                buffer, tile_bbox.dim(),
                /*buffer_offsets=*/buffer_offsets,
                /*image_offsets=*/tile_bbox.min(),
                float_grids[tile_id]->getAccessor(), this->args->output_type,
                tile_thresholds->bkg_thresh, this->args->upsample_z);
            if (args->histogram) {
              // histogram += hist(buffer, tile_bbox.dim(), buffer_offsets);
            }
          } else if (this->args->output_type == "mask") {
            mask_grids[tile_id] = openvdb::MaskGrid::create();
            convert_buffer_to_vdb_acc(
                buffer, tile_bbox.dim(),
                /*buffer_offsets=*/buffer_offsets,
                /*image_offsets=*/tile_bbox.min(),
                mask_grids[tile_id]->getAccessor(), this->args->output_type,
                tile_thresholds->bkg_thresh, this->args->upsample_z);
            if (args->histogram) {
              // histogram += hist(buffer, tile_bbox.dim(), buffer_offsets);
            }
          } else { // point

            std::vector<PositionT> positions;
            // use the last bkg_thresh calculated for metadata,
            // bkg_thresh is constant for each tile unless a specific % is
            // input by command line user
            convert_buffer_to_vdb(buffer, tile_bbox.dim(),
                                  /*buffer_offsets=*/buffer_offsets,
                                  /*image_offsets=*/tile_bbox.min(), positions,
                                  tile_thresholds->bkg_thresh,
                                  this->args->upsample_z);

            grids[tile_id] = create_point_grid(positions, this->image_lengths,
                                               get_transform(),
                                               this->args->foreground_percent);

#ifdef FULL_PRINT
            print_vdb_mask(grids[tile_id]->getConstAccessor(),
                           this->image_lengths);
#endif
          }
        },
        dense_tile, tile_thresholds, histogram);

    active_tiles[tile_id] = false;
#ifdef LOG
    if (args->input_type == "ims") {
      std::cout << "Completed tile " << tile_id + 1 << " of " << grid_tile_size
                << " in " << tile_timer.elapsed() << " s\n";
      //" converted in " << (tile_timer.elapsed() - convert_start) <<
    }
#endif
  } else {
    // note openvdb::initialize() must have been called before this point
    // otherwise seg faults will occur
    auto update_accessor = this->update_grid->getAccessor();

    if (stage == "value") {
      throw std::runtime_error(
          "Unimplemented: need to generate tile_threshold for raw images");
    }

    // dummy values, only used in unimplemented value stage
    auto tile_thresholds = new TileThresholds<image_t>(
        /*max*/ 2,
        /*min*/ 0,
        /*bkg_thresh*/ 0);

    image_t *tile; // passing raw image buffers is deprecated, only used vdb
                   // sparse grids
    process_tile(tile_id, tile, stage, tile_thresholds, update_accessor);
  }
} // if the tile is active

template <class image_t>
template <class Container>
void Recut<image_t>::update(std::string stage, Container &fifo) {

#ifdef LOG
  cout << "Start updating stage " << stage << '\n';
#endif
  auto timer = high_resolution_timer();

  // multi-grids for convert stage
  // assertm(this->topology_grid, "topology grid not initialized");
  std::vector<EnlargedPointDataGrid::Ptr> grids(this->grid_tile_size);
  std::vector<ImgGrid::Ptr> uint8_grids(this->grid_tile_size);
  std::vector<openvdb::FloatGrid::Ptr> float_grids(this->grid_tile_size);
  std::vector<openvdb::MaskGrid::Ptr> mask_grids(this->grid_tile_size);

  // auto histogram = Histogram<image_t>();
  HistV histogram;

  // Main march for loop
  // continue iterating until all tiles are finished
  // tiles can be (re)activated by neighboring tiles
  int outer_iteration_idx;
  for (outer_iteration_idx = 0; !are_tiles_finished(); outer_iteration_idx++) {

    // Create the custom task_arena with specified threads
    tbb::task_arena arena(args->user_thread_count);
    // loop through all possible tiles
    // only safe for conversion stages with more than 1 thread
    arena.execute([&] {
      tbb::parallel_for_each(rv::indices(grid_tile_size) | rng::to_vector,
                             [&](const auto tile_id) {
                               io_tile(tile_id, grids, uint8_grids, float_grids,
                                       mask_grids, stage, histogram);
                             }); // end one tile traversal
    });
  } // finished all tiles

  if (stage == "convert") {
    std::cout << "Start merge\n";
    auto finalize_start = timer.elapsed();

    if (args->output_type == "point") {

      this->topology_grid = merge_grids(grids);

      set_grid_meta(this->topology_grid, this->image_lengths,
                    args->foreground_percent, args->channel,
                    args->resolution_level, this->args->output_name,
                    args->upsample_z);

    } else {
      if (this->args->output_type == "float") {
        this->input_grid = merge_grids(float_grids);
        set_grid_meta(this->input_grid, this->image_lengths,
                      args->foreground_percent, args->channel,
                      args->resolution_level, this->args->output_name,
                      args->upsample_z);
      } else if (this->args->output_type == "uint8") {
        this->img_grid = merge_grids(uint8_grids);
        set_grid_meta(this->img_grid, this->image_lengths,
                      args->foreground_percent, args->channel,
                      args->resolution_level, this->args->output_name,
                      args->upsample_z);
      } else if (this->args->output_type == "mask") {
        this->mask_grid = merge_grids(mask_grids);
        set_grid_meta(this->mask_grid, this->image_lengths,
                      args->foreground_percent, args->channel,
                      args->resolution_level, this->args->output_name,
                      args->upsample_z);
      }

      if (args->histogram) {
        std::visit(
            [](auto &histogram) {
              std::ofstream hist_file;
              hist_file.open("hist.txt");
              hist_file << histogram;
              hist_file.close();
            },
            histogram);
      }
    }

    auto finalize_time = timer.elapsed() - finalize_start;
#ifdef LOG
    cout << "Grid finalize time: " << finalize_time << " s\n";
#endif
  }

  cout << "Finished stage: " << stage << '\n';
  cout << "Finished total updating within " << timer.elapsed_formatted()
       << '\n';

  {
    auto stage_acr = stage; // line up with paper
    if (stage == "convert")
      stage_acr = "VC";
    if (stage == "connected")
      stage_acr = "connected components";
    if (stage == "radius")
      stage_acr = "SDF radius";

    std::ofstream run_log;
    run_log.open(log_fn, std::ios::app);
    run_log << "Skeletonization: " << stage_acr << " time, "
            << timer.elapsed_formatted() << '\n';
    run_log.flush();
  }

} // end update()

/*
 * Convert block coordinates into linear index row-ordered
 */
template <class image_t>
inline VID_t Recut<image_t>::sub_block_to_block_id(const VID_t iblock,
                                                   const VID_t jblock,
                                                   const VID_t kblock) {
  return iblock + jblock * this->tile_block_lengths[0] +
         kblock * this->tile_block_lengths[0] * this->tile_block_lengths[1];
}

// Wrap-around rotate all values forward one
// This logic disentangles 0 % 32 from 32 % 32 results
// This function is abstract in that it can adjust coordinates
// of vid, block or tile
template <class image_t>
inline VID_t Recut<image_t>::rotate_index(VID_t img_coord, const VID_t current,
                                          const VID_t neighbor,
                                          const VID_t tile_block_size,
                                          const VID_t pad_block_size) {
  // when they are in the same block or index
  // then you simply need to account for the 1
  // voxel border region to get the correct coord
  // for this dimension
  if (current == neighbor) {
    return img_coord + 1; // adjust to padded block idx
  }
  // if it's in another block/tile it can only be 1 vox away
  // so make sure the coord itself is on the correct edge of its block
  // domain
  if (current == (neighbor + 1)) {
    assertm(img_coord == tile_block_size - 1,
            "Does not currently support diagonal connections or any ghost "
            "regions greater that 1");
    return 0;
  }
  if (current == (neighbor - 1)) {
    assertm(img_coord == 0,
            "Does not currently support diagonal connections or "
            "any ghost regions greater that 1");
    return pad_block_size - 1;
  }

  // failed
  assertm(false, "Does not currently support diagonal connections or any ghost "
                 "regions greater that 1");
  return 0; // for compiler
}

template <class image_t>
void Recut<image_t>::initialize_globals(const VID_t &grid_tile_size,
                                        const VID_t &tile_block_size) {

  auto timer = high_resolution_timer();

  auto is_boundary = [](auto coord) {
    for (int i = 0; i < 3; ++i) {
      if (coord[i]) {
        if (coord[i] == (LEAF_LENGTH - 1))
          return true;
      } else {
        return true;
      }
    }
    return false;
  };

  std::map<GridCoord, std::deque<VertexAttr>> inner;
  VID_t tile_id = 0;

  for (auto leaf_iter = this->topology_grid->tree().beginLeaf(); leaf_iter;
       ++leaf_iter) {
    auto origin = leaf_iter->getNodeBoundingBox().min();

    // per leaf fifo/pq resources
    inner[origin] = std::deque<VertexAttr>();
    heap_map[origin] = local_heap();

    // every topology_grid leaf must have a corresponding leaf explicitly
    // created
    auto update_leaf =
        new openvdb::tree::LeafNode<bool, LEAF_LOG2DIM>(origin, false);

    // init update grid with fixed topology (active state)
    for (auto ind = leaf_iter->beginIndexOn(); ind; ++ind) {
      // get coord
      auto coord = ind.getCoord();
      auto leaf_coord = coord_mod(
          coord, new_grid_coord(LEAF_LENGTH, LEAF_LENGTH, LEAF_LENGTH));
      if (is_boundary(leaf_coord)) {
        update_leaf->setActiveState(coord, true);
      }
    }
    this->update_grid->tree().addLeaf(update_leaf);
  }

  // fifo is a deque representing the vids left to
  // process at each stage
  this->map_fifo = inner;
  this->connected_map = inner;
  // global active vertex list

#ifdef LOG_FULL
  cout << "\tCreated fifos " << timer.elapsed() << 's' << '\n';
#endif
}

// Deduce lengths from the various input options
template <class image_t>
GridCoord Recut<image_t>::get_input_image_lengths(RecutCommandLineArgs *args) {
  GridCoord input_image_lengths = zeros();
  this->update_grid = openvdb::BoolGrid::create();
  if (this->input_is_vdb) { // running based of a vdb input

    // assertm(!args->convert_only,
    //"Convert only option is not valid from vdb to vdb pass a --seeds "
    //"directory to start reconstructions");

    auto timer = high_resolution_timer();
    auto base_grid = read_vdb_file(args->input_path);

    if (base_grid->isType<openvdb::FloatGrid>()) {
      this->args->input_type = "float";
      this->input_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
      // copy topology (bit-mask actives) to the topology grid
      // this->topology_grid =
      // copy_to_point_grid(this->input_grid, input_image_lengths,
      // this->args->foreground_percent);
      this->topology_grid = convert_float_to_point(this->input_grid);
      // cout << "float count " << this->input_grid->activeVoxelCount()
      //<< " point count "
      //<< openvdb::points::pointCount(this->topology_grid->tree()) << '\n';
      // assertm(this->input_grid->activeVoxelCount() ==
      // openvdb::points::pointCount(this->topology_grid->tree()),
      //"did no match");
      auto [lengths, _] = get_metadata(input_grid);
      input_image_lengths = lengths;
      append_attributes(this->topology_grid);
    } else if (base_grid->isType<EnlargedPointDataGrid>()) {
      this->args->input_type = "point";
      this->topology_grid =
          openvdb::gridPtrCast<EnlargedPointDataGrid>(base_grid);
      auto [lengths, _] = get_metadata(topology_grid);
      input_image_lengths = lengths;
      append_attributes(this->topology_grid);
    } else if (base_grid->isType<openvdb::MaskGrid>()) {
      this->args->input_type = "mask";
      this->mask_grid = openvdb::gridPtrCast<openvdb::MaskGrid>(base_grid);
      auto [lengths, _] = get_metadata(mask_grid);
      input_image_lengths = lengths;
    } else if (base_grid->isType<ImgGrid>()) {
      throw std::runtime_error(
          "VDB grid type 'uint8' not yet supported as an input image");
    } else {
      throw std::runtime_error("VDB grid type not recognized, only type "
                               "'float', 'point', 'mask', 'uint8' supported");
    }

    if (this->args->input_type != "mask" && this->args->seed_path == "") {
      throw std::runtime_error(
          "For soma segmentation you must pass a raw image or a mask grid. If "
          "you start reconstruction from float or point you must also specify "
          "a seeds path");
    }

#ifdef LOG
    // cout << "Read grid in: " << timer.elapsed() << " s\n";
#endif

  } else { // converting to a new grid from a raw image
    if (args->input_type == "tiff") {
      const auto tif_filenames = get_dir_files(args->input_path, ".tif");
      input_image_lengths = get_tif_dims(tif_filenames);
    } else if (args->input_type == "ims") {
#ifdef USE_HDF5
      auto imaris_bbox =
          imaris_image_bbox(args->input_path.generic_string(),
                            args->resolution_level, args->channel);
      input_image_lengths = imaris_bbox.dim();
#else
      throw std::runtime_error("HDF5 dependency required for input type ims");
#endif
    } else {
      throw std::runtime_error("unknown input type" + args->input_type);
    }
  }
  return input_image_lengths;
}

template <class image_t>
void Recut<image_t>::update_hierarchical_dims(const GridCoord &tile_lengths) {

  // determine the length of tiles in each dim
  // rounding up (ceil)
  rng::for_each(rv::indices(3), [this](int i) {
    this->grid_tile_lengths[i] =
        (this->image_lengths[i] + this->tile_lengths[i] - 1) /
        this->tile_lengths[i];
  });

  // the resulting tile size override the user inputted block size
  if (this->args->convert_only) {
    this->block_lengths[0] = this->tile_lengths[0];
    this->block_lengths[1] = this->tile_lengths[1];
    this->block_lengths[2] = this->tile_lengths[2];
  } else {
    this->block_lengths[0] = std::min(this->tile_lengths[0], LEAF_LENGTH);
    this->block_lengths[1] = std::min(this->tile_lengths[1], LEAF_LENGTH);
    this->block_lengths[2] = std::min(this->tile_lengths[2], LEAF_LENGTH);
  }

  // determine length of blocks that span an tile for each dim
  // this rounds up
  rng::for_each(rv::indices(3), [this](int i) {
    this->tile_block_lengths[i] =
        (this->tile_lengths[i] + this->block_lengths[i] - 1) /
        this->block_lengths[i];
  });

  this->grid_tile_size = coord_prod_accum(this->grid_tile_lengths);
  this->tile_block_size = coord_prod_accum(this->tile_block_lengths);

  this->active_tiles = std::vector(this->grid_tile_size, false);
}

template <class image_t> void Recut<image_t>::initialize() {

  // input type
  {
    if (fs::is_directory(args->input_path)) {
      this->input_is_vdb = false;
      this->args->input_type = "tiff";
    } else {
      auto path_extension = args->input_path.extension();
      if (path_extension == ".vdb") {
        this->input_is_vdb = true;
      } else {
        this->input_is_vdb = false;
        if (path_extension == ".ims") {
          this->args->input_type = "ims";
        } else {
          throw std::runtime_error(
              "Recut does not support single files of type: " +
              path_extension.string());
        }
      }
    }
  }

  if (args->convert_only && args->input_type == "ims") {
    args->user_thread_count = 1;
  }

#ifdef LOG
  cout << "User specified image " << args->input_path << '\n';
#endif

  // actual possible lengths
  auto input_image_lengths = get_input_image_lengths(args);

  // account and check requested args->image_offsets and args->image_lengths
  // lengths are always the side length of the domain on each dim, in x y z
  // order
  assertm(coord_all_lt(args->image_offsets, input_image_lengths),
          "input offset can not exceed dimension of image");

  // default image_offsets is {0, 0, 0}
  // which means start at the beginning of the image
  // this enforces the minimum extent to be 1 in each dim
  // set and no longer refer to args->image_offsets
  this->image_offsets = args->image_offsets;

  // protect faulty out of bounds input if lengths goes beyond
  // domain of full image
  auto max_len_after_off = coord_sub(input_image_lengths, this->image_offsets);

  // sanitize in each dimension
  // and protected from faulty offset values
  rng::for_each(rv::indices(3), [this, &max_len_after_off](int i) {
    if (this->args->image_lengths[i] > 0) {
      // use the input length if possible, or maximum otherwise
      this->image_lengths[i] =
          std::min(args->image_lengths[i], max_len_after_off[i]);
    } else {
      // -1,-1,-1 means use to the end of input image
      // -1,-1,-5 means length should go up to 5 from the last z
      this->image_lengths[i] =
          max_len_after_off[i] + args->image_lengths[i] + 1;
    }
  });
  this->image_bbox = openvdb::math::CoordBBox(
      this->image_offsets, this->image_offsets + this->image_lengths);

  // TODO move this clipping up to the read step for faster performance on sub
  // grids
  if (this->input_is_vdb) {
    if (args->input_type == "mask")
      this->mask_grid->clip(this->image_bbox);
    else
      this->topology_grid->clip(this->image_bbox);
  }

  // save to globals the actual size of the full image
  // accounting for the input offsets and lengths
  // these will be used throughout the rest of the program
  // for convenience
  this->image_size = coord_prod_accum(this->image_lengths);

  // Determine the size of each tile in each dim
  // the image size and offsets override the user inputted tile size
  // since an tile must be at least the size of the image
  rng::for_each(rv::indices(3), [this](int i) {
    this->tile_lengths[i] =
        (this->input_is_vdb || (this->args->tile_lengths[i] < 1))
            ? this->image_lengths[i]
            : std::min(this->args->tile_lengths[i], this->image_lengths[i]);
  });

  // set good defaults for conversion depending on tiff/ims
  if (!this->input_is_vdb) {
    // images are saved in separate z-planes for tiff, so conversion should
    // respect that for best performance
    // if it wasn't set by user on command line, then set default
    this->tile_lengths[2] =
        this->args->tile_lengths[2] < 1 ? 8 : this->args->tile_lengths[2];
  }

#ifdef LOG
  print_coord(this->image_lengths, "image");
#endif
  update_hierarchical_dims(this->tile_lengths);

  // set the prune radius if not passed at command line
  // based on the voxel size
  if (!this->args->prune_radius.has_value()) {
    this->args->prune_radius = anisotropic_factor(this->args->voxel_size);
#ifdef LOG
    std::cout << "voxel sizes:"
              << " x=" << this->args->voxel_size[0]
              << " y=" << this->args->voxel_size[1]
              << " z=" << this->args->voxel_size[2] << "\n";
    if (!this->args->convert_only) {
      std::cout << "prune radius: " << this->args->prune_radius.value()
                << " . Calculated by the anisotropic factor of voxel sizes.\n";
    }
#endif
  }
}

// reject unvisited vertices
// band can be optionally included
template <class image_t>
bool Recut<image_t>::filter_by_label(VertexAttr *v, bool accept_tombstone) {
  if (accept_tombstone) {
    if (v->tombstone()) {
      return false;
    }
  } else {
    assertm(!(v->unselected()), "BAND vertex was lost");
    if (v->tombstone() || v->unselected()) {
      return false;
    }
  }
  if (v->radius < 1) {
    std::cout << v->description();
    assertm(false, "can't accept a vertex with a radii < 1");
  }
  return true;
}

template <class image_t> void Recut<image_t>::adjust_parent() {

  auto adjust_parent = [this](const auto &flags_handle, auto &parents_handle,
                              const auto &radius_handle, const auto &ind,
                              auto leaf) {
    auto coord = ind.getCoord();
    auto parent = adjust_vertex_parent(this->topology_grid,
                                       parents_handle.get(*ind), coord);
    parents_handle.set(*ind, parent);
#ifdef FULL_PRINT
    std::cout << coord << " -> " << coord + parent << '\n';
#endif
  };

  auto all_valid = [](const auto &flags_handle, const auto &parents_handle,
                      const auto &radius_handle,
                      const auto &ind) { return is_valid(flags_handle, ind); };

  visit(this->topology_grid, all_valid, adjust_parent);
}

// template <class image_t>
// void Recut<image_t>::print_to_swc(std::string swc_path) {

// auto coord_to_swc_id = get_id_map();

// auto to_swc = [this, &coord_to_swc_id](
// const auto &flags_handle, const auto &parents_handle,
// const auto &radius_handle, const auto &ind, auto leaf) {
// auto coord = ind.getCoord();
// print_swc_line(coord, this->args->voxel_size, is_root(flags_handle, ind),
// radius_handle.get(*ind), parents_handle.get(*ind),
// this->image_bbox, this->out,
//[>map*/ coord_to_swc_id, /*adjust<] true);
//};

// this->out.open(swc_path);
// this->out << "#id type_id x y z radius parent_id\n";

// visit(this->topology_grid, keep_root, to_swc);
// visit(this->topology_grid, not_root, to_swc);

// if (this->out.is_open())
// this->out.close();
// #ifdef LOG
// cout << "Wrote output to " << swc_path << '\n';
// #endif
//}

template <class image_t> void Recut<image_t>::prune_branch() {
  auto filter_branch = [](const auto &flags_handle, const auto &parents_handle,
                          const auto &radius_handle, const auto &ind) {
    auto parents = parents_handle.get(*ind);
    return is_valid(flags_handle, ind) && !is_root(flags_handle, ind) &&
           ((parents[0] + parents[1] + parents[2]) < MIN_BRANCH_LENGTH);
  };

  visit(this->topology_grid, filter_branch, prunes_visited);
}

template <class image_t> void Recut<image_t>::prune_radii() {
  auto filter_radii = [](const auto &flags_handle, const auto &parents_handle,
                         const auto &radius_handle, const auto &ind) {
    return is_valid(flags_handle, ind) && !is_root(flags_handle, ind) &&
           (radius_handle.get(*ind) < MIN_RADII);
  };

  visit(this->topology_grid, filter_radii, prunes_visited);
}

template <class image_t>
void Recut<image_t>::partition_components(std::vector<Seed> seeds, bool prune) {

  openvdb::GridPtrVec grids;
#ifdef LOG
  print_point_count(this->topology_grid);
#endif

  auto global_timer = high_resolution_timer();
  // this copies only vertices that have already had flags marked as selected.
  // selected means they are reachable from a known vertex during traversal
  // in either a connected or value stage.
  auto float_grid = copy_selected(this->topology_grid);
  cout << "Topo to float time " << global_timer.elapsed_formatted() << '\n';

  auto total_timer = high_resolution_timer();
  {
    VID_t selected_count = float_grid->activeVoxelCount();
    std::ofstream run_log;
    run_log.open(log_fn, std::ios::app);
    run_log << "Skeletonization: topology grid active voxel count, "
            << this->topology_grid->activeVoxelCount() << '\n';
    run_log << "Skeletonization: topology grid selected voxel count, "
            << selected_count << '\n';
    run_log.flush();
    assertm(selected_count, "active voxels in float grid must be > 0");
  }

  // aggregate disjoint connected components
  global_timer.restart();
  std::vector<openvdb::FloatGrid::Ptr> components;
  vto::segmentActiveVoxels(*float_grid, components);
#ifdef LOG
  cout << "Segment count: " << components.size() << " in "
       << global_timer.elapsed_formatted() << '\n';
#endif

  global_timer.restart();
  auto output_topology = false;

  // you need to load the passed image grids if you are outputting windows
  auto window_grids =
      args->window_grid_paths |
      rv::transform([](const auto &gpath) { return read_vdb_file(gpath); }) |
      rng::to_vector; // force reading once now

#ifdef LOG
  cout << "Finished grid reads\n";
#endif

  auto process_component = [this, &seeds,
                            &window_grids](const auto component_pair) {
    auto [index, component] = component_pair;
    // all grid transforms across are consistent across recut, so enforce
    // the same interpretation for any new grid
    component->setTransform(get_transform());
    auto bbox = component->evalActiveVoxelBoundingBox();

    // filter all roots within this component
    auto component_seeds = seeds |
                           rv::remove_if([&component](const auto &seed) {
                             return !component->tree().isValueOn(seed.coord);
                           }) |
                           rng::to_vector;

    std::string prefix = "";
    if (component_seeds.size() > 1) {
      prefix = "a-multi-";
    }

    auto voxel_count = component->activeVoxelCount();
    if (voxel_count < SWC_MIN_LINE) {
      prefix = "discard-";
      // return; // skip
    }

    if (bbox.dim()[2] < MIN_Z_DEPTH) {
      prefix = "discard-";
      // return; // skip
    }

    // is a fresh run_dir
    auto component_dir_fn =
        this->run_dir / (prefix + "component-" + std::to_string(index));
    fs::create_directories(component_dir_fn);

#ifdef LOG
    auto component_log_fn =
        component_dir_fn / ("component-" + std::to_string(index) + "-log.csv");
    std::ofstream component_log;
    component_log.open(component_log_fn.string());
    component_log << std::fixed << std::setprecision(6);
    component_log << "Thread count, " << args->user_thread_count << '\n';
    component_log << "Soma count, " << component_seeds.size() << '\n';
    component_log << "Component active voxel count, " << voxel_count << '\n';
    component_log << "Mean shift factor, "
                  << this->args->mean_shift_factor.value_or(0) << '\n';
#endif

    // seeds are always in voxel units and output with respect to the whole
    // volume
    write_seeds(component_dir_fn, component_seeds, this->args->voxel_size);

    if (args->save_vdbs) { // save a grid corresponding to this component
      write_vdb_file({component}, component_dir_fn / "float.vdb");
    }

    auto timer = high_resolution_timer();
    auto [markers, coord_to_idx] = convert_float_to_markers(
        component, this->topology_grid, this->args->prune_radius.value());

    timer.restart();
    std::vector<MyMarker *> refined_markers;
    auto refined_markers_opt =
        this->args->mean_shift_factor.has_value()
            ? mean_shift(markers, this->args->mean_shift_max_iters,
                         this->args->mean_shift_factor.value(), coord_to_idx,
                         args->timeout)
            : markers;

    // if mean shifting didn't timeout
    std::vector<std::vector<MyMarker *>> trees;
    if (refined_markers_opt) {
      refined_markers = refined_markers_opt.value();
      auto mean_shift_elapsed = timer.elapsed();
      timer.restart();

      // rebuild coord to idx for prune
      auto coord_to_indices = create_coord_to_indices(refined_markers);
      timer.restart();

      // prune radius already set when converting from markers above
      auto pruned_markers = advantra_prune(
          refined_markers, /*prune_radius*/ this->args->prune_radius.value(),
          coord_to_indices);
      if (pruned_markers.size() < 3) {
        std::cerr
            << "Non fatal error: extracted pruned trees contains too few nodes "
               "skipping " +
                   std::to_string(index)
            << '\n';
        return; // skip
      }

#ifdef LOG
      component_log << "Component count, " << markers.size() << '\n';
      component_log << "TC count, " << pruned_markers.size() << '\n';
      component_log << "MS elapsed time, " << mean_shift_elapsed << '\n';
      component_log << "TC elapsed time, " << timer.elapsed() << '\n';
#endif

      // extract a new tree via bfs
      timer.restart();
      auto cluster = extract_trees(pruned_markers, true);
#ifdef LOG
      component_log << "ET, " << timer.elapsed() << '\n';
#endif
      timer.restart();

      if (!is_cluster_self_contained(cluster)) {
        std::cerr << "Non fatal error: extracted cluster not self contained, "
                     "skipping " +
                         std::to_string(index)
                  << '\n';
        return; // skip this component
      }

      adjust_parent_ptrs(cluster);

      auto pruned_cluster = prune_short_branches(cluster, args->voxel_size[0],
                                                 this->args->min_branch_length);

      if (!is_cluster_self_contained(pruned_cluster))
        throw std::runtime_error("Pruned cluster not self contained");

      // auto fixed_cluster = pruned_cluster;
      // if (!args->ignore_multifurcations) {
      // auto fixed_cluster = fix_trifurcations(pruned_cluster);
      //{ // check
      // auto trifurcations = tree_is_valid(fixed_cluster);
      // if (!trifurcations.empty()) {
      // auto soma = fixed_cluster[0];
      // std::cout << "Warning tree in component-" + std::to_string(index)
      //<< " with soma " << soma->x << ' ' << soma->y << ' '
      //<< soma->z << " has trifurcations listed below:\n";
      // rng::for_each(trifurcations, [](auto mismatch) {
      // std::cout << "    " << *mismatch << '\n';
      //});
      //// throw std::runtime_error("Tree has trifurcations" +
      //// std::to_string(index));
      //}
      // if (!is_cluster_self_contained(fixed_cluster)) {
      // std::cout << "Warning a tree in component-" + std::to_string(index)
      //<< " contains at least 1 node with an invalid parent\n";
      //// throw std::runtime_error("Trifurc cluster not self contained" +
      //// std::to_string(index));
      //}
      //}
      //}

      // auto trees = partition_cluster(fixed_cluster);
      trees = partition_cluster(pruned_cluster);

#ifdef LOG
      component_log << "TP, " << timer.elapsed() << '\n';
      component_log << "TP count, " << pruned_cluster.size() << '\n';
#endif
    }

    if (!window_grids.empty()) {
      // the first grid passed from CL sets the bbox for the
      // rest of the output grids
      ImgGrid::Ptr image_grid =
          openvdb::gridPtrCast<ImgGrid>(window_grids.front());
      auto [valued_window_grid, window_bbox] = create_window_grid(
          image_grid, component, component_log, args->voxel_size,
          component_seeds, args->min_window_um, args->output_type == "labels",
          args->expand_window_um);

      // write the first passed window
      auto window_fn = write_output_windows<ImgGrid::Ptr>(
          image_grid, component_dir_fn, component_log, index,
          /*output_vdb*/ false, /*paged*/ args->output_type != "labels",
          window_bbox, /*channel*/ 0);

      // if outputting crops/windows, offset SWCs coords to match window
      bbox = window_bbox;

      // for all other windows passed, skipping channel 0 since already
      // processed above
      rng::for_each(window_grids | rv::enumerate | rv::tail,
                    [&](const auto window_gridp) {
                      auto [channel, window_grid] = window_gridp;
                      auto mask_grid =
                          openvdb::gridPtrCast<openvdb::MaskGrid>(window_grid);
                      // write to disk
                      write_output_windows(
                          mask_grid, component_dir_fn, component_log, index,
                          /*output_vdb*/ false,
                          /*paged*/ args->output_type != "labels", window_bbox,
                          channel);
                    });

      // skip components that are 0s in the original image
      auto mm = vto::minMax(valued_window_grid->tree());
      if (args->run_app2 && (mm.max() > 0)) {
        auto read_timer = high_resolution_timer();
        // protect against possibly empty windows ending up in stats
        if (!window_fn.empty()) {
          // make app2 read the window to get accurate comparison of IO
          read_tiff_paged(window_fn);
#ifdef LOG
          component_log << "Read window time, "
                        << read_timer.elapsed_formatted() << '\n';
#endif
        }

        // for comparison/benchmark/testing purposes
        run_app2(valued_window_grid, component_seeds, component_dir_fn, index,
                 this->args->min_branch_length, component_log,
                 args->window_grid_paths.empty());
      }
    } // end window created if any

#ifdef LOG
    component_log << "Volume, " << bbox.volume() << '\n';
    component_log << "Bounding box, " << bbox << '\n';
#endif

    if (refined_markers_opt) {
#ifdef LOG
      VID_t total_leaves = rng::accumulate(
          trees | rv::transform([](auto tree) { return count_leaves(tree); }),
          0LL);
      VID_t total_furcations =
          rng::accumulate(trees | rv::transform([](auto tree) {
                            return count_furcations(tree);
                          }),
                          0LL);

      component_log << "Final leaf count, " << total_leaves << '\n';
      component_log << "Final branching node count, " << total_furcations
                    << '\n';
#endif

      rng::for_each(trees, [&, this](auto tree) {
        write_swc(tree, this->args->voxel_size, component_dir_fn, bbox,
                  /*bbox_adjust*/ !args->window_grid_paths.empty(),
                  this->args->output_type == "eswc");
        if (!parent_listed_above(tree)) {
          throw std::runtime_error("Tree is not properly sorted");
        }
      });

      std::cout << "Component " << index << " complete and safe to open\n";
    } else {
      std::cout << "Component " << index
                << " SWC timeout, image, seed, (and vdb saved)\n";
    }
  }; // for each component

  auto enum_components = components | rv::enumerate | rng::to_vector;
  tbb::task_arena arena(args->user_thread_count);
  arena.execute(
      [&] { tbb::parallel_for_each(enum_components, process_component); });
  // rng::for_each(enum_components, process_component);

  if (output_topology)
    write_vdb_file({this->topology_grid}, "final-point-grid.vdb");
  std::ofstream run_log;
  run_log.open(log_fn, std::ios::app);
  // only log this if it isn't occluded by app2 and window write times
  if (!(args->run_app2 || !args->window_grid_paths.empty())) {
    run_log << "TC+TP, " << global_timer.elapsed() << '\n';
  }
  run_log << "Aggregated prune, " << total_timer.elapsed_formatted() << '\n';
  run_log << "Neuron count, " << seeds.size() << '\n';
}

template <class image_t> void Recut<image_t>::convert_topology() {
  activate_all_tiles();

  // mutates input_grid
  auto stage = "convert";
  this->update(stage, map_fifo);
}

template <class image_t> void Recut<image_t>::start_run_dir_and_logs() {
  // Warning: if you want to fuse conversion with other stages you need
  // to assign to these variable before running convert, since they
  // rely on output_name and run dir to be set
  if (this->args->convert_only) {
    // reassign output_name from the default
    if (this->args->output_name == "out.vdb") {
      this->args->output_name = get_output_name(args);
    }

    this->run_dir = ".";
    this->log_fn =
        this->run_dir / (this->args->output_name + "-log-" +
                         std::to_string(args->user_thread_count) + ".csv");
#ifdef LOG
    std::ofstream convert_log(this->log_fn);
    convert_log << "Thread count, " << args->user_thread_count << '\n';
    convert_log << "Upsample z, " << args->upsample_z << '\n';
    convert_log << "Final voxel size, " << args->upsample_z << '\n';
    convert_log << "Original voxel count, "
                << coord_prod_accum(this->image_lengths) << '\n';
#endif

    // Reconstructing volume:
  } else {
    this->run_dir = get_unique_fn((fs::path(".") / "run-1").string());
    this->log_fn = this->run_dir / "log.csv";
    fs::create_directories(run_dir);
    std::cout << "All outputs will be written to: " << this->run_dir << '\n';
    std::ofstream run_log(log_fn);
    run_log << "Thread count, " << args->user_thread_count << '\n'
            << "Input: path, " << args->input_path << '\n'
            << "Input: type, " << args->input_type << '\n'
            << "Input: number of channels, " << args->window_grid_paths.size()
            << '\n'
            << "Input: x-axis voxel size in µm, " << args->voxel_size[0] << '\n'
            << "Input: y-axis voxel size in µm, " << args->voxel_size[1] << '\n'
            << "Input: z-axis voxel size in µm, " << args->voxel_size[2] << '\n'
            << "Input: voxel count, " << coord_prod_accum(this->image_lengths)
            << '\n';
    if (args->foreground_percent >= 0) {
      std::ostringstream out;
      out.precision(3);
      out << std::fixed << args->foreground_percent;
      run_log << "Preprocessing: Foreground in %, " << out.str() << '\n';
    } else if (args->background_thresh >= 0) {
      // setting fg would set background value
      // so only log if it was input without a fg %
      run_log << "Preprocessing: Background threshold, "
              << args->background_thresh << '\n';
    }
    run_log << "Output: type, " << args->output_type << '\n';
    if (args->seed_path != "") {
      run_log << "Seed detection: Seeds path, " << args->seed_path << '\n';
    }
    run_log << "Seed detection: morphological operations order, "
            << args->morphological_operations_order << '\n'
            << "Seed detection: morphological operations denoise steps, "
            << args->open_denoise << '\n'
            << "Seed detection: morphological operations close steps, "
            << args->close_steps << '\n'
            << "Seed detection: morphological operations open steps, "
            << args->open_steps << '\n'
            << "Seed detection: min allowed soma radius in µm, "
            << args->min_radius_um << '\n'
            << "Seed detection: max allowed soma radius in µm, "
            << args->max_radius_um << '\n'
            << "Skeletonization: neurites mean shift radius, "
            << args->mean_shift_factor.value_or(0) << '\n'
            << "Skeletonization: neurites prune radius, "
            << args->prune_radius.value_or(0) << '\n'
            << "Skeletonization: soma prune radius factor, "
            << SOMA_PRUNE_RADIUS << '\n'
            << "Skeletonization: min branch length µm, "
            << args->min_branch_length << '\n'
            << "Benchmarking: run app2, " << args->run_app2 << '\n';
    run_log.flush();
  }
}

template <class image_t> void Recut<image_t>::operator()() {

  // Thread count is enforced across recut and dependencies (openvdb)
  // as long as control object is alive
  std::unique_ptr<tbb::global_control> control;
#ifdef LOG
  std::cout << "Global max thread count " << args->user_thread_count << '\n';
#endif
  control.reset(new tbb::global_control(
      tbb::global_control::max_allowed_parallelism, args->user_thread_count));

  if (!args->second_grid.empty()) {
    // simply combine the passed grids then exit program immediately
    combine_grids(args->input_path.generic_string(), args->second_grid,
                  this->args->output_name);
    return;
  }

  // process the input args and parameters
  this->initialize();

  start_run_dir_and_logs();

  // if point.vdb was not already set by input
  if (!this->input_is_vdb) {
    // if (!args->convert_only) {
    // if (this->args->output_type != "point") {
    // throw std::runtime_error(
    //"If running reconstruction, output type must be type point");
    //}
    //}

    if (args->convert_only) {
      // converts to whatever output_type specifies
      convert_topology();

      openvdb::GridPtrVec grids;
      if (args->output_type == "float") {
        print_grid_metadata(this->input_grid);
        grids.push_back(this->input_grid);
      } else if (args->output_type == "uint8") {
        print_grid_metadata(this->img_grid);
        grids.push_back(this->img_grid);
      } else if (args->output_type == "mask") {
        print_grid_metadata(this->mask_grid);
        grids.push_back(this->mask_grid);
      } else if (args->output_type == "point") {
        print_grid_metadata(this->topology_grid);
        grids.push_back(this->topology_grid);
      }
      write_vdb_file(grids, this->args->output_name);
      // no more work to do, exiting
      return;

    } else { // fully reconstruct the image

      auto final_output_type = args->output_type;

      //// temporarily set output type to create necessary grids
      // if (args->seed_path.empty()) {
      //  if generating seeds then convert to mask first
      args->output_type = "mask";
      convert_topology();
      assertm(this->mask_grid, "Mask grid not properly set");
      // sets to this->mask_grid instead of reading from file
      //} else {
      // args->output_type = "point";
      // convert_topology();
      // append_attributes(this->topology_grid);
      //}

      // reset it
      args->output_type = final_output_type;
    }

    // set the z tile lengths to equallying whole image for reconstruction
    this->tile_lengths[2] = this->image_lengths[2];
    update_hierarchical_dims(this->tile_lengths);
    // assume starting from VDB for rest of run
    this->input_is_vdb = true;
  }

  auto [seeds, soma_sdf, neurite_sdf] =
      soma_segmentation(mask_grid, args, this->image_lengths, log_fn, run_dir);

  if (this->args->output_type == "labels") {
    exit(0); // exit
  } else if (seeds.empty()) {
    std::cerr
        << "No somas found, possibly make --open-steps lower or --close-steps "
           "higher (if using membrane labeling), also consider raising the fg "
           "percent. Note that passing --seeds forces all found somas to be "
           "filtered against those you specified, exiting...\n";
    exit(1);
  } else if (this->args->output_type == "seeds") {
    exit(0); // exit
  }

// build full SDF by extending known somas into reachable neurites
#ifdef LOG
  std::cout << "\tmasking step\n";
#endif
  auto timer = high_resolution_timer();
  // if user passed known seeds then use the pretermined merged sdf grid
  // else use the result of the open and close steps above
  auto somas_connected_to_neurites = vto::maskSdf(*soma_sdf, *neurite_sdf);
  //auto somas_connected_to_neurites = vto::levelSetRebuild(*temp);
  //auto somas_connected_to_neurites = vto::fogToSdf(*temp, 0);

  std::ofstream run_log;
  run_log.open(log_fn, std::ios::app);
  run_log << "Seed detection: masking time, " << timer.elapsed_formatted()
          << '\n'
          << "Seed detection: masked SDF voxel count, "
          << somas_connected_to_neurites->activeVoxelCount() << '\n';
  run_log.flush();

#ifdef LOG
  std::cout << "\tTopology to tree step\n";
#endif
  topology_to_tree(somas_connected_to_neurites, this->run_dir,
                   this->args->save_vdbs);

#ifdef LOG
  std::cout << "\tSDF to point step\n";
#endif
  assertm(somas_connected_to_neurites,
          "Topology grid must be set before starting reconstruction");
  this->topology_grid = convert_sdf_to_points(
      somas_connected_to_neurites, image_lengths, args->foreground_percent);

  initialize_globals(this->grid_tile_size, this->tile_block_size);

  // constrain topology to only those reachable from roots
  auto stage = "connected";
  {
    // starting from the roots connected stage saves all surface vertices into
    // fifo
    this->activate_vids(this->topology_grid, seeds, stage, this->map_fifo,
                        this->connected_map);
    // first stage of the pipeline
    this->update(stage, map_fifo);
  }

  // radius stage will consume fifo surface vertices
  {
    stage = "radius";
    this->setup_radius(map_fifo);
    this->update(stage, map_fifo);
    // redefine soma radii based off info read in original files
    adjust_soma_radii(seeds, this->topology_grid);
  }

  partition_components(seeds, false);
  exit(0);

  // old prune strategy
  //{
  //// starting from roots, prune stage will
  //// create final list of vertices
  // if (true) {
  // stage = "prune";
  // this->activate_vids(this->topology_grid, seeds, stage,
  // this->map_fifo, this->connected_map); this->update(stage, map_fifo);
  //// make all unpruned trace a back to a root
  //// any time you remove a node you need to ensure tree validity
  // adjust_parent();
  //}

  // prune_radii();
  // adjust_parent();

  //// produces bad reach-back artifact
  //// prune_branch();
  //// adjust_parent();

  // print_to_swc();
  //}
}
