#pragma once

#ifdef DENSE
#include "grid.hpp"
class Grid;
#else
#include "vertex_attr.hpp"
#endif

#include "recut_parameters.hpp"
#include "tile_thresholds.hpp"
#include <algorithm>
#include <bits/stdc++.h>
#include <bitset>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <openvdb/tools/Composite.h>
#include <set>
#include <type_traits>
#include <unistd.h>
#include <unordered_set>

// taskflow significantly increases load times, avoid loading it if possible
#ifdef TF
#include <taskflow/taskflow.hpp>
#endif

struct InstrumentedUpdateStatistics {
  int iterations;
  double total_time;
  double computation_time;
  double io_time;
  std::vector<uint16_t> interval_open_counts;

  // These refer to the FIFO sizes
  std::vector<uint64_t> max_sizes;
  std::vector<uint64_t> mean_sizes;

  InstrumentedUpdateStatistics(int iterations, double total_time,
                               double computation_time, double io_time,
                               std::vector<uint16_t> interval_open_counts)
      : iterations(iterations), total_time(total_time),
        computation_time(computation_time), io_time(io_time),
        interval_open_counts(interval_open_counts) {}
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
  VID_t image_size, grid_interval_size, interval_block_size;

  GridCoord image_lengths;
  GridCoord image_offsets;
  GridCoord interval_lengths;
  GridCoord block_lengths;
  GridCoord grid_interval_lengths;
  GridCoord interval_block_lengths;

#ifdef DENSE
  Grid grid;
  VID_t pad_block_length_x, pad_block_length_y, pad_block_length_z,
      pad_block_offset,
#ifdef USE_MMAP
      bool mmap_ = true;
#else
      bool mmap_ = false;
#endif
#endif

  bool input_is_vdb;
#ifdef USE_VDB
  EnlargedPointDataGrid::Ptr topology_grid;
#endif

  std::vector<OffsetCoord> const stencil{
      new_offset_coord(0, 0, -1), new_offset_coord(0, 0, 1),
      new_offset_coord(0, -1, 0), new_offset_coord(0, 1, 0),
      new_offset_coord(-1, 0, 0), new_offset_coord(1, 0, 0)};

  std::ofstream out;
  image_t *generated_image = nullptr;
  atomic<VID_t> global_revisits;
  RecutCommandLineArgs *args;
  RecutParameters *params;
  std::vector<std::vector<std::deque<VertexAttr>>> global_fifo;
  std::vector<std::vector<std::deque<VertexAttr>>> local_fifo;
  std::vector<std::vector<std::vector<VertexAttr>>> active_vertices;

  // interval specific global data structures
  vector<bool> active_intervals;
  vector<vector<atomwrapper<bool>>> active_blocks;
  vector<vector<atomwrapper<bool>>> processing_blocks;
  vector<vector<vector<bool>>> active_neighbors;
  vector<vector<map<VID_t, vector<VertexAttr>>>> updated_ghost_vec;

  Recut(RecutCommandLineArgs &args)
      : args(&args), params(&(args.recut_parameters())) {}

  void operator()();

  // to destroy the information for this run
  // so that it doesn't affect the next run
  // the vertices must be unmapped
  // done via `release()`
#ifdef DENSE
  inline void release() { grid.Release(); }
  inline VertexAttr *get_vertex_vid(VID_t interval_id, VID_t block_id,
                                    VID_t vid, VID_t *output_offset);
  template <typename vertex_t>
  void brute_force_extract(vector<vertex_t> &outtree, bool accept_band = false,
                           bool release_intervals = true);
  inline void get_block_coord(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  void get_interval_coord(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline void get_img_coord(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
#endif
  void initialize_globals(const VID_t &grid_interval_size,
                          const VID_t &interval_block_size);

  bool filter_by_vid(VID_t vid, VID_t find_interval_id, VID_t find_block_id);
  bool filter_by_label(VertexAttr *v, bool accept_band);
  void adjust_parent(bool to_swc_file);

  image_t get_img_val(const image_t *tile, GridCoord coord);
  template <typename T> bool get_vdb_val(T accessor, GridCoord coord);
  inline VID_t rotate_index(VID_t img_coord, const VID_t current,
                            const VID_t neighbor,
                            const VID_t interval_block_size,
                            const VID_t pad_block_size);
  int get_bkg_threshold(const image_t *tile, VID_t interval_vertex_size,
                        const double foreground_percent);
  inline VertexAttr *get_or_set_active_vertex(const VID_t interval_id,
                                              const VID_t block_id,
                                              const OffsetCoord offset,
                                              bool &found);
  inline VertexAttr *get_active_vertex(const VID_t interval_id,
                                       const VID_t block_id,
                                       const OffsetCoord offset);
  void place_vertex(const VID_t nb_interval_id, VID_t block_id, VID_t nb,
                    struct VertexAttr *dst, OffsetCoord msg_offsets,
                    std::string stage);
  bool check_blocks_finish(VID_t interval_id);
  bool are_intervals_finished();
  void activate_all_intervals();

  // indexing helpers that use specific global lengths from recut
  GridCoord id_interval_to_img_offsets(const VID_t interval_id);
  GridCoord id_block_to_interval_offsets(const VID_t block_id);
  inline VID_t id_img_to_block_id(const VID_t id);
  VID_t id_img_to_interval_id(const VID_t id);
  template <typename T> VID_t coord_img_to_block_id(T coord);
  template <typename T> VID_t coord_img_to_interval_id(T coord);
  GridCoord id_interval_block_to_img_offsets(VID_t interval_id, VID_t block_id);
  VID_t v_to_img_id(VID_t interval_id, VID_t block_id, VertexAttr *v);
  GridCoord v_to_img_coord(VID_t interval_id, VID_t block_id, VertexAttr *v);
  OffsetCoord v_to_off(VID_t interval_id, VID_t block_id, VertexAttr *v);

  void check_ghost_update(VID_t interval_id, VID_t block_id,
                          GridCoord dst_coord, VertexAttr *dst,
                          std::string stage);
  int get_parent_code(VID_t dst_id, VID_t src_id);
  template <typename T, typename T2, typename FlagsT, typename ParentsT>
  bool accumulate_connected(const image_t *tile, VID_t interval_id,
                            VID_t block_id, GridCoord dst_coord, T2 ind,
                            OffsetCoord offset_to_current, VID_t &revisits,
                            const TileThresholds<image_t> *tile_thresholds,
                            bool &found_adjacent_invalid, T vdb_accessor,
                            FlagsT flags_handle, ParentsT parents_handle);
  bool accumulate_value(const image_t *tile, VID_t interval_id,
                        GridCoord dst_coord, VID_t block_id,
                        struct VertexAttr *current, VID_t &revisits,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_background);
  bool is_covered_by_parent(VID_t interval_id, VID_t block_id,
                            VertexAttr *current);
  template <class Container, typename T, typename T2, typename IndT,
            typename RadiusT, typename FlagsT>
  void accumulate_prune(VID_t interval_id, VID_t block_id, GridCoord dst_coord,
                        IndT ind, T current, T2 current_vid,
                        bool current_unvisited, Container &fifo,
                        RadiusT radius_handle, FlagsT flags_handle);
  template <class Container, typename T, typename T2, typename FlagsT,
            typename RadiusT>
  void accumulate_radius(VID_t interval_id, VID_t block_id, GridCoord dst_coord,
                         T ind, T2 current_radius, Container &fifo,
                         FlagsT flags_handle, RadiusT radius_handle);
  template <class Container>
  void integrate_updated_ghost(const VID_t interval_id, const VID_t block_id,
                               std::string stage, Container &fifo);
  template <class Container> void dump_buffer(Container buffer);
  void adjust_vertex_parent(VertexAttr *vertex, GridCoord start_offsets);
  template <class Container>
  bool integrate_vertex(const VID_t interval_id, const VID_t block_id,
                        struct VertexAttr *updated_vertex,
                        bool ignore_KNOWN_NEW, std::string stage,
                        Container &fifo);
  template <class Container, typename T, typename T2>
  void march_narrow_band(const image_t *tile, VID_t interval_id, VID_t block_id,
                         std::string stage,
                         const TileThresholds<image_t> *tile_thresholds,
                         Container &fifo, T vdb_accessor, T2 leaf_iter);
  template <class Container, typename T, typename T2>
  void connected_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                      GridCoord offsets, std::string stage,
                      const TileThresholds<image_t> *tile_thresholds,
                      Container &fifo, VID_t revisits, T vdb_accessor,
                      T2 leaf_iter);
  template <class Container, typename T, typename T2>
  void radius_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                   GridCoord offsets, std::string stage,
                   const TileThresholds<image_t> *tile_thresholds,
                   Container &fifo, VID_t revisits, T vdb_accessor,
                   T2 leaf_iter);
  template <class Container, typename T, typename T2>
  void prune_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                  GridCoord offsets, std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T vdb_accessor,
                  T2 leaf_iter);
  void create_march_thread(VID_t interval_id, VID_t block_id);
#ifdef USE_MCP3D
  void load_tile(VID_t interval_id, mcp3d::MImage &mcp3d_tile);
  TileThresholds<image_t> *get_tile_thresholds(mcp3d::MImage &mcp3d_tile);
#endif
  template <class Container, typename T>
  std::atomic<double>
  process_interval(VID_t interval_id, const image_t *tile, std::string stage,
                   const TileThresholds<image_t> *tile_thresholds,
                   Container &fifo, T vdb_accessor);
  template <class Container>
  std::unique_ptr<InstrumentedUpdateStatistics>
  update(std::string stage, Container &fifo = nullptr,
         TileThresholds<image_t> *tile_thresholds = nullptr);
  GridCoord get_input_image_extents(bool force_regenerate_image,
                                    RecutCommandLineArgs *args);
  GridCoord get_input_image_lengths(bool force_regenerate_image,
                                    RecutCommandLineArgs *args);
  const std::vector<VID_t> initialize();
  template <typename vertex_t>
  void convert_to_markers(std::vector<vertex_t> &outtree,
                          bool accept_band = false);
  inline VID_t sub_block_to_block_id(VID_t iblock, VID_t jblock, VID_t kblock);
  template <class Container>
  void print_interval(VID_t interval_id, std::string stage, Container &fifo);
  template <class Container>
  void print_grid(std::string stage, Container &fifo);
  template <class Container> void setup_radius(Container &fifo);
  template <class Container>
  void activate_vids(const std::vector<VID_t> &root_vids, std::string stage,
                     Container &fifo);
  std::vector<VID_t> process_marker_dir(GridCoord grid_offsets,
                                        GridCoord grid_extents);
  void print_vertex(VID_t interval_id, VID_t block_id, VertexAttr *current,
                    GridCoord offsets);
  void set_parent_non_branch(const VID_t interval_id, const VID_t block_id,
                             VertexAttr *dst, VertexAttr *potential_new_parent);
  ~Recut<image_t>();
};

template <class image_t> Recut<image_t>::~Recut<image_t>() {
  if (this->params->force_regenerate_image) {
    // when initialize has been run
    // generated_image is no longer nullptr
    if (this->generated_image) {
      delete[] this->generated_image;
    }
  }
}

template <class image_t>
GridCoord Recut<image_t>::id_interval_to_img_offsets(const VID_t interval_id) {
  // return coord_add(
  // this->image_offsets,
  return coord_prod(id_to_coord(interval_id, this->grid_interval_lengths),
                    this->interval_lengths);
}

template <class image_t>
GridCoord Recut<image_t>::id_block_to_interval_offsets(const VID_t block_id) {
  // return coord_add(
  // this->image_offsets,
  return coord_prod(id_to_coord(block_id, this->interval_block_lengths),
                    this->block_lengths);
}

/**
 * returns interval_id, the interval domain this vertex belongs to
 * does not consider overlap of ghost regions
 */
template <class image_t>
template <typename T>
VID_t Recut<image_t>::coord_img_to_interval_id(T coord) {
  return coord_to_id(coord_div(coord, this->interval_lengths),
                     this->grid_interval_lengths);
}

/**
 * id : linear idx relative to unpadded image
 * returns interval_id, the interval domain this vertex belongs to
 * with respect to the original unpadded image
 * does not consider overlap of ghost regions
 */
template <class image_t>
VID_t Recut<image_t>::id_img_to_interval_id(const VID_t id) {
  return coord_img_to_interval_id(id_to_coord(id, this->image_lengths));
}

template <class image_t>
template <typename T>
VID_t Recut<image_t>::coord_img_to_block_id(T coord) {
  return coord_to_id(
      coord_div(coord_mod(coord, this->interval_lengths), this->block_lengths),
      this->interval_block_lengths);
}

/**
 * all block_nums are a linear row-wise idx, relative to current interval
 * vid : linear idx into the full domain inimg1d
 * the interval contributions are modded away
 * such that all block_nums are relative to a single
 * interval
 * returns block_id, the block domain this vertex belongs
 * in one of the intervals
 * Note: block_nums are renumbered within each interval
 * does not consider overlap of ghost regions
 */
template <class image_t>
VID_t Recut<image_t>::id_img_to_block_id(const VID_t vid) {
  return coord_img_to_block_id(id_to_coord(vid, this->image_lengths));
}

// first coord of block with respect to whole image
template <typename image_t>
GridCoord Recut<image_t>::id_interval_block_to_img_offsets(VID_t interval_id,
                                                           VID_t block_id) {
  return coord_add(id_interval_to_img_offsets(interval_id),
                   id_block_to_interval_offsets(block_id));
}

// recompute the vertex vid
template <typename image_t>
GridCoord Recut<image_t>::v_to_img_coord(VID_t interval_id, VID_t block_id,
                                         VertexAttr *v) {
  return coord_add(v->offsets,
                   id_interval_block_to_img_offsets(interval_id, block_id));
}

// recompute the vertex vid
template <typename image_t>
VID_t Recut<image_t>::v_to_img_id(VID_t interval_id, VID_t block_id,
                                  VertexAttr *v) {
  return coord_to_id(v_to_img_coord(interval_id, block_id, v));
}

// recompute the vertex vid
template <typename image_t>
OffsetCoord Recut<image_t>::v_to_off(VID_t interval_id, VID_t block_id,
                                     VertexAttr *v) {
  return coord_mod(v_to_img_coord(interval_id, block_id, v),
                   this->block_lengths);
}

// adds all markers to root_vids
//
template <class image_t>
std::vector<VID_t>
Recut<image_t>::process_marker_dir(const GridCoord grid_offsets,
                                   const GridCoord grid_extents) {
  std::vector<VID_t> root_vids;

  if (params->marker_file_path().empty())
    return root_vids;

  // allow either dir or dir/ naming styles
  if (params->marker_file_path().back() != '/')
    params->set_marker_file_path(params->marker_file_path().append("/"));

  cout << "marker dir path: " << params->marker_file_path() << '\n';
  assertm(fs::exists(params->marker_file_path()),
          "Marker file path must exist");

  std::vector<MyMarker> inmarkers;
  for (const auto &marker_file :
       fs::directory_iterator(params->marker_file_path())) {
    const auto marker_name = marker_file.path().filename().string();
    const auto full_marker_name = params->marker_file_path() + marker_name;
    inmarkers = readMarker_file(full_marker_name);

    // set intervals with root present as active
    for (auto &root : inmarkers) {
      auto adjusted = coord_add(new_grid_coord(root.x, root.y, root.z), ones());

      if (!(is_in_bounds(adjusted, grid_offsets, grid_extents)))
        continue;

      root_vids.push_back(coord_to_id(adjusted, this->image_lengths));

#ifdef FULL_PRINT
      cout << "Using marker at " << coord_to_str(adjusted) << '\n';
#endif
    }
  }
  return root_vids;
}

// activates
// the intervals of the leaf and reads
// them to the respective heaps
template <class image_t>
template <class Container>
void Recut<image_t>::setup_radius(Container &fifo) {
  for (size_t interval_id = 0; interval_id < grid_interval_size;
       ++interval_id) {
    for (size_t block_id = 0; block_id < interval_block_size; ++block_id) {
      if (!(fifo[interval_id][block_id].empty())) {
        active_intervals[interval_id] = true;
        active_blocks[interval_id][block_id].store(true);
#ifdef FULL_PRINT
        cout << "Set interval " << interval_id << " block " << block_id
             << " to active\n";
#endif
      }
    }
  }
}

// activates
// the intervals of the root and reads
// them to the respective heaps
template <class image_t>
template <class Container>
void Recut<image_t>::activate_vids(const std::vector<VID_t> &root_vids,
                                   std::string stage, Container &fifo) {
  assertm(!(root_vids.empty()), "Must have at least one root");

  if (stage == "connected") {
    init_root_attributes(this->topology_grid,
                         ids_to_coords(root_vids, this->image_lengths));
  }

  for (const auto &vid : root_vids) {
    auto interval_id = id_img_to_interval_id(vid);
    auto block_id = id_img_to_block_id(vid);

    auto coord = id_to_coord(vid, this->image_lengths);
#ifdef FULL_PRINT
    cout << "activate_vids(): setting interval " << interval_id << " block "
         << block_id << " to active ";
    cout << "for marker " << coord << '\n';
#endif

    active_intervals[interval_id] = true;
    active_blocks[interval_id][block_id].store(true);

    VertexAttr *msg_vertex;
    auto offsets = coord_mod(coord, this->block_lengths);
    if (stage == "connected") {
      // place a root with proper vid and parent of itself
      msg_vertex = &(this->active_vertices[interval_id][block_id].emplace_back(
          /*edge_state*/ 0, offsets, zeros_off()));
      this->local_fifo[interval_id][block_id].emplace_back(
          /*edge_state*/ 0, offsets, zeros_off());
    } else if (stage == "prune") {

#ifdef DENSE
      msg_vertex = get_vertex_vid(interval_id, block_id, coord, nullptr);
#else
      msg_vertex = get_active_vertex(interval_id, block_id, offsets);
      assertm(msg_vertex != nullptr, "get_active_vertex yielded nullptr");
#endif
      assertm(msg_vertex->valid_radius(),
              "activate vids didn't find valid radius");
      fifo[interval_id][block_id].emplace_back(/*edge_state*/ 0, offsets,
                                               zeros_off());
    } else {
      assertm(false, "stage behavior not yet specified");
    }

    // place ghost update accounts for
    // edges of intervals in addition to blocks
    // this only adds to update_ghost_vec if the root happens
    // to be on a boundary
    check_ghost_update(interval_id, block_id, coord, msg_vertex,
                       stage); // add to any other ghost zone blocks
  }
}

template <class image_t>
template <class Container>
void Recut<image_t>::print_grid(std::string stage, Container &fifo) {
  for (size_t interval_id = 0; interval_id < grid_interval_size;
       ++interval_id) {
    print_interval(interval_id, stage, fifo[interval_id]);
  }
}

template <class image_t>
template <class Container>
void Recut<image_t>::print_interval(VID_t interval_id, std::string stage,
                                    Container &fifo) {
#ifdef DENSE
  auto interval = grid.GetInterval(interval_id);

  if (interval->IsInMemory()) {
#ifdef LOG
    cout << "Print recut interval " << interval_id << " stage: " << stage
         << '\n';
#endif
  } else {
    if (this->mmap_) {
#ifdef LOG
      cout << "Recut interval " << interval_id << " stage: " << stage
           << " never loaded during run, skipping...\n";
#endif
      return;
    }
    assertm(!(interval->IsInMemory()), "can't reload into memory");
    interval->LoadFromDisk();
#ifdef LOG
    cout << "Recut interval " << interval_id << " stage: " << stage
         << " loaded during print\n";
#endif
  }
#endif // DENSE

  // these looping vars below are over the non-padded lengths of each interval
  auto interval_coord = id_to_coord(interval_id, this->grid_interval_lengths);
  auto adjusted = coord_prod(interval_coord, this->interval_lengths);

  for (int zi = adjusted[2]; zi < adjusted[2] + this->interval_lengths[2];
       zi++) {
    cout << "Z=" << zi << '\n';
    cout << "  | ";
    for (int xi = adjusted[0]; xi < adjusted[0] + this->interval_lengths[0];
         xi++) {
      cout << xi << " ";
    }
    cout << '\n';
    for (int xi = 0; xi < 2 * this->interval_lengths[1] + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = adjusted[0]; yi < adjusted[0] + this->interval_lengths[1];
         yi++) {
      cout << yi << " | ";
      for (int xi = adjusted[0]; xi < adjusted[0] + this->interval_lengths[0];
           xi++) {
        auto coord = new_grid_coord(xi, yi, zi);
        auto block_id = coord_img_to_block_id(coord);

#ifdef DENSE
        auto vid = coord_to_id(coord, this->image_lengths);
        auto v = get_vertex_vid(interval_id, block_id, vid, nullptr);
#else
        auto offsets = coord_mod(coord, this->block_lengths);
        auto v = get_active_vertex(interval_id, block_id, offsets);
        if (v == nullptr) {
          cout << "- ";
          continue;
        }
#endif

        if (stage == "radius") {
          if (v->valid_radius()) {
            cout << +(v->radius) << " ";
          } else {
            cout << "- ";
          }
        } else if (stage == "parent") {
          if (v->valid_parent()) {
            cout << v->parent << " ";
          } else {
            cout << "- ";
          }
        } else if (stage == "surface") {
          if (v->surface()) {
            cout << "L "; // L for leaf and because disambiguates selected S
          } else {
            cout << "- ";
          }
        } else if (stage == "label" || stage == "connected") {
          cout << v->label() << ' ';
        }
      }
      cout << '\n';
    }
    cout << '\n';
  }
}

template <typename T, typename T2> T absdiff(const T &lhs, const T2 &rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

template <class image_t>
template <typename T>
bool Recut<image_t>::get_vdb_val(T vdb_accessor, GridCoord coord) {
#ifdef FULL_PRINT
  cout << coord_to_str(coord) << '\n';
#endif
  return vdb_accessor.isValueOn(coord);
}

/*
 * Takes a grid coord
 * and converts it to linear index of current
 * tile buffer of interval currently processed
 * returning the value
 */
template <class image_t>
image_t Recut<image_t>::get_img_val(const image_t *tile, GridCoord coord) {
  // force_regenerate_image passes the whole image as the
  // tile so the img vid is the correct address regardless
  // of interval length sizes Note that force_regenerate_image
  // is mostly used in test cases to try different scenarios
  if (this->params->force_regenerate_image) {
    return tile[coord_to_id(coord, this->image_lengths)];
  }

  auto buffer_coord = coord_to_id(coord_mod(coord, this->interval_lengths),
                                  this->interval_lengths);
  return tile[buffer_coord];
}

/*
template <typename image_t>
bool Recut<image_t>::is_covered_by_parent(VID_t interval_id, VID_t block_id,
                                          VertexAttr *current) {
  VID_t i, j, k, pi, pj, pk;
  get_img_coord(current->vid, i, j, k);
  get_img_coord(current->parent, pi, pj, pk);
  auto x = static_cast<double>(i) - pi;
  auto y = static_cast<double>(j) - pj;
  auto z = static_cast<double>(k) - pk;
  auto vdistance = sqrt(x * x + y * y + z * z);

  auto parent_interval_id = id_img_to_interval_id(current->parent);
  auto parent_block_id = id_img_to_block_id(current->parent);

#ifdef DENSE
  auto parent = get_vertex_vid(parent_interval_id, parent_block_id,
                               current->parent, nullptr);
#else
  auto parent =
      get_active_vertex(parent_interval_id, parent_block_id, current->parent);
#endif

  if (static_cast<double>(parent->radius) >= vdistance) {
    return true;
  }
  return false;
}

template <typename image_t>
void Recut<image_t>::set_parent_non_branch(const VID_t interval_id,
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
#ifdef DENSE
      potential_new_parent = get_vertex_vid(
          interval_id, block_id, potential_new_parent->parent, nullptr);
#else
      potential_new_parent = get_active_vertex(interval_id, block_id,
                                               potential_new_parent->parent);
#endif
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
template <class Container, typename T, typename T2, typename IndT,
          typename RadiusT, typename FlagsT>
void Recut<image_t>::accumulate_prune(VID_t interval_id, VID_t block_id,
                                      GridCoord dst_coord, IndT ind, T current,
                                      T2 current_vid, bool current_unvisited,
                                      Container &fifo, RadiusT radius_handle,
                                      FlagsT flags_handle) {

#ifdef DENSE
  auto dst = get_vertex_vid(interval_id, block_id, dst_id, nullptr);
#else
  auto dst = get_active_vertex(interval_id, block_id,
                               coord_mod(dst_coord, this->block_lengths));
#endif
  if (dst == nullptr) { // never selected
    return;
  }

  auto add_prune_dst = [&]() {
    dst->prune_visit();
    fifo.push_back(*dst);
    check_ghost_update(interval_id, block_id, dst_coord, dst, "prune");
#ifdef FULL_PRINT
    std::cout << "  added dst " << dst_coord << " rad " << +(dst->radius)
              << '\n';
#endif
  };

  // check if dst is covered by current
  // dst can only be 1 hop away (adjacent) from current, therefore
  // all radii greater than 1 imply some redundancy in coverage
  // but this may be desired with DILATION_FACTOR higher than 1
  if (current->radius >= DILATION_FACTOR) {
    auto dst_was_updated = false;
    // dst itself can be used to pass messages
    // like modified radius and prune status
    // to other blocks / intervals
    auto update_radius = current->radius - 1;
    if ((!dst->prune_visited()) && (update_radius < dst->radius)) {
      // previously pruned vertex can transmits transitive
      // coverage info
      dst->radius = update_radius;
      dst_was_updated = true;
    }

    // dst should be covered by current
    // if it hasn't already by pruned
    if (!(dst->root() || dst->unvisited())) {
      dst->mark_unvisited();
      dst_was_updated = true;
    }

    if (dst_was_updated) {
#ifdef FULL_PRINT
      std::cout << "current covers  radius of: " << +(dst->radius) << " at dst "
                << dst_coord << " " << dst->label();
#endif
      add_prune_dst();
    }
  } else {

    // even if dst is not covered if it's already been
    // pruned or visited there's no more work to do
    if (!(dst->unvisited() || dst->prune_visited())) {
      add_prune_dst();
    }
  }
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current interval_id and block_id, since
 * it is potentially adding these vertices to the unique heap of interval_id
 * and block_id. only one parent when selected. If one of these vertexes on
 * the edge but still within interval_id and block_id domain is updated it
 * is the responsibility of check_ghost_update to take note of the update such
 * that this update is propagated to the relevant interval and block see
 * integrate_updated_ghost(). dst_coord : continuous vertex id VID_t of the dst
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <class Container, typename T, typename T2, typename FlagsT,
          typename RadiusT>
void Recut<image_t>::accumulate_radius(VID_t interval_id, VID_t block_id,
                                       GridCoord dst_coord, T ind,
                                       T2 current_radius, Container &fifo,
                                       FlagsT flags_handle,
                                       RadiusT radius_handle) {

  // note the current vertex can belong in the boundary
  // region of a separate block /interval and is only
  // within this block /interval's ghost region
  // therefore all neighbors / destinations of current
  // must be checked to make sure they protude into
  // the actual current block / interval region
  // current vertex is not always within this block and interval
  // and each block, interval have a ghost region
  // after filter in scatter this pointer arithmetic is always valid
#ifdef DENSE
  auto dst = get_vertex_vid(interval_id, block_id, dst_id, nullptr);
#else
  std::cout << "check dst: " << coord_to_str(dst_coord) << '\n';
  auto dst = get_active_vertex(interval_id, block_id,
                               coord_mod(dst_coord, this->block_lengths));
  if (dst == nullptr) {
    return;
  }
  std::cout << "\tdst not background\n";
#endif

  uint8_t updated_radius = 1 + current_radius;

  assertm((is_selected(flags_handle, ind) || is_root(flags_handle, ind)) ==
              ((dst->selected() || dst->root())),
          "don't match");

  if (dst->selected() || dst->root()) {

    // if radius not set yet it necessitates it is 1 higher than current OR an
    // update from another block / interval creates new lower updates
    // if (!(dst->valid_radius()) || (dst->radius > updated_radius)) {
    if (!(valid_radius(radius_handle, ind)) ||
        (radius_handle.get(*ind) > updated_radius)) {
#ifdef FULL_PRINT
      cout << "\tAdjacent higher at " << coord_to_str(dst_coord) << " label "
           << dst->label() << " radius " << +(dst->radius) << " current radius "
           << +(current_radius) << '\n';
#endif
      dst->radius = updated_radius;
      radius_handle.set(*ind, updated_radius);
      // construct a dst message
      fifo.push_back(*dst);
      check_ghost_update(interval_id, block_id, dst_coord, dst, "radius");
    }
  } else {
    assertm(false, "\tunselected neighbor was found");
  }
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current interval_id and block_id, since
 * it is potentially adding these vertices to the unique heap of interval_id
 * and block_id. only one parent when selected. If one of these vertexes on
 * the edge but still within interval_id and block_id domain is updated it
 * is the responsibility of check_ghost_update to take note of the update such
 * that this update is propagated to the relevant interval and block see
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <typename T, typename T2, typename FlagsT, typename ParentsT>
bool Recut<image_t>::accumulate_connected(
    const image_t *tile, VID_t interval_id, VID_t block_id, GridCoord dst_coord,
    T2 ind, OffsetCoord offset_to_current, VID_t &revisits,
    const TileThresholds<image_t> *tile_thresholds,
    bool &found_adjacent_invalid, T vdb_accessor, FlagsT flags_handle,
    ParentsT parents_handle) {

#ifdef FULL_PRINT
  cout << "\tcheck dst: " << coord_to_str(dst_coord);
  cout << " bkg_thresh " << +(tile_thresholds->bkg_thresh) << '\n';
#endif

  // skip backgrounds
  // the image voxel of this dst vertex is the primary method to exclude this
  // pixel/vertex for the remainder of all processing
  auto found_background = false;
  if (this->input_is_vdb) {
    auto dst_vox = get_vdb_val(vdb_accessor, dst_coord);
    if (!dst_vox)
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
#ifdef DENSE
  auto dst = get_vertex_vid(interval_id, block_id, dst_id, nullptr);
#else
  bool found;
  // new vertices automatically set as selected
  // this will invalidate any previous refs or iterators returned of active
  // vertices
  auto dst = get_or_set_active_vertex(
      interval_id, block_id, coord_mod(dst_coord, this->block_lengths), found);
  set_selected(flags_handle, ind);

  // skip already selected vertices too
  if (found) {
    revisits += 1;
    return false;
  }

#ifdef FULL_PRINT
  cout << "\tadded new dst to active set, vid: " << dst_coord << '\n';
#endif
#endif

  // ensure traces a path back to root in case no prune stage
  // parent will likely be mutated during prune stage
  dst->set_parent(offset_to_current);
  parents_handle.set(*ind, offset_to_current);

  local_fifo[interval_id][block_id].push_back(*dst);
  check_ghost_update(interval_id, block_id, dst_coord, dst, "connected");

  return true;
}

/*
 * this will place necessary updates towards regions in outside blocks
 * or intervals safely by leveraging the container of copies of the
 * updated vertices updated_ghost_vec. Vertices themselves act as
 * messages, which can be redundant or repeated when sent to their
 * destination domain since updated_ghost_vec acts like a fifo
 */
template <class image_t>
void Recut<image_t>::place_vertex(const VID_t nb_interval_id,
                                  const VID_t block_id, const VID_t nb_block_id,
                                  struct VertexAttr *dst,
                                  OffsetCoord msg_offsets, std::string stage) {

  // ASYNC option means that another block and thread can be started during
  // the processing of the current thread. If ASYNC is not defined then simply
  // save all possible updates for checking in the integrate_updated_ghost()
  // function after all blocks have been marched
#ifdef ASYNC
  if (stage != "value") {
    assertm(false, "not currently adjusted to work for non-value stage");
  }
  // if not currently processing, set atomically set to true and
  // potentially modify a neighbors heap
  // if neighbors heap is modified it will cause an async launch of a thread
  // during the processing of the current thread
#ifdef DENSE
  // mmap counts all intervals as in memory
  if (grid.GetInterval(nb_interval_id)->IsInMemory() &&
      processing_blocks[nb_interval_id][nb_block_id].compare_exchange_strong(
          false, true)) {
#else
  {
#endif
    // will check if below band in march narrow
    // use processing blocks to make sure no other neighbor of nb_block_id is
    // modifying nb_block_id heap
    bool dst_update_success =
        integrate_vertex(nb_interval_id, nb_block_id, dst, true, stage);
    if (dst_update_success) { // only update if it's true, allows for
      // remaining true
      active_blocks[nb_interval_id][nb_block_id].store(dst_update_success);
      active_intervals[nb_interval_id] = true;
#ifdef FULL_PRINT
      cout << "\t\t\tasync activate interval " << nb_interval_id << " block "
           << nb_block_id << '\n';
#endif
    }
    // Note: possible optimization here via explicit setting of memory
    // ordering on atomic
    processing_blocks[nb_interval_id][nb_block_id].store(
        false); // release nb_block_id heap
    // The other block isn't processing, so an update to it at here
    // is currently true for this iteration. It does not need to be checked
    // again in integrate_updated_ghost via adding it to the
    // updated_ghost_vec. Does not need to be added to active_neighbors for
    // same reason
    return;
  }
#endif // end of ASYNC

  // save vertex for later processing putting it in an overflow that will
  // be emptied in integrate_updated_ghost()
  // updated_ghost_vec[x][a][b] in domain of a, in ghost of block b
  // active_neighbors[x][a][b] in domain of b, in ghost of block a
  // in the synchronous case simply place all neighboring interval / block
  // updates in an overflow buffer update_ghost_vec for processing in
  // integrate_updated_ghost Note: that if the update is in a separate
  // interval it will be added to the same set as block_id updates from that
  // interval, this is because there is no need to separate updates based on
  // interval saving overhead
  active_intervals[nb_interval_id] = true;
  if (stage == "prune") {
    assertm(dst->valid_radius(), "selected must have a valid radius");
  }
  // when nb_interval_id != interval_id, the block_id is technically
  // incorrect, surprisingly this makes no difference for the correctness
  // since the block_id origination dimension is merely to prevent
  // race conditions
  // updated_ghost_vec[nb_interval_id][block_id][nb_block_id].emplace_back(
  // dst->edge_state, msg_offsets, dst->parent, dst->radius);
  // updated_ghost_vec[nb_interval_id][block_id].emplace(
  // nb_block_id, dst->edge_state, msg_offsets, dst->parent, dst->radius);
  updated_ghost_vec[nb_interval_id][block_id][nb_block_id].emplace_back(
      dst->edge_state, msg_offsets, dst->parent, dst->radius);
  active_neighbors[nb_interval_id][nb_block_id][block_id] = true;

#ifdef FULL_PRINT
  auto nb_block_coord = id_to_coord(nb_block_id, this->interval_block_lengths);
  cout << "\t\t\tplace_vertex(): interval " << nb_interval_id << " nb block "
       << coord_to_str(nb_block_coord) << " msg offsets "
       << coord_to_str(msg_offsets) << '\n';
#endif
}

/*
 * This function holds all the logic of whether the update of a vertex within
 * one intervals and blocks domain is adjacent to another interval and block.
 * If the vertex is covered by an adjacent region then it passes the vertex to
 * place_vertex for potential updating or saving. Assumes star stencil, no
 * diagonal connection in 3D this yields 6 possible block and or interval
 * connection corners.  block_id and interval_id are in linearly addressed
 * row-order. dst is always guaranteed to be within block_id and interval_id
 * region. dst has already been protected by global padding out of bounds from
 * guard in accumulate. This function determines if dst is in a border region
 * and which neighbor block / interval should be notified of adjacent change
 */
template <class image_t>
void Recut<image_t>::check_ghost_update(VID_t interval_id, VID_t block_id,
                                        GridCoord dst_coord, VertexAttr *dst,
                                        std::string stage) {

  auto dst_offsets = coord_mod(dst_coord, this->block_lengths);
  auto dst_interval_offsets = coord_mod(dst_coord, this->interval_lengths);
  auto block_coord = id_to_coord(block_id, this->interval_block_lengths);

#ifdef FULL_PRINT
  cout << "\t\tcheck_ghost_update(): on " << coord_to_str(dst_coord)
       << ", block " << coord_to_str(block_coord) << '\n';
#endif

  // check all 6 directions for possible ghost updates
  if (dst_offsets[0] == 0) {
    if (dst_coord[0] > 0) { // protect from image out of bounds
      VID_t nb = block_id - 1;
      VID_t nb_interval_id = interval_id; // defaults to current interval
      if (dst_interval_offsets[0] == 0) {
        nb_interval_id = interval_id - 1;
        // Convert block coordinates into linear index row-ordered
        nb = sub_block_to_block_id(this->interval_block_lengths[0] - 1,
                                   block_coord[1], block_coord[2]);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst,
                     new_offset_coord(this->block_lengths[0], dst_offsets[1],
                                      dst_offsets[2]),
                     stage);
    }
  }
  if (dst_offsets[1] == 0) {
    if (dst_coord[1] > 0) { // protect from image out of bounds
      VID_t nb = block_id - this->interval_block_lengths[0];
      VID_t nb_interval_id = interval_id; // defaults to current interval
      if (dst_interval_offsets[1] == 0) {
        nb_interval_id = interval_id - this->grid_interval_lengths[0];
        nb = sub_block_to_block_id(block_coord[0],
                                   this->interval_block_lengths[1] - 1,
                                   block_coord[2]);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst,
                     new_offset_coord(dst_offsets[0], this->block_lengths[1],
                                      dst_offsets[2]),
                     stage);
    }
  }
  if (dst_offsets[2] == 0) {
    if (dst_coord[2] > 0) { // protect from image out of bounds
      VID_t nb = block_id - this->interval_block_lengths[0] *
                                this->interval_block_lengths[1];
      VID_t nb_interval_id = interval_id; // defaults to current interval
      if (dst_interval_offsets[2] == 0) {
        nb_interval_id = interval_id - this->grid_interval_lengths[0] *
                                           this->grid_interval_lengths[1];
        nb = sub_block_to_block_id(block_coord[0], block_coord[1],
                                   this->interval_block_lengths[2] - 1);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst,
                     new_offset_coord(dst_offsets[0], dst_offsets[1],
                                      this->block_lengths[2]),
                     stage);
    }
  }

  if (dst_offsets[2] == this->block_lengths[0] - 1) {
    if (dst_coord[2] <
        this->image_lengths[2] - 1) { // protect from image out of bounds
      VID_t nb = block_id + this->interval_block_lengths[0] *
                                this->interval_block_lengths[1];
      VID_t nb_interval_id = interval_id; // defaults to current interval
      if (dst_interval_offsets[2] == this->interval_lengths[2] - 1) {
        nb_interval_id = interval_id + this->grid_interval_lengths[0] *
                                           this->grid_interval_lengths[1];
        nb = sub_block_to_block_id(block_coord[0], block_coord[1], 0);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst,
                     new_offset_coord(dst_offsets[0], dst_offsets[1], -1),
                     stage);
    }
  }
  if (dst_offsets[1] == this->block_lengths[1] - 1) {
    if (dst_coord[1] <
        this->image_lengths[1] - 1) { // protect from image out of bounds
      VID_t nb = block_id + this->interval_block_lengths[0];
      VID_t nb_interval_id = interval_id; // defaults to current interval
      if (dst_interval_offsets[1] == this->interval_lengths[1] - 1) {
        nb_interval_id = interval_id + this->grid_interval_lengths[0];
        nb = sub_block_to_block_id(block_coord[0], 0, block_coord[2]);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst,
                     new_offset_coord(dst_offsets[0], -1, dst_offsets[2]),
                     stage);
    }
  }
  if (dst_offsets[0] == this->block_lengths[2] - 1) {
    if (dst_coord[0] <
        this->image_lengths[0] - 1) { // protect from image out of bounds
      VID_t nb = block_id + 1;
      VID_t nb_interval_id = interval_id; // defaults to current interval
      if (dst_interval_offsets[0] == this->interval_lengths[0] - 1) {
        nb_interval_id = interval_id + 1;
        nb = sub_block_to_block_id(0, block_coord[1], block_coord[2]);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst,
                     new_offset_coord(-1, dst_offsets[1], dst_offsets[2]),
                     stage);
    }
  }
}

/* check and add current vertices in star stencil
 * if the current is greater the parent code will be greater
 * see struct definition in vertex_attr.h for full
 * list of VertexAttr parent codes and their meaning
 */
template <class image_t>
int Recut<image_t>::get_parent_code(VID_t dst_id, VID_t src_id) {
  auto src_gr = src_id > dst_id ? true : false; // current greater
  auto adiff = absdiff(dst_id, src_id);
  if ((adiff % image_lengths[0] * image_lengths[1]) == 0) {
    if (src_gr)
      return 5;
    else
      return 4;
  } else if ((adiff % this->image_lengths[0]) == 0) {
    if (src_gr)
      return 3;
    else
      return 2;
  } else if (adiff == 1) {
    if (src_gr)
      return 1;
    else
      return 0;
  }
  // FIXME check this
  cout << "get_parent_code failed at current " << src_id << " dst_id " << dst_id
       << " absdiff " << adiff << '\n';
  throw;
}

/*
 * returns true if the vertex updated to the proper interval and block false
 * otherwise This function takes no responsibility for updating the interval
 * as active or marking the block as active. It simply updates the correct
 * interval and blocks new vertex in that blocks unique heap. The caller is
 * responsible for making any changes to activity state of interval or
 * block.
 */
template <class image_t>
template <class Container>
bool Recut<image_t>::integrate_vertex(const VID_t interval_id,
                                      const VID_t block_id,
                                      struct VertexAttr *updated_vertex,
                                      bool ignore_KNOWN_NEW, std::string stage,
                                      Container &fifo) {
  assertm(updated_vertex, "integrate_vertex got nullptr updated vertex");

#ifdef DENSE
  auto dst =
      get_vertex_vid(interval_id, block_id, updated_vertex->vid, nullptr);

  // handle simpler radii stage and exit
  if (stage == "radius") {
    assertm(dst, "active vertex must already exist");
    if (dst->radius > updated_vertex->radius) {
      fifo.push_back(*dst);
      // deep copy into the shared memory location in the separate block
      *dst = *updated_vertex;
      assertm(dst->radius == updated_vertex->radius,
              "radius did not get copied");
      return true;
    }
    return false;
  }

  // handle simpler prune stage and exit
  if (stage == "prune") {
    assertm(dst, "active vertex must already exist");
    if (updated_vertex->root() || (*dst != *updated_vertex)) {
      fifo.push_back(*dst);
      // deep copy into the shared memory location in the separate block
      *dst = *updated_vertex;
      assertm(dst->valid_radius(), "dst must have a valid radius");
      return true;
    }
    return false;
  }

#else // not DENSE
  // doesn't matter if it's a root or not, it's now just a msg
  updated_vertex->mark_band();
  // adds to iterable but not to active vertices since its from outside domain
  if (stage == "connected") {
    local_fifo[interval_id][block_id].push_back(*updated_vertex);
  } else {
    // cout << "integrate vertex " << updated_vertex->description();
    fifo.push_back(*updated_vertex);
  }
  return true;
#endif

  return false;
} // end integrate_vertex

/* Core exchange step of the fastmarching algorithm, this processes and
 * empties the overflow buffer updated_ghost_vec in parallel. intervals and
 * blocks can receive all of their updates from the current iterations run of
 * march_narrow_band safely to complete the iteration
 */
template <class image_t>
template <class Container>
void Recut<image_t>::integrate_updated_ghost(const VID_t interval_id,
                                             const VID_t block_id,
                                             std::string stage,
                                             Container &fifo) {
  for (VID_t nb = 0; nb < interval_block_size; nb++) {
    // active_neighbors[x][a][b] in domain of b, in ghost of block a
    if (active_neighbors[interval_id][block_id][nb]) {
      // updated ghost cleared in march_narrow_band
      // iterate over all ghost points of block_id inside domain of block nb

      // updated_ghost_vec[x][a][b] in domain of a, in ghost of block b
      for (auto updated_vertex : updated_ghost_vec[interval_id][nb][block_id]) {

#ifdef FULL_PRINT
        cout << "integrate(): " << coord_to_str(updated_vertex.offsets)
             << " ghost of block id: " << block_id << " in domain block: " << nb
             << " interval: " << interval_id << '\n';
#endif
        // note fifo must respect that this vertex belongs to the domain
        // of nb, not block_id
        // this updates the value of the vertex of block: block_id
        // this vertex is in an overlapping ghost region
        // so the vertex is technically in the domain of nb
        // but the copy needs to be safely updated for march_narrow_band
        // to execute safely in parallel
        integrate_vertex(interval_id, block_id, &updated_vertex, false, stage,
                         fifo);
      } // end for each VertexAttr
      active_neighbors[interval_id][block_id][nb] = false; // reset to false
      // clear sets for all possible block connections of block_id from this
      // iter
      updated_ghost_vec[interval_id][nb][block_id].clear();
    } // for all active neighbor blocks of block_id
  }
  if (stage == "connected") {
    if (!(local_fifo[interval_id][block_id].empty())) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_id << " block: " << block_id
           << " to active\n";
#endif
      active_blocks[interval_id][block_id].store(true);
    }
  } else {
    if (!(fifo.empty())) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_id << " block: " << block_id
           << " to active\n";
#endif
      active_blocks[interval_id][block_id].store(true);
    }
  }
}

template <class image_t> void Recut<image_t>::activate_all_intervals() {
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    active_intervals[interval_id] = true;
  }
}

/*
 * If any interval is active return false, a interval is active if
 * any of its blocks are still active
 */
template <class image_t> bool Recut<image_t>::are_intervals_finished() {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Intervals active: ";
#endif
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    if (active_intervals[interval_id]) {
      tot_active++;
#ifdef LOG_FULL
      cout << interval_id << ", ";
#endif
    }
  }
  cout << '\n';
  if (tot_active == 0) {
    return true;
  } else {
#ifdef LOG_FULL
    cout << tot_active << " total intervals active" << '\n';
#endif
    return false;
  }
}

/*
 * If any block is active return false, a block is active if its
 * corresponding heap is not empty
 */
template <class image_t>
bool Recut<image_t>::check_blocks_finish(VID_t interval_id) {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Blocks active: ";
#endif
  for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
    if (active_blocks[interval_id][block_id].load()) {
      tot_active++;
#ifdef LOG_FULL
      cout << block_id << ", ";
#endif
    }
  }
  if (tot_active == 0) {
    return true;
  } else {
#ifdef LOG_FULL
    cout << '\n' << tot_active << " total blocks active" << '\n';
#endif
    return false;
  }
}

// https://github.com/HumanBrainProject/swcPlus/blob/master/SWCplus_specification.html
template <typename image_t>
void Recut<image_t>::print_vertex(VID_t interval_id, VID_t block_id,
                                  VertexAttr *current, GridCoord offsets) {
  auto coord = coord_add(current->offsets, offsets);
  std::ostringstream line;
  // n,type,x,y,z,radius,parent

  // n
  line << coord_to_id(coord, this->image_lengths) << ' ';

  // type_id
  if (current->root()) {
    line << "-1" << ' ';
  } else {
    line << '0' << ' ';
  }

  // coordinates
  line << coord[0] << ' ' << coord[1] << ' ' << coord[2] << ' ';

  // radius
  line << +(current->radius) << ' ';

  // parent
  auto parent_coord = coord_add(coord, current->parent);
  auto parent_vid = coord_to_id(parent_coord, this->image_lengths);
  if (current->root()) {
    line << "-1";
  } else {
    line << parent_vid;
  }

  line << '\n';

  if (this->out.is_open()) {
#pragma omp critical
    this->out << line.str();
  } else {
#pragma omp critical
    std::cout << line.str();
  }
}

// if the parent has been pruned then set the current
// parent further upstream
template <typename image_t>
void Recut<image_t>::adjust_vertex_parent(VertexAttr *vertex,
                                          GridCoord start_offsets) {
  VertexAttr *parent = vertex;
  VID_t block_id, interval_id;
  GridCoord coord, parent_coord;
  do {
    // find parent
    {
      coord = coord_add(parent->offsets, start_offsets);
      parent_coord = coord_add(coord, parent->parent);

      // get actual interval and block
      block_id = coord_img_to_block_id(parent_coord);
      interval_id = coord_img_to_interval_id(parent_coord);

      auto parent_offset = coord_mod(parent_coord, this->block_lengths);

      // get parent
      parent = get_active_vertex(interval_id, block_id, parent_offset);
      assertm(parent != nullptr, "inactive parents must be unreachable");
    }

    // update offsets for next loop iteration
    start_offsets = id_interval_block_to_img_offsets(interval_id, block_id);

  } while (parent->unvisited());
  // parent is now unpruned and traceable from vertex
  // so set vertex->parent to the new offset
  vertex->parent = coord_sub(parent_coord, coord);
}

template <class image_t>
template <class Container>
void Recut<image_t>::dump_buffer(Container buffer) {
  std::cout << "\n\nDump buffer\n";
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      for (auto &v : buffer[interval_id][block_id]) {
        std::cout << v.description();
      }
    }
  }
  std::cout << "\n\nFinished buffer dump\n";
}

template <class image_t>
template <class Container, typename T, typename T2>
void Recut<image_t>::connected_tile(
    const image_t *tile, VID_t interval_id, VID_t block_id, GridCoord offsets,
    std::string stage, const TileThresholds<image_t> *tile_thresholds,
    Container &fifo, VID_t revisits, T vdb_accessor, T2 leaf_iter) {

  if (local_fifo[interval_id][block_id].empty())
    return;

  // load flags
  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle =
      leaf_iter->attributeArray("flags");
  openvdb::points::AttributeWriteHandle<OffsetCoord> parents_handle =
      leaf_iter->attributeArray("parents");
  // if (input_is_vdb) {
  // flags_handle = leaf_iter->attributeArray("flags");

  // parents_handle = leaf_iter->attributeArray("parents");
  //}

  VertexAttr *current, *msg_vertex;
  VID_t visited = 0;
  while (!(local_fifo[interval_id][block_id].empty())) {

#ifdef LOG_FULL
    visited += 1;
#endif

    // msg_vertex might become undefined during scatter
    msg_vertex = &(local_fifo[interval_id][block_id].front());
    const bool in_domain = msg_vertex->selected() || msg_vertex->root();
    auto surface = msg_vertex->surface();

    auto current_coord = coord_add(msg_vertex->offsets, offsets);
    auto current_off = coord_mod(current_coord, this->block_lengths);

    // invalid can either be out of range of the entire global image or it
    // can be a background vertex which occurs due to pixel value below the
    // threshold, previously selected vertices are considered valid
    auto found_adjacent_invalid = false;
    auto valids =
        // star stencil offsets to img coords
        this->stencil |
        rng::views::transform([&current_coord](auto stencil_offset) {
          return coord_add(current_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this, &found_adjacent_invalid](auto coord_img) {
          if (is_in_bounds(coord_img, zeros(), this->image_lengths))
            return false;
          found_adjacent_invalid = true;
          return true;
        }) |
        // within leaf?
        rng::views::remove_if([&](auto coord_img) {
          auto mismatch = block_id != coord_img_to_block_id(coord_img);
          if (mismatch) {
            if (this->input_is_vdb) {
              if (!get_vdb_val(vdb_accessor, coord_img))
                found_adjacent_invalid = true;
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
        rng::views::remove_if([&](auto coord_img) {
          auto offset_to_current = coord_sub(current_coord, coord_img);
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // is background?  ...has side-effects
          return !accumulate_connected(
              tile, interval_id, block_id, coord_img, ind, offset_to_current,
              revisits, tile_thresholds, found_adjacent_invalid, vdb_accessor,
              flags_handle, parents_handle);
        }) |
        rng::to_vector; // force full evaluation via vector

    // protect from message values not inside
    // this block or interval such as roots from activate_vids
#ifdef DENSE
    auto vid = msg_vertex->vid;
    current = get_vertex_vid(interval_id, block_id, vid, nullptr);
#else
    if (in_domain) {
      current = get_active_vertex(interval_id, block_id, current_off);
      assertm(current != nullptr, "connected can't be passed a null");
      // msg_vertex aleady aware it is a surface
      if (surface) {
        current->mark_surface();
        if (this->input_is_vdb) {
          set_surface(flags_handle, leaf_iter->beginIndexVoxel(current_coord));
        }
        // save all surface vertices for the radius stage
        // each fifo corresponds to a specific interval_id and block_id
        // so there are no race conditions
        fifo.push_back(*current);
      }
    } else {
      // previous msg_vertex could have been invalidated by insertion in
      // accumulate_connected
      msg_vertex = &(local_fifo[interval_id][block_id].front());
      assertm(msg_vertex->band(), "if not selected it must be a band message");
      current = msg_vertex;
    }
    // safe to remove msg now
    local_fifo[interval_id][block_id].pop_front(); // remove it
#endif

    // ignore if already designated as surface
    // also prevents adding to fifo twice
    if (found_adjacent_invalid && !(current->surface())) {
#ifdef FULL_PRINT
      std::cout << "found surface vertex " << interval_id << " " << block_id
                << " " << coord_to_str(current_coord) << " label "
                << current->label() << '\n';
#endif
      current->mark_surface();
      if (this->input_is_vdb) {
        set_surface(flags_handle, leaf_iter->beginIndexVoxel(current_coord));
      }

      if (in_domain) {
        // save all surface vertices for the radius stage
        // each fifo corresponds to a specific interval_id and block_id
        // so there are no race conditions
        fifo.push_back(*current);
        assertm(current->selected() || current->root(), "must be selected");

#ifdef DENSE
        // ghost regions in the DENSE case have shared copies of ghost regions
        // so they would need to know about any changes to vertex state
        // otherwise surface status change is irrelevant to outside domains
        // at this stage
        // goes into updated_ghost_vec of neighbors if its on the edge with
        // them these then get added into neighbor local_fifo
        check_ghost_update(interval_id, block_id, current_coord, current,
                           stage);
#endif
      } else {
        // a message from an outside leaf is actually a surface and was
        // unaware so send a message directly to that leaf
        const auto check_block_id = coord_img_to_block_id(current_coord);
        const auto check_interval_id = coord_img_to_interval_id(current_coord);

        // leverage updated_ghost_vec to avoid race conditions
        // check_ghost_update never applies to vertices outside this leaf
        // updated_ghost_vec[check_interval_id][block_id][check_block_id]
        //.emplace_back(current->edge_state, current->offsets,
        // current->parent, current->radius);
        // updated_ghost_vec[check_interval_id][block_id].emplace(
        // check_block_id, current->edge_state, current->offsets,
        // current->parent, current->radius);
        updated_ghost_vec[check_interval_id][block_id][check_block_id]
            .emplace_back(current->edge_state, current->offsets,
                          current->parent, current->radius);
        active_neighbors[check_interval_id][check_block_id][block_id] = true;
      }
    }
  }
#ifdef LOG_FULL
  cout << "visited vertices: " << visited << '\n';
#endif
}

template <class image_t>
template <class Container, typename T, typename T2>
void Recut<image_t>::radius_tile(const image_t *tile, VID_t interval_id,
                                 VID_t block_id, GridCoord offsets,
                                 std::string stage,
                                 const TileThresholds<image_t> *tile_thresholds,
                                 Container &fifo, VID_t revisits,
                                 T vdb_accessor, T2 leaf_iter) {

  if (fifo.empty())
    return;

  // load read-only flags
  openvdb::points::AttributeHandle<uint8_t> flags_handle =
      leaf_iter->constAttributeArray("flags");
  // read-write radius
  openvdb::points::AttributeWriteHandle<uint8_t> radius_handle =
      leaf_iter->attributeArray("radius");

  VertexAttr *current; // for performance
  VID_t visited = 0;
  while (!(fifo.empty())) {
    // msg_vertex will be invalidated during scatter
    auto msg_vertex = &(fifo.front());
    fifo.pop_front();

    if (msg_vertex->band()) {
      // current can be from ghost region in different interval or block
      current = msg_vertex;
    } else {
      // current is safe during scatter
      current = get_active_vertex(interval_id, block_id, msg_vertex->offsets);
      assertm(current != nullptr,
              "get_active_vertex yielded nullptr radius_tile");
    }

    auto current_coord = coord_add(current->offsets, offsets);
    auto current_ind = leaf_iter->beginIndexVoxel(current_coord);
    // radius field can now be be mutated
    // set any vertex that shares a border with background
    // to the known radius of 1
    assertm(current->surface() == is_surface(flags_handle, current_ind),
            "surfaces don't match");
    if (current->surface()) {
      current->radius = 1;
      radius_handle.set(*current_ind, 1);
      // if in domain notify potential outside domains
      if (!(msg_vertex->band())) {
        check_ghost_update(interval_id, block_id, current_coord, current,
                           stage);
      }
    }

#ifdef LOG_FULL
    visited += 1;
#endif
    assertm(current->radius == radius_handle.get(*current_ind),
            "radii don't match");

    auto updated_inds =
        // star stencil offsets to img coords
        this->stencil |
        rng::views::transform([&current_coord](auto stencil_offset) {
          return coord_add(current_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this](auto coord_img) {
          return !is_in_bounds(coord_img, zeros(), this->image_lengths);
        }) |
        // within leaf?
        rng::views::remove_if([&](auto coord_img) {
          return block_id != coord_img_to_block_id(coord_img);
        }) |
        rng::views::transform([&](auto coord_img) {
          print_coord(coord_img, "in leaf");
          return coord_img;
        }) |
        // visit valid voxels
        rng::views::transform([&](auto coord_img) {
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // ...has side-effects
          accumulate_radius(interval_id, block_id, coord_img, ind,
                            current->radius, fifo, flags_handle, radius_handle);
          return ind;
        }) |
        rng::to_vector; // force evaluation
  }
}

template <class image_t>
template <class Container, typename T, typename T2>
void Recut<image_t>::prune_tile(const image_t *tile, VID_t interval_id,
                                VID_t block_id, GridCoord offsets,
                                std::string stage,
                                const TileThresholds<image_t> *tile_thresholds,
                                Container &fifo, VID_t revisits, T vdb_accessor,
                                T2 leaf_iter) {
  if (fifo.empty())
    return;

  openvdb::points::AttributeWriteHandle<uint8_t> radius_handle =
      leaf_iter->attributeArray("radius");
  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle =
      leaf_iter->attributeArray("flags");

  VertexAttr *current;
  bool current_outside_domain, covered;
  VID_t visited = 0;
  while (!(fifo.empty())) {
    covered = false;
    // fifo starts with only roots
    auto msg_vertex = &(fifo.front());
    auto current_outside_domain = false;
    fifo.pop_front();

#ifdef DENSE
    current = get_vertex_vid(interval_id, block_id, msg_vertex->vid, nullptr);
#else
    if (msg_vertex->band()) {
      current = msg_vertex;
      current_outside_domain = true;
    } else {
      current = get_active_vertex(interval_id, block_id, msg_vertex->offsets);
      assertm(current != nullptr,
              "get_active_vertex yielded nullptr radius_tile");
    }
    assertm(current->valid_radius(), "fifo must recover a valid_radius vertex");
#endif

#ifdef LOG_FULL
    visited += 1;
#endif

    auto current_coord = coord_add(current->offsets, offsets);
    auto current_ind = leaf_iter->beginIndexVoxel(current_coord);

    assertm(current->root() == is_root(flags_handle, current_ind),
            "roots don't match");

    // Parent
    // by default current will have a root of itself
    if (current->root()) {
      // OffsetCoord zeros{0, 0, 0};
      // current->set_parent(zeros);
      // parent_handle.set(*ind, zeros);
      current->prune_visit();
    }

#ifdef FULL_PRINT
    // all block ids are a linear row-wise idx, relative to current interval
    cout << '\n'
         << coord_to_str(current_coord) << " interval " << interval_id
         << " block " << block_id << " label " << current->label() << " radius "
         << +(current->radius) << '\n';
#endif

    // force full evaluation by saving to vector
    auto updated_inds =
        // star stencil offsets to img coords
        this->stencil |
        rng::views::transform([&current_coord](auto stencil_offset) {
          return coord_add(current_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this](auto coord_img) {
          return !is_in_bounds(coord_img, zeros(), this->image_lengths);
        }) |
        // within leaf?
        rng::views::remove_if([&](auto coord_img) {
          return block_id != coord_img_to_block_id(coord_img);
        }) |
        // visit valid voxels
        rng::views::transform([&](auto coord_img) {
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // ...has side-effects
          accumulate_prune(interval_id, block_id, coord_img, ind, current,
                           current_coord, current->unvisited(), fifo,
                           radius_handle, flags_handle);
          return ind;
        }) |
        rng::to_vector;
  } // end while over fifo
} // end prune_tile

template <class image_t>
template <class Container, typename T, typename T2>
void Recut<image_t>::march_narrow_band(
    const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds, Container &fifo,
    T vdb_accessor, T2 leaf_iter) {
  // first coord of block with respect to whole image
  auto block_img_offsets =
      id_interval_block_to_img_offsets(interval_id, block_id);

#ifdef LOG_FULL
  VID_t visited = 0;
  auto timer = new high_resolution_timer();
  cout << "\nMarching " << tree_to_str(interval_id, block_id) << ' '
       << coord_to_str(block_img_offsets) << '\n';
#endif

  VID_t revisits = 0;

  if (stage == "connected") {
    connected_tile(tile, interval_id, block_id, block_img_offsets, stage,
                   tile_thresholds, fifo, revisits, vdb_accessor, leaf_iter);
  } else if (stage == "radius") {
    radius_tile(tile, interval_id, block_id, block_img_offsets, stage,
                tile_thresholds, fifo, revisits, vdb_accessor, leaf_iter);
  } else if (stage == "prune") {
    prune_tile(tile, interval_id, block_id, block_img_offsets, stage,
               tile_thresholds, fifo, revisits, vdb_accessor, leaf_iter);
  } else {
    assertm(false, "Stage name not recognized");
  }

#ifdef LOG_FULL
  cout << "Marched interval: " << interval_id << " block: " << block_id
       << " visiting " << visited << " in " << timer->elapsed() << " s" << '\n';
#endif

  // Note: could set explicit memory ordering on atomic
  active_blocks[interval_id][block_id].store(false);
  processing_blocks[interval_id][block_id].store(
      false); // release block_id heap
} // end march_narrow_band

// return the nearest background threshold value that is closest to the
// arg foreground_percent for the given image region
// this function starts with a bkg_thresh of 0 and increments until it
// finds a bkg_thresh value which yields a foreground_percent equal to
// or above the requested
// if the absdiff of the final above threshold is greater than the last
// bkg_thresh below, the last bkg_thresh that is slightly below will
// be returned instead in effort of accuracy
template <class image_t>
int Recut<image_t>::get_bkg_threshold(const image_t *tile,
                                      VID_t interval_vertex_size,
                                      const double foreground_percent) {
#ifdef LOG
  cout << "Determine thresholding value on fg %: " << foreground_percent
       << '\n';
#endif
  assertm(foreground_percent >= 0., "foreground_percent must be 0 or positive");
  assertm(foreground_percent <= 1., "foreground_percent must be 1.0 or less");
  image_t above; // store next bkg_thresh value above desired bkg pct
  image_t below = 0;
  double above_diff_pct = 0.0; // pct bkg at next above value
  double below_diff_pct = 1.;  // last below percentage
  double desired_bkg_pct = 1. - foreground_percent;
  // test different background threshold values until finding
  // percentage above desired_bkg_pct or when all pixels set to background
  VID_t bkg_count;
  for (image_t local_bkg_thresh = 0;
       local_bkg_thresh <= std::numeric_limits<image_t>::max();
       local_bkg_thresh++) {

    // Count total # of pixels under current thresh
    bkg_count = 0;
#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
#pragma omp parallel for reduction(+ : bkg_count)
#endif
    for (VID_t i = 0; i < (VID_t)interval_vertex_size; i++) {
      if (tile[i] <= local_bkg_thresh) {
        ++bkg_count;
      }
    }

    // Check if above desired percent background
    double test_pct = bkg_count / static_cast<double>(interval_vertex_size);
    auto foreground_count = interval_vertex_size - bkg_count;

#ifdef FULL_PRINT
    cout << "bkg_thresh=" << local_bkg_thresh << " (" << test_pct
         << "%) background, total foreground count: " << foreground_count
         << '\n';
#endif
    double test_diff = abs(test_pct - desired_bkg_pct);

    // check whether we overshot, if above
    // then return this local_bkg_thresh
    // or the last if it was closer
    if (test_pct >= desired_bkg_pct) {

      if (test_diff < below_diff_pct)
        return local_bkg_thresh;
      else
        return below;
    }

    // set the below for next iteration
    below = local_bkg_thresh;
    below_diff_pct = test_diff;
  }
  return std::numeric_limits<image_t>::max();
}

template <class image_t>
template <class Container, typename T>
std::atomic<double>
Recut<image_t>::process_interval(VID_t interval_id, const image_t *tile,
                                 std::string stage,
                                 const TileThresholds<image_t> *tile_thresholds,
                                 Container &fifo, T vdb_accessor) {

  struct timespec presave_time, postmarch_time, iter_start,
      start_iter_loop_time, end_iter_time, postsave_time;
  double no_io_time;
  no_io_time = 0.0;

#ifdef DENSE
  // only load the intervals that are not already mapped or have been read
  // already calling load when already present will throw
  if (!(grid.GetInterval(interval_id)->IsInMemory())) {
    grid.GetInterval(interval_id)->LoadFromDisk();
  }
#endif

  // setup any border regions from activate_vids or setup functions
#ifdef USE_OMP_BLOCK
#pragma omp parallel for
#endif
  for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
    integrate_updated_ghost(interval_id, block_id, stage, fifo[block_id]);
  }

  // iterations over blocks
  // if there is a single block per interval than this while
  // loop will exit after one iteration
  clock_gettime(CLOCK_REALTIME, &start_iter_loop_time);
  VID_t inner_iteration_idx = 0;
  while (true) {
    clock_gettime(CLOCK_REALTIME, &iter_start);

#ifdef ASYNC

#ifdef TF
    tf::Executor executor(params->parallel_num());
    tf::ExecutorObserver *observer =
        executor.make_observer<tf::ExecutorObserver>();
    vector<tf::Taskflow *> prevent_destruction;
#endif // TF

    // if any active status for any block of interval_id is
    // true
    while (aimage_y_len_of(this->active_blocks[interval_id].begin(),
                           this->active_blocks[interval_id].end(),
                           [](atomwrapper<bool> i) { return i.load(); })) {

#ifdef TF
      prevent_destruction.emplace_back(new tf::Taskflow());
      bool added_task = false;
#endif // TF

      for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
        // if not currently processing, set atomically set to true and
        if (active_blocks[interval_id][block_id].load() &&
            processing_blocks[interval_id][block_id].compare_exchange_strong(
                false, true)) {
#ifdef LOG_FULL
          // cout << "Start active block_id " << block_id << '\n';
#endif

#ifdef TF
          // FIXME check passing tile ptr as ref
          prevent_destruction.back()->silent_emplace([=, &tile]() {
            march_narrow_band(tile, interval_id, block_id, stage,
                              tile_thresholds, vdb_accessor);
          });
          added_task = true;
#else
          async(launch::async, &Recut<image_t>::march_narrow_band, this, tile,
                interval_id, block_id, stage, tile_thresholds, vdb_accessor);
#endif // TF
        }
      }

#ifdef TF
      if (!(prevent_destruction.back()->empty())) {
        // cout << "start taskflow" << '\n';
        // returns a std::future object
        // it is non-blocking
        // note it is the tasks responsibility to set the
        // appropriate active block variables in order to
        // exit this loop all active blocks must be false
        executor.run(*(prevent_destruction.back()));
      }
#endif // TF

      this_thread::sleep_for(chrono::milliseconds(10));
    }

#else // OMP or sequential strategy

    // assertm(this->topology_grid, "Block count and size must match topology
    // grid leaf size at compile time");
    auto leaf_iter = this->topology_grid->tree().beginLeaf();
#ifdef USE_OMP_BLOCK
#pragma omp parallel for
#endif
    for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
      if (input_is_vdb)
        assertm(leaf_iter, "Block count and size must match topology grid leaf "
                           "size at compile time");
      march_narrow_band(tile, interval_id, block_id, stage, tile_thresholds,
                        fifo[block_id], vdb_accessor, leaf_iter);
      if (input_is_vdb)
        ++leaf_iter;
    }

#endif // ASYNC

#ifdef LOG_FULL
    cout << "Marched narrow band";
    clock_gettime(CLOCK_REALTIME, &postmarch_time);
    cout << " in " << diff_time(iter_start, postmarch_time) << " sec." << '\n';
#endif

    //#ifdef ASYNC
    // for(VID_t block_id = 0;block_id<interval_block_size;++block_id)
    //{
    // create_integrate_thread(interval_id, block_id);
    //}
    //#else

#ifdef USE_OMP_BLOCK
#pragma omp parallel for
#endif
    for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
      integrate_updated_ghost(interval_id, block_id, stage, fifo[block_id]);
    }

#ifdef LOG_FULL
    clock_gettime(CLOCK_REALTIME, &end_iter_time);
    cout << "inner_iteration_idx " << inner_iteration_idx << " in "
         << diff_time(iter_start, end_iter_time) << " sec." << '\n';
#endif

    if (check_blocks_finish(interval_id)) {
      // final_inner_iter = inner_iteration_idx;
      break;
    }
    inner_iteration_idx++;
  } // iterations per interval

  clock_gettime(CLOCK_REALTIME, &presave_time);

#ifdef DENSE
  if (!(this->mmap_)) {
    grid.GetInterval(interval_id)->SaveToDisk();
  }
#endif

  clock_gettime(CLOCK_REALTIME, &postsave_time);

  no_io_time = diff_time(start_iter_loop_time, presave_time);
  // computation_time += no_io_time;
#ifdef LOG_FULL
  cout << "Interval: " << interval_id << " (no I/O) within " << no_io_time
       << " sec." << '\n';
#ifdef DENSE
  if (!(this->mmap_))
    cout << "Finished saving interval in "
         << diff_time(presave_time, postsave_time) << " sec." << '\n';
#endif
#endif

  active_intervals[interval_id] = false;
  return no_io_time;
}

#ifdef USE_MCP3D

/*
 * The interval size and shape define the requested "view" of the image
 * an image view is referred to as a tile
 * that we will load at one time. There is a one to one mapping
 * of an image view and an interval. There is also a one to
 * one mapping between each voxel of the image view and the
 * vertex of the interval. Note the interval is an array of
 * initialized unvisited structs so they start off at arbitrary
 * location but are defined as they are visited. When mmap
 * is defined all intervals follow copy on write semantics
 * at the granularity of a system page from the same array of structs
 * file INTERVAL_BASE
 */
template <class image_t>
void Recut<image_t>::load_tile(VID_t interval_id, mcp3d::MImage &mcp3d_tile) {
#ifdef LOG
  struct timespec start, image_load;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

  auto tile_extents = this->interval_lengths;

  // read image data
  // FIXME check that this has no state that can
  // be corrupted in a shared setting
  // otherwise just create copies of it if necessary
  assertm(!(this->params->force_regenerate_image),
          "If USE_MCP3D macro is set, this->params->force_regenerate_image "
          "must be set to False");
  // read data from channel
  mcp3d_tile.ReadImageInfo(args->resolution_level(), true);
  // read data
  try {
    auto interval_offsets = id_interval_to_img_offsets(interval_id);
#ifdef LOG_FULL
    print_coord(interval_offsets, "interval_offsets");
#endif

    // mcp3d operates in z y x order
    // but still returns row-major (c-order) buffers
    mcp3d::MImageBlock block(
        {interval_offsets[2], interval_offsets[1], interval_offsets[0]},
        {tile_extents[2], tile_extents[1], tile_extents[0]});
    mcp3d_tile.SelectView(block, args->resolution_level());
    mcp3d_tile.ReadData(true, "quiet");
  } catch (...) {
    MCP3D_MESSAGE("error in mcp3d_tile io. neuron tracing not performed")
    throw;
  }
  //#ifdef FULL_PRINT
  // print_image_3D(mcp3d_tile.Volume<image_t>(0), {tile_extents[0],
  // tile_extents[1], tile_extents[2]});
  //#endif

#ifdef LOG
  clock_gettime(CLOCK_REALTIME, &image_load);
  cout << "Load image in " << diff_time(start, image_load) << " sec." << '\n';
#endif
}

// Calculate new tile thresholds or use input thresholds according
// to params and args this function has no sideffects outside
// of the returned tile_thresholds struct
template <class image_t>
TileThresholds<image_t> *
Recut<image_t>::get_tile_thresholds(mcp3d::MImage &mcp3d_tile) {

  auto tile_thresholds = new TileThresholds<image_t>();

  std::vector<int> interval_dims =
      mcp3d_tile.loaded_view().xyz_dims(args->resolution_level());
  VID_t interval_vertex_size = static_cast<VID_t>(interval_dims[0]) *
                               interval_dims[1] * interval_dims[2];

  // assign thresholding value
  // foreground parameter takes priority
  // Note if either foreground or background percent is equal to or greater
  // than 0 than it was changed by a user so it takes precedence over the
  // defaults
  if (params->foreground_percent() >= 0) {
    tile_thresholds->bkg_thresh =
        get_bkg_threshold(mcp3d_tile.Volume<image_t>(0), interval_vertex_size,
                          params->foreground_percent());
    // deprecated
    // tile_thresholds->bkg_thresh = mcp3d::TopPercentile<image_t>(
    // mcp3d_tile.Volume<image_t>(0), interval_dims,
    // params->foreground_percent());
#ifdef LOG
    cout << "Requested foreground percent: " << params->foreground_percent()
         << " yielded background threshold: " << tile_thresholds->bkg_thresh
         << '\n';
#endif
  } else { // if bkg set explicitly and foreground wasn't
    if (params->background_thresh() >= 0) {
      tile_thresholds->bkg_thresh = params->background_thresh();
    }
  }
  // otherwise: tile_thresholds->bkg_thresh default inits to 0

  // assign max and min ints for this tile
  if (this->args->recut_parameters().get_max_intensity() < 0) {
    if (params->convert_only_) {
      tile_thresholds->max_int = std::numeric_limits<image_t>::max();
      tile_thresholds->min_int = std::numeric_limits<image_t>::min();
    } else {
      // max and min members will be set
      tile_thresholds->get_max_min(mcp3d_tile.Volume<image_t>(0),
                                   interval_vertex_size);
    }
  } else if (this->args->recut_parameters().get_min_intensity() < 0) {
    // if max intensity was set but not a min, just use the bkg_thresh value
    if (tile_thresholds->bkg_thresh >= 0) {
      tile_thresholds->min_int = tile_thresholds->bkg_thresh;
    } else {
      if (params->convert_only_) {
        tile_thresholds->max_int = std::numeric_limits<image_t>::max();
        tile_thresholds->min_int = std::numeric_limits<image_t>::min();
      } else {
        // max and min members will be set
        tile_thresholds->get_max_min(mcp3d_tile.Volume<image_t>(0),
                                     interval_vertex_size);
      }
    }
  } else { // both values were set
    // either of these values are signed and default inited -1, casting
    // them to unsigned image_t would lead to hard to find errors
    assertm(this->args->recut_parameters().get_max_intensity() >= 0,
            "invalid user max");
    assertm(this->args->recut_parameters().get_min_intensity() >= 0,
            "invalid user min");
    // otherwise set global max min from recut_parameters
    tile_thresholds->max_int =
        this->args->recut_parameters().get_max_intensity();
    tile_thresholds->min_int =
        this->args->recut_parameters().get_min_intensity();
  }

#ifdef LOG_FULL
  cout << "max_int: " << +(tile_thresholds->max_int)
       << " min_int: " << +(tile_thresholds->min_int) << '\n';
  cout << "bkg_thresh value = " << +(tile_thresholds->bkg_thresh) << '\n';
  cout << "interval dims x " << interval_dims[2] << " y " << interval_dims[1]
       << " z " << interval_dims[0] << '\n';
#endif

  return tile_thresholds;
} // end load_tile()

#endif // only defined in USE_MCP3D is

// returns the execution time for updating the entire
// stage excluding I/O
// note that tile_thresholds has a default value
// of nullptr, see update declaration
template <class image_t>
template <class Container>
std::unique_ptr<InstrumentedUpdateStatistics>
Recut<image_t>::update(std::string stage, Container &fifo,
                       TileThresholds<image_t> *tile_thresholds) {

  atomic<double> computation_time;
  computation_time = 0.0;
  global_revisits = 0;
  auto interval_open_count = std::vector<uint16_t>(grid_interval_size, 0);
  TileThresholds<image_t> *local_tile_thresholds;
#ifdef NO_INTERVAL_RV
  auto visited_intervals = std::make_unique<bool[]>(grid_interval_size, false);
#endif
#ifdef SCHEDULE_INTERVAL_RV
  auto visited_intervals = std::make_unique<bool[]>(grid_interval_size, false);
  auto locked_intervals = std::make_unique<bool[]>(grid_interval_size, false);
  auto schedule_intervals = std::make_unique<int[]>(grid_interval_size, 0);
#endif

#ifdef LOG
  cout << "Start updating stage " << stage << '\n';
#endif
  auto timer = new high_resolution_timer();

#ifdef USE_VDB
  // note openvdb::initialize() must have been called before this point
  // otherewise seg faults will occur
  // if (this->input_is_vdb || params->convert_only_) {
  assertm(this->topology_grid, "topology grid not initialized");
  auto vdb_const_accessor = this->topology_grid->getConstAccessor();
  auto vdb_accessor = this->topology_grid->getAccessor();
  // print_vdb(vdb_accessor, {this->image_lengths[0], this->image_lengths[1],
  // this->image_lengths[1]});
  //}
  std::vector<EnlargedPointDataGrid::Ptr> grids(this->grid_interval_size);
#endif

  // Main march for loop
  // continue iterating until all intervals are finished
  // intervals can be (re)activated by neighboring intervals
  int outer_iteration_idx;
  for (outer_iteration_idx = 0; !are_intervals_finished();
       outer_iteration_idx++) {

    // loop through all possible intervals
#ifdef USE_OMP_INTERVAL
#pragma omp parallel for
#endif
    for (int interval_id = 0; interval_id < grid_interval_size; interval_id++) {
      // only start intervals that have active processing to do
      if (active_intervals[interval_id]) {

#ifdef NO_INTERVAL_RV
        // forbid all re-opening tile/interval to check performance
        if (visited_intervals[interval_id]) {
          active_intervals[interval_id] = false;
          continue;
        } else {
          visited_intervals[interval_id] = true;
        }
#endif

#ifdef SCHEDULE_INTERVAL_RV
        // number of iterations to wait before a visited
        // interval can do a final set of processing
        auto scheduling_iteration_delay = 2;
        if (locked_intervals[interval_id]) {
          active_intervals[interval_id] = false;
          continue;
        }
        // only visited_intervals need to be explicitly
        // scheduled
        if (visited_intervals[interval_id] &&
            schedule_intervals[interval_id] != outer_iteration_idx) {
          // keep it active so that it is processed when it
          // is scheduled
          continue;
        }

        // otherwise process this interval
        // TODO still need to schedule_intervals initialize
        if (visited_intervals[interval_id]) {
          // this interval_id will not be processed again
          // so ignore it's scheduling
          locked_intervals[interval_id] = true;
        } else {
          visited_intervals[interval_id] = true;
          // in 2 iterations, all of the intervals that it
          // activated will have been processed
          // this is an approximation of scheduling once
          // all neighbors have been processed
          // in reality it's not clear when all neighbors
          // of this interval_id will have been processed
          schedule_intervals[interval_id] =
              outer_iteration_idx + scheduling_iteration_delay;
        }
#endif
        interval_open_count[interval_id] += 1;

        image_t *tile;
        // tile_thresholds defaults to nullptr
        // local_tile_thresholds will be set explicitly
        // if user did not pass in valid tile_thresholds value
        local_tile_thresholds = tile_thresholds;

        if (this->input_is_vdb && !local_tile_thresholds) {
          local_tile_thresholds = new TileThresholds<image_t>(
              /*max*/ 2,
              /*min*/ 0,
              /*bkg_thresh*/ 0);
        }

        // pre-generated images are for testing, or when an outside
        // library wants to pass input images instead
        if (this->params->force_regenerate_image) {
          assertm(this->generated_image,
                  "Image not generated or set by intialize");
          tile = this->generated_image;
          // allows users to input a tile thresholds object when
          // calling to update
          if (!local_tile_thresholds) {
            // note these default thresholds apply to any generated image
            // thus they will only be replaced if we're reading a real image
            local_tile_thresholds = new TileThresholds<image_t>(
                /*max*/ 2,
                /*min*/ 0,
                /*bkg_thresh*/ 0);
          }
        }

#ifdef USE_MCP3D
        // mcp3d_tile must be kept in scope during the processing
        // of this interval otherwise dangling reference then seg fault
        // on image access so prevent destruction before calling
        // process_interval
        mcp3d::MImage *mcp3d_tile;
        if (!input_is_vdb) {
          mcp3d_tile =
              new mcp3d::MImage(args->image_root_dir(), {args->channel()});
          // tile is only needed for the value stage
          if (stage == "connected" || stage == "convert") {
            if (!(this->params->force_regenerate_image)) {
              load_tile(interval_id, *mcp3d_tile);
              if (!local_tile_thresholds) {
                auto thresh_start = timer->elapsed();
                local_tile_thresholds = get_tile_thresholds(*mcp3d_tile);
                computation_time =
                    computation_time + (timer->elapsed() - thresh_start);
              }
              tile = mcp3d_tile->Volume<image_t>(0);
            }
          }
        }
#else
        if (!(this->params->force_regenerate_image)) {
          assertm(false,
                  "If USE_MCP3D macro is not set, "
                  "this->params->force_regenerate_image must be set to True");
        }
#endif

        if (stage == "convert") {
#ifdef USE_VDB
          assertm(!this->input_is_vdb,
                  "input can't be vdb during convert stage");

          GridCoord no_offsets = {0, 0, 0};
          auto interval_offsets = id_interval_to_img_offsets(interval_id);

          GridCoord buffer_offsets =
              params->force_regenerate_image ? interval_offsets : no_offsets;
          GridCoord buffer_extents = params->force_regenerate_image
                                         ? this->image_lengths
                                         : this->interval_lengths;

          auto convert_start = timer->elapsed();

#ifdef FULL_PRINT
          print_image_3D(tile, coord_to_vec(buffer_extents));
#endif

          std::vector<PositionT> positions;
          // use the last bkg_thresh calculated for metadata,
          // bkg_thresh is constant for each interval unless a specific % is
          // input by command line user
          convert_buffer_to_vdb(tile, buffer_extents,
                                /*buffer_offsets=*/buffer_offsets,
                                /*image_offsets=*/interval_offsets, positions,
                                local_tile_thresholds->bkg_thresh);
          // grid_transform must use the same voxel size for all intervals
          // and be identical
          auto grid_transform = openvdb::math::Transform::createLinearTransform(VOXEL_SIZE);
          grids[interval_id] =
              create_point_grid(positions, this->image_lengths, grid_transform,
                                local_tile_thresholds->bkg_thresh);
#ifdef FULL_PRINT
          print_vdb(grids[interval_id]->getConstAccessor(),
                    coord_to_vec(this->image_lengths));
#endif
          computation_time =
              computation_time + (timer->elapsed() - convert_start);

          active_intervals[interval_id] = false;
#ifdef LOG
          cout << "Completed interval id: " << interval_id << '\n';
#endif
#endif
        } else {
          computation_time =
              computation_time +
              process_interval(interval_id, tile, stage, local_tile_thresholds,
                               fifo[interval_id], vdb_const_accessor);
        }
      } // if the interval is active

    } // end one interval traversal
  }   // finished all intervals

  if (stage == "convert") {
    auto finalize_start = timer->elapsed();

    assertm(params->convert_only_,
            "reduce grids only possible for convert_only stage");
    for (int i = 0; i < (this->grid_interval_size - 1); ++i) {
      // default op is copy
      // leaves grids[i] empty, copies all to grids[i+1]
      vb::tools::compActiveLeafVoxels(grids[i]->tree(), grids[i + 1]->tree());
    }

    this->topology_grid = grids[this->grid_interval_size - 1];

    //print_positions(this->topology_grid);

    auto finalize_time = timer->elapsed() - finalize_start;
    computation_time = computation_time + finalize_time;
#ifdef LOG
    cout << "Grid finalize time: " << finalize_time << " s\n";
#endif
  } else {
#ifdef RV
    cout << "Total ";
#ifdef NO_RV
    cout << "rejected";
#endif
    cout << " revisits: " << global_revisits << " vertices" << '\n';
#endif
  }

  auto total_update_time = timer->elapsed();
  auto io_time = total_update_time - computation_time;
#ifdef LOG
  cout << "Finished stage: " << stage << '\n';
  cout << "Finished total updating within " << total_update_time << " sec \n";
  cout << "Finished computation (no I/O) within " << computation_time
       << " sec \n";
  cout << "Finished I/O within " << io_time << " sec \n";
  cout << "Total interval iterations: " << outer_iteration_idx << '\n';
//", block iterations: "<< final_inner_iter + 1<< '\n';
#endif

  return std::make_unique<InstrumentedUpdateStatistics>(
      outer_iteration_idx, total_update_time,
      static_cast<double>(computation_time), io_time, interval_open_count);
} // end update()

/*
 * Convert block coordinates into linear index row-ordered
 */
template <class image_t>
inline VID_t Recut<image_t>::sub_block_to_block_id(const VID_t iblock,
                                                   const VID_t jblock,
                                                   const VID_t kblock) {
  return iblock + jblock * this->interval_block_lengths[0] +
         kblock * this->interval_block_lengths[0] *
             this->interval_block_lengths[1];
}

// Wrap-around rotate all values forward one
// This logic disentangles 0 % 32 from 32 % 32 results
// This function is abstract in that it can adjust coordinates
// of vid, block or interval
template <class image_t>
inline VID_t Recut<image_t>::rotate_index(VID_t img_coord, const VID_t current,
                                          const VID_t neighbor,
                                          const VID_t interval_block_size,
                                          const VID_t pad_block_size) {
  // when they are in the same block or index
  // then you simply need to account for the 1
  // voxel border region to get the correct coord
  // for this dimension
  if (current == neighbor) {
    return img_coord + 1; // adjust to padded block idx
  }
  // if it's in another block/interval it can only be 1 vox away
  // so make sure the coord itself is on the correct edge of its block
  // domain
  if (current == (neighbor + 1)) {
    assertm(img_coord == interval_block_size - 1,
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

/*
 * Returns a pointer to the VertexAttr within interval_id,
 * block_id, and img_vid (vid with respect to global image)
 * if not found, creates a new active vertex marked as selected
 */
template <class image_t>
inline VertexAttr *Recut<image_t>::get_or_set_active_vertex(
    const VID_t interval_id, const VID_t block_id, const OffsetCoord offsets,
    bool &found) {
  auto vertex = get_active_vertex(interval_id, block_id, offsets);
  if (vertex) {
    found = true;
    assertm(vertex->selected() || vertex->root(),
            "all active vertices must be selected");
    return vertex;
  } else {
    found = false;
    auto v =
        &(this->active_vertices[interval_id][block_id].emplace_back(offsets));
    v->mark_selected();
    return v;
  }
}

/*
 * Returns a pointer to the VertexAttr within interval_id,
 * block_id, and img_vid (vid with respect to global image)
 */
template <class image_t>
inline VertexAttr *
Recut<image_t>::get_active_vertex(const VID_t interval_id, const VID_t block_id,
                                  const OffsetCoord offsets) {
  for (auto &v : this->active_vertices[interval_id][block_id]) {
    if (v.offsets == offsets) {
      return &v;
    }
  }
  return nullptr;
}

template <class image_t>
void Recut<image_t>::initialize_globals(const VID_t &grid_interval_size,
                                        const VID_t &interval_block_size) {

  // active boolean for in interval domain in block_id ghost region, in
  // domain of block
  this->active_neighbors = vector<vector<vector<bool>>>(
      grid_interval_size,
      vector<vector<bool>>(interval_block_size,
                           vector<bool>(interval_block_size)));

  if (!(params->convert_only_)) {
    updated_ghost_vec = vector<vector<map<VID_t, vector<VertexAttr>>>>(
        grid_interval_size,
        vector<map<VID_t, vector<VertexAttr>>>(
            interval_block_size, map<VID_t, vector<VertexAttr>>()));
#ifdef LOG_FULL
    cout << "Created updated ghost vec" << '\n';
#endif
  }

  // Initialize.
  vector<vector<atomwrapper<bool>>> temp(
      grid_interval_size, vector<atomwrapper<bool>>(interval_block_size));
  for (auto interval = 0; interval < grid_interval_size; ++interval) {
    vector<atomwrapper<bool>> inner(interval_block_size);
    for (auto &e : inner) {
      e.store(false, memory_order_relaxed);
    }
    temp[interval] = inner;
  }
  this->active_blocks = temp;

  this->active_intervals = vector(grid_interval_size, false);

#ifdef LOG_FULL
  cout << "Created active blocks" << '\n';
#endif

  // Initialize.
  vector<vector<atomwrapper<bool>>> temp2(
      grid_interval_size, vector<atomwrapper<bool>>(interval_block_size));
  for (auto interval = 0; interval < grid_interval_size; ++interval) {
    vector<atomwrapper<bool>> inner(interval_block_size);
    for (auto &e : inner) {
      e.store(false, memory_order_relaxed);
    }
    temp2[interval] = inner;
  }
  this->processing_blocks = temp2;

#ifdef LOG_FULL
  cout << "Created processing blocks" << '\n';
#endif

  if (!params->convert_only_) {
    // fifo is a deque representing the vids left to
    // process at each stage
    this->global_fifo = std::vector<std::vector<std::deque<VertexAttr>>>(
        grid_interval_size, std::vector<std::deque<VertexAttr>>(
                                interval_block_size, std::deque<VertexAttr>()));

    this->local_fifo = std::vector<std::vector<std::deque<VertexAttr>>>(
        grid_interval_size, std::vector<std::deque<VertexAttr>>(
                                interval_block_size, std::deque<VertexAttr>()));

    // global active vertex list
    this->active_vertices = std::vector<std::vector<std::vector<VertexAttr>>>(
        grid_interval_size,
        std::vector<std::vector<VertexAttr>>(interval_block_size,
                                             std::vector<VertexAttr>()));
  }
}

// Deduce lengths from the various input options
template <class image_t>
GridCoord Recut<image_t>::get_input_image_lengths(bool force_regenerate_image,
                                                  RecutCommandLineArgs *args) {
  GridCoord input_image_lengths(3);
  if (this->params->force_regenerate_image) {
    // for generated image runs trust the args->image_lengths
    // to reflect the total global image domain
    // see get_args() in utils.hpp
    input_image_lengths = args->image_lengths;

    // FIXME placeholder grid
    this->topology_grid =
        create_vdb_grid(input_image_lengths, this->params->background_thresh());
    append_attributes(this->topology_grid);
  } else if (this->input_is_vdb) {

    assertm(!params->convert_only_,
            "Convert only option is not valid from vdb to vdb");

#ifndef USE_VDB
    assertm(false, "Input must either be regenerated, vdb or from image, "
                   "USE_VDB must be defined");
#endif

    std::string grid_name = "topology";
    auto timer = new high_resolution_timer();
    auto base_grid = read_vdb_file(args->image_root_dir(), grid_name);
    this->topology_grid =
        openvdb::gridPtrCast<EnlargedPointDataGrid>(base_grid);
    append_attributes(this->topology_grid);
#ifdef LOG
    cout << "Read grid in: " << timer->elapsed() << " s\n";
#endif

    input_image_lengths = get_grid_original_lengths(topology_grid);

  } else {
    // read from image use mcp3d library

#ifndef USE_MCP3D
    assertm(false, "Input must either be regenerated, vdb or from image, "
                   "USE_MCP3D image reading library must be defined");
#endif
    assertm(fs::exists(args->image_root_dir()),
            "Image root directory does not exist");

    // determine the image size
    mcp3d::MImage global_image(args->image_root_dir(), {args->channel()});
    // read data from channel
    global_image.ReadImageInfo(args->resolution_level(), true);
    if (global_image.image_info().empty()) {
      MCP3D_MESSAGE("no supported image formats found in " +
                    args->image_root_dir() + ", do nothing.")
      throw;
    }

    // save to __image_info__.json in corresponding dir
    // global_image.SaveImageInfo();

    // reflects the total global image domain
    auto temp = global_image.xyz_dims(args->resolution_level());
    // reverse mcp3d's z y x order for offsets and lengths
    input_image_lengths = new_grid_coord(temp[2], temp[1], temp[0]);
    // FIXME remove this, don't require
    this->topology_grid = create_vdb_grid(input_image_lengths);
    append_attributes(this->topology_grid);
  }
  return input_image_lengths;
}

template <class image_t> const std::vector<VID_t> Recut<image_t>::initialize() {

#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
  omp_set_num_threads(params->user_thread_count());
#ifdef LOG
  cout << "User specific thread count " << params->user_thread_count() << '\n';
  cout << "User specified image root dir " << args->image_root_dir() << '\n';
#endif
#endif
  struct timespec time0, time1, time2, time3;
  uint64_t root_64bit;

  // input type
  {
    auto path_extension =
        std::string(fs::path(args->image_root_dir()).extension());
    this->input_is_vdb = path_extension == ".vdb" ? true : false;
    // if (params->convert_only_ || this->input_is_vdb) {
    // openvdb::initialize();
    //}
    // FIXME should only apply to vdb
    openvdb::initialize();
    // throws if run more than once
    // EnlargedPointDataGrid::registerGrid();
  }

  // actual possible lengths
  auto input_image_lengths =
      get_input_image_lengths(this->params->force_regenerate_image, args);

  // account and check requested args->image_offsets and args->image_lengths
  // extents are always the side length of the domain on each dim, in x y z
  // order
  assertm(coord_all_lt(args->image_offsets, input_image_lengths),
          "input offset can not exceed dimension of image");

  // default image_offsets is {0, 0, 0}
  // which means start at the beginning of the image
  // this enforces the minimum extent to be 1 in each dim
  // set and no longer refer to args->image_offsets
  this->image_offsets = args->image_offsets;

  // protect faulty out of bounds input if extents goes beyond
  // domain of full image
  auto max_len_after_off = coord_sub(input_image_lengths, this->image_offsets);

  // sanitize in each dimension
  // and protected from faulty offset values
  for (int i = 0; i < 3; i++) {
    // -1,-1,-1 means use to the end of input image
    if (args->image_lengths[i] > 0) {
      // use the input length if possible, or maximum otherwise
      this->image_lengths[i] =
          std::min(args->image_lengths[i], max_len_after_off[i]);
    } else {
      this->image_lengths[i] = max_len_after_off[i];
    }
  }

  // save to globals the actual size of the full image
  // accounting for the input offsets and extents
  // these will be used throughout the rest of the program
  // for convenience
  this->image_size = coord_prod_accum(this->image_lengths);

  // Determine the size of each interval in each dim
  // the image size and offsets override the user inputted interval size
  // continuous id's are the same for current or dst intervals
  // round up (pad)
  if (this->params->convert_only_) {
    // explicitly set by get_args
    if (params->interval_length) {
      this->interval_lengths[0] = params->interval_length;
      this->interval_lengths[1] = params->interval_length;
      this->interval_lengths[2] = params->interval_length;
    } else {
      // images are saved in separate z-planes, so conversion should respect
      // that for best performance constrict so less data is allocated
      // especially in z dimension
      this->interval_lengths[0] = this->image_lengths[0];
      this->interval_lengths[1] = this->image_lengths[1];
      auto recommended_max_mem = GetAvailMem() / 16;
      // guess how many z-depth tiles will fit before a bad_alloc is likely
      auto simultaneous_tiles =
          static_cast<double>(recommended_max_mem) /
          (sizeof(image_t) * this->image_lengths[0] * this->image_lengths[1]);
      // assertm(simultaneous_tiles >= 1, "Tile x and y size too large to fit in
      // system memory (DRAM)");
      this->interval_lengths[2] = std::min(
          simultaneous_tiles, static_cast<double>(this->image_lengths[2]));
    }
  } else if (this->input_is_vdb) {
    this->interval_lengths[0] = this->image_lengths[0];
    this->interval_lengths[1] = this->image_lengths[1];
    this->interval_lengths[2] = this->image_lengths[2];
  } else {
    int default_size = 1024;
    if (params->interval_length) {
      // explicitly set by get_args
      default_size = params->interval_length;
    }
    this->interval_lengths[0] = std::min(default_size, this->image_lengths[0]);
    this->interval_lengths[1] = std::min(default_size, this->image_lengths[1]);
    this->interval_lengths[2] = std::min(default_size, this->image_lengths[2]);
  }

  // determine the length of intervals in each dim
  // rounding up (ceil)
  this->grid_interval_lengths[0] =
      (this->image_lengths[0] + this->interval_lengths[0] - 1) /
      this->interval_lengths[0];
  this->grid_interval_lengths[1] =
      (this->image_lengths[1] + this->interval_lengths[1] - 1) /
      this->interval_lengths[1];
  this->grid_interval_lengths[2] =
      (this->image_lengths[2] + this->interval_lengths[2] - 1) /
      this->interval_lengths[2];

  // the resulting interval size override the user inputted block size
  if (this->params->convert_only_) {
    this->block_lengths[0] = this->interval_lengths[0];
    this->block_lengths[1] = this->interval_lengths[1];
    this->block_lengths[2] = this->interval_lengths[2];
  } else {
    this->block_lengths[0] = std::min(this->interval_lengths[0], LEAF_LENGTH);
    this->block_lengths[1] = std::min(this->interval_lengths[1], LEAF_LENGTH);
    this->block_lengths[2] = std::min(this->interval_lengths[2], LEAF_LENGTH);
  }

  // determine length of blocks that span an interval for each dim
  // this rounds up
  this->interval_block_lengths[0] =
      (this->interval_lengths[0] + this->block_lengths[0] - 1) /
      this->block_lengths[0];
  this->interval_block_lengths[1] =
      (this->interval_lengths[1] + this->block_lengths[1] - 1) /
      this->block_lengths[1];
  this->interval_block_lengths[2] =
      (this->interval_lengths[2] + this->block_lengths[2] - 1) /
      this->block_lengths[2];

  this->grid_interval_size = coord_prod_accum(this->grid_interval_lengths);
  this->interval_block_size = coord_prod_accum(this->interval_block_lengths);

#ifdef LOG
  print_coord(this->image_lengths, "image");
  print_coord(this->interval_lengths, "interval");
  print_coord(this->block_lengths, "block");
  print_coord(this->interval_block_lengths, "interval block lengths");
  std::cout << "intervals per grid: " << grid_interval_size
            << " blocks per interval: " << interval_block_size << '\n';
#endif

#ifdef DENSE
  pad_block_length_x = this->block_lengths[0] + 2;
  pad_block_length_y = this->block_lengths[1] + 2;
  pad_block_length_z = this->block_lengths[2] + 2;
  pad_block_offset =
      pad_block_length_x * pad_block_length_y * pad_block_length_z;
  const VID_t grid_vertex_pad_size =
      pad_block_offset * interval_block_size * grid_interval_size;

  if (grid_interval_size > (2 << 16) - 1) {
    cout << "Number of intervals too high: " << grid_interval_size
         << " try increasing interval size";
    // assert(false);
  }
  if (interval_block_size > (2 << 16) - 1) {
    cout << "Number of blocks too high: " << interval_block_size
         << " try increasing block size";
    // assert(false);
  }

  clock_gettime(CLOCK_REALTIME, &time0);
  grid = Grid(grid_vertex_pad_size, interval_block_size, grid_interval_size,
              *this, this->mmap_);

  clock_gettime(CLOCK_REALTIME, &time2);

#ifdef LOG
  cout << "Created grid in " << diff_time(time0, time2) << " s";
  cout << " with total intervals: " << grid_interval_size << '\n';
#endif

#endif // DENSE

  clock_gettime(CLOCK_REALTIME, &time2);
  initialize_globals(grid_interval_size, interval_block_size);

  clock_gettime(CLOCK_REALTIME, &time3);

#ifdef LOG
  cout << "Initialized globals " << diff_time(time2, time3) << '\n';
#endif

  if (this->params->force_regenerate_image) {
    // This is where we set image to our desired values
    this->generated_image = new image_t[this->image_size];

    assertm(this->params->tcase > -1, "Mismatched tcase for generate image");
    assertm(this->params->slt_pct > -1,
            "Mismatched slt_pct for generate image");
    assertm(this->params->selected > 0,
            "Mismatched selected for generate image");
    assertm(this->params->root_vid != numeric_limits<uint64_t>::max(),
            "Root vid uninitialized");

    // create_image take the length of one dimension
    // of the image, currently assuming all test images
    // are cubes
    // sets all to 0 for tcase 4 and 5
    assertm(this->image_lengths[1] == this->image_lengths[2],
            "change create_image implementation to support non-cube images");
    assertm(this->image_lengths[0] == this->image_lengths[1],
            "change create_image implementation to support non-cube images");
    auto selected = create_image(this->params->tcase, this->generated_image,
                                 this->image_lengths[0], this->params->selected,
                                 this->params->root_vid);
    if (this->params->tcase == 3 || this->params->tcase == 5) {
      // only tcase 3 and 5 doens't have a total select count not known
      // ahead of time
      this->params->selected = selected;
    }

    // add the single root vid to the root_vids
    return {this->params->root_vid};

  } else {
    if (params->convert_only_) {
      return {0}; // dummy root vid
    } else {
      // adds all valid markers to root_vids vector and returns
      return process_marker_dir(this->image_offsets, this->image_lengths);
    }
  }
}

#ifdef DENSE
template <class image_t>
inline void Recut<image_t>::get_img_coord(const VID_t id, VID_t &i, VID_t &j,
                                          VID_t &k) {
  i = id % this->image_lengths[0];
  j = (id / this->image_lengths[0]) % this->image_lengths[1];
  k = (id / this->image_lengths[0] * this->image_lengths[1]) %
      this->image_lengths[2];
}

/* intervals are arranged in c-order in 3D, therefore each
 * intervals can be accessed via it's intervals coord or by a linear idx
 * id : the linear idx of this interval in the entire domain
 * i, j, k : the coord to this interval
 */
template <class image_t>
void Recut<image_t>::get_interval_coord(const VID_t id, VID_t &i, VID_t &j,
                                        VID_t &k) {
  i = id % this->grid_interval_lengths[0];
  j = (id / this->grid_interval_lengths[0]) % this->grid_interval_lengths[1];
  k = (id / (this->grid_interval_lengths[0] * this->grid_interval_lengths[1])) %
      this->grid_interval_lengths[2];
}

/*
 * blocks are arranged in c-order in 3D, therefore each block can
 * be accessed via it's block coord or by a linear idx
 * id : the blocks linear idx with respect to the individual
 *      interval
 * i, j, k : the equivalent coord to this block
 */
template <class image_t>
inline void Recut<image_t>::get_block_coord(const VID_t id, VID_t &i, VID_t &j,
                                            VID_t &k) {
  i = id % this->interval_block_lengths[0];
  j = (id / this->interval_block_lengths[0]) % this->interval_block_lengths[1];
  k = (id /
       (this->interval_block_lengths[0] * this->interval_block_lengths[1])) %
      this->interval_block_lengths[2];
}

/*
 * Returns a pointer to the VertexAttr within interval_id,
 * block_id, and img_vid (vid with respect to global image)
 * Note each block actually spans (interval_block_size + 2) ^ 3
 * total vertices in memory, this is because each block needs
 * to hold a redundant copy of all border regions of its cube
 * border regions are also denoted as "ghost" cells/vertices
 * this creates complexity in that the requested vid passed
 * to this function may be referring to a VID that's within
 * the bounds of a separate block if one were to refer to the
 * interval_block_size alone.
 */
template <class image_t>
inline VertexAttr *
Recut<image_t>::get_vertex_vid(const VID_t interval_id, const VID_t block_id,
                               const VID_t img_vid, VID_t *output_offset) {
  VID_t i, j, k, img_block_i, img_block_j, img_block_k;
  VID_t pad_img_block_i, pad_img_block_j, pad_img_block_k;
  i = j = k = 0;

  Interval *interval = grid.GetInterval(interval_id);
  assert(interval->IsInMemory());
  VertexAttr *vertex = interval->GetData(); // first vertex of entire interval

  // block start calculates starting vid of block_id's first vertex within
  // the global interval array of structs Note: every vertex within this block
  // (including the ghost region) will be contiguous between vertex and vertex
  // + (pad_block_offset - 1) stored in row-wise order with respect to the
  // cubic block blocks within the interval are always stored according to
  // their linear block num such that a block_id * the total number of padded
  // vertices in a block i.e. pad_block_offset or (interval_block_size + 2) ^
  // 2 yields offset to the offset to the first vertex of the block.
  VID_t block_start = pad_block_offset * block_id;
  auto first_block_vertex = vertex + block_start;

  // Find correct offset into block

  // first convert from tile id to non- padded block coords
  get_img_coord(img_vid, i, j, k);
  // in case interval_length isn't evenly divisible by block size
  // mod out any contributions from the interval
  auto ia = i % this->interval_lengths[0];
  auto ja = j % this->interval_lengths[1];
  auto ka = k % this->interval_lengths[2];
  // these are coordinates within the non-padded block domain
  // these values will be modified by rotate_index to account for padding
  img_block_i = ia % this->block_lengths[0];
  img_block_j = ja % this->block_lengths[1];
  img_block_k = ka % this->block_lengths[2];
  // cout << "\timg vid: " << img_vid << " img_block_i " << img_block_i
  //<< " img_block_j " << img_block_j << " img_block_k " << img_block_k<<'\n';

  // which block domain and interval does this img_vid actually belong to
  // ignoring ghost regions denoted nb_* since it may belong in the domain
  // of a neighbors block or interval all block ids are a linear row-wise
  // idx, relative to current interval
  int nb_block = (int)id_img_to_block_id(img_vid);
  int nb_interval = (int)id_img_to_interval_id(img_vid);
  // cout << "nb_interval " << nb_interval << " nb_block " << nb_block <<
  // '\n';

  // adjust block coordinates so they reflect the interval or block they
  // belong to also adjust based on actual 3D padding of block Rotate all
  // values forward one This logic disentangles 0 % 32 from 32 % 32 results
  // within a block, where ghost region is index -1 and interval_block_size
  if (interval_id == nb_interval) { // common case first
    if (nb_block == block_id) {     // grab the second common case
      pad_img_block_i = img_block_i + 1;
      pad_img_block_j = img_block_j + 1;
      pad_img_block_k = img_block_k + 1;
    } else {
      VID_t iblock, jblock, kblock, nb_iblock, nb_jblock, nb_kblock;
      // the block_id is a linear index into the 3D row-wise arrangement of
      // blocks, converting to coord makes adjustments easier
      get_block_coord(block_id, iblock, jblock, kblock);
      get_block_coord(nb_block, nb_iblock, nb_jblock, nb_kblock);
      assertm(absdiff(iblock, nb_iblock) + absdiff(jblock, nb_jblock) +
                      absdiff(kblock, nb_kblock) ==
                  1,
              "Does not currently support diagonal connections or any "
              "ghost "
              "regions greater that 1");
      pad_img_block_i =
          rotate_index(img_block_i, iblock, nb_iblock, this->block_lengths[0],
                       pad_block_length_x);
      pad_img_block_j =
          rotate_index(img_block_j, jblock, nb_jblock, this->block_lengths[1],
                       pad_block_length_y);
      pad_img_block_k =
          rotate_index(img_block_k, kblock, nb_kblock, this->block_lengths[2],
                       pad_block_length_z);
    }
  } else { // ignore block info, adjust based on interval
    // the interval_id is also linear index into the 3D row-wise arrangement
    // of intervals, converting to coord makes adjustments easier
    VID_t iinterval, jinterval, kinterval, nb_iinterval, nb_jinterval,
        nb_kinterval;
    get_interval_coord(interval_id, iinterval, jinterval, kinterval);
    get_interval_coord(nb_interval, nb_iinterval, nb_jinterval, nb_kinterval);
    // can only be 1 interval away
    assertm(absdiff(iinterval, nb_iinterval) +
                    absdiff(jinterval, nb_jinterval) +
                    absdiff(kinterval, nb_kinterval) ==
                1,
            "Does not currently support diagonal connections or any ghost "
            "regions greater that 1");
    // check that its in correct block of other interval, can only be 1 block
    // over note that all block ids are relative to their interval so this
    // requires a bit more logic to check, even when converting to coords
#ifndef NDEBUG
    VID_t iblock, jblock, kblock, nb_iblock, nb_jblock, nb_kblock;
    get_block_coord(block_id, iblock, jblock, kblock);
    get_block_coord(nb_block, nb_iblock, nb_jblock, nb_kblock);
    // overload rotate_index simply for the assert checks
    rotate_index(nb_iblock, iinterval, nb_iinterval,
                 this->interval_block_lengths[0], pad_block_length_x);
    rotate_index(nb_jblock, jinterval, nb_jinterval,
                 this->interval_block_lengths[1], pad_block_length_y);
    rotate_index(nb_kblock, kinterval, nb_kinterval,
                 this->interval_block_lengths[2], pad_block_length_z);
    // do any block dimensions differ than current
    auto idiff = absdiff(iblock, nb_iblock) != 0;
    auto jdiff = absdiff(jblock, nb_jblock) != 0;
    auto kdiff = absdiff(kblock, nb_kblock) != 0;
    if ((idiff && (jdiff || kdiff)) || (jdiff && (idiff || kdiff)) ||
        (kdiff && (idiff || jdiff))) {
      cout << "\t\tiblock " << iblock << " nb_iblock " << nb_iblock << '\n';
      cout << "\t\tjblock " << jblock << " nb_jblock " << nb_jblock << '\n';
      cout << "\t\tkblock " << kblock << " nb_kblock " << nb_kblock << '\n';
      assertm(false,
              "Does not currently support diagonal connections or a ghost "
              "regions greater that 1");
    }
#endif
    // checked by rotate that coord is 1 away
    pad_img_block_i = rotate_index(img_block_i, iinterval, nb_iinterval,
                                   this->block_lengths[0], pad_block_length_x);
    pad_img_block_j = rotate_index(img_block_j, jinterval, nb_jinterval,
                                   this->block_lengths[1], pad_block_length_y);
    pad_img_block_k = rotate_index(img_block_k, kinterval, nb_kinterval,
                                   this->block_lengths[2], pad_block_length_z);
  }

  // offset with respect to the padded block
  auto offset = pad_img_block_i + pad_block_length_x * pad_img_block_j +
                pad_img_block_k * pad_block_length_x * pad_block_length_y;
  assert(offset < pad_block_offset); // no valid offset is beyond this val

  if (output_offset)
    *output_offset = offset;

  VertexAttr *match = first_block_vertex + offset;
#ifdef FULL_PRINT
  // cout << "\t\tget vertex vid for tile vid: "<< img_vid<< " pad_img_block_i
  // "
  //<< pad_img_block_i << " pad_img_block_j " << pad_img_block_j
  //<< " pad_img_block_k " << pad_img_block_k<<'\n';
  ////cout << "\t\ti " << i << " j " << j << " k " << k<<'\n';
  ////cout << "\t\tia " << ia << " ja " << ja << " ka " << k<<'\n';
  // cout << "\t\tblock_num " << block_id << " nb_block " << nb_block
  //<< " interval num " << interval_id << " nb_interval num " << nb_interval
  //<<
  //'\n'; cout << "\t\toffset " << offset << " block_start " << block_start <<
  // '\n'; cout << "\t\ttotal interval size " << interval->GetNVertices() <<
  // '\n';
  ////assert(block_start + offset < interval->GetNVertices()); // no
  ////valid offset is beyond this val cout << "\t\tmatch-vid " << match->vid
  ///<<
#endif
  return match;
}

// note this function is only valid ifdef DENSE
template <class image_t>
template <typename vertex_t>
void Recut<image_t>::brute_force_extract(vector<vertex_t> &outtree,
                                         bool accept_band,
                                         bool release_intervals) {

  struct timespec time0, time1;
#ifdef FULL_PRINT
  cout << "Generating results." << '\n';
#endif
  clock_gettime(CLOCK_REALTIME, &time0);

  // FIXME terrible performance
  map<VID_t, MyMarker *> vid_to_marker_ptr; // hash set
  // create all valid new marker objects
  VID_t interval_id, block_id;
  for (VID_t vid = 0; vid < this->image_size; vid++) {
    if (filter_by_vid(vid, interval_id, block_id)) {

      auto vertex = get_vertex_vid(interval_id, block_id, vid, nullptr);

#ifdef FULL_PRINT
      // cout << "checking vertex " << *vertex << " at offset " << offset <<
      // '\n';
#endif
      if (filter_by_label(vertex, accept_band)) {
        // don't create redundant copies of same vid
        if (vid_to_marker_ptr.find(vertex->vid) ==
            vid_to_marker_ptr.end()) { // check not already added
#ifdef FULL_PRINT
          // cout << "\tadding vertex " << vertex->vid << '\n';
#endif
          VID_t i, j, k;
          i = j = k = 0;
          // get original i, j, k
          get_img_coord(vertex->vid, i, j, k);
          auto marker = new MyMarker(i, j, k);
          if (vertex->root()) {
            marker->type = 0;
          }
          marker->radius = vertex->radius;
          vid_to_marker_ptr[vertex->vid] =
              marker; // save this marker ptr to a map
          outtree.push_back(marker);
        } else {
          // check that all copied across blocks and intervals of a
          // single vertex all match same values
          // FIXME this needs to be moved to recut_test.cpp
          // auto previous_match = vid_to_marker_ptr[vertex->vid];
          // assert(*previous_match == *vertex);
          assertm(false, "Can't have two matching vids");
        }
      }
    }
  }

  for (VID_t vid = 0; vid < this->image_size; vid++) {
    if (filter_by_vid(vid, interval_id, block_id)) {
      auto vertex = get_vertex_vid(interval_id, block_id, vid, nullptr);
      if (filter_by_label(vertex, accept_band)) {
        // different copies have same values
        if (!(vertex->valid_parent()) && !(vertex->root())) {
          std::cout << "could not find a valid connection for non-root node: "
                    << vertex->description() << '\n';
          printf("with address of %p\n", (void *)vertex);
          std::cout << "in interval " << interval_id << '\n';
          assertm(!(vertex->root()), "incorrect valid_parent status");
          assertm(vertex->valid_parent() == false,
                  "incorrect valid_parent status");
          assertm(false, "must have a valid connection");
        }
        auto parent_vid = vertex->parent;
        auto marker = vid_to_marker_ptr[vertex->vid]; // get the ptr
        if (vertex->root()) {
          marker->parent = 0;
#ifdef FULL_PRINT
          cout << "found root at " << vertex->vid << '\n';
          printf("with address of %p\n", (void *)vertex);
#endif
          assertm(marker->parent == 0,
                  "a marker with a parent of 0, must be a root");
          assertm(marker->type == 0,
                  "a marker with a type of 0, must be a root");
        } else {
          auto parent = vid_to_marker_ptr[parent_vid]; // adjust
          marker->parent = parent;
          if (marker->parent == 0) {
            // failure condition
            std::cout << "\ninterval vid " << interval_id << '\n';
            std::cout << "block vid " << block_id << '\n';
            print_vertex(interval_id, block_id, vertex);
            assertm(marker->parent != 0,
                    "a non root marker must have a valid parent");
          }
          assertm(marker->type != 0,
                  "a marker with a type of 0, must be a root");
        }
      }
    }
  }

  // release both user-space or mmap'd data since info is all in outtree
  // now
  for (size_t interval_id = 0; interval_id < grid_interval_size;
       ++interval_id) {
    if (release_intervals) {
      grid.GetInterval(interval_id)->Release();
    }
  }

#ifdef LOG
  cout << "Total marker size: " << outtree.size() << " nodes" << '\n';
#endif

  clock_gettime(CLOCK_REALTIME, &time1);
#ifdef FULL_PRINT
  cout << "Finished generating results within " << diff_time(time0, time1)
       << " sec." << '\n';
#endif
}

#endif // DENSE

// reject unvisited vertices
// band can be optionally included
template <class image_t>
bool Recut<image_t>::filter_by_label(VertexAttr *v, bool accept_band) {
  if (accept_band) {
    if (v->unvisited()) {
      return false;
    }
  } else {
    assertm(!(v->band()), "BAND vertex was lost");
    if (v->unvisited() || v->band()) {
      return false; // skips unvisited 11XX XXXX and band 01XX XXXX
    }
  }
  if (v->radius < 1) {
    std::cout << v->description();
    assertm(false, "can't accept a vertex with a radii < 1");
  }
  return true;
};

template <class image_t> void Recut<image_t>::adjust_parent(bool to_swc_file) {
#ifdef LOG
  cout << "Start stage adjust_parent\n";
#endif

  if (to_swc_file) {
    this->out.open(this->args->swc_path());
    std::ostringstream line;
    line << "#id type_id x y z radius parent_id\n";
    this->out << line.str();
  }

  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    auto interval_img_offsets = id_interval_to_img_offsets(interval_id);
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      auto block_img_offsets = id_block_to_interval_offsets(block_id);
      auto offsets = coord_add(interval_img_offsets, block_img_offsets);
      for (auto &v : this->active_vertices[interval_id][block_id]) {
        if (filter_by_label(&v, false)) {
          adjust_vertex_parent(&v, offsets);
          print_vertex(interval_id, block_id, &v, offsets);
        }
      }
    }
  }

  if (this->out.is_open())
    this->out.close();
}

template <class image_t>
bool Recut<image_t>::filter_by_vid(VID_t vid, VID_t find_interval_id,
                                   VID_t find_block_id) {
  find_interval_id = id_img_to_interval_id(vid);
  if (find_interval_id >= grid_interval_size) {
    return false;
  }
  find_block_id = id_img_to_block_id(vid);
  if (find_block_id >= interval_block_size) {
    return false;
  }
#ifdef DENSE
  if (!(grid.GetInterval(find_interval_id)->IsInMemory())) {
    grid.GetInterval(find_interval_id)->LoadFromDisk();
  }
#endif
  return true;
};

// accept_band is a way to see pruned vertices still in active_vertex
template <class image_t>
template <typename vertex_t>
void Recut<image_t>::convert_to_markers(vector<vertex_t> &outtree,
                                        bool accept_band) {

  struct timespec time0, time1;
#ifdef FULL_PRINT
  cout << "Generating results." << '\n';
#endif
  clock_gettime(CLOCK_REALTIME, &time0);

  // get a mapping to stable address pointers in outtree such that a markers
  // parent is valid pointer when returning just outtree
  map<VID_t, MyMarker *> vid_to_marker_ptr;

  // iterate all active vertices ahead of time so each marker
  // can have a pointer to it's parent marker
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    auto interval_img_offsets = id_interval_to_img_offsets(interval_id);
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      auto block_img_offsets = id_block_to_interval_offsets(block_id);
      auto offsets = coord_add(interval_img_offsets, block_img_offsets);
      for (auto &vertex_value : this->active_vertices[interval_id][block_id]) {
        // FIXME add interval offset too
        auto vertex = &vertex_value;
        auto coord = coord_add(vertex->offsets, offsets);
        auto vid = coord_to_id(coord, this->image_lengths);
#ifdef FULL_PRINT
        print_vertex(interval_id, block_id, vertex, offsets);
#endif
        // create all valid new marker objects
        if (filter_by_label(vertex, accept_band)) {
          // don't create redundant copies of same vid
          // check that all copied across blocks and intervals of a
          // single vertex all match same values
          // FIXME this needs to be moved to recut_test.cpp
          // auto previous_match = vid_to_marker_ptr[vid];
          // assert(*previous_match == *vertex);
          if (vid_to_marker_ptr.count(vid)) {
            std::cout << interval_id << ' ' << block_id << ' ' << vid
                      << " has parent " << coord_to_str(vertex->parent) << '\n';
          }
          assertm(vid_to_marker_ptr.count(vid) == 0,
                  "Can't have two matching vids");
          // get original i, j, k
          auto marker = new MyMarker(coord[0], coord[1], coord[2]);
          if (vertex->root()) {
            // a marker with a type of 0, must be a root
            marker->type = 0;
          }
          marker->radius = vertex->radius;
          // save this marker ptr to a map
          vid_to_marker_ptr[vid] = marker;
          // std::cout << "\t " << interval_id << ' ' << block_id << ' ' <<
          // vid
          std::cout << "\t " << coord_to_str(coord) << " -> " << vertex->parent
                    << '\n';
          outtree.push_back(marker);
        }
      }
    }
  }

  // know that a pointer to all desired markers is known
  // iterate and complete the marker definition
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    auto interval_img_offsets = id_interval_to_img_offsets(interval_id);
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      auto block_img_offsets = id_block_to_interval_offsets(block_id);
      auto offsets = coord_add(interval_img_offsets, block_img_offsets);
      for (auto &vertex_value : this->active_vertices[interval_id][block_id]) {
        auto vertex = &vertex_value;
        auto coord = coord_add(vertex->offsets, offsets);
        auto vid = coord_to_id(coord, this->image_lengths);
        // only consider the same vertices as above
        if (filter_by_label(vertex, accept_band)) {
          auto parent_vid = coord_to_id(coord_add(coord, vertex->parent),
                                        this->image_lengths);
          assertm(vid_to_marker_ptr.count(vid),
                  "did not find vertex in marker map");
          auto marker = vid_to_marker_ptr[vid]; // get the ptr
          if (vertex->root()) {
            // a marker with a parent of 0, must be a root
            marker->parent = 0;
            //#ifdef FULL_PRINT
            // cout << "found root at " << vid << '\n';
            // printf("with address of %p\n", (void *)vertex);
            //#endif
          } else {
            assertm(vid_to_marker_ptr.count(vid),
                    "did not find vertex in marker map");
            auto parent = vid_to_marker_ptr[parent_vid]; // adjust
            marker->parent = parent;
            if (marker->parent == 0) {
              // failure condition
              std::cout << "interval vid " << interval_id << '\n';
              std::cout << "block vid " << block_id << '\n';
              print_vertex(interval_id, block_id, vertex, offsets);
              assertm(marker->parent != 0,
                      "a non root marker must have a valid parent");
            }
            assertm(marker->type != 0,
                    "a marker with a type of 0, must be a root");
          }
        }
      }
    }
  }

#ifdef LOG
  cout << "Total marker size: " << outtree.size() << " nodes" << '\n';
#endif

  clock_gettime(CLOCK_REALTIME, &time1);
#ifdef FULL_PRINT
  cout << "Finished generating results within " << diff_time(time0, time1)
       << " sec." << '\n';
#endif
}

template <class image_t> void Recut<image_t>::operator()() {
  std::string stage;
  // create a list of root vids
  auto root_vids = this->initialize();

  if (params->convert_only_) {
    activate_all_intervals();

    // mutates topology_grid
    stage = "convert";
    this->update(stage, global_fifo);

    openvdb::GridPtrVec grids;
    grids.push_back(this->topology_grid);
    write_vdb_file(grids, this->params->out_vdb_);

    // no more work to do, exiting
    return;
  }

  // starting from the roots connected stage saves all surface vertices into
  // fifo
  stage = "connected";
  this->activate_vids(root_vids, stage, global_fifo);
  this->update(stage, global_fifo);

  // radius stage will consume fifo surface vertices
  stage = "radius";
  this->setup_radius(global_fifo);
  this->update(stage, global_fifo);

  // starting from roots, prune stage will
  // create final list of vertices
  stage = "prune";
  this->activate_vids(root_vids, stage, global_fifo);
  this->update(stage, global_fifo);

#ifndef DENSE
  // aggregate results, adjust pruned parent if necessary
  auto to_swc_file = true;
  adjust_parent(to_swc_file);
#endif
}
