#pragma once

#include "app2_helpers.hpp"
#include "recut_parameters.hpp"
#include "tile_thresholds.hpp"
#include "utils.hpp"
#include <algorithm>
#include <bits/stdc++.h>
#include <bitset>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <execution>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/VolumeToSpheres.h> // for fillWithSpheres
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <unordered_set>

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
  openvdb::math::CoordBBox image_bbox;

  GridCoord image_lengths;
  GridCoord image_offsets;
  GridCoord interval_lengths;
  GridCoord block_lengths;
  GridCoord grid_interval_lengths;
  GridCoord interval_block_lengths;

  bool input_is_vdb;
#ifdef USE_VDB
  EnlargedPointDataGrid::Ptr topology_grid;
  openvdb::FloatGrid::Ptr input_grid;
  openvdb::BoolGrid::Ptr update_grid;
  openvdb::MaskGrid::Ptr mask_grid;
  ImgGrid::Ptr img_grid;
#endif

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
  image_t *generated_image = nullptr;
  atomic<VID_t> global_revisits;
  RecutCommandLineArgs *args;
  RecutParameters *params;
  std::map<GridCoord, std::deque<VertexAttr>> map_fifo;
  std::map<GridCoord, local_heap> heap_map;
  std::map<GridCoord, std::deque<VertexAttr>> connected_map;

  // interval specific global data structures
  vector<bool> active_intervals;

  Recut(RecutCommandLineArgs &args)
      : args(&args), params(&(args.recut_parameters())) {}

  void operator()();
  void print_to_swc();
  void adjust_parent();
  void prune_radii();
  void prune_branch();
  void convert_topology();

  void
  partition_components(std::vector<std::pair<GridCoord, uint8_t>> root_pair,
                       bool prune);

  void initialize_globals(const VID_t &grid_interval_size,
                          const VID_t &interval_block_size);

  bool filter_by_label(VertexAttr *v, bool accept_tombstone);

  image_t get_img_val(const image_t *tile, GridCoord coord);
  inline VID_t rotate_index(VID_t img_coord, const VID_t current,
                            const VID_t neighbor,
                            const VID_t interval_block_size,
                            const VID_t pad_block_size);
  int get_bkg_threshold(const image_t *tile, VID_t interval_vertex_size,
                        const double foreground_percent);
  template <typename T>
  void place_vertex(const VID_t nb_interval_id, VID_t block_id, VID_t nb,
                    struct VertexAttr *dst, GridCoord dst_coord,
                    OffsetCoord msg_offsets, std::string stage, T vdb_accessor);
  bool any_fifo_active(std::map<GridCoord, std::deque<VertexAttr>> &check_fifo);
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

  template <typename T>
  void check_ghost_update(VID_t interval_id, VID_t block_id,
                          GridCoord dst_coord, VertexAttr *dst,
                          std::string stage, T vdb_accessor);
  template <typename IndT, typename FlagsT, typename ParentsT, typename ValueT,
            typename PointIter, typename UpdateIter>
  bool accumulate_value(const image_t *tile, VID_t interval_id, VID_t block_id,
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
  bool accumulate_connected(const image_t *tile, VID_t interval_id,
                            VID_t block_id, GridCoord dst_coord, T2 ind,
                            OffsetCoord offset_to_current, VID_t &revisits,
                            const TileThresholds<image_t> *tile_thresholds,
                            bool &found_adjacent_invalid, PointIter point_leaf,
                            UpdateIter update_leaf, FlagsT flags_handle,
                            ParentsT parents_handle,
                            std::deque<VertexAttr> &connected_fifo);
  bool is_covered_by_parent(VID_t interval_id, VID_t block_id,
                            VertexAttr *current);
  template <class Container, typename IndT, typename RadiusT, typename FlagsT,
            typename UpdateIter>
  void accumulate_prune(VID_t interval_id, VID_t block_id, GridCoord dst_coord,
                        IndT ind, const uint8_t current_radius,
                        bool current_unvisited, Container &fifo,
                        RadiusT radius_handle, FlagsT flags_handle,
                        UpdateIter update_leaf);
  template <class Container, typename T, typename FlagsT, typename RadiusT,
            typename UpdateIter>
  void accumulate_radius(VID_t interval_id, VID_t block_id, GridCoord dst_coord,
                         T ind, const uint8_t current_radius, Container &fifo,
                         FlagsT flags_handle, RadiusT radius_handle,
                         UpdateIter update_leaf);
  void integrate_update_grid(
      EnlargedPointDataGrid::Ptr grid,
      vt::LeafManager<PointTree> grid_leaf_manager, std::string stage,
      std::map<GridCoord, std::deque<VertexAttr>> &fifo,
      std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo,
      openvdb::BoolGrid::Ptr update_grid, VID_t interval_id);
  template <class Container> void dump_buffer(Container buffer);
  template <typename T2>
  void march_narrow_band(const image_t *tile, VID_t interval_id, VID_t block_id,
                         std::string stage,
                         const TileThresholds<image_t> *tile_thresholds,
                         std::deque<VertexAttr> &connected_fifo,
                         std::deque<VertexAttr> &fifo, T2 leaf_iter);
  template <class Container, typename T2>
  void value_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                  std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T2 leaf_iter);
  template <class Container, typename T2>
  void connected_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                      std::string stage,
                      const TileThresholds<image_t> *tile_thresholds,
                      Container &connected_fifo, Container &fifo,
                      VID_t revisits, T2 leaf_iter);
  template <class Container, typename T2>
  void radius_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                   std::string stage,
                   const TileThresholds<image_t> *tile_thresholds,
                   Container &fifo, VID_t revisits, T2 leaf_iter);
  template <class Container, typename T2>
  void prune_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                  std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T2 leaf_iter);
  void create_march_thread(VID_t interval_id, VID_t block_id);
#ifdef USE_TINYTIFF
  vto::Dense<uint16_t, vto::LayoutXYZ> load_tile(const VID_t interval_id,
                                                 const std::string &dir);
  TileThresholds<image_t> *
  get_tile_thresholds(vto::Dense<uint16_t, vto::LayoutXYZ> &tile);
#endif
  template <typename T2>
  std::atomic<double>
  process_interval(VID_t interval_id, const image_t *tile, std::string stage,
                   const TileThresholds<image_t> *tile_thresholds,
                   T2 vdb_accessor);
  template <class Container>
  std::unique_ptr<InstrumentedUpdateStatistics>
  update(std::string stage, Container &fifo = nullptr,
         TileThresholds<image_t> *tile_thresholds = nullptr);
  GridCoord get_input_image_extents(bool force_regenerate_image,
                                    RecutCommandLineArgs *args);
  GridCoord get_input_image_lengths(bool force_regenerate_image,
                                    RecutCommandLineArgs *args);
  std::vector<std::pair<GridCoord, uint8_t>> initialize();
  inline VID_t sub_block_to_block_id(VID_t iblock, VID_t jblock, VID_t kblock);
  template <class Container> void setup_radius(Container &fifo);
  void
  activate_vids(EnlargedPointDataGrid::Ptr grid,
                const std::vector<std::pair<GridCoord, uint8_t>> &roots,
                const std::string stage,
                std::map<GridCoord, std::deque<VertexAttr>> &fifo,
                std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo);
  std::vector<std::pair<GridCoord, uint8_t>>
  process_marker_dir(const GridCoord grid_offsets,
                     const GridCoord grid_extents);
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

// adds all markers to root_pair
template <class image_t>
std::vector<std::pair<GridCoord, uint8_t>>
Recut<image_t>::process_marker_dir(const GridCoord grid_offsets,
                                   const GridCoord grid_extents) {

  auto local_bbox =
      openvdb::math::CoordBBox(grid_offsets, grid_offsets + grid_extents);

#ifdef LOG
  cout << "Processing region: " << local_bbox << '\n';
  print_point_count(this->topology_grid);
#endif

  // input handler
  {
    if (params->marker_file_path().empty())
      return {};

    // allow either dir or dir/ naming styles
    if (params->marker_file_path().back() != '/')
      params->set_marker_file_path(params->marker_file_path().append("/"));

#ifdef LOG
    cout << "marker dir path: " << params->marker_file_path() << '\n';
    assertm(fs::exists(params->marker_file_path()),
            "Marker file path must exist");
#endif
  }

  // gather all markers within directory
  auto inmarkers = std::vector<MyMarker>();
  rng::for_each(
      fs::directory_iterator(params->marker_file_path()),
      [&inmarkers, this](auto marker_file) {
        std::stringstream fn(marker_file.path().filename());
        const auto full_marker_name =
            params->marker_file_path() + marker_file.path().filename().string();
        auto markers = readMarker_file(full_marker_name);
        assertm(markers.size() == 1, "only 1 marker file per soma");

        if (markers[0].radius == 0) {
          std::string mass; // mass is the last number of the file name
          while (std::getline(fn, mass, '_')) {
          }
          assertm(!mass.empty(), "can not deduce radius size of input marker");
          markers[0].radius =
              static_cast<int>(std::cbrt(std::stoi(mass) / (4 * PI / 3)));
        }
#ifdef FULL_PRINT
        cout << "Read marker assigning radius: " << markers[0].radius << '\n';
#endif

        // delete this later
        // cout << "Warning: temporarily adjusting x,y,z of marker\n";
        // markers[0].x += this->image_offsets[0];
        // markers[0].y += this->image_offsets[1];
        // markers[0].z += this->image_offsets[2];

        inmarkers.insert(inmarkers.end(), markers.begin(), markers.end());
      });

  // transform to <coord, radius> of all somas/roots
  auto roots =
      inmarkers | rng::views::transform([this](auto marker) {
        return std::pair{
            ones() + GridCoord(marker.x / params->downsample_factor_,
                               marker.y / params->downsample_factor_,
                               upsample_idx(params->upsample_z_, marker.z)),
            static_cast<uint8_t>(marker.radius)};
      }) |
      rng::to_vector;

  auto filtered_roots =
      roots | rng::views::remove_if([this, &local_bbox](auto coord_radius) {
        auto [coord, radius] = coord_radius;
        if (local_bbox.isInside(coord)) {
          if (this->topology_grid->tree().isValueOn(coord)) {
            return false;
          } else {
#ifdef FULL_PRINT
            cout << "Warning: root at " << coord << " in image bbox "
                 << local_bbox
                 << " is not selected in the segmentation so it is "
                    "ignored. "
                    "May indicate the image and marker directories are  "
                    "mismatched or major inaccuracies in segmentation\n ";
#endif
          }
        } else {
#ifdef FULL_PRINT
          cout << "Warning: root at " << coord << " in image bbox "
               << local_bbox
               << " is not within the images bounding box so it is "
                  "ignored\n";
#endif
        }
        return true; // remove it
      }) |
      rng::to_vector;

#ifdef LOG
  cout << "Using " << filtered_roots.size() << " of " << roots.size()
       << " roots found in directory\n";
#endif

  return filtered_roots;
}

// activates
// the intervals of the leaf and reads
// them to the respective heaps
template <class image_t>
template <class Container>
void Recut<image_t>::setup_radius(Container &fifo) {
  for (size_t interval_id = 0; interval_id < grid_interval_size;
       ++interval_id) {
    active_intervals[interval_id] = true;
#ifdef FULL_PRINT
    cout << "Set interval " << interval_id << " to active\n";
#endif
  }
}

template <class image_t>
void Recut<image_t>::activate_vids(
    EnlargedPointDataGrid::Ptr grid,
    const std::vector<std::pair<GridCoord, uint8_t>> &roots,
    const std::string stage, std::map<GridCoord, std::deque<VertexAttr>> &fifo,
    std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo) {

  assertm(!(roots.empty()), "Must have at least one root");

  // Iterate over leaf nodes that contain topology (active)
  // checking for roots within them
  for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
    auto leaf_bbox = leaf_iter->getNodeBoundingBox();
    // std::cout << "Leaf BBox: " << leaf_bbox << '\n';

    // FILTER for those in this leaf
    auto leaf_roots = roots | rng::views::transform([](auto coord_radius) {
                        return coord_radius.first;
                      }) |
                      rng::views::remove_if([&leaf_bbox](GridCoord coord) {
                        return !leaf_bbox.isInside(coord);
                      }) |
                      rng::to_vector;

    if (leaf_roots.empty())
      continue;

    this->active_intervals[0] = true;
    // print_iter_name(leaf_roots, "\troots");

    // Set Values
    auto update_leaf = this->update_grid->tree().probeLeaf(leaf_bbox.min());
    assertm(update_leaf, "Update must have a corresponding leaf");

    rng::for_each(leaf_roots, [&update_leaf](auto coord) {
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

      rng::for_each(leaf_roots, [&](auto coord) {
        auto ind = leaf_iter->beginIndexVoxel(coord);
        if (this->args->type_ == "float") {
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

      rng::for_each(leaf_roots, [&](auto coord) {
        auto ind = leaf_iter->beginIndexVoxel(coord);
        assertm(ind,
                "All root coords must be filtered with respect to topology");
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

  auto parent =
      get_active_vertex(parent_interval_id, parent_block_id, current->parent);

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
      potential_new_parent = get_active_vertex(interval_id, block_id,
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
    VID_t interval_id, VID_t block_id, GridCoord dst_coord, IndT ind,
    const uint8_t current_radius, bool current_unvisited, Container &fifo,
    RadiusT radius_handle, FlagsT flags_handle, UpdateIter update_leaf) {
  if (ind && is_selected(flags_handle, ind)) {
#if FULL_PRINT
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
      // to other blocks / intervals
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
 * on VertexAttr that are within the current interval_id and block_id, since
 * it is potentially adding these vertices to the unique heap of interval_id
 * and block_id. only one parent when selected. If one of these vertexes on
 * the edge but still within interval_id and block_id domain is updated it
 * is the responsibility of check_ghost_update to take note of the update such
 * that this update is propagated to the relevant interval and block see
 * integrate_update_grid(). dst_coord : continuous vertex id VID_t of the dst
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <class Container, typename T, typename FlagsT, typename RadiusT,
          typename UpdateIter>
void Recut<image_t>::accumulate_radius(VID_t interval_id, VID_t block_id,
                                       GridCoord dst_coord, T ind,
                                       const uint8_t current_radius,
                                       Container &fifo, FlagsT flags_handle,
                                       RadiusT radius_handle,
                                       UpdateIter update_leaf) {
  // although current vertex can belong in the boundary
  // region of a separate block /interval it must be only
  // 1 voxel away (within this block /interval's ghost region)
  // therefore all neighbors / destinations of current
  // must be checked to make sure they protude into
  // the actual current block / interval region
  if (ind && is_selected(flags_handle, ind)) {
#if FULL_PRINT
    std::cout << "\tcheck foreground dst: " << coord_to_str(dst_coord) << '\n';
#endif

    const uint8_t updated_radius = 1 + current_radius;
    auto dst_radius = radius_handle.get(*ind);

    // if radius not set yet it necessitates it is 1 higher than current OR an
    // update from another block / interval creates new lower updates
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
 * operate on a VertexAttr (voxel) that are within the current interval_id and
 * block_id, since it is potentially adding these vertices to the unique heap of
 * interval_id and block_id. only one parent when selected. If one of these
 * vertexes on the edge but still within interval_id and block_id domain is
 * updated it is the responsibility of check_ghost_update to take note of the
 * update such that this update is propagated to the relevant interval and block
 * see vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <typename IndT, typename FlagsT, typename ParentsT, typename ValueT,
          typename PointIter, typename UpdateIter>
bool Recut<image_t>::accumulate_value(
    const image_t *tile, VID_t interval_id, VID_t block_id, GridCoord dst_coord,
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
template <typename T2, typename FlagsT, typename ParentsT, typename PointIter,
          typename UpdateIter>
bool Recut<image_t>::accumulate_connected(
    const image_t *tile, VID_t interval_id, VID_t block_id, GridCoord dst_coord,
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
 * or intervals safely by leveraging update_grid
 */
template <class image_t>
template <typename T>
void Recut<image_t>::place_vertex(const VID_t nb_interval_id,
                                  const VID_t block_id, const VID_t nb_block_id,
                                  struct VertexAttr *dst, GridCoord dst_coord,
                                  OffsetCoord msg_offsets, std::string stage,
                                  T vdb_accessor) {
  active_intervals[nb_interval_id] = true;
  vdb_accessor.setValueOn(dst_coord);

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
template <typename T>
void Recut<image_t>::check_ghost_update(VID_t interval_id, VID_t block_id,
                                        GridCoord dst_coord, VertexAttr *dst,
                                        std::string stage, T vdb_accessor) {
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
        place_vertex(nb_interval_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(this->block_lengths[0], dst_offsets[1],
                                      dst_offsets[2]),
                     stage, vdb_accessor);
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
        place_vertex(nb_interval_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], this->block_lengths[1],
                                      dst_offsets[2]),
                     stage, vdb_accessor);
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
        place_vertex(nb_interval_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], dst_offsets[1],
                                      this->block_lengths[2]),
                     stage, vdb_accessor);
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
        place_vertex(nb_interval_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], dst_offsets[1], -1),
                     stage, vdb_accessor);
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
        place_vertex(nb_interval_id, block_id, nb, dst, dst_coord,
                     new_offset_coord(dst_offsets[0], -1, dst_offsets[2]),
                     stage, vdb_accessor);
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
        place_vertex(nb_interval_id, block_id, nb, dst, dst_coord,
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
      stencil_offsets |
      rng::views::transform([&start_coord](auto stencil_offset) {
        return std::pair{/*rel. offset*/ stencil_offset,
                         coord_add(start_coord, stencil_offset)};
      }) |
      // get the corresponding leaf from update grid
      rng::views::transform([update_grid](auto coord_pair) {
        return std::pair{/*rel. offset*/ coord_pair.first,
                         update_grid->tree().probeConstLeaf(coord_pair.second)};
      }) |
      // does adj leaf have any border topology?
      rng::views::remove_if([](auto leaf_pair) {
        if (leaf_pair.second) {
          return leaf_pair.second->isEmpty(); // any of values true
        } else {
          return true;
        }
      }) |
      // for each adjacent leaf with values on
      rng::views::transform([&](auto leaf_pair) {
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
 * intervals and blocks can receive all of their updates from the current
 * iterations run of march_narrow_band safely to complete the iteration
 */
template <class image_t>
void Recut<image_t>::integrate_update_grid(
    EnlargedPointDataGrid::Ptr grid,
    vt::LeafManager<PointTree> grid_leaf_manager, std::string stage,
    std::map<GridCoord, std::deque<VertexAttr>> &fifo,
    std::map<GridCoord, std::deque<VertexAttr>> &connected_fifo,
    openvdb::BoolGrid::Ptr update_grid, VID_t interval_id) {
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
  //#ifdef LOG_FULL
  // cout << '\n' << tot_active << " total blocks active" << '\n';
  //#endif
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
template <class Container, typename T2>
void Recut<image_t>::value_tile(const image_t *tile, VID_t interval_id,
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
        this->stencil |
        rng::views::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this, &found_adjacent_invalid](auto coord_img) {
          if (this->image_bbox.isInside(coord_img))
            return false;
          found_adjacent_invalid = true;
          return true;
        }) |
        // within leaf?
        rng::views::remove_if([&](auto coord_img) {
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
        rng::views::remove_if([&](auto coord_img) {
          auto offset_to_current = coord_sub(msg_coord, coord_img);
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // is background?  ...has side-effects
          return !accumulate_value(tile, interval_id, block_id, coord_img, ind,
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
        // each fifo corresponds to a specific interval_id and block_id
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
    const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
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
        this->stencil |
        rng::views::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this, &found_adjacent_invalid](auto coord_img) {
          if (this->image_bbox.isInside(coord_img))
            return false;
          found_adjacent_invalid = true;
          return true;
        }) |
        // within leaf?
        rng::views::remove_if([&](auto coord_img) {
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
        rng::views::remove_if([&](auto coord_img) {
          auto offset_to_current = coord_sub(msg_coord, coord_img);
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // is background?  ...has side-effects
          return !accumulate_connected(
              tile, interval_id, block_id, coord_img, ind, offset_to_current,
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
        // each fifo corresponds to a specific interval_id and block_id
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
void Recut<image_t>::radius_tile(const image_t *tile, VID_t interval_id,
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
        this->stencil |
        rng::views::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this](auto coord_img) {
          return !this->image_bbox.isInside(coord_img);
        }) |
        // within leaf?
        rng::views::remove_if(
            [&](auto coord_img) { return !bbox.isInside(coord_img); }) |
        rng::views::transform([&](auto coord_img) { return coord_img; }) |
        // visit valid voxels
        rng::views::transform([&](auto coord_img) {
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // ...has side-effects
          accumulate_radius(interval_id, block_id, coord_img, ind,
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
void Recut<image_t>::prune_tile(const image_t *tile, VID_t interval_id,
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
    // all block ids are a linear row-wise idx, relative to current interval
    cout << '\n'
         << coord_to_str(msg_coord) << " interval " << interval_id << " block "
         << bbox.min() << " label " << current->label() << " radius "
         << +(current->radius) << '\n';
#endif

    // force full evaluation by saving to vector
    auto updated_inds =
        // star stencil offsets to img coords
        this->stencil |
        rng::views::transform([&msg_coord](auto stencil_offset) {
          return coord_add(msg_coord, stencil_offset);
        }) |
        // within image?
        rng::views::remove_if([this](auto coord_img) {
          return !this->image_bbox.isInside(coord_img);
        }) |
        // within leaf?
        rng::views::remove_if(
            [&](auto coord_img) { return !bbox.isInside(coord_img); }) |
        // visit valid voxels
        rng::views::transform([&](auto coord_img) {
          auto ind = leaf_iter->beginIndexVoxel(coord_img);
          // ...has side-effects
          accumulate_prune(interval_id, block_id, coord_img, ind,
                           current->radius, current->tombstone(), fifo,
                           radius_handle, flags_handle, update_leaf);
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
    const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds,
    std::deque<VertexAttr> &connected_fifo, std::deque<VertexAttr> &fifo,
    T2 leaf_iter) {
#ifdef FULL_PRINT
  auto timer = high_resolution_timer();
  auto loc = tree_to_str(interval_id, block_id);
  // cout << "\nMarching " << loc << ' ' << leaf_iter->origin() << '\n';
#endif

  VID_t revisits = 0;

  if (stage == "value") {
    value_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
               revisits, leaf_iter);
  } else if (stage == "connected") {
    connected_tile(tile, interval_id, block_id, stage, tile_thresholds,
                   connected_fifo, fifo, revisits, leaf_iter);
  } else if (stage == "radius") {
    radius_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
                revisits, leaf_iter);
  } else if (stage == "prune") {
    prune_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
               revisits, leaf_iter);
  } else {
    assertm(false, "Stage name not recognized");
  }

#ifdef FULL_PRINT
  // cout << "Marched " << loc << " in " << timer.elapsed() << " s" << '\n';
#endif

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
  auto timer = high_resolution_timer();
  assertm(foreground_percent >= 0., "foreground_percent must be 0 or positive");
  assertm(foreground_percent <= 100., "foreground_percent must be 100 or less");
  const double foreground_ratio = foreground_percent / 100;
  const double desired_bkg_pct = 1. - foreground_ratio;

  image_t above; // store next bkg_thresh value above desired bkg pct
  image_t below = 0;
  double above_diff_pct = 0.0; // pct bkg at next above value
  double below_diff_pct = 1.;  // last below percentage
  // test different background threshold values until finding
  // percentage above desired_bkg_pct or when all pixels set to background
  VID_t bkg_count;
  for (image_t local_bkg_thresh = 0;
       local_bkg_thresh <= std::numeric_limits<image_t>::max();
       ++local_bkg_thresh) {

    // Count total # of pixels under current thresh
    bkg_count = 0;
#if defined USE_OMP_BLOCK
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

#ifdef LOG
      cout << "Calculated bkg thresh in " << timer.elapsed() << '\n';
#endif
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
template <typename T2>
std::atomic<double> Recut<image_t>::process_interval(
    VID_t interval_id, const image_t *tile, std::string stage,
    const TileThresholds<image_t> *tile_thresholds, T2 vdb_accessor) {
  auto timer = high_resolution_timer();

  vt::LeafManager<PointTree> grid_leaf_manager(this->topology_grid->tree());

  integrate_update_grid(this->topology_grid, grid_leaf_manager, stage,
                        this->map_fifo, this->connected_map, this->update_grid,
                        interval_id);

  auto march_range = [&,
                      this](const openvdb::tree::LeafManager<
                            openvdb::points::PointDataTree>::LeafRange &range) {
    // for each leaf with active voxels i.e. containing topology
    for (auto leaf_iter = range.begin(); leaf_iter; ++leaf_iter) {
      auto block_id = coord_img_to_block_id(leaf_iter->origin());
      march_narrow_band(tile, interval_id, block_id, stage, tile_thresholds,
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

  // if there is a single block per interval than this while
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
                            this->update_grid, interval_id);
#ifdef LOG_FULL
      cout << "Integrated " << stage << " in "
           << timer.elapsed() - integrate_start << " sec.\n";
#endif
    }

  } // iterations per interval

  active_intervals[interval_id] = false;

#ifdef LOG_FULL
  cout << "Interval: " << interval_id << " in " << inner_iteration_idx
       << " iterations, total " << timer.elapsed() << " sec." << '\n';
#endif
  return timer.elapsed();
}

#ifdef USE_TINYTIFF

/*
 * The interval size and shape define the requested "view" of the image
 * an image view is referred to as a tile
 * that we will load at one time. There is a one to one mapping
 * of an image view and an interval. There is also a one to
 * one mapping between each voxel of the image view and the
 * vertex of the interval. Note the interval is an array of
 * initialized unvisited structs so they start off at arbitrary
 * location but are defined as they are visited.
 */
template <class image_t>
vto::Dense<uint16_t, vto::LayoutXYZ>
Recut<image_t>::load_tile(const VID_t interval_id, const std::string &dir) {
#ifdef LOG
  auto timer = high_resolution_timer();
#endif

  auto tile_extents = this->interval_lengths;
  auto tile_offsets = id_interval_to_img_offsets(interval_id);
  auto interval_max = (tile_offsets + tile_extents).offsetBy(-1);
  const auto bbox = CoordBBox(tile_offsets, interval_max);
#ifdef LOG
  cout << "From image, on ii " << interval_id << " requesting:\n";
  print_coord(tile_offsets, "tile_offsets");
  print_coord(tile_extents, "tile_extents");
  cout << bbox << '\n';
#endif

  const auto tif_filenames = get_dir_files(dir, ".tif"); // sorted
  auto interval_filenames = tif_filenames | rng::views::drop(tile_offsets[2]) |
                            rng::views::take(tile_extents[2]) | rng::to_vector;
  // bbox is inclusive, therefore substract 1
  // const auto dims = get_tif_dims(interval_filenames);
  // assertm(dims == bbox.dim(), "dims and bbox dims must match");

  // try {
  // returns row-major (c-order) buffers
  auto dense = read_tiff_planes(interval_filenames, bbox);
#ifdef LOG
  cout << "Load image in " << timer.elapsed() << " sec." << '\n';
#endif
  ////} catch (...) {
  //// throw std::runtime_error(
  ////"error in mcp3d_tile io. neuron tracing not performed");
  ////}
  return dense;
}

// Calculate new tile thresholds or use input thresholds according
// to params and args this function has no sideffects outside
// of the returned tile_thresholds struct
template <class image_t>
TileThresholds<image_t> *Recut<image_t>::get_tile_thresholds(
    vto::Dense<uint16_t, vto::LayoutXYZ> &tile) {
  auto tile_thresholds = new TileThresholds<image_t>();

  auto interval_dims = tile.bbox().dim();
  auto interval_vertex_size = coord_prod_accum(interval_dims);

  // assign thresholding value
  // foreground parameter takes priority
  // Note if either foreground or background percent is equal to or greater
  // than 0 than it was changed by a user so it takes precedence over the
  // defaults
  if (params->foreground_percent() >= 0) {
    auto timer = high_resolution_timer();
    // TopPercentile takes a fraction 0 -> 1, not a percentile
    tile_thresholds->bkg_thresh =
        bkg_threshold<image_t>(tile.data(), interval_vertex_size,
                               (params->foreground_percent()) / 100);

#ifdef LOG
    cout << "Requested foreground percent: " << params->foreground_percent()
         << " yielded background threshold: " << tile_thresholds->bkg_thresh
         << " in " << timer.elapsed() << " s\n";
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
      tile_thresholds->get_max_min(tile.data(), interval_vertex_size);
#ifdef LOG_FULL
      cout << "max_int: " << +(tile_thresholds->max_int)
           << " min_int: " << +(tile_thresholds->min_int) << '\n';
      cout << "bkg_thresh value = " << +(tile_thresholds->bkg_thresh) << '\n';
      cout << "interval dims " << interval_dims << '\n';
#endif
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
        tile_thresholds->get_max_min(tile.data(), interval_vertex_size);
#ifdef LOG_FULL
        cout << "max_int: " << +(tile_thresholds->max_int)
             << " min_int: " << +(tile_thresholds->min_int) << '\n';
        cout << "bkg_thresh value = " << +(tile_thresholds->bkg_thresh) << '\n';
        cout << "interval dims " << interval_dims << '\n';
#endif
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

  return tile_thresholds;
} // end load_tile()

#endif // USE_TINYTIFF

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
  auto timer = high_resolution_timer();

  // note openvdb::initialize() must have been called before this point
  // otherwise seg faults will occur
  auto update_accessor = this->update_grid->getAccessor();

  // multi-grids for convert stage
  // assertm(this->topology_grid, "topology grid not initialized");
  std::vector<EnlargedPointDataGrid::Ptr> grids(this->grid_interval_size);

  auto histogram = Histogram<image_t>();

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

        if (stage == "convert") {
#ifdef USE_TINYTIFF
          auto dense_tile = load_tile(interval_id, args->image_root_dir());
          if (!local_tile_thresholds) {
            auto thresh_start = timer.elapsed();
            local_tile_thresholds = get_tile_thresholds(dense_tile);
            computation_time =
                computation_time + (timer.elapsed() - thresh_start);
          }
          tile = dense_tile.data();
#else
          if (!(this->params->force_regenerate_image)) {
            assertm(this->input_is_vdb,
                    "If USE_TINYTIFF macro is not set, "
                    "input must either by VDB or "
                    "this->params->force_regenerate_image must be set to True");
          }
#endif

          assertm(!this->input_is_vdb,
                  "input can't be vdb during convert stage");

          GridCoord no_offsets = {0, 0, 0};
          auto interval_offsets = id_interval_to_img_offsets(interval_id);

          GridCoord buffer_offsets =
              params->force_regenerate_image ? interval_offsets : no_offsets;
          GridCoord buffer_extents = params->force_regenerate_image
                                         ? this->image_lengths
                                         : this->interval_lengths;

          auto convert_start = timer.elapsed();

#ifdef FULL_PRINT
          // print_image_3D(tile, buffer_extents);
#endif

          if (this->args->type_ == "uint8") {
            convert_buffer_to_vdb_acc(
                tile, buffer_extents,
                /*buffer_offsets=*/buffer_offsets,
                /*image_offsets=*/interval_offsets,
                this->img_grid->getAccessor(), this->args->type_,
                local_tile_thresholds->bkg_thresh, this->params->upsample_z_);
            if (params->histogram_) {
              histogram += hist(tile, buffer_extents, buffer_offsets);
            }
          } else if (this->args->type_ == "float") {
            convert_buffer_to_vdb_acc(
                tile, buffer_extents,
                /*buffer_offsets=*/buffer_offsets,
                /*image_offsets=*/interval_offsets,
                this->input_grid->getAccessor(), this->args->type_,
                local_tile_thresholds->bkg_thresh, this->params->upsample_z_);
            if (params->histogram_) {
              histogram += hist(tile, buffer_extents, buffer_offsets);
            }
          } else if (this->args->type_ == "mask") {
            convert_buffer_to_vdb_acc(
                tile, buffer_extents,
                /*buffer_offsets=*/buffer_offsets,
                /*image_offsets=*/interval_offsets,
                this->mask_grid->getAccessor(), this->args->type_,
                local_tile_thresholds->bkg_thresh, this->params->upsample_z_);
            if (params->histogram_) {
              histogram += hist(tile, buffer_extents, buffer_offsets);
            }
          } else { // point

            std::vector<PositionT> positions;
            // use the last bkg_thresh calculated for metadata,
            // bkg_thresh is constant for each interval unless a specific % is
            // input by command line user
            convert_buffer_to_vdb(tile, buffer_extents,
                                  /*buffer_offsets=*/buffer_offsets,
                                  /*image_offsets=*/interval_offsets, positions,
                                  local_tile_thresholds->bkg_thresh,
                                  this->params->upsample_z_);

            grids[interval_id] = create_point_grid(
                positions, this->image_lengths, get_transform(),
                this->params->foreground_percent());

#ifdef FULL_PRINT
            print_vdb_mask(grids[interval_id]->getConstAccessor(),
                           this->image_lengths);
#endif
          }
          computation_time =
              computation_time + (timer.elapsed() - convert_start);

          active_intervals[interval_id] = false;
#ifdef LOG
          cout << "Completed traversal of interval " << interval_id + 1
               << " of " << grid_interval_size << " in "
               << timer.elapsed() - convert_start << " s\n";
#endif
        } else {
          computation_time =
              computation_time + process_interval(interval_id, tile, stage,
                                                  local_tile_thresholds,
                                                  update_accessor);
        }
      } // if the interval is active

    } // end one interval traversal
  }   // finished all intervals

  if (stage == "convert") {
    auto finalize_start = timer.elapsed();

    if (args->type_ == "point") {

      assertm(params->convert_only_,
              "reduce grids only possible for convert_only stage");
      for (int i = 0; i < (this->grid_interval_size - 1); ++i) {
        // vb::tools::compActiveLeafVoxels(grids[i]->tree(), grids[i +
        // 1]->tree());
        if (leaves_intersect(grids[i + 1], grids[i])) {
          throw std::runtime_error(
              "Leaves intersect, can cause undefined behavior\n");
        }
        // leaves grids[i] empty, copies all to grids[i+1]
        grids[i + 1]->tree().merge(grids[i]->tree(),
                                   vb::MERGE_ACTIVE_STATES_AND_NODES);
      }
      this->topology_grid = grids[this->grid_interval_size - 1];

      set_grid_meta(this->topology_grid, this->image_lengths,
                    params->foreground_percent());
      this->topology_grid->tree().prune();

    } else {

      set_grid_meta(this->input_grid, this->image_lengths,
                    params->foreground_percent());

      if (this->args->type_ == "float") {
        this->input_grid->tree().prune();
      } else if (this->args->type_ == "uint8") {
        this->img_grid->tree().prune();
      } else if (this->args->type_ == "mask") {
        this->mask_grid->tree().prune();
      }

      if (params->histogram_) {
        auto write_to_file = [](auto out, std::string fn) {
          std::ofstream file;
          file.open(fn);
          file << out;
          file.close();
        };

        write_to_file(histogram, "hist.txt");
      }
    }

    auto finalize_time = timer.elapsed() - finalize_start;
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

  auto total_update_time = timer.elapsed();
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

template <class image_t>
void Recut<image_t>::initialize_globals(const VID_t &grid_interval_size,
                                        const VID_t &interval_block_size) {
  this->active_intervals = vector(grid_interval_size, false);

  auto timer = high_resolution_timer();

  if (!params->convert_only_) {

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
    VID_t interval_id = 0;

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
  }

#ifdef LOG
  cout << "Active leaf count: " << this->update_grid->tree().leafCount()
       << '\n';
#endif
#ifdef LOG_FULL
  cout << "\tCreated fifos " << timer.elapsed() << 's' << '\n';
#endif
}

// Deduce lengths from the various input options
template <class image_t>
GridCoord Recut<image_t>::get_input_image_lengths(bool force_regenerate_image,
                                                  RecutCommandLineArgs *args) {
  GridCoord input_image_lengths = zeros();
  this->update_grid = openvdb::BoolGrid::create();
  if (this->params->force_regenerate_image) {
    // for generated image runs trust the args->image_lengths
    // to reflect the total global image domain
    // see get_args() in utils.hpp
    input_image_lengths = args->image_lengths;

    // FIXME placeholder grid
    this->topology_grid = create_vdb_grid(input_image_lengths,
                                          this->params->foreground_percent());
    append_attributes(this->topology_grid);
  } else if (this->input_is_vdb) { // running based of a vdb input

    assertm(!params->convert_only_,
            "Convert only option is not valid from vdb to vdb");

#ifndef USE_VDB
    assertm(false, "Input must either be regenerated, vdb or from image, "
                   "USE_VDB must be defined");
#endif

    auto timer = high_resolution_timer();
    auto base_grid = read_vdb_file(args->image_root_dir());

#ifdef LOG
    cout << "VDB input type " << this->args->type_ << '\n';
#endif

    if (this->args->type_ == "float") {
      this->input_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
      // copy topology (bit-mask actives) to the topology grid
      // this->topology_grid =
      // copy_to_point_grid(this->input_grid, input_image_lengths,
      // this->params->foreground_percent());
      this->topology_grid = convert_float_to_point(this->input_grid);
      // cout << "float count " << this->input_grid->activeVoxelCount()
      //<< " point count "
      //<< openvdb::points::pointCount(this->topology_grid->tree()) << '\n';
      // assertm(this->input_grid->activeVoxelCount() ==
      // openvdb::points::pointCount(this->topology_grid->tree()),
      //"did no match");
      auto [lengths, _] = get_metadata(input_grid);
      input_image_lengths = lengths;
    } else if (this->args->type_ == "point") {
      this->topology_grid =
          openvdb::gridPtrCast<EnlargedPointDataGrid>(base_grid);
      auto [lengths, _] = get_metadata(topology_grid);
      input_image_lengths = lengths;

      // you need to use grid if you are outputing windows
      if (!params->output_windows_.empty()) {
        auto raw_grid = read_vdb_file(params->output_windows_);
        this->img_grid = openvdb::gridPtrCast<ImgGrid>(raw_grid);
      }
    }
    append_attributes(this->topology_grid);

#ifdef LOG
    cout << "Read grid in: " << timer.elapsed() << " s\n";
#endif

  } else { // converting to a new grid
#ifndef USE_TINYTIFF
    assertm(false, "Input must either be regenerated, use VDB or have the "
                   "USE_TINYTIFF macro for image reading");
#else
    const auto tif_filenames = get_dir_files(args->image_root_dir(), ".tif");
    input_image_lengths = get_tif_dims(tif_filenames);
#endif

    if (args->type_ == "float") {
      this->input_grid = openvdb::FloatGrid::create();
    } else if (args->type_ == "point") {
      this->topology_grid = create_vdb_grid(input_image_lengths,
                                            this->params->foreground_percent());
      append_attributes(this->topology_grid);
    } else if (args->type_ == "uint8") {
      this->img_grid = ImgGrid::create();
    } else if (args->type_ == "mask") {
      this->mask_grid = openvdb::MaskGrid::create();
    }
  }
  return input_image_lengths;
}

template <class image_t>
std::vector<std::pair<GridCoord, uint8_t>> Recut<image_t>::initialize() {

#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
  omp_set_num_threads(params->user_thread_count());
#ifdef LOG
  cout << "User specific thread count " << params->user_thread_count() << '\n';
  cout << "User specified image root dir " << args->image_root_dir() << '\n';
#endif
#endif
  struct timespec time2, time3;
  uint64_t root_64bit;

  // input type
  {
    auto path_extension =
        std::string(fs::path(args->image_root_dir()).extension());
    this->input_is_vdb = path_extension == ".vdb" ? true : false;
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
  this->image_bbox = openvdb::math::CoordBBox(
      this->image_offsets, this->image_offsets + this->image_lengths);

  // TODO move this clipping up to the read step
  if (!this->params->convert_only_) {
    this->topology_grid->clip(this->image_bbox);
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
    // images are saved in separate z-planes, so conversion should respect
    // that for best performance constrict so less data is allocated
    // especially in z dimension
    this->interval_lengths[0] = this->image_lengths[0];
    this->interval_lengths[1] = this->image_lengths[1];
    // explicitly set by get_args
    if (params->interval_length) {
      this->interval_lengths[2] = params->interval_length;
    } else {
      this->interval_lengths[2] = LEAF_LENGTH;
      // auto recommended_max_mem = GetAvailMem() / 16;
      // guess how many z-depth tiles will fit before a bad_alloc is likely
      // auto simultaneous_tiles =
      // static_cast<double>(recommended_max_mem) /
      //(sizeof(image_t) * this->image_lengths[0] * this->image_lengths[1]);
      // assertm(simultaneous_tiles >= 1, "Tile x and y size too large to fit
      // in system memory (DRAM)");
      // this->interval_lengths[2] = std::min(
      // simultaneous_tiles, static_cast<double>(this->image_lengths[2]));
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

  auto timer = high_resolution_timer();
  initialize_globals(grid_interval_size, interval_block_size);

#ifdef LOG
  cout << "Initialized globals " << timer.elapsed() << '\n';
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

    // add the single root vid to the roots
    return {
        std::pair{id_to_coord(this->params->root_vid, this->image_lengths), 1}};

  } else {
    if (params->convert_only_) {
      return {};
    } else {
      // adds all valid markers to roots vector and returns
      return process_marker_dir(this->image_offsets, this->image_lengths);
    }
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
};

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

template <class image_t> void Recut<image_t>::print_to_swc() {

  auto coord_to_swc_id = get_id_map();

  auto to_swc = [this, &coord_to_swc_id](
                    const auto &flags_handle, const auto &parents_handle,
                    const auto &radius_handle, const auto &ind, auto leaf) {
    auto coord = ind.getCoord();
    print_swc_line(coord, is_root(flags_handle, ind), radius_handle.get(*ind),
                   parents_handle.get(*ind), this->image_bbox, this->out,
                   /*map*/ coord_to_swc_id, /*adjust*/ true);
  };

  this->out.open(this->args->swc_path());
  this->out << "#id type_id x y z radius parent_id\n";

  visit(this->topology_grid, keep_root, to_swc);
  visit(this->topology_grid, not_root, to_swc);

  if (this->out.is_open())
    this->out.close();
#ifdef LOG
  cout << "Wrote output to " << this->args->swc_path() << '\n';
#endif
}

template <class image_t> void Recut<image_t>::prune_branch() {
  auto filter_branch = [](const auto &flags_handle, const auto &parents_handle,
                          const auto &radius_handle, const auto &ind) {
    auto parents = parents_handle.get(*ind);
    return is_valid(flags_handle, ind) && !is_root(flags_handle, ind) &&
           ((parents[0] + parents[1] + parents[2]) < MIN_LENGTH);
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
void Recut<image_t>::partition_components(
    std::vector<std::pair<GridCoord, uint8_t>> root_pair, bool prune) {

  openvdb::GridPtrVec grids;
#ifdef LOG
  print_point_count(this->topology_grid);
#endif

  // this copies only vertices that have already had flags marked as selected
  // selected means they are reachable from a known vertex during traversal
  // in either a connected or value stage
  auto float_grid = copy_selected(this->topology_grid);

  cout << "Float active count: " << float_grid->activeVoxelCount() << '\n';
  assertm(float_grid->activeVoxelCount(),
          "active voxels in float grid must be > 0");

  // aggregate disjoint connected components
  auto timer = high_resolution_timer();
  std::vector<openvdb::FloatGrid::Ptr> components;
  vto::segmentActiveVoxels(*float_grid, components);
  cout << "Segment count: " << components.size() << " in " << timer.elapsed()
       << " s\n";

  auto output_topology = false;

  auto counter = 0;

  rng::for_each(components, [this, &prune, &counter, float_grid,
                             output_topology,
                             &root_pair](const auto component) {
    // all grid transforms across are consistent across recut, so enforce the
    // same interpretation for any new grid
    component->setTransform(get_transform());

    // filter all roots within this component
    auto component_roots =
        root_pair |
        rng::views::remove_if([&component](const auto &coord_radius) {
          auto [coord, radius] = coord_radius;
          return !component->tree().isValueOn(coord);
        }) |
        rng::to_vector;

    if (component_roots.size() < 1)
      return; // skip

    if (component_roots.size() > 1) {
      cout << "Skipping component with more than 1 root\n";
      return; // skip
    }

    // FIXME delete this once performance improves
    auto voxel_count = component->activeVoxelCount();

    auto timer = high_resolution_timer();
    auto markers = convert_float_to_markers(component, this->topology_grid);
    // auto markers = convert_to_markers(this->topology_grid, false);
#ifdef LOG
    // cout << "Convert to markers in " << timer.elapsed() << '\n';
#endif

#ifdef LOG
    auto dir = "./component-" + std::to_string(counter);
    fs::remove_all(dir); // make sure it's an overwrite
    fs::create_directories(dir);
    auto name = dir + "/component-" + std::to_string(counter) + ".swc";
    // cout << name << " active count " << component->activeVoxelCount() << '
    // '
    //<< component->evalActiveVoxelBoundingBox() << '\n';
    // cout << "Marker count: " << markers.size() << '\n';
#endif

    timer.restart();
    auto pruned_markers = advantra_prune(markers, this->args->prune_radius_);
#ifdef LOG
    // cout << "Prune markers to size " << pruned_markers.size() << " in "
    //<< timer.elapsed() << '\n';
#endif

    timer.restart();
    // extract a new tree via bfs
    auto tree = advantra_extract_trees(pruned_markers, true);
#ifdef LOG
    // cout << "Extract trees to size: " << tree.size() << " in "
    //<< timer.elapsed() << '\n';
#endif

    auto filtered_tree = remove_short_leafs(tree);

    // start swc and add header metadata
    std::ofstream file;
    file.open(name);
    file << "# Component bounding volume: "
         << component->evalActiveVoxelBoundingBox() << '\n';
    file << "# id type_id x y z radius parent_id\n";

    // start a new blank map for coord to a unique swc id
    auto coord_to_swc_id = get_id_map();
    // iter those marker*
    rng::for_each(filtered_tree, [this, &file, &coord_to_swc_id,
                                  &component](const auto marker) {
      auto coord = GridCoord(marker->x, marker->y, marker->z);

      auto parent_coord =
          GridCoord(marker->parent->x, marker->parent->y, marker->parent->z);
      auto parent_offset = coord_sub(parent_coord, coord);
      // print_swc_line() expects an offset to a parent
      print_swc_line(coord,
                     /*is_root*/ marker->type == 0, marker->radius,
                     parent_offset, component->evalActiveVoxelBoundingBox(),
                     file, coord_to_swc_id,
                     /*bbox_adjust*/ !params->output_windows_.empty());
    });

    if (!params->output_windows_.empty()) {
      auto component_with_values =
          write_output_windows(this->img_grid, component, dir, counter);
      assertm(component_with_values->evalActiveVoxelBoundingBox() ==
                  component->evalActiveVoxelBoundingBox(),
              "transfered component have mismatched sizes");
      if (true) { // check against app2
        auto window = convert_vdb_to_dense(component_with_values);
        assertm(component_with_values->evalActiveVoxelBoundingBox() ==
                    window.bbox(),
                "converted component have mismatched sizes");

        // get a per window bkg_thresh, max, min
        auto tile_thresholds = get_tile_thresholds(window);

        auto component_markers =
            component_roots | rng::views::transform([](auto &coord_radius) {
              auto [coord, radius] = coord_radius;
              auto marker =
                  new MyMarker(static_cast<double>(coord.x()),
                               static_cast<double>(coord.y()),
                               static_cast<double>(coord.z()), radius);
              marker->type = 0; // mark as a root
              return marker;
            }) |
            rng::to_vector;

        rng::for_each(component_markers, [&dir](const auto marker) {
          // write marker file
          std::ofstream marker_file;
          auto mass = ((4 * PI) / 3.) * pow(marker->radius, 3);
          marker_file.open(dir + "/marker_" +
                           std::to_string(static_cast<int>(marker->x)) + "_" +
                           std::to_string(static_cast<int>(marker->y)) + "_" +
                           std::to_string(static_cast<int>(marker->z)) + "_" +
                           std::to_string(int(mass)));

          marker_file << "# x,y,z in original image\n";
          marker_file << marker->x << ',' << marker->y << ',' << marker->z
                      << '\n';
        });

        // start time
        auto timer = high_resolution_timer();

        // adjust component_markers to match window, just for
        // fastmarching_tree()
        rng::for_each(component_markers, [&window](auto &marker) {
          // subtracts the offset so that app2 is with respect to this window
          // for fastmarching_tree() and happ()
          adjust_marker(marker, -window.bbox().min());
        });

        // reconstruct
        std::vector<MyMarker *> app2_output_tree;
        std::vector<MyMarker> targets;
        fastmarching_tree(component_markers, targets, window.data(),
                          app2_output_tree, window.bbox().dim()[0],
                          window.bbox().dim()[1], window.bbox().dim()[2],
                          /* cnn_type*/ 1, tile_thresholds->bkg_thresh,
                          tile_thresholds->max_int, tile_thresholds->min_int);

        // prune run the seq prune from app2 to compare
        std::vector<MyMarker *> app2_output_tree_prune;
        happ(app2_output_tree, app2_output_tree_prune, window.data(),
             window.bbox().dim()[0], window.bbox().dim()[1],
             window.bbox().dim()[2], tile_thresholds->bkg_thresh,
             /*length thresh*/ 3.0, /*sr_ratio*/ 1. / 3);

        // adjust app2_output_tree_prune to match global image, for swc output
        if (false) {
          rng::for_each(app2_output_tree_prune, [&window](const auto marker) {
            // adds the offset so the swc is with respect to whole image
            adjust_marker(marker, window.bbox().min());
          });
        }

        // print
        auto app2_fn =
            dir + "/app2-component-" + std::to_string(counter) + ".swc";
        marker_to_swc_file(app2_fn, app2_output_tree_prune);

        cout << "Run APP2 in " << timer.elapsed() << '\n';
      }
    }

    ++counter;
  }); // for each component

  if (output_topology) {
    grids.push_back(this->topology_grid);
    write_vdb_file(grids, "final-point-grid.vdb");
  }
}

template <class image_t> void Recut<image_t>::convert_topology() {
  activate_all_intervals();

  // mutates input_grid
  auto stage = "convert";
  this->update(stage, map_fifo);

  openvdb::GridPtrVec grids;

  if (args->type_ == "float") {
    print_grid_metadata(this->input_grid);
    grids.push_back(this->input_grid);
  } else if (args->type_ == "uint8") {
    print_grid_metadata(this->img_grid);
    grids.push_back(this->img_grid);
  } else if (args->type_ == "mask_grid") {
    print_grid_metadata(this->mask_grid);
    grids.push_back(this->mask_grid);
  } else if (args->type_ == "point") {
    print_grid_metadata(this->topology_grid);
    grids.push_back(this->topology_grid);
  }

  write_vdb_file(grids, this->params->out_vdb_);
}

template <class image_t> void Recut<image_t>::operator()() {
  if (!params->second_grid_.empty()) {
    combine_grids(args->image_root_dir(), params->second_grid_,
                  this->params->out_vdb_);
    return;
  }

  // read the list of root vids
  auto root_pair = this->initialize();

  if (params->convert_only_) {
    convert_topology();
    // no more work to do, exiting
    return;
  }

  // constrain topology to only those reachable from roots
  auto stage = "connected";
  {
    // starting from the roots connected stage saves all surface vertices into
    // fifo
    this->activate_vids(this->topology_grid, root_pair, stage, this->map_fifo,
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
    adjust_soma_radii(root_pair, this->topology_grid);
  }

  partition_components(root_pair, false);

  // old prune strategy
  //{
  //// starting from roots, prune stage will
  //// create final list of vertices
  // if (true) {
  // stage = "prune";
  // this->activate_vids(this->topology_grid, root_pair, stage,
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
