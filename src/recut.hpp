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
#include <bitset>
#include <cstdlib>
#include <deque>

#include <fstream>
#include <future>
#include <iostream>
#include <map>
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
  // subscripts: come at the end of variable name to aid discoverability and
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
  // A 3D image has a dimension of image_length_x, image_length_y,
  // image_length_z. Therefore image_size = image_length_x * image_length_y *
  // image_length_z. If the program were keeping track of multiple images then
  // the variable image_count would record that number
  VID_t image_size, image_length_x, image_length_y, image_length_z,
      image_length_xy, interval_block_len_x, interval_block_len_y,
      interval_block_len_z, user_def_block_size, pad_block_length_x,
      pad_block_length_y, pad_block_length_z, pad_block_offset, block_length_x,
      block_length_y, block_length_z, interval_length_x, interval_length_y,
      interval_length_z, grid_interval_length_x, grid_interval_length_y,
      grid_interval_length_z, grid_interval_size, interval_block_size;

  std::vector<int> image_extents;
  std::ofstream out;
  image_t *generated_image = nullptr;
  bool mmap_;
  bool input_is_vdb;
#ifdef USE_VDB
  openvdb::v8_0::TopologyGrid::Ptr topology_grid;
#endif
  size_t iteration;
  float bound_band;
  float stride;
  double restart_factor;
  bool restart_bool;
  atomic<VID_t> global_revisits;
  VID_t vertex_issue; // default heuristic per thread for roughly best
  // performance
  RecutCommandLineArgs *args;
  RecutParameters *params;
  std::vector<std::vector<std::deque<VertexAttr>>> global_fifo;
  std::vector<std::vector<std::deque<VertexAttr>>> local_fifo;
  std::vector<std::vector<std::vector<VertexAttr>>> active_vertices;

  // interval specific global data structures
  vector<bool> active_intervals;
  vector<vector<atomwrapper<bool>>> active_blocks;
  vector<vector<atomwrapper<bool>>> processing_blocks;
  vector<vector<local_heap>> heap_vec;
  vector<vector<vector<bool>>> active_neighbors;
#ifdef CONCURRENT_MAP
  // runtime global data structures
  std::unique_ptr<ConcurrentMap64> updated_ghost_vec;
#else
  vector<vector<vector<vector<VertexAttr>>>> updated_ghost_vec;
#endif

  Recut() : mmap_(false){};
  Recut(RecutCommandLineArgs &args)
      : args(&args), params(&(args.recut_parameters())), global_revisits(0),
        user_def_block_size(args.recut_parameters().block_size()),
        mmap_(false) {

#ifdef USE_MMAP
    this->mmap_ = true;
#endif
    this->restart_bool = params->restart();
    this->restart_factor = params->restart_factor();
  }

  void run_pipeline();

  void operator()() { run_pipeline(); }

  // to destroy the information for this run
  // so that it doesn't affect the next run
  // the vertices must be unmapped
  // done via `release()`
#ifdef DENSE
  Grid grid;
  inline void release() { grid.Release(); }
  inline VertexAttr *get_vertex_vid(VID_t interval_id, VID_t block_id,
                                    VID_t vid, VID_t *output_offset);
  template <typename vertex_t>
  void brute_force_extract(vector<vertex_t> &outtree, bool accept_band = false,
                           bool release_intervals = true);
#endif
  void initialize_globals(const VID_t &grid_interval_size,
                          const VID_t &interval_block_size);

  bool filter_by_vid(VID_t vid, VID_t find_interval_id, VID_t find_block_id);
  bool filter_by_label(VertexAttr *v, bool accept_band);
  // template<typename T, typename T2>
  // void safe_increase(T &heap, T2* node) ;
  template <typename T, typename T2, typename TNew>
  void safe_update(T &heap, T2 *node, TNew new_field, std::string cmp_field);
  template <typename T, typename T2>
  void safe_push(T &heap, T2 *node, VID_t interval_id, VID_t block_id,
                 std::string cmp_field);
  void adjust_parent(bool print);
  template <typename T, typename T2>
  T2 safe_pop(T &heap, VID_t block_id, VID_t interval_id,
              std::string cmp_field);

  image_t get_img_val(const image_t *tile, VID_t vid);
  template <typename T> bool get_vdb_val(T accessor, VID_t vid);
  inline VID_t rotate_index(VID_t img_sub, const VID_t current,
                            const VID_t neighbor,
                            const VID_t interval_block_size,
                            const VID_t pad_block_size);
  int get_bkg_threshold(const image_t *tile, VID_t interval_vertex_size,
                        const double foreground_percent);
  inline VertexAttr *get_or_set_active_vertex(VID_t interval_id, VID_t block_id,
                                              VID_t vid, bool &found);
  inline VertexAttr *get_active_vertex(VID_t interval_id, VID_t block_id,
                                       VID_t vid);
  void place_vertex(const VID_t nb_interval_id, VID_t block_id, VID_t nb,
                    struct VertexAttr *dst, std::string stage);
  bool check_blocks_finish(VID_t interval_id);
  bool are_intervals_finished();
  void activate_all_intervals();
  inline VID_t get_img_vid(VID_t i, VID_t j, VID_t k);
  inline VID_t get_interval_id_vert_sub(const VID_t i, const VID_t j,
                                        const VID_t k);
  inline VID_t get_interval_id(const VID_t i, const VID_t j, const VID_t k);
  void get_interval_offsets(const VID_t interval_id,
                            vector<int> &interval_offsets,
                            vector<int> &interval_extents);
  void get_interval_subscript(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline void get_img_subscript(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline void get_block_subscript(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline VID_t get_block_id(const VID_t id);
  VID_t get_interval_id(VID_t vid);
  VID_t get_sub_to_interval_id(VID_t i, VID_t j, VID_t k);
  void check_ghost_update(VID_t interval_id, VID_t block_id,
                          struct VertexAttr *dst, std::string stage);
  int get_parent_code(VID_t dst_id, VID_t src_id);
  template <typename T>
  bool accumulate_connected(const image_t *tile, VID_t interval_id,
                            VID_t dst_id, VID_t block_id, VID_t current_vid,
                            VID_t &revisits,
                            const TileThresholds<image_t> *tile_thresholds,
                            bool &found_background, T vdb_accessor);
  bool accumulate_value(const image_t *tile, VID_t interval_id, VID_t dst_id,
                        VID_t block_id, struct VertexAttr *current,
                        VID_t &revisits,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_background);
  bool is_covered_by_parent(VID_t interval_id, VID_t block_id,
                            VertexAttr *current);
  template <class Container, typename T, typename T2>
  void accumulate_prune(VID_t interval_id, VID_t dst_id, VID_t block_id,
                        T current_parent, T2 current_vid,
                        bool current_unvisited,
                        struct VertexAttr &found_higher_parent, bool &covered,
                        bool enqueue_dsts, const bool dst_outside_domain,
                        Container &fifo);
  template <class Container, typename T>
  void accumulate_radius(VID_t interval_id, VID_t dst_id, VID_t block_id,
                         T current_radius, VID_t &revisits, int stride,
                         int pad_stride, Container &fifo);
  template <class Container, typename T>
  void update_neighbors(const image_t *tile, VID_t interval_id, VID_t block_id,
                        VertexAttr *current, VID_t current_vid, VID_t &revisits,
                        std::string stage,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_adjacent_invalid,
                        VertexAttr &found_higher_parent, Container &fifo,
                        bool &covered, T vdb_accessor,
                        bool current_outside_domain = false,
                        bool enqueue_dsts = false);
  template <class Container>
  void integrate_updated_ghost(const VID_t interval_id, const VID_t block_id,
                               std::string stage, Container &fifo);
  template <class Container> void dump_buffer(Container buffer);
  void adjust_vertex_parent(VertexAttr *vertex);
  template <class Container>
  bool integrate_vertex(const VID_t interval_id, const VID_t block_id,
                        struct VertexAttr *updated_vertex,
                        bool ignore_KNOWN_NEW, std::string stage,
                        Container &fifo);
  template <class Container, typename T>
  void march_narrow_band(const image_t *tile, VID_t interval_id, VID_t block_id,
                         std::string stage,
                         const TileThresholds<image_t> *tile_thresholds,
                         Container &fifo, T vdb_accessor);
  template <class Container, typename T>
  void connected_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                      std::string stage,
                      const TileThresholds<image_t> *tile_thresholds,
                      Container &fifo, VID_t revisits, T vdb_accessor);
  template <class Container, typename T>
  void value_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                  std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T vdb_accessor);
  template <class Container, typename T>
  void radius_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                   std::string stage,
                   const TileThresholds<image_t> *tile_thresholds,
                   Container &fifo, VID_t revisits, T vdb_accessor);
  template <class Container, typename T>
  void prune_tile_assign_parent(const image_t *tile, VID_t interval_id,
                                VID_t block_id, std::string stage,
                                const TileThresholds<image_t> *tile_thresholds,
                                Container &fifo, VID_t revisits,
                                T vdb_accessor);
  template <class Container, typename T>
  void prune_tile(const image_t *tile, VID_t interval_id, VID_t block_id,
                  std::string stage,
                  const TileThresholds<image_t> *tile_thresholds,
                  Container &fifo, VID_t revisits, T vdb_accessor);
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
  const std::vector<VID_t> initialize();
  template <typename vertex_t>
  void convert_to_markers(vector<vertex_t> &outtree, bool accept_band = false);
  VID_t parentToVID(struct VertexAttr *vertex);
  inline VID_t get_block_id(VID_t iblock, VID_t jblock, VID_t kblock);
  template <class Container>
  void print_interval(VID_t interval_id, std::string stage, Container &fifo);
  template <class Container>
  void print_grid(std::string stage, Container &fifo);
  template <class Container> void setup_radius(Container &fifo);
  template <class Container>
  void activate_vids(const std::vector<VID_t> &root_vids, std::string stage,
                     Container &fifo);
  std::vector<VID_t> process_marker_dir(vector<int> global_image_offsets,
                                        vector<int> global_image_extents);
  void print_vertex(VertexAttr *current);
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

/*
 * Does this subscript belong in the full image
 * accounting for the input offsets and extents
 * i, j, k : subscript of vertex in question
 * off : offsets in z y x order
 * end : sanitized end pixels order
 */
template <typename T, typename T2>
bool check_in_bounds(T i, T j, T k, T2 off, T2 end) {
  if (i < off[2])
    return false;
  if (j < off[1])
    return false;
  if (k < off[0])
    return false;
  if (i > off[2] + end[2])
    return false;
  if (j > off[1] + end[1])
    return false;
  if (k > off[0] + end[0])
    return false;
  return true;
}

// adds all markers to root_vids
//
template <class image_t>
std::vector<VID_t>
Recut<image_t>::process_marker_dir(vector<int> global_image_offsets,
                                   vector<int> global_image_extents) {
  vector<VID_t> root_vids;

  if (params->marker_file_path().empty())
    return root_vids;

  // allow either dir or dir/ naming styles
  if (params->marker_file_path().back() != '/')
    params->set_marker_file_path(params->marker_file_path().append("/"));

  cout << "marker dir path: " << params->marker_file_path() << '\n';
  assertm(fs::exists(params->marker_file_path()),
          "Marker file path must exist");

  vector<MyMarker> inmarkers;
  for (const auto &marker_file :
       fs::directory_iterator(params->marker_file_path())) {
    const auto marker_name = marker_file.path().filename().string();
    const auto full_marker_name = params->marker_file_path() + marker_name;
    inmarkers = readMarker_file(full_marker_name);

    // set intervals with root present as active
    for (auto &root : inmarkers) {
      // init state and phi for root
      VID_t i, j, k, x, y, z;
      x = root.x + 1;
      y = root.y + 1;
      z = root.z + 1;

      if (!(check_in_bounds(x, y, z, global_image_offsets,
                            global_image_extents)))
        continue;
      // adjust the vid according to the region of the image we are processing
      i = x - global_image_offsets[2];
      j = y - global_image_offsets[1];
      k = z - global_image_offsets[0];
      auto vid = get_img_vid(i, j, k);
      root_vids.push_back(vid);

#ifdef LOG_FULL
      cout << "Read marker at x " << x << " y " << y << " z " << z
           << " adjust to subscripts, i " << i << " j " << j << " k " << k
           << " vid " << vid << '\n';
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
  for (const auto &vid : root_vids) {
    auto interval_id = get_interval_id(vid);
    auto block_id = get_block_id(vid);

#ifdef FULL_PRINT
    cout << "activate_vids(): setting interval " << interval_id << " block "
         << block_id << " to active ";
    cout << "for marker vid " << vid << '\n';
#endif

    active_intervals[interval_id] = true;
    active_blocks[interval_id][block_id].store(true);

    VertexAttr *msg_vertex;
    if (stage == "connected") {
      // place a root with proper vid and parent of itself
      msg_vertex = &(this->active_vertices[interval_id][block_id].emplace_back(
          0, 0., vid, vid));
      this->local_fifo[interval_id][block_id].emplace_back(0, 0., vid, vid);
    } else if (stage == "prune") {

#ifdef DENSE
      msg_vertex = get_vertex_vid(interval_id, block_id, vid, nullptr);
#else
      msg_vertex = get_active_vertex(interval_id, block_id, vid);
#endif
      cout << "root radius " << +(msg_vertex->radius) << ' ' << interval_id
           << ' ' << block_id << ' ' << msg_vertex->vid << '\n';
      assertm(msg_vertex->valid_radius(),
              "activate vids didn't find valid radius");
      assertm(msg_vertex != nullptr, "get_active_vertex yielded nullptr");
      fifo[interval_id][block_id].emplace_back(0, 0., vid, vid);
    } else {
      assertm(false, "stage behavior not yet specified");
    }

    // place ghost update accounts for
    // edges of intervals in addition to blocks
    // this only adds to update_ghost_vec if the root happens
    // to be on a boundary
    check_ghost_update(interval_id, block_id, msg_vertex,
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
  VID_t i, j, k, xadjust, yadjust, zadjust;
  get_interval_subscript(interval_id, i, j, k);
  xadjust = i * interval_length_x;
  yadjust = j * interval_length_y;
  zadjust = k * interval_length_z;
  for (int zi = zadjust; zi < zadjust + interval_length_z; zi++) {
    cout << "Z=" << zi << '\n';
    cout << "  | ";
    for (int xi = xadjust; xi < xadjust + interval_length_x; xi++) {
      cout << xi << " ";
    }
    cout << '\n';
    for (int xi = 0; xi < 2 * interval_length_y + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = yadjust; yi < yadjust + interval_length_y; yi++) {
      cout << yi << " | ";
      for (int xi = xadjust; xi < xadjust + interval_length_x; xi++) {
        VID_t vid = ((VID_t)xi) + yi * image_length_x + zi * image_length_xy;
        auto block_id = get_block_id(vid);
        get_img_subscript(vid, i, j, k);
        assertm(xi == i, "i mismatch");
        assertm(yi == j, "j mismatch");
        assertm(zi == k, "k mismatch");
#ifdef FULL_PRINT
        // std::cout << "\nvid " << vid << " block " << block_id << " xi " <<
        // xi
        // << " yi " << yi << " zi " << zi << '\n';
#endif

#ifdef DENSE
        auto v = get_vertex_vid(interval_id, block_id, vid, nullptr);
#else
        auto v = get_active_vertex(interval_id, block_id, vid);
        if (v == nullptr) {
          cout << "- ";
          continue;
        }
#endif

        if (v->root()) {
          assertm(v->value == 0., "root value must be 0.");
          assertm(v->valid_value(), "root must have valid value");
        }
        // cout << +inimg1d[vid] << " ";
        // auto v = interval->GetData()[vid];
        // cout << i << '\n' << v.description() << " ";
        if (stage == "value") {
          if (v->valid_value()) {
            cout << v->value << " ";
          } else {
            cout << "- ";
          }
        } else if (stage == "radius") {
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
          // for now just print the first block of the
          // interval
          auto surface = fifo[block_id];
          bool match = false;
          for (auto check_v : surface) {
            if (v->vid == check_v.vid) {
              match = true;
            }
          }
          if (match) {
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

/* Note: this assumes all values are below 2<<16 - 1
 * otherwise they will overflow when cast to uint16 for
 * packing. Ordered such that one is placed at higher
 * bits than two
 */
uint32_t double_pack_key(const VID_t one, const VID_t two) {
  uint32_t final = (uint32_t)two;
  final |= (uint32_t)one << 16;
  return final;
}

/* Note: this assumes all values are below 2<<16 - 1
 * otherwise they will overflow when cast to uint16 for
 * packing. ordered such that one is placed at the highest
 * bit location, left to right while ascending.
 */
uint64_t triple_pack_key(const VID_t one, const VID_t two, const VID_t three) {
  uint64_t final = (uint64_t)three;
  final |= (uint64_t)two << 16;
  final |= (uint64_t)one << 32;
  return final;
}

template <typename T, typename T2> T absdiff(const T &lhs, const T2 &rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

template <typename T, typename T2, typename T3>
T3 min(const T &lhs, const T2 &rhs) {
  return lhs < rhs ? lhs : rhs;
}

/**
 * vid : linear idx relative to unpadded image
 * returns interval_id, the interval domain this vertex belongs to
 * with respect to the original unpadded image
 * does not consider overlap of ghost regions
 * uses get_sub_to_interval_id() to calculate
 */
template <class image_t> VID_t Recut<image_t>::get_interval_id(VID_t vid) {
  // get the subscripts
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  // cout << "i " << i << " j " << j << " k " << k << '\n';
  return get_sub_to_interval_id(i, j, k);
}

/**
 * i, j, k : subscripts of vertex relative to entire image
 * returns interval_id, the interval domain this vertex belongs to
 * does not consider overlap of ghost regions
 */
template <class image_t>
VID_t Recut<image_t>::get_sub_to_interval_id(VID_t i, VID_t j, VID_t k) {
  auto i_interval = i / interval_length_x;
  auto j_interval = j / interval_length_y;
  auto k_interval = k / interval_length_z;
  auto interval_id =
      i_interval + j_interval * grid_interval_length_x +
      k_interval * grid_interval_length_x * grid_interval_length_y;
  // cout << "i_interval " << i_interval << " j_interval " << j_interval
  //<< " k_interval " << k_interval << '\n';
  assert(interval_id < grid_interval_size);
  return interval_id;
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
template <class image_t> VID_t Recut<image_t>::get_block_id(const VID_t vid) {
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  // subtract away the interval influence on the block num
  auto ia = i % interval_length_x;
  auto ja = j % interval_length_y;
  auto ka = k % interval_length_z;
  // block in this interval
  auto i_block = ia / block_length_x;
  auto j_block = ja / block_length_y;
  auto k_block = ka / block_length_z;
  return i_block + j_block * interval_block_len_x +
         k_block * interval_block_len_x * interval_block_len_y;
}

/** safely removes minimum element from the heap
 * heap
 */
template <class image_t>
template <typename T, typename T2>
T2 Recut<image_t>::safe_pop(T &heap, VID_t block_id, VID_t interval_id,
                            std::string cmp_field) {
#ifdef HLOG_FULL
  // cout << "safe_pop ";
  assert(!(heap.empty()));
#endif
  T2 noderef = heap.top();
#ifdef HLOG_FULL
  // cout << "vid: " << noderef->vid ;
  assert(noderef->valid_handle());
#endif

  T2 node2 = heap.pop(block_id, cmp_field); // remove it

#ifdef HLOG_FULL
  assert(!(node2->valid_handle()));
  // cout << " handle removed" << " heap size: " << heap.size() << '\n';
#endif
  return node2;
}

// assign handle save to original vertex not the heap copy
template <class image_t>
template <typename T_heap, typename T2>
void Recut<image_t>::safe_push(T_heap &heap, T2 *node, VID_t interval_id,
                               VID_t block_id, std::string cmp_field) {
#ifdef HLOG_FULL
  cout << "safe push heap size " << heap.size() << " vid: " << node->vid
       << '\n'; //" handle: " << node->handles[block_id] ;
  assert(!(node->valid_handle()));
  assertm(current->valid_vid(), "must push a valid_vid vertex");
#endif
  heap.push(node, block_id, cmp_field);
#ifdef HLOG_FULL
  assert(node->valid_handle());
#endif
}

// assign handle save to original vertex not the heap copy
template <class image_t>
template <typename T, typename T2, typename TNew>
void Recut<image_t>::safe_update(T &heap, T2 *node, TNew new_field,
                                 std::string cmp_field) {
#ifdef HLOG_FULL
  assert(node->valid_handle());
  // cout << "safe update size " << heap.size() <<
  //" vid: " << node->vid << '\n';
#endif
  heap.update(node, 0, new_field,
              cmp_field); // block_id currently deprecated
}

/* get the interval linear idx from it's subscripts
 * not all linear indices are row-ordered and specific
 * to their hierarchical arrangment
 */
template <class image_t>
inline VID_t Recut<image_t>::get_interval_id(const VID_t i, const VID_t j,
                                             const VID_t k) {
  return k * (grid_interval_length_x * grid_interval_length_y) +
         j * grid_interval_length_x + i;
}

/* get the vid with respect to the entire image passed to the
 * recut program. Note this spans multiple tiles and blocks
 * Take the subscripts of the vertex or voxel
 * returns the linear idx into the entire domain
 */
template <class image_t>
inline VID_t Recut<image_t>::get_interval_id_vert_sub(const VID_t i,
                                                      const VID_t j,
                                                      const VID_t k) {
  return k * (interval_length_x * interval_length_y) + j * interval_length_x +
         i;
}

template <class image_t>
template <typename T>
bool Recut<image_t>::get_vdb_val(T vdb_accessor, VID_t vid) {
  VID_t x, y, z;
  x = y = z = 0;
  get_img_subscript(vid, x, y, z);
  openvdb::Coord xyz(x, y, z);
#ifdef FULL_PRINT
  cout << " x " << x << " y " << y << " z " << z << '\n';
#endif
  return vdb_accessor.getValue(xyz);
}

/*
 * Takes a global image vid
 * and converts it to linear index of current
 * tile of interval currently processed
 * returning the voxel value at that
 * location id -> sub_script i, j, k -> mod i, j, k against interval_length in
 * each dim -> convert modded subscripts to a new vid
 */
template <class image_t>
image_t Recut<image_t>::get_img_val(const image_t *tile, VID_t vid) {
  // force_regenerate_image passes the whole image as the
  // tile so the img vid is the correct address regardless
  // of interval length sizes Note that force_regenerate_image
  // is mostly used in test cases to try different scenarios
  if (this->params->force_regenerate_image) {
    return tile[vid];
  }
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  // mod out any contributions from the interval
  auto ia = i % interval_length_x;
  auto ja = j % interval_length_y;
  auto ka = k % interval_length_z;
  auto interval_vid = get_interval_id_vert_sub(ia, ja, ka);
#ifdef FULL_PRINT
  // cout<< "\ti: "<<i<<" j: "<<j <<" k: "<< k<< " dst vid: " << vid << '\n';
  // cout<< "\n\tia: "<<ia<<" ja: "<<ja <<" ka: "<< ka<< " interval vid: " <<
  // interval_vid << '\n';
#endif
  return tile[interval_vid];
}

template <typename image_t>
bool Recut<image_t>::is_covered_by_parent(VID_t interval_id, VID_t block_id,
                                          VertexAttr *current) {
  VID_t i, j, k, pi, pj, pk;
  get_img_subscript(current->vid, i, j, k);
  get_img_subscript(current->parent, pi, pj, pk);
  auto x = static_cast<double>(i) - pi;
  auto y = static_cast<double>(j) - pj;
  auto z = static_cast<double>(k) - pk;
  auto vdistance = sqrt(x * x + y * y + z * z);

  auto parent_interval_id = get_interval_id(current->parent);
  auto parent_block_id = get_block_id(current->parent);

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

template <class image_t>
template <class Container, typename T, typename T2>
void Recut<image_t>::accumulate_prune(VID_t interval_id, VID_t dst_id,
                                      VID_t block_id, T current_parent,
                                      T2 current_vid, bool current_unvisited,
                                      struct VertexAttr &found_higher_parent,
                                      bool &covered, bool enqueue_dsts,
                                      const bool dst_outside_domain,
                                      Container &fifo) {

#ifdef DENSE
  auto dst = get_vertex_vid(interval_id, block_id, dst_id, nullptr);
#else
  auto dst = get_active_vertex(interval_id, block_id, dst_id);
#endif
  if (dst == nullptr) { // never selected
    return;
  }

  // or if unselected but radius is valid meaning it's been pruned
  if (dst->unselected() && dst->prune_visited()) {
    assertm(dst->valid_radius(), "should always have a valid radius?");
  }
  if (dst->band()    // in the FIFO awaiting visit
      || dst->root() // a root already processed
      // or a previously pruned vertex which transmits transitive
      // coverage info from it's covered:
      || (dst->unselected() && dst->prune_visited() && dst->valid_radius())) {
    assertm(dst->valid_vid(), "selected must have a valid vid");
    assertm(dst->vid == dst_id,
            "get_active_vertex failed getting correct vertex");

    // dst can only be 1 hop away (adjacent) from current, therefore
    // all radii greater than 1 imply some redundancy in coverage
    // but this may be desired with DILATION_FACTOR higher than 1
    if (dst->radius >= DILATION_FACTOR) {
      covered = true;
      if (found_higher_parent.radius < (dst->radius - 1)) {
        found_higher_parent = *dst;
        // std::cout << "found higher " << dst->description() << '\n';
      }
      // end
#ifdef FULL_PRINT
      std::cout << "  radius of: " << +(dst->radius) << " at dst " << dst_id
                << " " << dst->label() << " covers current;";
#endif
    }
  }

  // after all neighbors have been checked, it's safe to know
  // what the final parent will will be, so you can safely pass
  // along the right parent to dst's by set_parent
  // so dst itself can be used to pass messages
  // beyond just vid to other blocks / intervals
  // and keep a contiguous path to a root
  if (enqueue_dsts) {
    if ((dst->selected() && !(dst->prune_visited())) && !dst_outside_domain) {
#ifdef FULL_PRINT
      std::cout << "  added dst " << dst_id << " rad " << +(dst->radius)
                << '\n';
#endif
      if (covered || current_unvisited) {
        dst->set_parent(current_parent);
      } else {
        dst->set_parent(current_vid);
      }
      dst->prune_visit();
      assertm(dst->valid_vid(), "selected must have a valid vid");
      assertm(dst->valid_radius(), "selected must have a valid radius");
      fifo.push_back(*dst);
      check_ghost_update(interval_id, block_id, dst, "prune");
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
 * integrate_updated_ghost(). dst_id : continuous vertex id VID_t of the dst
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <class Container, typename T>
void Recut<image_t>::accumulate_radius(VID_t interval_id, VID_t dst_id,
                                       VID_t block_id, T current_radius,
                                       VID_t &revisits, int stride,
                                       int pad_stride, Container &fifo) {

  // note the current vertex can belong in the boundary
  // region of a separate block /interval and is only
  // within this block /interval's ghost region
  // therefore all neighbors / destinations of current
  // must be checked to make sure they protude into
  // the actual current block / interval region
  // current vertex is not always within this block and interval
  // and each block, interval have a ghost region
  // after filter in update_neighbors this pointer arithmetic is always valid
#ifdef DENSE
  auto dst = get_vertex_vid(interval_id, block_id, dst_id, nullptr);
#else
  auto dst = get_active_vertex(interval_id, block_id, dst_id);
  if (dst == nullptr) {
    return;
  }
#endif

  uint8_t updated_radius = 1 + current_radius;
  if (dst->selected() || dst->root()) {
    assertm(dst->valid_vid(), "selected must have a valid vid");
    assertm(dst->vid == dst_id,
            "get_active_vertex failed getting correct vertex");

    // if radius not set yet it necessitates it is 1 higher than current OR an
    // update from another block / interval creates new lower updates
    if (!(dst->valid_radius()) || (dst->radius > updated_radius)) {
      dst->radius = updated_radius;
      fifo.push_back(*dst);
      check_ghost_update(interval_id, block_id, dst, "radius");

#ifdef FULL_PRINT
      cout << "\tAdjacent higher dst->vid " << dst->vid << " radius "
           << +(dst->radius) << "current radius " << +(current_radius) << '\n';
      if (dst->root()) {
        cout << "root radius " << +(dst->radius) << ' ' << interval_id << ' '
             << block_id << ' ' << dst->vid << '\n';
      }
#endif
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
 * integrate_updated_ghost(). dst_id : continuous vertex id VID_t of the dst
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
template <typename T>
bool Recut<image_t>::accumulate_connected(
    const image_t *tile, VID_t interval_id, VID_t dst_id, VID_t block_id,
    VID_t current_vid, VID_t &revisits,
    const TileThresholds<image_t> *tile_thresholds, bool &found_background,
    T vdb_accessor) {

#ifdef FULL_PRINT
  cout << "\tcheck dst vid: " << dst_id;
  cout << " bkg_thresh " << +(tile_thresholds->bkg_thresh) << '\n';
#endif

  // skip backgrounds
  // the image voxel of this dst vertex is the primary method to exclude this
  // pixel/vertex for the remainder of all processing
  if (this->input_is_vdb) {
    if (!get_vdb_val(vdb_accessor, dst_id))
      found_background = true;
  } else {
    auto dst_vox = get_img_val(tile, dst_id);
    if (dst_vox <= tile_thresholds->bkg_thresh) {
      found_background = true;
    }
  }

  if (found_background) {
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
  auto dst = get_or_set_active_vertex(interval_id, block_id, dst_id, found);
#endif

  // skip already selected vertices too
  if (found) {
    revisits += 1;
    return false;
  }

#ifdef FULL_PRINT
  cout << "\tadded new dst to active set, vid: " << dst_id << '\n';
#endif

  // ensure traces a path back to root in case no prune stage
  // parent will likely be mutated during prune stage
  dst->set_parent(current_vid);
  assertm(dst->valid_vid(), "selected must have a valid vid");
  local_fifo[interval_id][block_id].push_back(*dst);
  check_ghost_update(interval_id, block_id, dst, "connected");

  return true;
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current interval_id and block_id, since
 * it is potentially adding these vertices to the unique heap of interval_id
 * and block_id. only one parent when selected. If one of these vertexes on
 * the edge but still within interval_id and block_id domain is updated it
 * is the responsibility of check_ghost_update to take note of the update such
 * that this update is propagated to the relevant interval and block see
 * integrate_updated_ghost(). dst_id : continuous vertex id VID_t of the dst
 * vertex in question block_id : current block id current : minimum vertex
 * attribute selected
 */
template <class image_t>
bool Recut<image_t>::accumulate_value(
    const image_t *tile, VID_t interval_id, VID_t dst_id, VID_t block_id,
    struct VertexAttr *current, VID_t &revisits,
    const TileThresholds<image_t> *tile_thresholds, bool &found_background) {

  assertm(false, "deprecated");
  // dsts are always within this domain
#ifdef DENSE
  auto dst = get_vertex_vid(interval_id, block_id, dst_id, nullptr);
#else
  auto dst = get_active_vertex(interval_id, block_id, dst_id);
#endif

  auto dst_vox = get_img_val(tile, dst_id);

#ifdef FULL_PRINT
  cout << "\tcheck dst vid: " << dst_id;
  // cout << " addr: " << static_cast<void *>(dst);
  // cout << " val " << dst->value;
  cout << " pixel " << dst_vox;
  cout << " bkg_thresh " << +(tile_thresholds->bkg_thresh) << '\n';
#endif

  // skip backgrounds
  // the image voxel of this dst vertex is the primary method to exclude this
  // pixel/vertex for the remainder of all processing
  if (dst_vox <= tile_thresholds->bkg_thresh) {
#ifdef FULL_PRINT
    cout << "\t\tfailed tile_thresholds->bkg_thresh" << '\n';
#endif
    found_background = true;
    return false;
  }

  // solve for update value
  // dst_id and current->vid are linear idx relative to full image domain
  float new_field = static_cast<float>(
      current->value +
      (tile_thresholds->calc_weight(get_img_val(tile, current->vid)) +
       tile_thresholds->calc_weight(dst_vox)) *
          0.5);

  if (dst == nullptr) {
    // this->active_vertices[interval_id][block_id].emplace_back(dst_id);
    // dst = &(this->active_vertices[interval_id][block_id].back());
    dst = new VertexAttr(dst_id);
  }

  // this automatically excludes any root vertex since they have a
  // value of 0.
  if (dst->value > new_field) {

    if (dst->selected()) {
      revisits += 1;
#ifdef NO_RV
      return false;
#endif
    }

    { // update this vertex
      // add to band (01XX XXXX)
      dst->mark_band(dst_id);

      // traces a path back to root
      // parent will likely be mutated during prune stage
      dst->set_parent(current->vid);
      assertm(dst->valid_vid(), "selected must have a valid vid");

      // if a visited node doesn't have a vid it will cause
      // undefined behavior
      if (dst->valid_handle()) { // in block_id heap
        safe_update(heap_vec[interval_id][block_id], dst, new_field,
                    "value"); // increase priority, lowers value in min-heap
#ifdef FULL_PRINT
        cout << "\t\tupdate: change in heap for BAND value" << '\n';
#endif
      } else {
        dst->value = new_field;
        safe_push(heap_vec[interval_id][block_id], dst, interval_id, block_id,
                  "value");
#ifdef FULL_PRINT
        cout << "\t\tupdate: add to heap: " << dst->vid
             << " value: " << dst->value << '\n';
#endif
      }
      check_ghost_update(interval_id, block_id, dst, "value");

      assertm(dst->value == new_field,
              "Accumulate value did not properly set it's updated field");
      // all dsts are guaranteed within this domain
      this->active_vertices[interval_id][block_id].push_back(*dst);
    }
    return true;
  } else {
#ifdef FULL_PRINT
    cout << "\t\tfailed: no value change: " << dst->vid
         << " value: " << dst->value << '\n';
#endif
  }
  return false; // no update
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
                                  struct VertexAttr *dst, std::string stage) {

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
#ifdef CONCURRENT_MAP
  auto key = triple_pack_key(nb_interval_id, block_id, nb_block_id);
  auto mutator = updated_ghost_vec->insertOrFind(key);
  std::vector<struct VertexAttr> *vec = mutator.getValue();
  if (!vec) {
    vec = new std::vector<struct VertexAttr>;
  }
  vec->emplace_back(dst->edge_state, dst->value, dst->vid, dst->radius,
                    dst->parent);
  // Note: this block is unique to a single thread, therefore no other
  // thread could have this same key, since the keys are unique with
  // respect to their permutation. This means that we do not need to
  // protect from two threads modifying the same key simultaneously
  // in this design. If this did need to protected from see documentation
  // at preshing.com/20160201/new-concurrent-hash-maps-for-cpp/ for details
  mutator.assignValue(vec); // assign via mutator vs. relookup
#else
  if (stage == "prune") {
    assertm(dst->valid_radius(), "selected must have a valid radius");
  }
  // when nb_interval_id != interval_id, the block_id is technically
  // incorrect, surprisingly this makes no difference for the correctness
  // since the block_id origination dimension is merely to prevent
  // race conditions
  updated_ghost_vec[nb_interval_id][block_id][nb_block_id].emplace_back(
      dst->edge_state, dst->value, dst->vid, dst->radius, dst->parent);
#endif
  active_neighbors[nb_interval_id][nb_block_id][block_id] = true;

#ifdef FULL_PRINT
  VID_t i, j, k;
  get_block_subscript(nb_block_id, i, j, k);
  cout << "\t\t\tghost update stage " << stage << " interval " << nb_interval_id
       << " nb block " << nb_block_id << " block_id " << block_id << " block i "
       << i << " block j " << j << " block k " << k << " vid " << dst->vid
       << '\n';
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
 * Warning: heap_vec store pointers to the same
 * underlying VertexAttr data as in the interval, therefore out of order /
 * race condition changes are not protected against
 */
template <class image_t>
void Recut<image_t>::check_ghost_update(VID_t interval_id, VID_t block_id,
                                        struct VertexAttr *dst,
                                        std::string stage) {
  VID_t i, j, k, ii, jj, kk, iii, jjj, kkk;
  vector<VID_t> interval_subs = {0, 0, 0};
  i = j = k = ii = jj = kk = 0;
  VID_t id = dst->vid;
  get_img_subscript(id, i, j, k);
  ii = i % block_length_x;
  jj = j % block_length_y;
  kk = k % block_length_z;
  iii = i % interval_length_x;
  jjj = j % interval_length_y;
  kkk = k % interval_length_z;

  // check all 6 directions for possible ghost updates
  VID_t nb; // determine neighbor block
  VID_t nb_interval_id;
  VID_t iblock, jblock, kblock;
  get_block_subscript(block_id, iblock, jblock, kblock);

#ifdef FULL_PRINT
  cout << "\t\t\tcheck_ghost_update on vid " << dst->vid << " current block "
       << block_id << " block i " << iblock << " block j " << jblock
       << " block k " << kblock << '\n';
#endif

  // check all six sides
  if (ii == 0) {
    if (i > 0) { // protect from image out of bounds
      nb = block_id - 1;
      nb_interval_id = interval_id; // defaults to current interval
      if (iii == 0) {
        nb_interval_id = interval_id - 1;
        // Convert block subscripts into linear index row-ordered
        nb = get_block_id(interval_block_len_x - 1, jblock, kblock);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, stage);
    }
  }
  if (jj == 0) {
    if (j > 0) { // protect from image out of bounds
      nb = block_id - interval_block_len_x;
      nb_interval_id = interval_id; // defaults to current interval
      if (jjj == 0) {
        nb_interval_id = interval_id - grid_interval_length_x;
        nb = get_block_id(iblock, interval_block_len_y - 1, kblock);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, stage);
    }
  }
  if (kk == 0) {
    if (k > 0) { // protect from image out of bounds
      nb = block_id - interval_block_len_x * interval_block_len_y;
      nb_interval_id = interval_id; // defaults to current interval
      if (kkk == 0) {
        nb_interval_id =
            interval_id - grid_interval_length_x * grid_interval_length_y;
        nb = get_block_id(iblock, jblock, interval_block_len_z - 1);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, stage);
    }
  }

  if (kk == block_length_x - 1) {
    if (k < image_length_z - 1) { // protect from image out of bounds
      nb = block_id + interval_block_len_x * interval_block_len_y;
      nb_interval_id = interval_id; // defaults to current interval
      if (kkk == interval_length_z - 1) {
        nb_interval_id =
            interval_id + grid_interval_length_x * grid_interval_length_y;
        nb = get_block_id(iblock, jblock, 0);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, stage);
    }
  }
  if (jj == block_length_y - 1) {
    if (j < image_length_y - 1) { // protect from image out of bounds
      nb = block_id + interval_block_len_x;
      nb_interval_id = interval_id; // defaults to current interval
      if (jjj == interval_length_y - 1) {
        nb_interval_id = interval_id + grid_interval_length_x;
        nb = get_block_id(iblock, 0, kblock);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, stage);
    }
  }
  if (ii == block_length_z - 1) {
    if (i < image_length_x - 1) { // protect from image out of bounds
      nb = block_id + 1;
      nb_interval_id = interval_id; // defaults to current interval
      if (iii == interval_length_x - 1) {
        nb_interval_id = interval_id + 1;
        nb = get_block_id(0, jblock, kblock);
      }
      if ((nb >= 0) && (nb < interval_block_size)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, stage);
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
  if ((adiff % image_length_xy) == 0) {
    if (src_gr)
      return 5;
    else
      return 4;
  } else if ((adiff % image_length_x) == 0) {
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

// check and add dst vertices in star stencil
template <class image_t>
template <class Container, typename T>
void Recut<image_t>::update_neighbors(
    const image_t *tile, VID_t interval_id, VID_t block_id, VertexAttr *current,
    VID_t current_vid, VID_t &revisits, std::string stage,
    const TileThresholds<image_t> *tile_thresholds,
    bool &found_adjacent_invalid, VertexAttr &found_higher_parent,
    Container &fifo, bool &covered, T vdb_accessor, bool current_outside_domain,
    bool enqueue_dsts) {

  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(current->vid, i, j, k);
#ifdef FULL_PRINT
  // all block ids are a linear row-wise idx, relative to current interval
  VID_t block = get_block_id(current->vid);
  cout << "\ni: " << i << " j: " << j << " k: " << k << " stage: " << stage
       << " current vid: " << current->vid;
  if (stage == "value") {
    std::cout << " value: " << current->value;
  }
  if (stage == "radius" || stage == "prune") {
    std::cout << " radius: " << +(current->radius);
  }
  std::cout << " addr: " << static_cast<void *>(current) << " interval "
            << interval_id << " block " << block_id << " label "
            << current->label() << '\n';
  //" for block " << block_id << " within domain of block " << block << '\n';
#endif

  if (stage == "radius") {
    {
      if (!(current->valid_radius())) {
        cout << current->description();
        dump_buffer(this->active_vertices);
        assertm(current->valid_radius(),
                "radius must march outward from known radii");
      }
    }
  }

  // only supports +-1 in x, y, z
  VID_t dst_id;
  int x, y, z;
  bool dst_outside_domain;
  int stride, pad_stride;
  int z_stride = this->block_length_x * this->block_length_y;
  int z_pad_stride = this->pad_block_length_x * this->pad_block_length_y;

  // save current params before it possibly becomes undefined by iterator
  // invalidation
  auto current_radius = current->radius;
  auto current_unvisited = current->unvisited();
  auto current_parent = current->parent;

  for (int kk = -1; kk <= 1; kk++) {
    z = ((int)k) + kk;
    if (z < 0 || z >= image_length_z) {
      found_adjacent_invalid = true;
      continue;
    }
    for (int jj = -1; jj <= 1; jj++) {
      y = ((int)j) + jj;
      if (y < 0 || y >= image_length_y) {
        found_adjacent_invalid = true;
        continue;
      }
      for (int ii = -1; ii <= 1; ii++) {
        x = ((int)i) + ii;
        if (x < 0 || x >= image_length_x) {
          found_adjacent_invalid = true;
          continue;
        }
        int offset = abs(ii) + abs(jj) + abs(kk);
        // this ensures a start stencil,
        // exclude current, exclude diagonals
        if (offset == 0 || offset > 1) {
          continue;
        }
        dst_id = get_img_vid(x, y, z);

        // all block_nums and interval_nums are a linear
        // row-wise idx, relative to current interval
        auto dst_block_id = get_block_id(dst_id);
        auto dst_interval_id = get_sub_to_interval_id(x, y, z);

        // Filter all dsts that don't protude into current
        // block and interval region, ghost destinations
        // can not be added in to processing stack
        // ghost vertices can only be added in to the stack
        // during `integrate_updated_ghost()`
        dst_outside_domain =
            ((dst_interval_id != interval_id) || (dst_block_id != block_id))
                ? true
                : false;

        // only prune considers currents outside of the domain
        // prune uses these ghost regions (+-1) to pass messages
        // for example a root can be in the ghost region
        // and the adjacent vertex in this block needs to know it's
        // radius in order to be properly pruned
        if (stage == "prune") {
          // if current is already outside this block
          // never take a dst unless it projects back into the current
          // block and interval, otherwise you will go outside of the data race
          // safe data region
          if (current_outside_domain) {
            if (dst_outside_domain)
              continue;
          }
        } else {
          if (dst_outside_domain)
            continue;
        }

        // note although this is summed only one of ii,jj,kk
        // will be not equal to 0
        stride = ii + jj * this->block_length_x + kk * z_stride;
        pad_stride = ii + jj * this->pad_block_length_x + kk * z_pad_stride;
#ifdef FULL_PRINT
        cout << "x " << x << " y " << y << " z " << z << " dst_id " << dst_id
             << " vid " << current->vid << '\n';
#endif
        if (stage == "connected") {
          accumulate_connected(tile, interval_id, dst_id, block_id, current_vid,
                               revisits, tile_thresholds,
                               found_adjacent_invalid, vdb_accessor);
        } else if (stage == "radius") {
          accumulate_radius(interval_id, dst_id, block_id, current_radius,
                            revisits, stride, pad_stride, fifo);
        } else if (stage == "prune") {
          accumulate_prune(interval_id, dst_id, block_id, current_parent,
                           current_vid, current_unvisited, found_higher_parent,
                           covered, enqueue_dsts, dst_outside_domain, fifo);
        } else if (stage == "value") {
          accumulate_value(tile, interval_id, dst_id, block_id, current,
                           revisits, tile_thresholds, found_adjacent_invalid);
        }
      }
    }
  }
} // end update_neighbors()

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
      if (dst->valid_vid()) { // "dst must have a valid vid"
        fifo.push_back(*dst);
        // deep copy into the shared memory location in the separate block
        *dst = *updated_vertex;
        assertm(dst->radius == updated_vertex->radius,
                "radius did not get copied");
        return true;
      }
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
      assertm(dst->valid_vid(), "dst must have a valid vid");
      return true;
    }
    return false;
  }

  if (stage == "value") {
#ifdef NO_RV
    if (ignore_KNOWN_NEW) {
      if (dst->selected()) { // 10XX XXXX KNOWN NEW
        return false;
      }
    }
#endif

    // These are collected from nb, so only when nb updates are lower do we
    // need to update. However equal values are included so neighboring
    // blocks can successfully activated even when the update is
    // redundant or had correct ordering from the thread processing.
    // Note: although these updates are >= the solution will still
    // converge because update neighbors requires an update be strictly
    // less than, thus points with no update on a border will not be
    // added back and forth continously.
    // Only updates in the ghost region outside domain of block_id, in domain
    // of nb, therefore the updates must go into the heapvec of nb to
    // distinguish the internal handles used for either heaps handles[block_id]
    // is set for cells within block_id blocks internal domain or block_id ghost
    // cell region, it represents all cells added to heap block_id and all cells
    // that can be manipulated by thread block_id
    if ((dst->value > updated_vertex->value) ||
        (updated_vertex->surface() && !(dst->surface()))) {
      float old_val = dst->value;
      // a dst can only be marked as surface
      // it can never be reversed from surface to not
      dst->copy_edge_state(*updated_vertex);
      if (updated_vertex->root()) {
        dst->mark_root(updated_vertex->vid); // mark permanently as root
#ifdef FULL_PRINT
        cout << "\tfrom integrate_vertex(), mark root: " << dst->vid << '\n';
#endif
      } else {
        dst->mark_band(
            updated_vertex->vid); // ensure band in case of race condition from
        // threads during accumulate()
      }
      assert(dst->valid_vid());
      // if heap_vec does not contain this dst already
      if (dst->valid_handle()) { // already in the heap
        // FIXME adapt this to handle radius as well
        safe_update(heap_vec[interval_id][block_id], dst, updated_vertex->value,
                    stage); // increase priority, lowers value in min-heap
      } else {
        dst->value = updated_vertex->value;
        safe_push(heap_vec[interval_id][block_id], dst, interval_id, block_id,
                  stage);
      }
      return true;
    }
#ifdef FULL_PRINT
    cout << "\tfailed: no value change in heap" << '\n';
#endif
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
#ifdef CONCURRENT_MAP
      auto key = triple_pack_key(interval_id, nb, block_id);
      // get mutator so that doesn't have to reloaded when assigning
      auto mutator = updated_ghost_vec->find(key);
      std::vector<struct VertexAttr> *vec = mutator.getValue();
      // nullptr if not found, continuing saves a redundant mutator assign
      if (!vec) {
        active_neighbors[interval_id][nb][block_id] = false; // reset to false
        continue; // FIXME this should never occur because of active_neighbor
        // right?
      }
      assertm(false, "not implemented");

      for (struct VertexAttr updated_vertex : *vec) {
#else

      // updated_ghost_vec[x][a][b] in domain of a, in ghost of block b
      for (VertexAttr updated_vertex :
           updated_ghost_vec[interval_id][nb][block_id]) {
#endif

#ifdef FULL_PRINT
        cout << "integrate vid: " << updated_vertex.vid
             << " ghost of block id: " << block_id
             << " in block domain of block id: " << nb << " at interval "
             << interval_id << '\n';
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
#ifdef CONCURRENT_MAP
      vec->clear();
      // assign directly to the mutator to save a lookup
      mutator.assignValue(
          vec); // keep the same pointer just make sure it's empty
#else
      updated_ghost_vec[interval_id][nb][block_id].clear();
#endif
    } // for all active neighbor blocks of block_id
  }
  if (stage == "value") {
    if (!(heap_vec[interval_id][block_id].empty())) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_id << " block: " << block_id
           << " to active\n";
#endif
      active_blocks[interval_id][block_id].store(true);
    }
  } else if (stage == "connected") {
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

template <typename image_t>
void Recut<image_t>::print_vertex(VertexAttr *current) {
  // n,type,x,y,z,radius,parent
  // std::cout << current->description() << '\n';
  VID_t i, j, k;
  get_img_subscript(current->vid, i, j, k);
  auto parent_vid = std::string{"-"};
  if (current->valid_parent()) { // parent set properly
    parent_vid = std::to_string(current->parent);
  }
  if (current->root()) {
    parent_vid = "-1";
  }
  if (this->out.is_open()) {
#pragma omp critical
    this->out << current->vid << " " << i << " " << j << " " << k << " "
              << +(current->radius) << " " << parent_vid << '\n';
  } else {
#pragma omp critical
    std::cout << current->vid << " " << i << " " << j << " " << k << " "
              << +(current->radius) << " " << parent_vid << '\n';
  }
}

// if the parent has been pruned then set the current
// parent further upstream,
template <typename image_t>
void Recut<image_t>::adjust_vertex_parent(VertexAttr *vertex) {
  VertexAttr *parent = vertex;
  do {
    auto parent_vid = parent->parent;
    // get actual interval and block
    const auto block_id = get_block_id(parent_vid);
    const auto interval_id = get_interval_id(parent_vid);

    parent = get_active_vertex(interval_id, block_id, parent_vid);
    assertm(parent != nullptr, "parent must be active");

  } while (parent->unvisited());
  vertex->parent = parent->vid;
}

template <class image_t>
template <class Container>
void Recut<image_t>::dump_buffer(Container buffer) {
  std::cout << "\n\nDump buffer\n";
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      for (auto &v : buffer[interval_id][block_id]) {
        std::cout << v.description();
        // print_vertex(&v);
      }
    }
  }
  std::cout << "\n\nFinished buffer dump\n";
}

template <class image_t>
template <class Container, typename T>
void Recut<image_t>::connected_tile(
    const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds, Container &fifo,
    VID_t revisits, T vdb_accessor) {

  VertexAttr *current, *msg_vertex;
  VID_t visited = 0;
  while (!(local_fifo[interval_id][block_id].empty())) {

#ifdef LOG_FULL
    visited += 1;
#endif

    // msg_vertex might become undefined during call to update_neighbors
    msg_vertex = &(local_fifo[interval_id][block_id].front());
    const bool in_domain = msg_vertex->selected() || msg_vertex->root();
    auto vid = msg_vertex->vid;
    auto surface = msg_vertex->surface();

    // invalid can either be out of range of the entire global image or it
    // can be a background vertex which occurs due to pixel value below the
    // threshold, previously selected vertices are considered valid
    auto found_adjacent_invalid = false;
    auto covered = false;
    update_neighbors(tile, interval_id, block_id, msg_vertex, vid, revisits,
                     stage, tile_thresholds, found_adjacent_invalid,
                     *msg_vertex, fifo, covered, vdb_accessor);

    // protect from message values not inside
    // this block or interval such as roots from activate_vids
#ifdef DENSE
    current = get_vertex_vid(interval_id, block_id, vid, nullptr);
#else
    if (in_domain) {
      current = get_active_vertex(interval_id, block_id, vid);
      assertm(current != nullptr, "connected can't be passed a null");
      // msg_vertex aleady aware it is a surface
      if (surface) {
        current->mark_surface();
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
                << " " << current->vid << " label " << current->label() << '\n';
      // std::cout << "check_interval_id " << check_interval_id
      //<< " check_block_id " << check_block_id << '\n';
#endif
      current->mark_surface();

      if (in_domain) {
        // save all surface vertices for the radius stage
        // each fifo corresponds to a specific interval_id and block_id
        // so there are no race conditions
        fifo.push_back(*current);
        assertm(current->selected() || current->root(), "must be selected");

        auto test = get_active_vertex(interval_id, block_id, current->vid);
        if (!(test->surface())) {
          std::cout << current->description();
          std::cout << test->description();
          dump_buffer(this->active_vertices);
        }
        assertm(test->surface(), "recovered vertex wasn't marked as surface");
#ifdef DENSE
        // ghost regions in the DENSE case have shared copies of ghost regions
        // so they would need to know about any changes to vertex state
        // otherwise surface status change is irrelevant to outside domains
        // at this stage
        // goes into updated_ghost_vec of neighbors if its on the edge with them
        // these then get added into neighbor local_fifo
        check_ghost_update(interval_id, block_id, current, stage);
#endif
      } else {
        // a message from an outside leaf is actually a surface and was
        // unaware so send a message directly to that leaf
        const auto check_block_id = get_block_id(current->vid);
        const auto check_interval_id = get_interval_id(current->vid);

        // leverage updated_ghost_vec to avoid race conditions
        // check_ghost_update never applies to vertices outside this leaf
        updated_ghost_vec[check_interval_id][block_id][check_block_id]
            .emplace_back(current->edge_state, current->value, current->vid,
                          current->radius, current->parent);
        active_neighbors[check_interval_id][check_block_id][block_id] = true;
      }
    }
  }
}

template <class image_t>
template <class Container, typename T>
void Recut<image_t>::value_tile(const image_t *tile, VID_t interval_id,
                                VID_t block_id, std::string stage,
                                const TileThresholds<image_t> *tile_thresholds,
                                Container &fifo, VID_t revisits,
                                T vdb_accessor) {
  assertm(false, "deprecated");
  VertexAttr *current;
  VID_t visited = 0;
  while (!(heap_vec[interval_id][block_id].empty())) {

    if (restart_bool) {
      struct VertexAttr *top =
          heap_vec[interval_id][block_id].top(); // peek at top position
      if (top->value > bound_band) {
        break;
      }
    }

#ifdef LOG_FULL
    visited += 1;
#endif

    // remove from this intervals heap
    struct VertexAttr *msg_vertex = safe_pop<local_heap, struct VertexAttr *>(
        heap_vec[interval_id][block_id], block_id, interval_id, stage);

    // protect from message values added to the heap not inside
    // this block or interval roots are also added to the heap in initialize
    // to prevent having to load the full intervals into memory at that time
    // this retrieves the actual ptr to the vid of the root message min
    // values so that they can be properly initialized
#ifdef DENSE
    current = get_vertex_vid(interval_id, block_id, msg_vertex->vid, nullptr);
#else
    current = get_active_vertex(interval_id, block_id, msg_vertex->vid);
#endif
    if (current == nullptr) {
      current = new VertexAttr(msg_vertex->vid);
    }

    // preserve state of roots
    if (msg_vertex->root()) {
      current->value = 0.;
      current->vid = msg_vertex->vid;
      assertm(current->valid_vid(), "must recover a valid_vid vertex");
      // 0000 0000, selected no parent, all zeros
      current->mark_root(msg_vertex->vid);
      assertm(current->root(), "root value not set properly");
      assertm(current->value == 0, "root value not set properly");
    } else {
      current->mark_selected(); // set as KNOWN NEW
      current->vid = msg_vertex->vid;
      assertm(current->valid_vid(), "must recover a valid_vid vertex");
      // Note: any adjustments to current if in a neighboring domain
      // i.e. current->handles.contains(nb) are not in a data race
      // with the nb thread. All VertexAttr have redundant copies in each
      // ghost region
    }

    // invalid can either be out of range of the entire global image or it
    // can be a background vertex which occurs due to pixel value below the
    // threshold
    auto found_adjacent_invalid = false;
    auto covered = false;
    update_neighbors(tile, interval_id, block_id, current, current->vid,
                     revisits, stage, tile_thresholds, found_adjacent_invalid,
                     *current, fifo, covered, vdb_accessor);
    const auto check_block_id = get_block_id(current->vid);
    const auto check_interval_id = get_interval_id(current->vid);
    const bool in_domain =
        (check_block_id == block_id) && (check_interval_id == interval_id);

    // keep all such active vertices within domain
    if (in_domain) {
      // roots would be added twice since they are never added back to the
      // heap to prevent redundant other vertices set NO_RV to on
      this->active_vertices[interval_id][block_id].push_back(*current);
#ifdef DENSE
      current = get_vertex_vid(interval_id, block_id, current->vid, nullptr);
#else
      current = get_active_vertex(interval_id, block_id, current->vid);
#endif
      assertm(current != nullptr, "no current in value_tile");
    }

    // surface related logic
    if (found_adjacent_invalid || current->surface()) {
#ifdef FULL_PRINT
      std::cout << "found surface vertex " << interval_id << " " << block_id
                << " " << current->vid << " label " << current->label() << '\n';
      std::cout << "check_interval_id " << check_interval_id
                << " check_block_id " << check_block_id << '\n';
      if (current->surface()) {
        std::cout << "previously marked as suface\n";
      }
#endif
      current->mark_surface();

      assertm(current->selected() || current->root(),
              "surface was not selected");
      if (in_domain) {
        // save all surface vertices for the radius stage
        // each fifo corresponds to a specific interval_id and block_id
        // so there are no race conditions
        fifo.push_back(*current);
        check_ghost_update(interval_id, block_id, current, stage);
      } else {
        // leverage updated_ghost_vec to avoid race conditions
        // check_ghost_update never applies to vertices outside
        // of this block and interval domain
        updated_ghost_vec[check_interval_id][block_id][check_block_id]
            .emplace_back(current->edge_state, current->value, current->vid,
                          current->radius, current->parent);
        active_neighbors[check_interval_id][check_block_id][block_id] = true;
      }
    }
  }
}

template <class image_t>
template <class Container, typename T>
void Recut<image_t>::radius_tile(const image_t *tile, VID_t interval_id,
                                 VID_t block_id, std::string stage,
                                 const TileThresholds<image_t> *tile_thresholds,
                                 Container &fifo, VID_t revisits,
                                 T vdb_accessor) {
  VertexAttr *current; // for performance
  VID_t visited = 0;
  while (!(fifo.empty())) {
    // msg_vertex will be invalidated during call to update_neighbors
    auto msg_vertex = &(fifo.front());
    fifo.pop_front();

    if (msg_vertex->band()) {
      // current can be from ghost region in different interval or block
      current = msg_vertex;
    } else {
      // current is safe during call to update_neighbors
      current = get_active_vertex(interval_id, block_id, msg_vertex->vid);
      assertm(current != nullptr,
              "get_active_vertex yielded nullptr radius_tile");
    }
    assertm(current->valid_vid(), "fifo must recover a valid_vid vertex");

    // radius field can now be be mutated
    // set any vertex that shares a border with background
    // to the known radius of 1
    // if (current->surface() && (current->radius != 1)) {
    if (current->surface()) {
#ifdef FULL_PRINT
      cout << "radius_tile surface " << current->vid << '\n';
#endif
      current->radius = 1;
      // if in domain notify potential outside domains
      if (!(msg_vertex->band())) {
        check_ghost_update(interval_id, block_id, current, stage);
      }
    }

#ifdef LOG_FULL
    visited += 1;
#endif

    bool _ = false;
    update_neighbors(tile, interval_id, block_id, current, current->vid,
                     revisits, stage, tile_thresholds, _, *current, fifo, _,
                     vdb_accessor);
  }
}

template <class image_t>
template <class Container, typename T>
void Recut<image_t>::prune_tile(const image_t *tile, VID_t interval_id,
                                VID_t block_id, std::string stage,
                                const TileThresholds<image_t> *tile_thresholds,
                                Container &fifo, VID_t revisits,
                                T vdb_accessor) {
  VertexAttr *current;
  bool current_outside_domain, covered;
  VID_t visited = 0;
  while (!(fifo.empty())) {
    covered = false;
    // fifo starts with only roots
    auto msg_vertex = &(fifo.front());
    fifo.pop_front();

#ifdef DENSE
    current = get_vertex_vid(interval_id, block_id, msg_vertex->vid, nullptr);
#else
    if (msg_vertex->band()) {
      // current can be from ghost region in different interval or block
      current = msg_vertex;
    } else {
      current = get_active_vertex(interval_id, block_id, msg_vertex->vid);
      assertm(current != nullptr,
              "get_active_vertex yielded nullptr radius_tile");
    }
    assertm(current->valid_vid(), "fifo must recover a valid_vid vertex");
    assertm(current->valid_radius(), "fifo must recover a valid_radius vertex");
#endif

#ifdef LOG_FULL
    visited += 1;
#endif

    // Parent
    // by default current will have a root of itself
    if (current->root()) {
      current->set_parent(current->vid);
      current->prune_visit();
    }

    VID_t revisits;
    VertexAttr found_higher_parent = *current;

    bool enqueue_dsts = false;
    bool _ = false;
    // if current has already been pruned you already know it's covered
    if (current->unvisited()) {
      covered = true;
    } else {
      // check if covered
      update_neighbors(nullptr, interval_id, block_id, current, current->vid,
                       revisits, "prune", nullptr, _, found_higher_parent, fifo,
                       covered, vdb_accessor, current_outside_domain,
                       enqueue_dsts);
    }

    enqueue_dsts = true;
    // if current is covered second pass here sets all dst->parent to
    // current->parent
    update_neighbors(nullptr, interval_id, block_id, current, current->vid,
                     revisits, "prune", nullptr, _, found_higher_parent, fifo,
                     covered, vdb_accessor, current_outside_domain,
                     enqueue_dsts);

    if (covered) {
      // roots can never be added back in by accumulate_prune
      // so fifo is safe from infinite loops
      if (!((current->root()))) {
        // never visit again and never print it
        current->mark_unvisited();
#ifdef FULL_PRINT
        std::cout << "  covered vid " << current->vid << '\n';
#endif
        // get radii before adjusting parent is intended to
        // preserve coverage info as it decreases in range
        // only pruned / unselected vertices will have their radii mutated
        // to transitively pass coverage info across blocks
        // using only vertices
        if (found_higher_parent.vid != current->vid) {
#ifdef FULL_PRINT
          std::cout << "  refreshed radius to "
                    << +(found_higher_parent.radius - 1) << " from vid "
                    << found_higher_parent.vid << '\n';
#endif
          assertm(found_higher_parent.valid_radius(),
                  "higher parent invalid radius");
          assertm(found_higher_parent.radius != 0,
                  "higher parent can't have a radii of 0");
          current->radius = found_higher_parent.radius - 1;
        } else {
          if (current->radius != 0) {
            // if an adjacent (dst) is covering current and that adjacent is
            // not higher than current, respect dsts radius as truth and
            // do not refresh the transitive radius coverage
            // this has advantage of a radius always covering only its
            // hops disadvantage is that equal or lower values can not
            // "refresh" the coverage
            current->radius = current->radius - 1;
          }
        }

        if (!current_outside_domain) {
          check_ghost_update(interval_id, block_id, current, stage);
        }
        continue;
      }
    }

    if (!(current->root())) {
      current->mark_band();
    }
    if (!current_outside_domain) {
      check_ghost_update(interval_id, block_id, current, stage);
    }
  } // end while
} // end prune_tile

// template <class image_t>
// template <class Container, typename T>
// void Recut<image_t>::prune_tile_assign_parent(
// const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
// const TileThresholds<image_t> *tile_thresholds, Container &fifo,
// VID_t revisits, T vdb_accessor) {
// VertexAttr *current;
//// fifo only starts with roots
// while (!(fifo.empty())) {
// auto covered = false;
// uint64_t vid = fifo.front();
// fifo.pop_front();

//#ifdef DENSE
// current = get_vertex_vid(interval_id, block_id, vid, nullptr);
//#else
// current = get_active_vertex(interval_id, block_id, vid);
//#endif

// assertm(current != nullptr, "get_active_vertex yielded nullptr");
// const auto current_interval_id = get_interval_id(current->vid);
// const auto current_block_id = get_block_id(current->vid);
// bool current_outside_domain = false;
// if ((current_interval_id != interval_id) ||
//(current_block_id != block_id)) {
// current_outside_domain = true;
//}
// assertm(current->valid_vid(), "fifo must recover a valid_vid vertex");
// assertm(current->valid_radius(), "fifo must recover a valid_radius vertex");

//#ifdef LOG_FULL
// visited += 1;
//#endif

//// Parent
//// by default current will have a root of itself
// if (current->root()) {
// current->set_parent(current->vid);
// current->prune_visit();
//}

// VID_t revisits;
// VertexAttr found_higher_parent = *current;

// bool enqueue_dsts = false;
//// if current has already been pruned you already know it's covered
// if (current->unvisited()) {
// covered = true;
//} else {
//// check if covered
// update_neighbors(nullptr, interval_id, block_id, current, revisits,
//"prune", nullptr, false, found_higher_parent, fifo,
// covered, current_outside_domain, enqueue_dsts);
//}

// enqueue_dsts = true;
//// if current is covered second pass here sets all dst->parent to
//// current->parent
// update_neighbors(nullptr, interval_id, block_id, current, revisits, "prune",
// nullptr, false, found_higher_parent, fifo, covered,
// current_outside_domain, enqueue_dsts);

// if (covered) {
//// roots can never be added back in by accumulate_prune
//// so fifo is safe from infinite loops
// if (!((current->root()))) {
//// never visit again and never print it
// current->mark_unvisited();
//#ifdef FULL_PRINT
// std::cout << "  covered vid " << current->vid << '\n';
//#endif
//// get radii before adjusting parent is intended to
//// preserve coverage info as it decreases in range
//// only pruned / unselected vertices will have their radii mutated
//// to transitively pass coverage info across blocks
//// using only vertices
// if (found_higher_parent.vid != current->vid) {
//#ifdef FULL_PRINT
// std::cout << "  refreshed radius to "
//<< +(found_higher_parent.radius - 1) << " from vid "
//<< found_higher_parent.vid << '\n';
//#endif
// assertm(found_higher_parent.valid_radius(),
//"higher parent invalid radius");
// assertm(found_higher_parent.radius != 0,
//"higher parent can't have a radii of 0");
// current->radius = found_higher_parent.radius - 1;
//} else {
// if (current->radius != 0) {
//// if an adjacent (dst) is covering current and that adjacent is
//// not higher than current, respect dsts radius as truth and
//// do not refresh the transitive radius coverage
//// this has advantage of a radius always covering only its
//// hops disadvantage is that equal or lower values can not
//// "refresh" the coverage
// current->radius = current->radius - 1;
//}
//}

// if (!current_outside_domain) {
// check_ghost_update(interval_id, block_id, current, stage);
//}
// continue;
//}
//}

// if (!(current->root())) {
// current->mark_band();
//}
// if (!current_outside_domain) {
// check_ghost_update(interval_id, block_id, current, stage);
//}
//} // end while
//} // end prune_tile

template <class image_t>
template <class Container, typename T>
void Recut<image_t>::march_narrow_band(
    const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds, Container &fifo,
    T vdb_accessor) {
#ifdef LOG_FULL
  struct timespec time0, time1;
  VID_t visited = 0;
  clock_gettime(CLOCK_REALTIME, &time0);
  cout << "Start marching interval: " << interval_id << " block: " << block_id
       << '\n';
#endif

  VID_t revisits = 0;

  if (stage == "connected") {
    connected_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
                   revisits, vdb_accessor);
  } else if (stage == "radius") {
    radius_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
                revisits, vdb_accessor);
  } else if (stage == "prune") {
    prune_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
               revisits, vdb_accessor);
  } else if (stage == "value") {
    value_tile(tile, interval_id, block_id, stage, tile_thresholds, fifo,
               revisits, vdb_accessor);
  } else {
    assertm(false, "Stage name not recognized");
  }

#ifdef LOG_FULL
  clock_gettime(CLOCK_REALTIME, &time1);
  cout << "Marched interval: " << interval_id << " block: " << block_id
       << " visiting " << visited << " revisits " << revisits << " in "
       << diff_time(time0, time1) << " s" << '\n';
#endif

#ifdef RV
  // warning this is an atomic variable, which may incur costs such as
  // unwanted or unexpected serial behavior between blocks
  global_revisits += revisits;
#endif

  // Note: could set explicit memory ordering on atomic
  active_blocks[interval_id][block_id].store(false);
  processing_blocks[interval_id][block_id].store(
      false); // release block_id heap
} // end march_narrow_band

/* intervals are arranged in c-order in 3D, therefore each
 * intervals can be accessed via it's intervals subscript or by a linear idx
 * id : the linear idx of this interval in the entire domain
 * i, j, k : the subscript to this interval
 */
template <class image_t>
void Recut<image_t>::get_interval_subscript(const VID_t id, VID_t &i, VID_t &j,
                                            VID_t &k) {
  i = id % grid_interval_length_x;
  j = (id / grid_interval_length_x) % grid_interval_length_y;
  k = (id / (grid_interval_length_x * grid_interval_length_y)) %
      grid_interval_length_z;
}

template <class image_t>
void Recut<image_t>::get_interval_offsets(const VID_t interval_id,
                                          vector<int> &interval_offsets,
                                          vector<int> &interval_extents) {
  VID_t i, j, k;
  get_interval_subscript(interval_id, i, j, k);
  vector<int> subs = {static_cast<int>(k), static_cast<int>(j),
                      static_cast<int>(i)};
  interval_offsets = {0, 0, 0};
  interval_extents = {static_cast<int>(interval_length_z),
                      static_cast<int>(interval_length_y),
                      static_cast<int>(interval_length_x)};
  // increment the offset location to extract
  // args->image_offsets args->image_extents are in z y x order
  interval_offsets[2] += args->image_offsets[2];
  interval_offsets[1] += args->image_offsets[1];
  interval_offsets[0] += args->image_offsets[0];
  vector<int> szs = {(int)image_length_z, (int)image_length_y,
                     (int)image_length_x};
  std::vector<int> interval_lengths = {static_cast<int>(interval_length_z),
                                       static_cast<int>(interval_length_y),
                                       static_cast<int>(interval_length_x)};
  // don't constrain the extents to actual image
  // mcp3d pads with zero if requests go beyond memory
  // global command line extents have already been factored
  // into szs
  for (int i = 0; i < 3; i++) {
    interval_offsets[i] += subs[i] * interval_lengths[i];
  }
#ifdef LOG_FULL
  cout << "interval_id: " << interval_id;
  cout << " offset x " << interval_offsets[2] << " offset y "
       << interval_offsets[1] << " offset z " << interval_offsets[0] << '\n';
  cout << " extents x " << interval_extents[2] << " extents y "
       << interval_extents[1] << " extents z " << interval_extents[0] << '\n';
#endif
}

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

    bound_band += stride;

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

#ifdef USE_OMP_BLOCK
#pragma omp parallel for
#endif
    for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
      march_narrow_band(tile, interval_id, block_id, stage, tile_thresholds,
                        fifo[block_id], vdb_accessor);
    }

#endif // ASYNC

#ifdef LOG_FULL
    cout << "Marched narrow band";
    clock_gettime(CLOCK_REALTIME, &postmarch_time);
    if (restart_bool) {
      cout << " with bound_band: " << bound_band;
    }
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
  if (!(this->mmap_))
    cout << "Finished saving interval in "
         << diff_time(presave_time, postsave_time) << " sec." << '\n';
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

  vector<int> interval_offsets;
  vector<int> interval_extents;

  // read image data
  // FIXME check that this has no state that can
  // be corrupted in a shared setting
  // otherwise just create copies of it if necessary
  assertm(!(this->params->force_regenerate_image),
          "If USE_MCP3D macro is set, this->params->force_regenerate_image "
          "must be set "
          "to False");
  // read data from channel
  mcp3d_tile.ReadImageInfo(args->resolution_level(), true);
  // read data
  try {
    get_interval_offsets(interval_id, interval_offsets, interval_extents);
    // use unit strides only
    mcp3d::MImageBlock block(interval_offsets, interval_extents);
    mcp3d_tile.SelectView(block, args->resolution_level());
    mcp3d_tile.ReadData(true, "quiet");
  } catch (...) {
    MCP3D_MESSAGE("error in mcp3d_tile io. neuron tracing not performed")
    throw;
  }
#ifdef FULL_PRINT
  // print_image_3D(mcp3d_tile.Volume<image_t>(0), interval_extents);
#endif

#ifdef LOG
  clock_gettime(CLOCK_REALTIME, &image_load);
  cout << "Load image in " << diff_time(start, image_load) << " sec." << '\n';
#endif
#ifdef LOG_FULL
  cout << "interval offsets x " << interval_offsets[2] << " y "
       << interval_offsets[1] << " z " << interval_offsets[0] << '\n';
  cout << "interval extents x " << interval_extents[2] << " y "
       << interval_extents[1] << " z " << interval_extents[0] << '\n';
#endif
}

// Calculate new tile thresholds or use input thresholds according
// to params and args this function has no sideffects outside
// of the returned tile_thresholds struct
template <class image_t>
TileThresholds<image_t> *
Recut<image_t>::get_tile_thresholds(mcp3d::MImage &mcp3d_tile) {

  auto tile_thresholds = new TileThresholds<image_t>();

  vector<int> interval_dims =
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
    // tile_thresholds->bkg_thresh default inits to 0
  }

  // assign max and min ints for this tile
  if (this->args->recut_parameters().get_max_intensity() < 0) {
    // max and min members will be set
    tile_thresholds->get_max_min(mcp3d_tile.Volume<image_t>(0),
                                 interval_vertex_size);
  } else if (this->args->recut_parameters().get_min_intensity() < 0) {
    // if max intensity was set but not a min, just use the bkg_thresh value
    if (tile_thresholds->bkg_thresh >= 0) {
      tile_thresholds->min_int = tile_thresholds->bkg_thresh;
    } else {
      // max and min members will be set
      tile_thresholds->get_max_min(mcp3d_tile.Volume<image_t>(0),
                                   interval_vertex_size);
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
  auto interval_open_count = std::vector<uint16_t>(grid_interval_size, 0);
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
  assertm(this->topology_grid, "topology grid not initialized");
  auto vdb_const_accessor = this->topology_grid->getConstAccessor();
  auto vdb_accessor = this->topology_grid->getAccessor();
#endif

  // Main march for loop
  // continue iterating until all intervals are finished
  // intervals can be (re)activated by neighboring intervals
  bound_band = 0; // for restart
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
        TileThresholds<image_t> *local_tile_thresholds = tile_thresholds;

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
        mcp3d::MImage mcp3d_tile(
            args->image_root_dir(),
            {args->channel()}); // prevent destruction before calling
                                // process_interval
        // tile is only needed for the value stage
        if (stage == "value" || stage == "connected" || stage == "convert") {
          if (!(this->params->force_regenerate_image)) {
            if (!this->input_is_vdb) {
              // mcp3d_tile must be kept in scope during the processing
              // of this interval otherwise dangling reference then seg fault
              // on image access
              load_tile(interval_id, mcp3d_tile);
              if (!local_tile_thresholds) {
                local_tile_thresholds = get_tile_thresholds(mcp3d_tile);
              }
              tile = mcp3d_tile.Volume<image_t>(0);
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
          auto convert_start = timer->elapsed();

          vector<int> interval_offsets;
          vector<int> no_offsets = {0, 0, 0};
          vector<int> interval_extents;
          get_interval_offsets(interval_id, interval_offsets, interval_extents);

          vector<int> buffer_offsets =
              params->force_regenerate_image ? interval_offsets : no_offsets;
          vector<int> buffer_extents = params->force_regenerate_image
                                           ? this->image_extents
                                           : interval_extents;

          convert_buffer_to_vdb(tile, vdb_accessor, buffer_extents,
                                /*buffer_offsets=*/buffer_offsets,
                                /*image_offsets=*/interval_offsets,
                                local_tile_thresholds->bkg_thresh);

          active_intervals[interval_id] = false;
          computation_time =
              computation_time + (timer->elapsed() - convert_start);
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
  auto total_update_time = timer->elapsed();

  if (stage == "convert") {

    // finalize grid
    timer->restart();
    print_grid_metadata(topology_grid);

    topology_grid->pruneGrid();
    print_grid_metadata(topology_grid);
    computation_time = computation_time + timer->elapsed();

  } else {
#ifdef RV
    cout << "Total ";
#ifdef NO_RV
    cout << "rejected";
#endif
    cout << " revisits: " << global_revisits << " vertices" << '\n';
#endif
  }

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

/* get the vid with respect to the entire image passed to the
 * recut program. Note this spans multiple tiles and blocks
 * Take the subscripts of the vertex or voxel
 * returns the linear idx into the entire domain
 */
template <class image_t>
inline VID_t Recut<image_t>::get_img_vid(const VID_t i, const VID_t j,
                                         const VID_t k) {
  return k * image_length_xy + j * image_length_x + i;
}

/*
 * blocks are arranged in c-order in 3D, therefore each block can
 * be accessed via it's block subscript or by a linear idx
 * id : the blocks linear idx with respect to the individual
 *      interval
 * i, j, k : the equivalent subscript to this block
 */
template <class image_t>
inline void Recut<image_t>::get_block_subscript(const VID_t id, VID_t &i,
                                                VID_t &j, VID_t &k) {
  i = id % interval_block_len_x;
  j = (id / interval_block_len_x) % interval_block_len_y;
  k = (id / (interval_block_len_x * interval_block_len_y)) %
      interval_block_len_z;
}

/*
 * Convert block subscripts into linear index row-ordered
 */
template <class image_t>
inline VID_t Recut<image_t>::get_block_id(const VID_t iblock,
                                          const VID_t jblock,
                                          const VID_t kblock) {
  return iblock + jblock * interval_block_len_x +
         kblock * interval_block_len_x * interval_block_len_y;
}

template <class image_t>
inline void Recut<image_t>::get_img_subscript(const VID_t id, VID_t &i,
                                              VID_t &j, VID_t &k) {
  i = id % image_length_x;
  j = (id / image_length_x) % image_length_y;
  k = (id / image_length_xy) % image_length_z;
}

// Wrap-around rotate all values forward one
// This logic disentangles 0 % 32 from 32 % 32 results
// This function is abstract in that it can adjust subscripts
// of vid, block or interval
template <class image_t>
inline VID_t Recut<image_t>::rotate_index(VID_t img_sub, const VID_t current,
                                          const VID_t neighbor,
                                          const VID_t interval_block_size,
                                          const VID_t pad_block_size) {
  // when they are in the same block or index
  // then you simply need to account for the 1
  // voxel border region to get the correct subscript
  // for this dimension
  if (current == neighbor) {
    return img_sub + 1; // adjust to padded block idx
  }
  // if it's in another block/interval it can only be 1 vox away
  // so make sure the subscript itself is on the correct edge of its block
  // domain
  if (current == (neighbor + 1)) {
    assertm(img_sub == interval_block_size - 1,
            "Does not currently support diagonal connections or any ghost "
            "regions greater that 1");
    return 0;
  }
  if (current == (neighbor - 1)) {
    assertm(img_sub == 0, "Does not currently support diagonal connections or "
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
inline VertexAttr *
Recut<image_t>::get_or_set_active_vertex(const VID_t interval_id,
                                         const VID_t block_id,
                                         const VID_t img_vid, bool &found) {
  auto vertex = get_active_vertex(interval_id, block_id, img_vid);
  if (vertex) {
    found = true;
    assertm(vertex->selected() || vertex->root(),
            "all active vertices must be selected");
    return vertex;
  } else {
    found = false;
    auto v =
        &(this->active_vertices[interval_id][block_id].emplace_back(img_vid));
    v->mark_selected();
    return v;
  }
}

/*
 * Returns a pointer to the VertexAttr within interval_id,
 * block_id, and img_vid (vid with respect to global image)
 */
template <class image_t>
inline VertexAttr *Recut<image_t>::get_active_vertex(const VID_t interval_id,
                                                     const VID_t block_id,
                                                     const VID_t img_vid) {
  for (auto &v : this->active_vertices[interval_id][block_id]) {
    if (v.vid == img_vid) {
      return &v;
    }
  }
  return nullptr;
}

template <class image_t>
void Recut<image_t>::initialize_globals(const VID_t &grid_interval_size,
                                        const VID_t &interval_block_size) {

  // this->heap_vec.reserve(grid_interval_size);
  // for (int i = 0; i < grid_interval_size; i++) {
  // vector<local_heap> inner_vec;
  // this->heap_vec.reserve(interval_block_size);
  // for (int j = 0; j < interval_block_size; j++) {
  // local_heap test;
  // inner_vec.push_back(test);
  //}
  // this->heap_vec.push_back(inner_vec);
  //}

  //#ifdef LOG_FULL
  // cout << "Created global heap_vec" << '\n';
  //#endif

  // active boolean for in interval domain in block_id ghost region, in
  // domain of block
  this->active_neighbors = vector<vector<vector<bool>>>(
      grid_interval_size,
      vector<vector<bool>>(interval_block_size,
                           vector<bool>(interval_block_size)));

  if (!(params->convert_only_)) {
#ifdef CONCURRENT_MAP
    updated_ghost_vec = std::make_unique<ConcurrentMap64>();
#else
    updated_ghost_vec = vector<vector<vector<vector<struct VertexAttr>>>>(
        grid_interval_size,
        vector<vector<vector<struct VertexAttr>>>(
            interval_block_size,
            vector<vector<struct VertexAttr>>(interval_block_size,
                                              vector<struct VertexAttr>())));
#endif

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

  openvdb::initialize();
  // Create an empty grid with background value 0.
  this->topology_grid = openvdb::TopologyGrid::create();
  topology_grid->setName("topology");
  topology_grid->setCreator("recut");

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

  // for generated image runs trust the args->image_extents
  // to reflect the total global image domain
  // see get_args() in utils.hpp
  auto global_image_dims = args->image_extents;
  auto image_ext = std::string(fs::path(args->image_root_dir()).extension());
  this->input_is_vdb = image_ext == ".vdb" ? true : false;
  if (this->input_is_vdb) {

    assertm(!params->convert_only_,
            "Convert only option is not valid from vdb to vdb");
    std::string grid_name = "topology";
    this->topology_grid = read_vdb_file(args->image_root_dir(), grid_name);

    // read metadata
    auto active_voxel_dim = topology_grid->evalActiveVoxelDim();
    global_image_dims = {active_voxel_dim[2], active_voxel_dim[1],
                         active_voxel_dim[0]};
  } else {
#ifdef USE_MCP3D
    if (!(this->params->force_regenerate_image)) {
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
      global_image_dims = global_image.xyz_dims(args->resolution_level());
    }
#endif
  }

  // these are in z y x order
  vector<int> ext;

  // account for image_offsets and args->image_extents
  // extents are the length of the domain for each dim
  for (int i = 0; i < 3; i++) {
    // default image_offsets is {0, 0, 0}
    // which means start at the beginning of the image
    // this enforces the minimum extent to be 1 in each dim
    assertm(args->image_offsets[i] < global_image_dims[i],
            "input offset can not exceed dimension of image");
    // protect faulty out of bounds input if extents goes beyond
    // domain of full image, note: z, y, x order
    auto max_extent = global_image_dims[i] - args->image_offsets[i];
    if (args->image_extents[i]) {
      args->image_extents[i] = min(args->image_extents[i], max_extent);
    } else {
      // image_extents is set to grid_size for force_regenerate_image option,
      // otherwise 0,0,0 means use to the end of input image
      args->image_extents[i] = max_extent;
      // extents are now sanitized in each dimension
      // and protected from faulty offset values
    }
  }

  // save to globals the actual size of the full image
  // accounting for the input offsets and extents
  // these will be used throughout the rest of the program
  this->image_length_x = args->image_extents[2];
  this->image_length_y = args->image_extents[1];
  this->image_length_z = args->image_extents[0];
  this->image_length_xy = image_length_x * image_length_y;
  this->image_size = image_length_x * image_length_y * image_length_z;
  this->image_extents = {static_cast<int>(image_length_x),
                         static_cast<int>(image_length_y),
                         static_cast<int>(image_length_z)};

  // Determine the size of each interval in each dim
  // the image size and offsets override the user inputted interval size
  // continuous id's are the same for current or dst intervals
  // round up (pad)
  if (this->params->convert_only_) {
    // images are saved in separate z-planes, so conversion should respect that
    // for best performance
    // constrict so less data is allocated especially in z dimension
    this->interval_length_x = image_length_x;
    this->interval_length_y = image_length_y;
    this->interval_length_z = 1;
  } else {
    this->interval_length_x =
        min((VID_t)params->interval_size(), image_length_x);
    this->interval_length_y =
        min((VID_t)params->interval_size(), image_length_y);
    this->interval_length_z =
        min((VID_t)params->interval_size(), image_length_z);
  }

  // determine the length of intervals in each dim
  // rounding up (ceil)
  grid_interval_length_x =
      (image_length_x + interval_length_x - 1) / interval_length_x;
  grid_interval_length_y =
      (image_length_y + interval_length_y - 1) / interval_length_y;
  grid_interval_length_z =
      (image_length_z + interval_length_z - 1) / interval_length_z;

  // the resulting interval size override the user inputted block size
  block_length_x = min(interval_length_x, user_def_block_size);
  block_length_y = min(interval_length_y, user_def_block_size);
  block_length_z = min(interval_length_z, user_def_block_size);

  // determine length of blocks that span an interval for each dim
  // this rounds up
  interval_block_len_x =
      (interval_length_x + block_length_x - 1) / block_length_x;
  interval_block_len_y =
      (interval_length_y + block_length_y - 1) / block_length_y;
  interval_block_len_z =
      (interval_length_z + block_length_z - 1) / block_length_z;

  auto image_x_len_pad = interval_block_len_x * block_length_x;
  auto image_y_len_pad = interval_block_len_y * block_length_y;
  auto image_z_len_pad = interval_block_len_z * block_length_z;
  auto image_xy_len_pad =
      image_x_len_pad * image_y_len_pad; // saves recomputation occasionally

  this->grid_interval_size =
      grid_interval_length_x * grid_interval_length_y * grid_interval_length_z;
  this->interval_block_size =
      interval_block_len_x * interval_block_len_y * interval_block_len_z;
  pad_block_length_x = block_length_x + 2;
  pad_block_length_y = block_length_y + 2;
  pad_block_length_z = block_length_z + 2;
  pad_block_offset =
      pad_block_length_x * pad_block_length_y * pad_block_length_z;
  const VID_t grid_vertex_pad_size =
      pad_block_offset * interval_block_size * grid_interval_size;

#ifdef LOG
  cout << "block x, y, z size: " << block_length_x << ", " << block_length_y
       << ", " << block_length_z << " interval x, y, z size "
       << interval_length_x << ", " << interval_length_y << ", "
       << interval_length_z << " intervals: " << grid_interval_size
       << " blocks per interval: " << interval_block_size << '\n';
  cout << "image_length_x: " << image_length_x
       << " image_length_y: " << image_length_y
       << " image_length_z: " << image_length_z << '\n';
  cout << "interval_block_len_x: " << interval_block_len_x
       << " interval_block_len_y: " << interval_block_len_y
       << " interval_block_len_z: " << interval_block_len_z << '\n';
  cout << "image_x_len_pad: " << image_x_len_pad
       << " image_y_len_pad: " << image_y_len_pad
       << " image_z_len_pad: " << image_z_len_pad << " image_xy_len_pad "
       << image_xy_len_pad << '\n';
  // cout<< "image_offsets_x: "<< image_offsets[2] <<" image_offsets_y: "<<
  // image_offsets[1] <<" image_offsets_z: "<< image_offsets[0] << '\n';
#endif

#ifdef DENSE
  // we cast the interval id and block id to uint16 for use as a key
  // in the global variables maps, if total intervals or blocks exceed this
  // there would be overflow
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
    assertm(this->image_length_y == this->image_length_z,
            "change create_image implementation to support non-cube images");
    assertm(this->image_length_x == this->image_length_y,
            "change create_image implementation to support non-cube images");
    auto selected = create_image(this->params->tcase, this->generated_image,
                                 this->image_length_x, this->params->selected,
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
      return process_marker_dir(args->image_offsets, args->image_extents);
    }
  }
}

#ifdef DENSE
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

  // first convert from tile id to non- padded block subs
  get_img_subscript(img_vid, i, j, k);
  // in case interval_length isn't evenly divisible by block size
  // mod out any contributions from the interval
  auto ia = i % interval_length_x;
  auto ja = j % interval_length_y;
  auto ka = k % interval_length_z;
  // these are subscripts within the non-padded block domain
  // these values will be modified by rotate_index to account for padding
  img_block_i = ia % block_length_x;
  img_block_j = ja % block_length_y;
  img_block_k = ka % block_length_z;
  // cout << "\timg vid: " << img_vid << " img_block_i " << img_block_i
  //<< " img_block_j " << img_block_j << " img_block_k " << img_block_k<<'\n';

  // which block domain and interval does this img_vid actually belong to
  // ignoring ghost regions denoted nb_* since it may belong in the domain
  // of a neighbors block or interval all block ids are a linear row-wise
  // idx, relative to current interval
  int nb_block = (int)get_block_id(img_vid);
  int nb_interval = (int)get_interval_id(img_vid);
  // cout << "nb_interval " << nb_interval << " nb_block " << nb_block <<
  // '\n';

  // adjust block subscripts so they reflect the interval or block they belong
  // to also adjust based on actual 3D padding of block Rotate all values
  // forward one This logic disentangles 0 % 32 from 32 % 32 results within a
  // block, where ghost region is index -1 and interval_block_size
  if (interval_id == nb_interval) { // common case first
    if (nb_block == block_id) {     // grab the second common case
      pad_img_block_i = img_block_i + 1;
      pad_img_block_j = img_block_j + 1;
      pad_img_block_k = img_block_k + 1;
    } else {
      VID_t iblock, jblock, kblock, nb_iblock, nb_jblock, nb_kblock;
      // the block_id is a linear index into the 3D row-wise arrangement of
      // blocks, converting to subscript makes adjustments easier
      get_block_subscript(block_id, iblock, jblock, kblock);
      get_block_subscript(nb_block, nb_iblock, nb_jblock, nb_kblock);
      assertm(absdiff(iblock, nb_iblock) + absdiff(jblock, nb_jblock) +
                      absdiff(kblock, nb_kblock) ==
                  1,
              "Does not currently support diagonal connections or any "
              "ghost "
              "regions greater that 1");
      pad_img_block_i = rotate_index(img_block_i, iblock, nb_iblock,
                                     block_length_x, pad_block_length_x);
      pad_img_block_j = rotate_index(img_block_j, jblock, nb_jblock,
                                     block_length_y, pad_block_length_y);
      pad_img_block_k = rotate_index(img_block_k, kblock, nb_kblock,
                                     block_length_z, pad_block_length_z);
    }
  } else { // ignore block info, adjust based on interval
    // the interval_id is also linear index into the 3D row-wise arrangement
    // of intervals, converting to subscript makes adjustments easier
    VID_t iinterval, jinterval, kinterval, nb_iinterval, nb_jinterval,
        nb_kinterval;
    get_interval_subscript(interval_id, iinterval, jinterval, kinterval);
    get_interval_subscript(nb_interval, nb_iinterval, nb_jinterval,
                           nb_kinterval);
    // can only be 1 interval away
    assertm(absdiff(iinterval, nb_iinterval) +
                    absdiff(jinterval, nb_jinterval) +
                    absdiff(kinterval, nb_kinterval) ==
                1,
            "Does not currently support diagonal connections or any ghost "
            "regions greater that 1");
    // check that its in correct block of other interval, can only be 1 block
    // over note that all block ids are relative to their interval so this
    // requires a bit more logic to check, even when converting to subs
#ifndef NDEBUG
    VID_t iblock, jblock, kblock, nb_iblock, nb_jblock, nb_kblock;
    get_block_subscript(block_id, iblock, jblock, kblock);
    get_block_subscript(nb_block, nb_iblock, nb_jblock, nb_kblock);
    // overload rotate_index simply for the assert checks
    rotate_index(nb_iblock, iinterval, nb_iinterval, interval_block_len_x,
                 pad_block_length_x);
    rotate_index(nb_jblock, jinterval, nb_jinterval, interval_block_len_y,
                 pad_block_length_y);
    rotate_index(nb_kblock, kinterval, nb_kinterval, interval_block_len_z,
                 pad_block_length_z);
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
    // checked by rotate that subscript is 1 away
    pad_img_block_i = rotate_index(img_block_i, iinterval, nb_iinterval,
                                   block_length_x, pad_block_length_x);
    pad_img_block_j = rotate_index(img_block_j, jinterval, nb_jinterval,
                                   block_length_y, pad_block_length_y);
    pad_img_block_k = rotate_index(img_block_k, kinterval, nb_kinterval,
                                   block_length_z, pad_block_length_z);
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
  ////" match->value " << match->value << '\n' << '\n';
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
  map<VID_t, MyMarker *> tmp; // hash set
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
        assert(vertex->valid_vid());
        // don't create redundant copies of same vid
        if (tmp.find(vertex->vid) == tmp.end()) { // check not already added
#ifdef FULL_PRINT
          // cout << "\tadding vertex " << vertex->vid << '\n';
#endif
          VID_t i, j, k;
          i = j = k = 0;
          // get original i, j, k
          get_img_subscript(vertex->vid, i, j, k);
          // FIXME
          // set vid to be in context of entire domain of image
          // i += image_offsets[2]; // x
          // j += image_offsets[1]; // y
          // k += image_offsets[0]; // z
          // vertex->vid = get_img_vid(i, j, k);
          auto marker = new MyMarker(i, j, k);
          if (vertex->root()) {
            marker->type = 0;
          }
          marker->radius = vertex->radius;
          tmp[vertex->vid] = marker; // save this marker ptr to a map
          outtree.push_back(marker);
        } else {
          // check that all copied across blocks and intervals of a
          // single vertex all match same values other than handle
          // FIXME this needs to be moved to recut_test.cpp
          // auto previous_match = tmp[vertex->vid];
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
        // different copies have same values other than handle
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
        auto marker = tmp[vertex->vid]; // get the ptr
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
          auto parent = tmp[parent_vid]; // adjust
          marker->parent = parent;
          if (marker->parent == 0) {
            // failure condition
            std::cout << "\ninterval vid " << interval_id << '\n';
            std::cout << "block vid " << block_id << '\n';
            print_vertex(vertex);
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

template <class image_t> void Recut<image_t>::adjust_parent(bool print) {
#ifdef LOG
  cout << "Start stage adjust_parent\n";
#endif

  if (print)
    this->out.open(this->args->swc_path());

  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      for (auto &v : this->active_vertices[interval_id][block_id]) {
        adjust_vertex_parent(&v);
        if (print)
          print_vertex(&v);
      }
    }
  }

  if (this->out.is_open())
    this->out.close();
}

template <class image_t>
bool Recut<image_t>::filter_by_vid(VID_t vid, VID_t find_interval_id,
                                   VID_t find_block_id) {
  find_interval_id = get_interval_id(vid);
  if (find_interval_id >= grid_interval_size) {
    return false;
  }
  find_block_id = get_block_id(vid);
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

// only KNOWN_ROOT and KNOWN_NEW pass through this
// both are selected 0 and 0, skip
// band can be optionally included as well
template <class image_t>
bool Recut<image_t>::filter_by_label(VertexAttr *v, bool accept_band) {
  // unvisited vertices during vvalue by default
  // don't have a vid and should be ignored
  if (!(v->valid_vid())) {
    return false;
  }
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
  map<VID_t, MyMarker *> tmp;

  // iterate all active vertices
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      for (auto &v : this->active_vertices[interval_id][block_id]) {
        auto vertex = &v;
#ifdef FULL_PRINT
        print_vertex(vertex);
#endif
        assertm(vertex->valid_vid(), "no valid vid");
        assertm(filter_by_vid(vertex->vid, interval_id, block_id),
                "added to wrong container");
        // create all valid new marker objects
        if (filter_by_label(vertex, accept_band)) {
          // don't create redundant copies of same vid
          // check that all copied across blocks and intervals of a
          // single vertex all match same values other than handle
          // FIXME this needs to be moved to recut_test.cpp
          // auto previous_match = tmp[vertex->vid];
          // assert(*previous_match == *vertex);
          assertm(tmp.find(vertex->vid) == tmp.end(),
                  "Can't have two matching vids");
#ifdef FULL_PRINT
          // cout << "\tadding vertex " << vertex->vid << '\n';
#endif
          VID_t i, j, k;
          i = j = k = 0;
          // get original i, j, k
          get_img_subscript(vertex->vid, i, j, k);
          auto marker = new MyMarker(i, j, k);
          if (vertex->root()) {
            marker->type = 0;
          }
          marker->radius = vertex->radius;
          tmp[vertex->vid] = marker; // save this marker ptr to a map
          outtree.push_back(marker);
        }
      }
    }
  }

  // iterate all active vertices
  for (auto interval_id = 0; interval_id < grid_interval_size; ++interval_id) {
    for (auto block_id = 0; block_id < interval_block_size; ++block_id) {
      for (auto &v : this->active_vertices[interval_id][block_id]) {
        auto vertex = &v;
        auto parent_vid = vertex->parent;
        auto marker = tmp[vertex->vid]; // get the ptr
        if (vertex->root()) {
          marker->parent = 0;
          //#ifdef FULL_PRINT
          // cout << "found root at " << vertex->vid << '\n';
          // printf("with address of %p\n", (void *)vertex);
          //#endif
          assertm(marker->parent == 0,
                  "a marker with a parent of 0, must be a root");
          assertm(marker->type == 0,
                  "a marker with a type of 0, must be a root");
        } else {
          auto parent = tmp[parent_vid]; // adjust
          marker->parent = parent;
          if (marker->parent == 0) {
            // failure condition
            std::cout << "interval vid " << interval_id << '\n';
            std::cout << "block vid " << block_id << '\n';
            print_vertex(vertex);
            assertm(marker->parent != 0,
                    "a non root marker must have a valid parent");
          }
          assertm(marker->type != 0,
                  "a marker with a type of 0, must be a root");
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

template <class image_t> void Recut<image_t>::run_pipeline() {
  std::string stage;
  // create a list of root vids
  auto root_vids = this->initialize();

  if (params->convert_only_) {
    activate_all_intervals();

    // mutates topology_grid
    stage = "convert";
    auto tile_thresholds = new TileThresholds<image_t>(
        /*max*/ 2,
        /*min*/ 0,
        /*bkg_thresh*/ 0);
    this->update(stage, global_fifo, tile_thresholds);

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
  auto print = true;
  adjust_parent(print);
#endif
}
