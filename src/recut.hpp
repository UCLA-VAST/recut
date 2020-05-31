#pragma once

#include "grid.hpp"
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
#include <openssl/ssl.h>
#include <set>
#include <type_traits>
#include <unistd.h>
#include <unordered_set>

// taskflow significantly increases load times, avoid loading it if possible
#ifdef TF
#include <taskflow/taskflow.hpp>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

class Grid;
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
  // id: unique identifier of an instance
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
      grid_interval_length_z;

  Grid grid;
  image_t *generated_image = nullptr;
  bool mmap_;
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

#ifdef CONCURRENT_MAP
  // interval specific global data structures
  vector<vector<atomwrapper<bool>>> active_blocks;
  vector<vector<atomwrapper<bool>>> processing_blocks;
  vector<vector<local_heap>> heap_vec;
  // runtime global data structures
  std::unique_ptr<ConcurrentMap64> updated_ghost_vec;
  vector<vector<vector<bool>>> active_neighbors;
#else
  // interval specific global data structures
  vector<vector<atomwrapper<bool>>> active_blocks;
  vector<vector<atomwrapper<bool>>> processing_blocks;
  vector<vector<local_heap>> heap_vec;
  vector<vector<deque<VertexAttr *>>> surface_vec;
  // runtime global data structures
  vector<vector<vector<vector<struct VertexAttr>>>> updated_ghost_vec;
  vector<vector<vector<bool>>> active_neighbors;
#endif

  Recut(){};
  Recut(RecutCommandLineArgs &args)
      : args(&args), params(&(args.recut_parameters())), global_revisits(0),
        user_def_block_size(args.recut_parameters().block_size()),
        mmap_(false) {

#ifdef MMAP
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
  inline void release() { grid.Release(); }
  void initialize_globals(const VID_t &grid_interval_size,
                          const VID_t &interval_block_size);

  // template<typename T, typename T2>
  // void safe_increase(T &heap, T2* node) ;
  template <typename T, typename T2, typename TNew>
  void safe_update(T &heap, T2 *node, TNew new_field, std::string cmp_field);
  template <typename T, typename T2>
  void safe_push(T &heap, T2 *node, VID_t interval_id, VID_t block_id,
                 std::string cmp_field);
  template <typename T, typename T2>
  T2 safe_pop(T &heap, VID_t block_id, VID_t interval_id,
              std::string cmp_field);

  image_t get_img_val(const image_t *tile, VID_t vid);
  inline VID_t rotate_index(VID_t img_sub, const VID_t current,
                            const VID_t neighbor,
                            const VID_t interval_block_size,
                            const VID_t pad_block_size);
  int thresh_pct(const image_t *tile, VID_t interval_vertex_size,
                 const double foreground_percent);
  inline VertexAttr *get_attr_vid(VID_t interval_id, VID_t block_id, VID_t vid,
                                  VID_t *output_offset);
  void place_vertex(VID_t nb_interval_id, VID_t block_id, VID_t nb,
                    struct VertexAttr *dst, bool is_root, std::string stage);
  bool check_blocks_finish(VID_t interval_id);
  bool check_intervals_finish();
  inline VID_t get_img_vid(VID_t i, VID_t j, VID_t k);
  inline VID_t get_interval_id_vert_sub(const VID_t i, const VID_t j,
                                        const VID_t k);
  inline VID_t get_interval_id(const VID_t i, const VID_t j, const VID_t k);
  void get_interval_offsets(const VID_t interval_id,
                            vector<int> &interval_offsets,
                            vector<int> &interval_extents);
  void get_interval_subscript(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline void get_img_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline void get_block_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline VID_t get_block_id(VID_t id);
  VID_t get_interval_id(VID_t vid);
  VID_t get_sub_to_interval_id(VID_t i, VID_t j, VID_t k);
  void check_ghost_update(VID_t interval_id, VID_t block_id,
                          struct VertexAttr *dst, bool is_root,
                          std::string stage);
  int get_parent_code(VID_t dst_id, VID_t src_id);
  bool accumulate_value(const image_t *tile, VID_t interval_id, VID_t dst_id,
                        VID_t block_id, struct VertexAttr *current,
                        VID_t &revisits, int parent_code,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_background);
  void accumulate_radius(VID_t interval_id, VID_t dst_id, VID_t block_id,
                         struct VertexAttr *current, VID_t &revisits,
                         int parent_code, bool &found_adjacent_invalid,
                         VertexAttr *found_higher_parent,
                         VertexAttr *same_radius_adjacent, int stride,
                         int pad_stride);
  template <typename TNew>
  void vertex_update(VID_t interval_id, VID_t block_id, VertexAttr *dst,
                     TNew new_field, std::string stage);
  void update_neighbors(const image_t *tile, VID_t interval_id, VID_t block_id,
                        struct VertexAttr *current, VID_t &revisits,
                        std::string stage,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_adjacent_invalid,
                        VertexAttr *found_higher_parent,
                        VertexAttr *same_radius_adjacent);
  void integrate_updated_ghost(VID_t interval_id, VID_t block_id,
                               std::string stage);
  bool integrate_vertex(VID_t interval_id, VID_t block_id,
                        struct VertexAttr *updated_attr, bool ignore_KNOWN_NEW,
                        bool is_root, std::string stage);
  // void create_integrate_thread(VID_t interval_id, VID_t block_id) ;
  void march_narrow_band(const image_t *tile, VID_t interval_id, VID_t block_id,
                         std::string stage,
                         const TileThresholds<image_t> *tile_thresholds);
  void create_march_thread(VID_t interval_id, VID_t block_id);
#ifdef USE_MCP3D
  const TileThresholds<image_t> *load_tile(VID_t interval_id,
                                           mcp3d::MImage &mcp3d_tile);
#endif
  double process_interval(VID_t interval_id, const image_t *tile,
                          std::string stage,
                          const TileThresholds<image_t> *tile_thresholds);
  void update(std::string stage);
  std::vector<VID_t> initialize();
  template <typename vertex_t> void finalize(vector<vertex_t> &outtree);
  VID_t parentToVID(struct VertexAttr *attr);
  inline VID_t get_block_id(VID_t iblock, VID_t jblock, VID_t kblock);
  void print_interval(VID_t interval_id, std::string stage);
  void print_grid(std::string stage);
  void setup_radius();
  void setup_value(const std::vector<VID_t> root_vids);
  std::vector<VID_t> process_marker_dir(vector<int> global_image_offsets,
                                        vector<int> global_image_extents);
  ~Recut<image_t>();
};

template <class image_t> Recut<image_t>::~Recut<image_t>() { // FIXME
  // delete &grid;
  if (this->params->generate_image) {
    // when initialize has been run
    // generated_image is no longer nullptr
    if (this->generated_image)
      delete[] this->generated_image;
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
  // allow either dir or dir/ naming styles
  if (params->marker_file_path().back() != '/')
    params->set_marker_file_path(params->marker_file_path().append("/"));

  vector<MyMarker> inmarkers;
  vector<VID_t> root_vids;
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

      if (!check_in_bounds(x, y, z, global_image_offsets, global_image_extents))
        continue;
      // adjust the vid according to the region of the image we are processing
      i = x - global_image_offsets[2];
      j = y - global_image_offsets[1];
      k = z - global_image_offsets[0];
      auto vid = get_img_vid(i, j, k);
      root_vids.push_back(vid);

#ifdef LOG
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
template <class image_t> void Recut<image_t>::setup_radius() {
  // FIXME what affect does this have for radius?
  bool is_root = true; // changes behavior of check_ghost_update
  for (size_t interval_id = 0; interval_id < grid.GetNIntervals();
       ++interval_id) {
    Interval *interval = grid.GetInterval(interval_id);
    for (size_t block_id = 0; block_id < grid.GetNBlocks(); ++block_id) {
      if (!(this->surface_vec[interval_id][block_id].empty())) {
        interval->SetActive(true);
        active_blocks[interval_id][block_id].store(true);
#ifdef LOG
        cout << "Set interval " << interval_id << " block " << block_id
             << " to active\n";
#endif
      }
      // if (interval->get_valid_start()) {
      // auto vid = interval->get_start_vertex();
      // auto interval_id = get_interval_id(vid);
      // auto block_id = get_block_id(vid);

      // interval->SetActive(true);
      // VertexAttr *dummy_attr =
      // new VertexAttr(); // march is protect from dummy values like this
      // dummy_attr->vid = vid;
      // dummy_attr->radius = 1;

      // safe_push(this->heap_vec[interval_id][block_id], dummy_attr,
      // interval_id, block_id, "radius");
      // active_blocks[interval_id][block_id].store(true);
      //// place ghost update accounts for
      //// edges of intervals in addition to blocks
      //// this only adds to update_ghost_vec if the root happens
      //// to be on a boundary
      //// check_ghost_update(interval_id, dummy_attr, block_id, is_root,
      //// "radius"); // add to aimage_y_len other ghost zone blocks
    }
  }
}

// activates
// the intervals of the root and reads
// them to the respective heaps
template <class image_t>
void Recut<image_t>::setup_value(const std::vector<VID_t> root_vids) {
  bool is_root = true; // changes behavior of check_ghost_update
  assertm(!(root_vids.empty()), "Must have at least one root");
  for (const auto &vid : root_vids) {
    auto interval_id = get_interval_id(vid);
    auto block_id = get_block_id(vid);
    Interval *interval = grid.GetInterval(interval_id);

    interval->SetActive(true);
    VertexAttr *dummy_attr =
        new VertexAttr();       // march is protect from dummy values like this
    dummy_attr->mark_root(vid); // 0000 0000, selected no parent, all zeros
                                // indicates KNOWN_FIX root
    dummy_attr->value = 0.0;

    safe_push(this->heap_vec[interval_id][block_id], dummy_attr, interval_id,
              block_id, "value");
    active_blocks[interval_id][block_id].store(true);
    // place ghost update accounts for
    // edges of intervals in addition to blocks
    // this only adds to update_ghost_vec if the root happens
    // to be on a boundary
    check_ghost_update(interval_id, block_id, dummy_attr, is_root,
                       "value"); // add to aimage_y_len other ghost zone blocks
#ifdef LOG
    cout << "Set interval " << interval_id << " block " << block_id
         << " to active ";
    cout << "for marker vid " << vid << '\n';
#endif
  }
}

template <class image_t> void Recut<image_t>::print_grid(std::string stage) {
  for (size_t interval_id = 0; interval_id < grid.GetNIntervals();
       ++interval_id) {
    print_interval(interval_id, stage);
  }
}

template <class image_t>
void Recut<image_t>::print_interval(VID_t interval_id, std::string stage) {
  auto interval = grid.GetInterval(interval_id);

#ifdef LOG
  if (interval->IsInMemory()) {
    cout << "Print recut interval " << interval_id << " stage: " << stage
         << '\n';
  } else {
    cout << "Recut interval " << interval_id << " stage: " << stage
         << " never loaded returning without print\n";
    return;
  }
#endif

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
        // cout << "\nvid " << vid << " block " << block_id << " xadj " <<
        // xadjust
        //<< " yadj " << yadjust << " zadj " << zadjust << '\n';
        auto v = get_attr_vid(interval_id, block_id, vid, nullptr);
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
        } else if (stage == "surface") {
          // for now just print the first block of the
          // interval
          auto surface = this->surface_vec[interval_id][block_id];
          assertm(surface.size() != 0, "surface size is zero");
          if (std::count(surface.begin(), surface.end(), v)) {
            cout << "L "; // L for leaf and because disambiguates selected S
          } else {
            cout << "- ";
          }
        } else if (stage == "label") {
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
uint32_t double_pack_key(VID_t one, VID_t two) {
  uint32_t final = (uint32_t)two;
  final |= (uint32_t)one << 16;
  return final;
}

/* Note: this assumes all values are below 2<<16 - 1
 * otherwise they will overflow when cast to uint16 for
 * packing. ordered such that one is placed at the highest
 * bit location, left to right while ascending.
 */
uint64_t triple_pack_key(VID_t one, VID_t two, VID_t three) {
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
  assert(interval_id < grid.GetNIntervals());
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
template <class image_t> VID_t Recut<image_t>::get_block_id(VID_t vid) {
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
  assert(!heap.empty());
#endif
  T2 noderef = heap.top();
#ifdef HLOG_FULL
  // cout << "vid: " << noderef->vid ;
  assert(noderef->valid_handle());
#endif

  T2 node2 = heap.pop(block_id, cmp_field); // remove it

#ifdef HLOG_FULL
  assert(!node2->valid_handle());
  // cout << " handle removed" << " heap size: " << heap.size() << '\n';
#endif
  return node2;
}

// assign handle save to original attr not the heap copy
template <class image_t>
template <typename T_heap, typename T2>
void Recut<image_t>::safe_push(T_heap &heap, T2 *node, VID_t interval_id,
                               VID_t block_id, std::string cmp_field) {
#ifdef HLOG_FULL
  cout << "safe push heap size " << heap.size() << " vid: " << node->vid
       << '\n'; //" handle: " << node->handles[block_id] ;
  assert(!node->valid_handle());
  assert(node->valid_vid());
#endif
  heap.push(node, block_id, cmp_field);
#ifdef HLOG_FULL
  assert(node->valid_handle());
#endif
}

// assign handle save to original attr not the heap copy
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
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  // mod out aimage_y_len contributions from the interval
  auto ia = i % interval_length_x;
  auto ja = j % interval_length_y;
  auto ka = k % interval_length_z;
  auto interval_vid = get_interval_id_vert_sub(ia, ja, ka);
#ifdef FULL_PRINT
  // cout<< "\ti: "<<i<<" j: "<<j <<" k: "<< k<< " dst vid: " << vid << '\n';
  // cout<< "\n\tia: "<<ia<<" ja: "<<ja <<" ka: "<< ka<< " interval vid: " <<
  // interval_vid << '\n';
#endif
  auto val = tile[interval_vid];
  return val;
}

template <class image_t>
template <typename TNew>
void Recut<image_t>::vertex_update(VID_t interval_id, VID_t block_id,
                                   VertexAttr *dst, TNew new_field,
                                   std::string stage) {
  // if a visited node doesn't have a vid it will cause
  // undefined behavior
  if (dst->valid_handle()) { // in block_id heap
    safe_update(heap_vec[interval_id][block_id], dst, new_field,
                stage); // increase priority, lowers value in min-heap
#ifdef FULL_PRINT
    cout << "\t\tupdate: change in heap for BAND value" << '\n';
#endif
  } else {
    if (stage == "value") {
      dst->value = new_field;
    } else if (stage == "radius") {
      dst->radius = new_field;
    } else {
      assertm(false, "stage not recognized");
    }
    safe_push(heap_vec[interval_id][block_id], dst, interval_id, block_id,
              stage);
#ifdef FULL_PRINT
    cout << "\t\tupdate: add to heap: " << dst->vid << " value: " << dst->value
         << '\n';
#endif
  }
  check_ghost_update(interval_id, block_id, dst, false, stage);
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
void Recut<image_t>::accumulate_radius(
    VID_t interval_id, VID_t dst_id, VID_t block_id, struct VertexAttr *current,
    VID_t &revisits, int parent_code, bool &found_adjacent_invalid,
    VertexAttr *found_higher_parent, VertexAttr *same_radius_adjacent,
    int stride, int pad_stride) {

  // note the current vertex can belong in the boundary
  // region of a separate block /interval and is only
  // within this block /interval's ghost region
  // therefore all neighbors / destinations of current
  // must be checked to make sure they protude into
  // the actual current block / interval region
  // current vertex is not always within this block and interval
  // and each block, interval have a ghost region
  // after filter in update_neighbors this pointer arithmetic is always valid
  auto dst = get_attr_vid(interval_id, block_id, dst_id, nullptr);
  // if (dst->selected()) {
  // auto expected_dst_vid = current->vid + stride;
  // cout << " expected vid " << expected_dst_vid << '\n';
  // assertm(expected_dst_vid == dst->vid, "pointer arithmetic invalid");
  //}

  uint8_t updated_radius = 1 + current->radius;
  if (dst->selected()) {
    assertm(dst->valid_vid(), "selected must have a valid vid");
    assertm(dst->vid == dst_id, "get_attr_vid failed getting correct vertex");
    // FIXME would be more efficient
    // auto check_dst = current + pad_stride;
    // cout << "dst " << dst->vid << " check " << check_dst->vid << '\n';
    // assertm(dst->vid == check_dst->vid,
    //"get_attr_vid doesn't match pad stride arithmetic");
    if (!(dst->valid_radius()) || (dst->radius > updated_radius)) {
      // if radius has not been set yet
      // this necessitates it is 1 higher than current
      // OR an update from another block / interval
      // creates new lower updates
      found_higher_parent = dst;
      dst->radius = updated_radius;
#ifdef FULL_PRINT
      cout << "\tAdjacent higher"
           << " radius " << +(dst->radius) << '\n';
#endif
      this->surface_vec[interval_id][block_id].push_back(dst);
      check_ghost_update(interval_id, block_id, dst, false, "radius");
    } else if (dst->radius == current->radius) {
#ifdef FULL_PRINT
      cout << "\tAdjacent same\n";
#endif
      // aimage_y_len match adjacent same match will do
      same_radius_adjacent = dst;
    }
  } else {
    // exclude all unselected values, check that it is not a root value as well
    // since roots counterintuitively have a separate tag value than selected
    if (!(dst->root())) {
#ifdef FULL_PRINT
      cout << "\tunselected neighbor was found" << '\n';
#endif
      found_adjacent_invalid = true;
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
bool Recut<image_t>::accumulate_value(
    const image_t *tile, VID_t interval_id, VID_t dst_id, VID_t block_id,
    struct VertexAttr *current, VID_t &revisits, int parent_code,
    const TileThresholds<image_t> *tile_thresholds, bool &found_background) {

  assertm(dst_id < this->image_size, "Outside bounds of current interval");
  auto dst = get_attr_vid(interval_id, block_id, dst_id, nullptr);
  auto dst_vox = get_img_val(tile, dst_id);

#ifdef FULL_PRINT
  cout << "\tcheck dst vid: " << dst_id;
  cout << " pixel " << dst_vox;
  cout << " bkg_thresh " << +tile_thresholds->bkg_thresh << '\n';
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

  // this automatically excludes aimage_y_len root vertex since they have a
  // value of 0.
  if (dst->value > new_field) {

#ifdef RV
    if (dst->selected()) {
      revisits += 1;
#ifdef NO_RV
      return false;
#endif
    }
#endif

    // add to band (01XX XXXX)
    dst->mark_band(dst_id);
    dst->mark_connect(parent_code);
    assert(dst->valid_vid());

    vertex_update(interval_id, block_id, dst, new_field, "value");
    assertm(dst->value == new_field,
            "Accumulate value did not properly set it's updated field");
    return true;
  } else {
#ifdef FULL_PRINT
    cout << "\t\tfailed: no value change: " << dst->vid
         << " value: " << dst->value << '\n';
#endif
  }
  return false; // no update
}

template <class image_t>
void Recut<image_t>::place_vertex(VID_t nb_interval_id, VID_t block_id,
                                  VID_t nb, struct VertexAttr *dst,
                                  bool is_root, std::string stage) {

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
  if (grid.GetInterval(nb_interval_id)
          ->IsInMemory() // mmap counts all intervals as in memory
      && processing_blocks[nb_interval_id][nb].compare_exchange_strong(false,
                                                                       true)) {
    // will check if below band in march narrow
    // use processing blocks to make sure no other neighbor of nb is
    // modifying nb heap
    bool dst_update_success =
        integrate_vertex(nb_interval_id, nb, dst, true, is_root, stage);
    if (dst_update_success) { // only update if it's true, allows for
                              // remaining true
      active_blocks[nb_interval_id][nb].store(dst_update_success);
      grid.GetInterval(nb_interval_id)->SetActive(true);
#ifdef FULL_PRINT
      cout << "\t\t\tasync activate interval " << nb_interval_id << " block "
           << nb << '\n';
#endif
    }
    // Note: possible optimization here via explicit setting of memory
    // ordering on atomic
    processing_blocks[nb_interval_id][nb].store(false); // release nb heap
    // The other block isn't processing, so an update to it at here
    // is currently true for this iteration. It does not need to be checked
    // again in integrate_updated_ghost via adding it to the
    // updated_ghost_vec. Does not need to be added to active_neighbors for
    // same reason
    return;
  }
#endif

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
  grid.GetInterval(nb_interval_id)->SetActive(true);
#ifdef CONCURRENT_MAP
  auto key = triple_pack_key(nb_interval_id, block_id, nb);
  auto mutator = updated_ghost_vec->insertOrFind(key);
  std::vector<struct VertexAttr> *vec = mutator.getValue();
  if (!vec) {
    vec = new std::vector<struct VertexAttr>;
  }
  vec->emplace_back(dst->edge_state, dst->value, dst->vid, dst->radius);
  // Note: this block is unique to a single thread, therefore no other
  // thread could have this same key, since the keys are unique with
  // respect to their permutation. This means that we do not need to
  // protect from two threads modifying the same key simultaneously
  // in this design. If this did need to protected from see documentation
  // at preshing.com/20160201/new-concurrent-hash-maps-for-cpp/ for details
  mutator.assignValue(vec); // assign via mutator vs. relookup
#else
  updated_ghost_vec[nb_interval_id][block_id][nb].emplace_back(
      dst->edge_state, dst->value, dst->vid, dst->radius);
#endif
  active_neighbors[nb_interval_id][nb][block_id] = true;

#ifdef FULL_PRINT
  VID_t i, j, k;
  get_block_subscript(nb, i, j, k);
  cout << "\t\t\tghost update interval " << nb_interval_id << " nb block " << nb
       << " block i " << i << " block j " << j << " block k " << k << " vid "
       << dst->vid << '\n';
#endif
}

/*
 * This function holds all the logic of whether the update of a vertex within
 * one intervals and blocks domain is adjacent to another interval and block.
 * If the vertex is within an adjacent region then it passes the vertex to
 * place_vertex for potential updating or saving. Assumes star stencil, no
 * diagonal connection in 3D this yields 6 possible block and or interval
 * connection corners.  block_id and interval_id are in linearly addressed
 * row- order dst is always guaranteed to be within block_id and interval_id
 * region dst has already been protected from global padded out of bounds from
 * guard in accumulate. This function determines if dst is in a border region
 * and which neighbor block / interval should be notified of adjacent change
 * Warning: both update_ghost_vec and heap_vec store pointers to the same
 * underlying VertexAttr data, therefore out of order / race condition changes
 * are not protected against, however because only the first two values of
 * edge state can ever be changed by a separate thread this does not cause
 * aimage_y_len issues
 */
template <class image_t>
void Recut<image_t>::check_ghost_update(VID_t interval_id, VID_t block_id,
                                        struct VertexAttr *dst, bool is_root,
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

  VID_t tot_blocks = grid.GetNBlocks();

  // check all 6 directions for possible ghost updates
  VID_t nb; // determine neighbor block
  VID_t nb_interval_id;
  VID_t iblock, jblock, kblock;
  get_block_subscript(block_id, iblock, jblock, kblock);

#ifdef FULL_PRINT
  cout << "\t\t\tcurrent block "
       << " block i " << iblock << " block j " << jblock << " block k "
       << kblock << '\n';
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
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, is_root, stage);
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
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, is_root, stage);
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
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, is_root, stage);
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
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, is_root, stage);
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
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, is_root, stage);
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
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_id, block_id, nb, dst, is_root, stage);
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

// check and add current vertices in star stencil
template <class image_t>
void Recut<image_t>::update_neighbors(
    const image_t *tile, VID_t interval_id, VID_t block_id,
    struct VertexAttr *current, VID_t &revisits, std::string stage,
    const TileThresholds<image_t> *tile_thresholds,
    bool &found_adjacent_invalid, VertexAttr *found_higher_parent,
    VertexAttr *same_radius_adjacent) {

  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(current->vid, i, j, k);
#ifdef FULL_PRINT
  // all block ids are a linear row-wise idx, relative to current interval
  VID_t block = get_block_id(current->vid);
  cout << "\ni: " << i << " j: " << j << " k: " << k
       << " current vid: " << current->vid << " value: " << current->value
       << " interval " << interval_id << " block " << block_id << '\n';
  //" for block " << block_id << " within domain of block " << block << '\n';
#endif

  // only supports +-1 in x, y, z
  VID_t dst_id;
  int x, y, z;
  int parent_code;
  int stride, pad_stride;
  int z_stride = this->block_length_x * this->block_length_y;
  int z_pad_stride = this->pad_block_length_x * this->pad_block_length_y;
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
        if (offset == 0 || offset > 1)
          continue;
        dst_id = get_img_vid(x, y, z);

        // all block_nums and interval_nums are a linear
        // row-wise idx, relative to current interval
        auto nb = get_block_id(dst_id);
        auto ni = get_sub_to_interval_id(x, y, z);

        // Filter all dsts that don't protude into current
        // block and interval region, ghost destinations
        // can not be added in to processing stack
        // ghost vertices can only be added in to the stack
        // during `integrate_updated_ghost()`
        if (ni != interval_id)
          continue; // can't add verts of other intervals
        if (nb != block_id)
          continue; // can't add verts of other blocks

        parent_code = get_parent_code(dst_id, current->vid);
        // note although this is summed only one of ii,jj,kk
        // will be not equal to 0
        stride = ii + jj * this->block_length_x + kk * z_stride;
        pad_stride = ii + jj * this->pad_block_length_x + kk * z_pad_stride;
#ifdef FULL_PRINT
        cout << "x " << x << " y " << y << " z " << z << " dst_id " << dst_id
             << " vid " << current->vid << '\n';
#endif
        if (stage == "value") {
          accumulate_value(tile, interval_id, dst_id, block_id, current,
                           revisits, parent_code, tile_thresholds,
                           found_adjacent_invalid);
        } else if (stage == "radius") {
          accumulate_radius(interval_id, dst_id, block_id, current, revisits,
                            parent_code, found_adjacent_invalid,
                            found_higher_parent, same_radius_adjacent, stride,
                            pad_stride);
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
 * responsible for making aimage_y_len changes to activity state of interval or
 * block.
 */
template <class image_t>
bool Recut<image_t>::integrate_vertex(VID_t interval_id, VID_t block_id,
                                      struct VertexAttr *updated_attr,
                                      bool ignore_KNOWN_NEW, bool is_root,
                                      std::string stage) {
  // get attr
  auto dst = get_attr_vid(interval_id, block_id, updated_attr->vid, nullptr);

  // handle simpler radii stage and exit
  if (stage == "radius") {
    if (dst->radius > updated_attr->radius) {
      this->surface_vec[interval_id][block_id].push_back(dst);
      return true;
    }
    return false;
  }

#ifdef NO_RV
  if (ignore_KNOWN_NEW) {
    if (dst->selected()) { // 10XX XXXX KNOWN NEW
      return false;
    }
  }
#endif

#ifdef FULL_PRINT
  // cout << "\tiupdate, vid" << updated_attr->vid << '\n';
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
  // of nb, therefore the updates must go into the heapvec of nb
  // to distinguish the internal handles used for either heaps
  // handles[block_id] is set for cells within block_id blocks internal
  // domain or block_id ghost cell region, it represents all cells added to
  // heap block_id and all cells that can be manipulated by thread block_id
  assert(dst != NULL);
  assert(updated_attr != NULL);
  if (dst->value > updated_attr->value) {
    float old_val = dst->value;
    dst->copy_edge_state(*updated_attr); // copy connection to dst
    if (is_root) {
      dst->mark_root(updated_attr->vid); // mark permanently as root
#ifdef FULL_PRINT
      cout << "\tmark root: " << dst->vid << '\n';
#endif
    } else {
      dst->mark_band(
          updated_attr->vid); // ensure band in case of race condition from
                              // threads during accumulate()
    }
    assert(dst->valid_vid());
    // if heap_vec does not contain this dst already
    if (dst->valid_handle()) { // already in the heap
      // FIXME adapt this to handle radius as well
      safe_update(heap_vec[interval_id][block_id], dst, updated_attr->value,
                  stage); // increase priority, lowers value in min-heap
#ifdef FULL_PRINT
      // cout << "\tiupdate: change in heap" << " value: " << dst->value << "
      // oldval " << old_val << '\n';
#endif
    } else {
      dst->value = updated_attr->value;
      safe_push(heap_vec[interval_id][block_id], dst, interval_id, block_id,
                stage);
#ifdef FULL_PRINT
      // cout << "\tipush: add to heap: " << dst->vid << " value: " <<
      // dst->value << " oldval " << old_val << '\n';
#endif
    }
    return true;
  }
#ifdef FULL_PRINT
  cout << "\tfailed: no value change in heap" << '\n';
#endif
  return false;
} // end for each VertexAttr

/* Core exchange step of the fastmarching algorithm, this processes and
 * empties the overflow buffer updated_ghost_vec in parallel. intervals and
 * blocks can receive all of their updates from the current iterations run of
 * march_narrow_band safely to complete the iteration
 */
template <class image_t>
void Recut<image_t>::integrate_updated_ghost(VID_t interval_id, VID_t block_id,
                                             std::string stage) {
  VID_t tot_blocks = grid.GetNBlocks();
  for (VID_t nb = 0; nb < tot_blocks; nb++) {
    // active_neighbors[x][a][b] in domain of b, in ghost of block a
    if (active_neighbors[interval_id][block_id][nb]) {
      // updated ghost cleared in march_narrow_band
      // iterate over all ghost points of block_id inside domain of block nb
#ifdef CONCURRENT_MAP
      auto key = triple_pack_key(interval_id, nb, block_id);
      // get mutator so that doesn't have to reloaded when assigning
      auto mutator = updated_ghost_vec->find(key);
      std::vector<struct VertexAttr> *vec = mutator.getValue();
      // NULL if not found, continuing saves a redundant mutator assign
      if (!vec) {
        active_neighbors[interval_id][nb][block_id] = false; // reset to false
        continue; // FIXME this should never occur because of active_neighbor
                  // right?
      }

      for (struct VertexAttr updated_attr : *vec) {
#else
      for (struct VertexAttr updated_attr :
           updated_ghost_vec[interval_id][nb][block_id]) {
#endif

#ifdef FULL_PRINT
        cout << "integrate vid: " << updated_attr.vid
             << " ghost of block id: " << block_id
             << " in block domain of block id: " << nb << '\n';
#endif
        integrate_vertex(interval_id, block_id, &updated_attr, false, false,
                         stage);
      } // end for each VertexAttr
      active_neighbors[interval_id][nb][block_id] = false; // reset to false
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
    if (!heap_vec[interval_id][block_id].empty()) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_id << " block: " << block_id
           << " to active\n";
#endif
      active_blocks[interval_id][block_id].store(true);
    }
  } else if (stage == "radius") {
    if (!surface_vec[interval_id][block_id].empty()) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_id << " block: " << block_id
           << " to active\n";
#endif
      active_blocks[interval_id][block_id].store(true);
    }
  }
}

/*
 * If aimage_y_len interval is active return false, a interval is active if
 * aimage_y_len of its blocks are still active
 */
template <class image_t> bool Recut<image_t>::check_intervals_finish() {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Intervals active: ";
#endif
  for (auto interval_id = 0; interval_id < grid.GetNIntervals();
       ++interval_id) {
    if (grid.GetInterval(interval_id)->IsActive()) {
      tot_active++;
#ifdef LOG_FULL
      cout << interval_id << ", ";
#endif
    }
  }
  if (tot_active == 0) {
    return true;
  } else {
#ifdef LOG_FULL
    cout << '\n' << tot_active << " total intervals active" << '\n';
#endif
    return false;
  }
}

/*
 * If aimage_y_len block is active return false, a block is active if its
 * corresponding heap is not empty
 */
template <class image_t>
bool Recut<image_t>::check_blocks_finish(VID_t interval_id) {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Blocks active: ";
#endif
  for (auto block_id = 0; block_id < grid.GetNBlocks(); ++block_id) {
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

template <class image_t>
void Recut<image_t>::march_narrow_band(
    const image_t *tile, VID_t interval_id, VID_t block_id, std::string stage,
    const TileThresholds<image_t> *tile_thresholds) {

#ifdef LOG_FULL
  struct timespec time0, time1;
  VID_t visited = 0;
  VID_t start_size = heap_vec[interval_id][block_id].size();
  clock_gettime(CLOCK_REALTIME, &time0);
  if (!heap_vec[interval_id][block_id].empty()) {
    cout << "Start marching interval: " << interval_id << " block: " << block_id
         << '\n';
  }
#endif

  VID_t revisits = 0;
  VertexAttr *current;
  bool found_adjacent_invalid;
  if (stage == "radius") {
    while (!(this->surface_vec[interval_id][block_id].empty())) {
      // pointers must still be in memory
      VertexAttr *current = this->surface_vec[interval_id][block_id].front();
      this->surface_vec[interval_id][block_id].pop_front();

#ifdef LOG_FULL
      visited += 1;
#endif

      // invalid can either be out of range of the entire global image or it
      // can be a background vertex which occurs due to pixel value below the
      // threshold
      found_adjacent_invalid = false;
      VertexAttr *found_higher_parent = nullptr;
      VertexAttr *same_radius_adjacent = nullptr;
      update_neighbors(tile, interval_id, block_id, current, revisits, stage,
                       tile_thresholds, found_adjacent_invalid,
                       found_higher_parent, same_radius_adjacent);

      // Prune logic
      // set the proper parent, either a same or a higher adjacent
      // if (found_higher_parent && alsdkfj) {
      //}
      // you can only prune when the dst has a higher radii
      // and the current in question is
      // radii 2 or more and not a root
      // if ((current->radius >= 2) && !(current->root())) {
      //}
    }
  } else if (stage == "value") {
    while (!heap_vec[interval_id][block_id].empty()) {

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
      struct VertexAttr *dummy_min = safe_pop<local_heap, struct VertexAttr *>(
          heap_vec[interval_id][block_id], block_id, interval_id, stage);

      // protect from dummy values added to the heap not inside
      // this block or interval
      // roots are also added to the heap in initialize to prevent
      // having to load the full intervals into memory at that time
      // this retrieves the actual ptr to the vid of the root dummy min
      // values so that they can be properly initialized
      current = get_attr_vid(interval_id, block_id, dummy_min->vid, nullptr);

      // preserve state of roots
      if (!(dummy_min->root())) {
        current->mark_selected(); // set as KNOWN NEW
        assert(current->valid_vid());
        // Note: aimage_y_len adjustments to current if in a neighboring domain
        // i.e. current->handles.contains(nb) are not in a data race
        // with the nb thread. All VertexAttr have redundant copies in each
        // ghost region
      } else {
        current = dummy_min; // set as root and copy members
        cout << "set root at " << *current << '\n';
        assert(current->root());
        assertm(current->value == 0, "root value not set properly");
      }

      // invalid can either be out of range of the entire global image or it
      // can be a background vertex which occurs due to pixel value below the
      // threshold
      found_adjacent_invalid = false;
      update_neighbors(tile, interval_id, block_id, current, revisits, stage,
                       tile_thresholds, found_adjacent_invalid, nullptr,
                       nullptr);
      if (found_adjacent_invalid) {
        if (stage == "value") {
#ifdef FULL_PRINT
          cout << "found surface vertex " << current->vid << '\n';
#endif
          current->radius = 1;
          assertm(current->selected() || current->root(),
                  "surface was not selected");
          this->surface_vec[interval_id][block_id].push_back(current);
        }
      }
    }
  }

#ifdef LOG_FULL
  // if (visited > 0) {
  clock_gettime(CLOCK_REALTIME, &time1);
  cout << "Marched interval: " << interval_id << " block: " << block_id
       << " start size " << start_size << " visiting " << visited
       << " revisits " << revisits << " in " << diff_time(time0, time1) << " s"
       << '\n';
//}
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
}

// template <class image_t>
// void Recut<image_t>::create_march_thread(VID_t interval_id, VID_t
// block_id)
// {
//// process if block has been activated and is not currently being
//// processed by another task thread
// if (active_blocks[interval_id][block_id] &&
// !processing_blocks[interval_id][block_id]) {
////cout << "Start active block_id " << block_id << '\n';
// processing_blocks[interval_id][block_id] = true;
// thread(&Recut<image_t>::march_narrow_band, this, interval_id,
// block_id).detach();
////cout << "Ran through active block_id " << block_id << '\n';
//}
//}

// template <class image_t>
// void Recut<image_t>::create_integrate_thread(VID_t interval_id, VID_t
// block_id) { if
// (aimage_y_len_of(this->active_neighbors[interval_id][block_id].begin(),
// this->active_neighbors[interval_id][block_id].end(),
//[](bool i) {return i;})) {
//// Note: can optimize to taskflow if necessary
// thread(&Recut<image_t>::integrate_updated_ghost, this, interval_id,
// block_id).detach();
//}
//}

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
  vector<int> subs = {static_cast<int>( k ), static_cast<int>( j ), static_cast<int>( i )};
  interval_offsets = {0, 0, 0};
  interval_extents = {static_cast<int>( interval_length_z ), static_cast<int>( interval_length_y ), static_cast<int>( interval_length_x )};
  // increment the offset location to extract
  // args->image_offsets args->image_extents are in z y x order
  interval_offsets[2] += args->image_offsets[2];
  interval_offsets[1] += args->image_offsets[1];
  interval_offsets[0] += args->image_offsets[0];
  vector<int> szs = {(int)image_length_z, (int)image_length_y,
                     (int)image_length_x};
  std::vector<int> interval_lengths = {static_cast<int>( interval_length_z ), static_cast<int>( interval_length_y ), static_cast<int>( interval_length_x )};
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

// deprecated
template <class image_t>
int Recut<image_t>::thresh_pct(const image_t *tile, VID_t interval_vertex_size,
                               const double foreground_percent) {
#ifdef LOG
  cout << "Determine thresholding value" << '\n';
#endif
  int above = -1; // store next bkg_thresh value above desired bkg pct
  int below = -1;
  float above_diff_pct = 0.0; // pct bkg at next above value
  float below_diff_pct = 0.0; // last below percentage
  double bkg_pct = 1 - foreground_percent;
  // test different background threshold values until finding
  // percentage above bkg_pct or when all pixels set to background
  int bkg_count = 0;
  for (image_t local_bkg_thresh = 0;; local_bkg_thresh++) {
    // Count total # of pixels under current thresh
    bkg_count = 0;
#pragma omp parallel for reduction(+ : bkg_count)
    for (VID_t i = 0; i < (VID_t)interval_vertex_size; i++) {
      if (tile[i] <= local_bkg_thresh) {
        bkg_count += 1;
      }
    }

    // Check if above desired percent background
    float test_pct = bkg_count / (double)interval_vertex_size;
    // cout<<"local_bkg_thresh="<<local_bkg_thresh<<" ("<<100 *
    // test_pct<<"%)"<<'\n';
    float test_diff = abs(test_pct - bkg_pct);
    if (test_pct >= bkg_pct) {
      // failure here means:
      // all pixels labeled as background
      // if no thresh below was ever found
      assertm(below != -1, "All pixels value of 0");
      above = local_bkg_thresh;
      above_diff_pct = test_diff;
      if (above_diff_pct <= below_diff_pct)
        return above;
      else
        return below;
    } else {
      // this must be entered once
      below = local_bkg_thresh;
      below_diff_pct = test_diff;
    }
  }
}

template <class image_t>
double Recut<image_t>::process_interval(
    VID_t interval_id, const image_t *tile, std::string stage,
    const TileThresholds<image_t> *tile_thresholds) {

  struct timespec presave_time, postmarch_time, iter_start,
      start_iter_loop_time, end_iter_time, postsave_time;
  VID_t interval_block_size = grid.GetNBlocks();
  double no_io_time;
  no_io_time = 0.0;

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

    // if aimage_y_len active status for aimage_y_len block of interval_id is
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
                              tile_thresholds);
          });
          added_task = true;
#else
          async(launch::async, &Recut<image_t>::march_narrow_band, this, tile,
                interval_id, block_id, stage, tile_thresholds);
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

#else // OMP strategy

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
      march_narrow_band(tile, interval_id, block_id, stage, tile_thresholds);
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

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (VID_t block_id = 0; block_id < interval_block_size; ++block_id) {
      integrate_updated_ghost(interval_id, block_id, stage);
    }

    //#endif

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

  // this will be zero for mmap'd data
  // keep in memory for read write strategy as well since update ghost vec has
  // list of pointers
  if (!this->mmap_)
    grid.GetInterval(interval_id)->SaveToDisk();

  clock_gettime(CLOCK_REALTIME, &postsave_time);

  no_io_time = diff_time(start_iter_loop_time, presave_time);
  // global_no_io_time += no_io_time;
#ifdef LOG_FULL
  cout << "Interval: " << interval_id << " (no I/O) within " << no_io_time
       << " sec." << '\n';
  if (!this->mmap_)
    cout << "Finished saving interval in "
         << diff_time(presave_time, postsave_time) << " sec." << '\n';
#endif

  grid.GetInterval(interval_id)->SetActive(false);
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
const TileThresholds<image_t> *
Recut<image_t>::load_tile(VID_t interval_id, mcp3d::MImage &mcp3d_tile) {
#ifdef LOG
  struct timespec start, image_load;
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  auto tile_thresholds = new TileThresholds<image_t>();

  vector<int> interval_offsets;
  vector<int> interval_extents;
  vector<int> interval_dims;

  // read image data
  // FIXME check that this has no state that can
  // be corrupted in a shared setting
  // otherwise just create copies of it if necessary
  assertm(!(this->params->generate_image),
          "If USE_MCP3D macro is set, this->params->generate_image must be set "
          "to False");
  mcp3d_tile.ReadImageInfo(args->image_root_dir());
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
  print_image_3D(mcp3d_tile.Volume<image_t>(0), interval_extents);
#endif

#ifdef LOG
  clock_gettime(CLOCK_REALTIME, &image_load);
  cout << "Load image in " << diff_time(start, image_load) << " sec." << '\n';
#endif

  interval_dims = mcp3d_tile.loaded_view().view_xyz_dims();
  VID_t interval_vertex_size = static_cast<VID_t>(interval_dims[0]) *
                               interval_dims[1] * interval_dims[2];

  // assign thresholding value
  // foreground parameter takes priority
  // Note if either foreground or background percent is equal to or greater
  // than 0 than it was changed by a user so it takes precedence over the
  // defaults
  if (params->foreground_percent() >= 0) {
    tile_thresholds->bkg_thresh = mcp3d::TopPercentile<image_t>(
        mcp3d_tile.Volume<image_t>(0), interval_dims,
        params->foreground_percent());
#ifdef LOG_FULL
    cout << "Requested foreground percent: " << params->foreground_percent()
         << " yielded background threshold: " << tile_thresholds->bkg_thresh;
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
  cout << "bkg_thresh value = " << tile_thresholds->bkg_thresh << '\n';
  cout << "interval dims x " << interval_dims[2] << " y " << interval_dims[1]
       << " z " << interval_dims[0] << '\n';
  cout << "interval offsets x " << interval_offsets[2] << " y "
       << interval_offsets[1] << " z " << interval_offsets[0] << '\n';
  cout << "interval extents x " << interval_extents[2] << " y "
       << interval_extents[1] << " z " << interval_extents[0] << '\n';
#endif

  return tile_thresholds;
} // end load_tile()

#endif // only defined in USE_MCP3D is

template <class image_t> void Recut<image_t>::update(std::string stage) {
  // init all timers
  struct timespec update_start_time, update_finish_time;
  double global_no_io_time;
  global_no_io_time = 0.0;

  VID_t grid_interval_size = grid.GetNIntervals();

#ifdef LOG
  cout << "Start updating stage " << stage << '\n';
#endif
  clock_gettime(CLOCK_REALTIME, &update_start_time);

  bound_band = 0; // for restart
  // VID_t final_inner_iter = 0;
  VID_t outer_iteration_idx = 0;
  VID_t interval_id = 0;

  // begin processing all intervals
  // note this is a while since, intervals can be
  // reactivated, loop will continue until all intervals
  // are finished, see check_intervals_finish()
  // Main march for loop
  while (true) {

    // Manage iterations at interval level
    if (check_intervals_finish()) {
      break;
    }

    // only start intervals that have active processing to do
    if (grid.GetInterval(interval_id)->IsActive()) {

#ifdef LOG
      struct timespec interval_start, interval_load;
      clock_gettime(CLOCK_REALTIME, &interval_start);
#endif

      // only load the intervals that are not already mapped or have been read
      // already calling load when already present will throw
      if (!grid.GetInterval(interval_id)->IsInMemory())
        grid.GetInterval(interval_id)->LoadFromDisk();

#ifdef LOG
      clock_gettime(CLOCK_REALTIME, &interval_load);
      cout << "Load interval " << interval_id << " in "
           << diff_time(interval_start, interval_load) << " sec." << '\n';
#endif

      image_t *tile;
      const TileThresholds<image_t> *tile_thresholds =
          new TileThresholds<image_t>(1, 0, 0);
      // pre-generated images are for testing, or when an outside
      // project wants to input images instead
      if (this->params->generate_image) {
        assertm(this->generated_image,
                "Image not generated or set by intialize");
        tile = this->generated_image;
      }

#ifdef USE_MCP3D
      mcp3d::MImage
          mcp3d_tile; // prevent destruction before calling process_interval
      // tile is only needed for the value stage
      if (stage == "value") {
        if (!(this->params->generate_image)) {
          // mcp3d_tile must be kept in scope during the processing
          // of this interval otherwise dangling reference then seg fault
          // on image access
          tile_thresholds = load_tile(interval_id, mcp3d_tile);
          // FIXME need a setup image for both
          // This all needs to be designed in a way that keeps image around
          tile = mcp3d_tile.Volume<image_t>(0);
        }
      }
#else
      if (!(this->params->generate_image)) {
        assertm(false, "If USE_MCP3D macro is not set, "
                       "this->params->generate_image must be set to True");
      }
#endif

      global_no_io_time +=
          process_interval(interval_id, tile, stage, tile_thresholds);
    } // if the interval is active

    // rotate interval number until all finished
    interval_id = (interval_id + 1) % grid_interval_size;
    outer_iteration_idx++;
  } // end while for all active intervals
  clock_gettime(CLOCK_REALTIME, &update_finish_time);

#ifdef LOG
  cout << "Finished total updating within "
       << diff_time(update_start_time, update_finish_time) << " sec." << '\n';
  cout << "Finished marching (no I/O) within " << global_no_io_time << " sec."
       << '\n';
  cout << "Total interval iterations: " << outer_iteration_idx << '\n';
  //", block iterations: "<< final_inner_iter + 1<< '\n';
#endif

#ifdef RV
  cout << "Total ";
#ifdef NO_RV
  cout << "rejected";
#endif
  cout << " revisits: " << global_revisits << " vertices" << '\n';
#endif
}

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
template <class image_t>
inline VID_t Recut<image_t>::rotate_index(VID_t img_sub, const VID_t current,
                                          const VID_t neighbor,
                                          const VID_t interval_block_size,
                                          const VID_t pad_block_size) {
  if (current == neighbor)
    return img_sub + 1; // adjust to padded block idx
  // if it's in another block/interval it can only be 1 vox away
  // so make sure the subscript itself is on the correct edge of its block
  // domain
  if (current == (neighbor + 1))
    assertm(
        img_sub == interval_block_size - 1,
        "Does not currently support diagonal connections or aimage_y_len ghost "
        "regions greater that 1");
  return 0;
  if (current == (neighbor - 1))
    assertm(img_sub == 0, "Does not currently support diagonal connections or "
                          "aimage_y_len ghost regions greater that 1");
  return pad_block_size - 1;

  // failed
  assertm(
      false,
      "Does not currently support diagonal connections or aimage_y_len ghost "
      "regions greater that 1");
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
Recut<image_t>::get_attr_vid(const VID_t interval_id, const VID_t block_id,
                             const VID_t img_vid, VID_t *output_offset) {
  VID_t i, j, k, img_block_i, img_block_j, img_block_k;
  VID_t pad_img_block_i, pad_img_block_j, pad_img_block_k;
  i = j = k = 0;

  Interval *interval = grid.GetInterval(interval_id);
  assert(interval->IsInMemory());
  VertexAttr *attr = interval->GetData(); // first vertex of entire interval

  // block start calculates starting vid of block_id's first vertex within
  // the global interval array of structs Note: every vertex within this block
  // (including the ghost region) will be contiguous between attr and attr +
  // (pad_block_offset - 1) stored in row-wise order with respect to the cubic
  // block blocks within the interval are always stored according to their
  // linear block num such that a block_id * the total number of padded
  // vertices in a block i.e. pad_block_offset or (interval_block_size + 2) ^ 2
  // yields offset to the offset to the first vertex of the block.
  VID_t block_start = pad_block_offset * block_id;
  auto first_block_attr = attr + block_start;

  // Find correct offset into block

  // first convert from tile id to non- padded block subs
  get_img_subscript(img_vid, i, j, k);
  // in case interval_length isn't evenly divisible by block size
  // mod out aimage_y_len contributions from the interval
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
  // cout << "nb_interval " << nb_interval << " nb_block " << nb_block << '\n';

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
              "Does not currently support diagonal connections or aimage_y_len "
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
    assertm(
        absdiff(iinterval, nb_iinterval) + absdiff(jinterval, nb_jinterval) +
                absdiff(kinterval, nb_kinterval) ==
            1,
        "Does not currently support diagonal connections or aimage_y_len ghost "
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
    // cout << "\t\tiblock " << iblock << " nb_iblock " << nb_iblock << '\n';
    // cout << "\t\tjblock " << jblock << " nb_jblock " << nb_jblock << '\n';
    // cout << "\t\tkblock " << kblock << " nb_kblock " << nb_kblock << '\n';
    assertm(
        absdiff(iblock, nb_iblock) + absdiff(jblock, nb_jblock) +
                absdiff(kblock, nb_kblock) <=
            1,
        "Does not currently support diagonal connections or aimage_y_len ghost "
        "regions greater that 1");
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

  VertexAttr *match = first_block_attr + offset;
#ifdef FULL_PRINT
  // cout << "\t\tget attr vid for tile vid: "<< img_vid<< " pad_img_block_i "
  // << pad_img_block_i << " pad_img_block_j " << pad_img_block_j << "
  // pad_img_block_k " << pad_img_block_k<<'\n';
  ////cout << "\t\ti " << i << " j " << j << " k " << k<<'\n';
  ////cout << "\t\tia " << ia << " ja " << ja << " ka " << k<<'\n';
  // cout << "\t\tblock_num " << block_id << " nb_block " << nb_block << "
  // interval num " << interval_id << " nb_interval num " << nb_interval <<
  // '\n';;; cout << "\t\toffset " << offset << " block_start " << block_start
  // <<
  // '\n'; cout << "\t\ttotal interval size " << interval->GetNVertices() <<
  // '\n'; assert(block_start + offset < interval->GetNVertices()); // no
  // valid offset is beyond this val cout << "\t\tmatch-vid " << match->vid <<
  // " match->value " << match->value << '\n' << '\n';
#endif
  return match;
}

template <class image_t>
void Recut<image_t>::initialize_globals(const VID_t &grid_interval_size,
                                        const VID_t &interval_block_size) {

  this->heap_vec.reserve(grid_interval_size);
  for (int i = 0; i < grid_interval_size; i++) {
    vector<local_heap> inner_vec;
    this->heap_vec.reserve(interval_block_size);
    for (int j = 0; j < interval_block_size; j++) {
      local_heap test;
      inner_vec.push_back(test);
    }
    this->heap_vec.push_back(inner_vec);
  }

#ifdef LOG_FULL
  cout << "Created global heap_vec" << '\n';
#endif

  this->surface_vec = std::vector<std::vector<std::deque<VertexAttr *>>>(
      grid_interval_size, std::vector<std::deque<VertexAttr *>>(
                              interval_block_size, std::deque<VertexAttr *>()));

#ifdef LOG_FULL
  cout << "Created global surface_vec" << '\n';
#endif

  // active boolean for in interval domain in block_id ghost region, in
  // domain of block
  this->active_neighbors = vector<vector<vector<bool>>>(
      grid_interval_size,
      vector<vector<bool>>(interval_block_size,
                           vector<bool>(interval_block_size)));

#ifdef LOG_FULL
  cout << "Created active neighbors" << '\n';
#endif

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
}

template <class image_t> std::vector<VID_t> Recut<image_t>::initialize() {

#ifdef USE_OMP
  omp_set_num_threads(params->user_thread_count());
  cout << "User specific thread count " << params->user_thread_count() << '\n';
#endif

  struct timespec time0, time1, time2, time3;
  uint64_t root_64bit;

  // for generated image runs trust the args->image_extents
  // to reflect the total global image domain
  // see get_args() in utils.hpp
  auto global_image_dims = args->image_extents;
#ifdef USE_MCP3D
  if (!this->params->generate_image) {
    // determine the image size
    mcp3d::MImage global_image;
    global_image.ReadImageInfo(args->image_root_dir());
    if (global_image.image_info().empty()) {
      MCP3D_MESSAGE("no supported image formats found in " +
                    args->image_root_dir() + ", do nothing.")
      throw;
    }
    global_image.SaveImageInfo(); // save to __image_info__.json
    // reflects the total global image domain
    global_image_dims = global_image.xyz_dims(args->resolution_level());
  }
#endif

  // these are in z y x order
  vector<int> ext;

  // account for image_offsets and args->image_extents
  // extents are the length of the domain for each dim
  for (int i = 0; i < 3; i++) {
    // default image_offsets is {0, 0, 0}
    // this enforces the minimum extent to be 1 in each dim
    assertm(args->image_offsets[i] < global_image_dims[i],
            "input offset can not exceed dimension of image");
    // protect faulty out of bounds input if extents goes beyond
    // domain of full image, note: z, y, x order
    auto max_extent = global_image_dims[i] - args->image_offsets[i];
    if (args->image_extents[i]) {
      args->image_extents[i] = min(args->image_extents[i], max_extent);
    } else {
      // image_extents is set to grid_size for generate_image option, otherwise
      // 0,0,0 means use to the end of input image
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

  // the image size and offsets override the user inputted interval size
  // continuous id's are the same for current or dst intervals
  // round up (pad)
  // Determine the size of each interval in each dim
  // constrict so less data is allocated especially in z dimension
  interval_length_x = min((VID_t)params->interval_size(), image_length_x);
  interval_length_y = min((VID_t)params->interval_size(), image_length_y);
  interval_length_z = min((VID_t)params->interval_size(), image_length_z);

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

  const VID_t grid_interval_size =
      grid_interval_length_x * grid_interval_length_y * grid_interval_length_z;
  const VID_t interval_block_size =
      interval_block_len_x * interval_block_len_y * interval_block_len_z;
  pad_block_length_x = block_length_x + 2;
  pad_block_length_y = block_length_y + 2;
  pad_block_length_z = block_length_z + 2;
  pad_block_offset =
      pad_block_length_x * pad_block_length_y * pad_block_length_z;
  const VID_t grid_vertex_pad_size =
      pad_block_offset * interval_block_size * grid_interval_size;
  assertm(pad_block_offset * interval_block_size < MAX_INTERVAL_VERTICES,
          "Total vertices used by an interval can not exceed "
          "MAX_INTERVAL_VERTICES specified in vertex_attr.h");

#ifdef LOG
  cout << "block x, y, z size: " << block_length_x << ", " << block_length_y
       << ", " << block_length_z << " interval x, y, z size "
       << interval_length_x << ", " << interval_length_y << ", "
       << interval_length_z << " intervals: " << grid_interval_size
       << " blocks per interval: " << interval_block_size << '\n';
  cout << "image_length_x: " << image_length_x
       << " image_length_y: " << image_length_y
       << " image_length_z: " << image_length_z << '\n';
  cout << "nxblock: " << interval_block_len_x
       << " image_y_lenblock: " << interval_block_len_y
       << " image_z_lenblock: " << interval_block_len_z << '\n';
  cout << "image_x_len_pad: " << image_x_len_pad
       << " image_y_len_pad: " << image_y_len_pad
       << " image_z_len_pad: " << image_z_len_pad << " image_xy_len_pad "
       << image_xy_len_pad << '\n';
  // cout<< "image_offsets_x: "<< image_offsets[2] <<" image_offsets_y: "<<
  // image_offsets[1] <<" image_offsets_z: "<< image_offsets[0] << '\n';
#endif

  // we cast the interval id and block id to uint16 for use as a key
  // in the global variables maps, if total intervals or blocks exceed this
  // there would be overflow
  if (grid_interval_size > (2 << 16) - 1) {
    cout << "Number of intervals too high: " << grid_interval_size
         << " try increasing interval size";
    assert(false);
  }
  if (interval_block_size > (2 << 16) - 1) {
    cout << "Number of blocks too high: " << interval_block_size
         << " try increasing block size";
    assert(false);
  }

  // if (grid_vertex_pad_size > MAX_INTERVAL_VERTICES) {
  // cout << "Number of total vertices too high: " << grid_vertex_pad_size
  //<< " current max at: " << MAX_INTERVAL_VERTICES
  //<< " try increasing MAX_INTERVAL_BASE and rerunning interval base "
  //"generation in recut_test.hpp:CreateIntervalBase";
  // assert(false);
  //}

  clock_gettime(CLOCK_REALTIME, &time0);
  grid = Grid(grid_vertex_pad_size, interval_block_size, grid_interval_size,
              *this, this->mmap_);

  clock_gettime(CLOCK_REALTIME, &time2);

#ifdef LOG
  cout << "Created super interval in " << diff_time(time0, time2) << " s";
  cout << " with total intervals: " << grid_interval_size << '\n';
#endif

  initialize_globals(grid_interval_size, interval_block_size);

  clock_gettime(CLOCK_REALTIME, &time3);

#ifdef LOG
  cout << "Initialized globals" << diff_time(time2, time3) << '\n';
#endif

  if (this->params->generate_image) {
    const TileThresholds<image_t> tile_thresholds{1, 0, 0};
    // This is where we set image to our desired values
    this->generated_image = new image_t[this->image_size];

    assertm(this->params->tcase > -1, "Mismatched tcase for generate image");
    assertm(this->params->slt_pct > -1,
            "Mismatched slt_pct for generate image");
    assertm(this->params->selected > 0,
            "Mismatched selected for generate image");
    assertm(this->params->root_vid != numeric_limits<uint64_t>::max(),
            "Root vid uninitialized");

    // both get_grid and mesh_grid take the length of one dimension
    // of the image, currently assuming all test images
    // are cubes
    // sets all to 0 for tcase 4 and 5
    auto selected = get_grid(this->params->tcase, this->generated_image,
                             this->image_length_x);
    if (this->params->tcase == 4)
      mesh_grid(this->params->root_vid, this->generated_image,
                this->params->selected, this->image_length_x);
    else {
      this->params->selected = selected;
    }

    // add the single root vid to the root_vids
    return {this->params->root_vid};

  } else {
    // adds all valid markers to root_vids vector and returns
    return process_marker_dir(args->image_offsets, args->image_extents);
  }
}

template <class image_t>
template <typename vertex_t>
void Recut<image_t>::finalize(vector<vertex_t> &outtree) {

  struct timespec time0, time1;
#ifdef LOG
  cout << "Generating results." << '\n';
#endif
  clock_gettime(CLOCK_REALTIME, &time0);

  //#pragma omp declare reduction (merge : vector<vertex_t*> :
  // omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
  // create all marker objects
  if (is_same<vertex_t, VertexAttr>::value) {
    cout << "Warning: unimplemented section of finalize, behavior is "
            "prototype "
            "version"
         << '\n';
    // FIXME terrible performance, need an actual hash set impl, w/o buckets
    unordered_set<VID_t> tmp;
    for (size_t interval_id = 0; interval_id < grid.GetNIntervals();
         ++interval_id) {
      Interval *interval = grid.GetInterval(interval_id);
      if (!this->mmap_)
        grid.GetInterval(interval_id)->LoadFromDisk();
      //#pragma omp parallel for reduction(merge : outtree)
      for (struct VertexAttr *attr = interval->GetData();
           attr < interval->GetData() + interval->GetNVertices(); attr++) {
        // only KNOWN_ROOT and KNOWN_NEW pass through this
        // KNOWN_ROOT preserved 0000 0000 and created
        // if not selected 0 and 0, skip
        if (attr->unselected())
          continue; // skips unvisited 11XX XXXX and band 01XX XXXX
        assert(attr->valid_vid());
        // don't create redundant copies of same vid
        if (tmp.find(attr->vid) == tmp.end()) {
          VID_t i, j, k;
          i = j = k = 0;
          tmp.insert(attr->vid);
          // creates a copy of attr therfore interval can be safely deleted
          // upon ~Recut at the end of fastmarching_parallel()
          // function where program falls out of scope
          cout << "Warning: VertexAttr needs a valid cpy constructor defined"
               << '\n';
          // outtree.push_back(*attr);
        }
      }
      if (!this->mmap_)
        grid.GetInterval(interval_id)->Release();
    }
  } else if (is_same<vertex_t, MyMarker *>::value) {
#ifdef LOG
    cout << "Using MyMarker* type outtree" << '\n';
#endif
    // FIXME terrible performance
    map<VID_t, MyMarker *> tmp; // hash set
    // create all valid new marker objects
    for (size_t interval_id = 0; interval_id < grid.GetNIntervals();
         ++interval_id) {
      Interval *interval = grid.GetInterval(interval_id);
      if (this->mmap_) {
        if (!grid.GetInterval(interval_id)->IsInMemory())
          continue;
      } else {
        grid.GetInterval(interval_id)->LoadFromDisk();
      }

      struct VertexAttr *start = interval->GetData();
      for (VID_t offset = 0; offset < interval->GetNVertices(); offset++) {
        auto attr = start + offset;
        // only KNOWN_ROOT and KNOWN_NEW pass through this
        // KNOWN_ROOT preserved 0000 0000 and created
        // if not selected 0 and 0, skip
#ifdef FULL_PRINT
         cout << "checking attr " << *attr << " at offset " << offset << '\n';
#endif
        if (attr->vid == 21) { cout << "found vid 21\n";}
        if (attr->root()) {
          cout << "FOUND root " << attr << '\n';
        }
        if (attr->unselected() && !(attr->root()))
          continue; // skips unvisited 11XX XXXX and band 01XX XXXX
        assert(attr->valid_vid());
        // don't create redundant copies of same vid
        if (tmp.find(attr->vid) == tmp.end()) { // check not already added
#ifdef FULL_PRINT
          // cout << "\tadding attr " << attr->vid << '\n';
#endif
          VID_t i, j, k;
          i = j = k = 0;
          // get original i, j, k
          get_img_subscript(attr->vid, i, j, k);
          // FIXME
          // set vid to be in context of entire domain of image
          // i += image_offsets[2]; // x
          // j += image_offsets[1]; // y
          // k += image_offsets[0]; // z
          // attr->vid = get_img_vid(i, j, k);
          auto marker = new MyMarker(i, j, k);
          tmp[attr->vid] = marker; // save this marker ptr to a map
          outtree.push_back(marker);
        } else {
          // check that all copied across blocks and intervals of a
          // single vertex all match same values other than handle
          // FIXME this needs to be moved to recut_test.cpp
          // auto previous_match = tmp[attr->vid];
          // assert(*previous_match == *attr);
        }
      }
      // keep everything mmapped until all processing/reading is done
      if (!this->mmap_)
        grid.GetInterval(interval_id)->Release();
#ifdef LOG_FULL
      cout << "Total marker size : " << outtree.size() << " after interval "
           << interval_id << '\n';
#endif
    }
    // iterate through all possible, to assign parents correct pointer of
    // MyMarker
    for (size_t interval_id = 0; interval_id < grid.GetNIntervals();
         ++interval_id) {
      Interval *interval = grid.GetInterval(interval_id);
      if (this->mmap_) {
        if (!grid.GetInterval(interval_id)->IsInMemory())
          continue;
      } else {
        grid.GetInterval(interval_id)->LoadFromDisk();
      }
      for (struct VertexAttr *attr = interval->GetData();
           attr < interval->GetData() + interval->GetNVertices(); attr++) {
        // if not selected 0 and 0, skip
        if (attr->unselected())
          continue;
        // different copies have same values other than handle
        auto connects = attr->connections(image_length_x, image_length_y);
        if (connects.size() != 1) {
          cout << "Warning multi-connection (" << connects.size()
               << ") node:" << '\n'
               << attr->description() << '\n';
          for (const auto &elm : connects) {
            cout << elm << '\n';
          }
        }
        auto marker = tmp[attr->vid];      // get the ptr
        marker->parent = tmp[connects[0]]; // adjust
      }
      // release both user-space or mmap'd data since info is all in outtree
      // now
      grid.GetInterval(interval_id)->Release();
    }
  } else {
    assertm(false, "Outtree type passed not recognized. Supports "
                   "VertexAttr and MyMarker*");
  }

#ifdef LOG
  cout << "Total marker size before pruning: " << outtree.size() << " nodes"
       << '\n';
#endif

  clock_gettime(CLOCK_REALTIME, &time1);
#ifdef LOG
  cout << "Finished generating results within " << diff_time(time0, time1)
       << " sec." << '\n';
#endif
}

template <class image_t>
void Recut<image_t>::run_pipeline() {
  auto root_vids = this->initialize();

  this->setup_value(root_vids);
  this->update("value");

  this->setup_radius();
  this->update("radius");

  // this->setup_prune();
  // this->update("prune");
  this->finalize(this->args->output_tree);
}
