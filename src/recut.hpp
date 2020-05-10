#pragma once

#include "recut_parameters.hpp"
#include "super_interval.hpp"
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

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : (-(x)))
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

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

// use in conjunction with clock_gettime
inline double diff_time(struct timespec time1, struct timespec time2) {
  return time2.tv_sec - time1.tv_sec + (time2.tv_nsec - time1.tv_nsec) * 1e-9;
}

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

template <class image_t> struct TileThresholds {
  double max_int;
  double min_int;
  image_t bkg_thresh;
  const std::vector<double> givals{
      22026.5, 20368,   18840.3, 17432.5, 16134.8, 14938.4, 13834.9, 12816.8,
      11877.4, 11010.2, 10209.4, 9469.8,  8786.47, 8154.96, 7571.17, 7031.33,
      6531.99, 6069.98, 5642.39, 5246.52, 4879.94, 4540.36, 4225.71, 3934.08,
      3663.7,  3412.95, 3180.34, 2964.5,  2764.16, 2578.14, 2405.39, 2244.9,
      2095.77, 1957.14, 1828.24, 1708.36, 1596.83, 1493.05, 1396.43, 1306.47,
      1222.68, 1144.62, 1071.87, 1004.06, 940.819, 881.837, 826.806, 775.448,
      727.504, 682.734, 640.916, 601.845, 565.329, 531.193, 499.271, 469.412,
      441.474, 415.327, 390.848, 367.926, 346.454, 326.336, 307.481, 289.804,
      273.227, 257.678, 243.089, 229.396, 216.541, 204.469, 193.129, 182.475,
      172.461, 163.047, 154.195, 145.868, 138.033, 130.659, 123.717, 117.179,
      111.022, 105.22,  99.7524, 94.5979, 89.7372, 85.1526, 80.827,  76.7447,
      72.891,  69.2522, 65.8152, 62.5681, 59.4994, 56.5987, 53.856,  51.2619,
      48.8078, 46.4854, 44.2872, 42.2059, 40.2348, 38.3676, 36.5982, 34.9212,
      33.3313, 31.8236, 30.3934, 29.0364, 27.7485, 26.526,  25.365,  24.2624,
      23.2148, 22.2193, 21.273,  20.3733, 19.5176, 18.7037, 17.9292, 17.192,
      16.4902, 15.822,  15.1855, 14.579,  14.0011, 13.4503, 12.9251, 12.4242,
      11.9464, 11.4905, 11.0554, 10.6401, 10.2435, 9.86473, 9.50289, 9.15713,
      8.82667, 8.51075, 8.20867, 7.91974, 7.64333, 7.37884, 7.12569, 6.88334,
      6.65128, 6.42902, 6.2161,  6.01209, 5.81655, 5.62911, 5.44938, 5.27701,
      5.11167, 4.95303, 4.80079, 4.65467, 4.51437, 4.37966, 4.25027, 4.12597,
      4.00654, 3.89176, 3.78144, 3.67537, 3.57337, 3.47528, 3.38092, 3.29013,
      3.20276, 3.11868, 3.03773, 2.9598,  2.88475, 2.81247, 2.74285, 2.67577,
      2.61113, 2.54884, 2.48881, 2.43093, 2.37513, 2.32132, 2.26944, 2.21939,
      2.17111, 2.12454, 2.07961, 2.03625, 1.99441, 1.95403, 1.91506, 1.87744,
      1.84113, 1.80608, 1.77223, 1.73956, 1.70802, 1.67756, 1.64815, 1.61976,
      1.59234, 1.56587, 1.54032, 1.51564, 1.49182, 1.46883, 1.44664, 1.42522,
      1.40455, 1.3846,  1.36536, 1.3468,  1.3289,  1.31164, 1.29501, 1.27898,
      1.26353, 1.24866, 1.23434, 1.22056, 1.2073,  1.19456, 1.18231, 1.17055,
      1.15927, 1.14844, 1.13807, 1.12814, 1.11864, 1.10956, 1.10089, 1.09262,
      1.08475, 1.07727, 1.07017, 1.06345, 1.05709, 1.05109, 1.04545, 1.04015,
      1.03521, 1.0306,  1.02633, 1.02239, 1.01878, 1.0155,  1.01253, 1.00989,
      1.00756, 1.00555, 1.00385, 1.00246, 1.00139, 1.00062, 1.00015, 1};

  TileThresholds<image_t>(double max_int, double min_int, image_t bkg_thresh)
      : max_int(max_int), min_int(min_int), bkg_thresh(bkg_thresh) {}

  inline double calc_weight(image_t pixel) const {
    auto idx = (int)((pixel - this->min_int) / this->max_int * 255);
    assertm(idx < 256, "givals index can not exceed 255");
    assertm(idx >= 0, "givals index negative");
    return this->givals[idx];
  }
};

class SuperInterval;
template <class image_t> class Recut {
public:
  image_t *generated_image = nullptr;
  vector<VID_t> root_vids;
  bool mmap_;
  size_t iteration;
  int cnn_type;
  float bound_band;
  float stride;
  double restart_factor;
  bool restart_bool;
  image_t bkg_thresh;
  VID_t img_vox_num;
  // max and min set as double to align with look up table for value
  // estimation
  double max_int, min_int;
  int nthreads;
  VID_t nx, ny, nz, nxy, nx_block, ny_block, nz_block, user_def_block_size,
      x_pad_block_size, y_pad_block_size, z_pad_block_size, pad_block_offset,
      x_block_size, y_block_size, z_block_size, nxpad, nypad, nzpad, nxypad,
      x_interval_size, y_interval_size, z_interval_size, x_interval_num,
      y_interval_num, z_interval_num;
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
  SuperInterval super_interval;

  atomic<VID_t> global_revisits;
  VID_t vertex_issue; // default heuristic per thread for roughly best
                      // performance
  RecutCommandLineArgs *args;
  RecutParameters *params;
  vector<VID_t> interval_sizes;

  Recut(){};
  Recut(RecutCommandLineArgs &args)
      : args(&args), params(&(args.recut_parameters())), global_revisits(0),
        user_def_block_size(args.recut_parameters().block_size()), cnn_type(1),
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
  inline void release() { super_interval.Release(); }
  void initialize_globals(const VID_t &nintervals, const VID_t &nblocks);

  // template<typename T, typename T2>
  // void safe_increase(T &heap, T2* node) ;
  template <typename T, typename T2, typename TNew>
  void safe_update(T &heap, T2 *node, TNew new_field, std::string cmp_field);
  template <typename T, typename T2>
  void safe_push(T &heap, T2 *node, VID_t interval_num, VID_t block_num,
                 std::string cmp_field);
  template <typename T, typename T2>
  T2 safe_pop(T &heap, VID_t block_num, VID_t interval_num,
              std::string cmp_field);

  image_t get_img_val(const image_t *img, VID_t vid);
  inline VID_t rotate_index(VID_t img_sub, const VID_t current,
                            const VID_t neighbor, const VID_t block_size,
                            const VID_t pad_block_size);
  int thresh_pct(const image_t *img, VID_t interval_vert_num,
                 const double foreground_percent);
  void get_max_min(const image_t *img, VID_t interval_vert_num);
  inline VertexAttr *get_attr_vid(VID_t interval_num, VID_t block_num,
                                  VID_t vid, VID_t *output_offset);
  inline VertexAttr *get_attr(VID_t interval_num, VID_t block_num, VID_t ii,
                              VID_t jj, VID_t kk);
  void place_vertex(VID_t nb_interval_num, VID_t block_num, VID_t nb,
                    struct VertexAttr *dst, bool is_root, std::string stage);
  bool check_blocks_finish(VID_t interval_num);
  bool check_intervals_finish();
  inline VID_t get_block_offset(VID_t id, VID_t offset);
  inline VID_t get_img_vid(VID_t i, VID_t j, VID_t k);
  inline VID_t get_interval_id_vert_sub(const VID_t i, const VID_t j,
                                        const VID_t k);
  inline VID_t get_interval_id(const VID_t i, const VID_t j, const VID_t k);
  void get_interval_offsets(const VID_t interval_num,
                            vector<int> &interval_offsets,
                            vector<int> &interval_extents);
  void get_interval_subscript(const VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline VID_t get_vid(VID_t i, VID_t j, VID_t k);
  inline void get_img_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline void get_block_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k);
  inline VID_t get_block_num(VID_t id);
  VID_t get_interval_num(VID_t vid);
  VID_t get_sub_to_interval_num(VID_t i, VID_t j, VID_t k);
  void place_ghost_update(VID_t interval_num, VID_t block_num,
                          struct VertexAttr *dst, bool is_root,
                          std::string stage);
  int get_parent_code(VID_t dst_id, VID_t src_id);
  bool accumulate_value(const image_t *img, VID_t interval_num, VID_t dst_id,
                        VID_t block_num, struct VertexAttr *src,
                        VID_t &revisits, double factor, int parent_code,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_background);
  bool accumulate_radius(VID_t interval_num, VID_t dst_id, VID_t block_num,
                         struct VertexAttr *src, VID_t &revisits, double factor,
                         int parent_code, bool &found_adjacent_invalid);
  template <typename TNew>
  void vertex_update(VID_t interval_num, VID_t block_num, VertexAttr *dst,
                     TNew new_field, std::string stage);
  void update_neighbors(const image_t *img, VID_t interval_num,
                        struct VertexAttr *current, VID_t block_num,
                        VID_t &revisits, std::string stage,
                        const TileThresholds<image_t> *tile_thresholds,
                        bool &found_adjacent_invalid);
  void integrate_updated_ghost(VID_t interval_num, VID_t block_num,
                               std::string stage);
  bool integrate_vertex(VID_t interval_num, VID_t block_num,
                        struct VertexAttr *updated_attr, bool ignore_KNOWN_NEW,
                        bool is_root, std::string stage);
  // void create_integrate_thread(VID_t interval_num, VID_t block_num) ;
  void march_narrow_band(const image_t *img, VID_t interval_num,
                         VID_t block_num, std::string stage,
                         const TileThresholds<image_t> *tile_thresholds);
  void create_march_thread(VID_t interval_num, VID_t block_num);
#ifdef USE_MCP3D
  const TileThresholds<image_t> *load_tile(VID_t interval_num,
                                           mcp3d::MImage &mcp3d_tile);
#endif
  double process_interval(VID_t interval_num, image_t *tile, std::string stage,
                          const TileThresholds<image_t> *tile_thresholds);
  void update(std::string stage);
  template <typename vertex_t>
  void update_shardless(vector<vertex_t> roots, const vector<int> image_offsets,
                        image_t *inimg1d, VID_t nvid, int block_size_input,
                        VID_t *szs, double min_int, double max_int,
                        image_t bkg_thresh, bool restart, double restart_factor,
                        int nthreads);
  void initialize();
  template <typename vertex_t> void finalize(vector<vertex_t> &outtree);
  VID_t parentToVID(struct VertexAttr *attr);
  inline VID_t get_block_id(VID_t iblock, VID_t jblock, VID_t kblock);
  void print_interval(VID_t interval_num, std::string stage);
  void print_grid(std::string stage);
  void check_image(const image_t *img, VID_t size);
  void setup_radius();
  void setup_value();
  void process_marker_dir(vector<int> off, vector<int> end);
  ~Recut<image_t>();
};

template <class image_t> Recut<image_t>::~Recut<image_t>() { // FIXME
  // delete &super_interval;
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
  if (i > end[2])
    return false;
  if (j > end[1])
    return false;
  if (k > end[0])
    return false;
  return true;
}

// adds all markers to this->root_vids
template <class image_t>
void Recut<image_t>::process_marker_dir(vector<int> off, vector<int> end) {
  // allow either dir or dir/ naming styles
  if (params->marker_file_path().back() != '/')
    params->set_marker_file_path(params->marker_file_path().append("/"));

  vector<MyMarker> inmarkers;
  for (const auto &marker_file :
       fs::directory_iterator(params->marker_file_path())) {
    const auto marker_name = marker_file.path().filename().string();
    const auto full_marker_name = params->marker_file_path() + marker_name;
    inmarkers = readMarker_file(full_marker_name);

    // set intervals with root present as active
    for (auto &root : inmarkers) {
      // init state and phi for root
      VID_t i, j, k;
      i = root.x + 1;
      j = root.y + 1;
      k = root.z + 1;
      auto vid = get_img_vid(i, j, k);

      if (!check_in_bounds(i, j, k, off, end))
        continue;
      this->root_vids.push_back(vid);

#ifdef LOG
      cout << "Read marker x " << i << " y " << j << " z " << k << " vid "
           << vid << '\n';
#endif
    }
  }
}

// activates
// the intervals of the leaf and reads
// them to the respective heaps
// FIXME combine this with setup_value below
template <class image_t> void Recut<image_t>::setup_radius() {
  // FIXME what affect does this have for radius?
  bool is_root = true; // changes behavior of place_ghost_update
  for (size_t interval_num = 0; interval_num < super_interval.GetNIntervals();
       ++interval_num) {
    Interval *interval = super_interval.GetInterval(interval_num);
    for (size_t block_num = 0; block_num < super_interval.GetNBlocks();
         ++block_num) {
      if (!(this->surface_vec[interval_num][block_num].empty())) {
        interval->SetActive(true);
        active_blocks[interval_num][block_num].store(true);
#ifdef LOG
        cout << "Set interval " << interval_num << " block " << block_num
             << " to active\n";
#endif
      }
      // if (interval->get_valid_start()) {
      // auto vid = interval->get_start_vertex();
      // auto interval_num = get_interval_num(vid);
      // auto block_num = get_block_num(vid);

      // interval->SetActive(true);
      // VertexAttr *dummy_attr =
      // new VertexAttr(); // march is protect from dummy values like this
      // dummy_attr->vid = vid;
      // dummy_attr->radius = 1;

      // safe_push(this->heap_vec[interval_num][block_num], dummy_attr,
      // interval_num, block_num, "radius");
      // active_blocks[interval_num][block_num].store(true);
      //// place ghost update accounts for
      //// edges of intervals in addition to blocks
      //// this only adds to update_ghost_vec if the root happens
      //// to be on a boundary
      //// place_ghost_update(interval_num, dummy_attr, block_num, is_root,
      //// "radius"); // add to any other ghost zone blocks
    }
  }
}

// activates
// the intervals of the root and readds
// them to the respective heaps
template <class image_t> void Recut<image_t>::setup_value() {
  bool is_root = true; // changes behavior of place_ghost_update
  assertm(!(this->root_vids.empty()), "Must have at least one root");
  for (VID_t vid : this->root_vids) {
    auto interval_num = get_interval_num(vid);
    auto block_num = get_block_num(vid);
    Interval *interval = super_interval.GetInterval(interval_num);

    interval->SetActive(true);
    VertexAttr *dummy_attr =
        new VertexAttr();       // march is protect from dummy values like this
    dummy_attr->mark_root(vid); // 0000 0000, selected no parent, all zeros
                                // indicates KNOWN_FIX root
    dummy_attr->value = 0.0;

    safe_push(this->heap_vec[interval_num][block_num], dummy_attr, interval_num,
              block_num, "value");
    active_blocks[interval_num][block_num].store(true);
    // place ghost update accounts for
    // edges of intervals in addition to blocks
    // this only adds to update_ghost_vec if the root happens
    // to be on a boundary
    place_ghost_update(interval_num, block_num, dummy_attr, is_root,
                       "value"); // add to any other ghost zone blocks
#ifdef LOG
    cout << "Set interval " << interval_num << " block " << block_num
         << " to active ";
    cout << "for marker vid " << vid << '\n';
#endif
  }
}

template <class image_t> void Recut<image_t>::print_grid(std::string stage) {
  for (size_t interval_num = 0; interval_num < super_interval.GetNIntervals();
       ++interval_num) {
    print_interval(interval_num, stage);
  }
}

template <class image_t>
void Recut<image_t>::print_interval(VID_t interval_num, std::string stage) {
  auto interval = super_interval.GetInterval(interval_num);
  // assertm(interval->GetNVertices() == x_interval_size * y_interval_size *
  // z_interval_size, "Mismatched interval size");
  assertm(interval->IsInMemory(), "Can not print interval not in memory");
  cout << "Print recut interval " << interval_num << '\n';
  for (int zi = 0; zi < z_interval_size; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * y_interval_size + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < y_interval_size; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < x_interval_size; xi++) {
        VID_t vid = ((VID_t)xi) + yi * x_interval_size +
                    zi * x_interval_size * y_interval_size;
        auto block_num = get_block_num(vid);
        auto v = get_attr_vid(interval_num, block_num, vid, nullptr);
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
            cout << "NA ";
          }
        } else if (stage == "radius") {
          if (v->valid_radius()) {
            cout << +(v->radius) << " ";
          } else {
            cout << "NA ";
          }
        } else if (stage == "surface") {
          // for now just print the first block of the
          // interval
          auto surface = this->surface_vec[interval_num][block_num];
          assertm(surface.size() != 0, "surface size is zero");
          if (std::count(surface.begin(), surface.end(), v)) {
            cout << "S ";
          } else {
            cout << "- ";
          }
        }
      }
      cout << '\n';
    }
    cout << '\n';
  }
}

template <class image_t>
void Recut<image_t>::check_image(const image_t *img, VID_t size) {
  cout << "recut image " << '\n';
  for (VID_t i = 0; i < size; i++) {
    cout << i << " " << +(img[i]) << '\n';
    assert(img[i] <= 1);
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
 * returns interval_num, the interval domain this vertex belongs to
 * with respect to the original unpadded image
 * does not consider overlap of ghost regions
 * uses get_sub_to_interval_num() to calculate
 */
template <class image_t> VID_t Recut<image_t>::get_interval_num(VID_t vid) {
  // get the subscripts
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  return get_sub_to_interval_num(i, j, k);
}

/**
 * i, j, k : subscripts of vertex relative to entire image
 * returns interval_num, the interval domain this vertex belongs to
 * does not consider overlap of ghost regions
 */
template <class image_t>
VID_t Recut<image_t>::get_sub_to_interval_num(VID_t i, VID_t j, VID_t k) {
  auto i_interval = i / x_interval_size;
  auto j_interval = j / y_interval_size;
  auto k_interval = k / z_interval_size;
  auto interval_num = i_interval + j_interval * x_interval_num +
                      k_interval * x_interval_num * y_interval_num;
  assert(interval_num < super_interval.GetNIntervals());
  return interval_num;
}

/**
 * all block_nums are a linear row-wise idx, relative to current interval
 * vid : linear idx into the full domain inimg1d
 * the interval contributions are modded away
 * such that all block_nums are relative to a single
 * interval
 * returns block_num, the block domain this vertex belongs
 * in one of the intervals
 * Note: block_nums are renumbered within each interval
 * does not consider overlap of ghost regions
 */
template <class image_t> VID_t Recut<image_t>::get_block_num(VID_t vid) {
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  // subtract away the interval influence on the block num
  auto ia = i % x_interval_size;
  auto ja = j % y_interval_size;
  auto ka = k % z_interval_size;
  // block in this interval
  auto i_block = ia / x_block_size;
  auto j_block = ja / y_block_size;
  auto k_block = ka / z_block_size;
  return i_block + j_block * nx_block + k_block * nx_block * ny_block;
}

/** safely removes minimum element from the heap
 * heap
 */
template <class image_t>
template <typename T, typename T2>
T2 Recut<image_t>::safe_pop(T &heap, VID_t block_num, VID_t interval_num,
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

  T2 node2 = heap.pop(block_num, cmp_field); // remove it

#ifdef HLOG_FULL
  assert(!node2->valid_handle());
  // cout << " handle removed" << " heap size: " << heap.size() << '\n';
#endif
  return node2;
}

// assign handle save to original attr not the heap copy
template <class image_t>
template <typename T_heap, typename T2>
void Recut<image_t>::safe_push(T_heap &heap, T2 *node, VID_t interval_num,
                               VID_t block_num, std::string cmp_field) {
#ifdef HLOG_FULL
  cout << "safe push heap size " << heap.size() << " vid: " << node->vid
       << '\n'; //" handle: " << node->handles[block_num] ;
  assert(!node->valid_handle());
  assert(node->valid_vid());
  // auto attr = get_attr_vid(interval_num, block_num, node->vid);
  // assert(attr->vid == node->vid);
#endif
  heap.push(node, block_num, cmp_field);
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
              cmp_field); // block_num currently deprecated
}

/* get the interval linear idx from it's subscripts
 * not all linear indices are row-ordered and specific
 * to their hierarchical arrangment
 */
template <class image_t>
inline VID_t Recut<image_t>::get_interval_id(const VID_t i, const VID_t j,
                                             const VID_t k) {
  return k * (x_interval_num * y_interval_num) + j * x_interval_num + i;
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
  return k * (x_interval_size * y_interval_size) + j * x_interval_size + i;
}

/*
 * Takes a global image vid
 * and converts it to linear index of current
 * img of interval currently processed
 * returning the voxel value at that
 * location id -> sub_script i, j, k -> mod i, j, k against interval_size in
 * each dim -> convert modded subscripts to a new vid
 */
template <class image_t>
image_t Recut<image_t>::get_img_val(const image_t *img, VID_t vid) {
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
  // mod out any contributions from the interval
  auto ia = i % x_interval_size;
  auto ja = j % y_interval_size;
  auto ka = k % z_interval_size;
  auto interval_vid = get_interval_id_vert_sub(ia, ja, ka);
#ifdef FULL_PRINT
  // cout<< "\ti: "<<i<<" j: "<<j <<" k: "<< k<< " dst vid: " << vid << '\n';
  // cout<< "\n\tia: "<<ia<<" ja: "<<ja <<" ka: "<< ka<< " interval vid: " <<
  // interval_vid << '\n';
#endif
  auto val = img[interval_vid];
  return val;
}

template <class image_t>
template <typename TNew>
void Recut<image_t>::vertex_update(VID_t interval_num, VID_t block_num,
                                   VertexAttr *dst, TNew new_field,
                                   std::string stage) {
  // if a visited node doesn't have a vid it will cause
  // undefined behavior
  if (dst->valid_handle()) { // in block_num heap
    safe_update(heap_vec[interval_num][block_num], dst, new_field,
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
    safe_push(heap_vec[interval_num][block_num], dst, interval_num, block_num,
              stage);
#ifdef FULL_PRINT
    cout << "\t\tupdate: add to heap: " << dst->vid << " value: " << dst->value
         << '\n';
#endif
  }
  place_ghost_update(interval_num, block_num, dst, false, stage);
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current interval_num and block_num, since
 * it is potentially adding these vertices to the unique heap of interval_num
 * and block_num. only one parent when selected. If one of these vertexes on
 * the edge but still within interval_num and block_num domain is updated it
 * is the responsibility of place_ghost_update to take note of the update such
 * that this update is propagated to the relevant interval and block see
 * integrate_updated_ghost(). dst_id : continuous vertex id VID_t of the dst
 * vertex in question block_num : src block id src : minimum vertex attribute
 * selected returns true if this vertex is unselected (and not a root)
 */
template <class image_t>
bool Recut<image_t>::accumulate_radius(VID_t interval_num, VID_t dst_id,
                                       VID_t block_num, struct VertexAttr *src,
                                       VID_t &revisits, double factor,
                                       int parent_code,
                                       bool &found_adjacent_invalid) {

  auto dst = get_attr_vid(interval_num, block_num, dst_id, nullptr);

#ifdef FULL_PRINT
  cout << "\tcheck dst vid: " << dst_id;
#endif

  // exclude all unselected values, check that it is not a root value as well
  // since roots counterintuitively have a separate tag value than selected
  if (dst->unselected() && !(dst->root())) {
#ifdef FULL_PRINT
    cout << "\t\treturn true since unselected neighbor was found" << '\n';
#endif
    found_adjacent_invalid = true;
    return false;
  }

  // solve for update value
  // dst_id and src->vid are linear idx relative to full image domain
  assertm(src->valid_radius(), "src radius had an invalid field");
  auto new_field = src->radius + 1;

  if (dst->radius > new_field) {
    // TODO enforce this with a vector
    // assertm(!(dst->valid_radius), "A previously seen radius can not be "
    //"processed in current implementation");
    vertex_update(interval_num, block_num, dst, new_field, "radius");
    assertm(dst->radius == new_field,
            "Accumulate radius did not properly set it's updated field");
  } else {
#ifdef FULL_PRINT
    cout << "\t\tfailed: no radii change: " << dst->vid
         << " radii: " << +(dst->radius) << '\n';
#endif
  }
  return false; // it was a selected vertex
}

/**
 * accumulate is the core function of fast marching, it can only operate
 * on VertexAttr that are within the current interval_num and block_num, since
 * it is potentially adding these vertices to the unique heap of interval_num
 * and block_num. only one parent when selected. If one of these vertexes on
 * the edge but still within interval_num and block_num domain is updated it
 * is the responsibility of place_ghost_update to take note of the update such
 * that this update is propagated to the relevant interval and block see
 * integrate_updated_ghost(). dst_id : continuous vertex id VID_t of the dst
 * vertex in question block_num : src block id src : minimum vertex attribute
 * selected
 */
template <class image_t>
bool Recut<image_t>::accumulate_value(
    const image_t *img, VID_t interval_num, VID_t dst_id, VID_t block_num,
    struct VertexAttr *src, VID_t &revisits, double factor, int parent_code,
    const TileThresholds<image_t> *tile_thresholds, bool &found_background) {

  assertm(dst_id < this->img_vox_num, "Outside bounds of current interval");
  auto dst = get_attr_vid(interval_num, block_num, dst_id, nullptr);
  auto dst_vox = get_img_val(img, dst_id);

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
  // dst_id and src->vid are linear idx relative to full image domain
  float new_field = static_cast<float>(
      src->value + (tile_thresholds->calc_weight(get_img_val(img, src->vid)) +
                    tile_thresholds->calc_weight(dst_vox)) *
                       0.5 * factor);

  // this automatically excludes any root vertex since they have a value of 0.
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

    vertex_update(interval_num, block_num, dst, new_field, "value");
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
void Recut<image_t>::place_vertex(VID_t nb_interval_num, VID_t block_num,
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
  if (super_interval.GetInterval(nb_interval_num)
          ->IsInMemory() // mmap counts all intervals as in memory
      && processing_blocks[nb_interval_num][nb].compare_exchange_strong(false,
                                                                        true)) {
    // will check if below band in march narrow
    // use processing blocks to make sure no other neighbor of nb is
    // modifying nb heap
    bool dst_update_success =
        integrate_vertex(nb_interval_num, nb, dst, true, is_root, stage);
    if (dst_update_success) { // only update if it's true, allows for
                              // remaining true
      active_blocks[nb_interval_num][nb].store(dst_update_success);
      super_interval.GetInterval(nb_interval_num)->SetActive(true);
#ifdef FULL_PRINT
      cout << "\t\t\tasync activate interval " << nb_interval_num << " block "
           << nb << '\n';
#endif
    }
    // Note: possible optimization here via explicit setting of memory
    // ordering on atomic
    processing_blocks[nb_interval_num][nb].store(false); // release nb heap
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
  // interval it will be added to the same set as block_num updates from that
  // interval, this is because there is no need to separate updates based on
  // interval saving overhead
  super_interval.GetInterval(nb_interval_num)->SetActive(true);
#ifdef CONCURRENT_MAP
  auto key = triple_pack_key(nb_interval_num, block_num, nb);
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
  updated_ghost_vec[nb_interval_num][block_num][nb].emplace_back(
      dst->edge_state, dst->value, dst->vid, dst->radius);
#endif
  active_neighbors[nb_interval_num][nb][block_num] = true;

#ifdef FULL_PRINT
  VID_t i, j, k;
  get_block_subscript(nb, i, j, k);
  cout << "\t\t\tghost update interval " << nb_interval_num << " nb block "
       << nb << " block i " << i << " block j " << j << " block k " << k
       << " vid " << dst->vid << '\n';
#endif
}

/*
 * This function holds all the logic of whether the update of a vertex within
 * one intervals and blocks domain is adjacent to another interval and block.
 * If the vertex is within an adjacent region then it passes the vertex to
 * place_vertex for potential updating or saving. Assumes star stencil, no
 * diagonal connection in 3D this yields 6 possible block and or interval
 * connection corners.  block_num and interval_num are in linearly addressed
 * row- order dst is always guaranteed to be within block_num and interval_num
 * region dst has already been protected from global padded out of bounds from
 * guard in accumulate. This function determines if dst is in a border region
 * and which neighbor block / interval should be notified of adjacent change
 * Warning: both update_ghost_vec and heap_vec store pointers to the same
 * underlying VertexAttr data, therefore out of order / race condition changes
 * are not protected against, however because only the first two values of
 * edge state can ever be changed by a separate thread this does not cause any
 * issues
 */
template <class image_t>
void Recut<image_t>::place_ghost_update(VID_t interval_num, VID_t block_num,
                                        struct VertexAttr *dst, bool is_root,
                                        std::string stage) {
  VID_t i, j, k, ii, jj, kk, iii, jjj, kkk;
  vector<VID_t> interval_subs = {0, 0, 0};
  i = j = k = ii = jj = kk = 0;
  VID_t id = dst->vid;
  get_img_subscript(id, i, j, k);
  ii = i % x_block_size;
  jj = j % y_block_size;
  kk = k % z_block_size;
  iii = i % interval_sizes[2];
  jjj = j % interval_sizes[1];
  kkk = k % interval_sizes[0];

  VID_t tot_blocks = super_interval.GetNBlocks();

  // check all 6 directions for possible ghost updates
  VID_t nb; // determine neighbor block
  VID_t nb_interval_num;
  VID_t iblock, jblock, kblock;
  get_block_subscript(block_num, iblock, jblock, kblock);

#ifdef FULL_PRINT
  cout << "\t\t\tcurrent block "
       << " block i " << iblock << " block j " << jblock << " block k "
       << kblock << '\n';
#endif

  // check all six sides
  if (ii == 0) {
    if (i > 0) { // protect from image out of bounds
      nb = block_num - 1;
      nb_interval_num = interval_num; // defaults to current interval
      if (iii == 0) {
        nb_interval_num = interval_num - 1;
        // Convert block subscripts into linear index row-ordered
        nb = get_block_id(nx_block - 1, jblock, kblock);
      }
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_num, block_num, nb, dst, is_root, stage);
    }
  }
  if (jj == 0) {
    if (j > 0) { // protect from image out of bounds
      nb = block_num - nx_block;
      nb_interval_num = interval_num; // defaults to current interval
      if (jjj == 0) {
        nb_interval_num = interval_num - x_interval_num;
        nb = get_block_id(iblock, ny_block - 1, kblock);
      }
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_num, block_num, nb, dst, is_root, stage);
    }
  }
  if (kk == 0) {
    if (k > 0) { // protect from image out of bounds
      nb = block_num - nx_block * ny_block;
      nb_interval_num = interval_num; // defaults to current interval
      if (kkk == 0) {
        nb_interval_num = interval_num - x_interval_num * y_interval_num;
        nb = get_block_id(iblock, jblock, nz_block - 1);
      }
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_num, block_num, nb, dst, is_root, stage);
    }
  }

  if (kk == x_block_size - 1) {
    if (k < nz - 1) { // protect from image out of bounds
      nb = block_num + nx_block * ny_block;
      nb_interval_num = interval_num; // defaults to current interval
      if (kkk == z_interval_size - 1) {
        nb_interval_num = interval_num + x_interval_num * y_interval_num;
        nb = get_block_id(iblock, jblock, 0);
      }
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_num, block_num, nb, dst, is_root, stage);
    }
  }
  if (jj == y_block_size - 1) {
    if (j < ny - 1) { // protect from image out of bounds
      nb = block_num + nx_block;
      nb_interval_num = interval_num; // defaults to current interval
      if (jjj == y_interval_size - 1) {
        nb_interval_num = interval_num + x_interval_num;
        nb = get_block_id(iblock, 0, kblock);
      }
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_num, block_num, nb, dst, is_root, stage);
    }
  }
  if (ii == z_block_size - 1) {
    if (i < nx - 1) { // protect from image out of bounds
      nb = block_num + 1;
      nb_interval_num = interval_num; // defaults to current interval
      if (iii == x_interval_size - 1) {
        nb_interval_num = interval_num + 1;
        nb = get_block_id(0, jblock, kblock);
      }
      if ((nb >= 0) && (nb < tot_blocks)) // within valid block bounds
        place_vertex(nb_interval_num, block_num, nb, dst, is_root, stage);
    }
  }
}

/* check and add src vertices in star stencil
 * if the src is greater the parent code will be greater
 * see struct definition in vertex_attr.h for full
 * list of VertexAttr parent codes and their meaning
 */
template <class image_t>
int Recut<image_t>::get_parent_code(VID_t dst_id, VID_t src_id) {
  auto src_gr = src_id > dst_id ? true : false; // src greater
  auto adiff = absdiff(dst_id, src_id);
  if ((adiff % nxy) == 0) {
    if (src_gr)
      return 5;
    else
      return 4;
  } else if ((adiff % nx) == 0) {
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
  cout << "get_parent_code failed at src " << src_id << " dst_id " << dst_id
       << " absdiff " << adiff << '\n';
  throw;
}

// check and add src vertices in star stencil
template <class image_t>
void Recut<image_t>::update_neighbors(
    const image_t *img, VID_t interval_num, struct VertexAttr *current,
    VID_t block_num, VID_t &revisits, std::string stage,
    const TileThresholds<image_t> *tile_thresholds,
    bool &found_adjacent_invalid) {

  auto vid = current->vid;
  VID_t i, j, k;
  i = j = k = 0;
  get_img_subscript(vid, i, j, k);
#ifdef FULL_PRINT
  // all block_nums are a linear row-wise idx, relative to current interval
  VID_t block = get_block_num(vid);
  cout << "\ni: " << i << " j: " << j << " k: " << k << " src vid: " << vid
       << " value: " << current->value << '\n';
  //" for block " << block_num << " within domain of block " << block << '\n';
#endif

  // Warning: currently only supports cnn-type = 1
  // i.e. only +-1 in x, y, z
  VID_t dst_id;
  int w, h, d;
  int parent_code;
  for (int kk = -1; kk <= 1; kk++) {
    d = ((int)k) + kk;
    if (d < 0 || d >= nz) {
      found_adjacent_invalid = true;
      continue;
    }
    for (int jj = -1; jj <= 1; jj++) {
      h = ((int)j) + jj;
      if (h < 0 || h >= nz) {
        found_adjacent_invalid = true;
        continue;
      }
      for (int ii = -1; ii <= 1; ii++) {
        w = ((int)i) + ii;
        if (w < 0 || w >= nx) {
          found_adjacent_invalid = true;
          continue;
        }
        int offset = ABS(ii) + ABS(jj) + ABS(kk);
        if (offset == 0 || offset > cnn_type)
          continue;
        dst_id = get_img_vid(w, h, d);
        // all block_nums are a linear row-wise idx, relative to current
        // interval
        auto nb = get_block_num(dst_id);
        auto ni = get_sub_to_interval_num(w, h, d);
        if (ni != interval_num)
          continue; // can't add verts of other blocks
        if (nb != block_num)
          continue; // can't add verts of other blocks
        double factor =
            (offset == 1)
                ? 1.0
                : ((offset == 2) ? 1.414214 : ((offset == 3) ? 1.732051 : 0.0));
        parent_code = get_parent_code(dst_id, vid);
#ifdef FULL_PRINT
        cout << "w " << w << " h " << h << " d " << d << " dst_id " << dst_id
             << " vid " << vid << '\n';
#endif
        if (stage == "value") {
          accumulate_value(img, interval_num, dst_id, block_num, current,
                           revisits, factor, parent_code, tile_thresholds,
                           found_adjacent_invalid);
        } else if (stage == "radius") {
          if (found_adjacent_invalid)
            return;
          accumulate_radius(interval_num, dst_id, block_num, current, revisits,
                            factor, parent_code, found_adjacent_invalid);
          if (found_adjacent_invalid)
            return;
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
 * responsible for making any changes to activity state of interval or block.
 */
template <class image_t>
bool Recut<image_t>::integrate_vertex(VID_t interval_num, VID_t block_num,
                                      struct VertexAttr *updated_attr,
                                      bool ignore_KNOWN_NEW, bool is_root,
                                      std::string stage) {
  // get attr
  auto dst = get_attr_vid(interval_num, block_num, updated_attr->vid, nullptr);

  // handle simpler radii stage and exit
  if (stage == "radius") {
    if (dst->radius > updated_attr->radius) {
      this->surface_vec[interval_num][block_num].push_back(dst);
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
  // Only updates in the ghost region outside domain of block_num, in domain
  // of nb, therefore the updates must go into the heapvec of nb
  // to distinguish the internal handles used for either heaps
  // handles[block_num] is set for cells within block_num blocks internal
  // domain or block_num ghost cell region, it represents all cells added to
  // heap block_num and all cells that can be manipulated by thread block_num
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
      safe_update(heap_vec[interval_num][block_num], dst, updated_attr->value,
                  stage); // increase priority, lowers value in min-heap
#ifdef FULL_PRINT
      // cout << "\tiupdate: change in heap" << " value: " << dst->value << "
      // oldval " << old_val << '\n';
#endif
    } else {
      dst->value = updated_attr->value;
      safe_push(heap_vec[interval_num][block_num], dst, interval_num, block_num,
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
void Recut<image_t>::integrate_updated_ghost(VID_t interval_num,
                                             VID_t block_num,
                                             std::string stage) {
  VID_t tot_blocks = super_interval.GetNBlocks();
  for (VID_t nb = 0; nb < tot_blocks; nb++) {
    // active_neighbors[x][a][b] in domain of b, in ghost of block a
    if (active_neighbors[interval_num][block_num][nb]) {
      // updated ghost cleared in march_narrow_band
      // iterate over all ghost points of block_num inside domain of block nb
#ifdef CONCURRENT_MAP
      auto key = triple_pack_key(interval_num, nb, block_num);
      // get mutator so that doesn't have to reloaded when assigning
      auto mutator = updated_ghost_vec->find(key);
      std::vector<struct VertexAttr> *vec = mutator.getValue();
      // NULL if not found, continuing saves a redundant mutator assign
      if (!vec) {
        active_neighbors[interval_num][nb][block_num] = false; // reset to false
        continue; // FIXME this should never occur because of active_neighbor
                  // right?
      }

      for (struct VertexAttr updated_attr : *vec) {
#else
      for (struct VertexAttr updated_attr :
           updated_ghost_vec[interval_num][nb][block_num]) {
#endif

#ifdef FULL_PRINT
        cout << "integrate vid: " << updated_attr.vid
             << " ghost of block id: " << block_num
             << " in block domain of block id: " << nb << '\n';
#endif
        integrate_vertex(interval_num, block_num, &updated_attr, false, false,
                         stage);
      } // end for each VertexAttr
      active_neighbors[interval_num][nb][block_num] = false; // reset to false
      // clear sets for all possible block connections of block_num from this
      // iter
#ifdef CONCURRENT_MAP
      vec->clear();
      // assign directly to the mutator to save a lookup
      mutator.assignValue(
          vec); // keep the same pointer just make sure it's empty
#else
      updated_ghost_vec[interval_num][nb][block_num].clear();
#endif
    } // for all active neighbor blocks of block_num
  }
  if (stage == "value") {
    if (!heap_vec[interval_num][block_num].empty()) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_num << " block: " << block_num
           << " to active\n";
#endif
      active_blocks[interval_num][block_num].store(true);
    }
  } else if (stage == "radius") {
    if (!surface_vec[interval_num][block_num].empty()) {
#ifdef FULL_PRINT
      cout << "Setting interval: " << interval_num << " block: " << block_num
           << " to active\n";
#endif
      active_blocks[interval_num][block_num].store(true);
    }
  }
}

/*
 * If any interval is active return false, a interval is active if any of
 * its blocks are still active
 */
template <class image_t> bool Recut<image_t>::check_intervals_finish() {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Intervals active: ";
#endif
  for (auto interval_num = 0; interval_num < super_interval.GetNIntervals();
       ++interval_num) {
    if (super_interval.GetInterval(interval_num)->IsActive()) {
      tot_active++;
#ifdef LOG_FULL
      cout << interval_num << ", ";
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
 * If any block is active return false, a block is active if its
 * corresponding heap is not empty
 */
template <class image_t>
bool Recut<image_t>::check_blocks_finish(VID_t interval_num) {
  VID_t tot_active = 0;
#ifdef LOG_FULL
  cout << "Blocks active: ";
#endif
  for (auto block_num = 0; block_num < super_interval.GetNBlocks();
       ++block_num) {
    if (active_blocks[interval_num][block_num].load()) {
      tot_active++;
#ifdef LOG_FULL
      cout << block_num << ", ";
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
    const image_t *img, VID_t interval_num, VID_t block_num, std::string stage,
    const TileThresholds<image_t> *tile_thresholds) {

#ifdef LOG_FULL
  struct timespec time0, time1;
  VID_t visited = 0;
  VID_t start_size = heap_vec[interval_num][block_num].size();
  clock_gettime(CLOCK_REALTIME, &time0);
  if (!heap_vec[interval_num][block_num].empty()) {
    cout << "Start marching interval: " << interval_num
         << " block: " << block_num << '\n';
  }
#endif

  VID_t revisits = 0;
  VertexAttr *current;
  bool found_adjacent_invalid;
  if (stage == "radius") {
    while (!(this->surface_vec[interval_num][block_num].empty())) {
      // pointers must still be in memory
      VertexAttr *current = this->surface_vec[interval_num][block_num].front();
      this->surface_vec[interval_num][block_num].pop_front();
      // assertm(current != NULL, "pointers must still be in memory");
      // invalid can either be out of range of the entire global image or it
      // can be a background vertex which occurs due to pixel value below the
      // threshold
      found_adjacent_invalid = false;
      // FIXME these should be global
      auto z_pad_stride =
          static_cast<int64_t>(x_pad_block_size * y_pad_block_size);
      std::vector<int64_t> pad_strides{-1,
                                       1,
                                       -static_cast<int64_t>(x_pad_block_size),
                                       static_cast<int64_t>(x_pad_block_size),
                                       -z_pad_stride,
                                       z_pad_stride};
      auto z_stride = static_cast<int64_t>(x_block_size * y_block_size);
      std::vector<int64_t> strides{-1,
                                   1,
                                   -static_cast<int64_t>(x_block_size),
                                   static_cast<int64_t>(x_block_size),
                                   -z_stride,
                                   z_stride};
      VertexAttr *found_higher_parent = nullptr;
      VertexAttr *same_radius_adjacent = nullptr;
      VID_t i, j, k;
      i = j = k = 0;
      get_img_subscript(current->vid, i, j, k);
      cout << "\ncurrent->vid " << current->vid << " radius "
           << +(current->radius) << " i " << i << " j " << j << " k " << k
           << '\n';
      uint8_t updated_radius = 1 + current->radius;
      int64_t pad_stride, stride;
      VID_t expected_dst_vid;
      for (int idx = 0; idx < pad_strides.size(); idx++) {
        pad_stride = pad_strides[idx];
        stride = strides[idx];

        // note the current vertex can belong in the boundary
        // region of a separate block /interval and is only
        // within this block /interval's ghost region
        // therefore all neighbors / destinations of current
        // must be checked to make sure they protude into
        // the actual current block / interval region
        expected_dst_vid = current->vid + stride;
        get_img_subscript(expected_dst_vid, i, j, k);
        cout << "i " << i << " j " << j << " k " << k << " vid "
             << expected_dst_vid << '\n';
        auto dst_block = get_block_num(expected_dst_vid);
        auto dst_interval = get_interval_num(expected_dst_vid);

        // Filter all dsts that don't protude into current
        // block and interval region, ghost destinations
        // can not be added in to processing stack
        // ghost vertices can only be added in to the stack
        // during `integrate_updated_ghost()`
        if (dst_interval != interval_num) {
          continue; // can't add verts of other blocks
        }
        if (dst_block != block_num) {
          continue; // can't add verts of other blocks
        }

        // current vertex is not always within this block and interval
        // and each block, interval have a ghost region
        // after filter above this pointer arithmetic is always valid
        auto dst = current + pad_stride;
        cout << " radius " << +(dst->radius) << '\n';
        assertm(expected_dst_vid == dst->vid, "pointer arithmetic invalid");
        // all block_nums are a linear row-wise idx, relative to current
        // interval
        if (dst->selected()) {
          if (dst->radius == current->radius) {
            cout << "\tAdjacent same\n";
            // any match adjacent same match will do
            same_radius_adjacent = dst;
          } else if (!(dst->valid_radius()) || (dst->radius > updated_radius)) {
            // if radius has not been set yet
            // this necessitates it is 1 higher than current
            // OR an update from another block / interval
            // creates new lower updates
            cout << "\tAdjacent higher\n";
            found_higher_parent = dst;
            dst->radius = updated_radius;
            this->surface_vec[interval_num][block_num].push_back(dst);
            place_ghost_update(interval_num, block_num, dst, false, stage);
          }
        } else {
          cout << "\tUnselected\n";
        }
      }
      // set the proper parent, either a same or a higher adjacent
      // if (found_higher_parent && alsdkfj) {
      //}
      // you can only prune when the dst has a higher radii
      // and the current in question is
      // radii 2 or more and not a root
      // if ((current->radius >= 2) && !(current->root())) {
      //}
    }
  } else {
    while (!heap_vec[interval_num][block_num].empty()) {

      if (restart_bool) {
        struct VertexAttr *top =
            heap_vec[interval_num][block_num].top(); // peek at top position
        if (top->value > bound_band) {
          break;
        }
      }

#ifdef LOG_FULL
      visited += 1;
#endif

      // remove from this intervals heap
      struct VertexAttr *dummy_min = safe_pop<local_heap, struct VertexAttr *>(
          heap_vec[interval_num][block_num], block_num, interval_num, stage);

      // protect from dummy values added to the heap not inside
      // this block or interval
      // roots are also added to the heap in initialize to prevent
      // having to load the full intervals into memory at that time
      // this retrieves the actual ptr to the vid of the root dummy min
      // values so that they can be properly initialized
      current = get_attr_vid(interval_num, block_num, dummy_min->vid, nullptr);
      if (stage == "value") {

        // preserve state of roots
        if (!(dummy_min->root())) {
          current->mark_selected(); // set as KNOWN NEW
          assert(current->valid_vid());
          // Note: any adjustments to current if in a neighboring domain
          // i.e. current->handles.contains(nb) are not in a data race
          // with the nb thread. All VertexAttr have redundant copies in each
          // ghost region
        } else {
          current = dummy_min; // set as root and copy members
          assert(current->root());
          assertm(current->value == 0, "root value not set properly");
        }
      } else if (stage == "radius") {
        assertm(current->selected() || current->root(),
                "Can not process an invalid vertex");
        current->radius = dummy_min->radius;
        current->vid = dummy_min->vid;
      }

      // invalid can either be out of range of the entire global image or it
      // can be a background vertex which occurs due to pixel value below the
      // threshold
      found_adjacent_invalid = false;
      update_neighbors(img, interval_num, current, block_num, revisits, stage,
                       tile_thresholds, found_adjacent_invalid);
      if (found_adjacent_invalid) {
        if (stage == "value") {
#ifdef FULL_PRINT
          cout << "found surface vertex " << current->vid << '\n';
#endif
          current->radius = 1;
          assertm(current->selected(), "surface was not selected");
          this->surface_vec[interval_num][block_num].push_back(current);
        } else if (stage == "radius") {
          if (current->radius != 1) {
            // a vertex with an invalid adjacent (neighbor) always has a
            // radius of 1 put this vertex back for processing, in case it's
            // neighbors were improperly set before finding the invalid
            vertex_update(interval_num, block_num, current, 1, "radius");
            assertm(current->radius == 1, "radius was not properly set to 1");
          }
        }
      }
    }
  }

#ifdef LOG_FULL
  // if (visited > 0) {
  clock_gettime(CLOCK_REALTIME, &time1);
  cout << "Marched interval: " << interval_num << " block: " << block_num
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

  // save the last value to help start different stages
  // any last value is fine this does not need to be thread-safe
  if (stage == "value") {
#ifdef LOG_FULL
    cout << interval_num << ',' << block_num << current->description();
#endif
    assertm(current->selected() || current->root(),
            "Saving start, but vertex must be selected");
    super_interval.GetInterval(interval_num)->set_start_vertex(current->vid);
    super_interval.GetInterval(interval_num)->set_valid_start(true);
    // collect all border/surface vertices i.e. those that share at least one
    // adjacent edge with a background pixel
  }

  // Note: could set explicit memory ordering on atomic
  active_blocks[interval_num][block_num].store(false);
  processing_blocks[interval_num][block_num].store(
      false); // release block_num heap
}

// template <class image_t>
// void Recut<image_t>::create_march_thread(VID_t interval_num, VID_t
// block_num)
// {
//// process if block has been activated and is not currently being
//// processed by another task thread
// if (active_blocks[interval_num][block_num] &&
// !processing_blocks[interval_num][block_num]) {
////cout << "Start active block_num " << block_num << '\n';
// processing_blocks[interval_num][block_num] = true;
// thread(&Recut<image_t>::march_narrow_band, this, interval_num,
// block_num).detach();
////cout << "Ran through active block_num " << block_num << '\n';
//}
//}

// template <class image_t>
// void Recut<image_t>::create_integrate_thread(VID_t interval_num, VID_t
// block_num) { if
// (any_of(this->active_neighbors[interval_num][block_num].begin(),
// this->active_neighbors[interval_num][block_num].end(),
//[](bool i) {return i;})) {
//// Note: can optimize to taskflow if necessary
// thread(&Recut<image_t>::integrate_updated_ghost, this, interval_num,
// block_num).detach();
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
  i = id % x_interval_num;
  j = (id / x_interval_num) % y_interval_num;
  k = (id / (x_interval_num * y_interval_num)) % z_interval_num;
}

template <class image_t>
void Recut<image_t>::get_interval_offsets(const VID_t interval_num,
                                          vector<int> &interval_offsets,
                                          vector<int> &interval_extents) {
  VID_t i, j, k;
  get_interval_subscript(interval_num, i, j, k);
  vector<VID_t> subs = {k, j, i};
  interval_offsets = {0, 0, 0};
  interval_extents = {0, 0, 0};
  auto off = args->image_offsets();
  // increment the offset location to extract
  interval_offsets[2] +=
      off[2]; // args->image_offsets args->image_extents are in z y x order
  interval_offsets[1] += off[1];
  interval_offsets[0] += off[0];
  vector<int> szs = {(int)nz, (int)ny, (int)nx};
  for (int i = 0; i < 3; i++) {
    interval_offsets[i] += subs[i] * interval_sizes[i];
    // constrain the extents to actual image
    // global command line extents have already been factored
    // into szs
    interval_extents[i] =
        min(szs[i] - interval_offsets[i], (int)interval_sizes[i]);
  }
#ifdef LOG_FULL
  cout << "interval_num: " << interval_num;
  cout << " offset x " << interval_offsets[2] << " offset y "
       << interval_offsets[1] << " offset z " << interval_offsets[0] << '\n';
  cout << " extents x " << interval_extents[2] << " extents y "
       << interval_extents[1] << " extents z " << interval_extents[0] << '\n';
#endif
}

template <class image_t>
void Recut<image_t>::get_max_min(const image_t *img, VID_t interval_vert_num) {

  double elapsed_max_min = omp_get_wtime();

  // GI parameter min_int, max_int, li
  double local_max = 0; // maximum intensity, used in GI
  double local_min = std::numeric_limits<double>::max(); // max value
  //#pragma omp parallel for reduction(max:local_max)
  for (auto i = 0; i < interval_vert_num; i++) {
    if (img[i] > local_max) {
      local_max = img[i];
    }
  }
  //#pragma omp parallel for reduction(min:local_min)
  for (auto i = 0; i < interval_vert_num; i++) {
    if (img[i] < local_min) {
      local_min = img[i];
      // cout << "local_min" << +local_min << '\n';
    }
  }
  if (local_min == local_max) {
    cout << "Warning: max: " << local_max << "= min: " << local_min << '\n';
  } else if (local_min > local_max) {
    cout << "Error: max: " << local_max << "< min: " << local_min << '\n';
    throw;
  } else {
    local_max -= local_min;
  }
  this->max_int = local_max;
  this->min_int = local_min;

  elapsed_max_min = omp_get_wtime() - elapsed_max_min;

#ifdef LOG
  printf("Find max min wtime: %.1f s\n", elapsed_max_min);
#endif
}

// deprecated
template <class image_t>
int Recut<image_t>::thresh_pct(const image_t *img, VID_t interval_vert_num,
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
    for (VID_t i = 0; i < (VID_t)interval_vert_num; i++) {
      if (img[i] <= local_bkg_thresh) {
        bkg_count += 1;
      }
    }

    // Check if above desired percent background
    float test_pct = bkg_count / (double)interval_vert_num;
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
    VID_t interval_num, image_t *tile, std::string stage,
    const TileThresholds<image_t> *tile_thresholds) {

  struct timespec presave_time, postmarch_time, iter_start,
      start_iter_loop_time, end_iter_time, postsave_time;
  VID_t nblocks = super_interval.GetNBlocks();
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

    // if any active status for any block of interval_num is true
    while (any_of(this->active_blocks[interval_num].begin(),
                  this->active_blocks[interval_num].end(),
                  [](atomwrapper<bool> i) { return i.load(); })) {

#ifdef TF
      prevent_destruction.emplace_back(new tf::Taskflow());
      bool added_task = false;
#endif // TF

      for (VID_t block_num = 0; block_num < nblocks; ++block_num) {
        // if not currently processing, set atomically set to true and
        if (active_blocks[interval_num][block_num].load() &&
            processing_blocks[interval_num][block_num].compare_exchange_strong(
                false, true)) {
#ifdef LOG_FULL
          // cout << "Start active block_num " << block_num << '\n';
#endif

#ifdef TF
          // FIXME check passing tile ptr as ref
          prevent_destruction.back()->silent_emplace([=, &tile]() {
            march_narrow_band(tile, interval_num, block_num, stage,
                              tile_thresholds);
          });
          added_task = true;
#else
          async(launch::async, &Recut<tile>::march_narrow_band, this, tile,
                interval_num, block_num, stage, tile_thresholds);
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
    for (VID_t block_num = 0; block_num < nblocks; ++block_num) {
      march_narrow_band(tile, interval_num, block_num, stage, tile_thresholds);
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
    // for(VID_t block_num = 0;block_num<nblocks;++block_num)
    //{
    // create_integrate_thread(interval_num, block_num);
    //}
    //#else

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (VID_t block_num = 0; block_num < nblocks; ++block_num) {
      integrate_updated_ghost(interval_num, block_num, stage);
    }

    //#endif

#ifdef LOG_FULL
    clock_gettime(CLOCK_REALTIME, &end_iter_time);
    cout << "inner_iteration_idx " << inner_iteration_idx << " in "
         << diff_time(iter_start, end_iter_time) << " sec." << '\n';
#endif

    if (check_blocks_finish(interval_num)) {
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
    super_interval.GetInterval(interval_num)->SaveToDisk();

  clock_gettime(CLOCK_REALTIME, &postsave_time);

  no_io_time = diff_time(start_iter_loop_time, presave_time);
  // global_no_io_time += no_io_time;
#ifdef LOG_FULL
  cout << "Interval: " << interval_num << " (no I/O) within " << no_io_time
       << " sec." << '\n';
  if (!this->mmap_)
    cout << "Finished saving interval in "
         << diff_time(presave_time, postsave_time) << " sec." << '\n';
#endif

  super_interval.GetInterval(interval_num)->SetActive(false);
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
Recut<image_t>::load_tile(VID_t interval_num, mcp3d::MImage &mcp3d_tile) {
#ifdef LOG
  struct timespec start, image_load;
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  double max_int, min_int;
  image_t bkg_thresh;

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
    get_interval_offsets(interval_num, interval_offsets, interval_extents);
    // use unit strides only
    mcp3d::MImageBlock block(interval_offsets, interval_extents);
    mcp3d_tile.SelectView(block, args->resolution_level());
    mcp3d_tile.ReadData(true);
    // mcp3d_tile.ReadData(true, "quiet");
  } catch (...) {
    MCP3D_MESSAGE("error in mcp3d_tile io. neuron tracing not performed")
    throw;
  }

#ifdef LOG
  clock_gettime(CLOCK_REALTIME, &image_load);
  cout << "Load image in " << diff_time(start, image_load) << " sec." << '\n';
  // cout << "fg " << params->foreground_percent() << '\n';
#endif

  interval_dims = mcp3d_tile.loaded_view().view_xyz_dims();
  auto interval_vert_num =
      interval_dims[0] * (VID_t)interval_dims[1] * interval_dims[2];

  // assign thresholding value
  // foreground parameter takes priority
  // Note if either foreground or background percent is equal to or greater
  // than 0 than it was changed by a user so it takes precedence over the
  // defaults
  if (params->foreground_percent() >= 0) {
    bkg_thresh = mcp3d::TopPercentile<image_t>(mcp3d_tile.Volume<image_t>(0),
                                               interval_dims,
                                               params->foreground_percent());
  } else { // if bkg set explicitly and foreground wasn't
    if (params->background_thresh() >= 0) {
      bkg_thresh = params->background_thresh();
    } else {
      bkg_thresh = 0;
    }
  }

  // assign max and min ints for this tile
  if (this->args->recut_parameters().get_max_intensity() < 0) {
    get_max_min(mcp3d_tile.Volume<image_t>(0), interval_vert_num);
  } else if (this->args->recut_parameters().get_min_intensity() < 0) {
    if (bkg_thresh >= 0) {
      min_int = bkg_thresh;
    } else {
      get_max_min(mcp3d_tile.Volume<image_t>(0), interval_vert_num);
    }
  } else {
    // otherwise set global max min from recut_parameters
    max_int = this->args->recut_parameters().get_max_intensity();
    min_int = this->args->recut_parameters().get_min_intensity();
  }

#ifdef LOG_FULL
  cout << "max_int: " << +(max_int) << " min_int: " << +(min_int) << '\n';
  cout << "bkg_thresh value = " << bkg_thresh << '\n';
  cout << "interval dims x " << interval_dims[2] << " y " << interval_dims[1]
       << " z " << interval_dims[0] << '\n';
  cout << "interval offsets x " << interval_offsets[2] << " y "
       << interval_offsets[1] << " z " << interval_offsets[0] << '\n';
  cout << "interval extents x " << interval_extents[2] << " y "
       << interval_extents[1] << " z " << interval_extents[0] << '\n';
#endif
  // const TileThresholds<image_t> returnp{max_int, min_int, bkg_thresh};
  // return returnp;
  return new TileThresholds(max_int, min_int, bkg_thresh);
} // end load_tile()

#endif // only defined in USE_MCP3D is

template <class image_t> void Recut<image_t>::update(std::string stage) {
  // init all timers
  struct timespec update_start_time, update_finish_time;
  double global_no_io_time;
  global_no_io_time = 0.0;

  VID_t nintervals = super_interval.GetNIntervals();

#ifdef LOG
  cout << "Start updating stage " << stage << '\n';
#endif
  clock_gettime(CLOCK_REALTIME, &update_start_time);

  bound_band = 0; // for restart
  // VID_t final_inner_iter = 0;
  VID_t outer_iteration_idx = 0;
  VID_t interval_num = 0;

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
    if (super_interval.GetInterval(interval_num)->IsActive()) {

#ifdef LOG
      struct timespec interval_start, interval_load;
      clock_gettime(CLOCK_REALTIME, &interval_start);
#endif

      // only load the intervals that are not already mapped or have been read
      // already calling load when already present will throw
      if (!super_interval.GetInterval(interval_num)->IsInMemory())
        super_interval.GetInterval(interval_num)->LoadFromDisk();

#ifdef LOG
      clock_gettime(CLOCK_REALTIME, &interval_load);
      cout << "Load interval " << interval_num << " in "
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
      if (!(this->params->generate_image)) {
        // mcp3d_tile must be kept in scope during the processing
        // of this interval otherwise dangling reference then seg fault
        // on image access
        tile_thresholds = load_tile(interval_num, mcp3d_tile);
        // FIXME need a setup image for both
        // This all needs to be designed in a way that keeps image around
        cout << "before reassign to tile\n";
        tile = mcp3d_tile.Volume<image_t>(0);
      }
#else
      if (!(this->params->generate_image)) {
        assertm(false, "If USE_MCP3D macro is not set, "
                       "this->params->generate_image must be set to True");
      }
#endif

      global_no_io_time +=
          process_interval(interval_num, tile, stage, tile_thresholds);
    } // if the interval is active

    // rotate interval number until all finished
    interval_num = (interval_num + 1) % nintervals;
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

/* DEPRECATED
 */
template <class image_t>
VID_t Recut<image_t>::get_block_offset(const VID_t id, const VID_t offset) {
  VID_t i, j, k, raw_offset;
  raw_offset = id - offset;
  k = raw_offset / nxypad;
  raw_offset = raw_offset % nxypad;
  j = raw_offset / nxpad;
  i = raw_offset % x_block_size;
  // cout << "\tblock offset x " << i << " y " << j << " z " << k << '\n';
  return k * y_block_size * x_block_size + j * x_block_size + i;
}

/* get the vid with respect to the entire image passed to the
 * recut program. Note this spans multiple tiles and blocks
 * Take the subscripts of the vertex or voxel
 * returns the linear idx into the entire domain
 */
template <class image_t>
inline VID_t Recut<image_t>::get_img_vid(const VID_t i, const VID_t j,
                                         const VID_t k) {
  return k * nxy + j * nx + i;
}

/* DEPRECATED get the vid with respect to the entire padded image passed to
 * the recut program. Note this spans multiple tiles and blocks Take the
 * subscripts of the vertex or voxel returns the linear idx into the entire
 * domain after padding for blocks
 */
template <class image_t>
inline VID_t Recut<image_t>::get_vid(const VID_t i, const VID_t j,
                                     const VID_t k) {
  return k * nxypad + j * nxpad + i;
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
  i = id % nx_block;
  j = (id / nx_block) % ny_block;
  k = (id / (nx_block * ny_block)) % nz_block;
}

/*
 * Convert block subscripts into linear index row-ordered
 */
template <class image_t>
inline VID_t Recut<image_t>::get_block_id(const VID_t iblock,
                                          const VID_t jblock,
                                          const VID_t kblock) {
  return iblock + jblock * nx_block + kblock * nx_block * ny_block;
}

template <class image_t>
inline void Recut<image_t>::get_img_subscript(const VID_t id, VID_t &i,
                                              VID_t &j, VID_t &k) {
  i = id % nx;
  j = (id / nx) % ny;
  k = (id / nxy) % nz;
}

// Wrap-around rotate all values forward one
// This logic disentangles 0 % 32 from 32 % 32 results
template <class image_t>
inline VID_t Recut<image_t>::rotate_index(VID_t img_sub, const VID_t current,
                                          const VID_t neighbor,
                                          const VID_t block_size,
                                          const VID_t pad_block_size) {
  if (current == neighbor)
    return img_sub + 1; // adjust to padded block idx
  // if it's in another block/interval it can only be 1 vox away
  // so make sure the subscript itself is on the correct edge of its block
  // domain
  if (current == (neighbor + 1))
    assertm(img_sub == block_size - 1,
            "Does not currently support diagonal connections or any ghost "
            "regions greater that 1");
  return 0;
  if (current == (neighbor - 1))
    assertm(img_sub == 0, "Does not currently support diagonal connections or "
                          "any ghost regions greater that 1");
  return pad_block_size - 1;

  // failed
  assertm(false, "Does not currently support diagonal connections or any ghost "
                 "regions greater that 1");
}

/*
 * Returns a pointer to the VertexAttr within interval_num,
 * block_num, and img_vid (vid with respect to global image)
 * Note each block actually spans (block_size + 2) ^ 3
 * total vertices in memory, this is because each block needs
 * to hold a redundant copy of all border regions of its cube
 * border regions are also denoted as "ghost" cells/vertices
 * this creates complexity in that the requested vid passed
 * to this function may be referring to a VID that's within
 * the bounds of a separate block if one were to refer to the
 * block_size alone.
 */
template <class image_t>
inline VertexAttr *
Recut<image_t>::get_attr_vid(const VID_t interval_num, const VID_t block_num,
                             const VID_t img_vid, VID_t *output_offset) {
  VID_t i, j, k, img_block_i, img_block_j, img_block_k;
  VID_t pad_img_block_i, pad_img_block_j, pad_img_block_k;
  i = j = k = 0;

  Interval *interval = super_interval.GetInterval(interval_num);
  assert(interval->IsInMemory());
  VertexAttr *attr = interval->GetData(); // first vertex of entire interval

  // block start calculates starting vid of block_num's first vertex within
  // the global interval array of structs Note: every vertex within this block
  // (including the ghost region) will be contiguous between attr and attr +
  // (pad_block_offset - 1) stored in row-wise order with respect to the cubic
  // block blocks within the interval are always stored according to their
  // linear block num such that a block_num * the total number of padded
  // vertices in a block i.e. pad_block_offset or (block_size + 2) ^ 2 yields
  // offset to the offset to the first vertex of the block.
  VID_t block_start = pad_block_offset * block_num;
  auto first_block_attr = attr + block_start;

  // Find correct offset into block

  // first convert from img id to non- padded block subs
  get_img_subscript(img_vid, i, j, k);
  // in case interval_size isn't evenly divisible by block size
  // mod out any contributions from the interval
  auto ia = i % x_interval_size;
  auto ja = j % y_interval_size;
  auto ka = k % z_interval_size;
  // these are subscripts within the non-padded block domain
  // these values will be modified by rotate_index to account for padding
  img_block_i = ia % x_block_size;
  img_block_j = ja % y_block_size;
  img_block_k = ka % z_block_size;
  // cout << "\timg vid: "<< img_vid<< " img_block_i " << img_block_i << "
  // img_block_j " << img_block_j << " img_block_k " << img_block_k<<'\n';

  // which block domain and interval does this img_vid actually belong to
  // ignoring ghost regions denoted nb_* since it may belong in the domain of
  // a neighbors block or interval all block_nums are a linear row-wise idx,
  // relative to current interval
  int nb_block = (int)get_block_num(img_vid);
  int nb_interval = (int)get_interval_num(img_vid);

  // adjust block subscripts so they reflect the interval or block they belong
  // to also adjust based on actual 3D padding of block Rotate all values
  // forward one This logic disentangles 0 % 32 from 32 % 32 results within a
  // block, where ghost region is index -1 and block_size
  if (interval_num == nb_interval) { // common case first
    if (nb_block == block_num) {     // grab the second common case
      pad_img_block_i = img_block_i + 1;
      pad_img_block_j = img_block_j + 1;
      pad_img_block_k = img_block_k + 1;
    } else {
      VID_t iblock, jblock, kblock, nb_iblock, nb_jblock, nb_kblock;
      // the block_num is a linear index into the 3D row-wise arrangement of
      // blocks, converting to subscript makes adjustments easier
      get_block_subscript(block_num, iblock, jblock, kblock);
      get_block_subscript(nb_block, nb_iblock, nb_jblock, nb_kblock);
      assertm(absdiff(iblock, nb_iblock) + absdiff(jblock, nb_jblock) +
                      absdiff(kblock, nb_kblock) ==
                  1,
              "Does not currently support diagonal connections or any ghost "
              "regions greater that 1");
      pad_img_block_i = rotate_index(img_block_i, iblock, nb_iblock,
                                     x_block_size, x_pad_block_size);
      pad_img_block_j = rotate_index(img_block_j, jblock, nb_jblock,
                                     y_block_size, y_pad_block_size);
      pad_img_block_k = rotate_index(img_block_k, kblock, nb_kblock,
                                     z_block_size, z_pad_block_size);
    }
  } else { // ignore block info, adjust based on interval
    // the interval_num is also linear index into the 3D row-wise arrangement
    // of intervals, converting to subscript makes adjustments easier
    VID_t iinterval, jinterval, kinterval, nb_iinterval, nb_jinterval,
        nb_kinterval;
    get_interval_subscript(interval_num, iinterval, jinterval, kinterval);
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
    // over note that all block nums are relative to their interval so this
    // requires a bit more logic to check, even when converting to subs
#ifndef NDEBUG
    VID_t iblock, jblock, kblock, nb_iblock, nb_jblock, nb_kblock;
    get_block_subscript(block_num, iblock, jblock, kblock);
    get_block_subscript(nb_block, nb_iblock, nb_jblock, nb_kblock);
    // overload rotate_index simply for the assert checks
    rotate_index(nb_iblock, iinterval, nb_iinterval, nx_block,
                 x_pad_block_size);
    rotate_index(nb_jblock, jinterval, nb_jinterval, ny_block,
                 y_pad_block_size);
    rotate_index(nb_kblock, kinterval, nb_kinterval, nz_block,
                 z_pad_block_size);
    // cout << "\t\tiblock " << iblock << " nb_iblock " << nb_iblock << '\n';
    // cout << "\t\tjblock " << jblock << " nb_jblock " << nb_jblock << '\n';
    // cout << "\t\tkblock " << kblock << " nb_kblock " << nb_kblock << '\n';
    assertm(absdiff(iblock, nb_iblock) + absdiff(jblock, nb_jblock) +
                    absdiff(kblock, nb_kblock) <=
                1,
            "Does not currently support diagonal connections or any ghost "
            "regions greater that 1");
#endif
    // checked by rotate that subscript is 1 away
    pad_img_block_i = rotate_index(img_block_i, iinterval, nb_iinterval,
                                   x_block_size, x_pad_block_size);
    pad_img_block_j = rotate_index(img_block_j, jinterval, nb_jinterval,
                                   y_block_size, y_pad_block_size);
    pad_img_block_k = rotate_index(img_block_k, kinterval, nb_kinterval,
                                   z_block_size, z_pad_block_size);
  }

  // offset with respect to the padded block
  auto offset = pad_img_block_i + x_pad_block_size * pad_img_block_j +
                pad_img_block_k * x_pad_block_size * y_pad_block_size;
  assert(offset < pad_block_offset); // no valid offset is beyond this val

  if (output_offset)
    *output_offset = offset;

  VertexAttr *match = first_block_attr + offset;
#ifdef FULL_PRINT
  // cout << "\t\tget attr vid for img vid: "<< img_vid<< " pad_img_block_i "
  // << pad_img_block_i << " pad_img_block_j " << pad_img_block_j << "
  // pad_img_block_k " << pad_img_block_k<<'\n';
  ////cout << "\t\ti " << i << " j " << j << " k " << k<<'\n';
  ////cout << "\t\tia " << ia << " ja " << ja << " ka " << k<<'\n';
  // cout << "\t\tblock_num " << block_num << " nb_block " << nb_block << "
  // interval num " << interval_num << " nb_interval num " << nb_interval <<
  // '\n';;; cout << "\t\toffset " << offset << " block_start " << block_start
  // <<
  // '\n'; cout << "\t\ttotal interval size " << interval->GetNVertices() <<
  // '\n'; assert(block_start + offset < interval->GetNVertices()); // no
  // valid offset is beyond this val cout << "\t\tmatch-vid " << match->vid <<
  // " match->value " << match->value << '\n' << '\n';
#endif
  return match;
}

/* DEPRECATED
 */
template <class image_t>
inline VertexAttr *Recut<image_t>::get_attr(VID_t interval_num, VID_t block_num,
                                            VID_t ii, VID_t jj, VID_t kk) {
  Interval *interval = super_interval.GetInterval(interval_num);
  struct VertexAttr *attr = interval->GetData(); // first location of block
  VID_t block_start = pad_block_offset * block_num;
  auto offset =
      ii + x_pad_block_size * jj + kk * x_pad_block_size * y_pad_block_size;
  return attr + block_start + offset;
}

template <class image_t>
void Recut<image_t>::initialize_globals(const VID_t &nintervals,
                                        const VID_t &nblocks) {

  this->heap_vec.reserve(nintervals);
  for (int i = 0; i < nintervals; i++) {
    vector<local_heap> inner_vec;
    this->heap_vec.reserve(nblocks);
    for (int j = 0; j < nblocks; j++) {
      local_heap test;
      inner_vec.push_back(test);
    }
    this->heap_vec.push_back(inner_vec);
  }

#ifdef LOG_FULL
  cout << "Created global heap_vec" << '\n';
#endif

  this->surface_vec = std::vector<std::vector<std::deque<VertexAttr *>>>(
      nintervals, std::vector<std::deque<VertexAttr *>>(
                      nblocks, std::deque<VertexAttr *>()));

#ifdef LOG_FULL
  cout << "Created global surface_vec" << '\n';
#endif

  // active boolean for in interval domain in block_num ghost region, in
  // domain of block
  this->active_neighbors = vector<vector<vector<bool>>>(
      nintervals, vector<vector<bool>>(nblocks, vector<bool>(nblocks)));

#ifdef LOG_FULL
  cout << "Created active neighbors" << '\n';
#endif

#ifdef CONCURRENT_MAP
  updated_ghost_vec = std::make_unique<ConcurrentMap64>();
#else
  updated_ghost_vec = vector<vector<vector<vector<struct VertexAttr>>>>(
      nintervals, vector<vector<vector<struct VertexAttr>>>(
                      nblocks, vector<vector<struct VertexAttr>>(
                                   nblocks, vector<struct VertexAttr>())));
#endif

#ifdef LOG_FULL
  cout << "Created updated ghost vec" << '\n';
#endif

  // Initialize.
  vector<vector<atomwrapper<bool>>> temp(nintervals,
                                         vector<atomwrapper<bool>>(nblocks));
  for (auto interval = 0; interval < nintervals; ++interval) {
    vector<atomwrapper<bool>> inner(nblocks);
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
  vector<vector<atomwrapper<bool>>> temp2(nintervals,
                                          vector<atomwrapper<bool>>(nblocks));
  for (auto interval = 0; interval < nintervals; ++interval) {
    vector<atomwrapper<bool>> inner(nblocks);
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

template <class image_t> void Recut<image_t>::initialize() {

#ifdef USE_OMP
  omp_set_num_threads(params->user_thread_count());
  cout << "User specific thread count " << params->user_thread_count() << '\n';
#endif

  struct timespec time0, time1, time2, time3;
  uint64_t root_64bit;

  // for generated image runs trust the args->image_extents
  // to reflect the total global image domain
  // see get_args() in utils.hpp
  auto global_image_dims = args->image_extents();
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

  // these 3 are in z y x order
  vector<int> off;
  vector<int> ext;
  vector<int> end;

  // account for image_offsets and args->image_extents()
  off = args->image_offsets(); // default start is {0, 0, 0} for full image
  // image_extents is set to grid_size for generate_image option, otherwise
  // 0,0,0
  ext = args->image_extents(); // default is {0, 0, 0} for full image
  // protect faulty out of bounds input if extents goes beyond
  // domain of full image, note: z, y, x order
  auto endx = ext[2] ? min(off[2] + ext[2], global_image_dims[2])
                     : global_image_dims[2];
  auto endy = ext[1] ? min(off[1] + ext[1], global_image_dims[1])
                     : global_image_dims[1];
  auto endz = ext[0] ? min(off[0] + ext[0], global_image_dims[0])
                     : global_image_dims[0];
  end = {endz, endy, endx}; // sanitized end pixel in each dimension
  // protect faulty offset values
  assert(endx - off[2] > 0);
  assert(endy - off[1] > 0);
  assert(endz - off[0] > 0);
  // save to globals the actual size of the full image
  // accounting for the input offsets and extents
  // these will be used throughout the rest of the program
  this->nx = endx - off[2];
  this->ny = endy - off[1];
  this->nz = endz - off[0];
  this->nxy = nx * ny;
  this->img_vox_num = nx * ny * nz;

  // the image size and offsets override the user inputted interval size
  // continuous id's are the same for src or dst intervals
  // round up (pad)
  // Determine the size of each interval in each dim
  // constrict so less data is allocated especially in z dimension
  x_interval_size = min((VID_t)params->interval_size(), nx);
  y_interval_size = min((VID_t)params->interval_size(), ny);
  z_interval_size = min((VID_t)params->interval_size(), nz);
  interval_sizes = {z_interval_size, y_interval_size, x_interval_size};
  // determinze the number of intervals in each dim
  x_interval_num = (nx + x_interval_size - 1) / x_interval_size;
  y_interval_num = (ny + y_interval_size - 1) / y_interval_size;
  z_interval_num = (nz + z_interval_size - 1) / z_interval_size;

  // the resulting interval size override the user inputted block size
  x_block_size = min(x_interval_size, user_def_block_size);
  y_block_size = min(y_interval_size, user_def_block_size);
  z_block_size = min(z_interval_size, user_def_block_size);

  // determine number of blocks in one interval in each dim
  nx_block = (x_interval_size + x_block_size - 1) / x_block_size;
  ny_block = (y_interval_size + y_block_size - 1) / y_block_size;
  nz_block = (z_interval_size + z_block_size - 1) / z_block_size;
  nxpad = nx_block * x_block_size;
  nypad = ny_block * y_block_size;
  nzpad = nz_block * z_block_size;
  nxypad = nxpad * nypad; // saves recomputation occasionally

  const VID_t nintervals = x_interval_num * y_interval_num * z_interval_num;
  const VID_t nblocks = nx_block * ny_block * nz_block;
  x_pad_block_size = x_block_size + 2;
  y_pad_block_size = y_block_size + 2;
  z_pad_block_size = z_block_size + 2;
  pad_block_offset = x_pad_block_size * y_pad_block_size * z_pad_block_size;
  const VID_t nvid = pad_block_offset * nblocks * nintervals;
  assertm(pad_block_offset * nblocks < MAX_INTERVAL_VERTICES,
          "Total vertices used by an interval can not exceed "
          "MAX_INTERVAL_VERTICES specified in vertex_attr.h");

#ifdef LOG
  cout << "block x, y, z size: " << x_block_size << ", " << y_block_size << ", "
       << z_block_size << " interval x, y, z size " << x_interval_size << ", "
       << y_interval_size << ", " << z_interval_size
       << " intervals: " << nintervals << " blocks per interval: " << nblocks
       << '\n';
  cout << "nx: " << nx << " ny: " << ny << " nz: " << nz << '\n';
  cout << "nxblock: " << nx_block << " nyblock: " << ny_block
       << " nzblock: " << nz_block << '\n';
  cout << "nxpad: " << nxpad << " nypad: " << nypad << " nzpad: " << nzpad
       << " nxypad " << nxypad << '\n';
  // cout<< "image_offsets_x: "<< image_offsets[2] <<" image_offsets_y: "<<
  // image_offsets[1] <<" image_offsets_z: "<< image_offsets[0] << '\n';
#endif

  // we cast the interval num and block nums to uint16 for use as a key
  // in the global variables maps, if total intervals or blocks exceed this
  // there would be overflow
  if (nintervals > (2 << 16) - 1) {
    cout << "Number of intervals too high: " << nintervals
         << " try increasing interval size";
    assert(false);
  }
  if (nintervals > (2 << 16) - 1) {
    cout << "Number of blocks too high: " << nblocks
         << " try increasing block size";
    assert(false);
  }
  if (nvid > MAX_INTERVAL_VERTICES) {
    cout << "Number of total vertices too high: " << nvid
         << " current max at: " << MAX_INTERVAL_VERTICES
         << " try increasing MAX_INTERVAL_BASE and rerunning interval base "
            "generation in recut_test.hpp:CreateIntervalBase";
    assert(false);
  }

  clock_gettime(CLOCK_REALTIME, &time0);
  // Create SuperInterval
  super_interval = SuperInterval(nvid, nblocks, nintervals, *this, this->mmap_);

  clock_gettime(CLOCK_REALTIME, &time2);

#ifdef LOG
  cout << "Created super interval in " << diff_time(time0, time2) << " s";
  cout << " with total intervals: " << nintervals << '\n';
#endif

  initialize_globals(nintervals, nblocks);

  clock_gettime(CLOCK_REALTIME, &time3);

#ifdef LOG
  cout << "Initialized globals" << diff_time(time2, time3) << '\n';
#endif

  if (this->params->generate_image) {
    const TileThresholds<image_t> tile_thresholds{1, 0, 0};
    // This is where we set image to our desired values
    this->generated_image = new image_t[this->img_vox_num];
    // add the single root vid to the global state
    this->root_vids = {this->params->root_vid};

    assertm(root_vids.size() == 1,
            "Can only support 1 marker (root) at this time");
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
    auto selected =
        get_grid(this->params->tcase, this->generated_image, this->nx);
    if (this->params->tcase == 4)
      mesh_grid(this->root_vids[0], this->generated_image,
                this->params->selected, this->nx);
    else {
      this->params->selected = selected;
    }

  } else {
    // adds all markers to this->root_vids
    process_marker_dir(off, end);
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
    for (size_t interval_num = 0; interval_num < super_interval.GetNIntervals();
         ++interval_num) {
      Interval *interval = super_interval.GetInterval(interval_num);
      if (!this->mmap_)
        super_interval.GetInterval(interval_num)->LoadFromDisk();
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
        super_interval.GetInterval(interval_num)->Release();
    }
  } else if (is_same<vertex_t, MyMarker *>::value) {
#ifdef LOG
    cout << "Using MyMarker* type outtree" << '\n';
#endif
    // FIXME terrible performance
    map<VID_t, MyMarker *> tmp; // hash set
    // create all valid new marker objects
    for (size_t interval_num = 0; interval_num < super_interval.GetNIntervals();
         ++interval_num) {
      Interval *interval = super_interval.GetInterval(interval_num);
      if (this->mmap_) {
        if (!super_interval.GetInterval(interval_num)->IsInMemory())
          continue;
      } else {
        super_interval.GetInterval(interval_num)->LoadFromDisk();
      }

      struct VertexAttr *start = interval->GetData();
      // cout << "nvertices " << interval->GetNVertices() << '\n';
      for (VID_t offset = 0; offset < interval->GetNVertices(); offset++) {
        auto attr = start + offset;
        // only KNOWN_ROOT and KNOWN_NEW pass through this
        // KNOWN_ROOT preserved 0000 0000 and created
        // if not selected 0 and 0, skip
#ifdef FULL_PRINT
        // cout << "checking attr " << attr->vid << '\n';
#endif
        if (attr->unselected())
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
        super_interval.GetInterval(interval_num)->Release();
#ifdef LOG_FULL
      cout << "Total marker size : " << outtree.size() << " after interval "
           << interval_num << '\n';
#endif
    }
    // iterate through all possible, to assign parents correct pointer of
    // MyMarker
    for (size_t interval_num = 0; interval_num < super_interval.GetNIntervals();
         ++interval_num) {
      Interval *interval = super_interval.GetInterval(interval_num);
      if (this->mmap_) {
        if (!super_interval.GetInterval(interval_num)->IsInMemory())
          continue;
      } else {
        super_interval.GetInterval(interval_num)->LoadFromDisk();
      }
      for (struct VertexAttr *attr = interval->GetData();
           attr < interval->GetData() + interval->GetNVertices(); attr++) {
        // if not selected 0 and 0, skip
        if (attr->unselected())
          continue;
        // different copies have same values other than handle
        auto connects = attr->connections(nx, ny);
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
      super_interval.GetInterval(interval_num)->Release();
    }
  } else {
    MCP3D_INVALID_ARGUMENT("Outtree type passed not recognized. Supports "
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
// void Recut<image_t>::run_pipeline(std::vector<string> stages={"all"}) {
void Recut<image_t>::run_pipeline() {
  this->initialize();

  this->setup_value();
  this->update("value");

  this->setup_radius();
  this->update("radius");

  // this->setup_prune();
  // this->update("prune");
  this->finalize(this->args->output_tree);
}
