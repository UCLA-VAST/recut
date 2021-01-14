#pragma once

#include <atomic>
#include <numeric>
#include "markers.h"
#include "recut_parameters.hpp"
#include <algorithm> //min
#include <chrono>
#include <cstdlib> //rand srand
#include <ctime>   // for srand
#include <filesystem>
#include <math.h>
#include "range/v3/all.hpp"

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
#define assertm(exp, msg) assert(((void)msg, exp))
// be able to change pp values into std::string
#define XSTR(x) STR(x)
#define STR(x) #x

// the MyMarker operator< provided by library doesn't work
static const auto lt = [](const MyMarker* lhs, const MyMarker* rhs) {
  return std::tie(lhs->z, lhs->y, lhs->x) < std::tie(rhs->z, rhs->y, rhs->x);
};

static const auto eq = [](const MyMarker* lhs, const MyMarker* rhs) {
  //return *lhs == *rhs;
  //std::cout << "eq lhs " << lhs->description(2,2) << " rhs " << rhs->description(2,2) << '\n';
  return lhs->x == rhs->x && lhs->y == rhs->y && lhs->z == rhs->z;
};

// composing with pipe '|' is possible with actions and views
// passing a container to an action must be done by using std::move
const auto unique_count = [](std::vector<MyMarker*> v) {
  return rng::distance(std::move(v) | rng::actions::sort(lt) | rng::actions::unique(eq));
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

std::string get_data_dir() {
  return CMAKE_INSTALL_DATADIR;
}

VID_t get_central_sub(int grid_size) {
  return grid_size / 2 - 1; // place at center
}

VID_t get_central_vid(int grid_size) {
  auto sub = get_central_sub(grid_size);
  auto root_vid = (VID_t)sub * grid_size * grid_size + sub * grid_size + sub;
  return root_vid; // place at center
}

VID_t get_central_diag_vid(int grid_size) {
  auto sub = get_central_sub(grid_size);
  sub++; // add 1 to all x, y, z
  auto root_vid = (VID_t)sub * grid_size * grid_size + sub * grid_size + sub;
  return root_vid; // place at center diag
}

MyMarker *get_central_root(int grid_size) {
  VID_t x, y, z;
  x = y = z = get_central_sub(grid_size); // place at center
  auto root = new MyMarker();
  root->x = x;
  root->y = y;
  root->z = z;
  return root;
}

void get_img_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k,
    VID_t grid_size) {
  i = id % grid_size;
  j = (id / grid_size) % grid_size;
  k = (id / (grid_size * grid_size)) % grid_size;
}

std::vector<MyMarker *> vids_to_markers(std::vector<VID_t> vids,
    VID_t grid_size) {
  std::vector<MyMarker *> markers;
  for (const auto &vid : vids) {
    VID_t x, y, z;
    get_img_subscript(vid, x, y, z, grid_size);
    auto root = new MyMarker();
    root->x = x;
    root->y = y;
    root->z = z;
    markers.push_back(root);
  }
  return markers;
}

/* interval_size parameter is actually irrelevant due to
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

// interval_extents are in z, y, x order
template <typename T>
void print_marker_3D(T markers, std::vector<int> interval_extents,
    std::string stage) {
  for (int zi = 0; zi < interval_extents[0]; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * interval_extents[2] + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < interval_extents[1]; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < interval_extents[2]; xi++) {
        VID_t index = ((VID_t)xi) + yi * interval_extents[2] +
          zi * interval_extents[1] * interval_extents[2];
        //auto match = std::find(markers.begin(), markers.end(), index);
        //auto match = markers | find_if
        auto value = std::string{"-"};
        for (const auto& m : markers) {
          // z,y,x order!
          // this is xdim, ydim
          if (m->vid(interval_extents[2], interval_extents[1]) == index) {
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
}

// interval_extents are in z, y, x order
template <typename T>
void print_image_3D(T *inimg1d, std::vector<int> interval_extents) {
  for (int zi = 0; zi < interval_extents[0]; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * interval_extents[2] + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < interval_extents[1]; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < interval_extents[2]; xi++) {
        VID_t index = ((VID_t)xi) + yi * interval_extents[2] +
          zi * interval_extents[1] * interval_extents[2];
        // cout << i << " " ;
        cout << +inimg1d[index] << " ";
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
VID_t trace_mesh_image(VID_t id, uint16_t *inimg1d, const VID_t desired_selected, int grid_size) {
  VID_t i, j, k, ic, jc, kc;
  i = j = k = ic = kc = jc = 0;
  // set root to 1
  inimg1d[id] = 1;
  VID_t actual = 1; // count root
  srand(time(NULL));
  while (actual < desired_selected) {
    // calc i, j, k subs
    i = id % grid_size;
    j = (id / grid_size) % grid_size;
    k = (id / (grid_size * grid_size)) % grid_size;
    // std::cout << "previous " << id << " i " << i << " j " << j << " k " << k
    // << '\n';

    // try to find a suitable next location in one of the
    // six directions, if a proper direction was found
    // update the id and the subscripts to reflect
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
    get_img_subscript(id, ic, jc, kc, grid_size);
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

bool is_covered_by_parent(VID_t index, VID_t root_vid, int radius, VID_t grid_size) {
  VID_t i,j,k,pi,pj,pk;
  get_img_subscript(index, i, j, k, grid_size);
  get_img_subscript(root_vid, pi, pj, pk, grid_size);
  auto x = static_cast<double>(i) - pi;
  auto y = static_cast<double>(j) - pj;
  auto z = static_cast<double>(k) - pk;
  auto vdistance = sqrt(x * x + y * y + z * z);

  if (static_cast<double>(radius) >= vdistance) {
    return true;
  }
  return false;
}

/*
 * sets all to 1 for tcase 0
 * tcase4 : trace_mesh_image
 * tcase5 : sphere grid
 * tcase6 : reserved for real images throws error if passed
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
  double tcase2_factor = 2*PI;

  assertm(grid_size / 2 >= radius,
      "Can't specify a radius larger than grid_size / 2");
  auto root_x = static_cast<int>(get_central_sub(grid_size));
  auto root_y = static_cast<int>(get_central_sub(grid_size));
  auto root_z = static_cast<int>(get_central_sub(grid_size));
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
            sin(tcase2_factor * y) * sin(tcase2_factor * z);
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
  // must count total selected for all tcase 4 and above since it may not match desired
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
  assertm(false, "tcase not recognized: tcase 6 is reserved for reading real images\n tcase higher than 7 not specified");
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
  get_img_subscript(start, i, j, k, grid_size);
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

RecutCommandLineArgs get_args(int grid_size, int interval_size,
    int block_size, int slt_pct, int tcase,
    bool force_regenerate_image = false) {

  bool print = false;

  RecutCommandLineArgs args;
  auto params = args.recut_parameters();
  auto str_path = get_data_dir();
  params.set_marker_file_path(
      str_path + "/test_markers/" + std::to_string(grid_size) + "/tcase" +
      std::to_string(tcase) + "/slt_pct" + std::to_string(slt_pct) + "/");

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
    // zyx
    args.set_image_offsets({110, 229, 57});
    args.set_image_extents({grid_size, grid_size, grid_size});

    if (const char* env_p = std::getenv("TEST_IMAGE")) {
      std::cout << "Using $TEST_IMAGE environment variable: " << env_p << '\n';
      args.set_image_root_dir(std::string(env_p));
    } else {
      std::cout << "Warning likely fatal: must run: export TEST_IMAGE=\"abs/path/to/image\" to set the environment variable\n\n";
    }

    if (const char* env_p = std::getenv("TEST_MARKER")) {
      std::cout << "Using $TEST_MARKER environment variable: " << env_p << '\n';
      params.set_marker_file_path(std::string(env_p));
    } else {
      std::cout << "Warning likely fatal must run: export TEST_MARKER=\"abs/path/to/marker\" to set the environment variable\n\n";
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
    args.set_image_root_dir(
        str_path + "/test_images/" + std::to_string(grid_size) + "/tcase" +
        std::to_string(tcase) + "/slt_pct" + std::to_string(slt_pct));
  }

  // the total number of blocks allows more parallelism
  // ideally intervals >> thread count
  params.set_interval_size(interval_size);
  params.set_block_size(block_size);
  VID_t img_vox_num = grid_size * grid_size * grid_size;
  params.tcase = tcase;
  params.slt_pct = slt_pct;
  params.selected = img_vox_num * (slt_pct / (float)100);
  params.root_vid = get_central_vid(grid_size);
  params.set_user_thread_count(1);
#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
  params.set_user_thread_count(omp_get_max_threads());
#endif
  std::vector<int> extents = {grid_size, grid_size, grid_size};
  args.set_image_extents(extents);

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
    fn = fn + "marker";
    std::ofstream mf;
    mf.open(fn);
    mf << "# x,y,z\n";
    mf << x << "," << y << "," << z;
    mf.close();
    if (print)
      cout << "      Wrote marker: " << fn << '\n';
  }
}

#ifdef USE_MCP3D
void write_tiff(uint16_t *inimg1d, std::string base, int grid_size) {
  auto print = false;
#ifdef LOG
  print = true;
#endif

  bool rerun = false;
  if (!fs::exists(base) || rerun) {
    fs::remove_all(base); // make sure it's an overwrite
    if (print)
      cout << "      Delete old: " << base << '\n';
    fs::create_directories(base);
    // print_image(inimg1d, grid_size * grid_size * grid_size);
    for (int zi = 0; zi < grid_size; zi++) {
      std::string fn = base;
      fn = fn + "img_";
      // fn = fn + mcp3d::PadNumStr(zi, 9999); // pad to 4 digits
      fn = fn + std::to_string(zi); // pad to 4 digits
      std::string suff = ".tif";
      fn = fn + suff;
      VID_t start = zi * grid_size * grid_size;
      // cout << "fn: " << fn << " start: " << start << '\n';
      // print_image(&(inimg1d[start]), grid_size * grid_size);

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
      cout << "      Wrote test images at: " << base << '\n';
  }
}

void read_tiff(std::string fn, std::vector<int> image_offsets,
    std::vector<int> image_extents, mcp3d::MImage& image) {
  cout << "Read: " << fn << '\n';
  // read data from channel 0
  image.ReadImageInfo(0, true);
  try {
    // use unit strides only
    mcp3d::MImageBlock block(image_offsets, image_extents);
    image.SelectView(block, 0);
    image.ReadData(true, "quiet");
  } catch (...) {
    assertm(false, "error in image io. neuron tracing not performed");
  }
}
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

#ifdef USE_MMAP
  cout << "USE_MMAP" << '\n';
#endif

#ifdef ASYNC
  cout << "ASYNC" << '\n';
#endif

#ifdef TF
  cout << "TF" << '\n';
#endif

#ifdef CONCURRENT_MAP
  cout << "CONCURRENT_MAP" << '\n';
#endif

#ifdef USE_HUGE_PAGE
  cout << "USE_HUGE_PAGE" << '\n';
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

// use in conjunction with clock_gettime
inline double diff_time(struct timespec time1, struct timespec time2) {
  return time2.tv_sec - time1.tv_sec + (time2.tv_nsec - time1.tv_nsec) * 1e-9;
}

template<typename T>
struct CompareResults {
  T false_negatives;
  T false_positives;
  VID_t duplicate_count;
  VID_t match_count;

  CompareResults(T false_negatives, T false_positives, VID_t duplicate_count, VID_t match_count) :
    false_negatives(false_negatives), false_positives(false_positives),
    duplicate_count(duplicate_count) , match_count(match_count) {}

};

auto print = [](auto iter){
  rng::for_each(iter,
      [](auto i) { std::cout << i << ", "; });
  std::cout << '\n';
};

auto get_vids = [](auto tree, auto xdim, auto ydim) {
  return tree
    | rng::views::transform([&](auto v) {
        return v->vid(xdim, ydim); })
    | rng::to_vector;
};

auto get_vids_sorted = [](auto tree, auto xdim, auto ydim) {
  return get_vids(tree, xdim, ydim) | rng::action::sort;
};

// prints mismatches between two trees in uid sorted order
template <typename T, typename T2, typename T3>
auto compare_tree(T truth_tree, T check_tree, T2 xdim, T2 ydim, T3& recut) {

  bool print = false;
#ifdef LOG
  print = true;
#endif

  if (print)
    std::cout << "compare tree\n";
  // duplicate_count will be asserted to == 0 at caller
  VID_t duplicate_count = 0;
  duplicate_count += truth_tree.size() - unique_count(truth_tree);
  duplicate_count += check_tree.size() - unique_count(check_tree);

  auto truth_vids = get_vids_sorted(truth_tree, xdim, ydim);
  auto check_vids = get_vids_sorted(check_tree, xdim, ydim);

  if (print) {
    //std::cout << "truth_vids\n";
    //print(truth_vids);
    //std::cout << "check_vids\n";
    //print(check_vids);
  }

  auto matches = rng::views::set_intersection( truth_vids, check_vids);

  //for (auto&& vid : matches) {
  //const auto block = recut.get_block_id(vid);
  //std::cout <<  "match"<< " at: " << vid << " block: " << block << '\n';
  //}

  VID_t match_count = rng::distance(matches);
  if (print)
    std::cout <<  "match count: "<< match_count<< '\n';

  // move all this logic into the function body of tests
  auto get_negatives = [&](auto& tree, auto& matches, auto specifier) {
    return rng::views::set_difference( tree, matches)
      | rng::views::transform([&](auto vid) {
          const auto block = recut.get_block_id(vid);
          if (print)
          std::cout << "false " << specifier << " at: " << vid <<" block: " << block <<  '\n';
          return std::make_pair(vid, block);
          })
    | rng::to_vector ;
  };

  return new CompareResults<std::vector<std::pair<VID_t, VID_t>>>(
      get_negatives(truth_vids, matches, "negative"),
      get_negatives(check_vids, matches, "positive"),
      duplicate_count, match_count);
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

// avoid a dependency on boost.accumulators
// return mean of iterable and sample standard deviation
template <typename T>
std::tuple<double, double, double> iter_stats(T v) {

  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();
  double stdev;

  if (v.size() == 1) {
    stdev = 0.;
  } else {
    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

    // the mean is calculated from this data set the population mean
    // is unknown, therefore use the sample stdev (n - 1)
    stdev = std::sqrt(sq_sum / (v.size() - 1));
  }

  return {mean, sum, stdev};
}

template <typename I>
std::ostream open_swc_outputs(I root_vids) {
  std::ofstream out("out.swc");
  return out;
}

auto get_img_vid = [](const VID_t i, const VID_t j,
    const VID_t k,
    const VID_t image_length_x, const VID_t image_length_y) -> VID_t {
  auto image_length_xy = image_length_x * image_length_y;
  return k * image_length_xy + j * image_length_x + i;
};

// see DILATION_FACTOR definition or accumulate_prune() implementation
template <typename T, typename T2>
void create_coverage_mask_accurate(std::vector<MyMarker*>& markers, T* mask,
    T2 sz0, T2 sz1, T2 sz2) {
  assertm(sz0 == sz1, "must matching sizes for now");
  assertm(sz1 == sz2, "must matching sizes for now");
  auto sz01 = static_cast<VID_t>( sz0 * sz1);
  auto tol_sz = sz01 * sz2;
  VID_t count_selected_pixels = 0;
  for (VID_t index=0; index < tol_sz; ++index) {
    mask[index] = 0;
    // check all marker to see if their radius covers it
    for (const auto& marker : markers) {
      assertm(marker->radius > 0, "Markers must have a radius > 0");
      if (is_covered_by_parent(index, marker->vid(sz0, sz1), marker->radius - (DILATION_FACTOR - 1), sz0)) {
        mask[index] = 1;
        count_selected_pixels++;
        break;
      }
    }
  }
}

template <typename T, typename T2>
void create_coverage_mask(std::vector<MyMarker*>& markers, T* mask,
    T2 sz0, T2 sz1, T2 sz2) {
  auto sz01 = static_cast<VID_t>( sz0 * sz1);
  for (const auto& marker : markers) {
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
          //if (mask[ind] > 0) {
          //std::cout << "Warning: marker " << marker->description(sz0, sz1) <<
          //" is over covering at pixel " << x2 << " " << y2 << " " << z2 << '\n';
          //}
          mask[ind]++;
        }
      }
    }
  }
}

template <typename T, typename T2, typename T3>
auto check_coverage(const T mask, const T2 inimg1d, const VID_t tol_sz,
    T3 bkg_thresh) {
  VID_t match_count = 0;
  VID_t over_coverage = 0;
  std::vector<VID_t> false_negatives;
  std::vector<VID_t> false_positives;

  for (VID_t i=0; i < tol_sz; i++) {
    auto check = mask[i];
    // ground represents the original pixel value wheras
    // check merely indicates how many times a pixel was
    // covered in a pruning pass
    auto ground = inimg1d[i];
    //assertm(ground < 2, "this function only works on binarized images");

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

  return new CompareResults<std::vector<VID_t>>(
      false_negatives, false_positives, over_coverage,
      match_count);
}
