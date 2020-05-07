#pragma once

#include "recut_parameters.hpp"
#include <algorithm> //min
#include <cstdlib>   //rand srand
#include <ctime>     // for srand
#include <filesystem>
#include <math.h>
//#include "recut_prune.h"
#include "markers.h"
namespace fs = std::filesystem;

#ifdef USE_MCP3D
#include "common/mcp3d_common.hpp"
#include "common/mcp3d_utility.hpp" // PadNumStr
#include "image/mcp3d_image.hpp"
#include "image/mcp3d_voxel_types.hpp" // convert to CV type
#include "image/mcp3d_voxel_types.hpp"
#include <opencv2/opencv.hpp> // imwrite
#endif

#define PI 3.14159265

std::string get_curr() {
  fs::path full_path(fs::current_path());
  return fs::canonical(full_path).string();
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

/* interval_size parameter is actually irrelevant due to
 * copy on write, the chunk requested during reading
 * or mmapping is
 */
VID_t get_used_vertex_num(VID_t grid_size, VID_t block_size) {
  auto len = grid_size / block_size;
  auto total_blocks = len * len * len;
  auto pad_block_size = block_size + 2;
  auto pad_block_num = pad_block_size * pad_block_size * pad_block_size;
  // this is the total vertices that will be used including ghost cells
  auto interval_vert_num = pad_block_num * total_blocks;
  return interval_vert_num;
}

template <typename T> void print_image_3D(T *inimg1d, VID_t grid_size) {
  for (int zi = 0; zi < grid_size; zi++) {
    cout << "y | Z=" << zi << '\n';
    for (int xi = 0; xi < 2 * grid_size + 4; xi++) {
      cout << "-";
    }
    cout << '\n';
    for (int yi = 0; yi < grid_size; yi++) {
      cout << yi << " | ";
      for (int xi = 0; xi < grid_size; xi++) {
        VID_t index = ((VID_t)xi) + yi * grid_size + zi * grid_size * grid_size;
        // cout << i << " " ;
        cout << +inimg1d[index] << " ";
      }
      cout << '\n';
    }
    cout << '\n';
  }
}

void print_image(uint16_t *inimg1d, VID_t size) {
  cout << "print image " << '\n';
  for (VID_t i = 0; i < size; i++) {
    cout << i << " " << +inimg1d[i] << '\n';
  }
}

void get_img_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k,
                       VID_t grid_size) {
  i = id % grid_size;
  j = (id / grid_size) % grid_size;
  k = (id / (grid_size * grid_size)) % grid_size;
}

void mesh_grid(VID_t id, uint16_t *inimg1d, VID_t selected, int grid_size) {
  VID_t i, j, k, ic, jc, kc;
  i = j = k = ic = kc = jc = 0;
  // set root to 1
  inimg1d[id] = 1;
  VID_t actual = 1; // count root
  srand(time(NULL));
  while (actual < selected) {
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
    if (inimg1d[id] != 1) {
      inimg1d[id] = 1;
      actual++;
    }
  }
}

/*
 * sets all to 1 for tcase 0
 * sets all to 0 for tcase > 3
 * tcase5 = Sphere grid
 * takes an empty binarized inimg1d (all zeros)
 * and creates a central sphere of specified
 * radius directly in the center of the grid
 */
VID_t get_grid(int tcase, uint16_t *inimg1d, int grid_size) {
  VID_t selected = 0;

  // for tcase 5 sphere grid
  auto radius = grid_size / 4;
  radius = radius > 1 ? radius : 1; // clamp to 1

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
  for (int xi = 0; xi < grid_size; xi++) {
    for (int yi = 0; yi < grid_size; yi++) {
      for (int zi = 0; zi < grid_size; zi++) {
        VID_t index = ((VID_t)xi) + yi * grid_size + zi * grid_size * grid_size;
        x = xi * dh - .5;
        y = yi * dh - .5;
        z = zi * dh - .5;
        if (tcase == 0) {
          inimg1d[index] = 1;
        } else if (tcase == 1) {
          inimg1d[index] = (uint16_t)1 + .5 * sin(20 * PI * x) *
                                             sin(20 * PI * y) *
                                             sin(20 * PI * z);
        } else if (tcase == 2) {
          inimg1d[index] = (uint16_t)1 - .99 * sin(2 * PI * x) *
                                             sin(2 * PI * y) * sin(2 * PI * z);
        } else if (tcase == 3) {
          double r = sqrt(x * x + y * y);
          double R = sqrt(x * x + y * y + z * z);
          inimg1d[index] = 1;
        std:
          vector<double> Rvecs = {.15, .25, .35, .45};
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
            // if (r < .1) {
            // inimg1d = 0;
            // cout << "0";
            //}
            if (!condition0 != !condition1) { // xor / set difference
              inimg1d[index] = 0;
            } else {
              selected++;
            }
          }
        } else if (tcase == 4) { // 4 start with zero grid, and select in
                                 // function mesh_grid
          inimg1d[index] = 0;
        } else if (tcase == 5) {
          inimg1d[index] = 0;
          if ((xi >= xmin) && (xi <= xmax)) {
            if ((yi >= ymin) && (yi <= ymax)) {
              if ((zi >= zmin) && (zi <= zmax)) {
                inimg1d[index] = 1;
                selected++;
              }
            }
          }
        } else {
          assertm(false, "tcase specified not recognized");
        }
      }
    }
  }
  if (tcase < 3) {
    selected = grid_size * grid_size * grid_size;
  }
  return selected;
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

RecutCommandLineArgs get_args(int grid_size, int slt_pct, int tcase,
                              bool generate_image = false) {

  bool print = false;
#ifdef LOG
  print = true;
#endif

  RecutCommandLineArgs args;
  auto params = args.recut_parameters();
  auto str_path = get_curr();
  params.set_marker_file_path(
      str_path + "/test_markers/" + std::to_string(grid_size) + "/tcase" +
      std::to_string(tcase) + "/slt_pct" + std::to_string(slt_pct) + "/");
  // by setting the max intensities you do not need to recompute them
  // in the update function, this is critical for benchmarking
  params.set_max_intensity(1);
  params.set_min_intensity(0);
  // the total number of blocks allows more parallelism
  // ideally intervals >> thread count
  params.set_interval_size(grid_size);
  params.set_block_size(grid_size);
  VID_t img_vox_num = grid_size * grid_size * grid_size;

  if (generate_image) {
    params.generate_image = true;
    params.tcase = tcase;
    params.slt_pct = slt_pct;
    params.selected = img_vox_num * (slt_pct / (float)100);
    params.root_vid = get_central_vid(grid_size);
    params.set_user_thread_count(omp_get_max_threads());
    std::vector<int> extents = {grid_size, grid_size, grid_size};
    args.set_image_extents(extents);
  } else {
    args.set_image_root_dir(
        str_path + "/test_images/" + std::to_string(grid_size) + "/tcase" +
        std::to_string(tcase) + "/slt_pct" + std::to_string(slt_pct) + "/");
  }

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
  fs::remove_all(fn); // make sure it's an overwrite
  cout << "      Delete old: " << fn << '\n';
  fs::create_directories(fn);
  fn = fn + "marker";
  std::ofstream mf;
  mf.open(fn);
  mf << "# x,y,z\n";
  mf << x << "," << y << "," << z;
  mf.close();
  cout << "      Wrote marker: " << fn << '\n';
}

#ifdef USE_MCP3D
void write_tiff(uint16_t *inimg1d, std::string base, int grid_size) {
  fs::remove_all(base); // make sure it's an overwrite
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
      int cv_type = mcp3d::VoxelTypeToCVTypes(mcp3d::VoxelType::M16U, 1);
      cv::Mat m(grid_size, grid_size, cv_type, &(inimg1d[start]));
      cv::imwrite(fn, m);
    }

    // uint8_t* ptr = Plane(z, c, t);
    // std::vector<int> dims = {grid_size, grid_size, grid_size};
    // mcp3d::image::MImage mimg(dims); // defaults to uint16 format
  }
  cout << "      Wrote test images at: " << base << '\n';
}

uint16_t *read_tiff(std::string fn, int grid_size) {
  cout << "Read: " << fn << '\n';

  // auto full_img = cv::imread(fn, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  //// buffer for whole image
  // uint16_t* full_img_ptr = full_img.ptr<uint16_t>() ;
  // return full_img_ptr;

  // Mat test1(1000, 1000, CV_16U, Scalar(400));
  // imwrite("test.tiff", test1);
  // auto testfn = fn + "img_0000.tif";
  // cout << "Read: " << testfn << '\n';
  // cv::Mat test2 = cv::imread(testfn, cv::IMREAD_ANYDEPTH);
  // cout << test2 << '\n';
  // cout << test1.depth() << " " << test2.depth() << '\n';
  // cout << test2.at<unsigned short>(0,0) << '\n';

  vector<int> interval_offsets = {0, 0, 0}; // zyx
  vector<int> interval_extents = {grid_size, grid_size, grid_size};
  // read data
  mcp3d::MImage image;
  image.ReadImageInfo(fn);
  try {
    // use unit strides only
    mcp3d::MImageBlock block(interval_offsets, interval_extents);
    image.SelectView(block, 0);
    image.ReadData();
  } catch (...) {
    MCP3D_MESSAGE("error in image io. neuron tracing not performed")
    throw;
  }
  VID_t inimg1d_size = grid_size * grid_size * grid_size;
  uint16_t *img = new uint16_t[inimg1d_size];
  memcpy((void *)img, image.Volume<uint16_t>(0),
         inimg1d_size * sizeof(uint16_t));
  return img;
}
#endif

/*
 * test function based off APP2 by hanchuan peng to test accuracy of
 * fastmarching based calculate radius method
 */
template <class T>
uint16_t get_radius_accurate(const T *inimg1d, int grid_size, VID_t current_vid,
                             T thresh) {
  std::vector<int> sz = {grid_size, grid_size, grid_size};
  VID_t current_x, current_y, current_z;
  get_img_subscript(current_vid, current_x, current_y, current_z, grid_size);

  int max_r = grid_size / 2 - 1;
  int r;
  double tol_num, bak_num;
  int mx = current_x + 0.5;
  int my = current_y + 0.5;
  int mz = current_z + 0.5;
  // cout<<"mx = "<<mx<<" my = "<<my<<" mz = "<<mz<<'\n';
  int64_t x[2], y[2], z[2];

  tol_num = bak_num = 0.0;
  int64_t sz01 = sz[0] * sz[1];
  for (r = 1; r <= max_r; r++) {
    double r1 = r - 0.5;
    double r2 = r + 0.5;
    double r1_r1 = r1 * r1;
    double r2_r2 = r2 * r2;
    double z_min = 0, z_max = r2;
    for (int dz = z_min; dz < z_max; dz++) {
      double dz_dz = dz * dz;
      double y_min = 0;
      double y_max = sqrt(r2_r2 - dz_dz);
      for (int dy = y_min; dy < y_max; dy++) {
        double dy_dy = dy * dy;
        double x_min = r1_r1 - dz_dz - dy_dy;
        x_min = x_min > 0 ? sqrt(x_min) + 1 : 0;
        double x_max = sqrt(r2_r2 - dz_dz - dy_dy);
        for (int dx = x_min; dx < x_max; dx++) {
          x[0] = mx - dx, x[1] = mx + dx;
          y[0] = my - dy, y[1] = my + dy;
          z[0] = mz - dz, z[1] = mz + dz;
          for (char b = 0; b < 8; b++) {
            char ii = b & 0x01, jj = (b >> 1) & 0x01, kk = (b >> 2) & 0x01;
            if (x[ii] < 0 || x[ii] >= sz[0] || y[jj] < 0 || y[jj] >= sz[1] ||
                z[kk] < 0 || z[kk] >= sz[2])
              return r;
            else {
              tol_num++;
              long pos = z[kk] * sz01 + y[jj] * sz[0] + x[ii];
              if (inimg1d[pos] <= thresh) {
                bak_num++;
              }
              if ((bak_num / tol_num) > 0.0001)
                return r;
            }
          }
        }
      }
    }
  }
  return r;
}

/*
 * test function based off APP2 by hanchuan peng to test accuracy of
 * fastmarching based calculate radius method
 */
template <class T>
uint16_t get_radius_hanchuan_XY(const T *inimg1d, int grid_size,
                                VID_t current_vid, T thresh) {
  std::vector<int> sz = {grid_size, grid_size, grid_size};
  VID_t x, y, z;
  get_img_subscript(current_vid, x, y, z, grid_size);

  long sz0 = sz[0];
  long sz01 = sz[0] * sz[1];
  double max_r = grid_size / 2 - 1;
  if (max_r > sz[1] / 2)
    max_r = sz[1] / 2;

  double total_num, background_num;
  double ir;
  for (ir = 1; ir <= max_r; ir++) {
    total_num = background_num = 0;

    double dz, dy, dx;
    double zlower = 0, zupper = 0;
    for (dz = zlower; dz <= zupper; ++dz)
      for (dy = -ir; dy <= +ir; ++dy)
        for (dx = -ir; dx <= +ir; ++dx) {
          total_num++;

          double r = sqrt(dx * dx + dy * dy + dz * dz);
          if (r > ir - 1 && r <= ir) {
            int64_t i = x + dx;
            if (i < 0 || i >= sz[0])
              goto end1;
            int64_t j = y + dy;
            if (j < 0 || j >= sz[1])
              goto end1;
            int64_t k = z + dz;
            if (k < 0 || k >= sz[2])
              goto end1;

            if (inimg1d[k * sz01 + j * sz0 + i] <= thresh) {
              background_num++;

              if ((background_num / total_num) > 0.001)
                goto end1; // change 0.01 to 0.001 on 100104
            }
          }
        }
  }
end1:
  return ir;
}

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

#ifdef MMAP
  cout << "MMAP" << '\n';
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

#ifdef USE_OMP
  cout << "USE_OMP" << '\n';
#endif
}
