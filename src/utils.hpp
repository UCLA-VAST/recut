#pragma once

#include <filesystem>
#include <cstdlib> //rand srand
#include <ctime> // for srand
#include <math.h>
#include "recut_parameters.hpp"
namespace fs = std::filesystem;

using fs::exists;
using fs::remove;
using fs::remove_all;
using fs::directory_iterator;
using fs::canonical;
using fs::current_path;
using fs::path;
using fs::create_directories;
using std::to_string;

#define PI 3.14159265

std::string get_curr() {
    path full_path(current_path());
    return canonical(full_path).string();
}

void print_image(uint16_t* inimg1d, VID_t size) {
  cout << "print image " << endl;
  for (VID_t i=0; i < size; i++) {
    //cout << i << " " << +inimg1d[i] << endl;
    assert(inimg1d[i] <= 1);
  }
}

void get_img_subscript(VID_t id, VID_t &i, VID_t &j, VID_t &k, VID_t grid_size) {
  i = id % grid_size;
  j = (id / grid_size) % grid_size;
  k = (id / (grid_size * grid_size)) % grid_size;
}

void mesh_grid(VID_t id, uint16_t* inimg1d, VID_t selected, int grid_size) {
  VID_t i, j, k, ic, jc, kc;
  i=j=k=ic=kc=jc=0;
  // set root to 1
  inimg1d[id] = 1;
  VID_t actual = 1; // count root
  srand(time(NULL));
  while (actual < selected) {
    // calc i, j, k subs
    i = id % grid_size;
    j = (id / grid_size) % grid_size;
    k = (id / (grid_size * grid_size)) % grid_size;
    //std::cout << "previous " << id << " i " << i << " j " << j << " k " << k << endl;

    // try to find a suitable next location in one of the
    // six directions, if a proper direction was found
    // update the id and the subscripts to reflect
    // if this iterations direction is invalid skip to next
    long dir = rand() % 6;
    if (dir == 4) {
      if (k == 0) {continue;}
      k -= 1;
      id = id - grid_size * grid_size;
    } else if (dir == 2) {
      if (j == 0) {continue;}
      j -= 1;
      id = id - grid_size ;
    } else if (dir == 0) {
      if (i == 0) {continue;}
      i -= 1;
      id = id - 1 ;
    } else if (dir == 1) {
      if (i == grid_size - 1) {continue;}
      i += 1;
      id = id + 1 ;
    } else if (dir == 3) {
      if (j == grid_size - 1) {continue;}
      j += 1;
      id = id + grid_size ;
    } else if (dir == 5) {
      if (k == grid_size - 1) {continue;}
      k += 1;
      id = id + grid_size * grid_size ;
    }
    //std::cout << "node id" << id << " i " << i << " j " << j << " k " << k << endl;
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

// sets all to 1 for tcase 0
// sets all to 0 for tcase > 3
void get_grid(int tcase, uint16_t* inimg1d, int grid_size) {
  double dh = 1 / grid_size;
  double x, y, z;
  double w = 1 / 24;
  uint16_t trueval = 1;
  for (int xi=0; xi < grid_size; xi++) {
    for (int yi=0; yi < grid_size; yi++) {
      for (int zi=0; zi < grid_size; zi++) {
        int index = int(xi + yi * grid_size + zi * grid_size * grid_size);
        x = xi * dh - .5;
        y = yi * dh - .5;
        z = zi * dh - .5;
        if (tcase == 0) {
          inimg1d[index] = 1;
        } else if (tcase == 1) {
          inimg1d[index] = (uint16_t) 1 + .5 * sin(20 * PI * x) * sin(20 * PI * y) * sin(20 * PI * z);
        } else if (tcase == 2) {
          inimg1d[index] = (uint16_t) 1 - .99 * sin(2 * PI * x) * sin(2 * PI * y) * sin(2 * PI * z);
        } else if (tcase == 3) {
          double r = sqrt(x * x + y * y);
          double R = sqrt(x * x + y * y + z * z);
          inimg1d[index] = trueval;
          std:vector<double> Rvecs = {.15, .25, .35, .45};
          for (int ri=0; ri < Rvecs.size(); ri++) {
            double Rvec = Rvecs[ri];
            bool condition0, condition1;
            condition0 = condition1 = false;
            if (Rvec < R < Rvec + w) {
              condition0 = true;
            }
            if (ri == 0) {
              if (r < .05 ) {
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
            //if (r < .1) {
              //inimg1d = 0;
              //cout << "0";
            //}
            if (!condition0 != !condition1) { //xor / set difference
              inimg1d[index] = 0;
            }
          }
        } else { // 4 and 5 start with zero grids, and select
          inimg1d[index] = 0;
        }
      }
    }
  }
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
VID_t lattice_grid(VID_t start, uint16_t* inimg1d, int line_per_dim,
    int grid_size) {
  int interval = grid_size / line_per_dim; // roughly equiv
  std::vector<VID_t> x(line_per_dim + 1);
  std::vector<VID_t> y(line_per_dim + 1);
  std::vector<VID_t> z(line_per_dim + 1);
  VID_t i, j, k, count;
  i=j=k=0;
  count = 0;
  VID_t selected = 0;
  for (int count=0; count < grid_size; count+=interval) {
    x.push_back(count);
    y.push_back(count);
    z.push_back(count);
    std::cout << "Count " << count << endl;
  }
  get_img_subscript(start, i, j, k, grid_size);
  x.push_back(i);
  y.push_back(j);
  z.push_back(k);

  for (auto& xi : x) {
    for (auto& yi : y) {
      for (int zi; zi < grid_size; zi++) {
        int index = int(xi + yi * grid_size + zi * grid_size * grid_size);
        if (inimg1d[index] != 1) {
          inimg1d[index] = 1; // set to max
          selected++;
        }
      }
    }
  }

  for (auto& xi : x) {
    for (auto& zi : z) {
      for (int yi; yi < grid_size; yi++) {
        int index = int(xi + yi * grid_size + zi * grid_size * grid_size);
        if (inimg1d[index] != 1) {
          inimg1d[index] = 1; // set to max
          selected++;
        }
      }
    }
  }

  for (auto& yi : y) {
    for (auto& zi : z) {
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

VID_t get_central_sub(int grid_size) {
  return grid_size / 2 - 1; // place at center
}

VID_t get_central_vid(int grid_size) {
  auto sub = get_central_sub(grid_size);
  auto root_vid = (VID_t) sub * grid_size * grid_size + sub * grid_size + sub;
  return root_vid; // place at center
}

VID_t get_central_diag_vid(int grid_size) {
  auto sub = get_central_sub(grid_size);
  sub++; // add 1 to all x, y, z
  auto root_vid = (VID_t) sub * grid_size * grid_size + sub * grid_size + sub;
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

RecutCommandLineArgs get_args(int grid_size, int slt_pct, int tcase,
    bool generate_image=false, bool print=false) {
  RecutCommandLineArgs args;
  std::vector<int> extents = {grid_size, grid_size, grid_size};
  args.set_image_extents(extents);
  auto params = args.recut_parameters();
  auto str_path = get_curr();
  params.set_marker_file_path(str_path + "/test_markers/" + to_string(grid_size) + "/tcase" + to_string(tcase) + "/slt_pct" + to_string(slt_pct) + "/");
  params.set_max_intensity(1);
  params.set_min_intensity(0);
  VID_t img_vox_num = grid_size * grid_size * grid_size;

  if (generate_image) {
    params.generate_image = true;
    params.tcase = tcase;
    params.slt_pct = slt_pct;
    params.selected = img_vox_num * (slt_pct / (float) 100);
    params.root_vid = get_central_vid(grid_size);
    params.set_user_thread_count(omp_get_max_threads());
  } else {
    args.set_image_root_dir(str_path + "/test_images/" + to_string(grid_size) + "/tcase" + to_string(tcase) + "/slt_pct" + to_string(slt_pct) + "/");
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
  remove_all(fn); // make sure it's an overwrite
  cout << "      Delete old: " << fn << endl;
  create_directories(fn);
  fn = fn + "marker";
  std::ofstream mf;
  mf.open(fn);
  mf << "# x,y,z\n";
  mf << x << "," << y << "," << z;
  mf.close();
  cout << "      Wrote marker: " << fn << endl;
}

#ifdef IMAGE
void write_tiff(uint16_t* inimg1d, std::string base, int grid_size) {
  remove_all(base); // make sure it's an overwrite
  cout << "      Delete old: " << base << endl;
  create_directories(base);
  //print_image(inimg1d, grid_size * grid_size * grid_size);
  for (int zi=0; zi < grid_size; zi++) {
    std::string fn = base;
    fn = fn + "img_";
    //fn = fn + mcp3d::PadNumStr(zi, 9999); // pad to 4 digits
    fn = fn + std::to_string(zi); // pad to 4 digits
    std::string suff = ".tif";
    fn = fn + suff;
    VID_t start = zi * grid_size * grid_size;
    //cout << "fn: " << fn << " start: " << start << endl;
    //print_image(&(inimg1d[start]), grid_size * grid_size);

    { // cv write
      int cv_type = mcp3d::VoxelTypeToCVTypes(mcp3d::VoxelType::M16U, 1);
      cv::Mat m(grid_size, grid_size, cv_type, &(inimg1d[start]));
      cv::imwrite(fn, m);
    }

    //uint8_t* ptr = Plane(z, c, t);
    //std::vector<int> dims = {grid_size, grid_size, grid_size};
    //mcp3d::image::MImage mimg(dims); // defaults to uint16 format
  }
  cout << "      Wrote test images at: " << base << endl;
}

uint16_t* read_tiff(std::string fn, int grid_size ) {
  cout << "Read: " << fn << endl;

  //auto full_img = cv::imread(fn, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  //// buffer for whole image
  //uint16_t* full_img_ptr = full_img.ptr<uint16_t>() ;
  //return full_img_ptr;

  //Mat test1(1000, 1000, CV_16U, Scalar(400));
  //imwrite("test.tiff", test1);
  //auto testfn = fn + "img_0000.tif";
  //cout << "Read: " << testfn << endl;
  //cv::Mat test2 = cv::imread(testfn, cv::IMREAD_ANYDEPTH);
  //cout << test2 << endl;
  //cout << test1.depth() << " " << test2.depth() << endl;
  //cout << test2.at<unsigned short>(0,0) << endl;

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
  } catch(...) {
    MCP3D_MESSAGE("error in image io. neuron tracing not performed")
    throw;
  }
  VID_t inimg1d_size = grid_size * grid_size * grid_size;
  uint16_t* img = new uint16_t[inimg1d_size];
  memcpy((void*) img, image.Volume<uint16_t>(0), inimg1d_size * sizeof(uint16_t));
  return img;
}
#endif
