#include "../src/recut.hpp"
#include "app2_helpers.hpp"
#include "gtest/gtest.h"
#include <bits/stdc++.h>
#include <cstdlib> //rand
#include <ctime>   // for srand
#include <fcntl.h>
#include <fstream>
#include <iostream>
//#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

// catch an assertion and auto drop into
// intractive mode for gdb or rr
#define GTEST_BREAK_ON_FAILURE 1
// stop after first test failure
#define GTEST_FAIL_FAST 1

// using json = nlohmann::json;

#ifdef USE_MCP3D
#include "common/mcp3d_utility.hpp"    // PadNumStr
#include "image/mcp3d_voxel_types.hpp" // convert to CV type
#include <opencv2/opencv.hpp>          // imwrite
#endif

#define EXP_DEV_LOW .05

#ifdef USE_MCP3D
#define GEN_IMAGE false
#else
#define GEN_IMAGE true
#endif

// This source file is for functions or tests that include test macros only
// note defining a function that contains test function MACROS
// that returns a value will give strange: "void value not ignored as it ought
// to be" errors

template <class T, typename DataType>
void check_recut_error(T &recut, DataType *ground_truth, int grid_size,
    std::string stage, double &error_rate) {
  auto tol_sz = static_cast<VID_t>(grid_size) * grid_size * grid_size;
  // cout << "check image " << endl;
  double error_sum = 0.0;
  VID_t total_valid = 0;
  VertexAttr *v;
  for (int zi = 0; zi < recut.image_length_z; zi++) {
    for (int yi = 0; yi < recut.image_length_y; yi++) {
      for (int xi = 0; xi < recut.image_length_x; xi++) {
        VID_t vid = recut.get_img_vid(xi, yi, zi);
        auto interval_id = recut.get_interval_id(vid);
        auto block_id = recut.get_block_id(vid);
        auto interval = recut.grid.GetInterval(interval_id);
        v = nullptr;
        // if the entire interval not in memory make every possible
        // selected vertex with ground truth is also not valid
        if (interval->IsInMemory()) {
          v = recut.get_attr_vid(interval_id, block_id, vid, nullptr);
        }
        // cout << i << endl << v.description() << " ";
        if (stage == "value") {
          if (recut.generated_image[vid]) {
            ASSERT_NE(v, nullptr);
            bool valid_value = v->value != std::numeric_limits<uint8_t>::max();
            ASSERT_TRUE(valid_value);
          }
          if (v) {
            if (v->valid_value()) {
              ASSERT_EQ(recut.generated_image[vid], 1);
              error_sum += absdiff(ground_truth[vid], v->value);
              // cout << v->value << " ";
            }
          }
        } else if (stage == "radius") {
          if (recut.generated_image[vid]) {
            ASSERT_NE(v, nullptr);
            ASSERT_TRUE(v->valid_radius()) << " vid " << vid << " recut radius "
              << recut.generated_image[vid];
            error_sum += absdiff(ground_truth[vid], v->radius);
            ++total_valid;
          }
          if (v) {
            if (v->valid_radius()) {
              ASSERT_EQ(recut.generated_image[vid], 1);
            }
          }
        } else if (stage == "surface") {
          // if truth shows a value of 1 it is a surface
          // vertex, therefore surface_vec should also
          // contain this value
          if (ground_truth[vid] == 1) {
            ASSERT_NE(v, nullptr);
            auto found = false;
            for (auto iter : recut.surface_vec[interval_id][block_id]) {
              if (iter->vid == v->vid)
                found = true;
            }
            // if truth shows a value of 1 it is a surface
            // vertex, therefore surface_vec should also
            // contain this value
            ASSERT_TRUE(found) << "vid " << vid << " x" << xi << " y " << yi
              << " z " << zi << " v->vid " << v->vid << '\n';
          }
        }
      }
    }
  }
  if (stage != "surface") {
    ASSERT_EQ(total_valid, recut.params->selected);
    error_rate = 100 * error_sum / static_cast<double>(tol_sz);
  }
}

void check_image_error(uint16_t *inimg1d, uint16_t *baseline, uint16_t *check,
    int grid_size, VID_t selected, double &error_rate) {
  auto tol_sz = grid_size * grid_size * grid_size;
  VID_t total_valid = 0;
  // cout << "check image " << endl;
  VID_t error_sum = 0;
  for (VID_t i = 0; i < tol_sz; i++) {
    if (inimg1d[i]) {
      // cout << i << " " << +inimg1d[i] << endl;
      error_sum += absdiff(baseline[i], check[i]);
      ++total_valid;
    }
  }
  ASSERT_EQ(total_valid, selected);
  error_rate = 100 * error_sum / static_cast<double>(tol_sz);
}

void check_image_equality(uint16_t *inimg1d, uint16_t *check, int grid_size) {
  // cout << "check image " << endl;
  for (VID_t i = 0; i < (grid_size * grid_size * grid_size); i++) {
    // cout << i << " " << +inimg1d[i] << endl;
    ASSERT_LE(inimg1d[i], 1);
    ASSERT_EQ(inimg1d[i], check[i]);
  }
}

// make sure base interval is implemented in a read-only manner
void interval_base_immutable(VID_t nvid) {
  ASSERT_LE(nvid, MAX_INTERVAL_VERTICES);
  // check that interval_base has not changed
  auto base = INTERVAL_BASE;
  auto interval = new Interval(nvid, 0, base);
  interval->LoadFromDisk();
  for (int i = 0; i < nvid; i++) {
    ASSERT_TRUE(interval->GetData()[i].unvisited())
      << " i= " << i << " vid " << interval->GetData()[i].vid << endl;
    ASSERT_FALSE(interval->GetData()[i].valid_vid());
    ASSERT_FALSE(interval->GetData()[i].valid_value());
  }
}

void load_save(bool mmap_) {
  auto nvid = 64;
  auto fn = get_data_dir() + "/interval0.bin";
  auto base = INTERVAL_BASE;
  fs::remove(fn); // for safety, but done automatically via ~Interval
  ASSERT_FALSE(fs::exists(fn));
  ASSERT_TRUE(fs::exists(base));

  // create
  auto interval = new Interval(nvid, 0, base, mmap_);
  ASSERT_FALSE(interval->IsInMemory());
  interval->LoadFromDisk();
  ASSERT_TRUE(interval->IsInMemory());
  interval->GetData()[1].mark_root(0);
  interval->GetData()[1].vid = 1;
  ASSERT_TRUE(interval->GetData()->unvisited());

  // save
  interval->SaveToDisk();
  ASSERT_EQ(interval->GetFn(), fn);
  ASSERT_FALSE(interval->IsInMemory());
  ASSERT_TRUE(fs::exists(fn));

  // load and check
  interval->LoadFromDisk();
  ASSERT_EQ(interval->GetFn(), fn);
  ASSERT_TRUE(interval->IsInMemory());
  ASSERT_TRUE(interval->GetData()[1].root());
  ASSERT_EQ(interval->GetData()[1].vid, 1);
  ASSERT_TRUE(interval->GetData()->unvisited());
  delete interval;
  ASSERT_FALSE(fs::exists(fn));

  // check interval can't delete interval_base_64bit.bin
  // create
  auto interval2 = new Interval(nvid, 0, base, mmap_);
  interval2->LoadFromDisk();
  delete interval2;

  ASSERT_FALSE(fs::exists(fn));
  ASSERT_TRUE(fs::exists(base));

  ASSERT_NO_FATAL_FAILURE(interval_base_immutable(nvid));
}

//// assigns all of the intervals within the super interval
//// a unique vid even in padded regions. This is especially
//// useful for testing get_attr_vid is returning the correct
//// vid
// void assign_grid(Recut recut) {
//// test that get_attr_vid returns the right match in various scenarios
//// assign all vids for testing purposes
// VID_t index = 0;
// for (int i=0; i < recut.grid.GetNIntervals(); i++) {
// auto interval = recut.grid.GetInterval(i);
// auto attr = interval->GetData();
// for (int j=0; j < interval->GetNVertices(); j++) {
// auto current = attr + index; // get the appropriate vertex
// current->vid = index;
// index ++;
//}
//}
//}

void test_get_attr_vid(bool mmap, int grid_size, int interval_size,
    int block_size) {
  auto nvid = grid_size * grid_size * grid_size;
  double slt_pct = 100;
  int tcase = 0;
  auto args =
    get_args(grid_size, interval_size, block_size, slt_pct, tcase, GEN_IMAGE);

  auto root_vid = get_central_vid(grid_size);
  // x + 1, y + 1, z + 1 to the other central
  VID_t root_diag_vid = get_central_diag_vid(grid_size);
  std::vector<VID_t> vids = {root_vid, root_diag_vid};
  Recut recut = Recut<uint16_t>(args);
  recut.mmap_ = mmap;

  auto root_vids = recut.initialize();
  recut.setup_value(root_vids);

  ASSERT_EQ(recut.image_length_x, grid_size);
  ASSERT_EQ(recut.image_length_y, grid_size);
  ASSERT_EQ(recut.image_length_z, grid_size);

  // these tests make sure get_attr_vid always returns a pointer to underlying
  // vertex data and does not copy by value the vertex struct
  for (auto &vid : vids) {
    // cout << "TESTING vid " << vid << endl;
    ASSERT_LT(vid, nvid);
    auto current_interval_id = recut.get_interval_id(vid);
    auto current_block_id = recut.get_block_id(vid);
    std::vector<VID_t> test_blocks = {};
    test_blocks.push_back(current_block_id);
    std::vector<VID_t> test_intervals = {};
    test_intervals.push_back(current_interval_id);

    if (grid_size / interval_size == 2) {
      VID_t iinterval, jinterval, kinterval;
      // the interval is a linear index into the 3D row-wise arrangement of
      // interval, converting to subscript makes adjustments easier
      recut.get_interval_subscript(current_interval_id, iinterval, jinterval,
          kinterval);
      if (vid == root_vid) {
        test_intervals.push_back(
            recut.get_interval_id(iinterval + 1, jinterval, kinterval));
        test_intervals.push_back(
            recut.get_interval_id(iinterval, jinterval + 1, kinterval));
        test_intervals.push_back(
            recut.get_interval_id(iinterval, jinterval, kinterval + 1));
      } else if (vid == root_diag_vid) {
        test_intervals.push_back(
            recut.get_interval_id(iinterval - 1, jinterval, kinterval));
        test_intervals.push_back(
            recut.get_interval_id(iinterval, jinterval - 1, kinterval));
        test_intervals.push_back(
            recut.get_interval_id(iinterval, jinterval, kinterval - 1));
      }
    } else if (interval_size / block_size == 2) {
      VID_t iblock, jblock, kblock;
      // the block_id is a linear index into the 3D row-wise arrangement of
      // blocks, converting to subscript makes adjustments easier
      recut.get_block_subscript(current_block_id, iblock, jblock, kblock);
      if (vid == root_vid) {
        test_blocks.push_back(recut.get_block_id(iblock + 1, jblock, kblock));
        test_blocks.push_back(recut.get_block_id(iblock, jblock + 1, kblock));
        test_blocks.push_back(recut.get_block_id(iblock, jblock, kblock + 1));
      } else if (vid == root_diag_vid) {
        test_blocks.push_back(recut.get_block_id(iblock - 1, jblock, kblock));
        test_blocks.push_back(recut.get_block_id(iblock, jblock - 1, kblock));
        test_blocks.push_back(recut.get_block_id(iblock, jblock, kblock - 1));
      }
    }

    VID_t output_offset;
    for (auto &interval_id : test_intervals) {
      // cout << "interval_id " << interval_id << endl;
      recut.grid.GetInterval(interval_id)->LoadFromDisk();
      for (auto &block_id : test_blocks) {
        // cout << "\tblock_id " << block_id << endl;
        // TEST 1
        // Assert upstream changes (by pointer)
        auto attr =
          recut.get_attr_vid(interval_id, block_id, vid, &output_offset);
        attr->vid = vid;
        attr->value = 3;
        auto attr2 =
          recut.get_attr_vid(interval_id, block_id, vid, &output_offset);
        ASSERT_EQ(attr, attr2); // upstream changes?

        // TEST 2
        // assert get_attr_vid can handle dummy structs added in to heaps
        // and update real embedded values in interval accordingly
        VertexAttr *dummy_attr =
          new VertexAttr(); // march is protect from dummy values like this
        dummy_attr->mark_root(0); // 0000 0000, selected no parent, all zeros
        // indicates KNOWN_FIX root
        dummy_attr->value = 1.0;
        dummy_attr->vid = vid;
        auto embedded_attr = recut.get_attr_vid(
            interval_id, block_id, dummy_attr->vid, &output_offset);
        *embedded_attr = *dummy_attr;
        ASSERT_EQ(embedded_attr->vid, dummy_attr->vid);
        auto embedded_attr2 = recut.get_attr_vid(
            interval_id, block_id, dummy_attr->vid, &output_offset);
        ASSERT_EQ(embedded_attr2, embedded_attr); // same obj?
        ASSERT_EQ(embedded_attr2->vid, dummy_attr->vid);
        ASSERT_EQ(*embedded_attr2, *dummy_attr); // same values?
      }
      recut.grid.GetInterval(interval_id)->Release();
      ASSERT_FALSE(recut.grid.GetInterval(interval_id)->IsInMemory());
    }
  }
}

TEST(Heap, PushUpdate) {
  VID_t N = 1 << 10;
  bool update_values = true;

  std::vector<std::string> stages = {"value", "radius"};
  // float mval = std::numeric_limits<float>::max();
  float mval = 255;
  srand(time(0));
  std::vector<uint8_t> vr;
  std::vector<float> vv;
  for (auto &stage : stages) {
    NeighborHeap<VertexAttr> heap;
    uint8_t radius;
    float value;
    uint8_t updated_radius;
    float updated_value;
    for (VID_t i = 0; i < N; i++) {
      auto vert = new VertexAttr;
      ASSERT_FALSE(vert->valid_handle());
      if (stage == "radius") {
        radius = (uint8_t)rand() % std::numeric_limits<uint8_t>::max();
        updated_radius = (uint8_t)rand() % std::numeric_limits<uint8_t>::max();
        vert->radius = radius;
        vr.push_back(radius);
        ASSERT_EQ(vert->radius, radius);
      } else if (stage == "value") {
        value =
          static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / mval));
        updated_value =
          static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / mval));
        vert->value = value;
        vv.push_back(value);
      }
      heap.push(vert, 0, stage);
      ASSERT_TRUE(vert->valid_handle());

      /* for even element, update it with some random number
       * to make sure the update can still retain proper
       * ordering
       */
      if (update_values && (i % 2)) {
        // cout << "elems pre-update: " << " i " << i << endl;
        // heap.print(stage);

        if (stage == "radius") {
          // vert->radius = updated_radius;
          vr.pop_back();
          vr.push_back(updated_radius);
          heap.update(vert, 0, updated_radius, stage);
        } else {
          // vert->value = updated_value;
          vv.pop_back();
          vv.push_back(updated_value);
          heap.update(vert, 0, updated_value, stage);
        }
      }

      // cout << "elems: " << endl;
      // heap.print(stage);
    }

    // make sure all in heap are popped in
    // non-increasing order
    if (stage == "radius") {
      sort(vr.begin(), vr.end());
      for (auto &val : vr) {
        auto min_attr = heap.pop(0, stage);
        ASSERT_FALSE(min_attr->valid_handle());
        auto hval = min_attr->radius;
        ASSERT_EQ(val, hval) << "hval " << +hval << " val " << +val;
      }
    } else {
      sort(vv.begin(), vv.end());
      for (auto &val : vv) {
        auto min_attr = heap.pop(0, stage);
        ASSERT_FALSE(min_attr->valid_handle());
        auto hval = min_attr->value;
        ASSERT_EQ(val, hval) << "hval " << +hval << " val " << +val;
      }
    }
  }
}

/*
 * This test suite creates the basic initialized set of
 * VertexAttr for each new Interval to read from
 * The case cubed represents the max number of VertexAttrs
 * that may be requested/read
 * takes ~11 minutes for 3.7e9 (padded ~ 1024^3) vertices
 */
TEST(Install, CreateIntervalBase) {
  bool rerun = false; // change to true if MAX_INTERVAL_VERTICES or the vertex
  // struct has been changed in vertex_attr.h
  auto fn = INTERVAL_BASE;
  VID_t nvid = MAX_INTERVAL_VERTICES;
  bool mmap_flag = false;
  if (!fs::exists(fn) || rerun) {
    VID_t size = sizeof(VertexAttr) * nvid;
    cout << "Start creating " << fn << " cached total size: ~"
      << (size / (2 << 19)) << " MB"
      << " for max vertices: " << nvid << endl; // set the default values

    // make sure it's an overwrite
    if (fs::exists(fn)) {
      // delete previous
      assertm(unlink(fn) != -1, "unlink not successful");
      ASSERT_FALSE(fs::exists(fn));
    }
    struct VertexAttr *ptr;
    int fd;
    std::ofstream ofile;
    if (mmap_flag) {
      assertm((fd = open(fn, O_CREAT)) != -1, "open not successful");
      assert((ptr = (struct VertexAttr *)mmap(nullptr, size, PROT_WRITE,
              MAP_SHARED | MAP_ANONYMOUS, fd,
              0)) != MAP_FAILED);

      for (VID_t i = 0; i < nvid; i++) {
        auto v = ptr[i];
        v.edge_state = 192;
        v.vid = numeric_limits<VID_t>::max();
        v.handle = numeric_limits<VID_t>::max();
        v.value = numeric_limits<float>::max();
        // cout << i << endl << v.description() << endl;
      }

    } else {
      ofile.open(fn, ios::out | ios::binary); // read-mode
      ASSERT_TRUE(ofile.is_open());
      ASSERT_TRUE(ofile.good());
      struct VertexAttr *v = new VertexAttr;
      cout << v->description() << endl;
      v->edge_state = 192;
      v->vid = numeric_limits<VID_t>::max();
      v->handle = numeric_limits<VID_t>::max();
      v->value = numeric_limits<float>::max();
      for (VID_t i = 0; i < nvid; i++) {
        // write struct to file
        ofile.write((char *)v, sizeof(VertexAttr));
      }
    }

    if (mmap_flag) {
      assert(munmap((void *)ptr, size) == 0);
    } else {
      // ofile.write((char*)ptr, size);
      ASSERT_TRUE(ofile.good());

      // close file
      ofile.close();
      ASSERT_TRUE(fs::exists(fn));
    }

    cout << "finished interval creation" << endl;
  } else {
    cout << fn << " already exists" << endl;
  }
  // check the first few VertexAttr look correct
  ASSERT_NO_FATAL_FAILURE(interval_base_immutable(2));
}

TEST(Interval, LoadSaveIntervalMmap) {
  bool mmap_ = true;
  ASSERT_NO_FATAL_FAILURE(load_save(mmap_));
}

TEST(Interval, LoadSaveInterval) {
  bool mmap_ = false;
  ASSERT_NO_FATAL_FAILURE(load_save(mmap_));
}

TEST(Interval, GetAttrVid) {
  bool mmap_ = false;
  VID_t grid_size = 4;
  VID_t nvid = grid_size * grid_size * grid_size;

  ASSERT_NO_FATAL_FAILURE(
      test_get_attr_vid(mmap_, grid_size, grid_size, grid_size));
  // make sure base interval is implemented in a read-only manner
  ASSERT_NO_FATAL_FAILURE(
      interval_base_immutable(get_used_vertex_size(grid_size, grid_size)));
}

TEST(Interval, GetAttrVidMmap) {
  bool mmap_ = true;
  VID_t grid_size = 4;
  VID_t nvid = grid_size * grid_size * grid_size;

  ASSERT_NO_FATAL_FAILURE(
      test_get_attr_vid(mmap_, grid_size, grid_size, grid_size));
  // make sure base interval is implemented in a read-only manner
  ASSERT_NO_FATAL_FAILURE(
      interval_base_immutable(get_used_vertex_size(grid_size, grid_size)));
}

TEST(Interval, GetAttrVidMultiBlock) {
  VID_t grid_size = 8;
  VID_t block_size = grid_size / 2;
  // this is the total vertices that will be used including ghost cells
  auto interval_vert_num = get_used_vertex_size(grid_size, block_size);
  // this is the domain size not counting ghost cells
  VID_t nvid = grid_size * grid_size * grid_size;

  bool mmap_ = true;
  // single interval
  ASSERT_NO_FATAL_FAILURE(
      test_get_attr_vid(mmap_, grid_size, grid_size, grid_size / 2));

  ASSERT_NO_FATAL_FAILURE(interval_base_immutable(interval_vert_num));
}

TEST(Interval, GetAttrVidMultiInterval) {
  VID_t grid_size = 8;
  auto interval_size = grid_size / 2;
  VID_t nvid = grid_size * grid_size * grid_size;
  auto interval_vert_num = get_used_vertex_size(grid_size, interval_size);

  bool mmap_ = true;
  // single block per interval
  ASSERT_NO_FATAL_FAILURE(
      test_get_attr_vid(mmap_, grid_size, interval_size, interval_size));

  ASSERT_NO_FATAL_FAILURE(interval_base_immutable(interval_vert_num));
}

/*
 * Create the desired markers (seed locations) and images to be used by other
 * gtest functions below, this can be skipped once it has been run on a new
 * system for all the necessary configurations of grid_sizes, tcases and
 * selected percents
 */
TEST(Install, DISABLED_CreateImagesMarkers) {
  // change these to desired params
  // Note this will delete anything in the
  // same directory before writing
  // tcase 5 is deprecated
  //std::vector<int> grid_sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512};
  //std::vector<int> testcases = {4, 3, 2, 1, 0};
  //std::vector<double> selected_percents = {1, 10, 50, 100};
  std::vector<int> grid_sizes = {512, 1024};
  std::vector<int> testcases = {0};
  std::vector<double> selected_percents = {100};

  for (auto &grid_size : grid_sizes) {
    cout << "Create grids of " << grid_size << endl;
    auto root_vid = get_central_vid(grid_size);
    // FIXME save root marker vid
    long sz0 = (long)grid_size;
    long sz1 = (long)grid_size;
    long sz2 = (long)grid_size;
    VID_t tol_sz = sz0 * sz1 * sz2;

    uint16_t bkg_thresh = 0;
    int line_per_dim = 2; // for tcase 5 lattice grid

    VID_t x, y, z;
    x = y = z = get_central_sub(grid_size); // place at center
    VertexAttr *root = new VertexAttr();
    root->vid = (VID_t)z * sz0 * sz1 + y * sz0 + x;
    ASSERT_EQ(root->vid, root_vid);
    cout << "x " << x << "y " << y << "z " << z << endl;
    cout << "root vid " << root->vid << endl;

    for (auto &tcase : testcases) {
      cout << "  tcase " << tcase << endl;
      for (auto &slt_pct : selected_percents) {
        if ((tcase != 4) && (slt_pct != 100))
          continue;
        if ((tcase == 4) && (slt_pct > 50))
          continue;
        // never allow an image with less than 1 selected exist
        if ((tcase == 4) && (grid_size < 8) && (slt_pct < 10))
          continue;

        std::string base(get_data_dir());
        auto fn = base + "/test_images/";
        fn = fn + std::to_string(grid_size);
        fn = fn + "/tcase";
        fn = fn + std::to_string(tcase);
        std::string delim("/");
        fn = fn + delim;
        fn = fn + "slt_pct";
        fn = fn + std::to_string((int)slt_pct);
        fn = fn + delim;

        std::string fn_marker(base);
        fn_marker = fn_marker + "/test_markers/";
        fn_marker = fn_marker + std::to_string(grid_size);
        fn_marker = fn_marker + "/tcase";
        fn_marker = fn_marker + std::to_string(tcase);
        fn_marker = fn_marker + delim;
        fn_marker = fn_marker + "slt_pct";
        fn_marker = fn_marker + std::to_string((int)slt_pct);
        fn_marker = fn_marker + delim;

#ifdef USE_MCP3D
        VID_t selected = tol_sz * (slt_pct / (float)100); // for tcase 4
        // always select at least the root
        if (selected == 0)
          selected = 1;
        uint16_t *inimg1d = new uint16_t[tol_sz];
        // sets all to 0 for tcase 4
        create_image(tcase, inimg1d, grid_size, selected, root_vid);
        // if (tcase == 5) { // deprecated
        // selected = lattice_grid(root->vid, inimg1d, line_per_dim, grid_size);
        //}
        float actual_slt_pct = (selected / (float)tol_sz) * 100;
        cout << "    Actual num selected including root auto selection: "
          << selected << endl;
        cout << "    actual slt_pct: " << actual_slt_pct << "%" << endl;
        cout << "    for attempted slt_pct: " << slt_pct << "%" << endl;

        ASSERT_NE(inimg1d[root->vid], 0);
        ASSERT_NE(selected, 0);

        // check percent lines up
        // if the tcase can't pass this then raise the size or
        // slt pct to prevent dangerous usage
        ASSERT_NEAR(actual_slt_pct, slt_pct, 100 * EXP_DEV_LOW);

        write_tiff(inimg1d, fn, grid_size);
        delete[] inimg1d;
#endif

        // record the root
        write_marker(x, y, z, fn_marker);
      }
    }
  }
}

#ifdef USE_MCP3D
TEST(Image, ReadWrite) {
  auto grid_size = 2;
  auto tcase = 4;
  double slt_pct = 50;
  long sz0 = (long)grid_size;
  long sz1 = (long)grid_size;
  long sz2 = (long)grid_size;
  VID_t tol_sz = sz0 * sz1 * sz2;
  std::string fn(get_data_dir());
  fn = fn + "/test_images/ReadWriteTest/";
  vector<int> image_offsets = {0, 0, 0}; // zyx
  vector<int> image_extents = {grid_size, grid_size, grid_size};

  VID_t selected = tol_sz * (slt_pct / 100); // for tcase 4
  // cout << endl << "Select: " << selected << " (" << slt_pct << "%)" << endl;
  uint16_t *inimg1d = new uint16_t[tol_sz];
  create_image(tcase, inimg1d, grid_size, selected, get_central_vid(grid_size));
  write_tiff(inimg1d, fn, grid_size);
  mcp3d::MImage check, check3;
  read_tiff(fn, image_offsets, image_extents, check);
  // print_image(check, grid_size * grid_size * grid_size);
  ASSERT_NO_FATAL_FAILURE(
      check_image_equality(inimg1d, check.Volume<uint16_t>(0), grid_size));

  // run recut over the image, force it to run in read image
  // non-generated mode since MCP3D is guaranteed here
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase, false);
  auto recut = Recut<uint16_t>(args);
  recut();

  // check again
  read_tiff(fn, image_offsets, image_extents, check3);
  ASSERT_NO_FATAL_FAILURE(
      check_image_equality(inimg1d, check3.Volume<uint16_t>(0), grid_size));

  delete[] inimg1d;
}

#endif

TEST(Helpers, DISABLED_DoublePackKey) {
  {
    VID_t block_num = 5;
    VID_t nb_block_num = 13;
    uint32_t result = double_pack_key(block_num, nb_block_num);
    uint32_t actual = (uint32_t)block_num << 16;
    actual |= (uint32_t)nb_block_num;
    ASSERT_EQ(actual, result);
    // check a switch
    result = double_pack_key(nb_block_num, block_num);
    ASSERT_NE(actual, result);
  }
  {
    VID_t block_num = 1;
    VID_t nb_block_num = 1;
    uint32_t result = double_pack_key(block_num, nb_block_num);
    uint32_t actual = (uint32_t)1 << 16 | 1;
    ASSERT_EQ(actual, result);
  }
}

TEST(Helpers, DISABLED_TriplePackKey) {
  {
    VID_t interval_num = 2;
    VID_t block_num = 5;
    VID_t nb_block_num = 13;
    uint64_t result = triple_pack_key(interval_num, block_num, nb_block_num);
    uint64_t actual = (uint64_t)interval_num << 32;
    actual |= (uint64_t)block_num << 16;
    actual |= (uint64_t)nb_block_num;
    ASSERT_EQ(actual, result);
  }
  {
    VID_t interval_num = 1;
    VID_t block_num = 1;
    VID_t nb_block_num = 1;
    uint64_t result = triple_pack_key(interval_num, block_num, nb_block_num);
    uint64_t actual = (uint64_t)1 << 32 | 1 << 16 | 1;
    ASSERT_EQ(actual, result);
  }
}

#ifdef CONCURRENT_MAP
TEST(Helpers, DISABLED_ConcurrentMap) {
  // auto surface_map = std::make_unique<junction::ConcurrentMap_Leapfrog<
  // uint32_t, std::vector<VertexAttr *> *>>();
  // FIXME could be due to wrong type, std::vector in particular not allowed
  // no call to the destructor

  auto surface_map = std::make_unique<
    junction::ConcurrentMap_Leapfrog<uint32_t, VertexAttr *>>();

  std::vector<int> dim_range(2, 0);
  std::transform(dim_range.begin(), dim_range.end(), ++dim_range.begin(),
      std::bind2nd(plus<int>(), 1));

  for (VID_t interval_num : dim_range) {
    for (VID_t block_num : dim_range) {

      uint32_t key = double_pack_key(interval_num, block_num);
      cout << "interval_num " << interval_num << " block_num " << block_num
        << "key " << key << '\n';
      //{
      auto mutator = surface_map->insertOrFind(key);
      auto v = mutator.getValue();
      ASSERT_EQ(v, nullptr);
      // std::vector<struct VertexAttr *> *vec = mutator.getValue();
      // ASSERT_EQ(vec, nullptr);
      // if (!vec) {
      // vec = new std::vector<struct VertexAttr *>;
      //}
      v = new VertexAttr();
      v->vid = static_cast<VID_t>(key);
      // vec->push_back(v);
      // Note: this block is unique to a single thread, therefore no other
      // thread could have this same key, since the keys are unique with
      // respect to their permutation. This means that we do not need to
      // protect from two threads modifying the same key simultaneously
      // in this design. If this did need to protected from see documentation
      // at preshing.com/20160201/new-concurrent-hash-maps-for-cpp/ for
      // details
      mutator.exchangeValue(v); // assign via mutator vs. relookup
      // auto post = new std::vector<VertexAttr *>();
      // surface_map->assign(key, &(std::move(*post)));
      //}

      // std::vector<VertexAttr *> *check = surface_map->get(key);
      auto mutator2 = surface_map->insertOrFind(key);
      auto check = mutator2.getValue();
      ASSERT_NE(check, nullptr);
      ASSERT_EQ(check->vid, static_cast<VID_t>(key));
    }
  }

  for (VID_t interval_num : dim_range) {
    for (VID_t block_num : dim_range) {
      uint32_t key = double_pack_key(interval_num, block_num);
      cout << "interval_num " << interval_num << " block_num " << block_num
        << "key " << key << '\n';
      auto mutator = surface_map->insertOrFind(key);
      auto v = mutator.getValue();
      // std::vector<VertexAttr *> *vec = mutator.getValue();
      // ASSERT_NE(vec, nullptr);
      // VertexAttr *v = vec->back();
      ASSERT_NE(v, nullptr);
      ASSERT_EQ(v->vid, static_cast<VID_t>(key));
    }
  }
}
#endif

TEST(VertexAttr, ReadWriteInterval) {
  auto nvid = 4;
  auto ptr = new VertexAttr[nvid];
  size_t size = sizeof(VertexAttr) * nvid;
  // path logic
  auto base = get_data_dir() + "/test_data/";
  auto fn = base + "interval0.bin";
  // fs::remove_all(fn); // make sure it's an overwrite
  fs::create_directories(base);
  // cout << "fn: " << fn << endl;

  // open output
  std::ofstream ofile(fn, ios::out | ios::binary); // read-mode
  ASSERT_TRUE(ofile.is_open());
  ASSERT_TRUE(ofile.good());

  // write struct to file
  ofile.write((char *)ptr, size);
  ASSERT_TRUE(ofile.good());

  // close file
  ofile.close();

  VertexAttr *iptr = (VertexAttr *)malloc(size);
  std::ifstream ifile(fn,
      ios::in | ios::binary | ios::ate); // write-mode, end
  // open input
  ASSERT_TRUE(ifile.is_open());
  ASSERT_TRUE(ifile.good());
  auto rsize = ifile.tellg();
  ASSERT_EQ(rsize, size);

  ifile.seekg(0, ios::beg); // back to beginning
  ifile.read((char *)iptr, size);
  for (int i = 0; i < nvid; i++) {
    ASSERT_EQ(iptr[i], ptr[i]);
  }
  // close file
  ifile.close();

  delete[] ptr;
  free(iptr);
}

TEST(VertexAttr, Defaults) {
  auto v1 = new VertexAttr();
  ASSERT_FALSE(v1->root());
  ASSERT_EQ(v1->edge_state.field_, 192);
  ASSERT_TRUE(v1->unselected());
  ASSERT_TRUE(v1->unvisited());
  // FIXME reuse this test once, root marked as known new found
  // ASSERT_TRUE(v1->connections(1, 1).empty());
}

TEST(TileThresholds, AllTcases) {
  auto grid_size = 4;
  auto slt_pct = 50;
  auto grid_vertex_size = grid_size * grid_size * grid_size;
  auto print_image = true;
  std::vector<int> tcases = {1, 2, 3, 4, 5, 6};
  for (const auto &tcase : tcases) {
    auto args =
      get_args(grid_size, grid_size, grid_size, slt_pct, tcase, false);
    auto selected = args.recut_parameters().selected;
    auto recut = Recut<uint16_t>(args);
    auto inimg1d = std::make_unique<uint16_t[]>(grid_vertex_size);
    auto tile_thresholds = new TileThresholds<uint16_t>();
    uint16_t bkg_thresh;

#ifdef USE_MCP3D
    mcp3d::MImage image;
    if (tcase == 6) {
      read_tiff(args.image_root_dir(), args.image_offsets, args.image_extents,
          image);
      if (print_image) {
        print_image_3D(image.Volume<uint16_t>(0),
            {grid_size, grid_size, grid_size});
      }
      bkg_thresh = recut.get_bkg_threshold(image.Volume<uint16_t>(0),
          grid_vertex_size, slt_pct / 100.);
      tile_thresholds->get_max_min(image.Volume<uint16_t>(0), grid_vertex_size);

      cout << "tcase " << tcase << " bkg_thresh " << bkg_thresh << "\nmax "
        << tile_thresholds->max_int << " min " << tile_thresholds->min_int
        << '\n';
    } 
#endif

    if (tcase <=5) {
      create_image(tcase, inimg1d.get(), grid_size, selected,
          get_central_vid(grid_size));
      print_image_3D(inimg1d.get(), {grid_size, grid_size, grid_size});
      bkg_thresh =
        recut.get_bkg_threshold(inimg1d.get(), grid_vertex_size, slt_pct / 100.);
      tile_thresholds->get_max_min(inimg1d.get(), grid_vertex_size);

      cout << "tcase " << tcase << " bkg_thresh " << bkg_thresh << "\nmax "
        << tile_thresholds->max_int << " min " << tile_thresholds->min_int
        << '\n';
    }
  }
}

TEST(VertexAttr, MarkStatus) {
  auto v1 = new VertexAttr();
  v1->mark_root(0);
  ASSERT_TRUE(v1->root());

  v1->mark_band(0);
  ASSERT_FALSE(v1->selected());
  ASSERT_FALSE(v1->unvisited());
  ASSERT_FALSE(v1->root());
  ASSERT_TRUE(v1->unselected());

  v1->mark_selected();
  ASSERT_TRUE(v1->selected());
  ASSERT_FALSE(v1->unvisited());
  ASSERT_FALSE(v1->root());
  ASSERT_FALSE(v1->unselected());
}

TEST(VertexAttr, CopyOp) {
  auto v1 = new VertexAttr();
  auto v2 = new VertexAttr();
  v1->value = 1;
  v1->vid = 1;
  v1->edge_state.reset();
  v1->handle = 1;
  ASSERT_NE(*v1, *v2);
  *v2 = *v1;
  ASSERT_EQ(*v1, *v2);             // same values
  ASSERT_EQ(v1->value, v2->value); // check those values manually
  ASSERT_EQ(v1->vid, v2->vid);
  ASSERT_NE(v1->handle, v2->handle); // handles should never be copied
  ASSERT_EQ(v1->edge_state.field_, v2->edge_state.field_);
  ASSERT_NE(v1, v2); // make sure they not just the same obj
}

TEST(RecutPipeline, DISABLED_PrintDefaultInfo) {
  auto v1 = new VertexAttr();
  auto ps = sysconf(_SC_PAGESIZE);
  auto vs = sizeof(VertexAttr);
  cout << "sizeof vidt " << sizeof(VID_t) << " bytes" << std::scientific
    << endl;
  cout << "sizeof float " << sizeof(float) << " bytes" << endl;
  cout << "sizeof bitfield " << sizeof(bitfield) << " bytes" << endl;
  cout << "sizeof vertex " << vs << " bytes" << endl;
  cout << "sizeof 1024^3 interval " << sizeof(VertexAttr) << " GB" << endl;
  cout << "page size " << ps << " B" << endl;
  cout << "VertexAttr vertices per page " << ps / vs << endl;
  cout << "cube root of vertices per page " << (int)cbrt(ps / vs) << endl;
  cout << "AvailMem " << GetAvailMem() / (1024 * 1024 * 1024) << " GB" << endl;
  cout << "MAX_INTERVAL_VERTICES " << MAX_INTERVAL_VERTICES << std::scientific
    << endl;
  cout << "Vertices needed for a 1024^3 interval block size 4 : "
    << get_used_vertex_size(1024, 4) << std::scientific << endl;
  cout << "Vertices needed for a 2048^3 interval block size 4 : "
    << get_used_vertex_size(2048, 4) << std::scientific << endl;
}

TEST(Radius, Full) {
  bool print_all = true;
  bool check_xy = false;
  bool print_csv = false;

  int max_size = 8;
  // std::vector<int> grid_sizes = {max_size / 16, max_size / 8, max_size / 4,
  // max_size / 2, max_size};
  std::vector<int> grid_sizes = {max_size};
  std::vector<int> interval_sizes = {max_size};
  std::vector<int> block_sizes = {max_size};
  // tcase 5 is a sphere of radius grid_size / 4 centered
  // in the middle of an image
  std::vector<int> tcases = {5};
  int slt_pct = 100;
  uint16_t bkg_thresh = 0;
  std::unique_ptr<uint16_t[]> radii_grid_xy;
  if (print_csv) {
    cout << "name,iterations,error_rate(%)\n";
  }
  for (auto &grid_size : grid_sizes) {
    VID_t tol_sz = (VID_t)grid_size * grid_size * grid_size;
    auto radii_grid = std::make_unique<uint16_t[]>(tol_sz);
    if (check_xy) {
      radii_grid_xy = std::make_unique<uint16_t[]>(tol_sz);
    }
    for (auto &interval_size : interval_sizes) {
      for (auto &block_size : block_sizes) {
        for (auto &tcase : tcases) {
          auto args = get_args(grid_size, interval_size, block_size, slt_pct,
              tcase, true);

          // the total number of blocks allows more parallelism
          // ideally intervals >> thread count
          auto final_interval_size =
            interval_size > grid_size ? grid_size : interval_size;
          auto final_block_size = block_size > final_interval_size
            ? final_interval_size
            : block_size;

          std::ostringstream iteration_trace;
          // use this to tag and reconstruct data from json file
          iteration_trace << "grid_size " << grid_size << " interval_size "
            << final_interval_size << " block_size "
            << final_block_size;
          SCOPED_TRACE(iteration_trace.str());

          // run
          auto recut = Recut<uint16_t>(args);
          auto root_vids = recut.initialize();
          auto selected = args.recut_parameters().selected;

          if (print_all) {
            std::cout << "recut image grid" << endl;
            print_image_3D(recut.generated_image,
                {grid_size, grid_size, grid_size});
          }

          recut.setup_value(root_vids);
          recut.update("value");
          if (print_all) {
            recut.print_grid("value");
            recut.print_grid("surface");
            cout << "All surface vids: \n";
            for (auto &outer : recut.surface_vec) {
              for (auto &inner : outer) {
                for (auto &v : inner) {
                  cout << '\t' << v->vid << '\n';
                }
              }
            }
          }

          // Get accurate and approximate radii according to APP2
          // methods
          auto total_visited = 0;
          for (VID_t i = 0; i < tol_sz; i++) {
            if (recut.generated_image[i]) {
              // calculate radius with baseline accurate method
              radii_grid[i] = get_radius_accurate(recut.generated_image,
                  grid_size, i, bkg_thresh);
              if (check_xy) {
                // build original production version
                radii_grid_xy[i] = get_radius_hanchuan_XY(
                    recut.generated_image, grid_size, i, bkg_thresh);
              }
              ++total_visited;
            }
          }
          ASSERT_EQ(total_visited, selected);

          // make sure all surface vertices were identified correctly
          double xy_err, recut_err;
          ASSERT_NO_FATAL_FAILURE(check_recut_error(
                recut, radii_grid.get(), grid_size, "surface", recut_err));

          recut.setup_radius();
          // conducting update on radius consumes all recut.surface_vec values
          recut.update("radius");
          VID_t interval_num = 0;

          // Debug by eye
          if (print_all) {
            cout << "accuracy_radius\n";
            print_image_3D(radii_grid.get(), {grid_size, grid_size, grid_size});
            if (check_xy) {
              std::cout << "XY radii grid\n";
              print_image_3D(radii_grid_xy.get(),
                  {grid_size, grid_size, grid_size});
            }
            std::cout << "Recut radii\n";
            recut.print_grid("radius");
          }

          if (check_xy) {
            ASSERT_NO_FATAL_FAILURE(check_image_error(
                  recut.generated_image, radii_grid.get(), radii_grid_xy.get(),
                  grid_size, recut.params->selected, xy_err));
          }
          ASSERT_NO_FATAL_FAILURE(check_recut_error(
                recut, radii_grid.get(), grid_size, "radius", recut_err));
          ASSERT_NEAR(recut_err, 0., .001);

          if (print_csv) {
            if (check_xy) {
              std::cout << "\"xy_radius/" << grid_size << "\",1," << xy_err
                << '\n';
            }
            std::cout << "\"fast_marching_radius/" << grid_size << "\",1,"
              << recut_err << '\n';
          }
          std::ostringstream xy_stream, recut_stream;
          xy_stream << "XY Error " << iteration_trace.str();
          recut_stream << "Recut Error " << iteration_trace.str();
          RecordProperty(xy_stream.str(), xy_err);
          RecordProperty(recut_stream.str(), recut_err);
        }
      }
    }
  }
}

class RecutPipelineParameterTests
: public ::testing::TestWithParam<
  std::tuple<int, int, int, int, double, bool, bool>> {};

TEST_P(RecutPipelineParameterTests, ChecksIfFinalVerticesCorrect) {
  // record_property_pp_macros
#ifdef NO_INTERVAL_RV
  RecordProperty("NO_INTERVAL_RV", 1);
#endif
#ifdef SCHEDULE_INTERVAL_RV
  RecordProperty("SCHEDULE_INTERVAL_RV", 1);
#endif
#ifdef USE_OMP_BLOCK
  RecordProperty("USE_OMP_BLOCK", 1);
#endif
#ifdef USE_OMP_INTERVAL
  RecordProperty("USE_OMP_INTERVAL", 1);
#endif
#ifdef MMAP
  RecordProperty("MMAP", 1);
#endif
#ifdef NO_RV
  RecordProperty("NO_RV", 1);
#endif
#ifdef CONCURRENT_MAP
  RecordProperty("CONCURRENT_MAP", 1);
#endif
#ifdef FULL_PRINT
  // this significantly slows performance so it should be stamped to any performance stats
  RecordProperty("FULL_PRINT", 1);
#endif
#ifdef USE_HUGE_PAGE
  RecordProperty("USE_HUGE_PAGE", 1);
#endif
#ifdef USE_MCP3D
  RecordProperty("USE_MCP3D", 1);
#endif

  // documents the meaning of each tuple member
  auto grid_size = std::get<0>(GetParam());
  RecordProperty("grid_size", grid_size);
  auto interval_size = std::get<1>(GetParam());
  RecordProperty("interval_size", interval_size);
  auto block_size = std::get<2>(GetParam());
  RecordProperty("block_size", block_size);
  auto tcase = std::get<3>(GetParam());
  RecordProperty("tcase", tcase);
  double slt_pct = std::get<4>(GetParam());
  RecordProperty("slt_pct", slt_pct);
  bool check_against_selected = std::get<5>(GetParam());
  bool check_against_sequential = std::get<6>(GetParam());
  // regenerating image is random and done at run time
  // if you were to set to true tcase 4 would have a mismatch
  // with the loaded image
  bool force_regenerate_image = false;

  // shared params
  auto args = get_args(
      grid_size, interval_size, block_size, slt_pct, tcase,
      force_regenerate_image); // generate image so that you can read it below
  args.recut_parameters();
  // uint16_t is image_t here
  TileThresholds<uint16_t> *tile_thresholds;
  bool print_image = false;

  auto recut = Recut<uint16_t>(args);
  std::vector<VID_t> root_vids;
  root_vids = recut.initialize();
  recut.setup_value(root_vids);

#ifdef USE_MCP3D
  mcp3d::MImage image;
  if (check_against_sequential) {
    read_tiff(args.image_root_dir(), args.image_offsets,
        args.image_extents, image);

    if (print_image) {
      print_image_3D(image.Volume<uint16_t>(0),
          {grid_size, grid_size, grid_size});
    }
  }
#endif

  // establish the tile thresholds for the entire test run (recut and
  // sequential)
  if (tcase == 6) {
    //tile_thresholds = recut.get_tile_thresholds(image);
    // bkg_thresh table: 421 ~.01 foreground
    // if any pixels are found above or below these values it will fail
    tile_thresholds = new TileThresholds<uint16_t>(30000, 0, 421);
  } else {
    // note these default thresholds apply to any generated image
    // thus they will only be replaced if we're reading a real image
    tile_thresholds = new TileThresholds<uint16_t>(2, 0, 0);
  }
  RecordProperty("bkg_thresh", tile_thresholds->bkg_thresh);
  RecordProperty("max_int", tile_thresholds->max_int);
  RecordProperty("min_int", tile_thresholds->min_int);

  // update with fixed tile_thresholds for the entire update
  auto update_stats = recut.update("value", tile_thresholds);
  auto interval_open_count = update_stats->interval_open_count;
  // print vector
  std::ostringstream cat;
  std::copy(interval_open_count.begin(), interval_open_count.end(), std::ostream_iterator<int>(cat, ", "));
  std::cout << "interval reopens: " << cat.str() << '\n';

  auto sum_opens = std::accumulate(interval_open_count.begin(), interval_open_count.end(), 0);
  RecordProperty("Total tile reads", sum_opens);
  RecordProperty("Total intervals", interval_open_count.size());
  auto mean = sum_opens / (double)interval_open_count.size();
  RecordProperty("Mean tile reads", mean);

  recut.finalize(args.output_tree); // this fills args.output_tree
  //cout << "recut update no IO elapsed (s)" << recut_update_value_elapsed << '\n';
  double actual_slt_pct =
    (100. * args.output_tree.size()) / (grid_size * grid_size * grid_size);
  cout << "Selected " << actual_slt_pct << "% of pixels\n";
  RecordProperty("Foreground (%)", actual_slt_pct);
  RecordProperty("Foreground count (pixels)", args.output_tree.size());
  RecordProperty("Selected vertices/s", args.output_tree.size() / update_stats->total_time);
  RecordProperty("Selected vertices / computation (s)", args.output_tree.size() / update_stats->computation_time);
  RecordProperty("Selected vertices / IO (s)", args.output_tree.size() / update_stats->io_time);
  RecordProperty("Iterations", update_stats->iterations);
  RecordProperty("Value update computation (s)", update_stats->computation_time);
  RecordProperty("Value update IO (s)", update_stats->io_time);
  RecordProperty("Value update elapsed (s)", update_stats->total_time);
  RecordProperty("Value update computation / IO ratio", update_stats->computation_time / update_stats->io_time);
  RecordProperty("Grid / interval ratio", grid_size / interval_size);

  // pregenerated data has a known number of selected
  // pixels
  if (check_against_selected) {
    ASSERT_EQ(args.output_tree.size(), recut.params->selected);
  }

#ifdef USE_MCP3D
  // this runs the original sequential fastmarching algorithm
  // when using the real data, you don't know what the actual
  // selected number should be unless you compare it to another
  // reconstruction method or manual ground truth
  if (check_against_sequential) {
    // convert roots into markers (vector)
    std::vector<MyMarker *> root_markers;
    if (tcase == 6) {
      root_markers = vids_to_markers(root_vids, grid_size);
    } else {
      root_markers = {get_central_root(grid_size)};
    }

    std::vector<MyMarker *> sequential_output_tree;
    std::vector<MyMarker> targets;
    auto timer = new high_resolution_timer();
    fastmarching_tree(root_markers, targets, image.Volume<uint16_t>(0),
        sequential_output_tree, grid_size, grid_size, grid_size,
        1, tile_thresholds->bkg_thresh, tile_thresholds->max_int,
        tile_thresholds->min_int);

    cout << "sequential fastmarching elapsed (s)" << timer->elapsed() << '\n';
    // warning record property will auto cast to an int
    RecordProperty("Sequential elapsed (s)", timer->elapsed());
    RecordProperty("Recut speedup factor %", 100 * (update_stats->total_time / timer->elapsed()) ) ;

    //// would have to keep interval in memory for this to work
    // if ( sequential_output_tree.size() != args.output_tree.size()) {
    // cout << "recut's label did not match sequential:\n";
    // recut.print_grid("label");
    //}

    RecordProperty("Sequential tree size", sequential_output_tree.size());
    auto diff = absdiff(sequential_output_tree.size(), args.output_tree.size());
    RecordProperty("Error", diff);
    RecordProperty("Error rate (%)", 100 * (diff / sequential_output_tree.size()));
    ASSERT_EQ(sequential_output_tree.size(), args.output_tree.size());
  }
#endif
}

// ... check_against_selected, check_against_sequential
INSTANTIATE_TEST_CASE_P(
    RecutPipelineTests, RecutPipelineParameterTests,
    ::testing::Values(
      std::make_tuple(4, 4, 4, 0, 100., true, false), // 0
      std::make_tuple(4, 4, 4, 1, 100., true, false), // 1
      std::make_tuple(4, 4, 4, 2, 100., true, false), // 2
      std::make_tuple(4, 4, 4, 3, 100., true, false) // 3
#ifdef USE_MCP3D
      ,
      // check_against_sequential (final boolean below) currently uses MCP3D's reading of
      // image to ensure that both sequential and recut use the same
      // iamge, to include these test while testing against sequential
      // change the implementation so that the generated image from recut is saved and pass
      // it to fastmarching_tree
      std::make_tuple(4, 4, 4, 4, 50., true, true), // 4
      // multi-interval small
      std::make_tuple(4, 2, 2, 4, 50., true, true), // 5
      // multi-block small
      std::make_tuple(4, 4, 2, 4, 50., true, true),  // 6
      // make sure if bkg_thresh is 0, all vertices are selected for real
      std::make_tuple(4, 4, 4, 6, 100., true, true), // 7
      // make sure fastmarching_tree and recut produce exact match for real
      std::make_tuple(8, 8, 8, 6, 100., false, true), // 8
      // real data multi-block
      std::make_tuple(8, 8, 4, 6, 100., false, true), // 9
      // real data multi-interval
      std::make_tuple(8, 4, 4, 6, 100., false, true) // 10
#ifdef TEST_ALL_BENCHMARKS // test larger portions that must be verified for
      ,
      // interval grid ratio tests
      std::make_tuple(128, 64, 64, 6, 1, false, true), // 11
      std::make_tuple(256, 128, 128, 6, 1, false, true), // 12
      std::make_tuple(512, 256, 256, 6, 1, false, true), // 13
      std::make_tuple(1024, 512, 512, 6, 1, false, true), // 14
      std::make_tuple(2048, 1024, 1024, 6, 1, false, false), // out of memory for fastmarching_tree
      std::make_tuple(4096, 2048, 2048, 6, 1, false, false), // 16
      std::make_tuple(8192, 4096, 4096, 6, 1, false, false), // 17

      std::make_tuple(128, 32, 32, 6, 1, false, true), // 18
      std::make_tuple(256, 64, 64, 6, 1, false, true),
      std::make_tuple(512, 128, 128, 6, 1, false, true),
      std::make_tuple(1024, 256, 256, 6, 1, false, true),
      std::make_tuple(2048, 512, 512, 6, 1, false, false),
      std::make_tuple(4096, 1024, 1024, 6, 1, false, false),
      std::make_tuple(8192, 2048, 2048, 6, 1, false, false), // 24

      std::make_tuple(128, 16, 16, 6, 1, false, true), // 25
      std::make_tuple(256, 32, 32, 6, 1, false, true),
      std::make_tuple(512, 64, 64, 6, 1, false, true),
      std::make_tuple(1024, 128, 128, 6, 1, false, true),
      std::make_tuple(2048, 256, 256, 6, 1, false, false),
      std::make_tuple(4096, 512, 512, 6, 1, false, false),
      std::make_tuple(8192, 1024, 1024, 6, 1, false, false), // 31 // 31

      // record for benchmarks
      std::make_tuple(16, 16, 16, 6, 1, false, true),
      std::make_tuple(32, 32, 32, 6, 1, false, true),
      std::make_tuple(64, 64, 64, 6, 1, false, true),
      std::make_tuple(128, 128, 128, 6, 1, false, true),
      std::make_tuple(256, 256, 256, 6, 1, false, true),
      std::make_tuple(512, 512, 512, 6, 1, false, true),
      std::make_tuple(1024, 1024, 1024, 6, 1, false, true)
#endif
#endif
      ));

      int main(int argc, char **argv) {
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
      }
