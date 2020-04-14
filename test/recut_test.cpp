#include "gtest/gtest.h"
#include "../src/recut.hpp"
//#include "../external_tools/vaa3d/neuron_tracing/all_path_pruning2/heap.h"
//#include "fastmarching_tree.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <fcntl.h>
#include <bits/stdc++.h>
#include <cstdlib> //rand 
#include <ctime> // for srand

#ifdef USE_MCP3D
#include <opencv2/opencv.hpp> // imwrite
#include "image/mcp3d_voxel_types.hpp" // convert to CV type
#include "common/mcp3d_utility.hpp" // PadNumStr
#endif

#define EXP_DEV_LOW .05

#ifdef USE_MCP3D
#define GEN_IMAGE false
#else
#define GEN_IMAGE true
#endif

// This is for functions or tests that include test macros only
// note defining a function that contains test function MACROS
// that returns a value will give strange: void value not ignored as it ought to be errors

//VID_t compare_tree(const T seq, const T2 parallel, long
    //sz0, sz1, sz2) {
  //// get min of both sizes
  //VID_t tol_sz = seq.size() > parallel.size() ? parallel.size() : seq.size();
  //for (int i=0; i < tol_sz; i++) {
    //auto s = seq[i];
    //auto p = parallel[i];
    //VID_t sid = (VID_t) s->z * sz0 * sz1 + s->y * sz0 + s->x;
  //}
//}

template <class T, typename DataType>
void check_recut_error(T& recut, DataType* data, int grid_size, VID_t interval_num,
    std::string stage, double& error_rate) {
  auto tol_sz = static_cast<VID_t>(grid_size) * grid_size * grid_size;
  //cout << "check image " << endl;
  double error_sum = 0.0;
  VID_t total_valid = 0;
  auto interval = recut.super_interval.GetInterval(interval_num);
  assertm(interval->IsInMemory(), "Can not print interval not in memory");
  for (int zi=0; zi < recut.z_interval_size; zi++) {
    for (int yi=0; yi < recut.y_interval_size; yi++) {
      for (int xi=0; xi < recut.x_interval_size; xi++) {
        VID_t index = ((VID_t) xi) + yi * recut.x_interval_size + zi * recut.x_interval_size * recut.y_interval_size;
        VertexAttr* v = recut.get_attr_vid(interval_num, 0, index, nullptr);
        //cout << i << endl << v.description() << " ";
        if (stage == "value") {
          bool valid_value = v->value != std::numeric_limits<uint8_t>::max();
          if (recut.generated_image[index]) {
            ASSERT_TRUE(valid_value);
          }
          if (v->valid_value()) {
            ASSERT_EQ(recut.generated_image[index], 1);
          }
          if (v->valid_value()) {
            //error_sum += absdiff(data[index], v->value);
            //cout << v->value << " ";
          } 
        } else if (stage == "radius") {
          if (recut.generated_image[index]) {
            ASSERT_TRUE(v->valid_radius());
          }
          if (v->valid_radius()) {
            ASSERT_EQ(recut.generated_image[index], 1);
          }
          //if (v->valid_radius()) {
          if (recut.generated_image[index]) {
            error_sum += absdiff(data[index], v->radius);
            ++total_valid;
            //cout << +(v->radius) << " ";
          } 
        }
      }
    }
  }
  ASSERT_EQ(total_valid, recut.params->selected);
  error_rate = 100* error_sum / static_cast<double>(tol_sz);
}

void check_image_error(uint16_t* inimg1d, uint16_t* baseline, uint16_t* check,int grid_size,
    VID_t selected, double& error_rate) {
  auto tol_sz = grid_size * grid_size * grid_size;
  VID_t total_valid = 0;
  //cout << "check image " << endl;
  VID_t error_sum = 0;
  for (VID_t i=0; i < tol_sz; i++) {
    if (inimg1d[i]) {
      //cout << i << " " << +inimg1d[i] << endl;
      error_sum += absdiff(baseline[i], check[i]);
      ++total_valid;
    }
  }
  ASSERT_EQ(total_valid, selected);
  error_rate = 100* error_sum / static_cast<double>(tol_sz);
}

void check_image_equality(uint16_t* inimg1d, uint16_t* check,int grid_size) {
  //cout << "check image " << endl;
  for (VID_t i=0; i < (grid_size * grid_size * grid_size); i++) {
    //cout << i << " " << +inimg1d[i] << endl;
    ASSERT_LE(inimg1d[i] , 1);
    ASSERT_EQ(inimg1d[i], check[i]);
  }
}

// make sure base interval is implemented in a read-only manner
void interval_base_immutable(VID_t nvid) {
  ASSERT_LE(nvid, MAX_INTERVAL_VERTICES);
  // check that interval_base has not changed
  auto base = INTERVAL_BASE;
  auto interval = new Interval(0, nvid, 0, base);
  interval->LoadFromDisk();
  for (int i=0; i < nvid; i++) {
    ASSERT_TRUE(interval->GetData()[i].unvisited()) << " i= " << i << " vid " << interval->GetData()[i].vid << endl;
    ASSERT_FALSE(interval->GetData()[i].valid_vid());
    ASSERT_FALSE(interval->GetData()[i].valid_value());
  }
}

void load_save(bool mmap_) {
  auto nvid = 64;
  auto fn = get_curr() + "/interval0.bin";
  auto base = INTERVAL_BASE;
  remove(fn); // for safety, but done automatically via ~Interval
  ASSERT_FALSE(exists(fn));
  ASSERT_TRUE(exists(base));
  
  // create
  auto interval = new Interval(0, nvid, 0, base, mmap_);
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
  ASSERT_TRUE(exists(fn));

  //load and check
  interval->LoadFromDisk();
  ASSERT_EQ(interval->GetFn(), fn);
  ASSERT_TRUE(interval->IsInMemory());
  ASSERT_TRUE(interval->GetData()[1].root());
  ASSERT_EQ(interval->GetData()[1].vid, 1);
  ASSERT_TRUE(interval->GetData()->unvisited());
  delete interval;
  ASSERT_FALSE(exists(fn));

  // check interval can't delete interval_base_64bit.bin
  // create
  auto interval2 = new Interval(0, nvid, 0, base, mmap_);
  interval2->LoadFromDisk();
  delete interval2;
    
  ASSERT_FALSE(exists(fn));
  ASSERT_TRUE(exists(base));

  interval_base_immutable(nvid) ;
}

//// assigns all of the intervals within the super interval
//// a unique vid even in padded regions. This is especially
//// useful for testing get_attr_vid is returning the correct
//// vid 
//void assign_super_interval_vids(Recut recut) {
  //// test that get_attr_vid returns the right match in various scenarios
  //// assign all vids for testing purposes
  //VID_t index = 0;
  //for (int i=0; i < recut.super_interval.GetNIntervals(); i++) {
    //auto interval = recut.super_interval.GetInterval(i);
    //auto attr = interval->GetData();
    //for (int j=0; j < interval->GetNVertices(); j++) {
      //auto current = attr + index; // get the appropriate vertex
      //current->vid = index;
      //index ++;
    //}
  //}
//}

void test_get_attr_vid(bool mmap, int grid_size, int interval_size, int block_size)  {
  auto nvid = grid_size * grid_size * grid_size;
  double slt_pct = 100;
  int tcase = 0;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto params = args.recut_parameters();
  params.set_block_size(block_size);
  params.set_interval_size(interval_size);
  args.set_recut_parameters(params);

  auto root_vid = get_central_vid(grid_size);
  //x + 1, y + 1, z + 1 to the other central
  VID_t root_diag_vid = get_central_diag_vid(grid_size);
  std::vector<VID_t> vids = {root_vid, root_diag_vid};
  Recut recut = Recut<uint16_t>(args);
  recut.mmap_ = mmap;

  recut.initialize();
  recut.setup_value();

  ASSERT_EQ(recut.nx, grid_size);
  ASSERT_EQ(recut.ny, grid_size);
  ASSERT_EQ(recut.nz, grid_size);

  // these tests make sure get_attr_vid always returns a pointer to underlying vertex
  // data and does not copy by value the vertex struct
  for (auto& vid : vids) {
    //cout << "TESTING vid " << vid << endl;
    ASSERT_LT(vid, nvid);
    auto current_interval_num = recut.get_interval_num(vid);
    auto current_block_num = recut.get_block_num(vid);
    std::vector<VID_t> test_blocks = {};
    test_blocks.push_back(current_block_num);
    std::vector<VID_t> test_intervals = {};
    test_intervals.push_back(current_interval_num);

    if (grid_size / interval_size == 2) {
      VID_t iinterval, jinterval, kinterval; 
      // the interval is a linear index into the 3D row-wise arrangement of
      // interval, converting to subscript makes adjustments easier
      recut.get_interval_subscript(current_interval_num, iinterval, jinterval, kinterval);
      if (vid == root_vid) {
        test_intervals.push_back(recut.get_interval_id(iinterval + 1, jinterval, kinterval));
        test_intervals.push_back(recut.get_interval_id(iinterval, jinterval + 1, kinterval));
        test_intervals.push_back(recut.get_interval_id(iinterval, jinterval, kinterval + 1));
      } else if (vid == root_diag_vid) {
        test_intervals.push_back(recut.get_interval_id(iinterval - 1, jinterval, kinterval));
        test_intervals.push_back(recut.get_interval_id(iinterval, jinterval - 1, kinterval));
        test_intervals.push_back(recut.get_interval_id(iinterval, jinterval, kinterval - 1));
      }
    } else if (interval_size / block_size == 2) {
      VID_t iblock, jblock, kblock; 
      // the block_num is a linear index into the 3D row-wise arrangement of
      // blocks, converting to subscript makes adjustments easier
      recut.get_block_subscript(current_block_num, iblock, jblock, kblock);
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
    for (auto& interval_num : test_intervals ) {
      //cout << "interval_num " << interval_num << endl;
      recut.super_interval.GetInterval(interval_num)->LoadFromDisk();
      for (auto& block_num : test_blocks ) {
        //cout << "\tblock_num " << block_num << endl;
        // TEST 1
        // Assert upstream changes (by pointer)
        auto attr = recut.get_attr_vid(interval_num, block_num, vid, &output_offset);
        attr->vid = vid;
        attr->value = 3;
        auto attr2 = recut.get_attr_vid(interval_num, block_num, vid, &output_offset);
        ASSERT_EQ(attr , attr2); //upstream changes?

        // TEST 2
        // assert get_attr_vid can handle dummy structs added in to heaps
        // and update real embedded values in interval accordingly
        VertexAttr* dummy_attr = new VertexAttr(); // march is protect from dummy values like this
        dummy_attr->mark_root(0); // 0000 0000, selected no parent, all zeros indicates KNOWN_FIX root
        dummy_attr->value = 1.0;
        dummy_attr->vid = vid;
        auto embedded_attr = recut.get_attr_vid(interval_num, block_num, dummy_attr->vid, &output_offset);
        *embedded_attr = *dummy_attr;
        ASSERT_EQ(embedded_attr->vid , dummy_attr->vid);
        auto embedded_attr2 = recut.get_attr_vid(interval_num, block_num, dummy_attr->vid, &output_offset);
        ASSERT_EQ(embedded_attr2 , embedded_attr); // same obj?
        ASSERT_EQ(embedded_attr2->vid , dummy_attr->vid);
        ASSERT_EQ(*embedded_attr2 , *dummy_attr); // same values?
      }
      recut.super_interval.GetInterval(interval_num)->Release();
      ASSERT_FALSE(recut.super_interval.GetInterval(interval_num)->IsInMemory());
    }
  }
}

TEST (Heap, PushUpdate) {
  VID_t N = 1<<10;
  bool update_values = true;

  std::vector<std::string> stages = {"value", "radius"};
  //float mval = std::numeric_limits<float>::max();
  float mval = 255;
  srand(time(0));
  std::vector<uint8_t> vr;
  std::vector<float> vv;
  for (auto& stage : stages) {
    NeighborHeap<VertexAttr> heap;
    uint8_t radius;
    float value;
    uint8_t updated_radius;
    float updated_value;
    for (VID_t i=0; i < N; i++) {
      auto vert = new VertexAttr;
      ASSERT_FALSE(vert->valid_handle());
      if (stage == "radius" ) {
        radius = (uint8_t) rand() % std::numeric_limits<uint8_t>::max();
        updated_radius = (uint8_t) rand() % std::numeric_limits<uint8_t>::max();
        vert->radius = radius;
        vr.push_back(radius);
        ASSERT_EQ(vert->radius, radius);
      } else if (stage == "value") {
        value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/mval));
        updated_value = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/mval));
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
        //cout << "elems pre-update: " << " i " << i << endl;
        //heap.print(stage);

        if (stage == "radius" ) {
          //vert->radius = updated_radius;
          vr.pop_back();
          vr.push_back(updated_radius);
          heap.update(vert, 0, updated_radius, stage);
        } else {
          //vert->value = updated_value;
          vv.pop_back();
          vv.push_back(updated_value);
          heap.update(vert, 0, updated_value, stage);
        }
      }

      //cout << "elems: " << endl;
      //heap.print(stage);
    }

    // make sure all in heap are popped in 
    // non-increasing order
    if (stage == "radius") {
      sort(vr.begin(), vr.end());
      for (auto& val : vr) {
        auto min_attr = heap.pop(0, stage);
        ASSERT_FALSE(min_attr->valid_handle());
        auto hval = min_attr->radius;
        ASSERT_EQ(val, hval) << "hval " << +hval << " val " << +val;
      }
    } else {
      sort(vv.begin(), vv.end());
      for (auto& val : vv) {
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
TEST (Install, CreateIntervalBase) {
  bool rerun = false; // change to true if MAX_INTERVAL_VERTICES or the vertex struct has been changed in vertex_attr.h
  auto fn = INTERVAL_BASE;
  VID_t nvid = MAX_INTERVAL_VERTICES;
  bool mmap_flag = false;
  if (!exists(fn) || rerun) {
    VID_t size = sizeof(VertexAttr) * nvid;
    cout << "Start creating " << fn << " cached total size: ~" << (size /
        (2<<19)) << " MB" << " for max vertices: " <<
      nvid << endl; // set the default values

    // make sure it's an overwrite
    if (exists(fn)) {
      // delete previous
      assertm(unlink(fn) != -1, "unlink not successful"); 
      ASSERT_FALSE(exists(fn));
    }
    struct VertexAttr* ptr;
    int fd;
    std::ofstream ofile;
    if (mmap_flag) {
      assertm((fd = open(fn, O_CREAT)) != -1, "open not successful");
      assert((ptr = (struct VertexAttr*) mmap(nullptr, size, PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0)) != MAP_FAILED);

      for (VID_t i=0; i < nvid; i++) {
        auto v = ptr[i];
        v.edge_state = 192;
        v.vid = numeric_limits<VID_t>::max();
        v.handle = numeric_limits<VID_t>::max();
        v.value = numeric_limits<float>::max();
        //cout << i << endl << v.description() << endl;
      }

    } else {
      ofile.open(fn, ios::out | ios::binary); // read-mode
      ASSERT_TRUE(ofile.is_open());
      ASSERT_TRUE(ofile.good());
      struct VertexAttr* v  = new VertexAttr;
      cout << v->description() << endl;
      v->edge_state = 192;
      v->vid = numeric_limits<VID_t>::max();
      v->handle = numeric_limits<VID_t>::max();
      v->value = numeric_limits<float>::max();
      for (VID_t i=0; i < nvid; i++) {
        // write struct to file
        ofile.write((char *)v, sizeof(VertexAttr));
      }
    }

    if (mmap_flag) {
      assert(munmap((void*)ptr, size) == 0);
    } else {
      //ofile.write((char*)ptr, size);
      ASSERT_TRUE(ofile.good());

      //close file
      ofile.close();
      ASSERT_TRUE(exists(fn));
    }


    cout << "finished interval creation"<<endl;
  } else {
    cout << fn << " already exists" << endl;
  }
  // check the first few VertexAttr look correct
  interval_base_immutable(2) ;
}

TEST (Interval, LoadSaveMmap) {
  bool mmap_ = true;
  load_save(mmap_);
}

TEST (Interval, LoadSave) {
  bool mmap_ = false;
  load_save(mmap_);
}

TEST(Interval, GetAttrVid) {
  bool mmap_ = false;
  VID_t grid_size = 4;
  VID_t nvid = grid_size * grid_size * grid_size;

  test_get_attr_vid(mmap_, grid_size, grid_size, grid_size);
  // make sure base interval is implemented in a read-only manner
  interval_base_immutable(get_used_vertex_num(grid_size, grid_size)) ;
} 

TEST(Interval, GetAttrVidMmap) {
  bool mmap_ = true;
  VID_t grid_size = 4;
  VID_t nvid = grid_size * grid_size * grid_size;

  test_get_attr_vid(mmap_, grid_size, grid_size, grid_size);
  // make sure base interval is implemented in a read-only manner
  interval_base_immutable(get_used_vertex_num(grid_size, grid_size)) ;
}

TEST(Interval, GetAttrVidMultiBlock) {
  VID_t grid_size = 8;
  VID_t block_size = grid_size / 2;
  // this is the total vertices that will be used including ghost cells
  auto interval_vert_num = get_used_vertex_num(grid_size, block_size);
  // check it can handle this amount
  //interval_base_immutable(interval_vert_num) ;
  // this is the domain size not counting ghost cells
  VID_t nvid = grid_size * grid_size * grid_size;

  bool mmap_ = true;
  // single interval
  test_get_attr_vid(mmap_, grid_size, grid_size, grid_size / 2);

  interval_base_immutable(interval_vert_num) ;
} 

TEST(Interval, GetAttrVidMultiInterval) {
  VID_t grid_size = 8;
  auto interval_size = grid_size / 2;
  VID_t nvid = grid_size * grid_size * grid_size;
  auto interval_vert_num = get_used_vertex_num(grid_size, interval_size);
  //interval_base_immutable(interval_vert_num) ;

  bool mmap_ = true;
  // single block per interval
  test_get_attr_vid(mmap_, grid_size, interval_size, interval_size);

  interval_base_immutable(interval_vert_num) ;
} 


/* 
 * Create the desired markers (seed locations) and images to be used by other
 * gtest functions below, this can be skipped once it has been run on a new
 * system for all the necessary configurations of grid_sizes, tcases and
 * selected percents
 */
TEST (Install, DISABLED_CreateMarkersImages) {
  // change these to desired params
  // Note this will delete anything in the
  // same directory before writing
  // tcase 5 is deprecated
  std::vector<int> grid_sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512};
  std::vector<int> testcases = {4, 3, 2, 1, 0};
  std::vector<double> selected_percents = {1, 10, 50, 100};

  for (auto& grid_size : grid_sizes) {
    cout << "Create grids of " << grid_size << endl;
    auto root_vid = get_central_vid(grid_size);
    // FIXME save root marker vid
    long sz0 = (long) grid_size;
    long sz1 = (long) grid_size;
    long sz2 = (long) grid_size;
    VID_t tol_sz = sz0 * sz1 * sz2;

    uint16_t bkg_thresh = 0;
    int line_per_dim = 2; // for tcase 5 lattice grid

    VID_t x, y, z;
    x = y = z = get_central_sub(grid_size); // place at center
    VertexAttr* root = new VertexAttr();
    root->vid = (VID_t) z * sz0 * sz1 + y * sz0 + x;
    ASSERT_EQ(root->vid, root_vid);
    cout << "x " << x << "y " << y << "z " << z << endl;
    cout << "root vid " << root->vid << endl;

    for (auto& tcase : testcases) {
      cout << "  tcase " << tcase << endl;
      for (auto& slt_pct : selected_percents) {
        if ((tcase != 4) && (slt_pct != 100)) continue;
        if ((tcase == 4) && (slt_pct > 50)) continue;
        // never allow an image with less than 1 selected exist
        if ((tcase == 4) && (grid_size < 8) && (slt_pct < 10)) continue;

        std::string base(get_curr());
        auto fn = base + "/test_images/";
        fn = fn + std::to_string(grid_size);
        fn = fn + "/tcase";
        fn = fn + std::to_string(tcase);
        std::string delim("/");
        fn = fn + delim;
        fn = fn + "slt_pct";
        fn = fn + std::to_string((int) slt_pct);
        fn = fn + delim;

        std::string fn_marker(base);
        fn_marker = fn_marker + "/test_markers/";
        fn_marker = fn_marker + std::to_string(grid_size);
        fn_marker = fn_marker + "/tcase";
        fn_marker = fn_marker + std::to_string(tcase);
        fn_marker = fn_marker + delim;
        fn_marker = fn_marker + "slt_pct";
        fn_marker = fn_marker + std::to_string((int) slt_pct);
        fn_marker = fn_marker + delim;

#ifdef USE_MCP3D
        VID_t selected = tol_sz * (slt_pct / (float) 100); // for tcase 4
        // always select at least the root
        if (selected == 0) selected = 1;
        uint16_t* inimg1d = new uint16_t[tol_sz];
        // sets all to 0 for tcase 4
        get_grid(tcase, inimg1d, grid_size); 
        if (tcase == 4) 
          mesh_grid(root->vid, inimg1d, selected, grid_size); 
        if (tcase == 5) { 
          selected = lattice_grid(root->vid, inimg1d, line_per_dim,
              grid_size); 
        }
        float actual_slt_pct = (selected / (float) tol_sz) * 100; 
        cout << "    Actual num selected including root auto selection: " << selected << endl;
        cout << "    actual slt_pct: " << actual_slt_pct << "%" << endl;
        cout << "    for attempted slt_pct: " << slt_pct << "%" << endl;

        ASSERT_NE(inimg1d[root->vid], 0);
        ASSERT_NE(selected, 0);

        // check percent lines up
        // if the tcase can't pass this then raise the size or
        // slt pct to prevent dangerous usage
        ASSERT_NEAR(actual_slt_pct, slt_pct, 100 * EXP_DEV_LOW);

        write_tiff(inimg1d, fn, grid_size) ;
        delete[] inimg1d;
#endif

        // record the root
        write_marker(x, y, z, fn_marker);
      }
    }
  }
}

#ifdef USE_MCP3D
TEST (Image, ReadWrite) {
  auto grid_size = 2;
  auto tcase = 4;
  double slt_pct = 50;
  long sz0 = (long) grid_size;
  long sz1 = (long) grid_size;
  long sz2 = (long) grid_size;
  VID_t tol_sz = sz0 * sz1 * sz2;
  std::string fn(get_curr());
  fn = fn + "/test_images/ReadWriteTest/";

  VID_t selected = tol_sz * (slt_pct / 100); // for tcase 4
  //cout << endl << "Select: " << selected << " (" << slt_pct << "%)" << endl;
  uint16_t* inimg1d = new uint16_t[tol_sz];
  get_grid(tcase, inimg1d, grid_size); // sets all to 0 for tcase 4
  mesh_grid(get_central_vid(grid_size), inimg1d, selected, grid_size);
  //print_image(inimg1d, grid_size * grid_size * grid_size);
  write_tiff(inimg1d, fn, grid_size) ;
  uint16_t* check = read_tiff(fn, grid_size);
  //print_image(check, grid_size * grid_size * grid_size);
  check_image_equality(inimg1d, check, grid_size);

  // This extra dim was only useful for a different grid size
  // due to the cached __image_.json saved by mcp3d that causes errs
  //// write then check again
  //write_tiff(check, fn, grid_size);
  //uint16_t* check2 = read_tiff(fn, grid_size);
  //check_image_equality(inimg1d, check2, grid_size);

  // run recut over the image
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();

  // check again
  uint16_t* check3 = read_tiff(fn, grid_size);
  check_image_equality(inimg1d, check3, grid_size);

  delete[] inimg1d;
}
#endif

TEST (Helpers, DoublePackKey) {
  {
    VID_t block_num = 5;
    VID_t nb_block_num = 13;
    uint32_t result = double_pack_key(block_num, nb_block_num);
    uint32_t actual = (uint32_t) block_num << 16;
    actual |= (uint32_t) nb_block_num;
    ASSERT_EQ(actual, result);
    // check a switch
    result = double_pack_key(nb_block_num, block_num);
    ASSERT_NE(actual, result);
  }
  {
    VID_t block_num = 1;
    VID_t nb_block_num = 1;
    uint32_t result = double_pack_key(block_num, nb_block_num);
    uint32_t actual = (uint32_t) 1<<16 | 1;
    ASSERT_EQ(actual, result);
  }
}

TEST (Helpers, TriplePackKey) {
  {
    VID_t interval_num = 2;
    VID_t block_num = 5;
    VID_t nb_block_num = 13;
    uint64_t result = triple_pack_key(interval_num, block_num, nb_block_num);
    uint64_t actual = (uint64_t) interval_num << 32;
    actual |= (uint64_t) block_num << 16;
    actual |= (uint64_t) nb_block_num;
    ASSERT_EQ(actual, result);
  }
  {
    VID_t interval_num = 1;
    VID_t block_num = 1;
    VID_t nb_block_num = 1;
    uint64_t result = triple_pack_key(interval_num, block_num, nb_block_num);
    uint64_t actual = (uint64_t) 1<<32 | 1<<16 | 1;
    ASSERT_EQ(actual, result);
  }
}

TEST (VertexAttr, ReadWrite) {
  auto nvid = 4;
  auto ptr = new VertexAttr[nvid];
  size_t size = sizeof(VertexAttr) * nvid;
  // path logic
  auto base = get_curr() + "/test_data/";
  auto fn = base + "interval0.bin";
  //remove_all(fn); // make sure it's an overwrite
  create_directories(base);
  //cout << "fn: " << fn << endl;

  // open output
  std::ofstream ofile(fn, ios::out | ios::binary); // read-mode
  ASSERT_TRUE(ofile.is_open());
  ASSERT_TRUE(ofile.good());

  // write struct to file
  ofile.write((char*)ptr, size);
  ASSERT_TRUE(ofile.good());

  // close file
  ofile.close();

  VertexAttr* iptr = (VertexAttr*) malloc(size);
  std::ifstream ifile(fn, ios::in | ios::binary | ios::ate); // write-mode, end
  // open input
  ASSERT_TRUE(ifile.is_open());
  ASSERT_TRUE(ifile.good());
  auto rsize = ifile.tellg();
  ASSERT_EQ(rsize, size);

  ifile.seekg(0, ios::beg); // back to beginning
  ifile.read((char*)iptr, size);
  for (int i = 0; i < nvid; i++) {
    ASSERT_EQ(iptr[i], ptr[i]);
  }
  // close file
  ifile.close();

  delete[] ptr;
  free (iptr);
}

TEST(VertexAttr, Defaults) {
  auto v1 = new VertexAttr();
  ASSERT_FALSE(v1->root());
  ASSERT_EQ(v1->edge_state.field_ , 192);
  ASSERT_TRUE(v1->unselected());
  ASSERT_TRUE(v1->unvisited());
  // FIXME reuse this test once, root marked as known new found
  //ASSERT_TRUE(v1->connections(1, 1).empty());
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
  ASSERT_NE(*v1 , *v2);
  *v2 = *v1;
  ASSERT_EQ(*v1 , *v2); // same values
  ASSERT_EQ(v1->value , v2->value); // check those values manually
  ASSERT_EQ(v1->vid , v2->vid);
  ASSERT_NE(v1->handle , v2->handle); // handles should never be copied
  ASSERT_EQ(v1->edge_state.field_ , v2->edge_state.field_);
  ASSERT_NE(v1 , v2); // make sure they not just the same obj
}

TEST(Radius, DISABLED_Full) { 
  int max_size = 128;
  std::vector<int> grid_sizes = {max_size / 16, max_size / 8, max_size / 4, max_size / 2,
    max_size};
  // tcase 5 is a sphere of radius grid_size / 4 centered
  // in the middle of an image
  std::vector<int> tcases = {5};
  int slt_pct = 100;
  bool print_all = false;
  uint16_t bkg_thresh =0;
  cout << "name,iterations,error_rate(%)\n";
  for (auto& grid_size : grid_sizes) {
    VID_t tol_sz = (VID_t) grid_size * grid_size * grid_size;
    uint16_t* radii_grid = new uint16_t[tol_sz];
    uint16_t* radii_grid_xy = new uint16_t[tol_sz];
    for (auto& tcase : tcases) {
      auto args = get_args(grid_size, slt_pct, tcase, true);

      // adjust final runtime parameters
      auto params = args.recut_parameters();
      // the total number of blocks allows more parallelism
      // ideally intervals >> thread count
      params.set_interval_size(grid_size);
      params.set_block_size(grid_size);
      args.set_recut_parameters(params);

      // run
      auto recut= Recut<uint16_t>(args);
      recut.initialize();
      auto selected = args.recut_parameters().selected;

      recut.setup_value();
      recut.update("value");
      if (print_all) {
        recut.print_interval(0, "value");
      }

      recut.setup_radius();
      recut.update("radius");
      VID_t interval_num = 0;

      if (print_all) {
        std::cout << "recut image grid" << endl;
        print_image_3D(recut.generated_image, grid_size);
      }

      auto total_visited = 0; 
      for (VID_t i = 0; i < tol_sz; i++) {
        if (recut.generated_image[i]) {
          // calculate radius with baseline accurate method
          radii_grid[i] = get_radius_accurate(recut.generated_image, grid_size, i, bkg_thresh);
          // build original production version
          radii_grid_xy[i] = get_radius_hanchuan_XY(recut.generated_image, grid_size, i, bkg_thresh);
          ++total_visited;
        }
      }
      ASSERT_EQ(total_visited, selected);

      // Debug by eye
      if (print_all) {
        cout << "accuracy_radius\n";
        print_image_3D(radii_grid, grid_size);
        std::cout << "XY radii grid" << endl;
        print_image_3D(radii_grid_xy, grid_size);
        std::cout << "Recut radii\n";
        recut.print_interval(0, "radius");
      }

      double xy_err, recut_err;
      check_image_error(recut.generated_image, radii_grid, radii_grid_xy, grid_size, recut.params->selected, xy_err);
      check_recut_error(recut, radii_grid, grid_size, 0, "radius", recut_err);

      std::cout << "\"accurate_radius/" << grid_size << "\",1,0\n";
      std::cout << "\"xy_radius/" << grid_size << "\",1," << xy_err << '\n';
      std::cout << "\"fast_marching_radius/" << grid_size << "\",1," << recut_err << '\n';
    }
    delete[] radii_grid;
    delete[] radii_grid_xy;
  }
}

#ifdef APP2
TEST(DISABLED_EndToEnd, Full) {

  MCP3D_RUNTIME_ERROR("This API is currently deprecated use the main executable ./recut");
  int grid_size = 256;
  long sz0 = (long) grid_size;
  long sz1 = (long) grid_size;
  long sz2 = (long) grid_size;

  int N = 1;
  bool original = true;
  int line_per_dim = 2; // for tcase 4
  std::vector<int> image_offsets = {0, 0, 0};
  VID_t x, y, z;
  x = y = z = get_central_sub(grid_size); //place at center
  // for sequential runs
  std::vector<MyMarker> roots_seq;
  auto root_seq = new MyMarker();
  root_seq->x = x;
  root_seq->y = y;
  root_seq->z = z;
  roots_seq.push_back(*root_seq);

  std::vector<int> block_sizes = {128, grid_size};
  std::vector<double> restart_factors = {0};
  std::vector<int> testcases = {4};
  std::vector<double> percents = {1, 10};

  VID_t tol_sz = sz0 * sz1 * sz2;

  //size_t nthreads = thread::hardware_concurrency(); // max
  size_t nthreads = 32; // to model 16 cores of Yang lab threadripper
  omp_set_num_threads(nthreads);
  std::vector<double> errors;
  std::cout << "nthreads: " << nthreads << endl;
#ifdef ASYNC
  std::cout << "ASYNC" << endl;
#endif
  uint16_t bkg_thresh = 0;
  bool restart;
  bool is_break_accept = false;
  double error;
  cout << "sizeof double " << sizeof(error) << endl;
  uint16_t* inimg1d = new uint16_t[tol_sz];
  for (int run=0; run < N; run++) {
    VertexAttr* root = new VertexAttr();
    root->vid = (VID_t) z * sz0 * sz1 + y * sz0 + x;
    std::vector<VertexAttr> roots;
    roots.push_back(*root);
    for (int tcase : testcases) {
      cout << endl << endl << "Testcase: " << tcase << endl;

      for (double slt_pct : percents) {
        VID_t selected = tol_sz * (slt_pct / 100); // for tcase 4
        cout << endl << "Select: " << selected << " (" << slt_pct << "%)" << endl;
        if (slt_pct == 100) { tcase = 1; }
        get_grid(tcase, inimg1d, grid_size); // sets all to 0 for tcase 4
        if (tcase == 4) { mesh_grid(root->vid, inimg1d, selected, grid_size); }
        if (tcase == 5) { selected = lattice_grid(root->vid, inimg1d, line_per_dim, grid_size); }

        if (original && (tcase == 4)) {
          cout << endl << "Start sequential original fm"<< endl;
          std::vector<MyMarker*> outtree_seq;
          outtree_seq.reserve(selected);
          fastmarching_tree(roots_seq[0],
                 inimg1d,
                 outtree_seq,
                 sz0,
                 sz1,
                 sz2,
                 1,
                 bkg_thresh);
          error = absdiff((VID_t) outtree_seq.size(), selected) / (double) selected;
          cout << "Error: " << error * 100 << "%" << endl;
          cout << "Finish sequential original fm"<< endl << endl;
        }

        for (int block_size : block_sizes) {
          cout << endl << endl << "Block size: " << block_size << endl;
          for (double restart_factor : restart_factors) {
            cout << endl << "Restart factor: " << restart_factor << endl;
            restart = restart_factor > 0 ? true : false;
            //std::vector<VertexAttr> outtree; // sanitize outtree
            std::vector<MyMarker*> outtree; // sanitize outtree
            outtree.reserve(selected);
            fastmarching_tree_parallel(roots,
                   inimg1d,
                   outtree,
                   image_offsets,
                   sz0,
                   sz1,
                   sz2,
                   block_size,
                   bkg_thresh,
                   restart,
                   restart_factor,
                   is_break_accept,
                   nthreads);
            if (tcase == 4) {
              //assert(outtree.size() == selected);
            } else if (tcase == 5) {
              //assert(outtree.size() == selected);
            } else if (tcase < 4) {
              assert(outtree.size() == sz0 * sz1 * sz2);
            }
            error = absdiff((VID_t) outtree.size(), selected) / (double) selected;
            errors.push_back(error);
            cout << "Error: " << error * 100 << "%" << endl;
          } // end restart factors
        } // bs
      } // pcts
    } // tcases
  } // end run
  delete[] inimg1d;
}

TEST (RecutPipeline, DISABLED_4tcase4orig50) {
  auto grid_size = 4;
  double slt_pct = 50;
  auto selected = grid_size * grid_size * grid_size * (slt_pct / 100);
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  int tcase = 4;
  uint16_t bkg_thresh = 0;
  long sz0 = (long) grid_size;
  long sz1 = (long) grid_size;
  long sz2 = (long) grid_size;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  VID_t x, y, z;
  x = y = z = get_central_sub(grid_size); //place at center

  // read data
  uint16_t* inimg1d = read_tiff(args.image_root_dir(), grid_size);
  //print_image(inimg1d, grid_size * grid_size * grid_size);

  // for sequential runs
  std::vector<MyMarker> roots_seq;
  auto root_seq = new MyMarker();
  root_seq->x = x;
  root_seq->y = y;
  root_seq->z = z;
  roots_seq.push_back(*root_seq);
  cout << endl << "Start sequential original fm"<< endl;
  std::vector<MyMarker*> outtree_seq;
  outtree_seq.reserve(selected);
  fastmarching_tree(roots_seq[0],
         inimg1d,
         outtree_seq,
         sz0,
         sz1,
         sz2,
         1,
         bkg_thresh);
  EXPECT_NEAR(args.output_tree_seq.size(), selected, expected * EXP_DEV_LOW);
  //error = absdiff((VID_t) args.output_tree_seq.size(), selected) / (double) selected;
  //cout << "Error: " << error * 100 << "%" << endl;
}
#endif // APP2

TEST(RecutPipeline, DISABLED_PrintDefaultInfo) {
  auto v1 = new VertexAttr();
  auto ps = sysconf(_SC_PAGESIZE);
  auto vs = sizeof(VertexAttr);
  cout << "sizeof vidt " << sizeof(VID_t) << " bytes" << std::scientific << endl;
  cout << "sizeof float " << sizeof(float) << " bytes" << endl;
  cout << "sizeof bitfield " << sizeof(bitfield) << " bytes" << endl;
  cout << "sizeof vertex " << vs << " bytes" << endl;
  cout << "sizeof 1024^3 interval " << sizeof(VertexAttr) << " GB" << endl;
  cout << "page size " << ps << " B" << endl;
  cout << "VertexAttr vertices per page " << ps / vs << endl;
  cout << "cube root of vertices per page " << (int) cbrt(ps / vs) << endl;
  cout << "AvailMem " << GetAvailMem() / (1024 * 1024 * 1024) << " GB" << endl;
  cout << "MAX_INTERVAL_VERTICES " << MAX_INTERVAL_VERTICES << std::scientific << endl;
  cout << "Vertices needed for a 1024^3 interval block size 4 : " << get_used_vertex_num(1024, 4) << std::scientific << endl;
  cout << "Vertices needed for a 2048^3 interval block size 4 : " << get_used_vertex_num(2048, 4) << std::scientific << endl;
}

TEST (RecutPipeline, 4tcase0) {
  auto grid_size = 4;
  double slt_pct = 100; // NA
  int tcase = 0;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  ASSERT_EQ(args.output_tree.size() , grid_size * grid_size * grid_size);
}

TEST (RecutPipeline, DISABLED_4tcase0MultiInterval) {
  auto grid_size = 4;
  double slt_pct = 100; // NA
  int tcase = 0;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);

  auto params = args.recut_parameters();
  params.set_interval_size(2);
  args.set_recut_parameters(params);

  auto recut = Recut<uint16_t>(args);
  //recut.mmap_ = true;
  recut();

  interval_base_immutable(get_used_vertex_num(grid_size, grid_size / 2));
  ASSERT_EQ(args.output_tree.size() , grid_size * grid_size * grid_size);
}

TEST (RecutPipeline, 4tcase1) {
  auto grid_size = 4;
  double slt_pct = 100; // NA
  int tcase = 1;
  auto args = get_args(grid_size, slt_pct, tcase, 
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  ASSERT_EQ(args.output_tree.size() , grid_size * grid_size * grid_size);
}

TEST (RecutPipeline, 4tcase2) {
  auto grid_size = 4;
  double slt_pct = 100; // NA
  int tcase = 2;
  auto args = get_args(grid_size, slt_pct, tcase, 
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  ASSERT_EQ(args.output_tree.size() , grid_size * grid_size * grid_size);
}

TEST (RecutPipeline, 4tcase3) {
  auto grid_size = 4;
  double slt_pct = 100; // NA
  int tcase = 3;
  auto args = get_args(grid_size, slt_pct, tcase, 
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  ASSERT_EQ(args.output_tree.size() , grid_size * grid_size * grid_size);
}

TEST (RecutPipeline, 4tcase4) {
  auto grid_size = 4;
  double slt_pct = 50;
  int tcase = 4;
  auto args = get_args(grid_size, slt_pct, tcase, 
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  EXPECT_NEAR(args.output_tree.size() , (slt_pct / 100 ) * grid_size * grid_size * grid_size, expected * EXP_DEV_LOW );
}

TEST (RecutPipeline, DISABLED_ScratchPad) {
  auto grid_size = 512;
  double slt_pct = 10;
  int tcase = 4;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  std::vector<int> interval_sizes = {grid_size / 2, grid_size / 4};
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  for (auto& interval_size : interval_sizes) {

    // FIXME this API for setting block or interval needs to be simpler
    auto params = args.recut_parameters();
    params.set_interval_size(interval_size);
    params.set_block_size(interval_size);
    // by setting the max intensities you do not need to recompute them
    params.set_max_intensity(1);
    params.set_min_intensity(0);
    args.set_recut_parameters(params);

    auto recut = Recut<uint16_t>(args);
    recut();
    ASSERT_NEAR(args.output_tree.size(), expected, expected * EXP_DEV_LOW) << "Expected " << expected << " found " << args.output_tree.size() << endl;
  }
}

TEST (RecutPipeline, DISABLED_256tcase4sltpct10) {
  auto grid_size = 256;
  double slt_pct = 10;
  int tcase = 4;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  EXPECT_NEAR(args.output_tree.size(), expected, expected * EXP_DEV_LOW);
}

// FIXME Fixture not used
//class RecutPipelineFixture : public testing::TestWithParam<int> {
  //public:
    //~RecutPipelineFixture() override {delete &grid_size;}
    //void SetUp() override { grid_size = GetParam(); }
//// define class member variables that can be accessed 
//// like fixture member vars
//// GetParams()
 //protected:
   //int grid_size;
//};

//TEST_P(RecutPipelineFixture, check) {
  //cout << "PipelineFixture " << grid_size << endl;
//}

//INSTANTIATE_TEST_SUITE_P(RecutPipelineInstance,
    //RecutPipelineFixture, testing::Values(256, 512, 1025));

TEST (RecutPipeline, DISABLED_512tcase0sltpct100) {
  VID_t grid_size = 512;
  auto block_size = grid_size / 4;
  //interval_base_immutable(get_used_vertex_num(grid_size, block_size));
  double slt_pct = 100;
  int tcase = 0;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto params = args.recut_parameters();

  // set block size to large enough to prevent array overflow for now
  params.set_block_size(128);
  args.set_recut_parameters(params);

  auto recut = Recut<uint16_t>(args);
  recut.mmap_ = true;
  recut();
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  EXPECT_NEAR(args.output_tree.size(), expected, expected * EXP_DEV_LOW);
  interval_base_immutable(get_used_vertex_num(grid_size, block_size));
}

TEST (RecutPipeline, DISABLED_1024tcase4sltpct1) {
  auto grid_size = 1024;
  double slt_pct = 1;
  int tcase = 4;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto params = args.recut_parameters();

  // set block size to large enough to prevent array overflow for now
  params.set_block_size(128);
  args.set_recut_parameters(params);

  auto recut = Recut<uint16_t>(args);
  recut.mmap_ = true;
  recut();
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  EXPECT_NEAR(args.output_tree.size(), expected, expected * EXP_DEV_LOW);
}

#ifdef APP2
TEST (RecutPipeline, DISABLED_SequentialMatch1024) {
  auto grid_size = 1024;
  double slt_pct = 10;
  auto selected = grid_size * grid_size * grid_size * (slt_pct / 100);
  int tcase = 4;
  uint16_t bkg_thresh = 0;
  long sz0 = (long) grid_size;
  long sz1 = (long) grid_size;
  long sz2 = (long) grid_size;
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  VID_t x, y, z;
  x = y = z = get_central_sub(grid_size); //place at center

  // read data
  uint16_t* inimg1d = read_tiff(args.image_root_dir(), grid_size);
  //print_image(inimg1d, grid_size * grid_size * grid_size);

  // for sequential runs
  std::vector<MyMarker> roots_seq;
  auto root_seq = new MyMarker();
  root_seq->x = x;
  root_seq->y = y;
  root_seq->z = z;
  roots_seq.push_back(*root_seq);
  //cout << endl << "Start sequential original fm"<< endl;
  std::vector<MyMarker*> outtree_seq;
  VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  outtree_seq.reserve(selected);
  fastmarching_tree(roots_seq[0],
         inimg1d,
         outtree_seq,
         sz0,
         sz1,
         sz2,
         1,
         bkg_thresh);
  EXPECT_NEAR(outtree_seq.size(), selected, expected * EXP_DEV_LOW);
  //error = absdiff((VID_t) outtree_seq.size(), (VID_t) selected) / (double) selected;
  //cout << "Found: " << outtree_seq.size() << endl; // " Error: " << error * 100 << "%" << endl;
  auto recut = Recut<uint16_t>(args);
  recut();
  EXPECT_NEAR(outtree.size() , outtree_seq.size(), expected * EXP_DEV_LOW);
  delete[] inimg1d;
}

TEST (RecutPipeline, DISABLED_SequentialMatch256) {

  // set params for a 10%, 256 run
  auto grid_size = 256;
  double slt_pct = 10;
  int tcase = 4;
  auto selected = grid_size * grid_size * grid_size * (slt_pct / 100);
  RecutCommandLineArgs args;
  args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);

  // seq params
  uint16_t bkg_thresh = 0;
  long sz0 = (long) grid_size;
  long sz1 = (long) grid_size;
  long sz2 = (long) grid_size;
  VID_t x, y, z;
  x = y = z = get_central_sub(grid_size); //place at center

  // read data
  uint16_t* inimg1d = read_tiff(args.image_root_dir(), grid_size);

  // for sequential runs
  std::vector<MyMarker> roots_seq;
  auto root_seq = new MyMarker();
  root_seq->x = x;
  root_seq->y = y;
  root_seq->z = z;
  roots_seq.push_back(*root_seq);
  //cout << endl << "Start sequential original fm"<< endl;
  std::vector<MyMarker*> outtree_seq;
  outtree_seq.reserve(selected);
  fastmarching_tree(roots_seq[0],
         inimg1d,
         outtree_seq,
         sz0,
         sz1,
         sz2,
         1,
         bkg_thresh);
  EXPECT_NEAR(outtree_seq.size(), selected, selected * EXP_DEV_LOW);

  delete[] inimg1d;
}
#endif

TEST (RecutPipeline, DISABLED_StandardIntervalReadWrite256) {
  // set params for a 10%, 256 run
  auto grid_size = 256;
  double slt_pct = 10;
  int tcase = 4;
  auto selected = grid_size * grid_size * grid_size * (slt_pct / 100);

  // Read
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto recut = Recut<uint16_t>(args);
  recut();
  EXPECT_NEAR(args.output_tree.size(), selected, selected * EXP_DEV_LOW);
}

TEST (RecutPipeline, DISABLED_test_real_data) {
  VID_t grid_size = 256;
  double slt_pct = 1;
  int tcase = 4;
  std::vector<VID_t> walk_sizes = {grid_size, grid_size / 2,
    grid_size / 4, grid_size / 8, grid_size / 16};
  for (auto walk_size : walk_sizes) {
    cout << walk_size << endl;
    grid_size = walk_size;
    auto args = get_args(grid_size, slt_pct, tcase, false);
    auto selected = args.recut_parameters().selected;

    // adjust final runtime parameters
    auto params = args.recut_parameters();
    //args.set_image_root_dir("../../data/filled/");
    //params.set_marker_file_path("../../data/marker_files");
    args.set_image_root_dir("../../data/manual_modification/");
    params.set_marker_file_path("../../data/manual_modification/marker_files/");
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    params.set_interval_size(grid_size);
    params.set_block_size(grid_size);
    args.set_recut_parameters(params);

    // run
    auto recut= Recut<uint16_t>(args);
    recut();
    //ASSERT_NEAR(args.output_tree.size() , selected, selected * EXP_DEV_LOW);
  }
}

TEST (RecutPipeline, DISABLED_test_critical_loop) {
  VID_t grid_size = 256;
  double slt_pct = 1;
  int tcase = 4;
  std::vector<VID_t> walk_sizes = {grid_size, grid_size / 2,
    grid_size / 4, grid_size / 8, grid_size / 16};
  for (auto walk_size : walk_sizes) {
    cout << walk_size << endl;
    grid_size = walk_size;
    auto args = get_args(grid_size, slt_pct, tcase, GEN_IMAGE);
    auto selected = args.recut_parameters().selected;

    // adjust final runtime parameters
    auto params = args.recut_parameters();
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    params.set_interval_size(grid_size);
    params.set_block_size(grid_size);
    args.set_recut_parameters(params);

    // run
    auto recut= Recut<uint16_t>(args);
    recut();
    ASSERT_NEAR(args.output_tree.size() , selected, selected * EXP_DEV_LOW);
  }
}

TEST (RecutPipeline, DISABLED_Mmap256) {
  // set params for a 10%, 256 run
  auto grid_size = 256;
  double slt_pct = 10;
  int tcase = 4;
  auto selected = grid_size * grid_size * grid_size * (slt_pct / 100);

  // mmap section
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);
  auto recut_mmap = Recut<uint16_t>(args);
  recut_mmap.mmap_ = true;
  recut_mmap();
  EXPECT_NEAR(args.output_tree.size() , selected, selected * EXP_DEV_LOW);
}

TEST (RecutPipeline, DISABLED_MmapMultiInterval256) {
  // set params for a 10%, 256 run
  auto grid_size = 256;
  double slt_pct = 10;
  int tcase = 4;
  auto selected = grid_size * grid_size * grid_size * (slt_pct / 100);

  // mmap section, multi-interval
  auto args = get_args(grid_size, slt_pct, tcase,
      GEN_IMAGE);

  auto params = args.recut_parameters();
  params.set_interval_size(128);
  args.set_recut_parameters(params);

  auto recut_mmap_multi = Recut<uint16_t>(args);

  // set mmap
  recut_mmap_multi.mmap_ = true;

  recut_mmap_multi();
  EXPECT_NEAR(args.output_tree.size() , selected, selected * EXP_DEV_LOW);

}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
