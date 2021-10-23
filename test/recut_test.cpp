#include "app2_helpers.hpp"
#include "recut.hpp"
#include "gtest/gtest.h"
#include <cstdlib> //rand
#include <ctime>   // for srand
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <tbb/parallel_for.h>
#include <tuple>
#include <typeinfo>
#include <vector>

// catch an assertion and auto drop into
// intractive mode for gdb or rr
#define GTEST_BREAK_ON_FAILURE 1
// stop after first test failure
#define GTEST_FAIL_FAST 1

#ifdef USE_MCP3D
#include <common/mcp3d_utility.hpp>    // PadNumStr
#include <image/mcp3d_voxel_types.hpp> // convert to CV type
#include <opencv2/opencv.hpp>          // imwrite
#endif

#define EXP_DEV_LOW .05
#define NUMERICAL_ERROR .00001

#ifdef USE_MCP3D
#define GEN_IMAGE false
#else
#define GEN_IMAGE true
#endif

// This source file is for functions or tests that include test macros only
// note defining a function that contains test function MACROS
// that returns a value will give strange: "void value not ignored as it ought
// to be" errors

template <typename DataType>
void check_recut_error(Recut<uint16_t> &recut, DataType *ground_truth,
                       int grid_size, std::string stage, double &error_rate,
                       std::map<GridCoord, std::deque<VertexAttr>> fifo,
                       int ground_truth_selected_count,
                       bool strict_match = true) {
  auto tol_sz = static_cast<VID_t>(grid_size) * grid_size * grid_size;

  auto vdb_accessor = recut.topology_grid->getAccessor();

  double error_sum = 0.0;
  VID_t total_valid = 0;
  for (int zi = 0; zi < recut.image_lengths[2]; zi++) {
    for (int yi = 0; yi < recut.image_lengths[1]; yi++) {
      for (int xi = 0; xi < recut.image_lengths[0]; xi++) {
        // iteration vars
        auto coord = new_grid_coord(xi, yi, zi);
        auto leaf_iter = recut.topology_grid->tree().probeLeaf(coord);
        auto ind = leaf_iter->beginIndexVoxel(coord);
        auto value_on = leaf_iter->isValueOn(coord);

        auto coord_offset = coord - leaf_iter->origin();
        VID_t vid = coord_to_id(coord, recut.image_lengths);
        // auto interval_id = recut.id_img_to_interval_id(vid);

        auto find_vid = [&]() {
          for (const auto &local_vertex : fifo[leaf_iter->origin()]) {
            if (coord_all_eq(coord_offset, local_vertex.offsets))
              return true;
          }
          return false;
        };

        if (stage == "convert") {
          // std::cout << "type: " << typeid(value_on).name() << '\n';
          auto int_val = value_on ? 1 : 0;
          if (ground_truth[vid]) {
            ASSERT_TRUE(value_on) << coord;
            ++total_valid;
          }
          if (value_on) {
            ASSERT_EQ(ground_truth[vid], 1) << coord;
            error_sum += absdiff(ground_truth[vid], int_val);
          }
          continue;
        }

        // load
        openvdb::points::AttributeHandle<float> radius_handle(
            leaf_iter->constAttributeArray("pscale"));
        openvdb::points::AttributeHandle<uint8_t> flags_handle(
            leaf_iter->constAttributeArray("flags"));
        openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
            leaf_iter->constAttributeArray("parents"));

        if (stage == "radius") {
          if (ground_truth[vid]) {
            ASSERT_TRUE(value_on) << coord;
            ASSERT_TRUE(ind) << coord;
            // ASSERT_TRUE(radius_handle.get(*ind) > 0)
            //<< coord << " recut radius " << ground_truth[vid];
            error_sum += absdiff(ground_truth[vid], radius_handle.get(*ind));
            ++total_valid;
          } else if (value_on) {
            ASSERT_TRUE(ground_truth[vid] > 0) << coord;
            error_sum += absdiff(ground_truth[vid], radius_handle.get(*ind));
            if (strict_match) {
              ASSERT_EQ(radius_handle.get(*ind), ground_truth[vid]) << coord;
            }
            //}
          }
        } else if (stage == "surface") {
          if (strict_match) {
            // if truth shows a value of 1 it is a surface
            // vertex, therefore fifo should also
            // contain this value
            if (ground_truth[vid] == 1) {
              ASSERT_TRUE(value_on) << coord;
              ASSERT_TRUE(ind) << coord;
              // if truth shows a value of 1 it is a surface
              // vertex, therefore fifo should also
              // contain this value
              ASSERT_TRUE(find_vid()) << coord;
              ASSERT_TRUE(is_selected(flags_handle, ind)) << coord;
              ASSERT_TRUE(is_surface(flags_handle, ind)) << coord;
            } else if (value_on) {
              ASSERT_FALSE(find_vid()) << coord;
              ASSERT_FALSE(is_surface(flags_handle, ind)) << coord;
            }
          } else if (value_on) {
            // where strict_match=false all recut surface vertices will be in
            // ground truth, but not all ground_truth surfaces will in recut
            auto found = find_vid();
            if (is_surface(flags_handle, ind)) {
              ASSERT_TRUE(ground_truth[vid] == 1) << coord;
              ASSERT_TRUE(found) << coord;
            }
            if (found) {
              ASSERT_TRUE(ground_truth[vid] == 1) << coord;
              ASSERT_TRUE(is_surface(flags_handle, ind)) << coord;
            }
          }
        } else if (stage == "connected") {
          if (ground_truth[vid]) {
            ASSERT_TRUE(value_on) << coord;
            ASSERT_TRUE(ind) << coord;
            ASSERT_TRUE(is_selected(flags_handle, ind)) << coord;
          }
          if (value_on) {
            ASSERT_EQ(ground_truth[vid], 1) << coord;
          }
        }
      }
    }
  }
  if (stage != "surface" && stage != "convert") {
    ASSERT_EQ(total_valid, ground_truth_selected_count);
    error_rate = 100 * error_sum / static_cast<double>(tol_sz);
  }
}

template <typename T, typename T2> void check_parents(T markers, T2 grid_size) {
  VID_t counter;
  auto children_count = make_shared<uint8_t[]>(markers.size());
  for (auto &marker : markers) {
    auto current = marker;
    counter = 0;
    // breaks when this marker's parents made it back to a root
    while (current->type != 0) {
      ++counter;
      if (counter > markers.size()) {
        cout << "Marker caused infinite cycle by prune parent path\n";
        cout << current->description(grid_size, grid_size) << '\n';
        ASSERT_TRUE(false);
      }
      // only roots will have a parent == 0
      // and a root terminates this loop
      ASSERT_TRUE(current->parent != 0);
      VID_t index = current->parent->vid(grid_size, grid_size);
      children_count[index] += 1;
      // if it's not a root
      // can't have more than 2 children
      if (current->parent->type != 0) {
        // this condition is relaxed
        // EXPECT_LE(children_count[index], 2) << "at index " << index;
        if (children_count[index] > 2) {
          cout << "Warning children count " << +(children_count[index])
               << " at index " << index << '\n';
        }
      }
      if (current->parent == 0) {
        cout << "Non-root marker had an uninitialized parent\n";
        cout << current->description(grid_size, grid_size) << '\n';
        ASSERT_TRUE(false);
      }
      current = current->parent;
    }
    if (current->type != 0) {
      cout << "Marker never found a path back to a root an uninitialized "
              "parent\n";
      cout << current->description(grid_size, grid_size) << '\n';
      ASSERT_TRUE(false);
    }
    ASSERT_LT(counter, markers.size());
  }
  cout << "completed check_parents\n";
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

// template <typename image_t> struct WriteValueOp {

// explicit WriteValueOp(openvdb::Index64 index, image_t *buffer)
//: mIndex(index), buffer(buffer) {}

// void operator()(const openvdb::tree::LeafManager<
// openvdb::points::PointDataTree>::LeafRange &range) const {
// for (auto leafIter = range.begin(); leafIter; ++leafIter) {
// for (auto indexIter = leafIter->beginIndexAll(); indexIter; ++indexIter) {
// const openvdb::points::AttributeArray &array =
// leafIter->attributeArray(mIndex);
// openvdb::points::AttributeHandle<Coord> handle(array);
// handle.set(*indexIter, on);
//}
//}
//}

// openvdb::Index64 mIndex;
// const Coord on = Coord(0, 0, 0);
//};
//

//// Define a local function that retrieves a and b values from a CombineArgs
//// struct and then sets the result member to the maximum of a and b.
// struct Local {
// static inline void max(CombineArgs<float>& args) {
// if (args.b() > args.a()) {
//// Transfer the B value and its active state.
// args.setResult(args.b());
// args.setResultIsActive(args.bIsActive());
//} else {
//// Preserve the A value and its active state.
// args.setResult(args.a());
// args.setResultIsActive(args.aIsActive());
//}
//}
//};

TEST(VDB, InitializeGlobals) {
  // make big enough such that you have at least 2 blocks across each dim
  VID_t grid_size = 16;
  auto grid_extents = GridCoord(grid_size);
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = false;
  // generate an image buffer on the fly
  // then convert to vdb
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                       /*force_regenerate_image=*/false,
                       /*input_is_vdb=*/true);
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();

  auto update_accessor = recut.update_grid->getConstAccessor();
  auto topology_accessor = recut.topology_grid->getConstAccessor();

  if (print_all) {
    print_vdb_mask(topology_accessor, grid_extents);
    print_vdb_mask(update_accessor, grid_extents);
  }

  auto topology_is_on = new_grid_coord(3, 3, 3);

  ASSERT_TRUE(topology_accessor.isValueOn(topology_is_on));

  auto leaf_iter = recut.update_grid->tree().probeLeaf(topology_is_on);
  ASSERT_TRUE(leaf_iter);

  // should be false in update grid
  ASSERT_FALSE(leaf_iter->isValueOn(topology_is_on));
  ASSERT_FALSE(leaf_iter->getValue(topology_is_on));

  auto boundary_coord = new_grid_coord(7, 7, 7);

  ASSERT_TRUE(leaf_iter->isValueOn(boundary_coord));
  ASSERT_FALSE(leaf_iter->getValue(boundary_coord));
}

TEST(VDB, UpdateSemantics) {

  auto grid = openvdb::BoolGrid::create();

  openvdb::Coord coord(1, 1, 1);

  auto g_acc = grid->getAccessor();
  g_acc.setValueOn(coord);
  g_acc.setValueOff(coord);
  auto leaf_iter = grid->tree().probeLeaf(coord);
  ASSERT_TRUE(leaf_iter);

  ASSERT_FALSE(leaf_iter->isValueOn(coord));
  ASSERT_FALSE(leaf_iter->getValue(coord));
  // set topology on
  leaf_iter->setActiveState(coord, true);
  ASSERT_TRUE(leaf_iter->isValueOn(coord));
  ASSERT_FALSE(leaf_iter->getValue(coord));
  ASSERT_FALSE(leaf_iter->isInactive());

  // now set the value on
  leaf_iter->setValue(coord, true);
  ASSERT_TRUE(leaf_iter->getValue(coord));

  // clear all values, but leave active states intact
  leaf_iter->fill(false);
  ASSERT_FALSE(leaf_iter->getValue(coord));
  ASSERT_TRUE(leaf_iter->isValueOn(coord));
  ASSERT_FALSE(leaf_iter->isInactive());

  cout << "final view\n";
  for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
    auto val = *iter;
    auto val_coord = iter.getCoord();
    cout << val_coord << " " << val << '\n';
  }
}

TEST(Histogram, Add) {
  auto granularity = 8;
  auto histogram = Histogram<uint16_t>(granularity);
  histogram.bin_counts[0] = 1;
  histogram.bin_counts[1] = 1;

  auto rhistogram = Histogram<uint16_t>(granularity);
  rhistogram.bin_counts[0] = 2;
  rhistogram.bin_counts[1] = 2;

  auto combined = histogram;
  combined += rhistogram;
  for (auto [key, value] : combined.bin_counts) {
    ASSERT_EQ(value, histogram.bin_counts[key] + rhistogram.bin_counts[key]);
  }
}

TEST(Histogram, CallAndPrint) {
  auto n = 1 << 8;
  auto granularity = 8;
  auto print_all = true;

  {
    auto histogram = Histogram<uint16_t>(granularity);
    for (int i = 0; i < n; ++i) {
      histogram(i);
    }

    if (print_all) {
      std::cout << histogram.size() << '\n';
      std::cout << histogram;
    }

    ASSERT_EQ(histogram.size(), n / granularity)
        << histogram.size() << ',' << n / granularity;
    for (auto [lower_limit, bin_count] : histogram.bin_counts) {
      ASSERT_EQ(granularity, bin_count)
          << granularity * lower_limit << ',' << bin_count << '\n';
    }
  }

#ifdef USE_MCP3D
  {
    auto tcase = 0;
    auto grid_size = 2;
    auto args = get_args(grid_size, grid_size, grid_size, 100, tcase);
    mcp3d::MImage check(args.image_root_dir(), {"ch0"});
    read_tiff(args.image_root_dir(), args.image_offsets, args.image_lengths,
              check);
    auto histogram = hist(check.Volume<uint16_t>(0), args.image_lengths,
                          zeros(), granularity);
    ASSERT_EQ(histogram.size(), 1)
        << "all values are 1 so only first first bin should exist";
    for (const auto &[lower_limit, bin_count] : histogram.bin_counts) {
      ASSERT_EQ(lower_limit, 0)
          << "all values are 1 so only first first bin should exist";
      ASSERT_EQ(bin_count, grid_size * grid_size * grid_size)
          << "all values are 1 so only first first bin should exist";
    }
  }
#endif
}

TEST(VDB, IntegrateUpdateGrid) {
  // just large enough for a central block and surrounding blocks
  VID_t grid_size = 24;
  auto grid_extents = GridCoord(grid_size);
  // do no use tcase 4 since it is randomized and will not match
  // for the second read test
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = true;
  // generate an image buffer on the fly
  // then convert to vdb
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                       /*force_regenerate_image=*/false,
                       /*input_is_vdb=*/true);
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();
  // recut.activate_vids(root_coords, "connected", recut.map_fifo);
  auto update_accessor = recut.update_grid->getAccessor();
  auto topology_accessor = recut.topology_grid->getConstAccessor();

  if (print_all) {
    print_vdb_mask(topology_accessor, grid_extents);
    print_vdb_mask(update_accessor, grid_extents);
  }

  VID_t interval_id = 0;
  auto lower_corner = new_grid_coord(8, 8, 8);
  auto upper_corner = new_grid_coord(15, 15, 15);
  // both are in the same block
  auto update_leaf = recut.update_grid->tree().probeLeaf(lower_corner);
  ASSERT_TRUE(update_leaf);

  auto update_vertex = new VertexAttr();

  {
    auto stage = "connected";
    set_if_active(update_leaf, lower_corner);
    set_if_active(update_leaf, upper_corner);

    vt::LeafManager<PointTree> grid_leaf_manager(recut.topology_grid->tree());
    recut.integrate_update_grid(recut.topology_grid, grid_leaf_manager, stage,
                                recut.map_fifo, recut.connected_map,
                                recut.update_grid, interval_id);

    cout << "Finished integrate\n";

    auto check_matches = [&](auto block_list, auto corner) {
      auto matches =
          block_list | rng::views::transform([&](auto block_id) {
            auto block_img_offsets =
                recut.id_interval_block_to_img_offsets(interval_id, block_id);
            if (print_all)
              cout << recut.connected_map[block_img_offsets][0].offsets << '\n';
            return coord_all_eq(
                coord_add(block_img_offsets,
                          recut.connected_map[block_img_offsets][0].offsets),
                corner);
          }) |
          rng::to_vector;
      for (const auto match : matches) {
        ASSERT_TRUE(match);
      }
    };

    std::vector lower_adjacent_blocks{4, 10, 12};
    check_matches(lower_adjacent_blocks, lower_corner);

    std::vector upper_adjacent_blocks{14, 16, 22};
    check_matches(upper_adjacent_blocks, upper_corner);

    if (print_all) {
      print_vdb_mask(topology_accessor, grid_extents);
      print_all_points(recut.topology_grid, recut.image_bbox);
    }
  }
}

TEST(VDB, CreatePointDataGrid) {

  std::vector<EnlargedPointDataGrid::Ptr> grids(2);
  // openvdb::GridPtrVec grids(2);
  auto size = 2;
  auto lengths = new_grid_coord(size, size, size);
  auto loc1 = PositionT(0, 0, 0);
  auto loc1v = openvdb::Coord(loc1[0], loc1[1], loc1[2]);
  auto pos = 8192;
  auto loc2 = PositionT(pos, pos, pos);

  {
    std::vector<PositionT> positions;
    positions.push_back(loc1);
    auto grid = create_point_grid(positions, lengths, get_transform());
    auto points_tree = grid->tree();
    // Create a leaf iterator for the PointDataTree.
    auto leafIter = points_tree.beginLeaf();
    // Check that the tree has leaf nodes.
    ASSERT_TRUE(leafIter) << "No Leaf Nodes";
    // Retrieve the index from the descriptor.
    auto descriptor = leafIter->attributeSet().descriptor();
    openvdb::Index64 index = descriptor.find("P");
    // Check that the attribute has been found.
    ASSERT_NE(index, openvdb::points::AttributeSet::INVALID_POS)
        << "Invalid Attribute";

    // grids.push_back(grid);
    grids[0] = grid;
  }

  {
    std::vector<PositionT> positions;
    positions.push_back(loc2);
    auto grid = create_point_grid(positions, lengths, get_transform());
    grids[1] = grid;
    // grids.push_back(grid);
  }

  // EXPECT_FALSE(grids[1]->tree().isValueOn(loc1v));

  // default op is copy
  // leaves grids[0] empty, copies all to grids[1]
  vb::tools::compActiveLeafVoxels(grids[0]->tree(), grids[1]->tree());

  EXPECT_TRUE(grids[1]->tree().isValueOn(loc1v));
  // EXPECT_EQ(loc1, grids[0]->tree().getValue(loc1v));

  // do these have issue if the leaf doesn't already exist?
  // leaves grids[1] empty
  // grids[0]->tree().combineExtended(grids[1], Local::max);

  // csgUnion
  // compActiveLeafVoxels
  // compReplace

  //// Create a leaf manager for the points tree.
  // openvdb::tree::LeafManager<vp::PointDataTree> leafManager(points_tree);
  //// Create a new operator
  // WriteValueOp op(index);
  //// Evaluate in parallel
  // tbb::parallel_for(leafManager.leafRange(), op);
}

// TEST(Coord, CoordToId) {
// GridCoord coord(;
//}

TEST(VDB, ActivateVids) {
  VID_t grid_size = 8;
  auto grid_extents = GridCoord(grid_size);
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = false;
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                       /*force_regenerate_image=*/false,
                       /*input_is_vdb=*/true);
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();
  recut.activate_vids(recut.topology_grid, root_coords, "connected",
                      recut.map_fifo, recut.connected_map);

  if (print_all) {
    print_vdb_mask(recut.topology_grid->getConstAccessor(), grid_extents);
    print_all_points(recut.topology_grid, recut.image_bbox);
  }

  auto interval_id = 0;
  ASSERT_TRUE(recut.active_intervals[interval_id]);
  ASSERT_FALSE(recut.connected_map[GridCoord(0)].empty());

  GridCoord root(3, 3, 3);
  ASSERT_EQ(root, root_coords[0].first) << root_coords[0].first;
  auto leaf_iter = recut.topology_grid->tree().probeLeaf(root);
  auto ind = leaf_iter->beginIndexVoxel(root);
  ASSERT_TRUE(leaf_iter->isValueOn(root));

  {
    openvdb::points::AttributeHandle<uint8_t> flags_handle(
        leaf_iter->constAttributeArray("flags"));
    auto flag = flags_handle.get(*ind);
    ASSERT_TRUE(is_root(flags_handle, ind)) << flag;
  }

  // parents
  {
    openvdb::points::AttributeHandle<OffsetCoord> parents_handle(
        leaf_iter->constAttributeArray("parents"));
    OffsetCoord parent = parents_handle.get(*ind);
    ASSERT_FALSE(valid_parent(parents_handle, ind)) << parent;
    ASSERT_TRUE(coord_all_eq(zeros_off(), parent)) << parent;
  }
}

TEST(VDB, PriorityQueue) {
  VID_t grid_size = 8;
  auto grid_extents = GridCoord(grid_size);
  // do no use tcase 4 since it is randomized and will not match
  // for the second read test
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = true;
  // generate an image buffer on the fly
  // then convert to vdb
  auto args =
      get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
               /*force_regenerate_image=*/false,
               /*input_is_vdb=*/true,
               /* type =*/"float"); // priority queue must have type float
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();

  // TODO switch this to a more formal method
  // set the topology_grid mainly from file for this test
  // overwrite the attempt to convert float to pointgrid
  auto point_args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                             /*force_regenerate_image=*/false,
                             /*input_is_vdb=*/true,
                             /* type =*/"point");
  auto base_grid = read_vdb_file(point_args.image_root_dir());
  recut.topology_grid = openvdb::gridPtrCast<EnlargedPointDataGrid>(base_grid);
  append_attributes(recut.topology_grid);

  auto stage = "value";
  recut.activate_vids(recut.topology_grid, root_coords, stage, recut.map_fifo,
                      recut.connected_map);
  recut.update(stage, recut.map_fifo);

  if (print_all) {
    print_vdb_mask(recut.topology_grid->getConstAccessor(), grid_extents);
    print_all_points(recut.topology_grid, recut.image_bbox);
  }

  auto known_surface = new_grid_coord(1, 1, 1);
  auto known_selected = new_grid_coord(2, 2, 2);
  auto known_root = new_grid_coord(3, 3, 3);

  // they all are in the same leaf
  auto leaf_iter = recut.topology_grid->tree().probeLeaf(known_surface);

  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
      leaf_iter->attributeArray("flags"));

  {
    auto ind = leaf_iter->beginIndexVoxel(known_surface);
    ASSERT_TRUE(is_selected(flags_handle, ind));
    ASSERT_TRUE(is_surface(flags_handle, ind));
  }

  {
    auto ind = leaf_iter->beginIndexVoxel(known_selected);
    ASSERT_TRUE(is_selected(flags_handle, ind));
    ASSERT_FALSE(is_surface(flags_handle, ind));
  }

  {
    auto ind = leaf_iter->beginIndexVoxel(known_root);
    ASSERT_TRUE(is_selected(flags_handle, ind));
    ASSERT_TRUE(is_root(flags_handle, ind));
    ASSERT_FALSE(is_surface(flags_handle, ind));
  }
}

TEST(Utils, AdjustSomaRadii) {
  VID_t grid_size = 8;
  auto grid_extents = GridCoord(grid_size);
  // do no use tcase 4 since it is randomized and will not match
  // for the second read test
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = true;
  // generate an image buffer on the fly
  // then convert to vdb
  auto args =
      get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
               /*force_regenerate_image=*/false,
               /*input_is_vdb=*/true,
               /* type =*/"float"); // priority queue must have type float
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();

  // TODO switch this to a more formal method
  // set the topology_grid mainly from file for this test
  // overwrite the attempt to convert float to pointgrid
  auto point_args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                             /*force_regenerate_image=*/false,
                             /*input_is_vdb=*/true,
                             /* type =*/"point");
  auto base_grid = read_vdb_file(point_args.image_root_dir());
  recut.topology_grid = openvdb::gridPtrCast<EnlargedPointDataGrid>(base_grid);
  append_attributes(recut.topology_grid);
  //cout << root_coords[0].first << ' ' << +(root_coords[0].second) << '\n';

  // check all root radii are 0
  rng::for_each(root_coords, [&recut](const auto &coord_radius) {
    const auto [coord, radius] = coord_radius;
    const auto leaf = recut.topology_grid->tree().probeLeaf(coord);
    assertm(leaf, "corresponding leaf of passed root must be active");
    auto ind = leaf->beginIndexVoxel(coord);
    assertm(ind, "corresponding voxel of passed root must be active");

    // modify the radius value
    openvdb::points::AttributeWriteHandle<float> radius_handle(
        leaf->attributeArray("pscale"));

    auto previous_radius = radius_handle.get(*ind);
    ASSERT_EQ(previous_radius, 0);
  });

  adjust_soma_radii(root_coords, recut.topology_grid);

  // check all root radii have been changed
  rng::for_each(root_coords, [&recut](const auto &coord_radius) {
    const auto [coord, radius] = coord_radius;
    const auto leaf = recut.topology_grid->tree().probeLeaf(coord);
    assertm(leaf, "corresponding leaf of passed root must be active");
    auto ind = leaf->beginIndexVoxel(coord);
    assertm(ind, "corresponding voxel of passed root must be active");

    // modify the radius value
    openvdb::points::AttributeWriteHandle<float> radius_handle(
        leaf->attributeArray("pscale"));

    auto previous_radius = radius_handle.get(*ind);
    ASSERT_NE(previous_radius, 0);
  });
}

TEST(Utils, ConvertFloatToMarkers) {
}

TEST(VDB, Connected) {
  VID_t grid_size = 8;
  auto grid_extents = GridCoord(grid_size);
  // do no use tcase 4 since it is randomized and will not match
  // for the second read test
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = false;
  // generate an image buffer on the fly
  // then convert to vdb
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                       /*force_regenerate_image=*/false,
                       /*input_is_vdb=*/true);
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();
  auto stage = "connected";
  recut.activate_vids(recut.topology_grid, root_coords, stage, recut.map_fifo,
                      recut.connected_map);
  recut.update(stage, recut.map_fifo);

  if (print_all) {
    print_vdb_mask(recut.topology_grid->getConstAccessor(), grid_extents);
    print_all_points(recut.topology_grid, recut.image_bbox);
  }

  auto known_surface = new_grid_coord(1, 1, 1);
  auto known_selected = new_grid_coord(2, 2, 2);
  auto known_root = new_grid_coord(3, 3, 3);

  // they all are in the same leaf
  auto leaf_iter = recut.topology_grid->tree().probeLeaf(known_surface);

  openvdb::points::AttributeWriteHandle<uint8_t> flags_handle(
      leaf_iter->attributeArray("flags"));

  {
    auto ind = leaf_iter->beginIndexVoxel(known_surface);
    ASSERT_TRUE(is_selected(flags_handle, ind));
    ASSERT_TRUE(is_surface(flags_handle, ind));
  }

  {
    auto ind = leaf_iter->beginIndexVoxel(known_selected);
    ASSERT_TRUE(is_selected(flags_handle, ind));
    ASSERT_FALSE(is_surface(flags_handle, ind));
  }

  {
    auto ind = leaf_iter->beginIndexVoxel(known_root);
    ASSERT_TRUE(is_selected(flags_handle, ind));
    ASSERT_TRUE(is_root(flags_handle, ind));
    ASSERT_FALSE(is_surface(flags_handle, ind));
  }
}

// TEST(VDBWriteOnly, DISABLED_Any) {
// VID_t grid_size = 8;
// auto grid_extents = GridCoord(grid_size);
//// do no use tcase 4 since it is randomized and will not match
//// for the second read test
// auto tcase = 7;
// double slt_pct = 100;
// bool print_all = false;
//#ifdef LOG_FULL
//// print_all = true;
//#endif
//// auto str_path = get_data_dir();
//// auto fn = str_path + "/test_convert_only.vdb";
// auto fn = "test_convert_only.vdb";

//// generate an image buffer on the fly
//// then convert to vdb
// auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
//[>force_regenerate_image=<]true);
// auto recut = Recut<uint16_t>(args);
// recut.params->convert_only_ = true;
// recut.params->out_vdb_ = fn;
// ASSERT_FALSE(fs::exists(fn));
// recut();

// if (print_all) {
// std::cout << "recut image grid" << endl;
// print_image_3D(recut.generated_image, grid_extents);
//}

// if (print_all)
// print_vdb_mask(recut.topology_grid->getConstAccessor(), grid_extents);

// ASSERT_TRUE(fs::exists(fn));
//}

TEST(VDB, Convert) {
  VID_t grid_size = 8;
  VID_t interval_size = 8;
  GridCoord grid_extents(grid_size);
  // do no use tcase 4 since it is randomized and will not match
  // for the second read test
  auto tcase = 7;
  double slt_pct = 100;
  bool print_all = false;
  //#ifdef LOG_FULL
  print_all = true;
  //#endif
  auto str_path = get_data_dir();
  // auto fn = str_path + "/test_convert_only.vdb";

  // generate an image buffer on the fly
  // then convert to vdb
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
                       /*force_regenerate_image*/ true);
  auto recut = Recut<uint16_t>(args);
  recut.params->convert_only_ = true;
  // recut.params->out_vdb_ = fn;
  // ASSERT_FALSE(fs::exists(fn));
  recut.initialize();

  if (print_all) {
    std::cout << "recut image grid" << endl;
    print_image_3D(recut.generated_image, grid_extents);
  }

  /*
  if (recut.params->convert_only_) {
    recut.activate_all_intervals();
  }

  // mutates topology_grid
  auto stage = "convert";
  recut.update(stage, recut.map_fifo);

  if (print_all)
    print_vdb_mask(recut.topology_grid->getConstAccessor(), grid_extents);

  // don't write out the file since you will have read only paths in nix
  // environments ASSERT_TRUE(fs::exists(fn));

  double write_error_rate;
  EXPECT_NO_FATAL_FAILURE(
      check_recut_error(recut, recut.generated_image,
                        grid_size, stage, write_error_rate, recut.map_fifo,
                        recut.params->selected, true));
  ASSERT_NEAR(write_error_rate, 0., NUMERICAL_ERROR);
  */

  // test reading from a pre-generated image file of exact same as
  // recut.generated_image as long as tcase != 4
  // read from file and convert
  {
    auto args =
        get_args(grid_size, interval_size, interval_size, slt_pct, tcase,
                 /*force_regenerate_image=*/false,
                 /*input_is_vdb=*/false);

    auto recut_from_vdb_file = Recut<uint16_t>(args);
    recut_from_vdb_file.params->convert_only_ = true;

    // handle read from vdb
    recut_from_vdb_file.initialize();
    recut_from_vdb_file.activate_all_intervals();
    // mutates topology_grid
    auto stage = "convert";
    recut_from_vdb_file.update(stage, recut_from_vdb_file.map_fifo);

    if (print_all)
      print_vdb_mask(recut_from_vdb_file.topology_grid->getConstAccessor(),
                     grid_extents);

    // assert equals original grid above
    double read_from_file_error_rate;
    EXPECT_NO_FATAL_FAILURE(
        check_recut_error(recut_from_vdb_file,
                          /*ground_truth*/ recut.generated_image, grid_size,
                          stage, read_from_file_error_rate,
                          recut_from_vdb_file.map_fifo, recut.params->selected,
                          /*strict_match=*/true));

    ASSERT_NEAR(read_from_file_error_rate, 0., NUMERICAL_ERROR);
  }
}

/*
 * Create the desired markers (seed locations) and images to be used by other
 * gtest functions below, this can be skipped once it has been run on a new
 * directory or system for all the necessary configurations of grid_sizes,
 * tcases and selected percents install files into data/
 */
TEST(Install, DISABLED_CreateImagesMarkers) {
  // change these to desired params
  // Note this will delete anything in the
  // same directory before writing
  // tcase 5 is deprecated
  std::vector<int> grid_sizes = {2, 4, 8, 16, 24};

  //#ifdef TEST_ALL_BENCHMARKS
  // grid_sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512};
  //#endif

  std::vector<int> testcases = {7, 5, 4, 3, 2, 1, 0};
  std::vector<double> selected_percents = {1, 10, 50, 100};

  // std::vector<int> grid_sizes = {2, 4, 8, 1024};
  // std::vector<int> testcases = {7, 5, 4, 3, 2, 1, 0};
  // std::vector<double> selected_percents = {.1, .5, 1, 10, 50, 100};

  std::vector<int> no_offsets{0, 0, 0};
  auto print = false;

  for (auto &grid_size : grid_sizes) {
    auto grid_extents = GridCoord(grid_size);
    if (print)
      cout << "Create grids of " << grid_size << endl;
    auto root_vid = get_central_vid(grid_size);
    // FIXME save root marker vid
    auto sz0 = (long)grid_size;
    auto sz1 = (long)grid_size;
    auto sz2 = (long)grid_size;
    VID_t tol_sz = sz0 * sz1 * sz2;

    uint16_t bkg_thresh = 0;
    int line_per_dim = 2; // for tcase 5 lattice grid

    VID_t x, y, z;
    x = y = z = get_central_coord(grid_size); // place at center
    VertexAttr *root = new VertexAttr();
    root_vid = (VID_t)z * sz0 * sz1 + y * sz0 + x;
    if (print) {
      cout << "x " << x << "y " << y << "z " << z << endl;
      cout << "root vid " << root_vid << endl;
    }

    for (auto &tcase : testcases) {
      if (print)
        cout << "  tcase " << tcase << endl;
      for (auto &slt_pct : selected_percents) {
        if ((tcase != 4) && (slt_pct != 100))
          continue;
        if ((tcase == 4) && (slt_pct > 50))
          continue;
        // never allow an image with less than 1 selected exist
        if ((tcase == 4) && (grid_size < 8) && (slt_pct < 10))
          continue;
        if ((tcase == 4) && (slt_pct > 5))
          continue;

        auto radius = grid_size / 4;
        if ((tcase == 4) || (radius < 1)) radius = 1;

        std::string base(get_data_dir());
        std::string delim("/");
        std::string fn_marker(base);
        fn_marker = fn_marker + "/test_markers/";
        fn_marker = fn_marker + std::to_string(grid_size);
        fn_marker = fn_marker + "/tcase";
        fn_marker = fn_marker + std::to_string(tcase);
        fn_marker = fn_marker + delim;
        fn_marker = fn_marker + "slt_pct";
        fn_marker = fn_marker + std::to_string((int)slt_pct);
        // fn_marker = fn_marker + delim;
        // record the root
        write_marker(x, y, z, radius, fn_marker);

        auto image_dir = base + "/test_images/";
        image_dir = image_dir + std::to_string(grid_size);
        image_dir = image_dir + "/tcase";
        image_dir = image_dir + std::to_string(tcase);
        image_dir = image_dir + delim;
        image_dir = image_dir + "slt_pct";
        image_dir = image_dir + std::to_string((int)slt_pct);

        VID_t desired_selected;
        desired_selected = tol_sz * (slt_pct / (float)100); // for tcase 4
        // always select at least the root
        if (desired_selected == 0) {
          desired_selected = 1;
        }
        uint16_t *inimg1d = new uint16_t[tol_sz];
        VID_t actual_selected =
            create_image(tcase, inimg1d, grid_size, desired_selected, root_vid);

        float actual_slt_pct = (actual_selected / (float)tol_sz) * 100;
        if (print) {
          cout << "    Actual num selected including root auto selection: "
               << actual_selected << endl;
          cout << "    actual slt_pct: " << actual_slt_pct << "%" << endl;
          cout << "    for attempted slt_pct: " << slt_pct << "%" << endl;
        }

        // print_image_3D(inimg1d, grid_extents);
        ASSERT_NE(inimg1d[root_vid], 0) << " tcase " << tcase;
        ASSERT_NE(actual_selected, 0);

        // check percent lines up
        // if the tcase can't pass this then raise the size or
        // slt pct to prevent dangerous usage
        if (tcase == 4) {
          ASSERT_NEAR(actual_slt_pct, slt_pct, 100 * EXP_DEV_LOW);
        }

        auto float_grid = openvdb::FloatGrid::create();
        convert_buffer_to_vdb_acc(inimg1d, grid_extents, zeros(), zeros(),
                                  float_grid->getAccessor(), 0);
        set_grid_meta(float_grid, grid_extents, bkg_thresh);
        if (print) {
          // print_grid_metadata(float_grid); // already in create_point_grid
          cout << "float grid\n";
          print_vdb_mask(float_grid->getConstAccessor(), grid_extents);
        }

        // auto point_grid =
        // copy_to_point_grid(float_grid, grid_extents, bkg_thresh);
        // if (print) {
        // cout << "point grid\n";
        // print_vdb_mask(point_grid->getConstAccessor(), grid_extents);
        // for (auto iter = point_grid->cbeginValueOn(); iter; ++iter) {
        // std::cout << "Grid " << iter.getCoord() << " = " << *iter
        //<< std::endl;
        //}
        //}

        std::vector<PositionT> positions;
        convert_buffer_to_vdb(inimg1d, grid_extents, zeros(), zeros(),
                              positions, 0);
        auto topology_grid =
            create_point_grid(positions, grid_extents, get_transform());

        std::cout << "created vdb grid\n";

        topology_grid->tree().prune();

        if (print) {
          print_vdb_mask(topology_grid->getConstAccessor(), grid_extents);

          // print_all_points(
          // topology_grid,
          // openvdb::math::CoordBBox(
          // zeros(), grid_extents));

          // print_image_3D(inimg1d, grid_extents);
        }

#ifdef USE_MCP3D
        write_tiff(inimg1d, image_dir, grid_size);
#endif

        {
          openvdb::GridPtrVec grids;
          grids.push_back(topology_grid);
          auto fn = image_dir + "/point.vdb";
          write_vdb_file(grids, fn);
        }

        {
          openvdb::GridPtrVec grids;
          grids.push_back(float_grid);
          auto fn = image_dir + "/float.vdb";
          write_vdb_file(grids, fn);
        }

        delete[] inimg1d;
      }
    }
  }
}

TEST(VDB, ConvertDenseToVDB) {
  auto grid_size = 8;
  auto grid_extents = GridCoord(grid_size, grid_size, grid_size);
  auto tol_sz = coord_prod_accum(grid_extents);
  auto root_vid = get_central_vid(grid_size);
  auto root_coord = GridCoord(get_central_coord(grid_size));
  auto upsample_z = 5;
  auto print = true;
  auto extended_grid_extents =
      GridCoord(grid_size, grid_size, grid_size * upsample_z);

  cout << root_coord;

  auto idx = upsample_idx(root_coord[2], upsample_z);
  auto root_coord_upsampled = GridCoord(root_coord[0], root_coord[1], idx);

  cout << idx << '\n';

  // ASSERT_EQ(idx, root_coord[2] * upsample_z + upsample_z / 2);
  ASSERT_EQ(idx, root_coord[2] * upsample_z);

  uint16_t *inimg1d = new uint16_t[tol_sz];
  VID_t actual_selected = create_image(5, inimg1d, grid_size, 100, root_vid);

  if (print) {
    print_image_3D(inimg1d, grid_extents);
  }

  // auto topology_grid = create_vdb_grid(extended_grid_extents, 0);
  auto float_grid = openvdb::FloatGrid::create();
  convert_buffer_to_vdb_acc(inimg1d, extended_grid_extents, zeros(), zeros(),
                            float_grid->getAccessor(), 0, upsample_z);

  if (print) {
    cout << "float grid\n";
    print_vdb_mask(float_grid->getConstAccessor(), extended_grid_extents);
  }

  std::vector<PositionT> positions;
  convert_buffer_to_vdb(inimg1d, extended_grid_extents, zeros(), zeros(),
                        positions, 0, upsample_z);
  auto topology_grid =
      create_point_grid(positions, extended_grid_extents, get_transform());

  std::cout << "created vdb grid\n";

  topology_grid->tree().prune();

  if (print) {
    print_vdb_mask(topology_grid->getConstAccessor(), extended_grid_extents);

    // print_all_points(
    // topology_grid,
    // openvdb::math::CoordBBox(
    // zeros(), extended_grid_extents));

    // print_image_3D(inimg1d, extended_grid_extents);
  }
}

TEST(VDB, GetSetGridMeta) {
  float bkg_thresh = 0.;
  GridCoord grid_extents(1);
  auto grid = openvdb::FloatGrid::create();

  {
    set_grid_meta(grid, grid_extents, bkg_thresh);
    // print_grid_metadata(grid);
  }

  // check transferring metadata from one grid to another
  auto [lengths, bt] = get_metadata(grid);
  {
    ASSERT_EQ(bt, bkg_thresh);
    ASSERT_EQ(lengths[0], grid_extents[0]);
    ASSERT_EQ(lengths[1], grid_extents[1]);
    ASSERT_EQ(lengths[2], grid_extents[2]);
  }
}

#ifdef USE_MCP3D
TEST(Install, ImageReadWrite) {
  auto grid_size = 2;
  auto grid_extents = GridCoord(grid_size);
  auto tcase = 7;
  double slt_pct = 100;
  auto lengths = new_grid_coord(grid_size, grid_size, grid_size);
  auto tol_sz = coord_prod_accum(lengths);
  auto print_all = true;

  std::string fn(get_data_dir());
  // Warning: do not use directory names postpended with slash
  fn = fn + "/test_images/ReadWriteTest";
  auto image_lengths = GridCoord(grid_size);
  auto image_offsets = GridCoord(0);

  VID_t selected = tol_sz; // for tcase 4
  uint16_t *inimg1d = new uint16_t[tol_sz];
  create_image(tcase, inimg1d, grid_size, selected, get_central_vid(grid_size));
  cout << "Created image:\n";
  print_image_3D(inimg1d, grid_extents);

  write_tiff(inimg1d, fn, grid_size);
  mcp3d::MImage check(fn, {"ch0"});
  ASSERT_NE(check.n_channels(), 0);
  read_tiff(fn, image_offsets, image_lengths, check);
  // print_image(check, grid_size * grid_size * grid_size);
  cout << "Read image:\n";
  print_image_3D(check.Volume<uint16_t>(0), grid_extents);

  ASSERT_NO_FATAL_FAILURE(
      check_image_equality(inimg1d, check.Volume<uint16_t>(0), grid_size));
  cout << "first read passed\n";

  // run recut over the image, force it to run in read image
  // non-generated mode since MCP3D is guaranteed here
  // auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase,
  // false); auto recut = Recut<uint16_t>(args); recut();

  // check again
  cout << "second read\n";
  mcp3d::MImage check3(fn, {"ch0"});
  ASSERT_NE(check3.n_channels(), 0);
  read_tiff(fn, image_offsets, image_lengths, check3);
  ASSERT_NO_FATAL_FAILURE(
      check_image_equality(inimg1d, check3.Volume<uint16_t>(0), grid_size));

  delete[] inimg1d;
}

#endif // USE_MCP3D

TEST(VertexAttr, Defaults) {
  auto v1 = new VertexAttr();
  ASSERT_EQ(v1->edge_state.field_, 0);
  ASSERT_TRUE(v1->unselected());
  ASSERT_FALSE(v1->root());
}

TEST(TileThresholds, AllTcases) {
  VID_t grid_size = 4;
  auto grid_extents = GridCoord(grid_size);
  auto slt_pct = 50;
  auto grid_vertex_size = grid_size * grid_size * grid_size;
  auto print_image = false;
#ifdef FULL_PRINT
  print_image = true;
#endif
  // doesn't include real images tcase 6
  std::vector<int> tcases = {1, 2, 3, 4, 5, 7};
  for (const auto &tcase : tcases) {
    auto args =
        get_args(grid_size, grid_size, grid_size, slt_pct, tcase, false);
    auto selected = args.recut_parameters().selected;
    auto recut = Recut<uint16_t>(args);
    auto inimg1d = std::make_unique<uint16_t[]>(grid_vertex_size);
    auto tile_thresholds = new TileThresholds<uint16_t>();
    uint16_t bkg_thresh;

#ifdef USE_MCP3D
    mcp3d::MImage image(args.image_root_dir());
    if (tcase == 6) {
      read_tiff(args.image_root_dir(), args.image_offsets, args.image_lengths,
                image);
      if (print_image) {
        print_image_3D(image.Volume<uint16_t>(0), grid_extents);
      }
      bkg_thresh = recut.get_bkg_threshold(image.Volume<uint16_t>(0),
                                           grid_vertex_size, slt_pct / 100.);
      tile_thresholds->get_max_min(image.Volume<uint16_t>(0), grid_vertex_size);

      if (print_image) {
        cout << "tcase " << tcase << " bkg_thresh " << bkg_thresh << "\nmax "
             << tile_thresholds->max_int << " min " << tile_thresholds->min_int
             << '\n';
      }
    }
#endif

    if (tcase <= 5) {
      create_image(tcase, inimg1d.get(), grid_size, selected,
                   get_central_vid(grid_size));
      if (print_image) {
        print_image_3D(inimg1d.get(), grid_extents);
      }
      bkg_thresh = recut.get_bkg_threshold(inimg1d.get(), grid_vertex_size,
                                           slt_pct / 100.);
      tile_thresholds->get_max_min(inimg1d.get(), grid_vertex_size);

      if (print_image) {
        cout << "tcase " << tcase << " bkg_thresh " << bkg_thresh << "\nmax "
             << tile_thresholds->max_int << " min " << tile_thresholds->min_int
             << '\n';
      }
    }
  }
}

TEST(VertexAttr, MarkStatus) {
  auto v1 = new VertexAttr();
  v1->mark_root();
  ASSERT_TRUE(v1->root());

  v1->mark_selected();
  ASSERT_TRUE(v1->selected());
}

TEST(VertexAttr, CopyOp) {
  auto v1 = new VertexAttr();
  auto v2 = new VertexAttr();
  v1->offsets = new_offset_coord(1, 1, 1);
  v1->edge_state.reset();
  ASSERT_NE(*v1, *v2);
  *v2 = *v1;
  ASSERT_EQ(*v1, *v2); // same values
  ASSERT_EQ(v1->offsets, v2->offsets);
  ASSERT_EQ(v1->edge_state.field_, v2->edge_state.field_);
  ASSERT_NE(v1, v2); // make sure they not just the same obj
}

TEST(CompareTree, All) {
  std::vector<MyMarker *> truth;
  std::vector<MyMarker *> check;
  VID_t grid_size = 4;
  auto grid_extents = GridCoord(grid_size);
  auto interval_size = grid_size;
  VID_t block_size = 2;

  auto args = get_args(grid_size, interval_size, block_size, 100, 1);
  auto recut = Recut<uint16_t>(args);
  recut.initialize();

  std::vector<MyMarker *> false_negatives;
  std::vector<MyMarker *> false_positives;

  // block id 0
  truth.push_back(new MyMarker(0., 0., 0.));
  check.push_back(new MyMarker(0., 0., 0.));
  // test the MyMarker* lambda works
  ASSERT_TRUE(eq(truth[0], check[0]));

  // block id 1
  truth.push_back(new MyMarker(2., 1., 0.));
  check.push_back(new MyMarker(2., 1., 0.));

  // block id 2
  auto neg = new MyMarker(0., 2., 0.);
  truth.push_back(neg);
  false_negatives.push_back(neg);

  auto pos = new MyMarker(0., 2., 2.);
  check.push_back(pos);
  false_positives.push_back(pos);

  // block id 5
  truth.push_back(new MyMarker(0., 0., 2.));
  check.push_back(new MyMarker(0., 0., 2.));

  // test the MyMarker* lambda works
  ASSERT_TRUE(lt(neg, pos));

  auto a = new MyMarker(0., 0., 2.);
  auto b = new MyMarker(0., 0., 2.);
  // test the MyMarker* lambda works
  ASSERT_TRUE(eq(a, b));

  auto counter = unique_count(truth);
  // no repeats therefore count should be the same
  ASSERT_EQ(truth.size(), counter);

  // add repeats
  // block id 1
  truth.push_back(new MyMarker(2., 1., 0.));
  check.push_back(new MyMarker(2., 1., 0.));

  counter = unique_count(truth);
  // repeats therefore count should be diff
  ASSERT_EQ(truth.size(), counter + 1);

  auto results = compare_tree(truth, check, grid_extents);
  // make sure duplicates are found
  ASSERT_EQ(results->duplicate_count, 2);

  // remove duplicates
  truth.pop_back();
  check.pop_back();

  results = compare_tree(truth, check, grid_extents);
  // it's a problem if two markers with same vid are in a results vector
  ASSERT_EQ(results->duplicate_count, 0);

  auto get_mismatch = [&](auto false_negatives, auto check_false_negatives) {
    auto check = check_false_negatives |
                 rng::views::transform([](auto pairi) { return pairi.first; }) |
                 rng::to_vector | rng::action::sort;
    auto truth = get_vids_sorted(false_negatives, grid_extents);
    auto diff = rng::views::set_intersection(truth, check); // set_difference
    return rng::distance(diff); // return range length
  };

  auto count_mismatch_negatives =
      get_mismatch(false_negatives, results->false_negatives);
  auto count_mismatch_positives =
      get_mismatch(false_positives, results->false_positives);

  ASSERT_EQ(count_mismatch_positives, 1);
  ASSERT_EQ(count_mismatch_negatives, 1);

  //// check the compare tree worked properly
  // ASSERT_EQ(truth.size(),
  // results->match_count + results->false_positives.size());
  // ASSERT_EQ(truth.size(),
  // results->match_count + results->false_negatives.size());
}

/*
TEST(CoveredByParent, Full) {
  int grid_size = 8;
  int radius = 3;
  auto root_vid = 219;

  auto args = get_args(grid_size, grid_size, grid_size, 100, 5, true);

  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();

  // -3 steps in x y z respect.
  // +3 steps in x y z respect.
  std::vector<VID_t> check_trues = {216, 195, 27, 222, 243, 411};
  std::vector<VID_t> check_falses = {215, 187, 26, 223, 251, 412};
  auto root = new VertexAttr();
  root->offsets = coord_mod(id_to_coord(root_vid), recut.block_lengths);
  root->radius = radius;

  for (const auto &check : check_trues) {
    ASSERT_TRUE(is_covered_by_parent(check, root_vid, radius, grid_size));
    auto current = new VertexAttr();
    current->vid = check;
    current->parent = root_vid;
    // ASSERT_TRUE(recut.is_covered_by_parent(0,0,current)) << "for vid " <<
    // check;
  }

  for (const auto &check : check_falses) {
    ASSERT_FALSE(is_covered_by_parent(check, root_vid, radius, grid_size));
    auto current = new VertexAttr();
    current->vid = check;
    current->parent = root_vid;
    // ASSERT_FALSE(recut.is_covered_by_parent(0,0,current)) << "for vid " <<
    // check;
  }
}
*/

/*

TEST(CheckGlobals, DISABLED_AllFifo) {
  // minimal setup of globals
  int max_size = 8;
  auto args = get_args(max_size, max_size, max_size, 100, 0, true);
  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();
  std::list<VID_t> l(10);
  std::iota(l.begin(), l.end(), 0);

  bool found;
  for (auto vid : l) {
    auto offsets = id_to_coord(vid, recut.image_lengths);
    auto vertex = recut.get_or_set_active_vertex(0, 0, offsets, found);
    ASSERT_FALSE(found);
    ASSERT_FALSE(vertex->surface());
    ASSERT_TRUE(vertex->selected());
    ASSERT_TRUE(coord_all_eq(vertex->offsets, offsets));

    vertex->mark_root();
    vertex->mark_surface();

    ASSERT_TRUE(vertex->root());
    ASSERT_TRUE(vertex->surface());
    ASSERT_FALSE(vertex->selected());

    auto gvertex = recut.get_active_vertex(0, 0, offsets);
    ASSERT_TRUE(gvertex->root());
    ASSERT_TRUE(gvertex->surface());
    ASSERT_FALSE(gvertex->selected());

    gvertex->mark_selected();
    auto g2vertex = recut.get_active_vertex(0, 0, offsets);
    ASSERT_TRUE(g2vertex->selected());
    gvertex->mark_root();

    recut.map_fifo[0].push_back(*vertex);
    recut.connected_map[0].push_back(*vertex);
  }

  for (auto vid : l) {
    cout << "check vid: " << vid << '\n';
    cout << "fifo size: " << recut.connected_map[0].size() << '\n';
    auto offsets = id_to_coord(vid, recut.image_lengths);
    auto vertex = recut.get_or_set_active_vertex(0, 0, offsets, found);
    auto gvertex = recut.get_active_vertex(0, 0, offsets);
    ASSERT_NE(gvertex, nullptr);
    ASSERT_TRUE(gvertex->surface());
    ASSERT_TRUE(found);
    ASSERT_TRUE(vertex->root());
    ASSERT_TRUE(vertex->surface());

    auto msg_vertex = &(recut.connected_map[0].front());
    recut.connected_map[0].pop_front(); // remove it

    ASSERT_TRUE(coord_all_eq(msg_vertex->offsets, offsets));
    ASSERT_TRUE(coord_all_eq(msg_vertex->offsets, vertex->offsets));

    ASSERT_TRUE(msg_vertex->root());
    ASSERT_TRUE(msg_vertex->surface());

    auto global_vertex = &(recut.map_fifo[0].front());
    recut.map_fifo[0].pop_front(); // remove it

    ASSERT_TRUE(coord_all_eq(global_vertex->offsets, offsets));
    ASSERT_TRUE(coord_all_eq(global_vertex->offsets, vertex->offsets));

    ASSERT_TRUE(global_vertex->root());
    ASSERT_TRUE(global_vertex->surface());
  }
  ASSERT_TRUE(recut.connected_map[0].empty());
  ASSERT_TRUE(recut.map_fifo[0].empty());
}
*/

// TEST(Scale, DISABLED_InitializeGlobals) {
// auto grid_size = 2;
// auto args = get_args(grid_size, grid_size, grid_size, 100, 0);

// auto check_block_sizes = [&args](auto image_dims) {
// for (int block_length = 1 << 4; block_length > 4; block_length >>= 1) {
// auto recut = Recut<uint16_t>(args);
// auto block_lengths =
// new_grid_coord(block_length, block_length, block_length);
// auto interval_block_lengths = coord_div(image_dims, block_lengths);
// print_coord(interval_block_lengths, "\tinterval_block_lengths");
// auto interval_block_size = coord_prod_accum(interval_block_lengths);
// cout << "\tblock_length: " << block_length
//<< " interval_block_size: " << interval_block_size << '\n';
// recut.initialize_globals(1, interval_block_size);
//// delete recut;
//}
//};

// auto xy_log2dim = 14;
// auto z_log2dim = 9;
//{
// auto image_dims =
// new_grid_coord(1 << xy_log2dim, 1 << xy_log2dim, 1 << z_log2dim);
// print_coord(image_dims, "medium section");
// check_block_sizes(image_dims);
//}
//}

// TEST(VDB, DISABLED_FloatToPoint) {
// bool print_all = true;
// auto grid_size = 8;
// GridCoord lengths(grid_size);
// auto args = get_args(grid_size, grid_size, grid_size, 100, 7,
//[>force_regenerate_image=<]false,
//[>input_is_vdb=<]true,
//[>type<] "float");
// auto recut = Recut<uint16_t>(args);
// auto root_coords = recut.initialize();

// if (print_all) {
// print_vdb_mask(recut.input_grid->getConstAccessor(), lengths);
// print_vdb_mask(recut.topology_grid->getConstAccessor(), lengths);
// print_all_points(recut.topology_grid, recut.image_bbox);
//}

// auto leaf = recut.topology_grid->probeLeaf();
//{
// auto root_ind = leaf.beginIndexVoxel(root_coords[0].first);
// ASSERT_FALSE(root_ind);
//}

//// Determine the number of points / voxel and points / leaf.
// openvdb::Index LEAF_VOXELS =
// EnlargedPointDataGrid::TreeType::LeafNodeType::SIZE;
// openvdb::Index pointsPerLeaf = VOXEL_POINTS * LEAF_VOXELS;

//// add indices to topology grid
//// Iterate over the leaf nodes in the point tree.
// for (auto leafIter = recut.topology_grid->tree().beginLeaf(); leafIter;
//++leafIter) {
//// Initialize the voxel offsets
// openvdb::Index offset(0);
// for (openvdb::Index index = 0; index < LEAF_VOXELS; ++index) {
// offset += VOXEL_POINTS;
// leafIter->setOffsetOn(index, offset);
//}
//}

//{
// auto root_ind = leaf.beginIndexVoxel(root_coords[0].first);
// ASSERT_TRUE(root_ind);
//}

//// show correct
// print_all_points(recut.topology_grid, recut.image_bbox);

//// recut.activa();
//}

TEST(Update, EachStageIteratively) {
  bool print_all = false;
  bool print_csv = false;
  bool prune = true;
  // for circular test cases app2's accuracy_radius is not correct
  // in terms of hops, 1-norm, manhattan-distance, cell-centered euclidean
  // distance etc. to background voxel, it counts 1 for diagonals to a
  // background, for example 1 0 1 1 1 0 but it should be 1 0 2 1 1 0
  auto expect_exact_radii_match_with_app2 = false;
  auto expect_exact_radii_match_with_seq = true;
  // app2 has a 2D radii estimation which can also be compared
  bool check_xy = false;
  // you need to modify the vdb leaf size to accomplish this:
  bool check_against_sequential = false;

  int max_size = 16;
  // std::vector<int> grid_sizes = {max_size / 16, max_size / 8, max_size / 4,
  // max_size / 2, max_size};
  std::vector<int> grid_sizes = {max_size};
  std::vector<int> interval_sizes = {max_size};
  std::vector<int> block_sizes = {8}; //, max_size / 2, max_size / 4};
  std::vector<bool> input_is_vdbs = {true};
  // tcase 5 is a sphere of radius grid_size / 4 centered
  // in the middle of an image
  // tcase 7 is a square radius grid_size / 4
  // tcase 4 uses rand, different runs produce different results
  std::vector<int> tcases = {5};
  // on tcase 5 a sphere accuracy does not produce correct results according
  // to our definition of hops away, so you'll need to suppress the more
  // stringent tests in order avoid mistaken errors
  int slt_pct = 100;
  auto tile_thresholds = new TileThresholds<uint16_t>(2, 0, 0);
  std::unique_ptr<uint16_t[]> app2_xy_radii_grid;
  if (print_csv) {
    cout << "name,iterations,error_rate(%)\n";
  }
  for (auto input_is_vdb : input_is_vdbs) {
    if (input_is_vdb)
      cout << "Checking input_is_vdb\n";
    for (auto &grid_size : grid_sizes) {
      const VID_t tol_sz = (VID_t)grid_size * grid_size * grid_size;
      auto app2_accurate_radii_grid = std::make_unique<uint16_t[]>(tol_sz);
      auto seq_radii_grid = std::make_unique<uint16_t[]>(tol_sz);
      if (check_xy) {
        app2_xy_radii_grid = std::make_unique<uint16_t[]>(tol_sz);
      }
      for (auto &interval_size : interval_sizes) {
        if (interval_size > grid_size)
          continue;
        for (auto &block_size : block_sizes) {
          if (block_size > interval_size)
            continue;
          auto is_sequential_run =
              (grid_size == interval_size) && (grid_size == block_size) ? true
                                                                        : false;
          for (auto &tcase : tcases) {
            GridCoord grid_extents(grid_size);
            GridCoord interval_extents(interval_size);
            GridCoord block_extents(block_size);

            // Create ground truth refence for the rest of the loop body
            auto ground_truth_args =
                get_args(grid_size, interval_size, block_size, slt_pct, tcase);
            auto ground_truth_params = ground_truth_args.recut_parameters();
            auto ground_truth_image = std::make_unique<uint16_t[]>(tol_sz);
            ASSERT_NE(tcase, 4)
                << "tcase 4 will fail this test since a new random ground "
                   "truth will mismatch the read file\n";
            auto ground_truth_selected = create_image(
                tcase, ground_truth_image.get(), grid_size,
                ground_truth_params.selected, ground_truth_params.root_vid);

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
                            << final_block_size
                            << " input_is_vdb: " << input_is_vdb << '\n';
            SCOPED_TRACE(iteration_trace.str());
            cout << "\n\nStarting: " << iteration_trace.str();

            // Initialize Recut for this loop iteration args
            // always read from file and make sure it matches the ground truth
            // image generated on the fly above
            auto args =
                get_args(grid_size, interval_size, block_size, slt_pct, tcase,
                         /*force_regenerate_image=*/false, input_is_vdb);

            auto recut = Recut<uint16_t>(args);
            auto root_coords = recut.initialize();
            // auto recut_selected = args.recut_parameters().selected;

            if (print_all) {
              std::cout << "ground truth image grid" << endl;
              print_image_3D(ground_truth_image.get(), grid_extents);
              if (input_is_vdb) {
                auto vdb_accessor = recut.topology_grid->getConstAccessor();
                print_vdb_mask(vdb_accessor, interval_extents);
              }
            }

            // RECUT CONNECTED
            {
              auto stage = "connected";
              recut.activate_vids(recut.topology_grid, root_coords, "connected",
                                  recut.map_fifo, recut.connected_map);
              recut.update(stage, recut.map_fifo);
              if (print_all) {
                std::cout << "Recut connected\n";
                std::cout << iteration_trace.str();
                print_all_points(recut.topology_grid, recut.image_bbox, stage);
                std::cout << "Recut surface\n";
                std::cout << iteration_trace.str();
                print_all_points(recut.topology_grid, recut.image_bbox,
                                 "surface");
                auto total = 0;
                if (false) {
                  std::cout << "All surface vids: \n";
                  for (const auto inner : recut.map_fifo) {
                    std::cout << " Block " << inner.first << '\n';
                    for (auto &vertex : inner.second) {
                      total++;
                      cout << "\t" << vertex.description() << '\n';
                      ASSERT_TRUE(vertex.surface());
                      ASSERT_TRUE(vertex.root() || vertex.selected());
                    }
                  }
                  cout << "Surface vid total size " << total << '\n';
                }
              }
            }

            // APP2 RADIUS
            // Get accurate and approximate radii according to APP2
            // methods
            {
              auto total_visited = 0;
              for (VID_t i = 0; i < tol_sz; i++) {
                if (ground_truth_image[i]) {
                  // calculate radius with baseline accurate method
                  app2_accurate_radii_grid[i] =
                      get_radius_accurate(ground_truth_image.get(), grid_size,
                                          i, tile_thresholds->bkg_thresh);
                  if (check_xy) {
                    // build original production version
                    app2_xy_radii_grid[i] = get_radius_hanchuan_XY(
                        ground_truth_image.get(), grid_extents, i,
                        tile_thresholds->bkg_thresh);
                  }
                  ++total_visited;
                }
              }
              ASSERT_EQ(total_visited, ground_truth_selected);

              // Show everything in 3D
              if (print_all) {
                cout << "accuracy_radius\n";
                print_image_3D(app2_accurate_radii_grid.get(), grid_extents);
                if (check_xy) {
                  std::cout << "XY radii grid\n";
                  print_image_3D(app2_xy_radii_grid.get(), grid_extents);
                }
              }

              auto surface_count = 33;
              // make sure APP2 all surface vertices identified correctly
              double recut_vs_app2_accurate_surface_error;
              // only do strict_match comparison with other recut runs
              // because app2 has different radius/distance semantics
              // and will always fail
              EXPECT_NO_FATAL_FAILURE(check_recut_error(
                  recut, app2_accurate_radii_grid.get(), grid_size, "surface",
                  recut_vs_app2_accurate_surface_error, recut.map_fifo,
                  surface_count,
                  /*strict_match=*/false));
            }

            // RECUT RADIUS
            {
              recut.setup_radius(recut.map_fifo);
              // assert conducting update on radius consumes all fifo values
              recut.update("radius", recut.map_fifo);
              for (const auto &m : recut.map_fifo) {
                ASSERT_EQ(m.second.size(), 0);
              }
            }

            // COMPARE RADIUS RECUT AND APP2
            {
              // Debug by eye
              if (print_all) {
                std::cout << "Recut radii\n";
                std::cout << iteration_trace.str();
                print_all_points(recut.topology_grid, recut.image_bbox,
                                 "radius");
              }

              VID_t interval_num = 0;

              if (check_xy) {
                double xy_err;
                ASSERT_NO_FATAL_FAILURE(check_image_error(
                    ground_truth_image.get(), app2_accurate_radii_grid.get(),
                    app2_xy_radii_grid.get(), grid_size, recut.params->selected,
                    xy_err));
                std::ostringstream xy_stream;
                xy_stream << "XY Error " << iteration_trace.str();
                RecordProperty(xy_stream.str(), xy_err);
                if (print_csv) {
                  std::cout << "\"xy_radius/" << grid_size << "\",1," << xy_err
                            << '\n';
                }
              }

              // APP2 ACCURATE (3D) VS Recut
              double recut_vs_app2_accurate_radius_error;
              EXPECT_NO_FATAL_FAILURE(check_recut_error(
                  recut, app2_accurate_radii_grid.get(), grid_size, "radius",
                  recut_vs_app2_accurate_radius_error, recut.map_fifo,
                  ground_truth_selected));
              // see above comment on app2's accuracy_radius
              // the error is still recorded and radii are made sure to be
              // valid in the right locations
              if (expect_exact_radii_match_with_app2) {
                EXPECT_NEAR(recut_vs_app2_accurate_radius_error, 0., .001);
              }
              std::ostringstream recut_stream;
              recut_stream << "Recut Error " << iteration_trace.str();
              RecordProperty(recut_stream.str(),
                             recut_vs_app2_accurate_radius_error);

              if (print_csv) {
                std::cout << "\"fast_marching_radius/" << grid_size << "\",1,"
                          << recut_vs_app2_accurate_radius_error << '\n';
              }

              if (check_against_sequential) {
                // RECUT match with previous recut run exactly
                // check against is_sequential_run recut radii run
                if (is_sequential_run) {
                  // is_sequential_run needs to always be run first
                  for (int vid = 0; vid < tol_sz; ++vid) {
                    auto coord = id_to_coord(vid, recut.image_lengths);
                    if (recut.topology_grid->tree().isValueOn(coord)) {
                      auto leaf =
                          recut.topology_grid->tree().probeConstLeaf(coord);
                      openvdb::points::AttributeHandle<float> radius_handle(
                          leaf->constAttributeArray("pscale"));
                      auto ind = leaf->beginIndexVoxel(coord);
                      if (ind)
                        seq_radii_grid[vid] = radius_handle.get(*ind);
                      else
                        seq_radii_grid[vid] = 0;
                    } else {
                      seq_radii_grid[vid] = 0;
                    }
                  }
                } else {
                  if (print_all) {
                    cout << "recut sequential radii \n";
                    print_image_3D(seq_radii_grid.get(), grid_extents);
                  }
                  double recut_vs_recut_sequential_radius_error;
                  // radii are made sure to be valid in the right locations
                  EXPECT_NO_FATAL_FAILURE(check_recut_error(
                      recut, seq_radii_grid.get(), grid_size, "radius",
                      recut_vs_recut_sequential_radius_error, recut.map_fifo,
                      ground_truth_selected));
                  // exact match at every radii value
                  ASSERT_NEAR(recut_vs_recut_sequential_radius_error, 0.,
                              NUMERICAL_ERROR);
                }
              }
            }

            if (prune) {

              // APP2 PRUNE
              std::vector<MyMarker *> app2_output_tree;
              std::vector<MyMarker *> app2_output_tree_prune;
              {
                std::vector<MyMarker> targets;
                // convert roots into markers (vector)
                std::vector<MyMarker *> root_markers;
                if (tcase == 6) {
                  auto coords = root_coords |
                                rng::views::transform([](auto coord_radius) {
                                  return coord_radius.first;
                                }) |
                                rng::to_vector;
                  root_markers = coords_to_markers(coords);
                } else {
                  root_markers = {get_central_root(grid_size)};
                }
                fastmarching_tree(
                    root_markers, targets, ground_truth_image.get(),
                    app2_output_tree, grid_size, grid_size, grid_size, 1,
                    tile_thresholds->bkg_thresh, tile_thresholds->max_int,
                    tile_thresholds->min_int);

                // get app2 results for radius and pruning from app2
                // take recut results before pruningg and do pruning
                // with app2
                happ(app2_output_tree, app2_output_tree_prune,
                     ground_truth_image.get(), grid_size, grid_size, grid_size,
                     tile_thresholds->bkg_thresh, 0.);
                if (print_all) {
                  std::cout << "APP2 prune\n";
                  std::cout << iteration_trace.str();
                  print_marker_3D(app2_output_tree_prune, interval_extents,
                                  "label");

                  std::cout << "APP2 radius\n";
                  std::cout << iteration_trace.str();
                  print_marker_3D(app2_output_tree_prune, interval_extents,
                                  "radius");
                }
              }

              // RECUT PRUNE
              // starting from roots, prune stage will
              // find final list of vertices
              auto recut_output_tree_prune = std::vector<MyMarker *>();
              {
                // save the topologyto output_tree before starting
                std::cout << iteration_trace.str();
                args.output_tree =
                    convert_to_markers(recut.topology_grid, false);
                auto stage = std::string{"prune"};
                recut.activate_vids(recut.topology_grid, root_coords, stage,
                                    recut.map_fifo, recut.connected_map);
                recut.update(stage, recut.map_fifo);

                recut.adjust_parent();
                if (print_all) {
                  std::cout << "Recut prune\n";
                  std::cout << iteration_trace.str();
                  print_all_points(recut.topology_grid, recut.image_bbox,
                                   "label");

                  std::cout << "Recut radii post prune\n";
                  std::cout << iteration_trace.str();
                  print_all_points(recut.topology_grid, recut.image_bbox,
                                   "radius");

                  std::cout << "Recut post prune valid\n";
                  std::cout << iteration_trace.str();
                  print_all_points(recut.topology_grid, recut.image_bbox,
                                   "valid");
                }

                std::cout << iteration_trace.str();
                recut.print_to_swc();

                recut_output_tree_prune =
                    convert_to_markers(recut.topology_grid, false);
              }

              auto mask = std::make_unique<uint8_t[]>(tol_sz);
              create_coverage_mask_accurate(recut_output_tree_prune, mask.get(),
                                            grid_extents);
              auto results =
                  check_coverage(mask.get(), ground_truth_image.get(), tol_sz,
                                 tile_thresholds->bkg_thresh);
              auto recut_coverage_false_positives =
                  results->false_positives.size();
              auto recut_coverage_false_negatives =
                  results->false_negatives.size();

              auto app2_mask = std::make_unique<uint8_t[]>(tol_sz);
              create_coverage_mask_accurate(app2_output_tree_prune,
                                            app2_mask.get(), grid_extents);
              auto app2_results =
                  check_coverage(app2_mask.get(), ground_truth_image.get(),
                                 tol_sz, tile_thresholds->bkg_thresh);

              auto recut_vs_app2_coverage_results =
                  check_coverage(mask.get(), app2_mask.get(), tol_sz,
                                 tile_thresholds->bkg_thresh);

              if (print_all) {
                std::cout << "Recut coverage mask\n";
                std::cout << iteration_trace.str();
                print_image_3D(mask.get(), interval_extents);

                std::cout << "APP2 coverage mask\n";
                std::cout << iteration_trace.str();
                print_image_3D(app2_mask.get(), interval_extents);

                std::cout << "Ground truth image";
                std::cout << iteration_trace.str();
                print_image_3D(ground_truth_image.get(), interval_extents);
              }

              //// compare_tree will print to log matches, false positive and
              /// negative
              auto compare_tree_results =
                  compare_tree(app2_output_tree_prune, recut_output_tree_prune,
                               grid_extents);

              auto stage = std::string{"prune"};

              RecordProperty("False positives " + stage,
                             results->false_positives.size());
              RecordProperty("False negatives " + stage,
                             results->false_negatives.size());
              RecordProperty("Match count " + stage, results->match_count);
              RecordProperty("Match % " + stage, (100 * results->match_count) /
                                                     args.output_tree.size());
              RecordProperty("Duplicate count " + stage,
                             app2_results->duplicate_count);
              RecordProperty("Total nodes " + stage, args.output_tree.size());
              RecordProperty("Total pruned nodes " + stage,
                             recut_output_tree_prune.size());
              RecordProperty("Compression factor " + stage,
                             args.output_tree.size() /
                                 recut_output_tree_prune.size());

              stage = "APP2 prune";
              RecordProperty("False positives " + stage,
                             app2_results->false_positives.size());
              RecordProperty("False negatives " + stage,
                             app2_results->false_negatives.size());
              RecordProperty("Match count " + stage, app2_results->match_count);
              RecordProperty("Match % " + stage,
                             (100 * app2_results->match_count) /
                                 app2_output_tree.size());
              RecordProperty("Duplicate count " + stage,
                             app2_results->duplicate_count);
              RecordProperty("Total nodes " + stage, app2_output_tree.size());
              RecordProperty("Total pruned nodes " + stage,
                             app2_output_tree_prune.size());
              RecordProperty("Compression factor " + stage,
                             app2_output_tree.size() /
                                 app2_output_tree_prune.size());

              // app2 vs recut
              RecordProperty("APP2 vs recut prune marker false negatives",
                             compare_tree_results->false_negatives.size());
              RecordProperty("APP2 vs recut prune marker false positives",
                             compare_tree_results->false_positives.size());
              RecordProperty("APP2 vs recut prune marker duplicates",
                             compare_tree_results->duplicate_count);

              // make sure the swc is valid by checking all paths
              std::cout << iteration_trace.str();
              check_parents(recut_output_tree_prune, grid_size);

              // For most cases these will always fail since app2 and recut have
              // different hop distance semantics, however coverage is more
              // stringently kept constant between the two
              if (false) {
                std::cout << iteration_trace.str();
                EXPECT_EQ(compare_tree_results->false_negatives.size(), 0);
                EXPECT_EQ(compare_tree_results->false_positives.size(), 0);
                EXPECT_EQ(compare_tree_results->duplicate_count, 0);
              }

              // DILATION_FACTOR 2 makes an exact coverage with the background
              // image
              if (DILATION_FACTOR == 2) {
                // make sure the coverage topology (equivalent active voxels) is
                // the same, this only works if DILATION_FACTOR is
                std::cout << iteration_trace.str();
                EXPECT_EQ(recut_coverage_false_negatives, 0);
                EXPECT_EQ(recut_coverage_false_positives, 0);
              } else {
                // other DILATION_FACTOR s like 1 still have matching coverage
                // between app2 and recut tested for DF 1, 2 and tcase 7
                std::cout << iteration_trace.str();
                EXPECT_EQ(
                    recut_vs_app2_coverage_results->false_negatives.size(), 0);
                EXPECT_EQ(
                    recut_vs_app2_coverage_results->false_positives.size(), 0);
                EXPECT_EQ(recut_vs_app2_coverage_results->duplicate_count, 0);
              }
            }
          }
        }
      }
    }
  }
} // EachStageIteratively

class RecutPipelineParameterTests
    : public ::testing::TestWithParam<
          std::tuple<int, int, int, int, double, bool, bool>> {};

TEST_P(RecutPipelineParameterTests, DISABLED_ChecksIfFinalVerticesCorrect) {

  // documents the meaning of each tuple member
  auto grid_size = std::get<0>(GetParam());
  auto tol_sz = grid_size * grid_size * grid_size;
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
  bool check_against_app2 = std::get<6>(GetParam());

  bool check_coverage_results = false;
  // regenerating image is random and done at run time
  // if you were to set to true tcase 4 would have a mismatch
  // with the loaded image
  bool force_regenerate_image = true;
#ifdef USE_MCP3D
  // force_regenerate_image = false;
#endif
  bool prune = true;
  std::string stage;
  auto grid_extents = GridCoord(grid_size);
  auto interval_extents = GridCoord(interval_size);
  auto block_extents = GridCoord(block_size);

  // shared params
  // generate image so that you can read it below
  // first make sure it can pass
  auto input_is_vdb = true;
  auto args = get_args(grid_size, interval_size, block_size, slt_pct, tcase,
                       force_regenerate_image, input_is_vdb);
  cout << "args.image_root_dir() " << args.image_root_dir() << '\n';
  cout << "image lengths " << grid_extents << '\n';
  cout << "image offsets " << args.image_offsets << '\n';
  // uint16_t is image_t here
  TileThresholds<uint16_t> *tile_thresholds;
  bool print_all = false;

  auto recut = Recut<uint16_t>(args);
  auto root_coords = recut.initialize();
  cout << "recut.image_bbox " << recut.image_bbox << '\n';
  recut.activate_vids(recut.topology_grid, root_coords, "connected",
                      recut.map_fifo, recut.connected_map);

  auto dim = recut.image_bbox.dim().offsetBy(-1);
  cout << dim << '\n';
  cout << recut.image_lengths << '\n';
  ASSERT_EQ(coord_prod_accum(dim), coord_prod_accum(recut.image_lengths));

  std::unique_ptr<uint16_t[]> mask;
  if (check_against_app2) {
    // mask = create_vdb_mask(recut.topology_grid, recut.image_bbox);
    // test using the actual image buffer
    if (const char *env_p = std::getenv("TEST_FLOAT_IMAGE")) {
      std::cout << "Using $TEST_FLOAT_IMAGE environment variable: " << env_p
                << '\n';
      auto base_grid = read_vdb_file(env_p);
      auto input_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
      mask = copy_vdb_to_dense_buffer(input_grid, recut.image_bbox);
    } else {
      std::cout
          << "Warning likely fatal: must run: export "
             "TEST_FLOAT_IMAGE=\"abs/path/to/image\" to set the environment "
             "variable\n\n";
      exit(1);
    }
    if (print_all) {
      print_image_3D(mask.get(), grid_extents);
    }
  }

  // establish the tile thresholds for the entire test run (recut and
  // sequential)
  if (tcase == 6) {
    // tile_thresholds = recut.get_tile_thresholds(image);
    // bkg_thresh table: 421 ~.01 foreground
    // if any pixels are found above or below these values it will fail
    // tile_thresholds = new TileThresholds<uint16_t>(30000, 0, 421);
    tile_thresholds = new TileThresholds<uint16_t>(/*max value*/ 2, 0, 0);
  } else {
    // note these default thresholds apply to any generated image
    // thus they will only be replaced if we're reading a real image
    tile_thresholds = new TileThresholds<uint16_t>(/*max value*/ 2, 0, 0);
  }
  // std::cout << "Using bkg_thresh: " << tile_thresholds->bkg_thresh << '\n';
  RecordProperty("bkg_thresh", tile_thresholds->bkg_thresh);
  RecordProperty("max_int", tile_thresholds->max_int);
  RecordProperty("min_int", tile_thresholds->min_int);

  // Connected
  // update with fixed tile_thresholds for the entire update
  auto connected_update_stats =
      recut.update("connected", recut.map_fifo, tile_thresholds);

  if (print_all) {
    std::cout << "Recut connected\n";
    print_all_points(recut.topology_grid, recut.image_bbox, "label");
  }

  {
    auto interval_open_count = connected_update_stats->interval_open_counts;
    // print vector
    std::ostringstream cat;
    std::copy(interval_open_count.begin(), interval_open_count.end(),
              std::ostream_iterator<int>(cat, ", "));
#ifdef LOG
    std::cout << "Interval reopens " << cat.str() << '\n';
#endif

    auto [mean, sum, std_dev] =
        iter_stats(connected_update_stats->interval_open_counts);
    RecordProperty("Total tile reads", sum);
    RecordProperty("Tile reads mean", mean);
    RecordProperty("Tile reads std", std_dev);
    RecordProperty("Total intervals", interval_open_count.size());
  }

  {
    auto [mean, sum, std_dev] = iter_stats(connected_update_stats->mean_sizes);
    RecordProperty("Mean queue size mean", mean);
    RecordProperty("Mean queue size std", std_dev);
  }

  {
    auto [mean, sum, std_dev] = iter_stats(connected_update_stats->max_sizes);
    RecordProperty("Max queue size mean", mean);
    RecordProperty("Max queue size std", std_dev);
  }

  // RADIUS
  recut.setup_radius(recut.map_fifo);
  auto radius_update_stats = recut.update("radius", recut.map_fifo);

  if (print_all) {
    std::cout << "Recut radius\n";
    print_all_points(recut.topology_grid, recut.image_bbox, "radius");
  }

  // save the output_tree early before it is pruned to compare
  // to app2
  args.output_tree =
      convert_to_markers(recut.topology_grid, /*accept_band*/ false);

  // PRUNE
  auto recut_output_tree_prune = std::vector<MyMarker *>();
  if (prune) {
    stage = std::string{"prune"};
    recut.activate_vids(recut.topology_grid, root_coords, stage, recut.map_fifo,
                        recut.connected_map);
    auto prune_update_stats = recut.update(stage, recut.map_fifo);

    assertm(args.output_tree.size() != 0, "Can not have 0 selected output");
    recut_output_tree_prune.reserve(args.output_tree.size() / 100);

    if (print_all) {
      std::cout << "Recut prune\n";
      print_all_points(recut.topology_grid, recut.image_bbox, "label");

      std::cout << "Recut radii post prune\n";
      print_all_points(recut.topology_grid, recut.image_bbox, "radius");

      std::cout << "Recut post prune valid\n";
      print_all_points(recut.topology_grid, recut.image_bbox, "valid");

      // std::cout << "Recut parent post prune\n";
      // print_all_points(recut.topology_grid, recut.image_bbox, "parent");
    }

    recut.adjust_parent();

    recut.print_to_swc();

    recut_output_tree_prune = convert_to_markers(
        recut.topology_grid,
        /*accept_band*/ false); // this fills args.output_tree
  }

  double actual_slt_pct = (100. * args.output_tree.size()) / tol_sz;
#ifdef LOG
  cout << "Selected " << actual_slt_pct
       << "% of pixels, total count: " << args.output_tree.size() << '\n';
#endif
  RecordProperty("Foreground (%)", actual_slt_pct);
  RecordProperty("Foreground count (pixels)", args.output_tree.size());

  auto record_update_stats = [&](auto args, auto update_stats,
                                 std::string stage) {
    RecordProperty("Selected vertices/s" + stage,
                   args.output_tree.size() / update_stats->total_time);
    RecordProperty("Selected vertices / computation (s)" + stage,
                   args.output_tree.size() / update_stats->computation_time);
    RecordProperty("Selected vertices / IO (s)" + stage,
                   args.output_tree.size() / update_stats->io_time);
    RecordProperty("Iterations" + stage, update_stats->iterations);
    RecordProperty("Value update computation (ms)" + stage,
                   1000 * update_stats->computation_time);
    RecordProperty("Value update IO (ms)" + stage,
                   1000 * update_stats->io_time);
    RecordProperty("Value update elapsed (ms)" + stage,
                   1000 * update_stats->total_time);
    RecordProperty("Value update computation / IO ratio" + stage,
                   update_stats->computation_time / update_stats->io_time);
    RecordProperty("Grid / interval ratio" + stage, grid_size / interval_size);
  };

  // connected update stats
  record_update_stats(args, connected_update_stats.get(), "");
  record_update_stats(args, radius_update_stats.get(), "radius");

  // pregenerated data has a known number of selected
  // pixels
  if (check_against_selected) {
    ASSERT_EQ(args.output_tree.size(), recut.params->selected);
  }

#ifdef USE_MCP3D
  // this runs the original app2 fastmarching algorithm
  // when using the real data, you don't know what the actual
  // selected number should be unless you compare it to another
  // reconstruction method or manual ground truth
  if (check_against_app2) {

    // convert roots into markers (vector)
    std::vector<MyMarker *> root_markers;
    if (tcase == 6) {
      // cout << root_coords << '\n';
      // print_iter(root_coords);
      auto adjusted_roots = root_coords |
                            rng::views::transform([&args](auto coord) {
                              return coord.first - args.image_offsets;
                            }) |
                            rng::to_vector;
      // cout << adjusted_root << '\n';
      print_iter(adjusted_roots);
      root_markers = coords_to_markers(adjusted_roots);
    } else {
      root_markers = {get_central_root(grid_size)};
    }

    std::vector<MyMarker *> app2_output_tree;
    std::vector<MyMarker *> app2_output_tree_prune;
    std::vector<MyMarker> targets;
    auto timer = new high_resolution_timer();
    fastmarching_tree(root_markers, targets, mask.get(), app2_output_tree,
                      recut.image_lengths[0], recut.image_lengths[1],
                      recut.image_lengths[2], 1, tile_thresholds->bkg_thresh);
    // tile_thresholds->max_int, tile_thresholds->min_int);

    auto interval_extents = grid_extents;
    if (print_all) {
      std::cout << "APP2 value\n";
      print_marker_3D(app2_output_tree, interval_extents, "label");
    }

#ifdef LOG
    cout << "app2 fastmarching elapsed (s)" << timer->elapsed() << '\n';
#endif

    // warning record property will auto cast to an int
    RecordProperty("APP2 elapsed (ms)", 1000 * timer->elapsed());
    RecordProperty("Recut speedup factor %",
                   100 *
                       (connected_update_stats->total_time / timer->elapsed()));

    RecordProperty("APP2 tree size", app2_output_tree.size());
    auto diff = absdiff(app2_output_tree.size(), args.output_tree.size());
    RecordProperty("Error", diff);
    RecordProperty("Error rate (%)", 100 * (diff / app2_output_tree.size()));

    // compare_tree will print to log matches, false positive and negative
    auto results =
        compare_tree(app2_output_tree, args.output_tree, grid_extents);

    stage = "connected";
    RecordProperty("False positives " + stage, results->false_positives.size());
    RecordProperty("False negatives " + stage, results->false_negatives.size());
    RecordProperty("Match count " + stage, results->match_count);

    // it's a problem if two markers with same vid are in a results vector
    ASSERT_EQ(results->duplicate_count, 0);

    // check the compare tree worked properly
    ASSERT_EQ(args.output_tree.size(),
              results->match_count + results->false_positives.size());
    ASSERT_EQ(app2_output_tree.size(),
              results->match_count + results->false_negatives.size());

    EXPECT_EQ(results->false_positives.size(), 0);
    EXPECT_EQ(app2_output_tree.size(), args.output_tree.size());
    EXPECT_EQ(results->false_negatives.size(), 0);

    if (prune) {
      // run the seq version from app2 to compare
      happ(app2_output_tree, app2_output_tree_prune, mask.get(),
           recut.image_lengths[0], recut.image_lengths[1],
           recut.image_lengths[2], tile_thresholds->bkg_thresh, 0.);
      cout << "Finished app2 happ\n";

      if (print_all) {
        std::cout << "APP2 prune\n";
        print_marker_3D(app2_output_tree_prune, interval_extents, "label");

        std::cout << "APP2 radius\n";
        print_marker_3D(app2_output_tree_prune, interval_extents, "radius");
      }

      if (check_coverage_results) {
        auto prune_mask = std::make_unique<uint8_t[]>(tol_sz);
        create_coverage_mask_accurate(recut_output_tree_prune, prune_mask.get(),
                                      grid_extents);
        auto results = check_coverage(prune_mask.get(), mask.get(), tol_sz,
                                      tile_thresholds->bkg_thresh);

        auto app2_mask = std::make_unique<uint8_t[]>(tol_sz);
        create_coverage_mask_accurate(app2_output_tree_prune, app2_mask.get(),
                                      grid_extents);
        auto app2_results = check_coverage(app2_mask.get(), mask.get(), tol_sz,
                                           tile_thresholds->bkg_thresh);

        if (print_all) {
          std::cout << "Recut coverage mask\n";
          print_image_3D(prune_mask.get(), interval_extents);

          std::cout << "APP2 coverage mask\n";
          print_image_3D(app2_mask.get(), interval_extents);
        }

        // compare_tree will print to log matches, false positive and negative
        // results = compare_tree(app2_output_tree_prune,
        // recut_output_tree_prune, grid_size, grid_size, recut);

        stage = "prune";
        RecordProperty("False positives " + stage,
                       results->false_positives.size());
        RecordProperty("False negatives " + stage,
                       results->false_negatives.size());
        RecordProperty("Match count " + stage, results->match_count);
        RecordProperty("Match % " + stage,
                       (100 * results->match_count) / args.output_tree.size());
        RecordProperty("Duplicate count " + stage, results->duplicate_count);
        RecordProperty("Total nodes " + stage, args.output_tree.size());
        RecordProperty("Total pruned nodes " + stage,
                       recut_output_tree_prune.size());
        RecordProperty("Compression factor " + stage,
                       args.output_tree.size() /
                           recut_output_tree_prune.size());

        stage = "app2 prune";
        RecordProperty("False positives " + stage,
                       app2_results->false_positives.size());
        RecordProperty("False negatives " + stage,
                       app2_results->false_negatives.size());
        RecordProperty("Match count " + stage, app2_results->match_count);
        RecordProperty("Match % " + stage, (100 * app2_results->match_count) /
                                               app2_output_tree.size());
        RecordProperty("Duplicate count " + stage,
                       app2_results->duplicate_count);
        RecordProperty("Total nodes " + stage, app2_output_tree.size());
        RecordProperty("Total pruned nodes " + stage,
                       app2_output_tree_prune.size());
        RecordProperty("Compression factor " + stage,
                       app2_output_tree.size() / app2_output_tree_prune.size());

      } // check_coverage_results

      // it's a problem if two markers with same vid are in a results vector
      // ASSERT_EQ(results->duplicate_count, 0);

      //// check the compare tree worked properly
      // ASSERT_EQ(app2_output_tree_prune.size(),
      // results->match_count + results->false_positives.size());
      // ASSERT_EQ(recut_output_tree_prune.size(),
      // results->match_count + results->false_negatives.size());
      // EXPECT_EQ(app2_output_tree_prune.size(),
      // recut_output_tree_prune.size());

      check_parents(recut_output_tree_prune, grid_size);

      // EXPECT_EQ(results->false_positives.size(), 0)
      //<< "In comparison, app2 is " << app2_results->false_positives.size();
      // EXPECT_EQ(results->false_negatives.size(), 0)
      //<< "In comparison, app2 is " << app2_results->false_negatives.size();

      // print all SWC files for visualization of outputs
      marker_to_swc_file("app2-checks-if.swc", app2_output_tree_prune);
      // recut
      marker_to_swc_file("recut-checks-if.swc", recut_output_tree_prune);
    }
  }
#endif
} // ChecksIfFinalVerticesCorrect

// ... check_against_selected, check_against_app2
INSTANTIATE_TEST_CASE_P(
    RecutPipelineTests, RecutPipelineParameterTests,
    ::testing::Values(
        // check if using better parent characteristics for parent traversal
        // improves the overall swc quality and proofreadability
        std::make_tuple(2048, 2048, 2048, 6, 100., false, true) // 0
        // std::make_tuple(4, 4, 4, 0, 100., true, false), // 0
        // std::make_tuple(4, 4, 4, 1, 100., true, false), // 1
        // std::make_tuple(4, 4, 4, 2, 100., true, false), // 2
        // std::make_tuple(4, 2, 2, 2, 100., true, false)  // 3
        //#ifdef USE_MCP3D
        //,
        //// check_against_app2 (final boolean below) currently uses
        //// MCP3D's reading of image to ensure that both sequential and recut
        //// use the same iamge, to include these test while testing against
        //// sequential change the implementation so that the generated image
        //// from recut is saved and pass it to fastmarching_tree
        // std::make_tuple(4, 4, 4, 4, 50., true, true), // 4
        //// multi-interval small
        // std::make_tuple(4, 2, 2, 4, 50., true, true), // 5
        //// multi-block small
        // std::make_tuple(4, 4, 2, 4, 50., true, true) // 6
        //#ifdef TEST_ALL_BENCHMARKS // test larger portions that must be
        // verified for
        //// these must have TEST_IMAGE and TEST_MARKER environment variables
        //// set
        //,
        //// make sure if bkg_thresh is 0, all vertices are selected for real
        // std::make_tuple(4, 4, 4, 6, 100., false, true), // 7
        // make sure fastmarching_tree and recut produce exact match for real
        // std::make_tuple(8, 8, 8, 6, 100., false, true) // 8
        // real data multi-interval
        // std::make_tuple(8, 4, 4, 6, 100., false, true), // 10
        // interval grid ratio tests
        // std::make_tuple(128, 64, 64, 6, 1, false, true),    // 11
        // std::make_tuple(256, 128, 128, 6, 1, false, true),  // 12
        // std::make_tuple(512, 256, 256, 6, 1, false, true),  // 13
        // std::make_tuple(1024, 512, 512, 6, 1, false, true), // 14
        // std::make_tuple(2048, 1024, 1024, 6, 1, false,
        // false), // out of memory for fastmarching_tree
        //// std::make_tuple(4096, 2048, 2048, 6, 1, false, false), // 16
        //// std::make_tuple(8192, 4096, 4096, 6, 1, false, false), // 17

        // std::make_tuple(128, 32, 32, 6, 1, false, true), // 18
        // std::make_tuple(256, 64, 64, 6, 1, false, true),
        // std::make_tuple(512, 128, 128, 6, 1, false, true),
        // std::make_tuple(1024, 256, 256, 6, 1, false, true),
        // std::make_tuple(2048, 512, 512, 6, 1, false, false),
        // std::make_tuple(4096, 1024, 1024, 6, 1, false, false),
        //// std::make_tuple(8192, 2048, 2048, 6, 1, false, false), // 24

        // std::make_tuple(128, 16, 16, 6, 1, false, true), // 25
        // std::make_tuple(256, 32, 32, 6, 1, false, true),
        // std::make_tuple(512, 64, 64, 6, 1, false, true),
        // std::make_tuple(1024, 128, 128, 6, 1, false, true),
        // std::make_tuple(2048, 256, 256, 6, 1, false, false),
        // std::make_tuple(4096, 512, 512, 6, 1, false, false),
        // std::make_tuple(8192, 1024, 1024, 6, 1, false, false), // 31

        //// 1:1 grid:interval ratios
        // std::make_tuple(16, 16, 16, 6, 1, false, true), // 32
        // std::make_tuple(32, 32, 32, 6, 1, false, true),
        // std::make_tuple(64, 64, 64, 6, 1, false, true),
        // std::make_tuple(128, 128, 128, 6, 1, false, true),
        // std::make_tuple(256, 256, 256, 6, 1, false, true),
        // std::make_tuple(512, 512, 512, 6, 1, false, true),
        // std::make_tuple(1024, 1024, 1024, 6, 1, false, true), // 38

        //// block parallelism
        // std::make_tuple(512, 512, 8, 6, 1, false, true), // 39
        // std::make_tuple(512, 512, 16, 6, 1, false, true),
        // std::make_tuple(512, 512, 32, 6, 1, false, true),
        // std::make_tuple(512, 512, 64, 6, 1, false, true),
        // std::make_tuple(512, 512, 128, 6, 1, false, true),
        // std::make_tuple(512, 512, 256, 6, 1, false, true),
        // std::make_tuple(512, 512, 512, 6, 1, false, true),

        // std::make_tuple(1024, 512, 32, 6, 1, false, true), // 46
        // std::make_tuple(1024, 512, 64, 6, 1, false, true),
        // std::make_tuple(1024, 512, 128, 6, 1, false, true),
        // std::make_tuple(1024, 512, 256, 6, 1, false, true),
        // std::make_tuple(1024, 512, 512, 6, 1, false, true),

        // std::make_tuple(2048, 512, 32, 6, 1, false, false), // 49
        // std::make_tuple(2048, 512, 64, 6, 1, false, false),
        // std::make_tuple(2048, 512, 128, 6, 1, false, false),
        // std::make_tuple(2048, 512, 256, 6, 1, false, false),
        // std::make_tuple(2048, 512, 512, 6, 1, false, false),

        // std::make_tuple(4096, 512, 32, 6, 1, false, false), // 54
        // std::make_tuple(4096, 512, 64, 6, 1, false, false),
        // std::make_tuple(4096, 512, 128, 6, 1, false, false),
        // std::make_tuple(4096, 512, 256, 6, 1, false, false),
        // std::make_tuple(4096, 512, 512, 6, 1, false, false),

        // std::make_tuple(8192, 512, 32, 6, 1, false, false), // 59
        // std::make_tuple(8192, 512, 64, 6, 1, false, false),
        // std::make_tuple(8192, 512, 128, 6, 1, false, false),
        // std::make_tuple(8192, 512, 256, 6, 1, false, false),
        // std::make_tuple(8192, 512, 512, 6, 1, false, false),

        // std::make_tuple(1024, 1024, 128, 6, 1, false, false), // 63
        // std::make_tuple(2048, 2048, 128, 6, 1, false, false), // 64
        // std::make_tuple(4096, 4096, 128, 6, 1, false, false), // 65
        // std::make_tuple(8192, 8192, 128, 6, 1, false, false)
        //#endif
        //#endif
        ));

TEST(RecutPipeline, PrintDefaultInfo) {
  auto v1 = new VertexAttr();
  auto vs = sizeof(VertexAttr);
  cout << "sizeof vertex " << vs << " bytes" << endl;
  cout << "AvailMem " << GetAvailMem() / (1024 * 1024 * 1024) << " GB" << endl;
  cout << "Data directory referenced by this test binary: " << get_data_dir()
       << '\n';
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::Test::RecordProperty("GlobalProperty", XSTR(GIT_HASH));
  // record_property_pp_macros
#ifdef USE_OMP_BLOCK
  testing::Test::RecordProperty("USE_OMP_BLOCK", 1);
#endif
#ifdef USE_OMP_INTERVAL
  testing::Test::RecordProperty("USE_OMP_INTERVAL", 1);
#endif
#ifdef NO_RV
  testing::Test::RecordProperty("NO_RV", 1);
#endif
#ifdef FULL_PRINT
  // this significantly slows performance so it should be stamped to any
  // performance stats
  testing::Test::RecordProperty("FULL_PRINT", 1);
#endif
#ifdef USE_MCP3D
  testing::Test::RecordProperty("USE_MCP3D", 1);
#endif

  // warning: needs to be called once per executable before any related
  // function is called otherwise confusing seg faults ensue
  openvdb::initialize();
#ifdef CUSTOM_GRID
  EnlargedPointDataGrid::registerGrid();
#endif

  return RUN_ALL_TESTS();
}
