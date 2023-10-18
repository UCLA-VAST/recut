#pragma once

#include "config.hpp"
#include "markers.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

class RecutCommandLineArgs {
public:
  RecutCommandLineArgs()
      : input_path(std::string()), channel(0), resolution_level(0),
        image_offsets(0, 0, 0), image_lengths(-1, -1, -1), input_type("point"),
        output_type("swc"), run_app2(false),
        user_thread_count(tbb::info::default_concurrency()),
        tile_lengths({-1, -1, -1}), min_branch_length(MIN_BRANCH_LENGTH),
        convert_only(false), output_name("out.vdb"), background_thresh(-1),
        foreground_percent(FG_PCT), combine(false), histogram(false),
        window_grid_paths(std::vector<std::string>()),
        second_grid(std::string()), upsample_z(1), downsample_factor(1),
        max_intensity(-1), min_intensity(-1),
        expand_window_um(EXPAND_WINDOW_UM), min_window_um(MIN_WINDOW_UM),
        morphological_operations_order(1),
        min_radius_um(MIN_SOMA_RADIUS_UM), max_radius_um(MAX_SOMA_RADIUS_UM),
        voxel_size({1., 1., 1.}), save_vdbs(false), 
        ignore_multifurcations(false), close_topology(true),
        mean_shift_max_iters(4), timeout(MEAN_SHIFT_TIMEOUT),
        skeleton_grain(SKELETON_GRAIN), skeleton_grow(GROW_THRESHOLD),
        smooth_steps(SMOOTH_STEPS),
        mesh_grain(MESH_GRAIN), mean_shift_factor(0), 
        seed_action("find"), optimize_steps(5),
        disable_swc_scaling(false) {}

  static void PrintUsage();
  std::string MetaString();
  void PrintParameters() { std::cout << MetaString() << std::endl; }

  // setters
  void set_image_offsets(const GridCoord &image_offsets) {
    for (size_t i = 0; i < 3; ++i) {
      this->image_offsets[i] = image_offsets[i];
    }
  }

  void set_image_lengths(const GridCoord &image_lengths) {
    for (size_t i = 0; i < 3; ++i) {
      this->image_lengths[i] = image_lengths[i];
    }
  }

  std::vector<MyMarker *> output_tree;
  GridCoord image_offsets, image_lengths;

  std::filesystem::path input_path, seed_path;
  std::string input_type, output_type, output_name, second_grid, seed_action;
  std::vector<std::string> window_grid_paths;
  uint16_t user_thread_count, min_branch_length, resolution_level, channel,
      upsample_z, downsample_factor, morphological_operations_order,
      mean_shift_max_iters, smooth_steps;
  int background_thresh, max_intensity, min_intensity, tcase, timeout,
      skeleton_grow, optimize_steps;
  double foreground_percent, slt_pct;
  float min_window_um, expand_window_um, min_radius_um, max_radius_um,
      mean_shift_factor, skeleton_grain, mesh_grain;
  VID_t selected, root_vid;
  bool run_app2, convert_only, combine, histogram, save_vdbs,
      ignore_multifurcations, close_topology, disable_swc_scaling;
  std::array<int, 3> tile_lengths;
  std::array<double, 3> voxel_size;
  std::optional<float> prune_radius, soma_dilation;
  std::optional<int> close_steps, open_steps, saturate_edges, coarsen_steps;
  std::optional<std::filesystem::path> test;
};

RecutCommandLineArgs ParseRecutArgsOrExit(int argc, char *argv[]);
