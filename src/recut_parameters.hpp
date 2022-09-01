#pragma once

#include "config.hpp"
#include "markers.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <optional>

class RecutCommandLineArgs {
public:
  RecutCommandLineArgs()
      : input_path(std::string()), channel(0), resolution_level(0),
        image_offsets(0, 0, 0), image_lengths(-1, -1, -1), input_type("point"),
        output_type("point"), run_app2(false),
        user_thread_count(tbb::info::default_concurrency()),
        tile_lengths({-1, -1, -1}), min_branch_length(MIN_BRANCH_LENGTH),
        convert_only(true), output_name("out.vdb"), background_thresh(-1),
        foreground_percent(-0.01), combine(false), histogram(false),
        window_grid_paths(std::vector<std::string>()),
        second_grid(std::string()), upsample_z(1), downsample_factor(1),
        max_intensity(-1), min_intensity(-1), expand_window_um(0.),
        min_window_um(0.),
        voxel_size({1., 1., 1.}) {}

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

  std::string input_path, input_type, output_type, output_name, seed_path,
      second_grid;
  std::vector<std::string> window_grid_paths;
  uint16_t user_thread_count, min_branch_length, resolution_level,
      channel, upsample_z, downsample_factor;
  int background_thresh, max_intensity, min_intensity, tcase;
  double foreground_percent, slt_pct;
  float min_window_um, expand_window_um;
  VID_t selected, root_vid;
  bool run_app2, convert_only, combine, histogram;
  std::array<int, 3> tile_lengths;
  std::array<float, 3> voxel_size;
  std::optional<uint16_t> prune_radius;
};

RecutCommandLineArgs ParseRecutArgsOrExit(int argc, char *argv[]);
