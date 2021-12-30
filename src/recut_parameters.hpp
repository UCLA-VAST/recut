#pragma once

#include "config.hpp"
#include "markers.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
#include <omp.h>
#endif

class RecutParameters {
public:
  RecutParameters() {
    gsdt_ = false;
    output_windows_ = std::string();
    coverage_prune_ = true;
    prune_ = 1;
    allow_gap_ = false;
    histogram_ = false;
    background_thresh_ = -1;
    foreground_percent_ = -0.01;
    cnn_type_ = 2;
    sr_ratio_ = 1.0 / 3;
    cube_256_ = false;
    restart_factor_ = 0.0;
    downsample_factor_ = 1;
    restart_ = false;
    radius_from_2d_ = true;
    swc_resample_ = true;
    high_intensity_ = false;
    brightfield_ = false;
    marker_file_path_ = std::string();
    out_vdb_ = std::string();
    convert_only_ = false;
    parallel_num_ = 1; // default is the max hardware concurrency when not set
    max_intensity_ = -1;
    min_intensity_ = -1;
    // no getters or setters
    force_regenerate_image = false;
    tcase = -1; // -1 indicates it's not a test case
    slt_pct = -1;
    selected = 0;
    root_vid = std::numeric_limits<uint64_t>::max();
    interval_length = 0;
    upsample_z_ = 1; // i.e. no upsampling
  }
  std::string MetaString();
  // getters
  bool gsdt() const { return gsdt_; }
  bool coverage_prune() const { return coverage_prune_; }
  int prune() const { return prune_; }
  bool allow_gap() const { return allow_gap_; }
  int background_thresh() const { return background_thresh_; }
  double foreground_percent() const { return foreground_percent_; }
  int cnn_type() const { return cnn_type_; }
  double sr_ratio() const { return sr_ratio_; }
  double restart_factor() const { return restart_factor_; }
  bool restart() const { return restart_; }
  bool cube_256() const { return cube_256_; }
  bool radius_from_2d() const { return radius_from_2d_; }
  bool swc_resample() const { return swc_resample_; }
  bool high_intensity() const { return high_intensity_; }
  bool brightfield() const { return brightfield_; }
  std::string marker_file_path() const { return marker_file_path_; }
  bool paralel_num() const { return parallel_num_; }
  double get_max_intensity() { return max_intensity_; }
  double get_min_intensity() { return min_intensity_; }
  void set_restart_factor(double factor) { restart_factor_ = factor; }
  void set_restart(bool restart) { restart_ = restart; }
  void set_parallel_num(int num) { parallel_num_ = num; }
  void set_gsdt(bool is_gsdt) { gsdt_ = is_gsdt; }
  void set_histogram(bool histogram) { histogram_ = histogram; }
  void set_coverage_prune(bool coverage_prune) {
    coverage_prune_ = coverage_prune;
  }
  void set_prune(int prune) { prune_ = prune; }
  void set_allow_gap(bool allow_gap) { allow_gap_ = allow_gap; }
  void set_combine(std::string second_grid) { second_grid_ = second_grid; }
  void set_background_thresh(int background_thresh) {
    background_thresh_ = std::max(background_thresh, 0);
  }
  // setting foreground_percent forces
  // calculation of desired background_thresh during
  // get_tile_thresholds
  void set_foreground_percent(double foreground_percent) {
    foreground_percent_ = std::max(0.0, std::min(100.0, foreground_percent));
  }
  void set_cnn_type(int cnn_type) {
    if (cnn_type == 1 || cnn_type == 2 || cnn_type == 3)
      cnn_type_ = cnn_type;
  }
  void set_sr_ratio(double sr_ratio) {
    sr_ratio_ = std::max(1e-9, std::min(1.0, sr_ratio));
  }
  void set_cube_256(bool cube_256) { cube_256_ = cube_256; }
  void set_radius_from_2d(bool radius_from_2d) {
    radius_from_2d_ = radius_from_2d;
  }
  void set_swc_resample(bool swc_resample) { swc_resample_ = swc_resample; }
  void set_high_intensity(bool high_intensity) {
    high_intensity_ = high_intensity;
  }
  void set_brightfield(bool brightfield) { brightfield_ = brightfield; }
  void set_marker_file_path(const std::string &marker_file_path) {
    marker_file_path_ = marker_file_path;
  }
  void set_out_vdb(const std::string &out_vdb) { out_vdb_ = out_vdb; }
  void set_convert_only(const bool &convert_only) {
    convert_only_ = convert_only;
  }
  void set_max_intensity(double max_intensity) {
    max_intensity_ = max_intensity;
  }
  void set_min_intensity(double min_intensity) {
    min_intensity_ = min_intensity;
  }
  void set_output_windows(std::string fn) { output_windows_ = fn; }
  void set_downsample_factor(int factor) { downsample_factor_ = factor; }
  void set_upsample_z(int upsample_z) { upsample_z_ = upsample_z; }

  // no getters
  uint64_t selected, root_vid;
  bool force_regenerate_image, convert_only_, histogram_, gsdt_,
      coverage_prune_, allow_gap_, cube_256_, radius_from_2d_, swc_resample_,
      high_intensity_, brightfield_, restart_;
  int background_thresh_, cnn_type_, parallel_num_, prune_, interval_length,
      tcase, slt_pct, downsample_factor_, upsample_z_;
  double foreground_percent_, sr_ratio_, restart_factor_,
      max_intensity_, min_intensity_;
  std::string marker_file_path_, out_vdb_, second_grid_, output_windows_;
};

class RecutCommandLineArgs {
public:
  RecutCommandLineArgs()
      : recut_parameters_(RecutParameters{}), image_root_dir_(std::string()),
        swc_path_("out.swc"), channel(0), resolution_level_(0),
        image_offsets(0, 0, 0), image_lengths(-1, -1, -1), type_("point"),
        prune_radius_(ANISOTROPIC_FACTOR), run_app2(false),
        user_thread_count(tbb::info::default_concurrency()), interval_z(1),
        min_branch_length(MIN_BRANCH_LENGTH) {}

  static void PrintUsage();
  std::string MetaString();
  void PrintParameters() { std::cout << MetaString() << std::endl; }

  // getters
  RecutParameters &recut_parameters() { return recut_parameters_; }
  std::string image_root_dir() const { return image_root_dir_; }
  std::string swc_path() const { return swc_path_; }
  int resolution_level() const { return resolution_level_; }

  // setters
  void set_type(const std::string &type) { type_ = type; }

  void set_recut_parameters(RecutParameters &params) {
    recut_parameters_ = params;
  }

  void set_image_root_dir(const std::string &image_root_dir) {
    image_root_dir_ = image_root_dir;
  }

  void set_swc_path(const std::string &swc_path) { swc_path_ = swc_path; }

  void set_prune_radius(uint16_t pr) { prune_radius_ = pr; }

  void set_resolution_level(int resolution_level) {
    resolution_level_ = resolution_level;
  }

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

  RecutParameters recut_parameters_;
  std::string image_root_dir_, swc_path_, type_;
  uint16_t prune_radius_, user_thread_count, interval_z, min_branch_length, resolution_level_, channel;
  bool run_app2;
};

RecutCommandLineArgs ParseRecutArgsOrExit(int argc, char *argv[]);
