#include "../src/recut.hpp"
#include <benchmark/benchmark.h>

#ifdef USE_MCP3D
#define GEN_IMAGE false
#else
#define GEN_IMAGE true
#endif

/*
 * Benchmark loops over possible image sizes
 * grid_size is the length of one dimension
 * of this image cube
 * If you want to run a benchmark for only a specific
 * grid size of say 128, run:
 * `./recut_bench --benchmark_filter=bench_critical_loop/128
 */
static void bench_critical_loop(benchmark::State &state) {
  auto grid_size = state.range(0);
  double slt_pct = 1;
  int tcase = 4;
  auto args = get_args(grid_size, slt_pct, tcase, GEN_IMAGE);
  VID_t selected = args.recut_parameters().selected;

  // adjust final runtime parameters
  auto params = args.recut_parameters();
  // the total number of intervals allows more parallelism
  // ideally intervals >> thread count
  params.set_interval_size(grid_size);
  params.set_block_size(grid_size);
  // by setting the max intensities you do not need to recompute them
  // in the update function, this is critical for benchmarking
  params.set_max_intensity(1);
  params.set_min_intensity(0);
  args.set_recut_parameters(params);

  // Recut is templated by its image type
  auto recut = Recut<uint16_t>(args);

  // this creates the test image for this
  // grid_size it is likely the slowest part of each
  // bench run since it is the setup
  // this is why it is not in the performance
  // while loop below
  // When we do `perf recut_bench` this
  // initialize portion will still be included
  recut.initialize();

  // This is the performance loop where gbench
  // will execute until timing stats converge
  while (state.KeepRunning()) {
    // reactivates
    // the intervals of the root and readds
    // them to the respective heaps
    recut.setup_value();
    // from our gen img of initialize update
    // does fastmarching, updated vertices
    // are mmap'd
    recut.update("value");

    // to destroy the information for this run
    // so that it doesn't affect the next run
    // the vertices must be unmapped
    // done via `release()`
    recut.release();
    // benchmark::DoNotOptimize();
  }
  // items processed only refers to the total selected vertices
  // processed, note this is subject to the select percent
  // defined above as .01 which means on 1% of the total vertices
  // will be selected
  state.SetBytesProcessed(state.iterations() * selected * 26);
  // state.SetLabel(std::to_string(selected * 26 / 1024) + "kB");
  state.SetItemsProcessed(state.iterations() * selected);
}
BENCHMARK(bench_critical_loop)
    ->RangeMultiplier(2)
    ->Range(16, 256)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

static void fast_marching_radius(benchmark::State &state) {
  std::vector<int> tcases = {5};
  int slt_pct = 100;
  bool print_all = false;
  uint16_t bkg_thresh = 0;
  auto grid_size = state.range(0);

  for (auto &tcase : tcases) {
    auto args = get_args(grid_size, slt_pct, tcase, true);

    // adjust final runtime parameters
    auto params = args.recut_parameters();
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    params.set_interval_size(grid_size);
    params.set_block_size(grid_size);
    args.set_recut_parameters(params);

    // run
    auto recut = Recut<uint16_t>(args);
    recut.initialize();

    while (state.KeepRunning()) {
      // warning: pause and resume high overhead
      state.PauseTiming();
      recut.setup_value();
      recut.update("value");
      recut.setup_radius();
      state.ResumeTiming();

      recut.update("radius");
      recut.release();
    }
  }
}
BENCHMARK(fast_marching_radius)
    ->RangeMultiplier(2)
    ->Range(8, 128)
    ->Unit(benchmark::kMillisecond);

static void accurate_radius(benchmark::State &state) {
  std::vector<int> tcases = {5};
  int slt_pct = 100;
  bool print_all = false;
  uint16_t bkg_thresh = 0;
  auto grid_size = state.range(0);
  VID_t tol_sz = (VID_t)grid_size * grid_size * grid_size;
  uint16_t *radii_grid = new uint16_t[tol_sz];
  for (auto &tcase : tcases) {
    auto args = get_args(grid_size, slt_pct, tcase, true);

    // adjust final runtime parameters
    auto params = args.recut_parameters();
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    params.set_interval_size(grid_size);
    params.set_block_size(grid_size);
    args.set_recut_parameters(params);

    // run
    auto recut = Recut<uint16_t>(args);
    recut.initialize();
    VID_t interval_num = 0;

    while (state.KeepRunning()) {
      // calculate radius with baseline accurate method
      for (VID_t i = 0; i < tol_sz; i++) {
        if (recut.generated_image[i]) {
          radii_grid[i] = get_radius_accurate(recut.generated_image, grid_size,
                                              i, bkg_thresh);
        }
      }
    }
  }
  delete[] radii_grid;
}
BENCHMARK(accurate_radius)
    ->RangeMultiplier(2)
    ->Range(8, 128)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

static void xy_radius(benchmark::State &state) {
  std::vector<int> tcases = {5};
  int slt_pct = 100;
  bool print_all = false;
  uint16_t bkg_thresh = 0;
  auto grid_size = state.range(0);
  VID_t tol_sz = (VID_t)grid_size * grid_size * grid_size;
  uint16_t *radii_grid_xy = new uint16_t[tol_sz];
  for (auto &tcase : tcases) {
    auto args = get_args(grid_size, slt_pct, tcase, true);

    // adjust final runtime parameters
    auto params = args.recut_parameters();
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    params.set_interval_size(grid_size);
    params.set_block_size(grid_size);
    args.set_recut_parameters(params);

    // run
    auto recut = Recut<uint16_t>(args);
    recut.initialize();
    VID_t interval_num = 0;

    while (state.KeepRunning()) {
      // build original production version
      for (VID_t i = 0; i < tol_sz; i++) {
        if (recut.generated_image[i]) {
          radii_grid_xy[i] = get_radius_hanchuan_XY(recut.generated_image,
                                                    grid_size, i, bkg_thresh);
        }
      }
    }
  }
  delete[] radii_grid_xy;
}
BENCHMARK(xy_radius)
    ->RangeMultiplier(2)
    ->Range(8, 128)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
