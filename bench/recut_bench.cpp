#include "recut.hpp"
#include "../test/app2_helpers.hpp"
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
  // the total number of intervals allows more parallelism
  // ideally intervals >> thread count
  auto args =
      get_args(grid_size, grid_size, grid_size, slt_pct, tcase, GEN_IMAGE);
  VID_t selected = args.recut_parameters().selected;

  // adjust final runtime parameters
  auto params = args.recut_parameters();
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
  auto root_vids = recut.initialize();

  // This is the performance loop where gbench
  // will execute until timing stats converge
  for (auto _ : state) {
    // reactivates
    // the intervals of the root and readds
    // them to the respective heaps
    recut.activate_vids(root_vids, "value", recut.global_fifo);
    // from our gen img of initialize update
    // does fastmarching, updated vertices
    // are mmap'd
    recut.update("value", recut.global_fifo);

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

static void recut_radius_real_data(benchmark::State &state) {
  std::vector<int> tcases = {6};
  int slt_pct = 1;
  uint16_t bkg_thresh = 0;
  auto grid_size = state.range(0);
  auto interval_size = state.range(1);
  auto block_size = state.range(2);
  TileThresholds<uint16_t> *tile_thresholds;
  auto force_regenerate_image = false;

  for (auto &tcase : tcases) {
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    auto args = get_args(grid_size, interval_size, block_size, slt_pct, tcase,
                         force_regenerate_image);

    // establish the tile thresholds for the entire test run (recut and
    // sequential) to prevent unnecessary recalculations of theshodls
    if (tcase == 6) {
      // tile_thresholds = recut.get_tile_thresholds(image);
      // bkg_thresh table: 421 ~.01 foreground
      // if any pixels are found above or below these values it will fail
      tile_thresholds = new TileThresholds<uint16_t>(30000, 0, 421);
    } else {
      // note these default thresholds apply to any generated image
      // thus they will only be replaced if we're reading a real image
      tile_thresholds = new TileThresholds<uint16_t>(2, 0, 0);
    }

    // run
    auto recut = Recut<uint16_t>(args);
    auto root_vids = recut.initialize();

    for (auto _ : state) {
      // warning: pause and resume high overhead
      state.PauseTiming();
      recut.activate_vids(root_vids, "value", recut.global_fifo);
      recut.update("value", recut.global_fifo, tile_thresholds);
      recut.setup_radius(recut.global_fifo);
      state.ResumeTiming();

      recut.update("radius", recut.global_fifo);
      recut.release();
    }
  }
}
BENCHMARK(recut_radius_real_data)
    ->Args({512, 512, 16})
    ->Args({512, 512, 32})
    ->Args({512, 512, 64})
    ->Args({512, 512, 128})
    ->Args({512, 512, 256})
    ->Args({512, 512, 512})
    //->Args({1024, 512, 32})
    //->Args({1024, 512, 64})
    //->Args({1024, 512, 128})
    //->Args({1024, 512, 256})
    //->Args({1024, 512, 512})
    ->Unit(benchmark::kMillisecond);

static void recut_radius(benchmark::State &state) {
  std::vector<int> tcases = {5};
  int slt_pct = 100;
  bool print_all = false;
  uint16_t bkg_thresh = 0;
  auto grid_size = state.range(0);

  for (auto &tcase : tcases) {
    // the total number of blocks allows more parallelism
    // ideally intervals >> thread count
    auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase, true);

    // run
    auto recut = Recut<uint16_t>(args);
    auto root_vids = recut.initialize();

    for (auto _ : state) {
      // warning: pause and resume high overhead
      state.PauseTiming();
      recut.activate_vids(root_vids, "value", recut.global_fifo);
      recut.update("value", recut.global_fifo);
      recut.setup_radius(recut.global_fifo);
      state.ResumeTiming();

      recut.update("radius", recut.global_fifo);
      recut.release();
    }
  }
}
BENCHMARK(recut_radius)
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
    auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase, true);

    // run
    auto recut = Recut<uint16_t>(args);
    recut.initialize();
    VID_t interval_num = 0;

    for (auto _ : state) {
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
    auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase, true);

    // run
    auto recut = Recut<uint16_t>(args);
    recut.initialize();
    VID_t interval_num = 0;

    for (auto _ : state) {
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

#ifdef USE_MCP3D

static void load_exact_tile(benchmark::State &state) {
  auto tcase = 0;
  int slt_pct = 100;
  int grid_size = state.range(0);
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase, false);
  VID_t tol_sz = (VID_t)grid_size * grid_size * grid_size;

  for (auto _ : state) {
    mcp3d::MImage image;
    cout << args.image_root_dir() << '\n';
    read_tiff(args.image_root_dir(), {0, 0, 0},
              {grid_size, grid_size, grid_size}, image);
  }
  auto total_pixels = static_cast<VID_t>(grid_size) * grid_size * grid_size;
  state.SetBytesProcessed(state.iterations() * total_pixels * sizeof(uint16_t));
  state.SetItemsProcessed(state.iterations() * total_pixels);
}
BENCHMARK(load_exact_tile)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

static void load_tile_from_large_image(benchmark::State &state) {
  auto tcase = 6;
  int slt_pct = 100;
  int grid_size = state.range(0);
  int mid_pixel_in_total_xy_image = 4096;
  // note: z dim doesn't usually extend past 256 depth
  auto args = get_args(grid_size, grid_size, grid_size, slt_pct, tcase, false);
  VID_t tol_sz = (VID_t)grid_size * grid_size * grid_size;

  for (auto _ : state) {
    mcp3d::MImage image;
    cout << args.image_root_dir() << '\n';
    read_tiff(args.image_root_dir(),
              {mid_pixel_in_total_xy_image, mid_pixel_in_total_xy_image, 0},
              {grid_size, grid_size, grid_size}, image);
  }
  auto total_pixels = static_cast<VID_t>(grid_size) * grid_size * grid_size;
  state.SetBytesProcessed(state.iterations() * total_pixels * sizeof(uint16_t));
  state.SetItemsProcessed(state.iterations() * total_pixels);
}
BENCHMARK(load_tile_from_large_image)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

#endif

BENCHMARK_MAIN();
