#include <benchmark/benchmark.h>
#include "../src/recut.hpp"
#include <cstdlib> //rand srand
#include <ctime> // for srand

#ifdef IMAGE
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
static void bench_critical_loop(benchmark::State& state) {
  auto grid_size = 256;
  double slt_pct = 5;
  int tcase = 4;
  auto args = get_args(grid_size, slt_pct, tcase, GEN_IMAGE);
  VID_t selected = args.recut_parameters().selected;

  // adjust final runtime parameters
  auto params = args.recut_parameters();
  // the total number of intervals allows more parallelism
  // ideally intervals >> thread count
  params.set_interval_size(state.range(0));
  params.set_block_size(state.range(0));
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
  //state.PauseTiming();
  recut.initialize();
  //state.ResumeTiming();

  // This is the performance loop where gbench
  // will execute until timing stats converge
  while (state.KeepRunning()) {
    // from our gen img of initialize update
    // does fastmarching, updated vertices
    // are mmap'd
    recut.update();

    // to destroy the information for this run
    // so that it doesn't affect the next run
    // the vertices must be unmapped
    // done via `release()`, `reset` reactivates
    // the intervals of the root and readds
    // them to the respective heaps
    recut.reset();
    //benchmark::DoNotOptimize();
  }
  // items processed only refers to the total selected vertices
  // processed, note this is subject to the select percent
  // defined above as .01 which means on 1% of the total vertices
  // will be selected
  state.SetBytesProcessed(state.iterations() * selected * 26);
  state.SetLabel(std::to_string(selected * 26 / 1024) + "kB");
  state.SetItemsProcessed(state.iterations() * selected);
}
BENCHMARK(bench_critical_loop)->RangeMultiplier(2)->Range(16, 256)->ReportAggregatesOnly(true)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
