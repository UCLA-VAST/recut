#include <benchmark/benchmark.h>
#include "recut.hpp"
#include "../external_tools/vaa3d/neuron_tracing/all_path_pruning2/heap.h"
#include "fastmarching_tree.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <bits/stdc++.h>
#include <cstdlib> //rand srand
#include <ctime> // for srand
#include <opencv2/opencv.hpp> // imwrite
#include "image/mcp3d_voxel_types.hpp" // convert to CV type
#include "common/mcp3d_utility.hpp" // PadNumStr

//static void BM_multi_interval(benchmark::State& state) {
  //auto grid_size = 32;
  //double slt_pct = 100;
  //int tcase = 0;
  //auto args = get_args(grid_size, slt_pct, tcase);
  //std::vector<int> interval_sizes = {32, 16, 8};
  //VID_t expected = (slt_pct / 100 ) * grid_size * grid_size * grid_size;
  //for (auto& interval_size : interval_sizes) {
    //auto params = args.recut_parameters();
    //params.set_interval_size(interval_size);
    //params.set_block_size(interval_size);
    //// by setting the max intensities you do not need to recompute them
    //params.set_max_intensity(1);
    //params.set_min_intensity(0);
    //args.set_recut_parameters(params);

    //auto recut = Recut<uint16_t>(args);
    //recut.initialize();
    //recut.update();
  //}
  //return;
//}
//BENCHMARK(BM_multi_interval)->RangeMultiplier(1/2)->Range(128, 32)

BENCHMARK_MAIN();
