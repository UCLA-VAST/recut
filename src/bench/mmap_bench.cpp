#include <benchmark/benchmark.h>
#include "tbytesum1.hpp"

//#define TEST_INTERVAL "/curr/kdmarrett/mcp3d/bin/interval_base.bin"
#define TEST_INTERVAL "interval_base.bin"
#define RANGE_MULT 2<<9
//#define RANGE_MULT 1

//static void BM_fread(benchmark::State& state) {
  //for (auto _ : state) 
    //freadtst("fread", TEST_INTERVAL);
//}
//BENCHMARK(BM_fread);

//static void BM_ifstream(benchmark::State& state) {
  //for (auto _ : state) 
    //ifstreamtst("ifstream", TEST_INTERVAL, state.range(0));
//}
//BENCHMARK(BM_ifstream)->Range(2<<12, 2<<20);

//static void BM_read(benchmark::State& state) {
  //for (auto _ : state) 
    //readtst("read", TEST_INTERVAL);
//}
//BENCHMARK(BM_read);

static void BM_mmap(benchmark::State& state) {
  uint64_t sum;
  uint64_t actual_bytes = ((uint64_t) state.range(0)) * RANGE_MULT;
  for (auto _ : state) 
    sum = mmaptst("mmap", TEST_INTERVAL, actual_bytes);
  //state.counters("sum") = sum;
  //state.counters("bytes") = actual_bytes;
}
//BENCHMARK(BM_mmap)->Arg(2<<11)->Arg(2<<23);
BENCHMARK(BM_mmap)->RangeMultiplier(2)->Range(2<<20, 2<<25);
//BENCHMARK(BM_mmap)->Range(2<<11, 2<<29);

static void BM_pread(benchmark::State& state) {
  uint64_t sum;
  uint64_t actual_bytes = ((uint64_t) state.range(0)) * RANGE_MULT;
  for (auto _ : state) 
    sum = preadtst("pread", TEST_INTERVAL, actual_bytes);
  //state.counters("sum") = sum;
  //state.counters("bytes") = actual_bytes;
}
BENCHMARK(BM_pread)->Arg(2<<11)->Arg(2<<23);
//BENCHMARK(BM_pread)->RangeMultiplier(2)->Range(2<<20, 2<<25);
//BENCHMARK(BM_pread)->Range(2<<11, 2<<29);

BENCHMARK_MAIN();
