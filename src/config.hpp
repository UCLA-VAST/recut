// Define all preprocessor macros, templates and types to be used for
// configuration and runtime behavior

// overflows at ~2e9 nodes, ~ 1024 * 1024 * 1024 * 2
// typedef uint32_t VID_t; // overflows after ~2e9 ~= 2 1024^3 tiles
//#define INTERVAL_BASE "/tmp/interval_base_32.bin";

typedef uint64_t VID_t; // for multi-interval runs

// pre-generated array of vertices initialized wth desired default values,
// useful for mmap
#define INTERVAL_BASE                                                          \
  "/tmp/interval_base_64bit.bin" // must match the VID_t bit length type
// equivalent max allows up to interval actual size of 1024, 1024, 1024 with
// block_size 4 including padding (ghost cells) WARNING: if you change this
// number you need to rerun CreateIntervalBase function in recut_tests.cpp to
// save an interval at least that size at /tmp/
const VID_t MAX_INTERVAL_VERTICES = 3700000000;

// equivalent max allows up to interval actual size of 256, 256, 256 with
// block_size 4 including padding (ghost cells) Note this is ~786 MB, for
// VertexAttr of size 24 bytes WARNING: if you change this number you need to
// rerun CreateIntervalBase function in recut_tests.cpp to save an interval at
// least that size at /tmp/
// const VID_t MAX_INTERVAL_VERTICES = 32768000;
//#define INTERVAL_BASE "/mnt/huge/interval_base_64bit.bin" // must match the
// VID_t bit length type

// Define your logging level in order of increasing additive levels of
// specificity
#define LOG // overview logging details of the recut run, this suffices
// for
// basic timing info, granularity at interval level
#define LOG_FULL // roughly block by block processing granularity
// #define FULL_PRINT // vertex by vertex behavior
//#define HLOG_FULL // log the behavior of the block heap methods

// Define how revisits/reupdates to previous seen vertices is handled
//#define RV // count the number of revisits or attempted revisits of vertices
// and log to stdout
//#define NO_RV // reject any vertices from having new
// updated values after they have already been visited

// determines read speeds of vertex info from INTERVAL_BASE
#define MMAP
//#define USE_HUGE_PAGE
#define USE_MCP3D

// Note you can't have both OMP_BLOCK and OMP_INTERVAL defined
//#define USE_OMP_INTERVAL
//#define USE_OMP_BLOCK

// Parallel strategies other than OMP defined here
//#define ASYNC // run without TF macro to use the std::async instead of TF
// thread pool, warning much much slower not recommended #define TF // if
// defined, use CPP taskflow to use a workstealing thread pool for new blocks
// TF must be used in conjucntion with ASYNC macro to allow asynchronous starts
// of new blocks during the marching step
#ifdef TF
#define ASYNC
#endif

// Define which optimized data structures to utilize
//#define CONCURRENT_MAP
//#ifdef CONCURRENT_MAP
//#endif

// this creates tests that match the benchmarks
// to check their validity, warning this invokes
// a long form of all tests meaning execution will
// take a long time to complete
// during normal development this should be off
// and all tests should run in under 2s
//#define TEST_ALL_BENCHMARKS

//#define NO_INTERVAL_RV
//#define SCHEDULE_INTERVAL_RV
