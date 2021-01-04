// Define preprocessor macros, templates and types to be used for
// configuration and compile time behavior
// Note it is preferred to pass this macros to `cmake ..
// -DMACRO=ON` and add them to the root CMakelists.txt

// uint32_t overflows at ~2e9 nodes, ~ 1024 * 1024 * 1024 * 2
// typedef uint32_t VID_t; // overflows after ~2e9 ~= 2 1024^3 tiles
typedef uint64_t VID_t; // for multi-interval runs

// pre-generated array of vertices initialized wth desired default values,
// useful for mmap
// must match the VID_t bit length type
// on remote servers with mounted filesystems it is 
// recommended to place the interval in /tmp/ to avoid mmap issues
//#define INTERVAL_BASE "/tmp/interval_base_32.bin";
#define INTERVAL_BASE "../data/interval_base_64bit.bin"

// equivalent max allows up to interval actual size shown with
// block_size 4 including padding (ghost cells) WARNING: if you change this
// number you need to rerun CreateIntervalBase function in recut_tests.cpp to
// save an interval at least that size at /tmp/
// MAX_INTERVAL_VERTICES needs to be larger than just the processing region to account
// for ghost regions
// see utils.hpp:get_used_vertex_size()
// or recut_test.cpp:PrintDefaultInfo()
// Vertices needed for a 1024^3 interval block size 4 : 3623878656
// Vertices needed for a 2048^3 interval block size 4 : 28991029248
// Vertices needed for a 8^3 interval block size 2 : 4096
#ifdef TEST_ALL_BENCHMARKS
 const VID_t MAX_INTERVAL_VERTICES =  3623878656; 
#else
  const VID_t MAX_INTERVAL_VERTICES = 4096;
#endif

// equivalent max allows up to interval actual size of 256, 256, 256 with
// block_size 4 including padding (ghost cells) Note this is ~786 MB, for
// VertexAttr of size 24 bytes WARNING: if you change this number you need to
// rerun CreateIntervalBase function in recut_tests.cpp to save an interval at
// least that size at /tmp/
// const VID_t MAX_INTERVAL_VERTICES = 32768000;
//#define INTERVAL_BASE "/mnt/huge/interval_base_64bit.bin" // must match the
// VID_t bit length type

// Parallel strategies other than OMP defined here
//#define ASYNC // run without TF macro to use the std::async instead of TF
// thread pool, warning much much slower not recommended #define TF // if
// defined, use CPP taskflow to use a workstealing thread pool for new blocks
// TF must be used in conjucntion with ASYNC macro to allow asynchronous starts
// of new blocks during the marching step
#ifdef TF
#define ASYNC
#endif
