cmake_minimum_required(VERSION 2.8)

# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(3rd_party/googletest)
# build tests (targets: gtest_main, gtest)
add_subdirectory(3rd_party/benchmark)

# google benchmark
target_link_libraries("${PROJECT_NAME}-bench" benchmark)

# googletest
target_link_libraries("${PROJECT_NAME}-test" gtest)

project(RecutPipeline)
add_executable(recut recut_main.cpp)
install(TARGETS recut DESTINATION bin)
