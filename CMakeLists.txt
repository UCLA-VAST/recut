cmake_minimum_required(VERSION 2.8)

project(RecutPipeline)

# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(extern/googletest)
# build tests (targets: gtest_main, gtest)
add_subdirectory(extern/benchmark)

# google benchmark
#target_link_libraries("${PROJECT_NAME}-bench" benchmark)

# googletest
#target_link_libraries("${PROJECT_NAME}-test" gtest)

set(SRC src/recut_main.cpp)
add_executable(recut ${SRC})
install(TARGETS recut DESTINATION bin)
