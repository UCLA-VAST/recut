enable_testing()
# note we recommend at least 3.9 because of
# cpp 17 features that don't work
# on older version during linking
# we strongly suggest upgrading to this version
# or higher to build this code from source
cmake_minimum_required(VERSION 3.8)

project(recut)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# works for cmake 3.9+
# https://stackoverflow.com/questions/48825416/missing-openmp-c-flags-openmp-c-lib-names
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
  set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
  set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
  set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
  set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
  set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(STATUS "Using Clang compiler...")
  set(OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
  set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
  set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
  set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
  set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
  set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
endif()
find_package(OpenMP REQUIRED)

# CMake Preprocessor Defines:
# determine which targets to build
# a user of Recut in header only mode can ignore
# building of tests and benchmarks
# although CreateIntervalBase will need to be run at least once
# from the recut_test.hpp file
set(RECUT_ENABLE_CL_EXECUTABLE OFF CACHE BOOL "Enable Recut building its command line executable")
set(RECUT_ENABLE_TESTING ON CACHE BOOL "Enable Recut building its test targets")
set(RECUT_ENABLE_BENCHMARKING OFF CACHE BOOL "Enable Recut building its benchmarks targets")

# NIX-BUILD overrides some dependencies
option( FROM_NIX_BUILD OFF)
if (FROM_NIX_BUILD)
  message(STATUS " FROM_NIX_BUILD ON")
endif()

# nix-build handles gtest and gbenchmark libraries independently from submodules
# they are automatically added to the include path
if (RECUT_ENABLE_TESTING AND NOT FROM_NIX_BUILD)
  # build tests (targets: gtest, gtest_main)
  add_subdirectory(extern/googletest/googletest)
endif()

if (RECUT_ENABLE_BENCHMARKING AND NOT FROM_NIX_BUILD)
  # build google benchmark (target: benchmark)
  # do not build tests of benchmarking lib
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Don't install benchmark" FORCE)
  add_subdirectory(extern/benchmark)
endif()

#  this creates tests that match the benchmarks
#  to check their validity, warning this invokes
#  a long form of all tests meaning execution will
#  take a long time to complete
#  during normal development this should be off
#  and all tests should run in under 2s
# ignored if RECUT_ENABLE_TESTING is false
option(TEST_ALL_BENCHMARKS
  "Conduct full run of tests which can be used for results or benchmarks"
  OFF)

set( GIT_HASH OFF CACHE STRING "Hash of the latest git commit")
#  Scheduling
option( NO_INTERVAL_RV OFF)
option( SCHEDULE_INTERVAL_RV OFF)
option( USE_HUGE_PAGE OFF)
# Note you can't have both OMP_BLOCK and OMP_INTERVAL defined
option( USE_OMP_INTERVAL OFF)
option( USE_OMP_BLOCK OFF)
option( CONCURRENT_MAP "Define usage of the optimized map data structure" OFF)
# Define how revisits/reupdates to previous seen vertices is handled
option( RV "Count the number of revisits or attempted revisits of vertices" OFF)
option(NO_RV "Reject any vertices from having new updated values after they have already been visited" OFF)
# Define your logging level in order of increasing additive levels of specificity
# TODO this should by a command line option with -v, -vv, -vvv
option(LOG "Overview logging details of the recut run, this suffices for basic timing info, granularity at interval level" ON)
option(LOG_FULL "roughly block by block processing granularity" OFF)
option(FULL_PRINT "Print vertex by vertex behavior. Warning: will significantly slow performance" OFF)

option( USE_MMAP "Determines read speed of vertex info from INTERVAL_BASE" OFF)
option( USE_MCP3D "Use and link external library MCP3D into recut for image reading of TIF and HDF5 (.ims) file types" OFF)

set(MARKERS ${PROJECT_SOURCE_DIR}/src/markers.cpp)
set(PARAMS ${PROJECT_SOURCE_DIR}/src/recut_parameters.cpp)
set(APP2 ${PROJECT_SOURCE_DIR}/test/app2_helpers.hpp)

# These define the executable and command line processing
# if a recut command line tool is needed
set(SRC
  ${PROJECT_SOURCE_DIR}/src/recut_main.cpp
  ${PARAMS}
  ${MARKERS}
  )

# These define the core functionality of the library
# and can be used as header only
set(HEADER
  ${PROJECT_SOURCE_DIR}/src/recut.hpp
  ${PROJECT_SOURCE_DIR}/src/recut_parameters.hpp
  ${PROJECT_SOURCE_DIR}/src/grid.hpp
  ${PROJECT_SOURCE_DIR}/src/interval.hpp
  ${PROJECT_SOURCE_DIR}/src/vertex_attr.hpp
  ${PROJECT_SOURCE_DIR}/src/utils.hpp
  ${PROJECT_SOURCE_DIR}/src/config.hpp
  ${PROJECT_SOURCE_DIR}/src/tile_thresholds.hpp
  ${PROJECT_SOURCE_DIR}/src/markers.h # TODO remove this
  )

# must be defined before the executable or library
# is defined, otherwise it is ignored
# adding ../lib makes any executables truly relocatable
# without it if they are make install, RPATH
# will not link shared dependencies properly
#set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

if (RECUT_ENABLE_CL_EXECUTABLE)
  add_executable(recut ${SRC} ${HEADER} ${PARAMS})
endif()
if (RECUT_ENABLE_TESTING)
  add_executable(recut_test test/recut_test.cpp ${HEADER} ${MARKERS} ${PARAMS} ${APP2})
endif()
if (RECUT_ENABLE_BENCHMARKING)
  add_executable(recut_bench bench/recut_bench.cpp ${HEADER} ${MARKERS} ${PARAMS} ${APP2})
endif()

include_directories(include)
include_directories(src)
include_directories(/nix/store/1z56r6kx097b9j07r26kwhngdwjr03k1-openmp-7.1.0/include)

# variable and target dependent links
if (RECUT_ENABLE_TESTING )
  target_link_libraries(recut_test PRIVATE gtest)
endif()
if (RECUT_ENABLE_BENCHMARKING AND NOT FROM_NIX_BUILD)
  target_link_libraries(recut_bench PRIVATE benchmark)
endif()

# determine which targets to build
# a user of Recut in header only mode can ignore
# building of tests and benchmarks
set(ALL_TARGETS "")
if (RECUT_ENABLE_CL_EXECUTABLE)
  list(APPEND ALL_TARGETS recut)
endif()
if (RECUT_ENABLE_TESTING)
  list(APPEND ALL_TARGETS recut_test)
endif()
if (RECUT_ENABLE_BENCHMARKING)
  list(APPEND ALL_TARGETS recut_bench)
endif()

foreach(I ${ALL_TARGETS})
  message(STATUS "${I}")

  # Cmake accepts user preprocessor #define's on invocation
  # these are passed on to the preprocessor below
  if (LOG)
    target_compile_definitions(${I} PRIVATE LOG)
  endif()
  if (LOG_FULL)
    target_compile_definitions(${I} PRIVATE LOG_FULL)
  endif()
  if (FULL_PRINT)
    target_compile_definitions(${I} PRIVATE FULL_PRINT)
  endif()
  if (RV)
    target_compile_definitions(${I} PRIVATE RV)
  endif()
  if (NO_RV)
    target_compile_definitions(${I} PRIVATE NO_RV)
  endif()
  if (USE_OMP_BLOCK)
    target_compile_definitions(${I} PRIVATE USE_OMP_BLOCK)
  endif()
  if (USE_OMP_INTERVAL)
    target_compile_definitions(${I} PRIVATE USE_OMP_INTERVAL)
  endif()
  if (CONCURRENT_MAP)
    target_compile_definitions(${I} PRIVATE CONCURRENT_MAP)
  endif()
  if (USE_MMAP)
    target_compile_definitions(${I} PRIVATE USE_MMAP)
  endif()
  if (USE_MCP3D)
    target_compile_definitions(${I} PRIVATE USE_MCP3D)
  endif()
  if (NO_INTERVAL_RV)
    target_compile_definitions(${I} PRIVATE NO_INTERVAL_RV)
  endif()
  if (SCHEDULE_INTERVAL_RV)
    target_compile_definitions(${I} PRIVATE SCHEDULE_INTERVAL_RV)
  endif()
  if (USE_HUGE_PAGE)
    target_compile_definitions(${I} PRIVATE USE_HUGE_PAGE)
  endif()
  if (GIT_HASH)
    target_compile_definitions(${I} PRIVATE GIT_HASH)
  endif()

  target_compile_features(${I} PRIVATE cxx_std_17)
  # these are the actual flags passed during compilation
  target_compile_options(${I} PRIVATE -fno-omit-frame-pointer -g)
  install(TARGETS ${I} DESTINATION ${PROJECT_SOURCE_DIR}/bin)

  # other dependencies
  if (NOT FROM_NIX_BUILD) 
    target_include_directories(${I} PRIVATE extern/range-v3/include)
  endif()

  if (USE_OMP_INTERVAL OR USE_OMP_BLOCK)
    target_link_libraries(${I} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if (USE_MCP3D)
    target_include_directories(${I} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${I} PRIVATE
      #${PROJECT_SOURCE_DIR}/lib/libboost_filesystem.so.1.65.1
      ${PROJECT_SOURCE_DIR}/lib/libboost_serialization.so.1.65.1
      ${PROJECT_SOURCE_DIR}/lib/libboost_system.so.1.65.1
      ${PROJECT_SOURCE_DIR}/lib/libhdf5.so.100.1.0
      ${PROJECT_SOURCE_DIR}/lib/libhdf5.so.1.10.1
      ${PROJECT_SOURCE_DIR}/lib/libmcp3d_algorithm.so
      ${PROJECT_SOURCE_DIR}/lib/libmcp3d_common.so
      ${PROJECT_SOURCE_DIR}/lib/libmcp3d_image.so
      ${PROJECT_SOURCE_DIR}/lib/libmcp3d_project_structure.so
      ${PROJECT_SOURCE_DIR}/lib/libopencv_imgcodecs.so
      ${PROJECT_SOURCE_DIR}/lib/libopencv_core.so
      )
  endif()

  # make Recut libraries truly relocatable when linking from downstream programs
  set_target_properties(${I} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endforeach()

if (USE_MCP3D)
  message(STATUS "USE_MCP3D ON")
endif()

# test specific
if (RECUT_ENABLE_TESTING)
  if (TEST_ALL_BENCHMARKS)
    message(STATUS " TEST_ALL_BENCHMARKS ON")
    target_compile_definitions(recut_test PRIVATE TEST_ALL_BENCHMARKS)
  endif()

  add_test(NAME recut_test COMMAND ${PROJECT_SOURCE_DIR}/bin/recut_test)
endif()
