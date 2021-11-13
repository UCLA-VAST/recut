enable_testing()
# at least 3.9 for full cpp 17 features
# 3.14 for FetchContent
cmake_minimum_required(VERSION 3.14)

project("Recut" VERSION 0.9.0
  DESCRIPTION "A c++ library for parallel graph kernels"
  HOMEPAGE_URL "https://github.com/UCLA-VAST/recut-pipeline")

# if not user defined, set to sane default for unix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "..." FORCE)
endif()

message(STATUS "NIX_CFLAGS_COMPILE : $ENV{NIX_CFLAGS_COMPILE}")
message(STATUS "NIX_LDFLAGS : $ENV{NIX_LDFLAGS}")

if (DEFINED CMAKE_EXPORT_NO_PACKAGE_REGISTRY)
  message(STATUS "CMAKE_EXPORT_NO_PACKAGE_REGISTRY : ${CMAKE_EXPORT_NO_PACKAGE_REGISTRY}")
endif()

# get platforms paths for general, non-nix installs
include(GNUInstallDirs)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_INSTALL_DATADIR ${CMAKE_INSTALL_PREFIX}/data)
message(STATUS "CMAKE_INSTALL_DATADIR: ${CMAKE_INSTALL_DATADIR}")

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

# CMake Preprocessor Defines:
# determine which targets to build
# a user of Recut in header only mode can ignore
# building of tests and benchmarks
# although CreateIntervalBase will need to be run at least once
# from the recut_test.hpp file
set(RECUT_ENABLE_EXECUTABLE ON CACHE BOOL "Enable Recut building its command line executable")
set(RECUT_ENABLE_TESTING ON CACHE BOOL "Enable Recut building its test targets")
set(RECUT_ENABLE_BENCHMARKING OFF CACHE BOOL "Enable Recut building its benchmarks targets")

# forbid heavy config steps regardless of setting
# FIXME still ignored by ranges library
set(RANGE_V3_DOCS OFF CACHE BOOL "off" FORCE)
set(RANGE_V3_TESTS OFF CACHE BOOL "off" FORCE)
set(RANGE_V3_EXAMPLES OFF CACHE BOOL "off" FORCE)

# if doing a plain system install (not from nix or a package manager)
if (NOT CMAKE_EXPORT_NO_PACKAGE_REGISTRY)
  # nix-build handles gtest and gbenchmark libraries independently from submodules
  # they are automatically added to the include path
  option(FETCHCONTENT_UPDATES_DISCONNECTED "skips only update step" ON)
  include(FetchContent)

  FetchContent_Declare(
    Range-v3
    GIT_REPOSITORY "https://github.com/ericniebler/range-v3"
    GIT_TAG master
    )
  FetchContent_MakeAvailable(Range-v3)

  if (RECUT_ENABLE_TESTING)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.10.0
      )
    FetchContent_MakeAvailable(googletest)
  endif()

  if (RECUT_ENABLE_BENCHMARKING)
    # build google benchmark (target: benchmark)
    # do not build tests of benchmarking lib
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Don't install benchmark" FORCE)

    FetchContent_Declare(googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG master) # need master for benchmark::benchmark
    FetchContent_MakeAvailable(googlebenchmark)
  endif()

  if (USE_VDB)
    # Dependencies of OpenVDB TODO FIXME use nix for now
    FetchContent_Declare(
      Blosc
      GIT_REPOSITORY "https://github.com/Blosc/c-blosc.git"
      GIT_TAG master
      )
    FetchContent_MakeAvailable(Blosc)

    # TODO FIXME non-nix / plain cmake install will have to install all dependencies to OpenVDB
    #find_package(Blosc REQUIRED)
    #find_package(TBB REQUIRED)
    # etc...

    FetchContent_Declare(
      OpenVDB
      GIT_REPOSITORY "https://github.com/AcademySoftwareFoundation/openvdb.git"
      GIT_TAG master
      )
    FetchContent_MakeAvailable(OpenVDB)

  endif()

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
# Note you can't have both OMP_BLOCK and OMP_INTERVAL defined
option( USE_OMP_INTERVAL OFF)
option( USE_OMP_BLOCK OFF)
option( USE_VDB OFF)
# Define how revisits/reupdates to previous seen vertices is handled
option( RV "Count the number of revisits or attempted revisits of vertices" OFF)
option(NO_RV "Reject any vertices from having new updated values after they have already been visited" OFF)
# Define your logging level in order of increasing additive levels of specificity
# TODO this should by a command line option with -v, -vv, -vvv
option(LOG "Overview logging details of the recut run, this suffices for basic timing info, granularity at interval level" ON)
option(LOG_FULL "roughly block by block processing granularity" OFF)
option(FULL_PRINT "Print vertex by vertex behavior. Warning: will significantly slow performance" OFF)

option( USE_MCP3D "Use and link external library MCP3D into recut for image reading of TIF and HDF5 (.ims) file types" OFF)


set(MARKERS ${PROJECT_SOURCE_DIR}/src/markers.cpp)
set(PARAMS ${PROJECT_SOURCE_DIR}/src/recut_parameters.cpp)
set(APP2 ${PROJECT_SOURCE_DIR}/src/app2_helpers.hpp)

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
  ${PROJECT_SOURCE_DIR}/src/vertex_attr.hpp
  ${PROJECT_SOURCE_DIR}/src/utils.hpp
  ${PROJECT_SOURCE_DIR}/src/config.hpp
  ${PROJECT_SOURCE_DIR}/src/tile_thresholds.hpp
  ${PROJECT_SOURCE_DIR}/src/markers.h # TODO remove this
  )

# Uncomment if changing recut to export as a library
# must be defined before the executable or library
# is defined, otherwise it is ignored
# adding ../lib makes any executables truly relocatable
# without it if `make install`: then RPATH
# will not link shared dependencies properly
#set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

if (RECUT_ENABLE_EXECUTABLE)
  add_executable(recut ${SRC} ${HEADER} ${PARAMS} ${APP2})
endif()
if (RECUT_ENABLE_TESTING)
  add_executable(recut_test src/recut_test.cpp ${HEADER} ${MARKERS} ${PARAMS} ${APP2})
endif()
if (RECUT_ENABLE_BENCHMARKING)
  add_executable(recut_bench bench/recut_bench.cpp ${HEADER} ${MARKERS} ${PARAMS} ${APP2})
endif()

# all recut headers and source go in ./src/
include_directories(src)

# variable and target dependent links
if (RECUT_ENABLE_TESTING )
  target_link_libraries(recut_test PRIVATE gtest)
endif()
if (RECUT_ENABLE_BENCHMARKING)
  target_link_libraries(recut_bench PRIVATE benchmark::benchmark)
endif()

# determine which targets to build
# a user of Recut in header only mode can ignore
# building of tests and benchmarks
set(ALL_TARGETS "")
if (RECUT_ENABLE_EXECUTABLE)
  list(APPEND ALL_TARGETS recut)
endif()
if (RECUT_ENABLE_TESTING)
  list(APPEND ALL_TARGETS recut_test)
endif()
if (RECUT_ENABLE_BENCHMARKING)
  list(APPEND ALL_TARGETS recut_bench)
endif()

# required dependencies
find_package(range-v3)
find_package(OpenMP REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (TINYTIFF_PATH)
  set(TinyTIFFShared_DIR TINYTIFF_PATH)
  set(TinyTIFF_DIR TINYTIFF_PATH)
  find_package(TinyTIFF REQUIRED)
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
  if (USE_MCP3D)
    target_compile_definitions(${I} PRIVATE USE_MCP3D)
  endif()
  if (TINYTIFF_PATH)
    target_compile_definitions(${I} PRIVATE USE_TINYTIFF)
  endif()
  if (USE_VDB)
    target_compile_definitions(${I} PRIVATE USE_VDB)
  endif()
  if (NO_INTERVAL_RV)
    target_compile_definitions(${I} PRIVATE NO_INTERVAL_RV)
  endif()
  if (SCHEDULE_INTERVAL_RV)
    target_compile_definitions(${I} PRIVATE SCHEDULE_INTERVAL_RV)
  endif()
  if (GIT_HASH)
    target_compile_definitions(${I} PRIVATE GIT_HASH)
  endif()
  target_compile_definitions(${I} PRIVATE "CMAKE_INSTALL_DATADIR=\"${CMAKE_INSTALL_DATADIR}\"")

  # c++20 has compile errors with range-v3 library
  target_compile_features(${I} PRIVATE cxx_std_17)
  # these are the actual flags passed during compilation
  target_compile_options(${I} PRIVATE -fno-omit-frame-pointer -g)
  install(TARGETS ${I} DESTINATION ${CMAKE_INSTALL_BINDIR})

  target_link_libraries(${I} PRIVATE range-v3::range-v3)

  if (USE_OMP_INTERVAL OR USE_OMP_BLOCK)
    target_link_libraries(${I} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if (TEST_ALL_BENCHMARKS)
    target_compile_definitions(${I} PRIVATE TEST_ALL_BENCHMARKS)
  endif()

  target_link_libraries(${I} PRIVATE Threads::Threads)

  if (TINYTIFF_PATH)
    target_link_libraries(${I} PRIVATE TinyTIFF)
  endif()

  if (USE_MCP3D)
    message(STATUS "USE_MCP3D ON")
    find_package(Eigen3 REQUIRED)
    target_include_directories(${I} PRIVATE ${EIGEN3_INCLUDE_DIR})

    find_package(OpenCV REQUIRED core imgcodecs imgproc)
    target_include_directories(${I} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${I} PRIVATE ${OpenCV_LIBS})

    target_link_libraries(${I} PRIVATE
      libmcp3d_algorithm.so
      libmcp3d_image.so
      libmcp3d_common.so
      libmcp3d_project_structure.so
      )
  endif()

  if (USE_VDB)
    message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
    find_package(OpenVDB REQUIRED)
    target_link_libraries(${I} PRIVATE OpenVDB::openvdb)
    message(STATUS "Linked OpenVDB")
  endif()

  # make Recut libraries truly relocatable when linking from downstream programs
  set_target_properties(${I} PROPERTIES INSTALL_RPATH "$ORIGIN/../bin")
endforeach()

if (RECUT_ENABLE_TESTING)
  # You have to run installcheck such that the relevant files used at runtime
  # are in the data directory before running any other tests
  add_custom_target(installcheck COMMAND recut_test --gtest_also_run_disabled_tests --gtest_filter=Install.*)
  add_test(NAME TestAll COMMAND recut_test)
endif()
