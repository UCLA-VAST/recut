enable_testing()
# note this is set to 3.9 because of
# cpp 17 features that don't work
# on older version during linking
# we strongly suggest upgrading to this version
# or higher to build this code from source
cmake_minimum_required(VERSION 3.9)

project(recut)

# build tests (targets: gtest, gtest_main)
add_subdirectory(extern/googletest/googletest)
# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Don't install benchmark" FORCE)
add_subdirectory(extern/benchmark)

# works for cmake 3.9+
find_package(OpenMP REQUIRED)

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

#add_executable(recut ${SRC} ${HEADER} ${PARAMS})
add_executable(recut_test test/recut_test.cpp ${HEADER} ${MARKERS} ${PARAMS} ${APP2})
add_executable(recut_bench bench/recut_bench.cpp ${HEADER} ${MARKERS} ${PARAMS} ${APP2})

include_directories(include)

# Turf dependencies Add Junction (and Turf) build targets to the CMake project.
# This defines JUNCTION_ALL_INCLUDE_DIRS and JUNCTION_ALL_LIBRARIES.
add_subdirectory("extern/junction" junction)
include_directories(${JUNCTION_ALL_INCLUDE_DIRS})

target_link_libraries(recut_test PRIVATE gtest)
target_link_libraries(recut_bench PRIVATE benchmark)

#set(ALL_TARGETS recut recut_test recut_bench)
set(ALL_TARGETS recut_test recut_bench)
foreach(I ${ALL_TARGETS})
  message(STATUS "${I}")
  target_link_libraries(${I} PRIVATE ${JUNCTION_ALL_LIBRARIES})
  target_link_libraries(${I} PRIVATE OpenMP::OpenMP_CXX)
  target_compile_features(${I} PRIVATE cxx_std_17)
  #target_compile_definitions(${I} PUBLIC USE_HUGE_PAGE=${USE_HUGE_PAGE})
  # these are the actual flags passed during compilation
  target_compile_options(${I} PRIVATE -fno-omit-frame-pointer -g)
  install(TARGETS ${I} DESTINATION ${PROJECT_SOURCE_DIR}/bin)
  target_include_directories(${I} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${I} PRIVATE
    ${PROJECT_SOURCE_DIR}/lib/libboost_filesystem.so.1.65.1
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
  set_target_properties(${I} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endforeach()

add_test(NAME recut_test COMMAND ${PROJECT_SOURCE_DIR}/bin/recut_test)
