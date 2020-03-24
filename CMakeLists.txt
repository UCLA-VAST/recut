enable_testing()
# note this is set to 3.9 because of 
# cpp 17 features that we found not to work
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
  ${PROJECT_SOURCE_DIR}/src/super_interval.hpp 
  ${PROJECT_SOURCE_DIR}/src/interval.hpp 
  ${PROJECT_SOURCE_DIR}/src/vertex_attr.hpp 
  ${PROJECT_SOURCE_DIR}/src/utils.hpp
  ${PROJECT_SOURCE_DIR}/src/markers.h
  )

add_executable(recut ${SRC} ${HEADER})
add_executable(recut_test test/recut_test.cpp ${HEADER} ${MARKERS} ${PARAMS})
add_executable(recut_bench bench/recut_bench.cpp ${MARKERS} ${HEADER} ${PARAMS})

include_directories(include)

target_link_libraries(recut_test PUBLIC gtest)
target_link_libraries(recut_bench PUBLIC benchmark)

set(ALL_TARGETS recut recut_test recut_bench)
foreach(I ${ALL_TARGETS})
  message(STATUS "${I}")
  target_link_libraries(${I} PUBLIC OpenMP::OpenMP_CXX)
  target_compile_features(${I} PRIVATE cxx_std_17)
  # these are the actual flags passed during compilation
  target_compile_options(${I} PRIVATE -fno-omit-frame-pointer -g)
  set_target_properties(${I} PROPERTIES COMPILE_FLAGS "-save-temps") 
  install(TARGETS ${I} DESTINATION ${PROJECT_SOURCE_DIR}/bin)
endforeach()

add_test(NAME recut_test COMMAND ${PROJECT_SOURCE_DIR}/bin/recut_test)
