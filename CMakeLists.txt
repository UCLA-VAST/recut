enable_testing()
cmake_minimum_required(VERSION 2.8)

project(recut)

# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(extern/googletest)
# build tests (targets: gtest, gtest_main)
add_subdirectory(extern/benchmark)

set(SRC 
  ${PROJECT_SOURCE_DIR}/src/recut_main.cpp
  ${PROJECT_SOURCE_DIR}/src/recut_parameters.cpp)
set(HEADER
  ${PROJECT_SOURCE_DIR}/src/recut.hpp 
  ${PROJECT_SOURCE_DIR}/src/recut_parameters.hpp 
  ${PROJECT_SOURCE_DIR}/src/super_interval.hpp 
  ${PROJECT_SOURCE_DIR}/src/interval.hpp 
  ${PROJECT_SOURCE_DIR}/src/vertex_attr.hpp 
  ${PROJECT_SOURCE_DIR}/src/utils.hpp
  )
add_executable(recut ${SRC} ${HEADER})

# start edit (added by mrunal)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(recut Eigen3::Eigen)
# end edit

file(GLOB LOCAL_INCLUDE_LIBS "../lib/*.so")
target_link_libraries(recut PUBLIC ${LOCAL_LIBS})
target_compile_features(recut PRIVATE cxx_std_17)

set(BOOST_INCLUDE_PATH "/usr/include/boost")
include_directories(${BOOST_INCLUDE_PATH})
include_directories(include)

install(TARGETS recut DESTINATION bin)
