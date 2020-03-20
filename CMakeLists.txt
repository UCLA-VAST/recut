enable_testing()
cmake_minimum_required(VERSION 2.8)

project(recut)

# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(extern/googletest)
# build tests (targets: gtest, gtest_main)
add_subdirectory(extern/benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")
#set(CMAKE_CXX_FLAGS "/usr/bin/g++-7")

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

target_link_libraries(recut PUBLIC ${LOCAL_INCLUDE_LIBS})
target_compile_features(recut PRIVATE cxx_std_17)
set_property(TARGET recut PROPERTY CXX_STANDARD 17)
include_directories(include)

install(TARGETS recut DESTINATION bin)
