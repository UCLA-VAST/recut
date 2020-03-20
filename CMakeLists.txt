enable_testing()
cmake_minimum_required(VERSION 2.8)

project(recut)

# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(extern/googletest)
# build tests (targets: gtest, gtest_main)
add_subdirectory(extern/benchmark)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")

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

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#file(GLOB LOCAL_INCLUDE_LIBS "../lib/*.so")
#message(STATUS ${LOCAL_INCLUDE_LIBS})

set_target_properties(recut PROPERTIES
   CXX_STANDARD 17
   CXX_EXTENSIONS OFF 
   )

target_link_libraries(recut PUBLIC ${LOCAL_INCLUDE_LIBS})
target_compile_features(recut PRIVATE cxx_std_17)
# target_compile_options(recut PUBLIC -lboost -lboost_filesystem) # boost filesystem

# set(BOOST_INCLUDE_PATH "/usr/include/boost")
# include_directories(${BOOST_INCLUDE_PATH})
include_directories(include)

install(TARGETS recut DESTINATION bin)
