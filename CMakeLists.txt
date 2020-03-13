cmake_minimum_required(VERSION 2.8)

#set(BOOST_ROOT )
set(Boost_DEBUG 1)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
#set(BOOST_ROOT "/usr/share")
set(BOOST_INCLUDE_PATH "/usr/include/boost")
set(BOOST_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu")
#set(BOOST_LIBRARY_PATH "/usr/share/")
#set(BOOST_INCLUDE_PATH "/usr/include/boost")
#find_package(Boost REQUIRED )
#message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "BOOST_INCLUDE_PATH: ${BOOST_INCLUDE_PATH}")
message(STATUS "BOOST_LIBRARY_PATH: ${BOOST_LIBRARY_PATH}")

# This is needed if your Boost version is newer than your CMake version
# or if you have an old version of CMake (<3.5)
if(NOT TARGET Boost::filesystem)
    add_library(Boost::filesystem IMPORTED INTERFACE)
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
endif()

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

file(GLOB LOCAL_INCLUDE_LIBS "../lib/*.so")
target_link_libraries(recut PUBLIC ${LOCAL_LIBS})

#set(BOOST_LIBRARIES
    #${BOOST_LIBRARY_PATH}/libboost_serialization.so
    #${BOOST_LIBRARY_PATH}/libboost_filesystem.so
    #${BOOST_LIBRARY_PATH}/libboost_system.so)

#target_link_libraries(recut ${BOOST_LIBRARIES})
include_directories(${BOOST_INCLUDE_PATH})

#add_compile_options(-lboost -lboost_filesystem) # boost filesystem
#if(Boost_FOUND)
  #include_directories(${Boost_INCLUDE_DIRS})
  #add_executable(foo foo.cc)
  #target_link_libraries(foo ${Boost_LIBRARIES})
#endif()

include_directories(include)

install(TARGETS recut DESTINATION bin)
