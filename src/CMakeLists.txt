cmake_minimum_required(VERSION 2.8)
project(RecutPipeline)
add_executable(recut recut_main.cpp)
install(TARGETS recut DESTINATION bin)
