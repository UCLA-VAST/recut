# adds the following inputs to the executable
target_sources(recut PRIVATE basic_4dimage.cpp basic_4dimage.h basic_memory.cpp basic_memory.h mg_image_lib.cpp mg_image_lib.h mg_utilities.cpp mg_utilities.h stackutil.cpp stackutil.h v3d_basicdatatype.h)

target_include_directories(recut PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
