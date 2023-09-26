# adds the following input sources to the executables post-hoc
set(V3D_IMAGE_IO basic_4dimage.cpp basic_4dimage.h basic_memory.cpp basic_memory.h mg_image_lib.cpp mg_image_lib.h mg_utilities.cpp mg_utilities.h stackutil.cpp stackutil.h v3d_basicdatatype.h)

if (RECUT_ENABLE_EXECUTABLE)
  target_sources(recut PRIVATE ${V3D_IMAGE_IO})
  target_include_directories(recut PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
endif()

if (RECUT_ENABLE_TESTING)
  target_sources(recut_test PRIVATE ${V3D_IMAGE_IO})
  target_include_directories(recut_test PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
endif()
