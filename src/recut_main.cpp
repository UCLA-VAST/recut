#include "recut.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_VDB
  openvdb::initialize();
  // throws if run more than once
#ifdef CUSTOM_GRID
  EnlargedPointDataGrid::registerGrid();
#endif
#endif

  auto args = ParseRecutArgsOrExit(argc, argv);
  auto recut = Recut<uint16_t>(args);
  recut();

  return 0;
}
