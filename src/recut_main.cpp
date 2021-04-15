#include "recut.hpp"

int main(int argc, char *argv[]) {

#ifdef USE_VDB
  openvdb::initialize();
  // throws if run more than once
#ifdef CUSTOM_GRID
  EnlargedPointDataGrid::registerGrid();
#endif
#endif

  RecutCommandLineArgs args;
  // if command line arguments invalid, do not execute further
  if (!ParseRecutArgs(argc, argv, args))
    return 1;

  args.PrintParameters();

  auto recut = Recut<uint16_t>(args);
  recut();

  return 0;
}
