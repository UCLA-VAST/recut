#include "recut.hpp"

int main(int argc, char *argv[]) {

  // throws if run more than once
  openvdb::initialize();
  ImgGrid::registerGrid();

  auto args = ParseRecutArgsOrExit(argc, argv);
  auto recut = Recut<uint16_t>(args);
  recut();

  return 0;
}
