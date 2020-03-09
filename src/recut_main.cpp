#include <iostream>
#include <vector>
#include "recut.hpp"

using namespace std;

int main(int argc, char * argv[])
{

    mcp3d::RecutCommandLineArgs args;
    // if command line arguments invalid, do not execute further
    if (!mcp3d::ParseRecutArgs(argc, argv, args))
        return 1;

	args.PrintParameters();

    auto recut = Recut<uint16_t>(args);
    recut.initialize();
    recut.update();
    vector<MyMarker*> outtree;
    recut.finalize(outtree);

	return 0;
}
