#include <iostream>
#include <vector>
#include "recut.hpp"

int main(int argc, char * argv[])
{

    RecutCommandLineArgs args;
    // if command line arguments invalid, do not execute further
    if (!ParseRecutArgs(argc, argv, args))
        return 1;

	args.PrintParameters();

    auto recut = Recut<uint16_t>(args);
    recut.initialize();
    recut.update();
    std::vector<MyMarker*> outtree;
    recut.finalize(outtree);

	return 0;
}
