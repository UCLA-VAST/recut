### Installation
```
# don't forget --recursive flag otherwise you'll have confusing compilation errors
git clone --recursive https://github.com/UCLA-VAST/recut-pipeline.git
cd recut
mkdir build
cd build
cmake ..
make [-j 8]
make install [-j 8]
```

If you are not on CDSC's n1 host, you will need to generate an
"IntervalBase" file.  For performance reasons, Recut relies on this
pregenerated file, created in `/tmp/`, at runtime. 

Additionally, if image reading capabilities are turned on via the Cmake
USE_MCP3D variable, then for testing purposes a set of sample `test_images/`
and `test_markers/` will be pregenerated before running any other tests. All of
these files above can be generated via:

```
cd ../bin
./recut_test --gtest_filter=Install."*"
# Optionally run the entire test suite
./recut_test
```

If all test passed the installation is successful

### Dependencies
This program relies on: 
- Cmake (version 3.17 or newer)
  brings all necessary c++17 features
- google-test and google-benchmark library submodules (already included via `git
  --recursive ...`, auto built/linked through cmake, see
  `recut/CMakeLists.txt` for details)
- Optionally: `mcp3d::image` an image reading library for Tiff, Imaris file types see below 
- Optionally: python3.8 matplotlib, gdb, clang-tools, linux-perf
- Note: to increase reproducibility and dependencies issues we recommend developing within the Nix package environment (see the Troubleshooting section)

#### Image reading with MCP3D library
If you need image reading and writing capabilities rerun cmake and install
like so:
```
rm -rf build
cmake -B build -D USE_MCP3D=ON
cd build
make 
make install
cd ../bin
# install the test images like so:
./recut_test --gtest_filter=Install."*" 

# you may also want to run the full set of test benchmarks by instead defining
cmake -B build -D USE_MCP3D=ON -D TEST_ALL_BENCHMARKS=ON
```

##### MCP3D dependencies
- Boost development environment:
  e.g. `sudo apt-get install libboost-all-dev`
- openssl
- libtiff
- mpich (optional)

#### Troubleshooting
Some of Recut's dependencies require latest releases then you may have installed on your system, for example cmake.
In these scenarios, or if you're running into compile time issues we recommend running a pinned version of software via
the Nix package manager. You can install Nix on any Linux distribution, MacOS and Windows (via WSL) via the recommended multi-user installation:

`
sh <(curl -L https://nixos.org/nix/install) --daemon
# check installation
nix-shell --version
`

Now if you run:
`
nix-shell
`

from Recut's base directory you should enter the nix-shell where an isolated development environment is downloaded and loaded for you which includes cmake and all other dependencies needed for development.

### Usage
```
cd recut/bin
./recut_test 
```
