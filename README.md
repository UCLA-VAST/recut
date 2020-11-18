[![built with
nix](https://builtwithnix.org/badge.svg)](https://builtwithnix.org)
![Nix](https://github.com/UCLA-VAST/recut-pipeline/workflows/Nix/badge.svg)

## Installation
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

If you have *any* errors in the above steps see the Troubleshooting section below. We recommend a tested version of cmake and all other dependencies, as opposed to installing a version on your system by hand, as also explained in the Troubleshooting section.

```
cd ../bin
./recut_test --gtest_filter=Install."*"
# Check installation by running the test suite
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
Some of Recut's dependencies require later releases then you may have
installed on your system, for example cmake.  In these scenarios, or if you're
running into compile time issues we recommend running a pinned version of
software via the Nix package manager. To our knowledge, Nix is the state of the art 
in terms of software reproducibility, package and dependency management, and solving
versioning issues in multi-language repositories.

You can install Nix on any Linux
distribution, MacOS and Windows (via WSL) with:

```
# just for your user
curl -L https://nixos.org/nix/install | sh
# or for a multi-user installation, instead run
sh <(curl -L https://nixos.org/nix/install) --daemon

# check installation
nix-shell --version
```

The library employs some system features that have not been fully tested in containers / Docker. As such we recommend installing with the Nix package manager on a bear bones linux, MacOS, WSL machine or similar VM for now.

Now if you run:
`
nix-shell
`

from Recut's base directory you should enter the nix-shell where an isolated development environment is downloaded and loaded for you which includes cmake and all other dependencies needed for development.

#### Internal notes
If you are on CDSC's n1 host, you will need to change the name of the generated file specified by
`#define INTERVAL_BASE ...` to something new by changing it in `src/config.hpp`.  For performance reasons, Recut creates this pregenerated file with name defined by `INTERVAL_BASE`, in the `/tmp/` directory in your temporary filesystem. After installation, recut will use this file at runtime. 

Additionally, if image reading capabilities are turned on via the Cmake
USE_MCP3D variable, then for testing purposes a set of sample `test_images/`
will be pregenerated before running any other tests, see the Image reading with MCP3D section for details.

## Usage
```
cd recut/bin
./recut_test 
```

Note the binary file for image reading is currently turned off in CMakeLists by default.

## Documentation
This repository began as a fork of the out-of-memory graph processing framework detailed [here](https://vast.cs.ucla.edu/~chiyuze/pub/icde16.pdf)

The execution pattern and partitioning strategy much more strongly resembles this [paper]( https://arxiv.org/abs/1811.00009), however no public implementation for it was provided.
Reading this second paper is a fast way to understand the overall design and execution pattern of Recut.
