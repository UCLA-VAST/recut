[![built with
nix](https://builtwithnix.org/badge.svg)](https://builtwithnix.org)
![Nix](https://github.com/UCLA-VAST/recut-pipeline/workflows/build/badge.svg)

## Quick Install (strongly recommended)
On linux command line run:
```
# install nix package manager, <10 minutes
curl -L https://nixos.org/nix/install | sh
git clone https://github.com/UCLA-VAST/recut-pipeline.git
cd recut-pipeline
# build and install recut and its dependencies 
# takes up to 2 hours, no other input required
nix-env -f . -i
```
test your installation by running:
```
recut
```

## CMake Only Installation
The following are the commands are required for a CMake and git based installation
```
git clone git@github.com:UCLA-VAST/recut-pipeline
cd recut-pipeline
mkdir build
cd build
cmake ..
make [-j 8]
sudo make install [-j 8]
# required to generate test and interval data
sudo make installcheck
# Optionally, run all tests by running the test suite
make test
```

If all test passed the installation is successful. If you have *any* errors in the above steps see the Troubleshooting section below. We recommend a tested version of cmake and all other dependencies, as opposed to installing a version on your system by hand, as also explained in the Troubleshooting section.


### Dependencies
This program relies on: 
- Cmake (version 3.17 or newer)
  for proper gcc support of all necessary c++17 features
- Optionally: google-test and google-benchmark library submodules ( auto built/linked through cmake, see
  `recut/CMakeLists.txt` for details)
- Optionally: `mcp3d` an image reading library for Tiff, Imaris/HDF5 file types see below 
- Optionally: python3.8 matplotlib, gdb, clang-tools, linux-perf
- Note: to increase reproducibility and dependencies issues we recommend developing within the Nix package environment (see the Troubleshooting section)

#### Image reading with MCP3D library
If you need image reading and writing capabilities rerun cmake with the `-D USE_MCP3D=ON`
flag like so
```
rm -rf build
cmake -B build -D USE_MCP3D=ON
# you may also want to run the full set of test benchmarks by instead defining
cmake -B build -D USE_MCP3D=ON -D TEST_ALL_BENCHMARKS=ON
cd build
make 
sudo make install
# install the test images like so:
sudo make installcheck
# run all tests
./recut_test 
```

#### Troubleshooting
Some of Recut's dependencies require later releases then you may have
installed on your system, for example CMake. In these scenarios, or if you're
running into compile time issues we recommend running a pinned version of
all software via the Nix package manager. To our knowledge, Nix is the state of the art 
in terms of software reproducibility, package and dependency management, and solving
versioning issues in multi-language repositories.

You can install Nix on any Linux
distribution, MacOS and Windows (via WSL) with:

```
# just for your user 
curl -L https://nixos.org/nix/install | sh
# if you need to share packages between users on a system via a multi-user installation, instead run
sh <(curl -L https://nixos.org/nix/install) --daemon

# check installation
nix-shell --version
```

The library employs some system features that have not been fully tested in containers / Docker. As such we recommend installing with the Nix package manager on a bear bones linux, MacOS, WSL machine or similar container / VM for now.

Now if you run:
`
nix-shell
`

from Recut's base directory you should enter the nix-shell where an isolated development environment is downloaded and loaded for you which includes cmake and all other dependencies needed for development.

## Usage
Once recut is installed you can see example usage by running:
```
recut
```
If you installed globally via CMake you can run the executables by:
```
recut
# or for tests
recut_test 
```

If you have nix installed (recommended) you can also run the same CI tests with:
`nix-build`

All pushes will run `nix-build` via github-actions, so you should run this anyway locally before pushing
to make sure the CI system won't fail.

## Documentation
This repository began as a fork of the out-of-memory graph processing framework detailed [here](https://vast.cs.ucla.edu/~chiyuze/pub/icde16.pdf)

The execution pattern and partitioning strategy much more strongly resembles this [paper]( https://arxiv.org/abs/1811.00009), however no public implementation for it was provided.
Reading this second paper is a fast way to understand the overall design and execution pattern of Recut.
