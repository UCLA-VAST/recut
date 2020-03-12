### Installation
```
git clone --recursive https://github.com/UCLA-VAST/recut-pipeline.git
cd recut
mkdir build bin
cd build
cmake ..
make 
make install
```
Note for performance recut relies on a pregenerated file
in tmp at runtime. Also for testing purposes a set of 
sample `test_images/` and `test_markers/` should be pregenerated
before running any tests. All of these files above can be 
generated via:
```
cd recut/bin
./recut_tests --gtest_also_run_disabled --gtest_filter=INSTALL.*

### Dependencies
This program relies on: 
- the `mcp3d::image` library (included) 
- the google-test and google-benchmark library (included via git
submodules and auto builds through cmake, see `recut/CMakeLists.txt`)
- currently the boost development environmet `sudo apt-get install boost-devel`

### Usage
```
cd recut/bin
./recut --help
```
