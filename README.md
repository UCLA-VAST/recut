### Installation
```
# don't forget --recursive flag otherwise you'll have confusing compilation errors
git clone --recursive https://github.com/UCLA-VAST/recut-pipeline.git
cd recut
mkdir bin
cmake -B build -DCMAKE_BUILD_TYPE=[Debug/Release]
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
./recut_test --gtest_also_run_disabled_tests --gtest_filter=Install.*
```

### Dependencies
This program relies on: 
- currently the boost development environment:
  sudo apt-get install libboost-all-dev
- the google-test and google-benchmark library (included via git
submodules and auto builds through cmake, see `recut/CMakeLists.txt`)
- Optionally: `mcp3d::image` library (included) 

### Usage
```
cd recut/bin
./recut --help
```
