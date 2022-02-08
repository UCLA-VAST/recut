[![built with
nix](https://builtwithnix.org/badge.svg)](https://builtwithnix.org)
![Nix](https://github.com/UCLA-VAST/recut-pipeline/workflows/build/badge.svg)

## Quick Install With Nix (strongly recommended)
Make sure `git` is installed and follow the directions to install the Nix package manager for your OS [here](https://nixos.org/download.html).
Next, on your command line run:
```
git clone git@github.com:UCLA-VAST/recut
cd recut
# build and install recut and its dependencies globally
# takes up to 15 minutes, no other input required
nix-env -f . -i
```
Test your installation by running:
```
recut
```

## Usage
Once recut is installed globally you can see the example usage by running on the command line:
```
recut
```
Recut has several main functions: 
1. Compressing image volumes to a sparse format (VDB grids) 
2. Using VDB grids to reconstruct neurons into a set of SWC trees
3. Creating volumetric image windows centered around points or neurons of interest by reflating from the sparse VDB format.

The first argument passed to recut is the path to the input which can either be a directory for .tiff files or a distinct file in the case of .ims or .vdb files. The ordering of all other arguments is arbitrary. You call recut to process the inputs using an action, for example `--convert` or `--combine`. If no action is described, the default behavior is to do an end-to-end reconstruction from input to reconstructed cell swcs. Where possible, arguments have assumed default values for the most common expected behavior. The following are some use cases.

### Conversion
Convert the folder `ch0` into a VDB point grid:
```
recut ch0 --convert --input-type tiff --output-type point
```
This creates a .vdb file in the ch0 folder. The name is tagged with information about the conversion process for example the argument values used. This is done to preserve info about the source image and to lower the likelihood of overwriting a costly previous conversion with a generic name. Explicit names are helpful for humans but if you want to pass a simpler name you can do so.

Convert the folder again, but this time only take z-planes of 30 through 45 and name it `subset.vdb`
```
recut ch0 --convert subset.vdb --input-type tiff --output-type point --image-offsets 0 0 30 --image-lengths -1 -1 16
```

.vdbs are a binary format, you can only view information or visualize them with softare that explicitly supports them. However for quick info, Recut installs some of the VDB libraries command line tools.

List the exhausitive info and metatdata about the VDB grid:
```
vdb_print -l -m subset.vdb
```
You'll notice that vdb grid's metadata has been stamped with information about the arguments used during conversion to help distinguish files with different characteristics. If you have no other way to match the identity of VDB with an original image, refer to the original bounding extents which can often uniquely identify an image. The original bounding extents of the image used are preserved as the coordinate frame of all points and voxels regardless of any offset or length arguments passed.

#### Image Inference Conversions
The highest quality *reconstructions* currently involve running the MCP3D pipeline's neurite and soma segmentation and connected component stage followed by recut conversion to a point grid followed by recut's reconstruction of that point grid. MCP3D's connected component stage will output the soma locations `marker_files` for use in reconstruction as shown below. MCP3D's segmentation will output a directory of tiff files of the binarized segmented image, with all background set to 0, therefore converting like we did before:
```
recut ch0 --convert point.vdb --input-type tiff --output-type point
```
will automatically use the background threshold of 0 and place foreground everywhere else. This is quite efficient the inference outputs tends to have only .05% of voxels labeled as foreground.

#### Image Raw Conversions
While the process above produces the highest quality *reconstructions* with smaller likelihoods of path breaks, we often want to view the original image or a separate channel of the original image for proofreading and outputting of windows. The highest fidelity way of doing this currently is to convert using a guessed foreground percentage of the image.

```
recut test.ims --convert uint8.vdb --input-type ims --output-type uint8 --channel 0 --fg-percent .05
recut test.ims --convert mask.vdb --input-type ims --output-type mask --channel 1 --fg-percent 10
```
Note that is only necessary if you want to output windows while doing a reconstruction.

### Reconstruct

If you've created a point grid for an image named `point.vdb`, the following would reconstruct the image based off of a directory of files which note the coordinates of starting locations (somas). This directory in the following example is shown as `marker_files`:
```
recut point.vdb --seeds marker_files
```

If you created a corresponding VDB grid of type uint8 for channel 0 and type mask for channel 1 you can also output a window for each swc used by:
```
recut point.vdb --seeds marker_files --output-windows uint8.vdb mask.vdb
```

### Combine
Not finished.

## Developer Usage
If you'd like a virtual environment with all of Recut's dependencies and python provided for you can run:
`
nix-shell
`

from Recut's base directory. This enters an isolated development environment where tools are downloaded and loaded for you, which includes cmake and all other headers/libraries needed for development.

To run the tests:
```
recut_test 
```
If you have nix installed (recommended) you can also run the same CI tests with:
`nix-build`

All pushes will run `nix-build` via github-actions, so you should run this anyway locally before pushing
to make sure the regression tests and CI system won't fail.

## Scientific Motivation
This repository began as a fork of the out-of-memory graph processing framework detailed [here](https://vast.cs.ucla.edu/~chiyuze/pub/icde16.pdf)

The execution pattern and partitioning strategy much more strongly resembles this [paper]( https://arxiv.org/abs/1811.00009), however no public implementation for it was provided.
Reading this second paper is a fast way to understand the overall design and execution pattern of Recut.

## CMake Only Installation (Deprecated)
The following are the commands are required for a CMake and git based installation. If taking this route, the OpenVDB c++ library may need to be installed with CMake first.
```
git clone git@github.com:UCLA-VAST/recut-pipeline
cd recut
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
- Libtiff for reading and writing TIFF images
- OpenVDB for holding sparse data structures in-memory
- Optionally: google-test and google-benchmark library submodules ( auto built/linked through cmake, see
  `recut/CMakeLists.txt` for details)
- Optionally: HDF5 an image reading library for Imaris/HDF5 file types
- Optionally: python3.8 matplotlib, gdb, clang-tools, linux-perf
- Note: to increase reproducibility and dependencies issues we recommend developing within the Nix package environment (see the Troubleshooting section)

#### Troubleshooting
Some of Recut's dependencies require later releases then you may have
installed on your system, for example CMake. In these scenarios, or if you're
running into compile time issues we recommend running a pinned version of
all software via the Nix package manager. To our knowledge, Nix is the state of the art 
in terms of software reproducibility, package and dependency management, and solving
versioning issues in multi-language repositories.  You can install Nix on any Linux
distribution, MacOS and Windows (via WSL).
