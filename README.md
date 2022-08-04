[![built with
nix](https://builtwithnix.org/badge.svg)](https://builtwithnix.org)

## Quick Install With Nix (strongly recommended)
Make sure `git` and `curl` is installed and available from your command line and copy and paste the single command to install the Nix package manager for your operating system found [here](https://nixos.org/download.html).

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
### Update to the latest version
If some time has passed since your original installation and you need to update to the latest version, from the Recut github directory run:
```
git pull origin master
nix-env -f . -i
```

## Usage
Once recut is installed globally you can see the example usage by running on the command line:
```
recut
```
Recut has several principle usages: 
1. Compressing image volumes to a sparse format [VDB grids](https://www.openvdb.org/documentation/doxygen/faq.html) 
2. Using VDB grids to reconstruct neurons into a set of SWC trees.
3. Creating volumetric image windows for bounding boxes of interest, for example around neurons for proofreading. This is accomplished efficiently and in parallel by uncompressing from the sparse VDB format into .tiff which can be more widely consumed by visualization or machine learning software.
4. Filtration and quality control of neurons based on domain specific features. Since recut takes a batch process approach where large volumes of neuron tissue are reconstructed concurrently, methods that filter and discard neurons based on tuned features automatically are critical.
5. Converting .ims or .vdb files to .tiff and other image to image conversions.

The first argument passed to recut is the path to the input which can either be a directory for .tiff files or a distinct file in the case of .ims or .vdb files. The ordering of all other arguments is arbitrary. The default behavior is to convert either an .ims file or .tiff directory to a VDB file and exit. To do an end-to-end reconstruction pass an .ims, .tiff, or point VDB as input and specify a `--seeds` directory. Arguments have assumed default values for the most common expected behavior.

## Warning
Where possible Recut attempts to use the maximum threads available to your system by default. This can be a problem when converting large images or when using the `--output-windows` with multiple large grids since each thread is grabbing large chunks of images and operating on them in DRAM. Meanwhile reconstruction alone consumes very little memory since it operates on images that have already been compressed to VDB grids. In general you should use the system with the maximum amount of DRAM that you can. When you are still limited on DRAM you should lower the thread count used by recut by specifying the threads like `--parallel 2`. When recut consumes too much memory you will see erros like `Segmentation fault` or `SIGKILL`. Lower the thread count until the process can complete, you can monitor the dynamic usage of your DRAM during execution by running the command `htop` in a separate terminal. This can be helpful to guage what parallel factor you want to use.

### Conversion
Convert the folder `ch0` into a VDB point grid:
```
recut ch0 --input-type tiff --output-type point --fg-percent .05
```
This creates a .vdb file in the ch0 folder. The name is tagged with information about the conversion process for example the argument values used. This is done to preserve info about the source image and to lower the likelihood of overwriting a costly previous conversion with a generic name. Explicit names are helpful for humans but if you want to pass a simpler name you can do so.

Convert the folder again, but this time only take z-planes of 0 through 15 and name it `subset.vdb`
```
recut ch0 -o subset.vdb --input-type tiff --output-type point --image-lengths -1 -1 16 --fg-percent .05
```

.vdbs are a binary format, you can only view information or visualize them with software that explicitly supports them. However for quick info, Recut installs some of the VDB libraries command line tools.

List the exhausitive info and metatdata about the VDB grid:
```
vdb_print -l -m subset.vdb
```
You'll notice that vdb grid's metadata has been stamped with information about the arguments used during conversion to help distinguish files with different characteristics. If you have no other way to match the identity of VDB with an original image, refer to the original bounding extents which can often uniquely identify an image. The original bounding extents of the image used are preserved as the coordinate frame of all points and voxels regardless of any length arguments passed. Note that the command line tools `vdb_view`and `vdb_print` as well as Houdini will not work with type uint8 grids.

#### Image Inference Conversions
The highest quality *reconstructions* currently involve running the MCP3D pipeline's neurite and soma segmentation and connected component stage followed by recut conversion to a point grid followed by recut's reconstruction of that point grid. MCP3D's connected component stage will output the soma locations `marker_files` for use in reconstruction as shown below. MCP3D's segmentation will output a directory of tiff files of the binarized segmented image, with all background set to 0, therefore converting like we did before:
```
recut ch0 -o point.vdb --input-type tiff --output-type point
```
Note that this time we left off the `--fg-percent` flag so Recut will automatically cut off pixels valued `0` while saving foreground pixels everywhere else. This is only recommended for inference outputs since their pixel intensity values have already been binarized (0 or 1) and they tend to have only .05% of voxels labeled as foreground (i.e. value of 1).

#### Image Raw Conversions
While the process above produces the highest quality *reconstructions* with smaller likelihoods of path breaks, we often want a quick and dirty way to view the original image or a separate channel of the original image for proofreading via the `--output-windows` flag during reconstruction. One way of doing this is to convert using a guessed foreground percentage of the image like so:

```
recut test.ims -o uint8.vdb --input-type ims --output-type uint8 --channel 0 --fg-percent .05
recut test.ims -o mask.vdb --input-type ims --output-type mask --channel 1 --fg-percent 10
```
Note that doing multiple image conversions is only necessary if you want to output windows (create cropped images) while doing a reconstruction. Note the size of VDBs on disk reflect mainly the active voxel count of your conversion, for light microscopy data it's rare that you will want foreground percentages above 1%.

It's possible to background threshold based off a known raw pixel intensity value, for example if you have already background thresholded, clamped, and or background subtracted an image. In such cases simply convert like below, inputing your known intensity value:
```
recut ch0 -o point.vdb --input-type tiff --output-type point --bg-thresh 127
```
While this results in the fastest conversions it is rarely useful since it is highly sensitive to image normalization and preprocessing.

It is possible to run the full Recut pipeline on raw images but it may require guesswork in selecting the right foreground percentage for your image as shown below.
Either way you will want an automated way of detecting seed points (somas) see the Seeds section in the documentation below.
```
# an example flow on raw images
recut ch0 -o point.vdb --input-type tiff --output-type point --fg-percent .05
# you would need to write the coordinates and radii of known seed points (somas) by hand in marker_files/
recut point.vdb --seeds marker_files
```
### Reconstruct

If you've created a point grid for an image, for example named `point.vdb`, the following would reconstruct the image based off of a directory of files which note the coordinates of starting locations (somas). This directory in the following example is shown as `marker_files`:
```
recut point.vdb --seeds marker_files
```
This will create a folder in your current directory `run-1` which has folder for each component of neurons and their respective SWC outputs.

If you created a corresponding VDB grid of type uint8 for channel 0 and type mask for channel 1 you can also output a window for the bounding box each individual swc for proofreading by:
```
recut point.vdb --seeds marker_files --output-windows uint8.vdb mask.vdb
```
This will create a folder in your current directory `run-2` which has a folder for each component of neurons along with its compressed TIFF file for the bounding volume of the component for the uint8.vdb and mask.vdb grid passed to be used in a SWC viewing software.

Instead of outputting windows from the original image as demonstrated above, it's also possible to use the binarized inference image which Recut uses for reconstruction. To create bounding box windows around each swc first convert the inferenced image into a VDB grid of type mask named for example `inference-mask` then pass it as an argument like so:
```
recut point.vdb --seeds marker_files --output-windows uint8.vdb mask.vdb inference-mask.vdb
```
Now each component folder will have a TIFF window from the original image, the 1-bit channel 1, and the binarized inference of the original inference. Having different image windows (labels) with corresponding SWCs can be a very efficient way to retrain or build better neural network models since it removes much of the manual human annotation steps.

#### Seeds
Even in inferenced neural tissue of internal data, only about 20% of foreground voxels are reachable from known soma locations. In order for Recut to build trees it must traverse from a seed (soma) point. These seed points are generated by the MCP3D's connected component stage which runs after the inference stage. If you wish to generate soma locations via a separate method, output all somas in the image into separate files in the same folder. Each file contains a single line with the coordinate and radius information separated by commas like so:
`X,Y,Z,RADIUS`

#### Outputs
Within the directory `recut` is invoked from, a new folder named `run-1` will be created which contains a set of folders for each connected component connected to at least 1 seed point. The folders prepended with `a-multi...` contain multiple somas (seed points), therefore these particular folders contain multiple SWC files (trees) within them. If you ran the reconstruction passing different images to `--output-windows` these folders will also contain compressed tiff volumes for the bounding box of all trees within the component for proofreading or training. You can do further analysis or run quality control on these outputs if you install [StdSwc](http://neuromorpho.org/StdSwc1.21.jsp) and run `[recut_root_dir]/scripts/batch-std-swc.sh run-1` for the run directory generated. For each tree in the run directory a new corresponding text file will be placed alongside it logging any warnings for the proofreader. These logs are prepended with `stdlog-...`.

### Combine
You call recut to process the inputs using an action, for example `--combine`. 

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
git clone git@github.com:UCLA-VAST/recut
cd recut
mkdir build
cd build
cmake ..
make [-j 8]
sudo make install [-j 8]
# required to generate test data
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
- Optionally: python3.8 matplotlib, gdb, clang-tools, linux-perf for development
- Note: to increase reproducibility and dependencies issues we recommend developing within the Nix package environment (see the Troubleshooting section)

#### Troubleshooting
Some of Recut's dependencies require later releases then you may have
installed on your system, for example CMake. In these scenarios, or if you're
running into compile time issues we recommend running a pinned version of
all software via the Nix package manager. To our knowledge, Nix is the state of the art 
in terms of software reproducibility, package and dependency management, and solving
versioning issues in multi-language repositories.  You can install Nix on any Linux
distribution, MacOS and Windows (via WSL).

#### Cite
If you find this software helpful please consider citing the [preprint](https://www.biorxiv.org/content/10.1101/2021.12.07.471686v2.full).
