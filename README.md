[![built with
nix](https://builtwithnix.org/badge.svg)](https://builtwithnix.org)

# Quick Install With Nix (strongly recommended)
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

# Features
Recut has several principle usages: 
1. Compressing image volumes to a sparse format [VDB grids](https://www.openvdb.org/documentation/doxygen/faq.html) (`.vdb`)
2. Segment, reconstruct and skeletonize cells in medical images
3. Creating volumetric image windows for bounding boxes of interest, for example around cells for supervised training labels or proofreading. This is accomplished efficiently and in parallel by uncompressing from the sparse VDB format into `.tiff` which can be more widely consumed by visualization or machine learning software. For example, the cell-based U-net inference software in Recut's pipeline directory consumes `.tiff`s
4. Filtration and quality control of skeletonized cells (trees) based on domain-specific features. Since recut takes a batch process approach where large volume medical images are reconstructed concurrently, methods that filter and discard cells automatically are critical
5. Image conversion tools between `.ims`,`.vdb`, and `.tiff` for visualization, reconstruction and machine learning

# Usage 
Once recut is installed globally you can see the example usage by running on the command line:
```
recut
```
The first argument passed to recut is the path to the input which can either be a directory of `.tiff` files or an `.ims` or `.vdb` file. The ordering of all other arguments is arbitrary. The default behavior is to run an end-to-end reconstruction of the passed `.ims` or `.tiff` image.  Arguments have assumed default values for the most common expected behavior.

### Acceptable Input Formats
The first argument passed to recut is its input.
The list of possible inputs to recut are listed when printing recut's command line help via: `recut`. More specifically recut can take a directory of grayscale 8 or 16-bit 2D TIF planes, 16-bit Imaris (HDF5) files, or several VDB types.

### Reconstruct

The most common use case is to segment and skeletonize cell bodies from an image. If you had a set of 2D TIF planes in a folder called `ch0` you could reconstruct cells from the volume like so:
``` 
recut ch0
```
This will create a folder in your current directory `run-1` which has a folder for each component of neurons and their respective skeletonized SWC (tree-format) outputs.

### Morphological Operations and Seed Segmentation
In order to find the cell bodies of branching neurons more effectively, recut accepts a parameter to conduct morphological opening like so:
```
recut ch0 --open-steps 5
```
This will erase background noise, small islands, and thin projections in the image like the neurites that branch off a cell body. With these projections erased, the true cell body (tree root) are recovered quite robustly.

While the command above works for images with cells with clearly filled interiors, some imaging techniques only label the cell surface (contour). In such cases, we need to morphologically close the image before opening such that holes and valleys are filled. We do so like this: 
```
recut ch0 --input-type mask --close-steps 8 --open-steps 5
```

For brain volumes with voxel size [1,1,1] in um, we found a morphological closing step of 8 followed by a morphological opening step of 5 with a foreground percent of .1 to be best for segmenting hollow cell body (seed) regions. The estimated cell body location and coordinates will be created in the new run directory under `seeds/`. 

#### Seeds
Even in inferenced neural tissue of internal data, only about 20% of foreground voxels are reachable from known seed locations. In order for Recut to build trees it must traverse from a seed (cell body) point. These seed points can be picked from the image by hand or from the U-net model in the `recut/pipeline` folder. If you wish to generate seed locations via a separate method, output all seed in the image into separate files in the same folder. Each file contains a single line with the coordinate and radius information separated by commas like so:
`X,Y,Z,RADIUS`

## Conversions
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
The highest quality *reconstructions* currently involve running the MCP3D pipeline's neurite and cell body segmentation and connected component stage followed by recut conversion to a point grid followed by recut's reconstruction of that point grid. MCP3D's connected component stage will output the cell body locations `marker_files` for use in reconstruction as shown below. MCP3D's segmentation will output a directory of tiff files of the binarized segmented image, with all background set to 0, therefore converting like we did before:
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
Either way you will want to make sure your seed points (cell bodies) are properly detected see the morphological operations section in the documentation.
```
# do an explicit conversion
recut ch0 -o mask.vdb --output-type mask --fg-percent .05
# pass the output to a reconstruction run
recut mask.vdb --close-steps 8 --open-steps 5
```
which is equivalent to:
```
recut ch0 --fg-percent .05 --close-steps 8 --open-steps 5
```

### Output Windows (crops, ROIs, labels, etc.)
If you created a corresponding VDB grid of type uint8 for channel 0 and type mask for channel 1 you can also output a window for the bounding box each individual swc for proofreading by:
```
recut ch0 --output-windows uint8.vdb mask.vdb
```
This will create a folder in your current directory `run-2` which has a folder for each component of neurons along with its compressed TIFF file for the bounding volume of the component for the uint8.vdb and mask.vdb grid passed to be used in a SWC viewing software.

Instead of outputting windows from the original image as demonstrated above, it's also possible to use the binarized inference image which Recut uses for reconstruction. To create bounding box windows around each swc first convert the inferenced image into a VDB grid of type mask named for example `inference-mask` then pass it as an argument like so:
```
recut ch0 --output-windows uint8.vdb mask.vdb inference-mask.vdb
```
Now each component folder will have a TIFF window from the original image, the 1-bit channel 1, and the binarized inference of the original inference. Having different image windows (labels) with corresponding SWCs can be a very efficient way to retrain or build better neural network models since it removes much of the manual human annotation steps.

If your components have path breaks it is recommended to use the flags `--min-window` or `--expand-window` this will increase the output window sizes in attempt to capture all the branch extensions or surrounding context for proofreading.

#### Outputs
Within the directory `recut` is invoked from, a new folder named `run-1` will be created which contains a set of folders for each connected component connected to at least 1 seed point. The folders prepended with `a-multi...` contain multiple cell bodies (seed points), therefore these particular folders contain multiple SWC files (trees) within them. If you ran the reconstruction passing different images to `--output-windows` these folders will also contain compressed tiff volumes for the bounding box of all trees within the component for proofreading or training. You can do further analysis or run quality control on these outputs if you install [StdSwc](http://neuromorpho.org/StdSwc1.21.jsp) and run `[recut_root_dir]/scripts/batch-std-swc.sh run-1` for the run directory generated. For each tree in the run directory a new corresponding text file will be placed alongside it logging any warnings for the proofreader. These logs are prepended with `stdlog-...`.

## Other Usages
#### Reconstruction from known seeds
We recommend allowing Recut to find seeds automatically, but its still possible to pass in custom seeds for a particular run.
Passing in seeds will filter the connected components (3D segmented blobs) in the image to only those that contain a passed seed.
The radius and volume of the passed seeds are ignored and recut will create an image accurate radii and volume estimation in the runs `seeds` directory. 
For analysis of biological features, we strongly recommend using the volume in the filename in the `seeds` over the radius or especially the radius of SWC files, even those that have been proofread. This is because recut's volume estimate is an exact pixel coverage of the seed at the given foreground percent of background threshold.
The following would reconstruct the image based off of a directory of files which note the coordinates of starting locations (somas, seeds or roots of the tree). This directory in the following example is shown as `marker_files`:
```
recut mask.vdb --seeds marker_files
```

#### Training Labels
The output windows generated by Recut can also be used to (re)train new neural network models using the script in `recut/pipeline/python/train_model.py`. By default this script expects 16-bit 2D series grayscale window crops with corresponding 8-bit RGB TIFFs to assign a class label. For our usages so far a red value of 255 indicates a voxel of a seed, and a green value of 255 indicates a voxel of a neurite in the corresponding image. Both sets of image and label files are used for training but the creation of these windows is simplified by specifying `--output-type labels` like so:

`recut image_dir --seeds marker_files --output-windows uint8.vdb mask.vdb --output-type labels --voxel-size 1 1 1`

Specifying `--output-type labels` saves the window as a 2D series instead of the default compressed 3D (multi-page) tiff and only saves the z-planes of the component around the seed/soma. This is so 2D labels can be created/modified by hand and compared to their RGB labels efficiently.

#### Combine
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

### CMake Only Installation (Deprecated)
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

### Known Issues
Where possible Recut attempts to use the maximum threads available to your system by default. This can be a problem when converting large images or when using the `--output-windows` with multiple large grids since each thread is grabbing large chunks of images and operating on them in DRAM. Meanwhile reconstruction alone consumes very little memory since it operates on images that have already been compressed to VDB grids. In general you should use the system with the maximum amount of DRAM that you can. When you are still limited on DRAM you should lower the thread count used by recut by specifying the threads like `--parallel 2`. When recut consumes too much memory you will see erros like `Segmentation fault` or `SIGKILL`. Lower the thread count until the process can complete, you can monitor the dynamic usage of your DRAM during execution by running the command `htop` in a separate terminal. This can be helpful to guage what parallel factor you want to use.

### Troubleshooting
Some of Recut's dependencies require later releases then you may have
installed on your system, for example CMake. In these scenarios, or if you're
running into compile time issues we recommend running a pinned version of
all software via the Nix package manager. To our knowledge, Nix is the state of the art 
in terms of software reproducibility, package and dependency management, and solving
versioning issues in multi-language repositories.  You can install Nix on any Linux
distribution, MacOS and Windows (via WSL).

# Scientific Motivation
The execution pattern and partitioning strategy resembles this [paper]( https://arxiv.org/abs/1811.00009), however we build on top of the [VDB library](https://github.com/AcademySoftwareFoundation/openvdb) for performance in sparse large-scale settings. 

# Cite
If you find this software helpful please consider citing the [preprint](https://www.biorxiv.org/content/10.1101/2021.12.07.471686v2.full).
