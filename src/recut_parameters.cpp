#ifdef USE_MCP3D
#include <image/mcp3d_image_utils.hpp>
#endif
#include "recut_parameters.hpp"

void RecutCommandLineArgs::PrintUsage() {
  std::cout << "Basic usage : recut <image file or dir> [--seeds <marker_dir>] "
               "[--type point/uint8/mask/float/ims/tiff] "
               "[--convert <output_vdb_file_name>] "
               "[--bkg-thresh <int>] [--fg-percent <double>]\n\n";
  "[--image-offsets <int> [<int>] [<int>]] "
  "[--image-lengths <int> [<int>] [<int>]] "
  "[--channel <int>] "
  "[--resolution-level <int>] "
  "\nNote: neurite+soma images are binarized and do not need bkg-thresh or "
  "fg-percent specified";

  std::cout << "<image file or dir>  file or directory of input image(s)\n";
  std::cout << "--seeds              directory of files which represent known "
               "root/soma locations\n";
  std::cout
      << "--convert            [-cv] convert image file and exit defaults to "
         "out.vdb\n";
  std::cout
      << "--type               input type img: 'ims', 'tiff' | VDB: 'point', "
         "'uint8', 'mask' or 'float'\n";
  std::cout
      << "--prune-radius       larger values decrease node sampling density "
         "along paths, default 5 the z anisotropic factor\n";
  // std::cout << "--max                set max image voxel raw value allowed, "
  //"computed automatically when --bg_thresh or --fg-percent are "
  //"specified\n";
  // std::cout << "--min                set min image voxel raw value allowed, "
  //"computed automatically when --bg_thresh or --fg-percent are "
  //"specified\n";
  std::cout << "--channel            [-c] channel number, default 0\n";
  std::cout << "--resolution-level   [-rl] resolution level default 0 "
               "(original resolution)\n";
  std::cout
      << "--image-offsets      [-io] offsets of subvolume, in x y z order "
         "default 0 0 0\n";
  std::cout
      << "--image-lengths      [-ie] lengths of subvolume, in x y z order "
         "defaults"
         " to max range from offset start to max length in each axis (-1, -1, "
         "-1)\n";
  std::cout
      << "--bg-thresh          [-bt] background threshold value desired\n";
  std::cout
      << "--min-branch-length  prune leaf branches lower, defaults to 20\n";
  std::cout
      << "--fg-percent         [-fp] auto calculate a bg-thresh closest to a "
         "foreground \% between (0-100], overriding any --bg-thresh args. "
         "Value of .08 yields ~8 in 10,000 voxels "
         "as foreground per z-plane\n";
  // std::cout << "--prune              [-pr] prune 0 false, 1 true; defaults to
  // 1 "
  //"(automatically prunes)\n";
  std::cout
      << "--parallel           [-pl] thread count defaults to max hardware "
         "threads\n";
  std::cout
      << "--output-windows     specify uint8 vdb file for which to create "
         "windows surrounding each neuron cluster/component\n";
  std::cout
      << "--chunk-lengths   dimensions for fg percentages and conversion, "
         "defaults to image sizes\n";
  std::cout
      << "--downsample-factor  for images scaled down in x and z dimension "
         "scale the marker files by specified factor\n";
  std::cout << "--upsample-z         during --convert only z-dimension will be "
               "upsampled (copied) by specified factor, default is 1 i.e. no "
               "upsampling\n";
  std::cout
      << "--run-app2           for benchmarks and comparisons runs app2 on "
         "the vdb passed to --output-windows\n";
  std::cout << "--help               [-h] print this example usage summary\n";
}

std::string RecutCommandLineArgs::MetaString() {
  std::stringstream meta_stream;
  meta_stream << "image file/dir = " << image_root_dir << '\n';
  meta_stream << "channel = " << channel << '\n';
  meta_stream << "prune radius = " << prune_radius << '\n';
  meta_stream << "resolution level = " << resolution_level << '\n';
  meta_stream << "offsets (xyz) = " << image_offsets[0] << " "
              << image_offsets[1] << " " << image_offsets[2] << '\n';
  meta_stream << "lengths (xyz) = " << image_lengths[0] << " "
              << image_lengths[1] << " " << image_lengths[2] << '\n';
  meta_stream << "foreground_percent = " << foreground_percent << '\n';
  meta_stream << "background_thresh = " << background_thresh << '\n';
  meta_stream << "seeds directory = "
              << (seed_path.empty() ? "none" : seed_path) << '\n';
  return meta_stream.str();
}

RecutCommandLineArgs ParseRecutArgsOrExit(int argc, char *argv[]) {
  RecutCommandLineArgs args;
  if (argc < 2) {
    RecutCommandLineArgs::PrintUsage();
    exit(0);
  }
  try {
    if ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)) {
      RecutCommandLineArgs::PrintUsage();
      exit(0);
    } else {
      // global volume and channel selection
      args.image_root_dir = argv[1];
    }
    // if the switch is given, parameter(s) corresponding to the switch is
    // expected
    for (int i = 2; i < argc; ++i) {
      if (strcmp(argv[i], "--convert") == 0 || strcmp(argv[i], "-cv") == 0) {
        if (!((i+1) >= argc || argv[i+1][0] == '-')) {
          args.output_name = argv[i+1];
          ++i;
        }
        args.convert_only = true;
      } else if (strcmp(argv[i], "--seeds") == 0 ||
                 strcmp(argv[i], "-s") == 0) {
        args.seed_path = argv[i + 1];
        ++i;
      } else if (strcmp(argv[i], "--resolution-level") == 0 ||
                 strcmp(argv[i], "-rl") == 0) {
        args.resolution_level = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--image-offsets") == 0 ||
                 strcmp(argv[i], "-io") == 0) {
        GridCoord offsets(3);
        for (int j = 0; j < 3; ++j) {
          offsets[j] = atoi(argv[i + 1]);
          ++i;
        }
        args.set_image_offsets(offsets);
      } else if (strcmp(argv[i], "--image-lengths") == 0 ||
                 strcmp(argv[i], "-ie") == 0) {
        GridCoord lengths(3);
        for (int j = 0; j < 3; ++j) {
          lengths[j] = atoi(argv[i + 1]);
          ++i;
        }
        args.set_image_lengths(lengths);
      } else if (strcmp(argv[i], "--input-type") == 0) {
        auto arg = std::string(argv[i + 1]);
        if (arg == "float" || arg == "point" || arg == "uint8" ||
            arg == "mask" || arg == "ims" || arg == "tiff") {
          args.input_type = (argv[i + 1]);
        } else {
          cerr << "--input-type option must be one of "
                  "[float,point,uint8,mask,ims,tiff]\n";
          exit(1);
        }
        ++i;
      } else if (strcmp(argv[i], "--output-type") == 0) {
        auto arg = std::string(argv[i + 1]);
        if (arg == "float" || arg == "point" || arg == "uint8" ||
            arg == "mask" || arg == "ims" || arg == "tiff") {
          args.output_type = (argv[i + 1]);
        } else {
          cerr << "--output-type option must be one of "
                  "[float,point,uint8,mask,ims,tiff]\n";
          exit(1);
        }
        ++i;
      } else if (strcmp(argv[i], "--channel") == 0 ||
                 strcmp(argv[i], "-c") == 0) {
        args.channel = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--prune-radius") == 0) {
        args.prune_radius = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--bg-thresh") == 0 ||
                 strcmp(argv[i], "-bt") == 0) {
        args.background_thresh = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--fg-percent") == 0 ||
                 strcmp(argv[i], "-fp") == 0) {
        args.foreground_percent = atof(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--min-branch-length") == 0) {
        args.min_branch_length = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--chunk-lengths") == 0) {
        for (int j = 0; j < 3; ++j) {
          args.interval_lengths[j] = atoi(argv[i + 1]);
          ++i;
        }
      } else if (strcmp(argv[i], "--parallel") == 0 ||
                 strcmp(argv[i], "-pl") == 0) {
        args.user_thread_count = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--downsample-factor") == 0) {
        args.downsample_factor = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--combine") == 0) {
        args.combine = argv[i + 1];
        ++i;
      } else if (strcmp(argv[i], "--histogram") == 0) {
        args.histogram = true;
      } else if (strcmp(argv[i], "--output-windows") == 0) {
        args.output_windows = argv[i + 1];
        ++i;
      } else if (strcmp(argv[i], "--upsample-z") == 0) {
        args.upsample_z = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--run-app2") == 0) {
        args.run_app2 = true;
        ++i;
      } else {
        std::cout << "unknown option \"" << argv[i] << "\"  ...exiting\n\n";
        RecutCommandLineArgs::PrintUsage();
        exit(1);
      }
    }
    return args;
  } catch (const exception &e) {
    std::cout << e.what() << '\n';
    RecutCommandLineArgs::PrintUsage();
    exit(1);
  }
}
