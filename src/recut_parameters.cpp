#include "recut_parameters.hpp"

void RecutCommandLineArgs::PrintUsage() {
  std::cout << "Basic usage : recut <image file or dir> [--seeds <marker_dir>] "
               "[--type point/uint8/mask/float/ims/tiff] "
               "[-o <output_vdb_file_name>] "
               "[--bkg-thresh <int>] [--fg-percent <double>]\n\n";
  //"[--image-offsets <int> [<int>] [<int>]] "
  "[--image-lengths <int> [<int>] [<int>]] "
  "[--channel <int>] "
  "[--resolution-level <int>] "
  "\nNote: neurite+soma images are binarized and do not need bkg-thresh or "
  "fg-percent specified";

  std::cout << "<image file or dir>  file or directory of input image(s)\n";
  std::cout << "--seeds              directory of files which represent known "
               "root/soma locations, seeds are required to reconstruct\n";
  std::cout << "--output-name        [-o] give converted vdb a custom name "
               "defaults to "
               "naming with useful image attributes\n";
  std::cout << "--input-type         input type img: 'ims', 'tiff' | "
               "VDB: 'point', "
               "'uint8', 'mask' or 'float'\n";
  std::cout << "--output-type        output type img: 'ims', 'tiff' | "
               "VDB: 'point', "
               "'uint8', 'mask' or 'float' | 'swc', 'eswc', 'labels'\n";
  // std::cout << "--max                set max image voxel raw value allowed, "
  //"computed automatically when --bg_thresh or --fg-percent are "
  //"specified\n";
  // std::cout << "--min                set min image voxel raw value allowed, "
  //"computed automatically when --bg_thresh or --fg-percent are "
  //"specified\n";
  std::cout << "--channel            [-c] channel number, default 0\n";
  std::cout << "--resolution-level   [-rl] resolution level default 0 "
               "(original resolution)\n";
  // std::cout
  //<< "--image-offsets      [-io] offsets of subvolume, in x y z order "
  //"default 0 0 0\n";
  std::cout << "--voxel-size         um lengths of voxel in x y z order "
               "default 1.0 1.0 1.0 determines prune radius\n";
  std::cout
      << "--prune-radius       larger values decrease node sampling density "
         "along paths, default is set by the anisotropic factor of "
         "--voxel-size\n";
  std::cout
      << "--image-lengths      [-ie] lengths of subvolume, in x y z order "
         "defaults"
         " to max range from start to max length in each axis (-1, -1, "
         "-1)\n";
  std::cout << "--bg-thresh          [-bt] all pixels greater than this passed "
               "intensity value are treated as foreground\n";
  std::cout
      << "--min-branch-length  prune leaf branches lower, defaults to 20\n";
  std::cout
      << "--fg-percent         [-fp] auto calculate a bg-thresh value closest "
         "to the passed "
         "foreground \% between (0-100], overriding any --bg-thresh args. "
         "Value of .08 yields ~8 in 10,000 voxels "
         "as foreground per z-plane\n";
  // std::cout << "--prune              [-pr] prune 0 false, 1 true; defaults to
  // 1 "
  //"(automatically prunes)\n";
  std::cout
      << "--parallel           [-pl] thread count defaults to max hardware "
         "threads\n";
  std::cout << "--output-windows     list 1 or more uint8 vdb files in channel "
               "order to create "
               "image windows for each neuron cluster/component\n";
  std::cout
      << "--tile-lengths       dimensions for fg percentages and conversion, "
         "defaults to image sizes\n";
  std::cout
      << "--downsample-factor  for images scaled down in x and z dimension "
         "scale the marker files by specified factor\n";
  std::cout
      << "--upsample-z         during conversion only z-dimension will be "
         "upsampled (copied) by specified factor, default is 1 i.e. no "
         "upsampling\n";
  std::cout
      << "--min-window     windows by default only extend to bounding volume "
         "of their component, this value specifies the minimum window border "
         "surrounding seeds, if no um value is passed it will use "
      << MIN_WINDOW_UM << " um\n";
  std::cout
      << "--expand-window        windows by default only extend to bounding "
         "volume of their component, this allows specifying an expansion "
         "factor around seeds, if no um value is passed it will use "
      << EXPAND_WINDOW_UM << " um\n";
  std::cout
      << "--open-steps         # of iterations of morphological opening\n";
  std::cout
      << "--close-steps        # of iterations of morphological closing\n";
  std::cout
      << "--run-app2           for benchmarks and comparisons runs app2 on "
         "the vdb passed to --output-windows\n";
  std::cout << "--help               [-h] print this example usage summary\n";
}

std::string RecutCommandLineArgs::MetaString() {
  std::stringstream meta_stream;
  meta_stream << "image file/dir = " << input_path << '\n';
  meta_stream << "channel = " << channel << '\n';
  meta_stream << "resolution level = " << resolution_level << '\n';
  // meta_stream << "offsets (xyz) = " << image_offsets[0] << " "
  //<< image_offsets[1] << " " << image_offsets[2] << '\n';
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
    exit(0); // allow CI to test the binary, and receive a success signal
  }
  try {
    if ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)) {
      RecutCommandLineArgs::PrintUsage();
      exit(0);
    } else {
      // global volume and channel selection
      args.input_path = argv[1];
    }
    // if the switch is given, parameter(s) corresponding to the switch is
    // expected
    for (int i = 2; i < argc; ++i) {
      if (strcmp(argv[i], "--output-name") == 0 || strcmp(argv[i], "-o") == 0) {
        if (!((i + 1) >= argc || argv[i + 1][0] == '-')) {
          args.output_name = argv[i + 1];
          ++i;
        }
      } else if (strcmp(argv[i], "--seeds") == 0 ||
                 strcmp(argv[i], "-s") == 0) {
        args.seed_path = argv[i + 1];
        args.convert_only = false;
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
      } else if (strcmp(argv[i], "--voxel-size") == 0) {
        for (int j = 0; j < 3; ++j) {
          args.voxel_size[j] = atof(argv[i + 1]);
          ++i;
        }
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
            arg == "mask" || arg == "ims" || arg == "tiff" || arg == "eswc" ||
            arg == "swc" || arg == "labels") {
          args.output_type = (argv[i + 1]);
        } else {
          cerr << "--output-type option must be one of "
                  "[float,point,uint8,mask,ims,tiff,swc,eswc,labels]\n";
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
      } else if (strcmp(argv[i], "--tile-lengths") == 0) {
        for (int j = 0; j < 3; ++j) {
          args.tile_lengths[j] = atoi(argv[i + 1]);
          ++i;
        }
      } else if (strcmp(argv[i], "--parallel") == 0 ||
                 strcmp(argv[i], "-pl") == 0) {
        args.user_thread_count = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--min-window") == 0) {
        if ((i + 1 < argc) && (argv[i + 1][0] != '-')) {
          args.min_window_um = atof(argv[i + 1]);
          ++i;
        } else {
          args.min_window_um = MIN_WINDOW_UM;
        }
      } else if (strcmp(argv[i], "--expand-window") == 0) {
        if ((i + 1 < argc) && (argv[i + 1][0] != '-')) {
          args.expand_window_um = atof(argv[i + 1]);
          ++i;
        } else {
          args.min_window_um = EXPAND_WINDOW_UM;
        }
      } else if (strcmp(argv[i], "--open-steps") == 0) {
        args.open_steps = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--close-steps") == 0) {
        args.close_steps = atoi(argv[i + 1]);
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
        // need to pass in at least 1 grid name
        if ((i + 1 == argc) || (argv[i + 1][0] == '-')) {
          RecutCommandLineArgs::PrintUsage();
          exit(1);
        }
        while ((i + 1 < argc) && (argv[i + 1][0] != '-')) {
          args.window_grid_paths.push_back(argv[i + 1]);
          ++i;
        }
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

  // logic checks
  if (args.run_app2 && args.window_grid_paths.empty()) {
    RecutCommandLineArgs::PrintUsage();
    std::logic_error(
        "If run-app2 option is set an output-window must be passed");
  }
}
