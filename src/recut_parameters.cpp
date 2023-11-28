#include "recut_parameters.hpp"

void RecutCommandLineArgs::PrintUsage() {
  std::cout << "Basic usage : recut <image or VDB> "
               "--voxel-size X Y Z (in µm) "
               "[--seeds <marker_dir>] "
               "[--output-type swc/eswc/seeds/uint8/mask/float/tiff] "
               "[--fg-percent <double>]\n\n";
  std::cout << "<image or VDB>         file or directory of input image(s)\n";
  std::cout << "--voxel-size           µm lengths of voxel in x y z order "
               "default 1.0 1.0 1.0\n";
  std::cout << "--fg-percent           [-fp] auto calculate a bg-thresh value "
               "closest "
               "to the passed "
               "foreground % recommended between (0.1-1], only applies to "
               "image to VDB stage, "
               "defaults to value of "
            << FG_PCT << '\n';
  std::cout << "--seeds <path>         optionally pass either a directory "
               "of SWC files with 1 soma node per file, or a single *.swc file containing a single soma, "
               "fills the seeds as spheres directly on to the mask image, "
               "treating them as ground truth for reconstruction\n";
  std::cout << "--seed-action <action> can either be 'force' which forces the locations passed by users to be used for final reconstruction"
               " whereas 'find' finds the closest point in the skeleton to the passed seed, find-valent finds the point within the soma radius * dilation with the maximum connections\n";
  std::cout
      << "--parallel             [-pl] thread count defaults to max hardware "
         "threads\n";
  std::cout
      << "--output-windows       list 1 or more uint8 vdb files in channel "
         "order to create "
         "image windows for each neuron cluster/component\n";
  std::cout << "--input-type           inferred by default, valid inputs "
               "include image types ('ims' or 'tiff') or VDB types ('uint8' or "
               "'mask')\n";
  std::cout
      << "--output-type          valid output types include image types "
         "('ims' or 'tiff'), VDB types "
         "('uint8' or 'mask'), tree types ('swc' or 'eswc'), and misc. types "
         "('labels' or 'seeds'), default output is 'swc'\n";
  std::cout << "--close-steps          morphological closing level "
               "to fill hollow signals inside somata or to join path breaks "
               "defaults to roughly "
            << SOMA_CLOSE_FACTOR << " / [voxel size]\n";
  std::cout
      << "--preserve-topology    do not apply morphological closing to the "
         "neurites of the image; defaults to closing both somas and topology "
         "(neurites)\n";
  std::cout << "--open-steps           iterations of morphological opening, "
               "this will roughly erase neurites and blobs with voxel radius "
               "smaller than the integer value passed, thus yielding only the "
               "comparatively large somata; "
               "defaults to "
            << OPEN_FACTOR
            << " / [voxel size] for soma inference and soma intersection runs, "
               "for --seeds X force, opening is not applied\n";
  std::cout
      << "--image-offsets        [-io] offsets of subvolume, in x y z order "
         "default 0 0 0\n";
  std::cout << "--coarsen-steps        determines granularity of final skeletons, lower "
               "values result in higher detailed skeletons (SWC trees) with "
               "more skeletal nodes; default is " << COARSEN_STEPS << '\n';
            //<< COARSEN_FACTOR << " / [voxel size]\n";
  //std::cout << "--skeleton-grain       granularity of final skeletons, lower "
               //"value result in higher detailed skeletons (SWC trees) with "
               //"more skeletal nodes; default is "
            //<< SKELETON_GRAIN << '\n';
  //std::cout << "--skeleton-grow        affects granularity of final skeletons, "
               //"higher value results in higher detailed skeletons (SWC trees) "
               //"with more points; default is "
            //<< GROW_THRESHOLD << '\n';
  std::cout << "--smooth-steps         higher values smooths the node "
               "positions and radius of final skeletons, "
               "improving uniformity along paths; defaults to "
            << SMOOTH_STEPS << '\n';
  std::cout << "--soma-dilation        factor to multiply computed soma size "
               "by to collapse somal nodes, works in combination with 'find-valent' and 'force' actions "
               "defaults to "
            << FIND_SOMA_DILATION << " or " << FORCE_SOMA_DILATION << " respectively\n";
  std::cout
      << "--image-lengths        [-ie] lengths of subvolume as x y z "
         "defaults"
         " to max range from start to max length in each axis which could be "
         "specified by -1 -1 -1\n";
  // std::cout << "--bg-thresh            [-bt] all pixels greater than this
  // passed " "intensity value are treated as foreground\n";
  //std::cout << "--min-branch-length    prune leaf branches lower than path "
               //"length, defaults to "
            //<< MIN_BRANCH_LENGTH << " µm\n";
  // std::cout << "--mean-shift         radius to mean shift nodes towards local
  // " "mean which aids pruning; default 0\n";
  // std::cout
  //<< "--mean-shift-iters   max iterations allowed for mean shift "
  //"convergence; most smoothing converges by the default 4 iterations\n";
  // std::cout
  //<< "--tile-lengths         dimensions for fg percentages and conversion, "
  //"defaults to image sizes\n";
  // std::cout
  //<< "--downsample-factor    for images scaled down in x and z dimension "
  //"scale the marker files by specified factor\n";
  std::cout
      << "--upsample-z           during conversion only z-dimension will be "
         "upsampled (copied) by specified factor, default is 1 i.e. no "
         "upsampling\n";
  std::cout
      << "--min-window           windows by default only extend to bounding "
         "volume "
         "of their component, this value (in µm units) specifies the minimum "
         "window border "
         "surrounding seeds, if no value is passed it will use "
      << MIN_WINDOW_UM << " µm\n";
  std::cout
      << "--expand-window        windows by default only extend to bounding "
         "volume of their component, this allows specifying an expansion "
         "factor around seeds, if no µm value is passed it will use "
      << EXPAND_WINDOW_UM << " µm\n";
  // std::cout << "--order               morphological operations (open/close) "
  //"order An integer between 1 to 5 that defines the mathematical "
  //"complexity (order) of operations "
  //"defaults to 1\n";
  std::cout << "--min-radius           min allowed radius of the soma in µm "
               "used in the soma detection phase, defaults to "
            << MIN_SOMA_RADIUS_UM << " µm\n";
  std::cout << "--max-radius           max allowed radius of the soma in µm "
               "used in the soma detection phase, defaults to "
            << MAX_SOMA_RADIUS_UM << " µm\n";
  std::cout << "--save-vdbs            save intermediate VDB grids during "
               "reconstruction transformations\n";
  // std::cout
  //<< "--run-app2             for benchmarks and comparisons runs app2 on "
  //"the vdb passed to --output-windows\n";
  // std::cout << "--timeout             time in minutes to automatically cancel
  // pruning of a single component\n";
  std::cout << "--disable-swc-scaling  this flag outputs all SWCs in image voxel units\n";
  std::cout << "--help                 [-h] print this example usage summary\n";
}

std::string RecutCommandLineArgs::MetaString() {
  std::stringstream meta_stream;
  meta_stream << "image file/dir = " << input_path << '\n';
  meta_stream << "lengths (xyz) = " << image_lengths[0] << " "
              << image_lengths[1] << " " << image_lengths[2] << '\n';
  meta_stream << "foreground_percent = " << foreground_percent << '\n';
  meta_stream << "background_thresh = " << background_thresh << '\n';
  meta_stream << "seeds directory = " << (seed_path == "" ? "none" : seed_path)
              << '\n';
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
      // canonical removes the trailing slash
      args.input_path = std::filesystem::canonical(argv[1]);
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
        // need to pass in at least 1 seed path or file
        if ((i + 1 == argc) || (argv[i + 1][0] == '-')) {
          RecutCommandLineArgs::PrintUsage();
          exit(1);
        }
        // canonical removes the trailing slash
        args.seed_path = std::filesystem::canonical(argv[i + 1]);
        if (!std::filesystem::exists(args.seed_path)) {
          cerr << "--seeds path does not exist\n";
          exit(1);
        }
        if (!fs::is_directory(args.seed_path) && (fs::is_regular_file(args.seed_path) && (args.seed_path.extension() != ".swc"))) {
          cerr << "--seeds must be a directory of markers, swcs or a single *.swc file\n";
          exit(1);
        }
        ++i;
      } else if (strcmp(argv[i], "--seed-action") == 0 ||
                 strcmp(argv[i], "-s") == 0) {
        args.seed_action = argv[i + 1];

        if (!((args.seed_action == "force") || (args.seed_action == "find") 
              || (args.seed_action == "find-valent")))
          throw std::runtime_error("unrecognized seed action");
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
        args.image_offsets = offsets;
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
        args.image_lengths = lengths;
      } else if (strcmp(argv[i], "--input-type") == 0) {
        auto arg = std::string(argv[i + 1]);
        std::transform(arg.begin(), arg.end(), arg.begin(),
                       [](auto c) { return std::tolower(c); });
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
        std::transform(arg.begin(), arg.end(), arg.begin(),
                       [](auto c) { return std::tolower(c); });
        if (arg == "float" || arg == "point" || arg == "uint8" ||
            arg == "mask" || arg == "ims" || arg == "tiff" || arg == "eswc" ||
            arg == "swc" || arg == "labels" || arg == "seeds") {
          args.output_type = (argv[i + 1]);
          if (arg == "mask" || arg == "float" || arg == "uint8" ||
              arg == "point") {
            args.convert_only = true;
          }
        } else {
          cerr << "--output-type option must be one of "
                  "[float,point,uint8,mask,ims,tiff,swc,eswc,labels,seeds]\n";
          exit(1);
        }
        ++i;
      } else if (strcmp(argv[i], "--save-vdbs") == 0) {
        args.save_vdbs = true;
      } else if (strcmp(argv[i], "--ignore-multifurcations") == 0) {
        args.ignore_multifurcations = true;
      } else if (strcmp(argv[i], "--channel") == 0 ||
                 strcmp(argv[i], "-c") == 0) {
        args.channel = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--soma-dilation") == 0) {
        args.soma_dilation = atof(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--skeleton-grain") == 0) {
        args.skeleton_grain = atof(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--skeleton-grow") == 0) {
        args.skeleton_grow = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--smooth-steps") == 0) {
        auto val = atoi(argv[i + 1]);
        args.smooth_steps = val;
        ++i;
      } else if (strcmp(argv[i], "--coarsen-steps") == 0) {
        auto val = atoi(argv[i + 1]);
        args.coarsen_steps = val;
        ++i;
      } else if (strcmp(argv[i], "--saturate-edges") == 0) {
        auto val = atoi(argv[i + 1]);
        args.saturate_edges = val;
        ++i;
      } else if (strcmp(argv[i], "--optimize-steps") == 0) {
        auto val = atoi(argv[i + 1]);
        args.optimize_steps = val;
        ++i;
      } else if (strcmp(argv[i], "--bg-thresh") == 0 ||
                 strcmp(argv[i], "-bt") == 0) {
        args.background_thresh = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--fg-percent") == 0 ||
                 strcmp(argv[i], "-fp") == 0) {
        args.foreground_percent = atof(argv[i + 1]);
        ++i;
      //} else if (strcmp(argv[i], "--min-branch-length") == 0) {
        //args.min_branch_length = atoi(argv[i + 1]);
        //++i;
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
        }
      } else if (strcmp(argv[i], "--expand-window") == 0) {
        if ((i + 1 < argc) && (argv[i + 1][0] != '-')) {
          args.expand_window_um = atof(argv[i + 1]);
          ++i;
        }
      } else if (strcmp(argv[i], "--open-steps") == 0) {
        args.open_steps = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--close-steps") == 0) {
        auto val = atoi(argv[i + 1]);
        args.close_steps = val < 1 ? 1 : val; // must be at least 1
        ++i;
      } else if (strcmp(argv[i], "--preserve-topology") == 0) {
        args.close_topology = false;
      } else if (strcmp(argv[i], "--order") == 0) {
        args.morphological_operations_order = atoi(argv[i + 1]);
        ++i;
        if (args.morphological_operations_order < 1 ||
            args.morphological_operations_order > 5) {
          std::cerr << "--order should be between 1 to 5!\n";
          exit(1);
        }
      } else if (strcmp(argv[i], "--min-radius") == 0) {
        args.min_radius_um = atof(argv[i + 1]);
        ++i;
        if (args.min_radius_um < 0) {
          std::cerr << "--min-radius should be a positive float!\n";
          exit(1);
        }
      } else if (strcmp(argv[i], "--max-radius") == 0) {
        args.max_radius_um = atof(argv[i + 1]);
        ++i;
        if (args.max_radius_um <= 0) {
          std::cerr << "--max-radius should be a positive float!\n";
          exit(1);
        }
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
      } else if (strcmp(argv[i], "--disable-swc-scaling") == 0) {
        args.disable_swc_scaling = true;
      } else if (strcmp(argv[i], "--upsample-z") == 0) {
        args.upsample_z = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--test") == 0) {
        args.test = std::filesystem::canonical(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--run-app2") == 0) {
        args.run_app2 = true;
      } else if (strcmp(argv[i], "--timeout") == 0) {
        args.timeout = 60 * atoi(argv[i + 1]);
        ++i;
      } else {
        std::cerr << "unknown option \"" << argv[i] << "\"  ...exiting\n\n";
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
