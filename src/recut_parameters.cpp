#ifdef USE_MCP3D
#include <image/mcp3d_image_utils.hpp>
#endif
#include "recut_parameters.hpp"

using std::cout;
using std::exception;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

string RecutParameters::MetaString() {
  stringstream meta_stream;
  meta_stream << "# foreground_percent = " << foreground_percent_ << '\n';
  meta_stream << "# background_thresh = " << background_thresh_ << '\n';
  meta_stream << "# parallel = " << parallel_num_ << '\n';
  meta_stream << "# seeds directory = "
              << (marker_file_path_.empty() ? "none" : marker_file_path_)
              << '\n';
  return meta_stream.str();
}

void RecutCommandLineArgs::PrintUsage() {
  cout << "Basic usage : recut <image_root_dir> [--seeds <marker_dir>] "
          "[--type point/uint8/mask/float] "
          "[--convert <output_vdb_file_name>] "
          "[--bkg-thresh <int>] [--fg-percent <double>]\n\n";
          "[--image-offsets <int> [<int>] [<int>]] "
          "[--image-lengths <int> [<int>] [<int>]] "
          "\nNote: neurite+soma images are binarized and do not need bkg-thresh or fg-percent specified";
        // "[--channel <dir>] "
          //"[--outswc <swc_file>] "
          // "[--resolution-level <int>] "

  cout << "<image_root_dir>     directory for input image\n";
  cout << "--seeds              directory of files which represent known root/soma locations\n";
  cout << "--convert            [-cv] convert image file and exit defaults to "
          "out.vdb\n";
  cout << "--type               VDB input grid type: 'point', 'uint8', 'mask' or 'float'\n";
  cout << "--prune-radius       larger values decrease node sampling density "
          "along paths, default 5 the z anisotropic factor\n";
  //cout << "--max                set max image voxel raw value allowed, "
          //"computed automatically when --bg_thresh or --fg-percent are "
          //"specified\n";
  //cout << "--min                set min image voxel raw value allowed, "
          //"computed automatically when --bg_thresh or --fg-percent are "
          //"specified\n";
  //cout << "--channel            [-c] directory of channel image default ch0\n";
  //cout << "--resolution-level   [-rl] resolution level to perform tracing at. "
          //"default is 0, ie original resolution\n";
  cout << "--image-offsets      [-io] offsets of subvolume, in x y z order "
          "default 0 0 0\n";
  cout << "--image-lengths      [-ie] lengths of subvolume, in x y z order "
          "defaults"
          " to max range from offset start to max length in each axis (-1, -1, "
          "-1)\n";
  cout << "--bg-thresh          [-bt] background threshold value desired\n";
  cout << "--min-branch-length  prune leaf branches lower, defaults to 20\n";
  cout << "--fg-percent         [-fp] auto calculate a bg-thresh closest to a "
          "foreground \% between (0-100], overriding any --bg-thresh args. "
          "Value of .08 yields ~8 in 10,000 voxels "
          "as foreground per z-plane\n";
  //cout << "--prune              [-pr] prune 0 false, 1 true; defaults to 1 "
          //"(automatically prunes)\n";
  cout << "--parallel           [-pl] thread count defaults to max hardware "
          "threads\n";
  cout << "--output-windows     specify uint8 vdb file for which to create "
          "windows surrounding each neuron cluster/component\n";
  cout << "--interval-z         z-depth of fg percentages and conversion, defaults to 1\n"; 
  cout << "--downsample-factor  for images scaled down in x and z dimension "
          "scale the marker files by specified factor\n";
  cout << "--upsample-z         during --convert only z-dimension will be "
          "upsampled (copied) by specified factor, default is 1 i.e. no "
          "upsampling\n";
  cout << "--run-app2           for benchmarks and comparisons runs app2 on the vdb passed to --output-windows\n";
  cout << "--help               [-h] print this example usage summary\n";
}

string RecutCommandLineArgs::MetaString() {
  stringstream meta_stream;
  meta_stream << "# image root dir = " << image_root_dir_ << '\n';
  //meta_stream << "# channel = " << channel_ << '\n';
  meta_stream << "prune radius = " << channel_ << '\n';
  //meta_stream << "# resolution level = " << resolution_level_ << '\n';
  meta_stream << "# offsets (xyz) = " << image_offsets[0] << " "
              << image_offsets[1] << " " << image_offsets[2] << '\n';
  meta_stream << "# lengths (xyz) = " << image_lengths[0] << " "
              << image_lengths[1] << " " << image_lengths[2] << '\n';
  meta_stream << recut_parameters_.MetaString();
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
      args.set_image_root_dir(argv[1]);
    }
    // if the switch is given, parameter(s) corresponding to the switch is
    // expected
    for (int i = 2; i < argc; ++i) {
      if (strcmp(argv[i], "--convert") == 0 || strcmp(argv[i], "-cv") == 0) {
        ++i;
        if (!(i >= argc || argv[i][0] == '-')) {
          args.recut_parameters().set_out_vdb(argv[i]);
        } else {
          // still convert but set to default file name
          // only vdb is supported
          args.recut_parameters().set_out_vdb("out.vdb");
        }
        args.recut_parameters().set_convert_only(true);
      } else if (strcmp(argv[i], "--seeds") == 0 ||
                 strcmp(argv[i], "-s") == 0) {
        args.recut_parameters().set_marker_file_path(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--resolution-level") == 0 ||
                 strcmp(argv[i], "-rl") == 0) {
        args.set_resolution_level(atoi(argv[i + 1]));
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
      } else if (strcmp(argv[i], "--type") == 0) {
        auto arg = std::string(argv[i + 1]);
        if (arg == "float" || arg == "point" || arg == "uint8" ||
            arg == "mask") {
          args.set_type(argv[i + 1]);
        } else {
          cerr << "--type option must be one of [float,point]\n";
          exit(1);
        }
        ++i;
      } else if (strcmp(argv[i], "--channel") == 0 ||
                 strcmp(argv[i], "-c") == 0) {
        args.set_channel(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--prune-radius") == 0) {
        args.set_prune_radius(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--bg-thresh") == 0 ||
                 strcmp(argv[i], "-bt") == 0) {
        args.recut_parameters().set_background_thresh(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--min") == 0) {
        args.recut_parameters().set_min_intensity(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--max") == 0) {
        args.recut_parameters().set_max_intensity(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--fg-percent") == 0 ||
                 strcmp(argv[i], "-fp") == 0) {
        args.recut_parameters().set_foreground_percent(atof(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--min-branch-length") == 0) {
        args.min_branch_length = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--interval-z") == 0) {
        args.interval_z = atoi(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--sr-ratio") == 0 ||
                 strcmp(argv[i], "-sr") == 0) {
        args.recut_parameters().set_sr_ratio(atof(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--prune") == 0 ||
                 strcmp(argv[i], "-pr") == 0) {
        args.recut_parameters().set_prune(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--cnn-type") == 0 ||
                 strcmp(argv[i], "-ct") == 0) {
        args.recut_parameters().set_cnn_type(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--parallel") == 0 ||
                 strcmp(argv[i], "-pl") == 0) {
        args.user_thread_count = atoi(argv[i+1]);
        ++i;
      } else if (strcmp(argv[i], "--downsample-factor") == 0) {
        args.recut_parameters().set_downsample_factor(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--gsdt") == 0 ||
                 strcmp(argv[i], "-gs") == 0) {
        args.recut_parameters().set_gsdt(true);
      } else if (strcmp(argv[i], "--allow-gap") == 0 ||
                 strcmp(argv[i], "--ag") == 0) {
        args.recut_parameters().set_allow_gap(true);
      } else if (strcmp(argv[i], "--combine") == 0) {
        args.recut_parameters().set_combine(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--histogram") == 0) {
        args.recut_parameters().set_histogram(true);
      } else if (strcmp(argv[i], "--output-windows") == 0) {
        args.recut_parameters().set_output_windows(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--upsample-z") == 0) {
        args.recut_parameters().set_upsample_z(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--run-app2") == 0) {
        args.run_app2 = true;
        ++i;
      } else {
        cout << "unknown option \"" << argv[i] << "\"  ...exiting\n\n";
        RecutCommandLineArgs::PrintUsage();
        exit(1);
      }
    }
    return args;
  } catch (const exception &e) {
    cout << e.what() << '\n';
    RecutCommandLineArgs::PrintUsage();
    exit(1);
  }
}
