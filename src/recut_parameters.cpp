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
  meta_stream << "# max_int = " << max_intensity_ << '\n';
  meta_stream << "# min_int = " << min_intensity_ << '\n';
  //meta_stream << "# restart = " << restart_ << '\n';
  meta_stream << "# parallel = " << parallel_num_ << '\n';
  meta_stream << "# interval_size = " << interval_size_ << '\n';
  meta_stream << "# block_size = " << block_size_ << '\n';
  meta_stream << "# marker_file_path = "
              << (marker_file_path_.empty() ? "none" : marker_file_path_)
              << '\n';
  return meta_stream.str();
}

void RecutCommandLineArgs::PrintUsage() {
  cout << "Basic usage : recut <image_root_dir> <marker_dir> [--channel <int>] "
          "[--outswc <swc_file>] "
          "[--resolution-level <int>] [--image-offsets <int> [<int>] [<int>]] "
          "[--image-extents <int> [<int>] [<int>]] "
          "[--bkg-thresh <int>] [--fg-percent <double>]\n\n";

  cout << "<image_root_dir>     directory for input image\n";
  cout << "<marker_dir>         directory containing all marker "
          "files which represent known soma locations\n";
  cout << "--max                set max image voxel raw value allowed, "
          "computed automatically when --bg_thresh or --fg-percent are specified\n";
  cout << "--min                set min image voxel raw value allowed, "
          "computed automatically when --bg_thresh or --fg-percent are specified\n";
  cout << "--channel            [-c] channel number of image default 0\n";
  cout << "--outswc             [-os] output tracing result default is "
          "<imagename>_tracing.swc\n";
  cout << "--resolution-level   [-rl] resolution level to perform tracing at. "
          "default is 0, ie original resolution\n";
  cout << "--image-offsets      [-io] offsets of subvolume, in z y x order default 0 0 0\n";
  cout << "--image-extents      [-ie] extents of subvolume, in z y x order defaults"
          " to max range from offset start to max length in each axis\n";
  cout << "--bg-thresh          [-bt] background threshold value desired\n";
  cout << "--fg-percent         [-fp] default 0.01, percent of voxels to be "
          "considered foreground. overrides --bg-thresh\n";
  cout << "--prune              [-pr] prune 0 false, 1 true; defaults to 1 "
          "(automatically prunes)\n";
  cout << "--parallel           [-pl] thread count "
          "defaults to max hardware threads\n";
  cout << "--interval-size      [-is] interval size length "
          "defaults to interval cubes of edge length 1024\n";
  cout << "--block-size         [-bs] block size length "
          "defaults to block cubes of edge length 64\n";
  //cout << "--restart            [-rs] enforce parallel restarts default: "
          //"false, if true with no number passed restart factor defaults to 4\n\n";
}

string RecutCommandLineArgs::MetaString() {
  stringstream meta_stream;
  meta_stream << "# image root dir = " << image_root_dir_ << '\n';
  meta_stream << "# channel = " << channel_ << '\n';
  meta_stream << "# resolution level = " << resolution_level_ << '\n';
  meta_stream << "# offsets (zyx) = " << image_offsets[0] << " "
              << image_offsets[1] << " " << image_offsets[2] << '\n';
  meta_stream << "# extents (zyx) = " << image_extents[0] << " "
              << image_extents[1] << " " << image_extents[2] << '\n';
  meta_stream << recut_parameters_.MetaString();
  return meta_stream.str();
}

bool ParseRecutArgs(int argc, char *argv[], RecutCommandLineArgs &args) {
  if (argc < 3) {
    RecutCommandLineArgs::PrintUsage();
    return false;
  }
  try {
    // global volume and channel selection
    args.set_image_root_dir(argv[1]);
    args.recut_parameters().set_marker_file_path(argv[2]);
    // if the switch is given, parameter(s) corresponding to the switch is
    // expected
    for (int i = 3; i < argc; ++i) {
      // subvolume selection arguments
      if (strcmp(argv[i], "--resolution-level") == 0 ||
          strcmp(argv[i], "-rl") == 0) {
        args.set_resolution_level(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--image-offsets") == 0 ||
                 strcmp(argv[i], "-io") == 0) {
        vector<int> offsets;
        for (int j = 0; j < 3; ++j) {
          int offset = atoi(argv[i + 1]);
          offsets.push_back(offset);
          ++i;
        }
        args.set_image_offsets(offsets);
      } else if (strcmp(argv[i], "--image-extents") == 0 ||
                 strcmp(argv[i], "-ie") == 0) {
        vector<int> extents;
        for (int j = 0; j < 3; ++j) {
          int extent = atoi(argv[i + 1]);
          extents.push_back(extent);
          ++i;
        }
        args.set_image_extents(extents);
      } else if (strcmp(argv[i], "--outswc") == 0 ||
                 strcmp(argv[i], "-os") == 0) {
        args.set_swc_path(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "--channel") == 0 ||
                 strcmp(argv[i], "-c") == 0) {
        args.set_channel(atoi(argv[i + 1]));
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
      } else if (strcmp(argv[i], "--length-thresh") == 0 ||
                 strcmp(argv[i], "-lt") == 0) {
        args.recut_parameters().set_length_thresh(atof(argv[i + 1]));
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
        int max_threads = 1;
#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
        max_threads = omp_get_max_threads();
#endif
        int current_threads = max_threads;
        cout << "max threads available to CPU = " << max_threads << '\n';
        ++i;
        if (!(i >= argc || argv[i][0] == '-')) {
          args.recut_parameters().set_parallel_num(atoi(argv[i]));
          current_threads = atoi(argv[i]);
        } else {
          args.recut_parameters().set_parallel_num(max_threads);
        }
        cout << "using total threads = " << current_threads << '\n';
#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
        omp_set_num_threads(current_threads);
#endif
      } else if (strcmp(argv[i], "--interval-size") == 0 ||
                 strcmp(argv[i], "-is") == 0) {
        args.recut_parameters().set_interval_size(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "--block-size") == 0 ||
                 strcmp(argv[i], "-bs") == 0) {
        args.recut_parameters().set_block_size(atoi(argv[i + 1]));
        ++i;
      //} else if (strcmp(argv[i], "--restart") == 0 ||
                 //strcmp(argv[i], "-rs") == 0) {
        //args.recut_parameters().set_restart(true);
        //args.recut_parameters().set_restart_factor(4.0);
        //if (!(i + 1 >= argc || argv[i + 1][0] == '-')) {
          //args.recut_parameters().set_restart_factor(atof(argv[i + 1]));
          //if (atof(argv[i + 1]) <=
              //0.00000001) { // parse double has issues with 0
            //args.recut_parameters().set_restart(false);
          //}
          //++i;
        //}
      } else if (strcmp(argv[i], "--gsdt") == 0 || strcmp(argv[i], "-gs") == 0) {
        args.recut_parameters().set_gsdt(true);
      } else if (strcmp(argv[i], "--allow-gap") == 0 ||
                 strcmp(argv[i], "--ag") == 0) {
        args.recut_parameters().set_allow_gap(true);
      } else {
        cout << "unknown option \"" << argv[i] << "\"  ...exiting\n\n";
        RecutCommandLineArgs::PrintUsage();
        exit(1);
      }
    }
    // give default swc path if not given
    if (args.swc_path().empty()) {
      string z_start = to_string(args.image_offsets[0]),
             y_start = to_string(args.image_offsets[1]),
             x_start = to_string(args.image_offsets[2]);
      string z_end = to_string(args.image_offsets[0] + args.image_extents[0]),
             y_end = to_string(args.image_offsets[1] + args.image_extents[1]),
             x_end = to_string(args.image_offsets[2] + args.image_extents[2]);
      args.set_swc_path(args.image_root_dir() + "tracing_z" + z_start + "_" +
                        z_end + "_y" + y_start + "_" + y_end + "_x" + x_start +
                        "_" + x_end + ".swc");
    }
    // if neither background threshold nor foreground percent given, set
    // foreground percent to 0.01
    if (args.recut_parameters().background_thresh() < 0 &&
        args.recut_parameters().foreground_percent() < 0)
      args.recut_parameters().set_foreground_percent(0.01);
    return true;
  } catch (const exception &e) {
    cout << e.what() << '\n';
    RecutCommandLineArgs::PrintUsage();
    return false;
  }
}
