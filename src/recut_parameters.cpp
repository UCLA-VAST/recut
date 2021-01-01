#ifdef USE_MCP3D
#include <image/mcp3d_image_utils.hpp>
#endif
#include "recut_parameters.hpp"

using std::cout;
using std::endl;
using std::exception;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

string RecutParameters::MetaString() {
  stringstream meta_stream;
  meta_stream << "# foreground_percent = " << foreground_percent_ << endl;
  meta_stream << "# background_thresh = " << background_thresh_ << endl;
  meta_stream << "# max_int = " << max_intensity_ << endl;
  meta_stream << "# min_int = " << min_intensity_ << endl;
  meta_stream << "# length_thresh = " << length_thresh_ << endl;
  meta_stream << "# SR_ratio = " << sr_ratio_ << endl;
  meta_stream << "# gsdt = " << gsdt_ << endl;
  meta_stream << "# restart = " << restart_ << endl;
  meta_stream << "# parallel = " << parallel_num_ << endl;
  meta_stream << "# allow_gap = " << allow_gap_ << endl;
  meta_stream << "# cnn_type = " << cnn_type_ << endl;
  meta_stream << "# radius_from_2d = " << radius_from_2d_ << endl;
  meta_stream << "# swc_resample = " << swc_resample_ << endl;
  meta_stream << "# cube_256 = " << cube_256_ << endl;
  meta_stream << "# marker_file_path = "
              << (marker_file_path_.empty() ? "none" : marker_file_path_)
              << endl;
  return meta_stream.str();
}

void RecutCommandLineArgs::PrintUsage() {
  cout << "Usage : recut <image_root_dir> <channel> [-inmarker <marker_dir>] "
          "[-outswc <swc_file>] "
          "[-resolution-level <int>] [-image-offsets <int> [<int>] [<int>]] "
          "[-image-extents <int> [<int>] [<int>]]"
          "[-bkg-thresh <int>] [-fg-percent <double>] [-gsdt] [-cnn-type "
          "<int>] [-length-thresh <double>] [-allow_gap] [-pl [<int>]]"
       << endl;
  cout << endl;
  cout << "-inmarker           [-im] input marker file directory all marker "
          "files represent known soma locations"
       << endl;
  cout << "-outswc             [-os] output tracing result, default is "
          "<imagename>_tracing.swc"
       << endl;
  cout << "-resolution-level   [-rl] resolution level to perform tracing at. "
          "default is 0, ie original resolution"
       << endl;
  cout << "-image-offsets      [-io] offsets of subvolume, in zyx order. each "
          "axis has default offset value 0 if not provided"
       << endl;
  cout << "-image-extents      [-ie] extents of subvolume, in zyx order. each "
          "axis has extent from offset to axis maximum range if not provided"
       << endl;
  cout << "-bg-thresh          [-bt] background threshold value used in GSDT "
          "and tree construction when no target marker"
       << endl;
  cout << "-max                      set the max voxel value manually such "
          "that no extra traversal of image is done"
       << endl;
  cout << "-min                      set the min voxel value manually such "
          "that no extra traversal of image is done"
       << endl;
  cout << "-fg-percent         [-fp] default 0.01, percent of voxels to be "
          "considered foreground. overrides -bg-thresh"
       << endl;
  cout << "-length-thresh      [-lt] default 1.0, the length threshold value "
          "for hierarchy pruning(hp)"
       << endl;
  cout << "-sr-ratio           [-sr] default 1/3, signal/redundancy ratio "
          "threshold"
       << endl;
  cout << "-gsdt               [-gs] perform GSDT for original image" << endl;
  cout << "-cnn-type           [-ct] default 2. connection type 1 for 6 "
          "neighbors, 2 for 18 neighbors, 3 for 26 neighbors"
       << endl;
  cout << "-allow_gap          [-ag] accept one background point between "
          "foreground during tree construction when only one marker"
       << endl;
  cout << "-prune              [-pr] prune 0 false, 1 true; defaults to 1 "
          "(automatically prunes)"
       << endl;
  cout << "-parallel           [-pl] parallelize with specified # of threads; "
          "defaults to max threads, otherwise -pl THREAD-NUM is used"
       << endl;
  cout << "-interval_size      [-is] interval size for in memory size to "
          "parallelize ; defaults to interval cubes of edge length 1024"
       << endl;
  cout << "-block_size         [-bs] block size for thread level parallelize "
          "option; defaults to cubes of edge length 32"
       << endl;
  cout << "-restart            [-rs] enforce parallel restarts; default is "
          "false, if true with no number passed restart factor defaults to 4"
       << endl;
  cout << endl;
}

string RecutCommandLineArgs::MetaString() {
  stringstream meta_stream;
  meta_stream << "# image root dir = " << image_root_dir_ << endl;
  meta_stream << "# channel = " << channel_ << endl;
  meta_stream << "# resolution level = " << resolution_level_ << endl;
  meta_stream << "# offsets (zyx) = " << image_offsets[0] << " "
              << image_offsets[1] << " " << image_offsets[2] << endl;
  meta_stream << "# extents (zyx) = " << image_extents[0] << " "
              << image_extents[1] << " " << image_extents[2] << endl;
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
    args.set_channel(atoi(argv[2]));
    // if the switch is given, parameter(s) corresponding to the switch is
    // expected
    for (int i = 3; i < argc; ++i) {
      // subvolume selection arguments
      if (strcmp(argv[i], "-resolution-level") == 0 ||
          strcmp(argv[i], "-rl") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -resolution-level")
        args.set_resolution_level(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-image-offsets") == 0 ||
                 strcmp(argv[i], "-io") == 0) {
        vector<int> offsets;
        for (int j = 0; j < 3; ++j) {
          // if (i + 1 >= argc || argv[i + 1][0] == '-')
          // MCP3D_INVALID_ARGUMENT("missing parameter(s) for -image-offsets")
          int offset = atoi(argv[i + 1]);
          offsets.push_back(offset);
          ++i;
        }
        args.set_image_offsets(offsets);
      } else if (strcmp(argv[i], "-image-extents") == 0 ||
                 strcmp(argv[i], "-ie") == 0) {
        vector<int> extents;
        for (int j = 0; j < 3; ++j) {
          // if (i + 1 >= argc || argv[i + 1][0] == '-')
          // MCP3D_INVALID_ARGUMENT("missing parameter for -image-extents")
          int extent = atoi(argv[i + 1]);
          extents.push_back(extent);
          ++i;
        }
        args.set_image_extents(extents);
      } else if (strcmp(argv[i], "-outswc") == 0 ||
                 strcmp(argv[i], "-os") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -outswc")
        args.set_swc_path(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "-inmarker") == 0 ||
                 strcmp(argv[i], "-im") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -inmarker")
        args.recut_parameters().set_marker_file_path(argv[i + 1]);
        ++i;
      } else if (strcmp(argv[i], "-bg-thresh") == 0 ||
                 strcmp(argv[i], "-bt") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -bg-thresh")
        args.recut_parameters().set_background_thresh(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-min") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -min")
        args.recut_parameters().set_min_intensity(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-max") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -max")
        args.recut_parameters().set_max_intensity(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-fg-percent") == 0 ||
                 strcmp(argv[i], "-fp") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -fg-percent")
        args.recut_parameters().set_foreground_percent(atof(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-length-thresh") == 0 ||
                 strcmp(argv[i], "-lt") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -length-thresh")
        args.recut_parameters().set_length_thresh(atof(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-sr-ratio") == 0 ||
                 strcmp(argv[i], "-sr") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -sr-ratio")
        args.recut_parameters().set_sr_ratio(atof(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-prune") == 0 ||
                 strcmp(argv[i], "-pr") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -prune")
        args.recut_parameters().set_prune(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-cnn-type") == 0 ||
                 strcmp(argv[i], "-ct") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -cnn-type")
        args.recut_parameters().set_cnn_type(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-parallel") == 0 ||
                 strcmp(argv[i], "-pl") == 0) {
        int max_threads = 1;
#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
        max_threads = omp_get_max_threads();
#endif
        int current_threads = max_threads;
        cout << "max threads available to CPU = " << max_threads << endl;
        ++i;
        if (!(i >= argc || argv[i][0] == '-')) {
          args.recut_parameters().set_parallel_num(atoi(argv[i]));
          current_threads = atoi(argv[i]);
        } else {
          args.recut_parameters().set_parallel_num(max_threads);
        }
        cout << "using total threads = " << current_threads << endl;
#if defined USE_OMP_BLOCK || defined USE_OMP_INTERVAL
        omp_set_num_threads(current_threads);
#endif
      } else if (strcmp(argv[i], "-interval_size") == 0 ||
                 strcmp(argv[i], "-is") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -interval_size")
        args.recut_parameters().set_interval_size(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-block_size") == 0 ||
                 strcmp(argv[i], "-bs") == 0) {
        // if (i + 1 >= argc || argv[i + 1][0] == '-')
        // MCP3D_INVALID_ARGUMENT("missing parameter for -block_size")
        args.recut_parameters().set_block_size(atoi(argv[i + 1]));
        ++i;
      } else if (strcmp(argv[i], "-restart") == 0 ||
                 strcmp(argv[i], "-rs") == 0) {
        args.recut_parameters().set_restart(true);
        args.recut_parameters().set_restart_factor(4.0);
        if (!(i + 1 >= argc || argv[i + 1][0] == '-')) {
          args.recut_parameters().set_restart_factor(atof(argv[i + 1]));
          if (atof(argv[i + 1]) <=
              0.00000001) { // parse double has issues with 0
            args.recut_parameters().set_restart(false);
          }
          ++i;
        }
      } else if (strcmp(argv[i], "-gsdt") == 0 || strcmp(argv[i], "-gs") == 0)
        args.recut_parameters().set_gsdt(true);
      else if (strcmp(argv[i], "-allow-gap") == 0 ||
               strcmp(argv[i], "-ag") == 0)
        args.recut_parameters().set_allow_gap(true);
      else
        cout << "unknown option " << argv[i] << endl;
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
    cout << e.what() << endl;
    // MCP3D_MESSAGE("invalid command line arguments. neuron tracing not
    // performed")
    RecutCommandLineArgs::PrintUsage();
    return false;
  }
}
