#pragma once

#include "utils.hpp"
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/FastSweeping.h> //maskSDF
#include <openvdb/tools/TopologyToLevelSet.h>
#include <tbb/parallel_for_each.h>

// adds all markers to seeds
// recut operates in voxel units (image space) therefore whenever marker/node
// information is input into recut it converts it from um units into voxel
// units marker_base: some programs like APP2 consider input markers should be
// offset by 1 recut operates with 0 offset of input markers and SWC nodes
std::vector<Seed> process_marker_dir(
    std::string seed_path,
    std::array<float, 3> voxel_size = std::array<float, 3>{1, 1, 1},
    int marker_base = 0) {

  // input handler
  if (seed_path.empty())
    return {};

  auto min_voxel_size = min_max(voxel_size).first;

  // gather all markers within directory
  auto seeds = std::vector<Seed>();

  rng::for_each(
      fs::directory_iterator(seed_path), [&](const auto &marker_file) {
        if (!fs::is_directory(marker_file)) {
          auto markers =
              readMarker_file(fs::absolute(marker_file), marker_base);
          assertm(markers.size() == 1, "only 1 marker file per soma");
          auto marker = markers[0];

          std::string fn = marker_file.path().filename().string();
          auto numbers =
              fn | rv::split('_') | rng::to<std::vector<std::string>>();
          if (numbers.size() != 5)
            throw std::runtime_error(
                "Marker file names must be in format marker_x_y_z_volume");
          //  volume is the last number of the file name
          uint64_t volume = std::stoull(numbers.back());

          if (marker.radius == 0) {
            marker.radius =
                static_cast<uint8_t>(std::cbrt(volume) / (4 / 3 * PI) + 0.5);
          }

          // ones() + GridCoord(marker.x / args->downsample_factor,
          // marker.y / args->downsample_factor,
          // upsample_idx(args->upsample_z, marker.z)),

          // convert from world space (um) to image space (pixels)
          // these are the offsets around the coordinate to keep
          seeds.emplace_back(
              GridCoord(std::round(marker.x / voxel_size[0]),
                        std::round(marker.y / voxel_size[1]),
                        std::round(marker.z / voxel_size[2])),
              static_cast<uint8_t>(marker.radius / min_voxel_size + 0.5),
              marker.radius, volume);
        }
      });

#ifdef LOG
  std::cout << '\t' << seeds.size() << " seeds found in directory\n";
#endif

  return seeds;
}

openvdb::FloatGrid::Ptr create_seed_sphere_grid(std::vector<Seed> seeds) {
  auto timer = high_resolution_timer();

  auto component_grids = seeds | rv::transform([](const Seed &seed) {
                           int voxel_size = 1;
                           return vto::createLevelSetSphere<openvdb::FloatGrid>(
                               seed.radius, seed.coord.asVec3s(), voxel_size,
                               RECUT_LEVEL_SET_HALF_WIDTH);
                         }) |
                         rng::to_vector;
#ifdef LOG
  std::cout << "\tFinished create seed spheres in " << timer.elapsed() << '\n';
#endif
  timer.restart();

  // TODO replace with sumMergeOp which is parallel and more efficient
  auto merged = merge_grids(component_grids);
#ifdef LOG
  std::cout << "\tFinished sphere merge in " << timer.elapsed() << '\n';
#endif
  return merged;
}

// establish highest performance most basic filter for morphological
// operations like dilation, erosion, opening, closing
// Of all connected components, keep those whose central coordinate is an
// active voxel in the point topology and estimate the radius given the
// bbox of the component. If passing known seeds, then filter all
// components to only those components which contain the passed seed
// If you already filtered the grid with seed intersection there's no need
// to refilter by known seeds
auto create_seed_pairs = [](std::vector<openvdb::FloatGrid::Ptr> components,
                            openvdb::FloatGrid::Ptr topology_sdf,
                            std::array<float, 3> voxel_size,
                            float min_radius_um, float max_radius_um,
                            std::string output_type,
                            std::vector<Seed> known_seeds = {}) {
  std::vector<Seed> seeds;
  std::vector<openvdb::FloatGrid::Ptr> filtered_components;
  auto removed_by_inactivity = 0;
  auto removed_by_known_seeds = 0;
  auto removed_by_incorrect_sphere = 0;
  auto removed_by_radii = 0;
  auto max_voxel_size = min_max(voxel_size).second;
  for (auto component : components) {
    openvdb::Vec4s sphere;
    std::vector<openvdb::Vec4s> spheres;
    // make a temporary copy to possibly mutate
    // auto dilated_sdf = component->deepCopy();
    // establish the filter for opening
    auto filter = create_morph_filter(component);
    if (component->activeVoxelCount() < 1) {
      ++removed_by_incorrect_sphere;
      continue;
    }

    // std::cout << "\tstarting sdf count " << dilated_sdf->activeVoxelCount()
    //<< '\n';
    // for (int i=0; (i < 5) && (spheres.size() < 1) &&
    // (dilated_sdf->activeVoxelCount() >=1); ++i) {
    //  it's possible to force this function to return spheres with a
    //  certain range of radii, but we'd rather see what the raw radii
    //  it returns for now and let the min and max radii filter them
    vto::fillWithSpheres(*component, spheres,
                         /* min, max total count of spheres allowed */ {1, 1},
                         /* overlapping*/ false);

    // dilate
    // filter->offset(-1);
    // std::cout << "\tdilated sdf count " << dilated_sdf->activeVoxelCount()
    //<< '\n';
    //}

    if (spheres.size() < 1) {
      ++removed_by_incorrect_sphere;
      continue;
    } else if (spheres.size() == 1) {
      sphere = spheres[0];
    } else { // get the sphere with max radii
      sphere = *std::max_element(spheres.begin(), spheres.end(),
                                 [](openvdb::Vec4s &fst, openvdb::Vec4s &snd) {
                                   return fst[3] < snd[3];
                                 });
    }

    auto coord_center = GridCoord(sphere[0], sphere[1], sphere[2]);
    auto radius_voxels = sphere[3];

    // auto seed = sdf_to_seed(component);
    // if (!seed) {
    //++removed_by_incorrect_sphere;
    // continue;
    //}
    // auto [coord_center, radius_voxels] = *seed;
    // std::cout << "radius voxels " << radius_voxels << '\n';

    if (topology_sdf->tree().isValueOn(coord_center)) {
      if (!known_seeds.empty()) {
        // convert to fog so that isValueOn returns whether it is
        // within the
        vto::sdfToFogVolume(*component);
        // component if no known seed is an active voxel in this
        // component then remove this component
        if (rng::none_of(known_seeds, [&component](const auto &known_seed) {
              return component->tree().isValueOn(known_seed.coord);
            })) {
          ++removed_by_known_seeds;
          continue;
        }
      }
    } else {
      ++removed_by_inactivity;
      continue;
    }

    // the radius is calculated by the distance of the center point
    // to the nearest surface point voxel sizes can be anisotropic,
    // and in such cases the radii is likely set by the distance (in
    // voxels) along the lowest resolution (the largest voxel length)
    // dimension Example for a voxel size of [.2, .2, 1], if the
    // radius returned was 3, it would be the actual radii_um is 3 um
    // along the z-dimension therefore scale with the voxel dimension
    // with the largest length
    auto radius_um = radius_voxels * std::pow(max_voxel_size, 3);
    if (radius_um > max_radius_um || !radius_um)
      std::cout << radius_um << " ";
    if (min_radius_um <= radius_um && radius_um <= max_radius_um) {
      // round to the nearest 8-bit unsigned integer between 0 and 255
      auto radius = static_cast<uint8_t>(radius_voxels + 0.5);
      radius = radius < 1 ? 1 : radius; // clamp to at least 1
      seeds.emplace_back(coord_center, radius, radius_um,
                         adjust_volume_by_voxel_size(
                             component->activeVoxelCount(), voxel_size));
      filtered_components.push_back(component);
    } else {
      ++removed_by_radii;
    }
  }
#ifdef LOG
  std::cout << "\tseeds removed by radii min and max criteria "
            << removed_by_radii << '\n';
  if (removed_by_inactivity)
    std::cerr << "\tWarning: seeds removed by inactivity "
              << removed_by_inactivity << '\n';
  if (removed_by_known_seeds)
    std::cerr << "\tWarning: seeds removed by known seeds "
              << removed_by_known_seeds << '\n';
  if (removed_by_incorrect_sphere)
    std::cerr << "\tWarning: seeds removed by incorrect sphere "
              << removed_by_incorrect_sphere << '\n';
#endif
  return std::make_pair(seeds, components);
};

// define the filter for the morphological operations to the level set
// morpho operations can only occur on SDF level sets, not on FOG topologies
// the morphological ops with filter below mutate sdf_grid in place
// erode then dilate --> morphological opening
// dilate then erode --> morphological closing
// negative offset means dilate
// positive offset means erode
std::unique_ptr<vto::LevelSetFilter<openvdb::FloatGrid>>
create_filter(openvdb::FloatGrid::Ptr sdf_grid,
              int morphological_operations_order) {
  auto filter =
      std::make_unique<vto::LevelSetFilter<openvdb::FloatGrid>>(*sdf_grid);

  switch (morphological_operations_order) {
  case 1:
    std::cout << "\t1st order morphological operations\n";
    filter->setSpatialScheme(openvdb::math::FIRST_BIAS);
    break;
  case 2:
    std::cout << "\t2nd order morphological operations\n";
    filter->setSpatialScheme(openvdb::math::SECOND_BIAS);
    break;
  case 3:
    std::cout << "\t3rd order morphological operations\n";
    filter->setSpatialScheme(openvdb::math::THIRD_BIAS);
    break;
  case 4:
    std::cout << "\t4th order morphological operations\n";
    filter->setSpatialScheme(openvdb::math::WENO5_BIAS);
    break;
  case 5:
    std::cout << "\t5th order morphological operations\n";
    filter->setSpatialScheme(openvdb::math::HJWENO5_BIAS);
    break;
  default:
    std::cout << "\tunexpected value for argument --order "
              << morphological_operations_order << "\n"
              << "\t1st order morphological operations\n";
    filter->setSpatialScheme(openvdb::math::FIRST_BIAS);
  }
  filter->setTemporalScheme(openvdb::math::TVD_RK1);
  return filter;
}

// replace the original grid, with a grid only containing the soma component
// this prevents the soma labels and their grid from contain other components
template <typename GridT>
std::optional<std::pair<GridT, CoordBBox>>
find_soma_component(Seed seed, GridT grid,
                    openvdb::FloatGrid::Ptr keep_if_empty_grid = nullptr,
                    int channel = 0) {

  // build the bounding box:
  // center the window around the center of the soma and give it a uniform width
  // as all other crops/windows
  auto offset = GridCoord(SOMA_LABEL_LENGTH / 2);
  auto bbox = CoordBBox(seed.coord - offset, seed.coord + offset);
  vb::BBoxd clipBox(bbox.min().asVec3d(), bbox.max().asVec3d());

  if (!grid) {
    return std::nullopt; // do nothing
  }

  // if the surface SDF (keep_if_empty_grid) was passed then only write
  // windows where it is empty, this is vital for visualizing the problem
  // somas the somas that get deleted for unknown reasons
  if (keep_if_empty_grid) {
    const auto output_grid = vto::clip(*keep_if_empty_grid, clipBox);
    if (output_grid->activeVoxelCount() > 0) {
      return std::nullopt; // do nothing
    }
  }

  grid = vto::clip(*grid, clipBox);

  if (!grid || (grid->activeVoxelCount() == 0)) {
    return std::nullopt; // do nothing
  }

  if (channel) {
    std::vector<GridT> window_components;
    // works on grids of arbitrary type, placing all disjoint segments
    // (components) in decreasing size order in window_components
    vto::segmentActiveVoxels(*grid, window_components);

    if (window_components.size() == 0) {
      std::cout << "\tNo window components\n";
      return std::nullopt; // do nothing
    }

    // find the component that has the seed within it
    auto known_component =
        window_components | rv::filter([&seed](auto component) {
          auto mask = vto::extractEnclosedRegion(*component);
          auto test = vto::sdfInteriorMask(*component);
          if (mask) {
            std::cout << "\tenclosed voxel count: " << mask->activeVoxelCount()
                      << " on " << mask->tree().isValueOn(seed.coord) << '\n';
          }
          if (test) {
            std::cout << "\tinterior voxel count: " << test->activeVoxelCount()
                      << " on " << test->tree().isValueOn(seed.coord) << '\n';
          }
          std::cout << '\n';
          return mask ? mask->tree().isValueOn(seed.coord) : false;
          // return component->tree().isValueOn(seed.coord);
        }) |
        rng::to_vector;

    if (known_component.size() == 0) {
      std::cout << "\tNo known component\n";
    }

    // otherwise use the largest (first) component in the window
    grid = known_component.size() ? known_component.front()
                                  : window_components.front();
  }
  return std::make_pair(grid, bbox);
}

// takes a set of seeds and their corresponding sdf/isosurface
// and writes to TIF their uint8
std::vector<openvdb::FloatGrid::Ptr>
find_soma_components(std::vector<Seed> seeds, openvdb::FloatGrid::Ptr sdf_grid,
                     int threads = 1) {
  std::vector<openvdb::FloatGrid::Ptr> temp_vec;
  temp_vec.reserve(seeds.size());

  // tbb::task_arena arena(threads);
  // arena.execute([&] {
  // tbb::parallel_for_each(
  // rng::for_each(seeds | rv::enumerate | rng::to_vector, [&](auto element) {
  for (int i = 0; i < seeds.size(); ++i) {
    // auto [index, seed] = element;
    auto seed = seeds[i];
    std::cout << "Checking " << i << '\n';
    auto opt = find_soma_component(seed, sdf_grid, nullptr, 1);
    if (opt) {
      auto [ptr, _] = opt.value();
      temp_vec[i] = ptr;
    } else {
      std::cout << "\tno window\n";
    }
  }
  //});
  //});

  // keep valid ptrs
  std::vector<openvdb::FloatGrid::Ptr> final_vec;
  for (auto ptr : temp_vec) {
    if (ptr)
      final_vec.push_back(ptr);
  }
  return final_vec;
}

// You need to convert a SDF grid in its entirety to an interior mask grid
// before cropping. If you crop an SDF before converting to a mask grid
// the inside voxels may not be enclosed and may be counted as outside
template <typename GridT>
GridT create_label(Seed seed, fs::path dir, GridT grid,
                   openvdb::FloatGrid::Ptr keep_if_empty_grid = nullptr,
                   int index = 0, bool output_vdb = true, int channel = 0,
                   bool paged = true) {

  auto opt = find_soma_component(seed, grid, keep_if_empty_grid, channel);
  if (!opt)
    return nullptr;
  auto [clipped_grid, bbox] = opt.value();
  grid = clipped_grid;

  if (grid) {
    // prepare the directory and log
    fs::create_directories(dir);
    std::ofstream runtime;
    runtime.open(dir / ("log.csv"));

    write_output_windows(grid, dir, runtime, index, output_vdb, paged, bbox,
                         channel);
    return grid;
  }

  return nullptr;
}

fs::path soma_dir(fs::path dir, Seed seed) {
  return dir / ("soma-" + std::to_string(seed.coord.x()) + '-' +
                std::to_string(seed.coord.y()) + '-' +
                std::to_string(seed.coord.z()));
}

// takes a set of seeds and their corresponding sdf/isosurface
// and writes to TIF their uint8
template <typename GridT>
void create_labels(std::vector<Seed> seeds, fs::path dir,
                   ImgGrid::Ptr image = nullptr, GridT sdf_grid = nullptr,
                   openvdb::FloatGrid::Ptr keep_if_empty_grid = nullptr,
                   int threads = 1, bool output_vdb = false,
                   bool paged = true) {

  fs::create_directories(dir);
  // if uint8 is passed to create crop windows
  if (image) {
    tbb::task_arena arena(threads);
    arena.execute([&] {
      tbb::parallel_for_each(
          seeds | rv::enumerate | rng::to_vector, [&](auto element) {
            auto [index, seed] = element;
            create_label(seed, soma_dir(dir, seed), image, keep_if_empty_grid,
                         index, output_vdb, 0, paged);
            auto sdf_soma =
                create_label(seed, soma_dir(dir, seed), sdf_grid,
                             keep_if_empty_grid, index, output_vdb, 1, paged);
          });
    });
  }

  // if (components) {
  // auto elements = rng::zip(seeds, components.value());
  // rng::for_each(element | rv::enumerate, [&](auto element) {
  // auto [seed, component, index] = element;
  // std::cout << "soma " << index << '\n';

  // auto bbox = component->evalActiveVoxelBoundingBox();
  // std::cout << '\t' << bbox << '\n';
  // std::cout << '\t' << component->activeVoxelCount << '\n';

  //// convert to mask type
  // auto mask = vto::extractEnclosedRegion(component);
  // create_label(seed, soma_dir(index), sdf_grid, keep_if_empty_grid,
  // index, output_vdb, channel, paged); write_vdb_file({mask}, dir / "soma-"
  // + index);
  //});
  //}
}

std::pair<std::vector<Seed>, openvdb::FloatGrid::Ptr>
soma_segmentation(openvdb::MaskGrid::Ptr mask_grid, RecutCommandLineArgs *args,
                  GridCoord image_lengths, fs::path log_fn, fs::path run_dir) {

  auto timer = high_resolution_timer();
  assertm(mask_grid, "Mask grid must be set before starting soma segmentation");

  // for labels output .. load the uint8 image
  ImgGrid::Ptr image;
  if (args->output_type == "labels") {
    if (args->window_grid_paths.empty()) {
      std::cerr << "--output-type labels must also pass --output-windows "
                   "uint8.vdb, exiting...\n";
      exit(1);
    }

    // you need to load the passed image grids if you are outputting windows
    auto window_grids =
        args->window_grid_paths |
        rv::transform([](const auto &gpath) { return read_vdb_file(gpath); }) |
        rng::to_vector; // force reading once now

    image = openvdb::gridPtrCast<ImgGrid>(window_grids.front());
  }

  // if (args->save_vdbs && args->input_type != "mask")
  // write_vdb_file({mask_grid}, run_dir / "mask.vdb");

  std::ofstream run_log;
  run_log.open(log_fn, std::ios::app);

// mask grids are a fog volume of sparse active values in space
// change the fog volume into an SDF by holding values on the border between
// active an inactive voxels
// the new SDF wraps (dilates by 1) the original active voxels, and
// additionally holds values across the interface of the surface
//
// this function additionally adds a morphological closing step such that
// holes and valleys in the SDF are filled
#ifdef LOG
  std::cout << "starting seed (soma) detection:\n";
  std::cout << "\tmask to sdf step\n";
#endif
  // resulting SDF is slightly modified by a closing op of 1 step which has a
  // very minimal effect this API does not allow 0 closing
  timer.restart();
  // if there is no pre-opening step (open_denoise) then do closing at the
  // time of conversion to sdf for performance gain
  auto sdf_grid = vto::topologyToLevelSet(
      *mask_grid, /*halfwidth voxels*/ RECUT_LEVEL_SET_HALF_WIDTH,
      /*closing steps*/ args->open_denoise == 0 ? args->close_steps : 0);

  // the raw image is only needed if you are not closing topology
  openvdb::FloatGrid::Ptr raw_image_sdf;
  if (!args->close_topology) {
    // get an unaltered sdf copy of the image, you must close at least 1 step
    raw_image_sdf = vto::topologyToLevelSet(
        *mask_grid, /*halfwidth voxels*/ RECUT_LEVEL_SET_HALF_WIDTH,
        /*closing steps*/ 1);
  }

  run_log << "Seed detection: mask to SDF conversion time, "
          << timer.elapsed_formatted() << '\n'
          << "Seed detection: SDF (topology) voxel count, "
          << sdf_grid->activeVoxelCount() << '\n';
  run_log.flush();

  auto known_seeds = process_marker_dir(args->seed_path, args->voxel_size);
  std::vector<openvdb::FloatGrid::Ptr> final_soma_sdfs;
  openvdb::FloatGrid::Ptr final_soma_sdf_merged;
  if (known_seeds.size()) {
    // when --seeds are passed gather them and turn them into a joined SDF grid
    openvdb::FloatGrid::Ptr known_seed_sphere_merged =
        create_seed_sphere_grid(known_seeds);
    if (args->save_vdbs)
      write_vdb_file({known_seed_sphere_merged}, run_dir / "known_seeds.vdb");

    // labels need somas as intact / unmodified therefore they should
    // not be unioned
    if (args->output_type != "labels") {
      // this strategies permanently modify the  mask_grid for both soma
      // detection and neurite reconstruction, the image grid uint8 that is
      // output in windows is unaffected however
      // fills in spheres where the user passed seeds are
      // known to be located
      timer.restart();
      vto::csgUnion(*sdf_grid, *known_seed_sphere_merged, true, true);
      std::cout << "\tFinished csgUnion in " << timer.elapsed() << '\n';

      // if (args->save_vdbs) {
      // write_vdb_file({sdf_grid}, run_dir / "union.vdb");
      //}

      timer.restart();
      final_soma_sdfs =
          find_soma_components(known_seeds, sdf_grid, args->user_thread_count);
      std::cout << "\tFinished finding " << final_soma_sdfs.size()
                << " soma components in " << timer.elapsed() << '\n';
    }

    // TODO replace with sumMergeOp which is parallel and more efficient
    timer.restart();
    final_soma_sdf_merged = merge_grids(final_soma_sdfs);
    std::cout << "\tFinished merge final somas in " << timer.elapsed() << '\n';
  }

  // if (args->save_vdbs)
  // write_vdb_file({sdf_grid}, run_dir / "sdf.vdb");

  // TODO find enclosed regions and log

  auto filter = create_filter(sdf_grid, args->morphological_operations_order);

  // open a bit to denoise specifically in brain surfaces
  if (args->open_denoise > 0) {
#ifdef LOG
    std::cout << "\topen denoise step: open = " << args->open_denoise << "\n";
#endif
    timer.restart();
    filter->offset(args->open_denoise);
    filter->offset(-args->open_denoise);
    run_log << "Seed detection: denoise time, " << timer.elapsed_formatted()
            << "\n";
  } else {
    run_log << "Seed detection: denoise time, 00:00:00:00 d:h:m:s\n";
  }
  run_log << "Seed detection: denoised SDF voxel count, "
          << sdf_grid->activeVoxelCount() << '\n';
  run_log.flush();

  // close to fill the holes inside somata where cell nuclei are
  if (args->open_denoise > 0) {
#ifdef LOG
    std::cout << "\tclose step: close = " << args->close_steps << "\n";
#endif
    timer.restart();
    filter->offset(-args->close_steps);
    filter->offset(args->close_steps);
    run_log << "Seed detection: closing time, " << timer.elapsed_formatted()
            << '\n'
            << "Seed detection: closed SDF voxel count, "
            << sdf_grid->activeVoxelCount() << '\n';
    run_log.flush();
  }

  auto closed_sdf = sdf_grid->deepCopy();
  if (args->save_vdbs)
    write_vdb_file({closed_sdf}, run_dir / "closed_sdf.vdb");

  // open again to filter axons and dendrites
  if (args->open_steps > 0) {
#ifdef LOG
    std::cout << "\topen step: open = " << args->open_steps << "\n";
#endif
    timer.restart();
    filter->offset(args->open_steps);
    filter->offset(-args->open_steps);
    run_log << "Seed detection: opening time, " << timer.elapsed_formatted()
            << "\n";
  } else {
    run_log << "Seed detection: opening time, 00:00:00:00 d:h:m:s\n";
  }
  run_log << "Seed detection: opened SDF voxel count, "
          << sdf_grid->activeVoxelCount() << "\n";
  run_log.flush();

  if (args->save_vdbs)
    write_vdb_file({sdf_grid}, run_dir / "opened_sdf.vdb");

  if (known_seeds.size()) {
    if (args->output_type == "labels") {
      //// convert active regions and capped concativities like hollow centers
      // openvdb::BoolGrid::Ptr opened_bool_grid =
      // vto::extractEnclosedRegion(*sdf_grid);
      create_labels(known_seeds, run_dir / "ml-train-and-test", image, sdf_grid,
                    nullptr, args->user_thread_count, args->save_vdbs);
#ifdef LOG
      std::cout << "\tLabel creation completed and safe to open\n";
#endif
      return std::make_pair(std::vector<Seed>{}, nullptr);
    }
  }

  // only overwrite final_soma_sdfs if user did not pass their own seeds
  if (known_seeds.size()) {
#ifdef LOG
    std::cout << "\tsegmentation step\n";
#endif
    timer.restart();
    // turn the whole sdf image into a vector of sdf for each connected
    // component
    vto::segmentSDF(*sdf_grid, final_soma_sdfs);
    run_log << "Seed detection: segmentation time, "
            << timer.elapsed_formatted() << '\n'
            << "Seed detection: initial seed count, " << final_soma_sdfs.size()
            << '\n';
    run_log.flush();
  }

// build full SDF by extending known somas into reachable neurites
#ifdef LOG
  std::cout << "\tmasking step\n";
#endif
  timer.restart();
  // if user passed known seeds then use the pretermined merged sdf grid
  // else use the result of the open and close steps above
  auto somas_connected_to_neurites =
      vto::maskSdf(known_seeds.size() ? *final_soma_sdf_merged : *sdf_grid,
                   args->close_topology ? *closed_sdf : *raw_image_sdf);
  run_log << "Seed detection: masking time, " << timer.elapsed_formatted()
          << '\n'
          << "Seed detection: masked SDF voxel count, "
          << somas_connected_to_neurites->activeVoxelCount() << '\n';
  run_log.flush();

// if (args->save_vdbs)
// write_vdb_file({somas_connected_to_neurites}, run_dir / "connected_sdf.vdb");

// adds all valid markers to roots vector
// filters by user input seeds if available
#ifdef LOG
  std::cout << "\tcreate seed pairs step\n";
  std::cout << "\tmin allowed radius is " << args->min_radius_um << " µm\n";
  std::cout << "\tmax allowed radius is " << args->max_radius_um << " µm\n";
#endif
  timer.restart();
  // If you already filtered the grid with seed intersection there's no need
  // to refilter by known seeds
  auto [seeds, filtered_components] = create_seed_pairs(
      final_soma_sdfs, somas_connected_to_neurites, args->voxel_size,
      args->min_radius_um, args->max_radius_um, args->output_type,
      args->seed_intersection ? std::vector<Seed>{} : known_seeds);
#ifdef LOG
  std::cout << "\tsaving " << seeds.size() << " seed coordinates to file ...\n";
#endif
  write_seeds(run_dir, seeds, args->voxel_size);

  run_log << "Seed detection: seed pairs creation time, "
          << timer.elapsed_formatted() << '\n'
          << "Seed detection: final seed count, " << seeds.size() << '\n';
  run_log.flush();

  // if (seeds.size() && args->output_type == "labels") {
  // auto zipped = rv::zip(seeds, filtered_components, rv::iota(0));
  // auto dir = run_dir / "true-positive-labels";
  // tbb::task_arena arena(args->user_thread_count);
  // arena.execute([&] {
  // tbb::parallel_for_each(zipped | rng::to_vector, [&](auto element) {
  // auto [seed, component, index] = element;
  //// auto [seed, component] = seedp;

  //// convert active regions and capped concativities like hollow
  //// centers
  // openvdb::BoolGrid::Ptr bool_component =
  // vto::extractEnclosedRegion(*sdf_grid);
  // create_label(seed, soma_dir(dir, seed), image, nullptr, index,
  // args->save_vdbs, 0);
  // create_label(seed, soma_dir(dir, seed), bool_component, nullptr, index,
  // args->save_vdbs, 1);
  //});
  //});
  //}

  return std::make_pair(seeds, somas_connected_to_neurites);
}
