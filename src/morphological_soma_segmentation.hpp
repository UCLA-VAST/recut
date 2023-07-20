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

std::pair<openvdb::FloatGrid::Ptr, std::vector<openvdb::FloatGrid::Ptr>>
create_seed_sphere_grid(std::vector<Seed> seeds) {
  auto component_grids = seeds | rv::transform([](const Seed &seed) {
                           int voxel_size = 1;
                           return vto::createLevelSetSphere<openvdb::FloatGrid>(
                               seed.radius, seed.coord.asVec3s(), voxel_size,
                               RECUT_LEVEL_SET_HALF_WIDTH);
                         }) |
                         rng::to_vector;
  auto merged = merge_grids(component_grids);
  return std::make_pair(merged, component_grids);
}

// this strategies permanently modify the  mask_grid for both soma
// detection and neurite reconstruction, the image grid uint8 that is
// output in windows is unaffected however
// fills in spheres where the user passed seeds are
// known to be located
void fill_seeds(openvdb::MaskGrid::Ptr mask_grid, std::vector<Seed> seeds) {
  auto mask_accessor = mask_grid->getAccessor();
  rng::for_each(seeds, [&mask_accessor](Seed seed) {
    for (const auto coord : sphere_iterator(seed.coord, seed.radius)) {
      mask_accessor.setValueOn(coord);
    }
  });
}

std::optional<GridCoord> mean_location(openvdb::MaskGrid::Ptr mask_grid) {
  GridCoord sum = zeros();
  uint64_t counter = 0;
  for (auto iter = mask_grid->cbeginValueOn(); iter.test(); ++iter) {
    auto val_coord = iter.getCoord();
    sum += val_coord;
    ++counter;
  }
  if (counter)
    return coord_div(sum, GridCoord(counter));
  return std::nullopt;
}

// establish highest performance most basic filter for morphological
// operations like dilation, erosion, opening, closing
// Of all connected components, keep those whose central coordinate is an
// active voxel in the point topology and estimate the radius given the
// bbox of the component. If passing known seeds, then filter all
// components to only those components which contain the passed seed
// If you already filtered the grid with seed intersection there's no need
// to refilter by known seeds
auto create_seed_pairs = [](std::vector<openvdb::MaskGrid::Ptr> components,
                            std::array<float, 3> voxel_size,
                            float min_radius_um, float max_radius_um,
                            std::string output_type,
                            std::vector<Seed> known_seeds = {}) {
  std::vector<Seed> seeds;
  std::vector<openvdb::MaskGrid::Ptr> filtered_components;
  auto removed_by_known_seeds = 0;
  auto empty_components = 0;
  auto removed_by_radii = 0;
  auto max_voxel_size = min_max(voxel_size).second;
  for (auto component : components) {
    auto volume_voxels = component->activeVoxelCount();
    if (volume_voxels < 1) {
      ++empty_components;
      continue;
    }

    // estimate the radii from the volume of active voxels
    float radius_voxels = std::cbrtf((volume_voxels * 3) / (4 * PI));
    auto coord = mean_location(component);
    GridCoord coord_center;
    if (coord) {
      coord_center = coord.value();
    } else {
      ++empty_components;
      continue;
    }

    /*
    if (!known_seeds.empty()) {
      // component if no known seed is an active voxel in this
      // component then remove this component
      if (rng::none_of(known_seeds, [&component](const auto &known_seed) {
            return component->tree().isValueOn(known_seed.coord);
          })) {
        ++removed_by_known_seeds;
        continue;
      }
    }
    */

    // the radius is calculated by the distance of the center point
    // to the nearest surface point voxel sizes can be anisotropic,
    // and in such cases the radii is likely set by the distance (in
    // voxels) along the lowest resolution (the largest voxel length)
    // dimension Example for a voxel size of [.2, .2, 1], if the
    // radius returned was 3, it would be the actual radii_um is 3 um
    // along the z-dimension therefore scale with the voxel dimension
    // with the largest length
    auto radius_um = radius_voxels * std::pow(max_voxel_size, 3);
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
  if (removed_by_known_seeds)
    std::cerr << "\tWarning: seeds removed by known seeds "
              << removed_by_known_seeds << '\n';
  if (empty_components)
    std::cerr << "\tWarning: seeds removed by empty component "
              << empty_components << '\n';
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
    filter->setSpatialScheme(openvdb::math::FIRST_BIAS);
    break;
  case 2:
    filter->setSpatialScheme(openvdb::math::SECOND_BIAS);
    break;
  case 3:
    filter->setSpatialScheme(openvdb::math::THIRD_BIAS);
    break;
  case 4:
    filter->setSpatialScheme(openvdb::math::WENO5_BIAS);
    break;
  case 5:
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

  auto clipped_grid = vto::clip(*grid, clipBox);

  if (!clipped_grid || (clipped_grid->activeVoxelCount() == 0)) {
    return std::nullopt; // do nothing
  }

  /* Warning if you uncomment this you can no longer use segmentSDF
  if (channel) {
    std::vector<openvdb::FloatGrid::Ptr> window_components;
    // works on grids of arbitrary type, placing all disjoint segments
    // (components) in decreasing size order in window_components
    // causes seg fault
    // vto::segmentActiveVoxels(*clipped_grid, window_components);
    vto::segmentSDF(*clipped_grid, window_components);

    if (window_components.size() == 0) {
      std::cout << "\tNo window components\n";
      return std::nullopt; // do nothing
    }

    // find the component that has the seed within it
    std::vector<openvdb::FloatGrid::Ptr> known_component;
    for (int i = 0; i < window_components.size(); ++i) {
      openvdb::FloatGrid::Ptr component = window_components[i];
      if (component->evalActiveVoxelBoundingBox().isInside(seed.coord)) {
        known_component.push_back(component);
        continue;
      }
      // if (!component || component->activeVoxelCount() == 0)
      // continue;
      // auto mask = vto::extractEnclosedRegion(*component);
      // auto fog = component->deepCopy();
      // vto::sdfToFogVolume(*fog);
      //// auto test = vto::sdfInteriorMask(*component);
      // if (mask) {
      // std::cout << "\tenclosed voxel count: " << mask->activeVoxelCount()
      //<< " on " << mask->tree().isValueOn(seed.coord) << '\n';
      //}
      //// if (test) {
      //// std::cout << "\tinterior voxel count: " << test->activeVoxelCount()
      ////<< " on " << test->tree().isValueOn(seed.coord) << '\n';
      ////}
      // if (fog) {
      // std::cout << "\tfog voxel count: " << fog->activeVoxelCount()
      //<< " on " << fog->tree().isValueOn(seed.coord) << '\n';
      //}
      // std::cout << '\n';
      // if (mask && mask->activeVoxelCount()) {
      // if (mask->tree().isValueOn(seed.coord))
      // known_component.push_back(component);
      //}
    }

    // auto known_component =
    // window_components | rv::filter([&seed](auto component) {
    // if (!component || component->activeVoxelCount() == 0)
    // return false;
    // auto mask = vto::extractEnclosedRegion(*component);
    //// auto test = vto::sdfInteriorMask(*component);
    // if (mask) {
    // std::cout << "\tenclosed voxel count: " << mask->activeVoxelCount()
    //<< " on " << mask->tree().isValueOn(seed.coord) << '\n';
    //}
    //// if (test) {
    //// std::cout << "\tinterior voxel count: " << test->activeVoxelCount()
    ////<< " on " << test->tree().isValueOn(seed.coord) << '\n';
    ////}
    // std::cout << '\n';
    // if (mask && mask->activeVoxelCount())
    // return mask->tree().isValueOn(seed.coord);
    // return false;
    //// return component->tree().isValueOn(seed.coord);
    //}) |
    // rng::to_vector;

    if (known_component.size() == 0) {
      std::cout << "\tNo known component\n";
    }

    // otherwise use the largest (first) component in the window
    clipped_grid = known_component.size() > 0 ? known_component.front()
                                              : window_components.front();
  }
  */
  return std::make_pair(clipped_grid, bbox);
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
void create_labels(std::vector<Seed> seeds, fs::path dir,
                   ImgGrid::Ptr image = nullptr,
                   openvdb::FloatGrid::Ptr sdf_grid = nullptr,
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
}

std::vector<Seed> soma_segmentation(const openvdb::MaskGrid::Ptr mask_grid,
                                    std::vector<Seed> seeds,
                                    RecutCommandLineArgs *args,
                                    GridCoord image_lengths, fs::path log_fn,
                                    fs::path run_dir) {

  auto timer = high_resolution_timer();
  assertm(mask_grid, "Mask grid must be set before starting soma segmentation");
  auto neurite_mask = mask_grid->deepCopy();

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

  std::ofstream run_log;
  run_log.open(log_fn, std::ios::app);

  /*
  openvdb::FloatGrid::Ptr final_soma_sdf_merged;
  if (seeds.size()) {
    // when --seeds are passed gather them and turn them into a joined SDF
    // grid
    auto finals = create_seed_sphere_grid(seeds);
    final_soma_sdf_merged = finals.first;
    final_soma_sdfs = finals.second;
  }
  */

  // open again to filter axons and dendrites
  if (args->open_steps) {
#ifdef LOG
    std::cout << "\tStart morphological open = " << args->open_steps.value()
              << "\n";
#endif
    timer.restart();
    openvdb::tools::erodeActiveValues(neurite_mask->tree(),
                                      args->open_steps.value());
    openvdb::tools::pruneInactive(neurite_mask->tree());
    openvdb::tools::dilateActiveValues(neurite_mask->tree(),
                                       args->open_steps.value());
    run_log << "Seed detection: opening time, " << timer.elapsed_formatted()
            << "\n";
    run_log << "Seed detection: opened SDF voxel count, "
            << neurite_mask->activeVoxelCount() << "\n";
    run_log.flush();
#ifdef LOG
    std::cout << "\tEnd morphological open\n";
#endif
  }

  if (seeds.size()) {
    if (args->output_type == "labels") {
      create_labels(seeds, run_dir / "ml-train-and-test", image, nullptr,
                    nullptr, args->user_thread_count, args->save_vdbs);
#ifdef LOG
      std::cout << "\tLabel creation completed and safe to open\n";
#endif
      return std::vector<Seed>{};
    }
    /*
    if (args->seed_intersection) {
      timer.restart();
      std::cout << "\tFinished csgIntersection in " << timer.elapsed() << '\n';
    }
    */
  }

  // only overwrite final_soma_sdfs if user did not pass their own seeds
  // or seed intersection is on
  std::vector<openvdb::MaskGrid::Ptr> final_soma_sdfs;
  // if (args->seed_intersection || seeds.size() == 0) {
  if (seeds.size() == 0) {
#ifdef LOG
    std::cout << "\tsegmentation step\n";
#endif
    timer.restart();
    // turn the whole sdf image into a vector of sdf for each connected
    // component
    vto::segmentActiveVoxels(*neurite_mask, final_soma_sdfs);
    // vto::extractActiveVoxelSegmentMasks(*neurite_mask, final_soma_sdfs);
    run_log << "Seed detection: segmentation time, "
            << timer.elapsed_formatted() << '\n'
            << "Seed detection: initial seed count, " << final_soma_sdfs.size()
            << '\n';
    run_log.flush();

#ifdef LOG
    std::cout << "\tcreate seed pairs step\n";
    std::cout << "\tmin allowed radius is " << args->min_radius_um << " µm\n";
    std::cout << "\tmax allowed radius is " << args->max_radius_um << " µm\n";
#endif
    timer.restart();
    // adds all valid markers to roots vector
    // filters by user input seeds if available
    // If you already filtered the grid with seed intersection there's no need
    // to refilter by known seeds
    auto pairs = create_seed_pairs(
        final_soma_sdfs, args->voxel_size, args->min_radius_um,
        args->max_radius_um, args->output_type,
        args->seed_intersection ? std::vector<Seed>{} : seeds);
    seeds = pairs.first;
    final_soma_sdfs = pairs.second;
    run_log << "Seed detection: seed pairs creation time, "
            << timer.elapsed_formatted() << '\n';
    run_log.flush();
  }

#ifdef LOG
  std::cout << "\tsaving " << seeds.size() << " seed coordinates to file ...\n";
#endif
  run_log << "Seed detection: final seed count, " << seeds.size() << '\n';
  run_log.flush();
  write_seeds(run_dir, seeds, args->voxel_size);

  return seeds;
}
