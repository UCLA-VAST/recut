#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>

void topology_to_tree(openvdb::FloatGrid::Ptr topology, fs::path run_dir,
                      bool save_vdbs = false) {
  write_vdb_file({topology}, run_dir / "whole-topology.vdb");
  std::vector<openvdb::FloatGrid::Ptr> components;
  vto::segmentSDF(*topology, components);

  rng::for_each(components | rv::enumerate | rng::to_vector, [&](auto cpair) {
    auto [index, component] = cpair;
    std::vector<openvdb::Vec3s> points;
    // quad index list, which can be post-processed to
    // find a triangle mesh
    std::vector<openvdb::Vec4I> quads;
    vto::volumeToMesh(*component, points, quads);
    std::cout << "Component active voxel count: "
              << component->activeVoxelCount() << '\n';
    std::cout << "points size: " << points.size() << '\n';

    // write to file
    auto component_dir_fn = run_dir / ("component-" + std::to_string(index));
    fs::create_directories(component_dir_fn);
    std::ofstream mesh_file;
    mesh_file.open(component_dir_fn / ("mesh_graph.gel"), std::ios::app);

    rng::for_each(points, [&](auto point) {
      mesh_file << "n " << point[0] << ' ' << point[1] << ' ' << point[2]
                << '\n';
    });

    // list all connections of the 0-indexed points
    rng::for_each(quads, [&](auto quad) {
      for (int i = 0; i < 4; ++i) {
        mesh_file << "c " << quad[i] << ' ' << quad[(i + 1) % 4] << '\n';
      }
    });

    if (save_vdbs) {
      write_vdb_file({component}, component_dir_fn / "sdf.vdb");
    }
  });

  exit(0);
}
