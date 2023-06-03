#include <GEL/Geometry/Graph.h>
#include <GEL/Geometry/graph_skeletonize.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>

void topology_to_tree(openvdb::FloatGrid::Ptr topology, fs::path run_dir,
                      bool save_vdbs = false) {
  write_vdb_file({topology}, run_dir / "whole-topology.vdb");
  std::vector<openvdb::FloatGrid::Ptr> components;
  // vto::segmentSDF(*topology, components);
  vto::segmentActiveVoxels(*topology, components);

  rng::for_each(
      components | rv::take(1) | rv::enumerate | rng::to_vector,
      [&](auto cpair) {
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
        auto component_dir_fn =
            run_dir / ("component-" + std::to_string(index));
        fs::create_directories(component_dir_fn);
        std::ofstream mesh_file;
        mesh_file.open(component_dir_fn / ("mesh_graph.gel"), std::ios::app);

        Geometry::AMGraph3D g;
        rng::for_each(points, [&](auto point) {
          auto p = CGLA::Vec3d(point[0], point[1], point[2]);
          auto node_id = g.add_node(p);
          std::cout << "add node: " << node_id << '\n';
          mesh_file << "n " << point[0] << ' ' << point[1] << ' ' << point[2]
                    << '\n';
        });

        // list all connections of the 0-indexed points
        rng::for_each(quads, [&](auto quad) {
          for (int i = 0; i < 4; ++i) {
            auto a = quad[i];
            auto b = quad[(i + 1) % 4];
            g.connect_nodes(a, b);
            std::cout << "c " << a << ' ' << b << '\n';
            mesh_file << "c " << a << ' ' << b << '\n';
          }
        });

        // make local separators
        auto separators = local_separators(g, Geometry::SamplingType::Basic);
        auto [skeleton, mapping] = skeleton_from_node_set_vec(g, separators);

        if (save_vdbs) {
          write_vdb_file({component}, component_dir_fn / "sdf.vdb");
        }
      });

  exit(0);
}
