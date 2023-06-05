#include <GEL/Geometry/Graph.h>
#include <GEL/Geometry/graph_skeletonize.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>

auto euc_dist = [](auto a, auto b) -> float {
  std::array<float, 3> diff = {
      static_cast<float>(a[0]) - static_cast<float>(b[0]),
      static_cast<float>(a[1]) - static_cast<float>(b[1]),
      static_cast<float>(a[2]) - static_cast<float>(b[2])};
  return std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
};

std::set<size_t> find_soma_nodes(Geometry::AMGraph3D graph, std::vector<Seed> seeds) {
  std::set<size_t> soma_ids;
  rng::for_each(seeds, [&](Seed seed) {
    int max_val = 0;
    std::optional<size_t> max_index;
    for (auto i : graph.node_ids()) {
      if (euc_dist(graph.pos[i].get(), seed.coord) < seed.radius) {
        auto valence = graph.valence(i);
        if (valence > max_val) {
          max_index = i;
          max_val = valence;
        }
      }
    }
    if (max_index)
      soma_ids.insert(max_index.value());
    else
      std::cout << "Warning no seed found\n";
  });
  return soma_ids;
}

void topology_to_tree(openvdb::FloatGrid::Ptr topology, fs::path run_dir,
                      std::vector<Seed> seeds, bool save_vdbs = false) {
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
        auto [component_graph, mapping] =
            skeleton_from_node_set_vec(g, separators);

        // sweep through various soma ids
        auto soma_ids = find_soma_nodes(component_graph, seeds);
        if (soma_ids.size() == 0) {
          std::cout << "Warning no soma_ids found for component " << index
                    << '\n';
          // assign a soma randomly if none are found
          soma_ids.insert(0);
        }

        // std::unordered_map<GridCoord, MyMarker *> coord_to_marker_ptr;
        // save this marker ptr to a map
        // coord_to_marker_ptr.emplace(coord, marker);
        std::vector<MyMarker *> component_tree;
        for (auto i : component_graph.node_ids()) {
          auto color = component_graph.node_color[i].get();
          // radius is in the green channel
          auto radius = color[1];
          auto pos = g.pos[i].get();
          component_tree.emplace_back(pos[0], pos[1], pos[2], radius);
        }

        // assign all parents via BFS
        // traverse graph outwards from any soma, does not matter which
        size_t root = *soma_ids.begin();
        auto ms_tree = minimum_spanning_tree(component_graph, root);
        Geometry::BreadthFirstSearch bfs(ms_tree);
        bfs.add_init_node(root);
        while (bfs.Prim_step()) {
          auto last = bfs.get_last();
          auto marker = ms_tree[last];
          if (last == soma_id) {
            marker->parent = 0;
            marker->type = 0;
          } else {
            marker->parent = component_tree(bfs.pred[last]));
          }
        }

        auto trees = partition_cluster(component_tree);

        if (save_vdbs) {
          write_vdb_file({component}, component_dir_fn / "sdf.vdb");
        }
      });

  exit(0);
}
