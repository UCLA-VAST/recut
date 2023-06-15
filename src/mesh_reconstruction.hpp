#pragma once

#include "tree_ops.hpp"
#include <GEL/Geometry/Graph.h>
#include <GEL/Geometry/graph_io.h>
#include <GEL/Geometry/graph_skeletonize.h>
#include <openvdb/tools/FastSweeping.h> // fogToSdf
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

std::set<size_t> find_soma_nodes(Geometry::AMGraph3D graph,
                                 std::vector<Seed> seeds) {
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

void write_graph(Geometry::AMGraph3D &g, fs::path fn) {
  // std::ofstream graph_file;
  // graph_file.open(fn, std::ios::app);
  // for (auto i : g.node_ids()) {
  // auto pos = g.pos[i].get();
  // graph_file << "n " << pos[0] << ' ' << pos[1] << ' ' << pos[2] << '\n';
  //}
  // for (auto i : g.edge_ids()) {
  // graph_file << "c " << a << ' ' << b << '\n';
  //}
  // graph_file.close();
  Geometry::graph_save(fn.generic_string(), g);

  std::cout << "Wrote: " << fn << " nodes: " << g.no_nodes()
            << " edges: " << g.no_edges() << '\n';
}

template <typename Poly>
auto unroll_polygons(std::vector<Poly> polys, Geometry::AMGraph3D &g,
                     unsigned int order) {
  // list all connections of the 0-indexed points
  rng::for_each(polys, [&](auto poly) {
    for (unsigned int i = 0; i < order; ++i) {
      auto a = poly[i];
      auto b = poly[(i + 1) % order];
      g.connect_nodes(a, b);
    }
  });
};

Geometry::AMGraph3D vdb_to_mesh(openvdb::FloatGrid::Ptr component,
                                RecutCommandLineArgs *args) {
  std::vector<openvdb::Vec3s> points;
  // quad index list, which can be post-processed to
  // find a triangle mesh
  std::vector<openvdb::Vec4I> quads;
  std::vector<openvdb::Vec3I> tris;
  // vto::volumeToMesh(*component, points, quads);
  vto::volumeToMesh(*component, points, tris, quads, 0, args->mesh_grain);
  std::cout << "Component active voxel count: " << component->activeVoxelCount()
            << '\n';
  std::cout << "points size: " << points.size() << '\n';

  Geometry::AMGraph3D g;
  rng::for_each(points, [&](auto point) {
    auto p = CGLA::Vec3d(point[0], point[1], point[2]);
    auto node_id = g.add_node(p);
  });

  unroll_polygons(tris, g, 3);
  unroll_polygons(quads, g, 4);

  return g;
}

void topology_to_tree(openvdb::FloatGrid::Ptr topology, fs::path run_dir,
                      std::vector<Seed> seeds, RecutCommandLineArgs *args) {
  write_vdb_file({topology}, run_dir / "whole-topology.vdb");
  std::vector<openvdb::FloatGrid::Ptr> components;
  // vto::segmentSDF(*topology, components);
  vto::segmentActiveVoxels(*topology, components);
  std::cout << "Total components: " << components.size() << '\n';

  rng::for_each(
      components | rv::take(1) | rv::enumerate | rng::to_vector,
      [&](auto cpair) {
        auto [index, fog] = cpair;
        auto component_dir_fn =
            run_dir / ("component-" + std::to_string(index));
        fs::create_directories(component_dir_fn);

        auto component = vto::fogToSdf(*fog, 0);
        if (args->save_vdbs) {
          write_vdb_file({component}, component_dir_fn / "sdf.vdb");
        }
        auto g = vdb_to_mesh(component, args);
        write_graph(g, component_dir_fn / ("mesh.graph"));

        uint desired_node_count = 1000;
        auto msg = Geometry::multiscale_graph(g, desired_node_count, true);
        g = msg.layers.back(); // get the coarsest (smallest) graph
                               // representation in the hierarchy
        write_graph(g, component_dir_fn / ("coarse.graph"));

        // classic local separators
        // auto adv_samp_thresh = 8; // higher is higher quality at cost of
        // runtime auto separators = local_separators(g,
        // Geometry::SamplingType::None, args->skeleton_grain, adv_samp_thresh);

        uint grow_threshold =
            64; // 4 had lowest resolution, higher is finer granularity
        // multi-scale is faste and scales linearly with input graph size at the
        // cost of difficulty in choosing a grow threshold
        auto separators =
            multiscale_local_separators(g, Geometry::SamplingType::Advanced,
                                        grow_threshold, args->skeleton_grain);
        auto [component_graph, mapping] =
            skeleton_from_node_set_vec(g, separators);
        write_graph(g, component_dir_fn / ("skeleton.graph"));

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
          // auto radius = color[1];
          auto radius = 1;
          auto pos = g.pos[i].get();
          // TODO
          auto marker = new MyMarker(pos[0], pos[1], pos[2], radius);
          // component_tree.emplace_back(pos[0], pos[1], pos[2], radius);
          component_tree.push_back(marker);
        }

        // assign all parents via BFS
        // traverse graph outwards from any soma, does not matter which
        Geometry::BreadthFirstSearch bfs(component_graph);
        size_t root = *soma_ids.begin();
        bfs.add_init_node(root);
        while (bfs.Prim_step()) {
          auto last = bfs.get_last();
          auto marker = component_tree[last];
          if (soma_ids.find(last) != soma_ids.end()) {
            // soma
            marker->parent = 0;
            marker->type = 0;
          } else {
            marker->parent = component_tree[bfs.pred[last]];
          }
        }

        auto trees = partition_cluster(component_tree);

        // TODO setTransform?
        auto bbox = component->evalActiveVoxelBoundingBox();
        rng::for_each(trees, [&](auto tree) {
          write_swc(tree, args->voxel_size, component_dir_fn, bbox,
                    /*bbox_adjust*/ !args->window_grid_paths.empty(),
                    args->output_type == "eswc");
          if (!parent_listed_above(tree)) {
            throw std::runtime_error("Tree is not properly sorted");
          }
        });
      });

  exit(0);
}
