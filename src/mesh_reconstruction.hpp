#pragma once

#include "tree_ops.hpp"
#include <GEL/Geometry/Graph.h>
#include <GEL/Geometry/graph_io.h>
#include <GEL/Geometry/graph_skeletonize.h>
#include <GEL/Geometry/KDTree.h>
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

  // build kdtree, highly efficient for nearest point computations
  Geometry::KDTree<CGLA::Vec3d, int> tree;
  for (auto i : graph.node_ids()) {
    CGLA::Vec3d p0 = graph.pos[i];
    tree.insert(p0, i);
  }
  tree.build();

  std::set<size_t> soma_ids;
  rng::for_each(seeds, [&](Seed seed) {
    // find existing skeletal node within the radius of the soma
    auto coord = seed.coord;
    CGLA::Vec3d p0(coord[0], coord[1], coord[2]);
    std::vector<CGLA::Vec3d> keys;
    std::vector<int> vals;
    int within_count = tree.in_sphere(p0, seed.radius, keys, vals);

    // pick skeletal node within radii with highest number of edges (valence)
    int max_valence = 0;
    std::optional<size_t> max_index;
    for (int i = 0; i < within_count; ++i) {
      int idx = vals[i];
      auto valence = graph.valence(idx);
      if (valence > max_valence) {
        max_index = idx;
        max_valence = valence;
      }
    }

    if (max_index)
      soma_ids.insert(max_index.value());
    else
      std::cout << "Warning lost 1 seed during skeletonization\n";
  });
  return soma_ids;
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

    // quads can not be directly skeletonized by the local separator approach
    // so you must create connections along a diagonal
    // do both diagonals for coarser final skeletons
    if (order == 4) {
      g.connect_nodes(poly[0], poly[2]);
      g.connect_nodes(poly[1], poly[3]);
    }
  });
};

Geometry::AMGraph3D vdb_to_graph(openvdb::FloatGrid::Ptr component,
                                 RecutCommandLineArgs *args) {
  std::vector<openvdb::Vec3s> points;
  // quad index list, which can be post-processed to
  // find a triangle mesh
  std::vector<openvdb::Vec4I> quads;
  std::vector<openvdb::Vec3I> tris;
  vto::volumeToMesh(*component, points, quads);
  // vto::volumeToMesh(*component, points, tris, quads, 0, args->mesh_grain);
  std::cout << "Component active voxel count: " << component->activeVoxelCount()
            << '\n';
  std::cout << "points size: " << points.size() << '\n';

  Geometry::AMGraph3D g;
  rng::for_each(points, [&](auto point) {
    auto p = CGLA::Vec3d(point[0], point[1], point[2]);
    auto node_id = g.add_node(p);
  });

  // unroll_polygons(tris, g, 3);
  unroll_polygons(quads, g, 4);

  return g;
}

HMesh::Manifold vdb_to_mesh(openvdb::FloatGrid::Ptr component,
                            RecutCommandLineArgs *args) {
  std::vector<openvdb::Vec3s> points;
  // quad index list, which can be post-processed to
  // find a triangle mesh
  std::vector<openvdb::Vec4I> quads;
  std::vector<openvdb::Vec3I> tris;
  vto::volumeToMesh(*component, points, quads);
  // vto::volumeToMesh(*component, points, tris, quads, 0, args->mesh_grain);
  // std::cout << "Component active voxel count: " <<
  // component->activeVoxelCount()
  //<< '\n';
  // std::cout << "points size: " << points.size() << '\n';

  // convert points to GEL vertices
  auto vertices = points | rv::transform([](auto point) {
                    return CGLA::Vec3f(point[0], point[1], point[2]);
                  }) |
                  rng::to_vector;

  // convert polygonal faces to manifold edges
  std::vector<int> faces, indices;
  auto triangle_count = 2 * quads.size();
  faces.reserve(triangle_count);
  indices.reserve(3 * triangle_count);
  rng::for_each(quads | rv::enumerate, [&](auto quadp) {
    // this quad has 2 tris devoted to it
    auto [i, quad] = quadp;
    faces.push_back(3);
    faces.push_back(3);

    // FIXME check this to in counter clockwise order
    // this quad has 6 indices devoted to it
    auto offset = 2 * 3 * i;
    // triangle 0 1 2
    indices.push_back(quad[0]);
    indices.push_back(quad[1]);
    indices.push_back(quad[2]);

    // quads can not be directly skeletonized by the local separator approach
    // so you must create connections along a diagonal
    // triangle 0 2 3
    indices.push_back(quad[0]);
    indices.push_back(quad[2]);
    indices.push_back(quad[3]);
  });

  HMesh::Manifold m;
  HMesh::build(m, vertices.size(), reinterpret_cast<float *>(&vertices[0]),
               faces.size(), &faces[0], &indices[0]);

  return m;
}

// Taken directly from PyGEL library
Geometry::AMGraph3D graph_from_mesh(HMesh::Manifold &m) {
  HMesh::VertexAttributeVector<Geometry::AMGraph::NodeID> v2n;
  Geometry::AMGraph3D g;

  for (auto v : m.vertices())
    v2n[v] = g.add_node(m.pos(v));
  for (auto h : m.halfedges()) {
    HMesh::Walker w = m.walker(h);
    if (h < w.opp().halfedge())
      g.connect_nodes(v2n[w.opp().vertex()], v2n[w.vertex()]);
  }

  return g;
}

std::optional<std::vector<MyMarker *>>
vdb_to_markers(openvdb::FloatGrid::Ptr fog, std::vector<Seed> component_seeds,
               int index, RecutCommandLineArgs *args,
               fs::path component_dir_fn) {
  auto component = vto::fogToSdf(*fog, 0);
  if (args->save_vdbs) {
    write_vdb_file({component}, component_dir_fn / "sdf.vdb");
  }

  HMesh::Manifold m;
  try {
    m = vdb_to_mesh(component, args);
  } catch (...) {
    return std::nullopt;
  }
  if (!(HMesh::valid(m) && m.no_vertices()))
    return std::nullopt;
  HMesh::obj_save(component_dir_fn / ("mesh.obj"), m);

  auto g = graph_from_mesh(m);
  // graph_save(component_dir_fn / ("mesh.graph"), g);

  // multi-scale is faster and scales linearly with input graph size at
  // the cost of difficulty in choosing a grow threshold
  auto separators =
      multiscale_local_separators(g, Geometry::SamplingType::Advanced,
                                  args->skeleton_grow, args->skeleton_grain);
  auto [component_graph, mapping] = skeleton_from_node_set_vec(g, separators);
  graph_save(component_dir_fn / ("skeleton.graph"), component_graph);

  // sweep through various soma ids
  auto soma_ids = find_soma_nodes(component_graph, component_seeds);
  if (soma_ids.size() == 0) {
    std::cout << "Warning no soma_ids found for component " << index << '\n';
    // assign a soma randomly if none are found
    soma_ids.insert(0);
  }

  std::vector<MyMarker *> component_tree;
  for (auto i : component_graph.node_ids()) {
    auto color = component_graph.node_color[i].get();
    // radius is in the green channel
    auto radius = color[1]; // RGB
    auto pos = component_graph.pos[i].get();
    auto marker = new MyMarker(pos[0], pos[1], pos[2], radius);
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
      // std::cout << "->parent " << bfs.pred[last] << '\n';;
      marker->parent = component_tree[bfs.pred[last]];
    }
  }

  /*
  for (auto i : soma_ids) {
    std::cout << "soma " << i << '\n';
  }

  for (auto i : component_graph.node_ids()) {
    auto current = component_tree[i];
    if ((current->type != 0) && !current->parent)
      std::cout << i << ' ' << *current << '\n';
  }
  */

  return component_tree;
}

void topology_to_tree(openvdb::FloatGrid::Ptr topology, fs::path run_dir,
                      std::vector<Seed> seeds, RecutCommandLineArgs *args) {
  write_vdb_file({topology}, run_dir / "whole-topology.vdb");
  std::vector<openvdb::FloatGrid::Ptr> all_components;
  vto::segmentActiveVoxels(*topology, all_components);

  auto components = all_components | rv::remove_if([&seeds](auto component) {
                      // convert to fog so that isValueOn returns whether it is
                      // within the
                      // auto fog = vto::sdfToFogVolume(*component); // or find
                      // extractEnclosedRegion or sdfInteriorMask auto fog =
                      // vto::extractEnclosedRegion(*component);
                      // component if no known seed is an active voxel in this
                      // component then remove this component
                      return rng::none_of(seeds, [component](const auto &seed) {
                        return component->tree().isValueOn(seed.coord);
                      });
                    }) |
                    rng::to_vector;
  std::cout << "Total components: " << components.size() << '\n';

  rng::for_each(components | rv::enumerate | rng::to_vector, [&](auto cpair) {
    auto [index, fog] = cpair;
    auto component_dir_fn = run_dir / ("component-" + std::to_string(index));
    fs::create_directories(component_dir_fn);

    auto component_tree_opt =
        vdb_to_markers(fog, seeds, index, args, component_dir_fn);
    auto component_tree = component_tree_opt.value();

    auto trees = partition_cluster(component_tree);

    auto bbox = fog->evalActiveVoxelBoundingBox();
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
