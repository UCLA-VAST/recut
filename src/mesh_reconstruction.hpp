#pragma once

#include "tree_ops.hpp"
#include <GEL/Geometry/Graph.h>
#include <GEL/Geometry/graph_io.h>
#include <GEL/Geometry/graph_util.h>
#include <GEL/Geometry/graph_skeletonize.h>
//#include <GEL/Geometry/graph_util.h>
#include <GEL/Geometry/KDTree.h>
#include <GEL/Util/AttribVec.h>
#include <openvdb/tools/FastSweeping.h> // fogToSdf
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/VolumeToMesh.h>

auto euc_dist = [](auto a, auto b) -> float {
  std::array<float, 3> diff = {
      static_cast<float>(a[0]) - static_cast<float>(b[0]),
      static_cast<float>(a[1]) - static_cast<float>(b[1]),
      static_cast<float>(a[2]) - static_cast<float>(b[2])};
  return std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
};

// build kdtree, highly efficient for nearest point computations
Geometry::KDTree<CGLA::Vec3d, int> build_kdtree(Geometry::AMGraph3D &graph) {
  Geometry::KDTree<CGLA::Vec3d, int> tree;
  for (auto i : graph.node_ids()) {
    CGLA::Vec3d p0 = graph.pos[i];
    tree.insert(p0, i);
  }
  tree.build();
  return tree;
}

// find existing skeletal node within the radius of the soma
std::vector<int> within_sphere(Seed &seed,
                               Geometry::KDTree<CGLA::Vec3d, int> &tree,
                               float soma_dilation) {
  auto coord = seed.coord;
  CGLA::Vec3d p0(coord[0], coord[1], coord[2]);
  std::vector<CGLA::Vec3d> keys;
  std::vector<int> vals;
  tree.in_sphere(p0, soma_dilation * seed.radius, keys, vals);
  return vals;
}

// add seeds passed as new nodes in the graph,
// nearby skeletal nodes are merged into the seeds,
// seeds inherit edges and delete nodes within 3D radius soma_dilation *
// seed.radius the original location and radius of the seeds are preserved
std::vector<long unsigned int> force_soma_nodes(Geometry::AMGraph3D &graph,
                                                std::vector<Seed> &seeds,
                                                float soma_dilation) {

  // add the known seeds to the skeletonized graph
  // aggregate their seed ids so they are not mistakenly merged
  // below
  auto seed_ids =
      seeds | rv::transform([&](Seed seed) {
        auto p = CGLA::Vec3d(seed.coord[0], seed.coord[1], seed.coord[2]);
        auto soma_id = graph.add_node(p);
        // set radius by previously computed radius
        graph.node_color[soma_id] =
            CGLA::Vec3f(0, soma_dilation * seed.radius, 0);
        return soma_id;
      }) |
      rng::to_vector;
  auto seed_pairs = rv::zip(seeds, seed_ids);

  // the graph is complete so build a data structure that
  // is fast at finding nodes within a 3D radial distance
  auto tree = build_kdtree(graph);

  std::set<int> deleted_nodes;
  rng::for_each(seed_pairs, [&](auto seedp) {
    auto [seed, seed_id] = seedp;

    auto within_sphere_ids = within_sphere(seed, tree, soma_dilation);
    // iterate all nodes within a 3D radial distance from this known seed
    for (auto id : within_sphere_ids) {
      // if this node within the sphere isn't a known seed then merge it into
      // the seed at the center of the radial sphere
      if (std::find(seed_ids.begin(), seed_ids.end(), id) ==
          std::end(seed_ids)) {
        // the tree is unaware of nodes that have previously been deleted
        // so you must specifically filter out those so they do not error
        if (std::find(deleted_nodes.begin(), deleted_nodes.end(), id) ==
            std::end(deleted_nodes)) {
          // keep the original location
          graph.merge_nodes(id, seed_id, /*average location*/ false);
          deleted_nodes.insert(id);
        }
      }
    }
  });

  return seed_ids;
}

std::set<size_t> find_soma_nodes(Geometry::AMGraph3D &graph,
                                 std::vector<Seed> seeds, float soma_dilation) {

  auto tree = build_kdtree(graph);

  std::set<size_t> soma_ids;
  rng::for_each(seeds, [&](Seed seed) {
    auto within_sphere_ids = within_sphere(seed, tree, soma_dilation);

    // pick skeletal node within radii with highest number of edges (valence)
    int max_valence = 0;
    std::optional<size_t> max_index;
    for (auto id : within_sphere_ids) {
      auto valence = graph.valence(id);
      if (valence > max_valence) {
        max_index = id;
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

openvdb::FloatGrid::Ptr mask_to_sdf(openvdb::MaskGrid::Ptr mask) {
  /*
   this has a weird bug
  struct Local {
    static inline void op(const openvdb::MaskGrid::ValueOnCIter &iter,
                          openvdb::FloatGrid::ValueAccessor &accessor) {
      if (iter.isVoxelValue()) {
        accessor.setValue(iter.getCoord(), 1.);
      } else {
        openvdb::CoordBBox bbox;
        iter.getBoundingBox(bbox);
        accessor.getTree().fill(bbox, 1.);
      }
    }
  };

  vto::transformValues(mask->cbeginValueOn(), *float_grid, Local::op);
*/

  auto float_grid = openvdb::FloatGrid::create();
  auto accessor = float_grid->getAccessor();
  // TODO speed up by iterating through leaves then values
  for (auto iter = mask->cbeginValueOn(); iter; ++iter) {
    accessor.setValue(iter.getCoord(), 1.);
  }
  return vto::fogToSdf(*float_grid, 0);
}

float get_radius(Util::AttribVec<Geometry::AMGraph::NodeID, CGLA::Vec3f> &node_color, size_t i) {
  auto color = node_color[i].get();
  // radius is in the green channel
  return color[1]; // RGB
}

// laplacian smoothing, controllable by number of repeated iterations
// and the smoothing strength (alpha) at each iteration
// a high number of iterations allows the smoothing effect to diffuse
// globally throughout the graph, whereas a low iteration count and a high
// alpha make the effect much more local
// 1 iteration and 1 alpha are the defaults in the GEL library and in the
// corresponding paper (Baerentzen) all skeletons qualitatively looked smooth
// after only 1 iteration at 1 alpha
void smooth_graph_pos_rad(Geometry::AMGraph3D &g, const int iter, const float alpha) {
  auto lsmooth = [](Geometry::AMGraph3D &g, float _alpha) {
    auto convert_radius = [](float radius) {
      return CGLA::Vec3f(0, radius, 0);
    };

    Util::AttribVec<Geometry::AMGraph::NodeID, CGLA::Vec3d> new_pos(
        g.no_nodes(), CGLA::Vec3d(0));
    Util::AttribVec<Geometry::AMGraph::NodeID, CGLA::Vec3f> new_radius(
        g.no_nodes(), CGLA::Vec3f(0));
    for (auto n : g.node_ids()) {
      double wsum = 0;
      auto N = g.neighbors(n);
      for (auto nn : N) {
        double w = 1.0;
        new_pos[n] += w * g.pos[nn];
        new_radius[n] += convert_radius(w * get_radius(g.node_color, nn));
        wsum += w;
      }
      double alpha = N.size() == 1 ? 0 : _alpha;
      new_pos[n] = (alpha)*new_pos[n] / wsum + (1.0 - alpha) * g.pos[n];
      new_radius[n] = convert_radius((alpha)*get_radius(new_radius, n) / wsum +
                                     (1.0 - alpha) * get_radius(g.node_color, n));
    }
    return std::make_pair(new_pos, new_radius);
  };

  for (int i = 0; i < iter; ++i) {
    auto [npos, nradius] = lsmooth(g, alpha);
    g.pos = npos;
    g.node_color = nradius;
  }
}

std::optional<std::pair<Geometry::AMGraph3D, std::vector<long unsigned int>>>
vdb_to_skeleton(openvdb::MaskGrid::Ptr mask, std::vector<Seed> component_seeds,
                int index, RecutCommandLineArgs *args,
                fs::path component_dir_fn, int threads) {
  auto component = mask_to_sdf(mask);
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

  // multi-scale is faster and scales linearly with input graph size at
  // the cost of difficulty in choosing a grow threshold
  Geometry::AMGraph3D component_graph;
  try {
    auto separators =
        multiscale_local_separators(g, Geometry::SamplingType::Advanced,
                                    args->skeleton_grow, args->skeleton_grain, 
                                    /*opt steps*/ 0, threads, 
                                    /*quiet*/true);
    auto ppair = skeleton_from_node_set_vec(g, separators);
    component_graph = ppair.first;
  } catch (...) {
    std::cerr << "Skeletonization failed for " << (component_dir_fn) << '\n';
    return std::nullopt;
  }
  
  // prune all leaf vertices (valency 1) whose only neighbor has valency > 2
  // as these tend to be spurious branches 
  Geometry::prune(component_graph);

  // TODO fix trifurcations

  smooth_graph_pos_rad(component_graph, args->smooth_iters, /*alpha*/ 1);

  // sweep through various soma ids
  auto soma_ids =
      force_soma_nodes(component_graph, component_seeds, args->soma_dilation);
  if (soma_ids.size() == 0) {
    std::cout << "Warning no soma_ids found for component " << index << '\n';
    // assign a soma randomly if none are found
    soma_ids.push_back(0);
  }
  graph_save(component_dir_fn / ("skeleton.graph"), component_graph);

  return std::make_pair(component_graph, soma_ids);
  
  /*
  std::vector<MyMarker *> component_tree;
  for (auto i : component_graph.node_ids()) {
    auto radius = get_radius(component_graph.node_color, i);
    auto pos = component_graph.pos[i].get();
    auto marker = new MyMarker(pos[0], pos[1], pos[2], radius);
    component_tree.push_back(marker);
  }

  std::vector<bool> visited(component_tree.size());

  // do BFS from each known soma in the component
  rng::for_each(soma_ids, [&](auto soma_id) {
    std::queue<size_t> q;
    q.push(soma_id);
    // mark soma
    auto marker = component_tree[soma_id];
    marker->parent = 0;
    marker->type = 0;
    visited[soma_id] = true;

    // traverse rest of tree
    size_t last = soma_id;
    while (q.size()) {
      size_t id = q.front();
      q.pop();
      for (auto nb_id : component_graph.neighbors(id)) {
        // skip other somas or visited
        if (!visited[nb_id] && std::find(soma_ids.begin(), soma_ids.end(),
                                         nb_id) == std::end(soma_ids)) {
          auto marker = component_tree[nb_id];
          // add current id as parent to all discovered
          marker->parent = component_tree[id];
          visited[nb_id] = true;
          q.push(nb_id);
        }
      }
    }
  });

  return component_tree;
  */
}

void write_swcs(Geometry::AMGraph3D component_graph, std::vector<long unsigned int> soma_ids,
                    std::array<float, 3> voxel_size,
                    std::filesystem::path component_dir_fn = ".",
                    CoordBBox bbox = {}, bool bbox_adjust = false,
                    bool is_eswc = false, bool voxel_units = false) {

  // each vertex in the graph has a single parent (id) which is
  // determined via BFS traversal
  std::vector<int> parent_table(component_graph.no_nodes(), -1);

  // do BFS from each known soma in the component
  rng::for_each(soma_ids, [&](auto soma_id) {
    // init q with soma
    std::queue<size_t> q;
    q.push(soma_id);

    // start swc and add header metadata
    auto pos = component_graph.pos[soma_id].get();
    auto file_name_base = "tree-with-soma-xyz-" +
                        std::to_string(pos[0]) + "-" +
                        std::to_string(pos[1]) + "-" +
                        std::to_string(pos[2]);
    auto coord_to_swc_id = get_id_map();
    std::ofstream swc_file;
    swc_file.open(component_dir_fn / (file_name_base + ".swc"));
    swc_file << "# Crop windows bounding volume: " << bbox << '\n'
           << "# id type_id x y z radius parent_id in units: " << (voxel_units ? "voxel" : "um") << '\n';

    // traverse rest of tree
    parent_table[soma_id] = soma_id; // a soma technically has no parent
    while (q.size()) {
      size_t id = q.front();
      q.pop();

      auto pos = component_graph.pos[id].get();
      auto radius = get_radius(component_graph.node_color, id);
      // can only be this trees root, not possible for somas from other trees to enter in to q
      auto is_root = id == soma_id;
      auto parent_id = parent_table[id];
      assertm(id == parent_id, "Soma improperly found");

      auto coord = std::array<double, 3>{pos[0], pos[1], pos[2]};
      std::array<double, 3> parent_coord;
      if (is_root) {
        parent_coord = coord;
      } else {
        auto lpos = component_graph.pos[parent_id].get();
        parent_coord = std::array<double, 3>{lpos[0], lpos[1], lpos[2]};
      }

      // expects an offset to a parent
      print_swc_line(coord, is_root, radius, parent_coord, bbox, swc_file,
                   coord_to_swc_id, voxel_size, bbox_adjust, false, voxel_units);

      // add all neighbors of current to q
      for (auto nb_id : component_graph.neighbors(id)) {
        // do not add other somas or previously visited to the q
        if (parent_table[nb_id] < 0 && std::find(soma_ids.begin(), soma_ids.end(),
                                         nb_id) == std::end(soma_ids)) {
          // add current id as parent to all neighbors
          parent_table[nb_id] = id; // permanent assignment of parent
          q.push(nb_id);
        }
      }
    }
  });
}

  /*
 if (is_eswc) {
    std::ofstream ano_file;
    ano_file.open(component_dir_fn / (file_name_base + ".ano"));
    ano_file << "APOFILE=" << file_name_base << ".ano.apo\n"
             << "SWCFILE=" << file_name_base << ".ano.eswc\n";
    ano_file.close();

    auto marker = tree[0];
    std::ofstream apo_file;
    apo_file.open(component_dir_fn / (file_name_base + ".ano.apo"));
    apo_file << std::fixed << std::setprecision(SWC_PRECISION);
    // 56630,,,,2452.761,4745.697,3057.039,
    // 0.000,0.000,0.000,314.159,0.000,,,,0,0,255
    apo_file
        << "##n,orderinfo,name,comment,z,x,y, "
           "pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b\n";
    // ...skip assigning a node id (n)
    apo_file << ',';
    // orderinfo,name,comment
    apo_file << ",,,";
    // z,x,y
    apo_file << voxel_size[2] * marker->z << ',' << voxel_size[0] * marker->x
             << ',' << voxel_size[1] * marker->y << ',';
    // pixmax,intensity,sdev,
    apo_file << "0.,0.,0.,";
    // volsize
    apo_file << marker->radius * marker->radius * marker->radius;
    // mass,,,, color_r,color_g,color_b
    apo_file << "0.,,,,0,0,255\n";
    apo_file.close();
    // iter those marker*

    std::ofstream eswc_file;
    eswc_file.open(component_dir_fn / (file_name_base + ".ano.eswc"));
    eswc_file << "# id type_id x y z radius parent_id"
              << " seg_id level mode timestamp TFresindex\n";

    coord_to_swc_id = get_id_map();
    rng::for_each(tree, [&](const auto marker) {
      auto coord = std::array<double, 3>{marker->x, marker->y, marker->z};
      auto is_root = marker->type == 0;
      auto parent_coord =
          is_root ? coord
                  : std::array<double, 3>{marker->parent->x, marker->parent->y,
                                          marker->parent->z};
      // expects an offset to a parent
      print_swc_line(coord, is_root, marker->radius, parent_coord, bbox,
                     eswc_file, coord_to_swc_id, voxel_size, bbox_adjust, true);
    });
  }
  */

