#pragma once

#include "tree_ops.hpp"
#include <GEL/Geometry/Graph.h>
#include <GEL/Geometry/graph_io.h>
#include <GEL/Geometry/graph_util.h>
#include <GEL/Geometry/graph_skeletonize.h>
#include <GEL/Geometry/KDTree.h>
#include <GEL/Util/AttribVec.h>
#include <openvdb/tools/FastSweeping.h> // fogToSdf
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/VolumeToMesh.h>

class Node { 
  public:
    CGLA::Vec3d pos;
    float radius;
};

using NodeID = Geometry::AMGraph3D::NodeID;

float get_radius(Util::AttribVec<NodeID, CGLA::Vec3f> &node_color, NodeID i) {
  auto color = node_color[i].get();
  // radius is in the green channel
  return color[1]; // RGB
}

CGLA::Vec3f convert_radius(float radius) {
  return {0, radius, 0};
}

Node get_node(Geometry::AMGraph3D g, NodeID i) {
  Node n{g.pos[i], get_radius(g.node_color, i)};
  return n;
}

auto euc_dist = [](auto a, auto b) -> float {
  std::array<float, 3> diff = {
    static_cast<float>(a[0]) - static_cast<float>(b[0]),
    static_cast<float>(a[1]) - static_cast<float>(b[1]),
    static_cast<float>(a[2]) - static_cast<float>(b[2])};
  return std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
};

// build kdtree, highly efficient for nearest point computations
Geometry::KDTree<CGLA::Vec3d, NodeID> build_kdtree(Geometry::AMGraph3D &graph) {
  Geometry::KDTree<CGLA::Vec3d, NodeID> tree;
  for (auto i : graph.node_ids()) {
    CGLA::Vec3d p0 = graph.pos[i];
    tree.insert(p0, i);
  }
  tree.build();
  return tree;
}

// find existing skeletal node within the radius of the soma
std::vector<NodeID> within_sphere(Node &node,
    Geometry::KDTree<CGLA::Vec3d, NodeID> &tree,
    float soma_dilation=1) {

  std::vector<CGLA::Vec3d> _;
  std::vector<NodeID> vals;
  tree.in_sphere(node.pos, soma_dilation * node.radius, _, vals);
  return vals;
}

// nodes with high valence and large radius have a bug with FEQ remesh
// so you may need to artificially set the radius to something small then
// merge an ideal sphere after meshing
// in use cases where the graph is not transformed into a mesh this is
// not necessary
void merge_local_radius(Geometry::AMGraph3D &graph, std::vector<Seed> &seeds,
    float soma_dilation=1., bool keep_radius_small=false) {
  // the graph is complete so build a data structure that
  // is fast at finding nodes within a 3D radial distance
  auto tree = build_kdtree(graph);

  std::set<NodeID> deleted_nodes;
  rng::for_each(seeds, [&](Seed seed) {
      auto original_pos = CGLA::Vec3d(seed.coord[0], seed.coord[1], seed.coord[2]);
      Node n{original_pos, static_cast<float>(seed.radius)};

      std::vector<NodeID> within_sphere_ids = within_sphere(n, tree, soma_dilation);

      auto new_seed_id = graph.merge_nodes(within_sphere_ids);

      // set the position and radius explicitly rather than averaging the merged nodes
      graph.pos[new_seed_id] = original_pos;

      graph.node_color[new_seed_id] = CGLA::Vec3f(0, keep_radius_small ? 1 : seed.radius, 0);
  });

  graph = Geometry::clean_graph(graph);
}

// add seeds passed as new nodes in the graph,
// nearby skeletal nodes are merged into the seeds,
// seeds inherit edges and delete nodes within 3D radius soma_dilation *
// seed.radius the original location and radius of the seeds are preserved
// returns coords since id's are invalidated by future mutations to graph
std::vector<GridCoord> force_soma_nodes(Geometry::AMGraph3D &graph,
    std::vector<Seed> &seeds,
    float soma_dilation) {

  auto soma_coords =
    seeds | rv::transform([](Seed seed) { return seed.coord; }) 
    | rng::to_vector;

  // add the known seeds to the skeletonized graph
  merge_local_radius(graph, seeds, soma_dilation);

  return soma_coords;
}

void check_soma_ids(NodeID nodes, std::vector<NodeID> soma_ids) {
  rng::for_each(soma_ids, [&](NodeID soma_id) {
      if (soma_id >= nodes) {
      throw std::runtime_error("Impossible soma id found");
      }
      });
}

std::vector<GridCoord> find_soma_nodes(Geometry::AMGraph3D &graph,
    std::vector<Seed> seeds, float soma_dilation, bool highest_valence=false) {

  auto tree = build_kdtree(graph);

  std::vector<GridCoord> soma_coords;
  rng::for_each(seeds, [&](Seed seed) {
      std::optional<NodeID> max_index;

      if (highest_valence) {
      auto original_pos = CGLA::Vec3d(seed.coord[0], seed.coord[1], seed.coord[2]);
      Node n{original_pos, static_cast<float>(seed.radius)};
      auto within_sphere_ids = within_sphere(n, tree, soma_dilation);

      // pick skeletal node within radii with highest number of edges (valence)
      int max_valence = 0;
      for (auto id : within_sphere_ids) {
      auto valence = graph.valence(id);
      if (valence > max_valence) {
      max_index = id;
      max_valence = valence;
      }
      }
      } else { // find closest point
      auto coord = seed.coord;
      CGLA::Vec3d p0(coord[0], coord[1], coord[2]);
      CGLA::Vec3d key;
      NodeID val;
      double dist = 1000;
      bool found = tree.closest_point(p0, dist, key, val);
      if (found)
        max_index = val;
      }

      if (max_index) {
        auto pos = graph.pos[max_index.value()];
        soma_coords.emplace_back(pos[0], pos[1], pos[2]);
      } else {
        std::cout << "Warning lost 1 seed during skeletonization\n";
      }
  });
  return soma_coords;
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

  Geometry::AMGraph3D g;
  rng::for_each(points, [&](auto point) {
      auto p = CGLA::Vec3d(point[0], point[1], point[2]);
      auto node_id = g.add_node(p);
      });

  // unroll_polygons(tris, g, 3);
  unroll_polygons(quads, g, 4);

  return g;
}

// naive multifurcation fix, force non-soma vertices to have at max 3 neighbors
Geometry::AMGraph3D fix_multifurcations(Geometry::AMGraph3D &graph,
    std::vector<GridCoord> soma_coords) {

  // loop over all vertices until no remaining multifurcations are found
  for (auto multifurc_id : graph.node_ids()) {
    auto pos = graph.pos[multifurc_id];
    auto current_coord = GridCoord(pos[0], pos[1], pos[2]);
    // if not a soma
    if (rng::find(soma_coords, current_coord) == rng::end(soma_coords)) {
      while (graph.valence(multifurc_id) > 3) { // is a multifurcation
        auto neighbors = graph.neighbors(multifurc_id);
        auto to_reattach = neighbors[0]; // picked at random
        auto to_extend = neighbors[1]; // picked at random

        // build a new averaged node
        CGLA::Vec3d pos1 = graph.pos[multifurc_id];
        CGLA::Vec3d pos2 = graph.pos[to_extend];
        auto rad = (get_radius(graph.node_color, multifurc_id) + 
          get_radius(graph.node_color, to_extend)) / 2;
        auto pos3 = CGLA::Vec3d((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2, (pos1[2] + pos2[2]) / 2);
        auto new_path_node = graph.add_node(pos3);
        graph.node_color[new_path_node] = convert_radius(rad);

        graph.connect_nodes(new_path_node, to_extend);
        graph.connect_nodes(new_path_node, multifurc_id);
        graph.connect_nodes(new_path_node, to_reattach);
        graph.disconnect_nodes(multifurc_id, to_reattach);
        graph.disconnect_nodes(multifurc_id, to_extend);
        // remove invalidated edges such that edge counts
        // are correct in future iterations
        graph = Geometry::clean_graph(graph);
      }
    }
  }
  return graph;
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
Geometry::AMGraph3D mesh_to_graph(HMesh::Manifold &m) {
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

// Taken directly from PyGEL library
std::pair<std::vector<openvdb::Vec3s>, std::vector<openvdb::Vec4I>> mesh_to_polygons(HMesh::Manifold &m) {
  HMesh::VertexAttributeVector<Geometry::AMGraph::NodeID> v2n;
  Geometry::AMGraph3D g;

  std::vector<openvdb::Vec3s> points;
  std::vector<openvdb::Vec4I> quads;

  for (auto v : m.vertices()) {
    auto pos = m.pos(v);
    points.emplace_back(pos[0], pos[1], pos[2]);
  }

  // iterate all polygons of manifold
  for (HMesh::FaceID poly : m.faces()) {
    // iterate all vertices of this polygon
    std::vector<unsigned int> poly_indices;
    for (auto v : m.incident_vertices(poly)) {
      poly_indices.push_back(static_cast<unsigned int>(v.get_index()));
    }

    if (poly_indices.size() == 4) {
      quads.emplace_back(poly_indices[0], poly_indices[1], poly_indices[2], poly_indices[3]);
    } else if (poly_indices.size() == 6) {
      quads.emplace_back(poly_indices[0], poly_indices[1], poly_indices[2], poly_indices[3]);
      quads.emplace_back(poly_indices[0], poly_indices[3], poly_indices[4], poly_indices[5]);
    } else if (poly_indices.size() == 8) {
      quads.emplace_back(poly_indices[0], poly_indices[1], poly_indices[2], poly_indices[3]);
      quads.emplace_back(poly_indices[3], poly_indices[4], poly_indices[7], poly_indices[0]);
      quads.emplace_back(poly_indices[4], poly_indices[5], poly_indices[6], poly_indices[7]);
    } else {
      throw std::runtime_error("Unimplemented how to quadrize a " + std::to_string(poly_indices.size()) + "-ary polygon");
    }
  }

  return std::make_pair(points, quads);
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

  /*
  test_multi_graph () {
    auto msg = Geometry::multiscale_graph(g, args->skeleton_grow, true);
    std::cout << "layers: " << msg.layers.size() << '\n';
    int i=0;
    for (auto layer : msg.layers) {
      std::cout << "layer " << i << '\n';
      std::cout << "  vertex count: " << layer.no_nodes() << '\n';
      timer.restart();
      auto separators =
        multiscale_local_separators(layer, Geometry::SamplingType::Advanced,
            args->skeleton_grow, args->skeleton_grain,
            0, threads,
            false);
      auto [component_graph, _] = skeleton_from_node_set_vec(layer, separators);
      std::cout << "  skeleton node count: " << component_graph.no_nodes() << '\n';
      std::cout << "  ls time: " << timer.elapsed() << '\n';
      //graph_save(component_dir_fn / ("skeleton" + i + ".graph"), component_graph);
      ++i;
    } }
  */

std::vector<NodeID> get_invalid_radii(Geometry::AMGraph3D &g) {
  g = Geometry::clean_graph(g);
  return rv::iota(0, static_cast<int>(g.no_nodes())) | 
    rv::transform([](auto i) { return static_cast<NodeID>(i); }) |
    rv::remove_if([&g](NodeID i) {
      auto rad = get_radius(g.node_color, i);
      return rad >= .001;
    }) 
    | rng::to_vector;
}

// set an invalid radii to be the average radii of its neighbors
void fix_invalid_radii(Geometry::AMGraph3D &g, std::vector<NodeID> invalids) {
  rng::for_each(invalids, [&](NodeID i) {
      auto nbs = g.neighbors(i);
      float radius = 0;
      rng::for_each(nbs, [&](NodeID nb) {
          radius += get_radius(g.node_color, nb);
      });
      g.node_color[i] = CGLA::Vec3f(0, radius / static_cast<float>(nbs.size()), 0);
  });
}

std::vector<NodeID> get_completely_within(NodeID i, Geometry::AMGraph3D &g) {
    // rebuilding the kdtree, per index is extremely inefficient,
    // however at each node, certain vertices are potentially being deleted
    // so this protects already deleted nodes from populating later searches
    auto tree = build_kdtree(g);

    auto n = get_node(g, i);
    auto radius = get_radius(g.node_color, i);
    std::vector<NodeID> within_sphere_ids = within_sphere(n, tree);
    // some nodes are partially within, others are completely within
    // the curren node
    return within_sphere_ids | rv::remove_if([&](NodeID j){ 
        return radius >= get_radius(g.node_color, j) + euc_dist(g.pos[i], g.pos[j]);
        }) | rng::to_vector;
}

void fix_node_within_another(Geometry::AMGraph3D &g) {
  for (NodeID i=0; i < g.no_nodes(); ++i) {
    auto completely_within = get_completely_within(i, g);

    // it's only legal to merge another node if it is also a neighbor
    // merging a neighbor can make a second, third, etc. degree neighbor
    // mergeable afterwards
    for (bool merged = false; merged; merged = false) {
      auto neighbors = g.neighbors(i);

      // protect from checking if a node is a neighbor of itself
      // keep nodes that are within and a directly linked neighbor only
      auto mergeables = completely_within | rv::remove_if([&](NodeID j) {
          return i == j || rng::find(neighbors, j) == rng::end(neighbors);
          }) | rng::to_vector; 

      // also merge this node if necessary
      mergeables.push_back(i);
      // FIXME seg fault here at merge nodes
      // probably due to j node already being invalidated
      // switch to merge_nodes(list) interface instead
      if (mergeables.size() > 1) {
        auto pos = g.pos[i];
        NodeID new_i = g.merge_nodes(mergeables);
        // restore original position of i
        // so that you don't create more within nodes
        g.pos[new_i] = pos;
        g = Geometry::clean_graph(g);
        merged = true;
      }
    }
  }
}

// throws if 1 node is a neighbor of another and one of them is
// completely within the other 
int count_nodes_within_another(Geometry::AMGraph3D &g) {
  int count=0;
  for (NodeID i=0; i < g.no_nodes(); ++i) {
    auto completely_within = get_completely_within(i, g);

    auto neighbors = g.neighbors(i);
    for (auto j : completely_within) {
      if (i != j && rng::find(neighbors, j) != rng::end(neighbors))
        ++count;
    }
  }
  return count;
}

std::optional<std::pair<Geometry::AMGraph3D, std::vector<GridCoord>>>
vdb_to_skeleton(openvdb::FloatGrid::Ptr component, std::vector<Seed> component_seeds,
    int index, RecutCommandLineArgs *args,
    fs::path component_dir_fn, std::ofstream& component_log, int threads, bool save_graphs = false) {

  auto timer = high_resolution_timer();
  auto g = vdb_to_graph(component, args);
  component_log << "vdb to graph, " << timer.elapsed() << '\n';

  if (args->coarsen_steps) {
    timer.restart();
    auto msg = Geometry::multiscale_graph(g, args->skeleton_grow, true);
    auto last_layer_index = msg.layers.size() - 1;
    auto layer_index = args->coarsen_steps.value() > last_layer_index ? last_layer_index : args->coarsen_steps.value();
    g = msg.layers[layer_index];
    component_log << "coarsen, " << timer.elapsed() << '\n';
  }

  if (args->saturate_edges) {
    timer.restart();
    Geometry::saturate_graph(g, args->saturate_edges.value());
    component_log << "saturate edges, " << timer.elapsed() << '\n';
  }

  if (save_graphs)
    graph_save(component_dir_fn / ("mesh.graph"), g);

  timer.restart();
  // multi-scale is faster and scales linearly with input graph size at
  // the cost of difficulty in choosing a grow threshold
  auto separators =
    multiscale_local_separators(g, Geometry::SamplingType::Advanced,
        args->skeleton_grow, args->skeleton_grain,
        /*opt steps*/ args->optimize_steps, threads,
        false);
  auto [component_graph, _] = skeleton_from_node_set_vec(g, separators);
  component_log << "msls, " << timer.elapsed() << '\n';

  // prune all leaf vertices (valency 1) whose only neighbor has valency > 2
  // as these tend to be spurious branches
  Geometry::prune(component_graph);

  smooth_graph_pos_rad(component_graph, args->smooth_steps, /*alpha*/ 1);

  // sweep through various soma ids
  std::vector<GridCoord> soma_coords;
  if (args->seed_action == "force")
    soma_coords = force_soma_nodes(component_graph, component_seeds, args->soma_dilation.value());
  else if (args->seed_action == "find")
    soma_coords = find_soma_nodes(component_graph, component_seeds, args->soma_dilation.value());
  else if (args->seed_action == "find-valent")
    soma_coords = find_soma_nodes(component_graph, component_seeds, args->soma_dilation.value(), true);

  if (soma_coords.size() == 0) {
    std::cerr << "Warning no soma_coords found for component " << index << '\n';
    return std::nullopt;
  }

  //int illegal_nodes = count_nodes_within_another(component_graph);
  //component_log << "Original within nodes, " << illegal_nodes << '\n';
  //if (illegal_nodes)
    //fix_node_within_another(component_graph);
  //component_log << "Post-fix within nodes, " << count_nodes_within_another(component_graph) << '\n';

  // multifurcations are only important for rules of SWC standard
  component_graph = fix_multifurcations(component_graph, soma_coords);
  //component_log << "Final within nodes, " << count_nodes_within_another(component_graph) << '\n';
  auto invalids = get_invalid_radii(component_graph);
  if (invalids.size() > 0) {
    component_log << "Invalid radii, " << invalids.size() << '\n';
    fix_invalid_radii(component_graph, invalids);
    invalids = get_invalid_radii(component_graph);
    component_log << "Final invalid radii, " << invalids.size() << '\n';
  }

  if (save_graphs)
    graph_save(component_dir_fn / ("skeleton.graph"), component_graph);

  return std::make_pair(component_graph, soma_coords);
}

void write_ano_file(fs::path component_dir_fn, std::string file_name_base) {
  std::ofstream ano_file;
  ano_file.open(component_dir_fn / (file_name_base + ".ano"));
  ano_file << "APOFILE=" << file_name_base << ".ano.apo\n"
    << "SWCFILE=" << file_name_base << ".ano.eswc\n";
  ano_file.close();
}

void write_apo_file(fs::path component_dir_fn, std::string file_name_base, std::array<double, 3> pos,
    float radius, std::array<double, 3> voxel_size) {
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
  apo_file << voxel_size[2] * pos[2] << ',' << voxel_size[0] * pos[0]
    << ',' << voxel_size[1] * pos[1] << ',';
  // pixmax,intensity,sdev,
  apo_file << "0.,0.,0.,";
  // volsize
  apo_file << radius * radius * radius;
  // mass,,,, color_r,color_g,color_b
  apo_file << "0.,,,,0,0,255\n";
  apo_file.close();
}

std::string swc_name(Node &n, std::array<double, 3> voxel_size) {
  std::ostringstream out;
  out <<  std::fixed << std::setprecision(SWC_PRECISION);
  out << '[';
  out << n.pos[0] * voxel_size[0] << ',';
  out << n.pos[1] * voxel_size[1] << ',';
  out << n.pos[2] * voxel_size[2] << ']';
  out << "-r=" << n.radius << "-µm";
  return out.str();
}

void write_swcs(Geometry::AMGraph3D component_graph, std::vector<GridCoord> soma_coords,
    std::array<double, 3> voxel_size,
    std::filesystem::path component_dir_fn = ".",
    CoordBBox bbox = {}, bool bbox_adjust = false,
    bool is_eswc = false, bool disable_swc_scaling = false) {

  // each vertex in the graph has a single parent (id) which is
  // determined via BFS traversal
  std::vector<NodeID> parent_table(component_graph.no_nodes(), -1);

  // scan the graph to find the final set of soma_ids
  std::vector<NodeID> soma_ids;
  for (NodeID i=0; i < component_graph.no_nodes(); ++i) {
    auto pos = component_graph.pos[i];
    auto current_coord = GridCoord(pos[0], pos[1], pos[2]);
    if (std::find(soma_coords.begin(),
          soma_coords.end(), current_coord) != std::end(soma_coords))
      soma_ids.push_back(i);
  }

  // do BFS from each known soma in the component
  for (auto soma_id : soma_ids) {
      // init q with soma
      std::queue<NodeID> q;
      q.push(soma_id);

      // start swc and add header metadata
      auto pos = component_graph.pos[soma_id];
      Node n{pos, get_radius(component_graph.node_color, soma_id)};
      auto file_name_base = swc_name(n, voxel_size);
      auto coord_to_swc_id = get_id_map();

      // traverse rest of tree
      parent_table[soma_id] = soma_id; // a soma technically has no parent
                                       // start file per soma, write header info
      std::ofstream swc_file;
      if (is_eswc) {

      auto soma_pos = component_graph.pos[soma_id].get();
      auto soma_coord = std::array<double, 3>{pos[0], pos[1], pos[2]};
      auto soma_radius = get_radius(component_graph.node_color, soma_id);

      write_apo_file(component_dir_fn, file_name_base, soma_coord, soma_radius, voxel_size);
      write_ano_file(component_dir_fn, file_name_base);

      swc_file.open(component_dir_fn / (file_name_base + ".ano.eswc"));
      swc_file << "# id type_id x y z radius parent_id"
        << " seg_id level mode timestamp TFresindex\n";
      } else {
        swc_file.open(component_dir_fn / (file_name_base + ".swc"));
        swc_file << "# Crop windows bounding volume: " << bbox << '\n'
          << "# id type_id x y z radius parent_id in units: " << (disable_swc_scaling ? "voxel" : "um") << '\n';
      }

      while (q.size()) {
        NodeID id = q.front();
        q.pop();

        auto pos = component_graph.pos[id].get();
        auto radius = get_radius(component_graph.node_color, id);
        // can only be this trees root, not possible for somas from other trees to enter in to q
        auto is_root = id == soma_id;
        auto parent_id = parent_table[id];

        auto coord = std::array<double, 3>{pos[0], pos[1], pos[2]};
        std::array<double, 3> parent_coord;
        if (is_root) {
          parent_coord = coord;
        } else {
          auto lpos = component_graph.pos[parent_id].get();
          parent_coord = std::array<double, 3>{lpos[0], lpos[1], lpos[2]};
        }

        print_swc_line(coord, is_root, radius, parent_coord, bbox, swc_file,
            coord_to_swc_id, voxel_size, bbox_adjust, is_eswc, disable_swc_scaling);

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
  }

  //for (auto parent : parent_table) {
  //if (parent < 0)
  //std::cerr << "Error: graph to tree lost vertex, report this issue\n";
  //}
}

std::pair<Seed, Geometry::AMGraph3D>
swc_to_graph(filesystem::path marker_file, std::array<double, 3> voxel_size,
    GridCoord image_offsets = zeros(), bool save_file = false) {
  ifstream ifs(marker_file);
  if (ifs.fail()) {
    throw std::runtime_error("Unable to open marker file " + marker_file.string());
  }

  auto min_voxel_size = min_max(voxel_size).first;
  std::vector<Seed> seeds;
  Geometry::AMGraph3D g;
  NodeID count = 0;
  while (ifs.good()) {
    if (ifs.peek() == '#' || ifs.eof()) {
      ifs.ignore(1000, '\n');
      continue;
    }
    NodeID id, type, parent_id;
    double x_um,y_um,z_um,radius_um;
    ifs >> id;
    ifs.ignore(10, ' ');
    ifs >> type;
    ifs.ignore(10, ' ');
    ifs >> x_um;
    ifs.ignore(10, ' ');
    ifs >> y_um;
    ifs.ignore(10, ' ');
    ifs >> z_um;
    ifs.ignore(10, ' ');
    ifs >> radius_um;
    ifs.ignore(10, ' ');
    ifs >> parent_id;
    ifs.ignore(1000, '\n');
    parent_id -= 1; // need to adjust to 0-indexed

    //translate by image offsets only for data originating from windowed runs 
    x_um += image_offsets[0];
    y_um += image_offsets[1];
    z_um += image_offsets[2];

    // translate from um units into integer voxel units
    double x,y,z,radius;
    radius = std::round(radius_um / min_voxel_size);
    x = std::round(x_um / voxel_size[0]);
    y = std::round(y_um / voxel_size[1]);
    z = std::round(z_um / voxel_size[2]);
    auto p = CGLA::Vec3d(x, y, z);
    auto coord = GridCoord(x, y, z);

    // add it to the graph
    auto node_id = g.add_node(p);
    g.node_color[node_id] = CGLA::Vec3f(0, radius, 0);

    // somas are nodes that have a parent of -1 (adjusted to -2) or have an index
    // of themselves
    if (parent_id == -2 || parent_id == node_id) {
      auto volume = static_cast<uint64_t>((4 / 3) * PI * std::pow(radius, 3));
      std::array<double, 3> coord_um{x_um, y_um, z_um};
      seeds.emplace_back(coord, coord_um, radius, radius_um, volume);
    } else {
      g.connect_nodes(node_id, parent_id);
    }
  }

  if (seeds.size() != 1) {
    throw std::runtime_error("Warning: SWC files are trees which by definition must have only 1 root (soma), provided file: " + 
        marker_file.generic_string() + " has " + std::to_string(seeds.size()));
  }
  
  if (save_file)
    graph_save(marker_file.stem().string() + ".graph", g);

  return std::make_pair(seeds.front(), g);
}

openvdb::FloatGrid::Ptr skeleton_to_surface(Geometry::AMGraph3D skeleton) {

  // feq requires an explicit radii vector
  std::vector<double> radii;
  radii.reserve(skeleton.no_nodes());
  for (NodeID i = 0; i < skeleton.no_nodes(); ++i) {
    radii.push_back(get_radius(skeleton.node_color, i));
  }

  // generate a mesh manifold then create a list of points and
  // polygons (quads or tris)
  std::cout << "Start graph to manifold" << '\n';
  auto manifold = graph_to_FEQ(skeleton, radii);

  HMesh::obj_save("mesh.obj", manifold);

  // translate manifold into something vdb will understand
  // points ands quads
  std::cout << "Start manifold to polygons" << '\n';
  auto [points, quads] = mesh_to_polygons(manifold);

  // convert the mesh into a surface (level set) vdb
  std::cout << "Start polygons to level set" << '\n';
  vto::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(points, quads);
  return vto::meshToVolume<openvdb::FloatGrid>(mesh, *get_transform());
}

// returns a polygonal mesh
openvdb::FloatGrid::Ptr swc_to_segmented(filesystem::path marker_file,
    std::array<double, 3> voxel_size, GridCoord image_offsets, bool save_vdbs = false, 
    std::string name = "") {
  auto [seed, skeleton] = swc_to_graph(marker_file, voxel_size, image_offsets);
  std::cout << seed << '\n';
  std::vector<Seed> seeds{seed};

  auto invalids = get_invalid_radii(skeleton);
  if (invalids.size()) 
    fix_invalid_radii(skeleton, invalids);

  merge_local_radius(skeleton, seeds, 1, true);

  if (save_vdbs)
    graph_save(marker_file.stem().string() + ".graph", skeleton);

  if (false) {
    auto soma_coords = rv::transform(seeds, [](Seed seed) {
        return seed.coord;
        }) | rng::to_vector;
    write_swcs(skeleton, soma_coords, voxel_size, "test");
  }

  auto level_set = skeleton_to_surface(skeleton);

  // merge soma on top
  {
    // only 1 soma allowed per swc
    auto seed = seeds[0];
    auto soma_sdf = vto::createLevelSetSphere<openvdb::FloatGrid>(
       seed.radius, seed.coord.asVec3s(), 1.,
       RECUT_LEVEL_SET_HALF_WIDTH);
    vto::csgUnion(*level_set, *soma_sdf); // empties merged_somas grid into ls
  }

  if (save_vdbs)
    write_vdb_file({level_set}, "surface-" + (name.empty() ? marker_file.stem().string() : name) + ".vdb");

  return level_set;
}

// Get the total count of interior voxels of a level set (surface)
uint64_t voxel_count(openvdb::FloatGrid::Ptr level_set) {
  // convert the surface into a segmented (filled) foreground grid
  openvdb::BoolGrid::Ptr enclosed = vto::extractEnclosedRegion(*level_set);
  return enclosed->activeVoxelCount();
}

void calculate_recall_precision(openvdb::FloatGrid::Ptr truth, 
    openvdb::FloatGrid::Ptr test, bool save_vdbs = true) {
  double recall, precision_d, f1, iou;

  std::cout << "truth active voxel count, " << voxel_count(truth) << '\n';
  std::cout << "test active voxel count, " << voxel_count(test) << '\n';

  // calculate true positive pixels
  // this is total count of matching truth and test pixels
  auto true_positive = vto::csgIntersectionCopy(*truth, *test);
  if (save_vdbs)
    write_vdb_file({true_positive}, "surface-true-positive.vdb");
  auto true_positive_count = voxel_count(true_positive);

  // calculate the count of false positive pixels
  auto false_positive = vto::csgDifferenceCopy(*test, *true_positive);
  if (save_vdbs)
    write_vdb_file({false_positive}, "surface-false-positive.vdb");
  auto false_positive_count = voxel_count(false_positive);

  // calculate the count of false positive pixels
  auto false_negative = vto::csgDifferenceCopy(*truth, *true_positive);
  if (save_vdbs)
    write_vdb_file({false_negative}, "surface-false-negative.vdb");
  auto false_negative_count = voxel_count(false_negative);

  // calculation union
  auto union_ls = vto::csgUnionCopy(*truth, *test);
  auto union_count = voxel_count(union_ls);

  recall = true_positive_count / (true_positive_count + false_negative_count);
  precision_d = true_positive_count / (true_positive_count + false_positive_count);
  f1 = 2 * precision_d * recall / (precision_d + recall);
  iou = true_positive_count / union_count;

  std::cout << "true positive count, " << true_positive_count << '\n';
  std::cout << "false positive count, " << false_positive_count << '\n';
  std::cout << "false negative count, " << false_negative_count << '\n';
  std::cout << "recall, " << recall << '\n';
  std::cout << "precision, " << precision_d << '\n';
  std::cout << "F1, " << f1 << '\n';
  std::cout << "IoU, " << iou << '\n';
}
