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
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/VolumeToMesh.h>

using AMGraph3D = Geometry::AMGraph3D;
using NodeID = AMGraph3D::NodeID;
using NodeSet = AMGraph3D::NodeSet;
using Pos = CGLA::Vec3d;
using KDTree = Geometry::KDTree<Pos, NodeID>;

class Node { 
  public:
    Pos pos;
    float radius;

    friend std::ostream &operator<<(std::ostream &os, const Node &n) {
      os << std::fixed << std::setprecision(SWC_PRECISION);
      os << "[" << std::to_string(n.pos[0]) << ", " +
                std::to_string(n.pos[1]) + ", " +
                std::to_string(n.pos[2]) +
                "], radius: " + std::to_string(n.radius) + '\n';
      return os;
    }

};

float get_radius(const AMGraph3D &g, NodeID i) {
  auto color = g.node_color[i].get();
  // radius is in the green channel
  return color[1]; // RGB
}

CGLA::Vec3f convert_radius(float radius) {
  // radius is in the green channel
  return {0, radius, 0};
}

void set_radius(AMGraph3D &g, float v, NodeID i) {
  g.node_color[i] = convert_radius(v);
}

Node get_node(const AMGraph3D &g, NodeID i) {
  Node n{g.pos[i], get_radius(g, i)};
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
KDTree build_kdtree(const AMGraph3D &graph) {
  KDTree tree;
  for (auto i : graph.node_ids()) {
    Pos p0 = graph.pos[i];
    tree.insert(p0, i);
  }
  tree.build();
  return tree;
}

// find existing skeletal node within the radius of the soma
std::vector<NodeID> within_sphere(const Node &node,
    const KDTree &tree,
    float soma_dilation=1) {

  std::vector<Pos> _;
  std::vector<NodeID> vals;
  tree.in_sphere(node.pos, soma_dilation * node.radius, _, vals);
  return vals;
}

// nodes with high valence and large radius have a bug with FEQ remesh
// so you may need to artificially set the radius to something small then
// merge an ideal sphere after meshing
// in use cases where the graph is not transformed into a mesh this is
// not necessary
std::vector<NodeID> merge_local_radius(AMGraph3D &graph, std::vector<Node> &nodes,
    float soma_dilation=1., bool keep_radius_small=false) {
  // the graph is complete so build a data structure that
  // is fast at finding nodes within a 3D radial distance
  auto tree = build_kdtree(graph);

  auto ids = nodes | rv::transform([&](const Node &node) {
      auto original_pos = Pos(node.pos[0], node.pos[1], node.pos[2]);
      Node n{original_pos, static_cast<float>(node.radius)};

      std::vector<NodeID> within_sphere_ids = within_sphere(n, tree, soma_dilation);

      auto new_seed_id = graph.merge_nodes(within_sphere_ids);

      // set the position and radius explicitly rather than averaging the merged nodes
      graph.pos[new_seed_id] = original_pos;

      set_radius(graph, keep_radius_small ? 1 : node.radius, new_seed_id);
      return new_seed_id;
  }) | rng::to_vector;

  graph.cleanup();
  return ids;
}

auto to_coord = [](Pos p) {
  GridCoord c;
  for (int i=0; i < 3; ++i)
    c[i] = static_cast<int>(std::round(p[i]));
  return c;
};

auto to_node = [](Seed seed) {
  // no need to round since int -> double
  Pos p{static_cast<double>(seed.coord[0]),
    static_cast<double>(seed.coord[1]), static_cast<double>(seed.coord[2])}; 
  return Node{p, seed.radius};
};

// add seeds passed as new nodes in the graph,
// nearby skeletal nodes are merged into the seeds,
// seeds inherit edges and delete nodes within 3D radius soma_dilation *
// seed.radius the original location and radius of the seeds are preserved
// returns coords since id's are invalidated by future mutations to graph
std::vector<GridCoord> force_soma_nodes(AMGraph3D &graph,
    std::vector<Seed> &seeds,
    float soma_dilation) {

  auto nodes = seeds | rv::transform(to_node) 
    | rng::to_vector;

  // add the known seeds to the skeletonized graph
  merge_local_radius(graph, nodes, soma_dilation);

  return seeds | rv::transform(&Seed::coord) | rng::to_vector;
}

void check_soma_ids(NodeID nodes, std::vector<NodeID> soma_ids) {
  rng::for_each(soma_ids, [&](NodeID soma_id) {
      if (soma_id >= nodes) {
      throw std::runtime_error("Impossible soma id found");
      }
      });
}

std::vector<GridCoord> find_soma_nodes(AMGraph3D &graph,
    std::vector<Seed> seeds, float soma_dilation, bool highest_valence=false) {

  auto tree = build_kdtree(graph);

  std::vector<GridCoord> soma_coords;
  rng::for_each(seeds, [&](Seed seed) {
    std::optional<NodeID> max_index;

    if (highest_valence) {
      auto original_pos = Pos(seed.coord[0], seed.coord[1], seed.coord[2]);
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

      if (max_index) { //found
        Node n {graph.pos[max_index.value()], get_radius(graph, max_index.value())};
        std::vector nodes{n};
        merge_local_radius(graph, nodes, soma_dilation);
      }
    } else { // find closest point
      auto coord = seed.coord;
      Pos p0(coord[0], coord[1], coord[2]);
      Pos key;
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
auto unroll_polygons(std::vector<Poly> polys, AMGraph3D &g,
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

AMGraph3D vdb_to_graph(openvdb::FloatGrid::Ptr component,
    RecutCommandLineArgs *args) {
  std::vector<openvdb::Vec3s> points;
  // quad index list, which can be post-processed to
  // find a triangle mesh
  std::vector<openvdb::Vec4I> quads;
  std::vector<openvdb::Vec3I> tris;
  vto::volumeToMesh(*component, points, quads);
  // vto::volumeToMesh(*component, points, tris, quads, 0, args->mesh_grain);

  AMGraph3D g;
  rng::for_each(points, [&](auto point) {
      auto p = Pos(point[0], point[1], point[2]);
      auto node_id = g.add_node(p);
      });

  // unroll_polygons(tris, g, 3);
  unroll_polygons(quads, g, 4);

  return g;
}

// naive multifurcation fix, force non-soma vertices to have at max 3 neighbors
AMGraph3D fix_multifurcations(AMGraph3D &graph,
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
        Pos pos1 = graph.pos[multifurc_id];
        Pos pos2 = graph.pos[to_extend];
        auto rad = (get_radius(graph, multifurc_id) + 
          get_radius(graph, to_extend)) / 2;
        auto pos3 = Pos((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2, (pos1[2] + pos2[2]) / 2);
        auto new_path_node = graph.add_node(pos3);
        set_radius(graph, rad, new_path_node);

        graph.connect_nodes(new_path_node, to_extend);
        graph.connect_nodes(new_path_node, multifurc_id);
        graph.connect_nodes(new_path_node, to_reattach);
        graph.disconnect_nodes(multifurc_id, to_reattach);
        graph.disconnect_nodes(multifurc_id, to_extend);
        // remove invalidated edges such that edge counts
        // are correct in future iterations
        graph.cleanup();
      }
    }
  }
  return graph;
}

void scale_radii(AMGraph3D &g, double scaling_factor, std::vector<GridCoord> exclude_coords) {
  for (auto id : g.node_ids()) {
    auto pos = g.pos[id];
    auto current_coord = GridCoord(pos[0], pos[1], pos[2]);
    if (rng::find(exclude_coords, current_coord) == rng::end(exclude_coords)) {
      set_radius(g, get_radius(g, id) * scaling_factor, id);
    }
  }
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
  // FIXME these shoudl match the rest of the API with double not float
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
AMGraph3D mesh_to_graph(HMesh::Manifold &m) {
  HMesh::VertexAttributeVector<NodeID> v2n;
  AMGraph3D g;

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
  HMesh::VertexAttributeVector<NodeID> v2n;
  AMGraph3D g;

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
     this has a weird bug but is the canonical and fast way to do it
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
// above 1 alpha caused zig-zagged behavior that was unusable
// additionally pin the position of critical nodes those with valence > 2
// like branches and somas
// leaf vertices are already pinned in radius and pos
void smooth_graph_pos_rad(AMGraph3D &g, const int iter, const float alpha) {
  auto lsmooth = [](AMGraph3D &g, float _alpha) {

    Util::AttribVec<NodeID, Pos> new_pos(
        g.no_nodes(), Pos(0));
    Util::AttribVec<NodeID, CGLA::Vec3f> new_radius(
        g.no_nodes(), CGLA::Vec3f(0));

    auto radius = [](auto attr, auto n) {
      auto color = attr[n].get();
      // radius is in the green channel
      return color[1]; // RGB
    };

    for (auto n : g.node_ids()) {
      double wsum = 0;
      auto N = g.neighbors(n);
      for (auto nn : N) {
        double w = 1.0;
        new_pos[n] += w * g.pos[nn];
        new_radius[n] += convert_radius(w * get_radius(g, nn));
        wsum += w;
      }
      double alpha = N.size() == 1 ? 0 : _alpha;
      new_pos[n] = (alpha)*new_pos[n] / wsum + (1.0 - alpha) * g.pos[n];
      new_radius[n] = convert_radius((alpha)*radius(new_radius, n) / wsum +
          (1.0 - alpha) * get_radius(g, n));
    }
    return std::make_pair(new_pos, new_radius);
  };

  for (int i = 0; i < iter; ++i) {
    auto [npos, nradius] = lsmooth(g, alpha);
    // leaf vertices are already pinned in radius and pos above if you
    // observe the weighting scheme
    if (g.valence(i) < 3)
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
      std::cout << "  ls time: " << timer.elapsed_formatted() << '\n';
      //graph_save(component_dir_fn / ("skeleton" + i + ".graph"), component_graph);
      ++i;
    } }
  */

std::vector<NodeID> get_invalid_radii(const AMGraph3D &g) {
  return rv::iota(0, static_cast<int>(g.no_nodes())) | 
    rv::transform([](auto i) { return static_cast<NodeID>(i); }) |
    rv::remove_if([&g](NodeID i) {
      auto rad = get_radius(g, i);
      return rad >= .001;
    }) 
    | rng::to_vector;
}

// set an invalid radii to be the average radii of its neighbors
void fix_invalid_radii(AMGraph3D &g, std::vector<NodeID> invalids) {
  rng::for_each(invalids, [&](NodeID i) {
      auto nbs = g.neighbors(i);
      float radius = 0;
      rng::for_each(nbs, [&](NodeID nb) {
          radius += get_radius(g, nb);
      });
      set_radius(g, radius / static_cast<float>(nbs.size()), i);
  });
}

std::vector<NodeID> get_completely_within(const AMGraph3D &g, NodeID current, const KDTree &kdtree) {
    auto n = get_node(g, current);
    auto radius = get_radius(g, current);
    std::vector<NodeID> within_sphere_ids = within_sphere(n, kdtree);
    // keep a list of node ids that can't escape (are completely within) the radius 
    // of the current node, these nodes will be deleted since their
    // volumetric contributions are 0
    return within_sphere_ids | rv::filter([&](NodeID nb){ 
        // keep the nbs that are within current
        return radius >= get_radius(g, nb) + euc_dist(g.pos[current], g.pos[nb]);
        }) | rng::to_vector;
}

bool in(std::set<NodeID> invalidated, NodeID i) {
  return invalidated.count(i) != 0;
}

void same_position(const AMGraph3D &g) {
  for (auto i : g.node_ids()) {
    auto pos = g.pos[i];
    for (auto j : g.node_ids()) {
      if (i != j && pos == g.pos[j]) {
        std::cout << "i " << i << " pos " << pos << '\n';
        std::cout << " j " << j << " pos " << pos << '\n';
        auto nbs = g.neighbors(i);
        std::cout << rv::all(nbs) << '\n';
        nbs = g.neighbors(j);
        std::cout << rv::all(nbs) << '\n';
        throw std::runtime_error("Position already filled by another");
      }
    }
  }
}

void fix_same_position(AMGraph3D &g) {
  for (auto i : g.node_ids()) {
    auto pos = g.pos[i];
    for (auto j : g.node_ids()) {
      if (i != j && pos == g.pos[j]) {
        auto nbs = g.neighbors(i);
        auto nbs2 = g.neighbors(j);
        if (nbs == nbs2)
          g.merge_nodes(i, j); 
      }
    }
  }
}

void fix_node_within_another(AMGraph3D &g, std::set<NodeID> 
    enclosers) {

  // iterate the set of nodes that are known to enclose at least 1 other
  // node, it's possible that some enclosers enclose eachother
  // since each iteration deletes covered nodes, its possible other
  // enclosers are invalidated, therefore a list of previously deleted
  // nodes must be kept to protect from accessing invalid (deleted) nodes
  // from a previous iteration
  std::set<NodeID> invalidated;
  for (NodeID encloser : enclosers) {
    // protect from previously deleted enclosers 
    if (in(invalidated, encloser)) continue;

    // save known pos/rad of this node
    auto radius = get_radius(g, encloser);
    auto pos = g.pos[encloser];

    // rebuilding the kdtree, per index is inefficient,
    // however at each node, certain vertices are potentially being deleted
    // so this protects already deleted nodes from populating later searches
    auto kdtree = build_kdtree(g);

    auto completely_within = get_completely_within(g, encloser, kdtree);

    // it's only legal to merge another node if it is also a neighbor
    // merging a neighbor can make a second, third, etc. degree neighbor
    // mergeable afterwards
    bool performed_a_merge;
    do {
      performed_a_merge = false;
      auto neighbors = g.neighbors(encloser);

      // protect from checking if a node is a neighbor of itself
      // which causes UB
      // keep nodes that are within and a directly linked neighbor only
      auto mergeables = completely_within | rv::remove_if([&](NodeID within) { return encloser == within; }) 
        | rv::remove_if([&](NodeID within) { return in(invalidated, within); })
        | rv::remove_if([&](NodeID within) { return rng::find(neighbors, within) == rng::end(neighbors); })
        | rng::to_vector; 

      // safe to merge encloser now
      mergeables.push_back(encloser);

      // if there are other mergeable nodes besides encloser
      if (mergeables.size() > 1) {
        NodeID new_encloser = g.merge_nodes(mergeables);
        // the enclosers are the ground truth, but merge nodes
        // sets an averaged pos and radii, instead:
        // restore original position and radius of encloser
        // so that you don't create more within nodes
        g.pos[new_encloser] = pos;
        set_radius(g, radius, new_encloser);
        // keep track of invalidated nodes so they don't cause errors
        // in subsequent iterations
        for (auto merged : mergeables)
          invalidated.insert(merged);
        performed_a_merge = true;
      }
    } while (performed_a_merge);
  }
  // removes invalidated nodes, also invalidates all previous node ids
  g.cleanup();
}

std::set<NodeID> count_nodes_within_another(const AMGraph3D &g) {

  auto kdtree = build_kdtree(g);

  std::set<NodeID> enclosers;
  for (NodeID i : g.node_ids()) {
    auto completely_within = get_completely_within(g, i, kdtree);

    auto neighbors = g.neighbors(i);
    for (auto j : completely_within) {
      if (i != j && rng::find(neighbors, j) != rng::end(neighbors)) {
        //std::cout << get_node(g, i);
        //std::cout << '\t' << get_node(g, j);
        //std::cout << '\t' << euc_dist(g.pos[i], g.pos[j]) << '\n';
        // keep track that node i, encloses other nodes
        enclosers.insert(i);
      }
    }
  }
  return enclosers;
}

// modifies g such that no edges are longer than max_edge...
// the graph and the max edge distance are in voxel units
void resample(AMGraph3D &g, double max_edge_distance_voxels) {
  bool found_longer_edge = true;
  while (found_longer_edge) {
    found_longer_edge = false;
    for (int i=0; i < g.no_nodes(); ++i) {
      auto current_pos = g.pos[i];
      auto nbs = g.neighbors(i);
      for (auto nb : nbs) {
        auto nb_pos = g.pos[nb];
        auto dist = euc_dist(current_pos, nb_pos);
        if (dist > max_edge_distance_voxels) {
          found_longer_edge = true;

          // create an averaged pos new node
          auto new_pos = current_pos + nb_pos;
          new_pos *= .5;
          auto new_node_id = g.add_node(new_pos);

          // get an average radius
          auto new_radius = (get_radius(g, i) + get_radius(g, nb)) / 2;
          set_radius(g, new_radius, new_node_id);

          // remove edge to eachother
          g.disconnect_nodes(i, nb);

          // connect the new node between the others
          g.connect_nodes(new_node_id, i);
          g.connect_nodes(new_node_id, nb);
        }
      }
    }
  }
}

// sampling rate along neurites and the smoothing achieved are quite related
// and you need to keep the behavior of native outputs and proofreads as similar
// and standardized as possible, that's why these are bound into a single function
// call to keep the behavior more coupled throughout calls from recut
void standardize_sampling_smoothing(AMGraph3D& g, const std::array<double,3> 
    voxel_size, const int iter) {
  resample(g, MAX_EDGE_LENGTH_UM / voxel_size[0]);
  // don't change the alpha it leads to poor results
  smooth_graph_pos_rad(g, iter, /*alpha*/ 1);
}

std::optional<std::pair<AMGraph3D, std::vector<GridCoord>>>
vdb_to_skeleton(openvdb::FloatGrid::Ptr component, std::vector<Seed> component_seeds,
    int index, RecutCommandLineArgs *args,
    fs::path component_dir_fn, std::ofstream& component_log, int threads, bool save_graphs = false) {

  auto timer = high_resolution_timer();
  auto g = vdb_to_graph(component, args);
  component_log << "vdb to graph, " << timer.elapsed_formatted() << '\n';

  if (args->coarsen_steps.value()) {
    timer.restart();
    auto msg = Geometry::multiscale_graph(g, args->skeleton_grow, true);
    auto last_layer_index = msg.layers.size() - 1;
    auto layer_index = args->coarsen_steps.value() > last_layer_index ? last_layer_index : args->coarsen_steps.value();
    g = msg.layers[layer_index];
    component_log << "coarsen, " << timer.elapsed_formatted() << '\n';
  }

  if (args->saturate_edges) {
    timer.restart();
    Geometry::saturate_graph(g, args->saturate_edges.value());
    component_log << "saturate edges, " << timer.elapsed_formatted() << '\n';
  }

  /*
  if (save_graphs)
    graph_save(component_dir_fn / ("mesh.graph"), g);
  */

  timer.restart();
  // multi-scale is faster and scales linearly with input graph size at
  // the cost of difficulty in choosing a grow threshold
  auto separators =
    multiscale_local_separators(g, Geometry::SamplingType::Advanced,
        args->skeleton_grow, args->skeleton_grain,
        /*opt steps*/ args->optimize_steps, threads,
        false);
  auto [component_graph, _] = skeleton_from_node_set_vec(g, separators);
  component_log << "msls, " << timer.elapsed_formatted() << '\n';

  // prune all leaf vertices (valency 1) whose only neighbor has valency > 2
  // as these tend to be spurious branches
  Geometry::prune(component_graph);

  standardize_sampling_smoothing(component_graph, args->voxel_size, args->smooth_steps.value());

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

  // if the images are inherently anisotropic, you need to scale the nodes radii according to
  // the anisotropic radii, this leaves soma nodes alone
  if (args->anisotropic_scaling.has_value())
    scale_radii(component_graph, args->anisotropic_scaling.value(), soma_coords);

  {
    auto illegal_nodes = count_nodes_within_another(component_graph);
    component_log << "Original within nodes, " << illegal_nodes.size() << '\n';
    if (illegal_nodes.size())
      fix_node_within_another(component_graph, illegal_nodes);
    illegal_nodes = count_nodes_within_another(component_graph);
    //assertm(illegal_nodes.size() == 0, "fix node within another not functional");
    if (illegal_nodes.size() != 0)
      component_log << "Post-fix within nodes, " << illegal_nodes.size() << '\n';
  }

  fix_same_position(component_graph);
  same_position(component_graph);

  // multifurcations are only important for rules of SWC standard
  component_graph = fix_multifurcations(component_graph, soma_coords);
  
  auto invalids = get_invalid_radii(component_graph);
  if (invalids.size() > 0) {
    component_log << "Invalid radii, " << invalids.size() << '\n';
    fix_invalid_radii(component_graph, invalids);
    invalids = get_invalid_radii(component_graph);
    //assertm(invalids.size() == 0, "fix invalid radii not functional");
    if (invalids.size() != 0)
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
    float unscaled_radius, std::array<double, 3> voxel_size) {
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
  auto v = min_max(voxel_size).first;
  auto radius = unscaled_radius * v;
  // volsize
  apo_file << radius * radius * radius;
  // mass,,,, color_r,color_g,color_b
  apo_file << "0.,,,,0,0,255\n";
  apo_file.close();
}

// converts the trees ( single ) root location and radius to world-space um units
// and names the output file accordingly, Note that this uniquely identifies
// trees and also allows rerunning recut from known seed/soma locations later
std::string swc_name(Node &n, std::array<double, 3> voxel_size, bool bbox_adjust=false,
    CoordBBox bbox = {}) {
  std::ostringstream out;
  out <<  std::fixed << std::setprecision(SWC_PRECISION);
  out << '[';
  out << n.pos[0] * voxel_size[0] << ',';
  out << n.pos[1] * voxel_size[1] << ',';
  out << n.pos[2] * voxel_size[2] << ']';

  auto min_voxel_size = min_max(voxel_size).first;
  out << "-r=" << n.radius * min_voxel_size;
  if (bbox_adjust) {
    auto off = bbox.min();
    out << "-offset=" << '(' << off[0] * voxel_size[0] << ',' << off[1] * voxel_size[1] << ',' << off[2] * voxel_size[2] << ')';
  }
  out << "-µm";
  return out.str();
}

// n,type,x,y,z,radius,parent
// for more info see:
// http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
// https://github.com/HumanBrainProject/swcPlus/blob/master/SWCplus_specification.html
auto print_swc_line = [](NodeID id, NodeID parent_id, 
    std::array<double, 3> swc_coord, bool is_root,
    float unscaled_radius, 
    CoordBBox bbox, std::ofstream &out,
    std::array<double, 3> voxel_size,
    bool bbox_adjust = true, bool is_eswc = false,
    bool disable_swc_scaling = false) {
  std::ostringstream line;

  auto scale_coord = [&](std::array<double, 3> &coord) {
    for (int i = 0; i < 3; ++i)
      coord[i] *= voxel_size[i];
  };

  //if (!bbox_adjust || is_eswc) {
  if (!disable_swc_scaling) {
    scale_coord(swc_coord);
  }

  if (bbox_adjust) { // implies output window crops is set
    std::array<double, 3> window_start = {static_cast<double>(bbox.min().x()),
      static_cast<double>(bbox.min().y()),
      static_cast<double>(bbox.min().z())};
    if (!disable_swc_scaling)
      scale_coord(window_start);

    auto subtract = [](std::array<double, 3> &l,
        const std::array<double, 3> r) {
      for (int i = 0; i < 3; ++i)
        l[i] -= r[i];
    };

    // adjust the coordinates to the components bbox
    subtract(swc_coord, window_start);
  }

  // n
  line << std::to_string(id) << ' ';

  // type_id
  if (is_root) {
    line << "1 ";
  } else {
    line << "3 ";
  }
  line << std::fixed << std::setprecision(SWC_PRECISION);

  // coordinates
  line << swc_coord[0] << ' ' << swc_coord[1] << ' ' << swc_coord[2] << ' ';

  auto v = min_max(voxel_size).first;
  auto radius = unscaled_radius * v;
  line << radius << ' ';

  // parent
  if (is_root) {
    // only the first line of the file can have a parent of -1
    // any other should connect to themselves
    line << std::to_string(is_eswc ? 0 : -1);
  } else {
    line << std::to_string(parent_id);
  }

  if (is_eswc) {
    // " seg_id level mode timestamp TFresindex";
    line << " 0 0 0 0 1";
  }

  line << '\n';

  if (out.is_open()) {
    out << line.str();
  } else {
    std::cout << line.str();
  }
};

void write_swcs(const AMGraph3D &component_graph, std::vector<GridCoord> soma_coords,
    std::array<double, 3> voxel_size,
    std::filesystem::path component_dir_fn = ".",
    CoordBBox bbox = {}, bool bbox_adjust = false,
    bool is_eswc = false, bool disable_swc_scaling = false) {

  // each vertex in the graph has a single parent (id) which is
  // determined via BFS traversal
  std::vector<std::optional<NodeID>> parent_table(component_graph.no_nodes());

  // scan the graph to find the final set of soma_ids
  std::vector<NodeID> soma_ids;
  for (NodeID i : component_graph.node_ids()) {
    auto pos = component_graph.pos[i];
    auto current_coord = GridCoord(pos[0], pos[1], pos[2]);
    if (std::find(soma_coords.begin(),
          soma_coords.end(), current_coord) != std::end(soma_coords))
      soma_ids.push_back(i);
  }

  std::set<NodeID> visited_node_ids;
  // do BFS from each known soma in the component
  for (auto soma_id : soma_ids) {
      std::set<NodeID> visited_swc_ids;
      NodeID swc_id = 1; // are 1-indexed
      // 1-indexed, nothing at 0
      std::vector<std::optional<NodeID>> swc_ids(component_graph.no_nodes() + 1);

      // init q with soma
      std::queue<NodeID> q;
      q.push(soma_id);

      // start swc and add header metadata
      auto pos = component_graph.pos[soma_id];
      Node n{pos, get_radius(component_graph, soma_id)};
      auto file_name_base = swc_name(n, voxel_size, bbox_adjust, bbox);

      // traverse rest of tree
      parent_table[soma_id] = soma_id; // a soma technically has no parent
                                       // start file per soma, write header info
      std::ofstream swc_file;
      if (is_eswc) {
        auto soma_pos = component_graph.pos[soma_id].get();
        auto soma_coord = std::array<double, 3>{pos[0], pos[1], pos[2]};
        auto soma_radius = get_radius(component_graph, soma_id);

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
        auto radius = get_radius(component_graph, id);
        // can only be this trees root, not possible for somas from other trees to enter in to q
        auto is_root = id == soma_id;
        auto parent_id = parent_table[id].value();

        auto coord = std::array<double, 3>{pos[0], pos[1], pos[2]};

        assertm(visited_node_ids.count(id) == 0, "Node already visited");
        assertm(visited_swc_ids.count(swc_id) == 0, "SWC already assigned");
        visited_node_ids.insert(id);
        visited_swc_ids.insert(swc_id);
        // invalid if root
        auto swc_parent_id = is_root ? swc_id : swc_ids[parent_id].value();
        print_swc_line(swc_id, swc_parent_id, coord, is_root, radius, 
            bbox, swc_file, voxel_size, bbox_adjust, is_eswc, disable_swc_scaling);
        // mark this swc's id
        swc_ids[id] = swc_id;
        ++swc_id;

        // add all neighbors of current to q
        for (auto nb_id : component_graph.neighbors(id)) {
          // do not add other somas or previously visited to the q
          if (nb_id != id 
              && !parent_table[nb_id].has_value() 
              && std::find(soma_ids.begin(), soma_ids.end(), nb_id) == std::end(soma_ids)) {
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

// SWCs are by default in um world space, recut always operates in pixel space
// you can translate from world to pixel space via a voxel size.
// some inputs are unscaled see --disable_swc_scaling, in those cases
// you can leave the coordinates and radii as is
// the soma is guaranteed to be the first node of the returned graph at index 0
// throws if more than 1 soma or if not at first line
std::pair<Seed, AMGraph3D>
swc_to_graph(filesystem::path swc_file, std::array<double, 3> voxel_size,
    GridCoord image_offsets = zeros(), bool disable_swc_scaling=false, bool save_file = false) {
  ifstream ifs(swc_file);
  if (ifs.fail()) {
    throw std::runtime_error("Unable to open marker file " + swc_file.string());
  }

  if (disable_swc_scaling)  {
    // leave pos and radii as is (in pixel space)
    voxel_size = {{1,1,1}};
  }

  auto min_voxel_size = min_max(voxel_size).first;

  std::vector<Seed> seeds;
  AMGraph3D g;
  std::vector<std::pair<NodeID, NodeID>> edges;
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
    radius = radius_um / min_voxel_size;
    x = std::round(x_um / voxel_size[0]);
    y = std::round(y_um / voxel_size[1]);
    z = std::round(z_um / voxel_size[2]);
    auto p = Pos(x, y, z);

    // add it to the graph
    auto node_id = g.add_node(p);
    if ((node_id + 1) != id) {
      std::cout << "node id " << node_id << ' ' << " id " << id << '\n';
      throw std::runtime_error("SWC ids are improperly numbered");
    }
    set_radius(g, radius, node_id);

    // somas are nodes that have a parent of -1 (adjusted to -2) or have an index
    // of themselves
    if (parent_id == -2 || parent_id == node_id) {
      auto volume = static_cast<uint64_t>(std::round((4. / 3.) * PI * std::pow(radius, 3)));
      std::array<double, 3> coord_um{x_um, y_um, z_um};
      // the coord of the soma is used as an identity
      auto coord = GridCoord(x, y, z);
      seeds.emplace_back(coord, coord_um, radius, radius_um, volume);
      if (id != 1)
        throw std::runtime_error("Soma node must be first line of SWC");
    } else {
      //g.connect_nodes(node_id, parent_id);
      edges.emplace_back(node_id, parent_id);
    }
  }

  // connect all the nodes according to saved bidirectional edges
  rng::for_each(edges, [&](auto p) {
    g.connect_nodes(p.first, p.second);
  });

  if (seeds.size() != 1) {
    throw std::runtime_error("Warning: SWC files are trees which by definition must have only 1 root (soma), provided file: " + 
        swc_file.generic_string() + " has " + std::to_string(seeds.size()));
  }
  
  if (save_file)
    graph_save(swc_file.stem().string() + ".graph", g);

  return std::make_pair(seeds.front(), g);
}

//openvdb::FloatGrid::Ptr skeleton_to_surface(const AMGraph3D &skeleton) {
//}

int count_self_connected(const AMGraph3D &g) {
  std::vector<NodeID> self_connected;
  for (NodeID n : g.node_ids()) {
    for (NodeID nn: g.neighbors(n)) {
      if (n == nn)
        self_connected.push_back(n);
    }
  }
  return self_connected.size();
}

openvdb::FloatGrid::Ptr skeleton_to_surface_feq(const AMGraph3D &skeleton) {

  // feq requires an explicit radii vector
  std::vector<double> radii;
  radii.reserve(skeleton.no_nodes());
  for (NodeID i : skeleton.node_ids()) {
    radii.push_back(get_radius(skeleton, i));
  }

  // generate a mesh manifold then create a list of points and
  // polygons (quads or tris)
  //std::cout << "Start graph to manifold" << '\n';
  auto manifold = graph_to_FEQ(skeleton, radii);

  //HMesh::obj_save("mesh.obj", manifold);

  // translate manifold into something vdb will understand
  // points ands quads
  //std::cout << "Start manifold to polygons" << '\n';
  auto [points, quads] = mesh_to_polygons(manifold);

  // convert the mesh into a surface (level set) vdb
  //std::cout << "Start polygons to level set" << '\n';
  vto::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(points, quads);
  return vto::meshToVolume<openvdb::FloatGrid>(mesh, *get_transform());
}

void remove_from_graph(AMGraph3D &g, std::vector<GridCoord> coords) {
  // loop over all vertices until no remaining multifurcations are found
  for (auto i : g.node_ids()) {
    auto pos = g.pos[i];
    auto current_coord = GridCoord(pos[0], pos[1], pos[2]);
    if (rng::find(coords, current_coord) != rng::end(coords)) {
      //std::cout << "nb count " << g.neighbors(i).size() << '\n';
      g.remove_node(i);
      g.cleanup();
    }
  }
}

std::vector<int> node_valencies(const AMGraph3D &g) {
  std::vector<int> valencies;
  for (NodeID n : g.node_ids()) {
    auto nbs = g.neighbors(n);
    valencies.push_back(nbs.size());
  }

  // greatest to smallest
  std::sort(valencies.begin(), valencies.end(), std::greater<int>());
  for (auto v : valencies)
    std::cout << v << '\n';
  return valencies;
}

/*
std::vector<AMGraph3D> split_graph(const AMGraph3D& g, const NodeID n) {
    using NodeQueue = std::queue<NodeID>;

    auto is_n = [n](auto i) { return i == n; };

    NodeSet s_visited;
    std::vector<AMGraph3D> subgs;
    // map from the origin node ids to the new set
    std::vector<std::optional<NodeID>> to_new_parent;
    // its okay to make this anew for each subgraph since
    // they do not touch each other and will never connect
    // aka, we will never ask to translate a node from another
    // subgraph, optional would throw if you did
    to_new_parent.reserve(g.no_nodes());
    for (auto old_id : rv::iota(static_cast<NodeID>(0), g.no_nodes()) | rv::remove_if(is_n)) {
        if(s_visited.count(old_id)==0)
        {
            NodeQueue Q;
            Q.push(old_id);
            s_visited.insert(old_id);
            AMGraph3D subg;
            while(!Q.empty())
            {
                NodeID current_old_id = Q.front();
                Q.pop();

                auto new_id = subg.add_node(g.pos[old_id]);
                set_radius(subg, get_radius(g, old_id), new_id);
                // connect back to the new parent that added this to the queue
                auto new_parent_id = to_new_parent[current_old_id];
                if (new_parent_id.has_value()) subg.connect_nodes(new_id, new_parent_id.value());

                for(auto old_nb: g.neighbors(current_old_id))
                    if(!is_n(old_nb) && s_visited.count(old_nb)==0) {
                        Q.push(old_nb);
                        to_new_parent[old_nb] = new_id;
                        s_visited.insert(old_nb);
                    }
            }
            subgs.push_back(subg);
        }
    }
    return subgs;
}
*/

// partition a graph into separate connected components at node n
// n is not present in the returned subgraphs
std::vector<AMGraph3D> split_graph(const AMGraph3D &g, const NodeID n) {

  // definitions
  auto is_n = [n](auto i) { return i == n; };

  // add the old connectivity to the new subgraph
  // making sure to use the new set of node ids
  auto copy_connections = [&is_n](const AMGraph3D &g, AMGraph3D &subg, 
      auto to_new, NodeID new_id, NodeID old_id) {

    std::vector<NodeID> old_nbs = g.neighbors(old_id);
    auto new_nbs = old_nbs | rv::remove_if(is_n) 
                        | rv::transform([&to_new](auto old_id) { return to_new[old_id].value(); });
    for (auto new_nb : new_nbs)
      subg.connect_nodes(new_id, new_nb); // bi-direc, order is irrel.
  };

  // a set is a list of NodeIDs indexing into the old graph
  // this creates a new graph copying over the old position and 
  // radius from the original graph
  // also copies over connectivity which requires translating the 
  // node ids of the old edges into their equivalents in the new graph
  auto set_to_graph = [&](NodeSet nset) -> AMGraph3D {

    AMGraph3D subg;
    // map from the origin node ids to the new set
    // its okay to make this anew for each subgraph since
    // they do not touch each other and will never connect
    // aka, we will never ask to translate a node from another
    // subgraph, optional would throw if you did
    std::vector<std::optional<NodeID>> to_new(g.no_nodes());
    assertm(to_new.size(), "to_new must match exact size");

    // create all nodes and save an index mapping ahead of time
    for (auto const old_id : nset) {
      auto new_id = subg.add_node(g.pos[old_id]);
      set_radius(subg, get_radius(g, old_id), new_id);
      to_new[old_id] = new_id;
    }

    // safe to establish all connections among new ids of subg
    // the values of nset are old ids
    // nset maintains order and new_ids are linearly ordered [0..len(subg))
    for (auto const & [new_id, old_id] : nset | rv::enumerate)
        copy_connections(g, subg, to_new, new_id, old_id);

    return subg;
  };

  // build a set of all nodes other than n
  NodeSet s;
  for (auto id : rv::iota(static_cast<NodeID>(0), g.no_nodes()) | rv::remove_if(is_n))
    s.insert(id);
  assertm(s.size() == (g.no_nodes() -1), "Set must be 1 smaller than graph");

  // n is not present in any of the returned sets
  auto sets = Geometry::connected_components(g, s);

  if (false) { // verify subgraphs used for debugging
    auto sz = sets.size();
    auto nbs = g.neighbors(n);
    auto nbs_sz = nbs.size();
    if (sz > nbs_sz)  {
      std::cout << "Subgraph size: " << sz << '\n';
      std::cout << "nbs size: " << nbs_sz << '\n';
      throw std::runtime_error("Subgraphs must be <= original nbs");
    }
  }

  return sets | rv::transform(set_to_graph) | rng::to_vector;
}

template <typename T>
void stamp_graph(T mask_accessor, AMGraph3D& g) {
  for (auto i : g.node_ids()) {
    for (const auto coord : sphere_iterator(to_coord(g.pos[i]), get_radius(g, i))) {
      mask_accessor.setValueOn(coord);
    }
  }
}

openvdb::MaskGrid::Ptr graph_neurites_to_mask(AMGraph3D& skeleton, Seed seed) {
  // removes the soma which is always at index 0
  auto subgraphs = split_graph(skeleton, 0);

  auto mask = openvdb::MaskGrid::create();
  auto mask_accessor = mask->getAccessor();
  for (auto& subgraph : subgraphs)  {
    // each skeletal node is upsampled to be 1 voxel apart, to create a smooth contiguous surface
    resample(subgraph, 1);
    stamp_graph(mask_accessor, subgraph);
  }

  return mask;
}

openvdb::MaskGrid::Ptr graph_to_mask(AMGraph3D& skeleton) {
  auto mask = openvdb::MaskGrid::create();
  auto mask_accessor = mask->getAccessor();

  // each skeletal node is upsampled to be 1 voxel apart, to create a smooth contiguous surface
  resample(skeleton, 1);
  stamp_graph(mask_accessor, skeleton);

  return mask;
}

// returns a mask grid corresponding to the segmented volume of the swc
openvdb::MaskGrid::Ptr swc_to_mask(filesystem::path swc_file,
    std::array<double, 3> voxel_size, GridCoord image_offsets, 
    bool save_vdbs = false, std::string name = "", bool disable_swc_scaling=false,
    bool neurites_only=false, bool save_swcs = false) {

  bool merge_soma_on_top = false;
  auto [seed, skeleton] = swc_to_graph(swc_file, voxel_size, image_offsets, disable_swc_scaling);

  auto mask = neurites_only ? graph_neurites_to_mask(skeleton, seed) : graph_to_mask(skeleton);

  if (save_vdbs)
    write_vdb_file({mask}, "mask-" + (name.empty() ? swc_file.stem().string() : name) + ".vdb");

  // delete this
  //openvdb::FloatGrid::Ptr level_set = vto::topologyToLevelSet(mask, RECUT_LEVEL_SET_HALF_WIDTH, 0, 1, 0, 0);
  //if (save_vdbs)
    //write_vdb_file({level_set}, "surface-" + (name.empty() ? swc_file.stem().string() : name) + ".vdb");

  return mask;
}

// returns a level set float grid
openvdb::FloatGrid::Ptr swc_to_surface(filesystem::path swc_file,
    std::array<double, 3> voxel_size, GridCoord image_offsets, 
    bool save_vdbs = false, std::string name = "", bool disable_swc_scaling=false,
    bool neurites_only=false, bool save_swcs = false) {

  bool merge_soma_on_top = false;
  auto [seed, skeleton] = swc_to_graph(swc_file, voxel_size, image_offsets, disable_swc_scaling);
  std::vector<Seed> seeds{seed};

  auto invalids = get_invalid_radii(skeleton);
  if (invalids.size()) {
    std::cout << "Original invalid radii, " << invalids.size() << '\n';
    fix_invalid_radii(skeleton, invalids);
  }

  // for any nodes within the soma collapse them
  // re-establish a known seed id
  {
    auto nodes = seeds | rv::transform(to_node) 
      | rng::to_vector;
    merge_local_radius(skeleton, nodes, 
        /*don't dilate soma at all*/1);
  }

  // you must correct for nodes within another before calling
  // graph to FEQ
  {
    auto illegal_nodes = count_nodes_within_another(skeleton);
    std::cout << "Original within nodes, " << illegal_nodes.size() << " of " << skeleton.no_nodes() << '\n';
    if (!illegal_nodes.empty())
      fix_node_within_another(skeleton, illegal_nodes);
  }

  NodeID seed_id;
  //auto seed_id = get_closest_id(skeleton, to_node(seed));
  {
    auto tree = build_kdtree(skeleton);
    Pos found_pos;
    Pos seed_pos(seed.coord[0], seed.coord[1], seed.coord[2]);
    double d = 0.00001f;
    assertm(tree.closest_point(seed_pos, d, found_pos, 
          seed_id), "Could not find seed");
  }

  if (save_vdbs)
    graph_save(swc_file.stem().string() + ".graph", skeleton);

  if (save_swcs) {
    auto soma_coords = rv::transform(seeds, [](Seed seed) {
        return seed.coord;
        }) | rng::to_vector;
    write_swcs(skeleton, soma_coords, voxel_size);
  }

  // check SWC health
  //node_valencies(skeleton);
  {
    auto invalids = get_invalid_radii(skeleton);
    if (invalids.size())
      throw std::runtime_error("invalid radii found");
  }
  if (count_self_connected(skeleton))
    throw std::runtime_error("Self connected nodes found");

  // delete the soma from the graph because it can have >10 valency
  // which causes graph_to_FEQ to seg fault
  // the soma sphere could be fused on top of the level set later if desired
  auto subgraphs = split_graph(skeleton, seed_id);
  if ((skeleton.no_nodes() - 1) != rng::accumulate(subgraphs | rv::transform([](auto s) { return s.no_nodes(); }), 0))
    throw std::runtime_error("Split graph loses nodes");

  //for (auto [i, subg] : subgraphs | rv::enumerate)
    //graph_save("subg" + std::to_string(i) + ".graph", subg);

  openvdb::FloatGrid::Ptr level_set;
  {
    auto timer = high_resolution_timer();
    auto level_sets = subgraphs | rv::transform(skeleton_to_surface_feq) | rng::to_vector;
    std::cout << "Graph -> surface elapsed: " << timer.elapsed() << "s\n";

    // empties/nullifies level sets into accumulator level set
    timer.restart();
    level_set = level_sets.front();
    for (int i=1; i < subgraphs.size(); ++i)
      vto::csgUnion(*level_set, *level_sets[i]);

    if (!neurites_only) {
      auto soma_sdf = vto::createLevelSetSphere<openvdb::FloatGrid>(
         seed.radius, seed.coord.asVec3s(), 1.,
         RECUT_LEVEL_SET_HALF_WIDTH);
       //empties/nullifies the soma_sdf grid
       //merge the two on top of each other
      vto::csgUnion(*level_set, *soma_sdf);
    }

    // if level sets overlap at all, their merged values may no longer form a proper surface 
    // without resurfacing
    level_set = vto::levelSetRebuild(*level_set);
    std::cout << "Surface merge elapsed: " << timer.elapsed() << "s\n";
  }

  if (save_vdbs)
    write_vdb_file({level_set}, "surface-" + (name.empty() ? swc_file.stem().string() : name) + ".vdb");

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
  auto true_positive = truth->deepCopy();
  {
    auto test_copy = test->deepCopy();
    vto::csgIntersection(*true_positive, *test_copy);
  }
  if (save_vdbs)
    write_vdb_file({true_positive}, "surface-true-positive.vdb");
  auto true_positive_count = voxel_count(true_positive);

  // calculate the count of false positive pixels
  auto false_positive = test->deepCopy();
  {
    auto true_positive_copy = true_positive->deepCopy();
    vto::csgDifference(*false_positive, *true_positive_copy);
  }
  if (save_vdbs)
    write_vdb_file({false_positive}, "surface-false-positive.vdb");
  auto false_positive_count = voxel_count(false_positive);

  // calculate the count of false positive pixels
  auto false_negative = truth->deepCopy();
  {
    auto true_positive_copy = true_positive->deepCopy();
    vto::csgDifference(*false_negative, *true_positive_copy);
  }
  if (save_vdbs)
    write_vdb_file({false_negative}, "surface-false-negative.vdb");
  auto false_negative_count = voxel_count(false_negative);

  // calculation union
  auto union_ls = truth->deepCopy();
  {
    auto test_copy = test->deepCopy();
    vto::csgUnion(*union_ls, *test_copy);
  }
  auto union_count = voxel_count(union_ls);

  recall = true_positive_count / static_cast<double>(true_positive_count + false_negative_count);
  precision_d = true_positive_count / static_cast<double>(true_positive_count + false_positive_count);
  f1 = 2 * precision_d * recall / static_cast<double>(precision_d + recall);
  iou = true_positive_count / static_cast<double>(union_count);

  std::cout << "true positive count, " << true_positive_count << '\n';
  std::cout << "false positive count, " << false_positive_count << '\n';
  std::cout << "false negative count, " << false_negative_count << '\n';
  std::cout << "recall, " << recall << '\n';
  std::cout << "precision, " << precision_d << '\n';
  std::cout << "F1, " << f1 << '\n';
  std::cout << "IoU, " << iou << '\n';
}

// surface must be a level set
// title is the name of the statistic your computing for example 
// skeletal recall or precision
double calculate_skeleton_within_mask(AMGraph3D &g, const openvdb::MaskGrid::Ptr mask, std::string title) {

  auto accessor = mask->getConstAccessor();

  auto is_inside = [&](NodeID i) {
    return accessor.isValueOn(to_coord(g.pos[i]));
  };

  auto within_frac = static_cast<double>(rng::distance(
        rv::iota(static_cast<NodeID>(0), g.no_nodes())
        | rv::filter(is_inside))) / g.no_nodes();

  std::cout << title << ": ";
  std::cout << within_frac << '\n';

  return within_frac;
}

// surface must be a level set
// title is the name of the statistic your computing for example 
// skeletal recall or precision
double calculate_skeleton_within_surface(AMGraph3D &g, openvdb::FloatGrid::Ptr surface, std::string title) {
  auto is_outside_surface = [&g, surface](NodeID i) {
    return surface->tree().getValue(to_coord(g.pos[i])) > 0;
  };

  auto within_frac = static_cast<double>(rng::distance(
        rv::iota(static_cast<NodeID>(0), g.no_nodes())
        | rv::remove_if(is_outside_surface))) / g.no_nodes();

  std::cout << title << ": ";
  std::cout << within_frac << '\n';

  return within_frac;
}

// soma is guaranteed to be the first in the graph
void scale_neurites(AMGraph3D& g) {
  // skip the soma
  for (auto i=1; i < g.no_nodes(); ++i)
    set_radius(g, get_radius(g, i) * ANISOTROPIC_FACTOR, i);
}

