struct ArrayHasher {
  std::size_t operator()(const std::array<double, 3> &a) const {
    std::size_t h = 0;

    for (auto e : a) {
      h ^= std::hash<int>{}(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

auto get_id_map = []() {
  std::unordered_map<std::array<double, 3>, uint32_t, ArrayHasher>
      coord_to_swc_id;
  // add a dummy value that will never be on to the map so that real indices
  // start at 1
  coord_to_swc_id[{std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()}] = 0;
  return coord_to_swc_id;
};

// also return a mapping from coord to id
auto create_tree_indices = [](std::vector<MyMarker *> &tree) {
  // start a new blank map for coord to a unique swc id
  auto coord_to_swc_id = get_id_map();

  // iter those marker*
  auto indices =
      tree | rv::transform([&coord_to_swc_id](const auto marker) {
        auto coord = std::array<double, 3>{marker->x, marker->y, marker->z};
        // roots have parents of themselves
        auto parent_coord =
            marker->type
                ? std::array<double, 3>{marker->parent->x, marker->parent->y,
                                        marker->parent->z}
                : coord;

        // also accumulate the mapping
        find_or_assign(parent_coord, coord_to_swc_id);
        auto id = find_or_assign(coord, coord_to_swc_id);

        // build the indices
        return id;
      }) |
      rng::to_vector;

  return std::make_pair(indices, coord_to_swc_id);
};

// trees only, can not pass cluster
auto parent_listed_above = [](std::vector<MyMarker *> &tree) {
  // start a new blank map for coord to a unique swc id
  auto coord_to_swc_id = get_id_map();

  // iter those marker*
  for (const auto marker : tree) {
    auto coord = std::array<double, 3>{marker->x, marker->y, marker->z};
    auto parent_coord = marker->type ? std::array<double, 3>{marker->parent->x,
                                                             marker->parent->y,
                                                             marker->parent->z}
                                     : coord;

    // add this current id to the mapping regardless
    find_or_assign(coord, coord_to_swc_id);

    // roots have parents of themselves so they will always be added
    // in above
    auto val = coord_to_swc_id.find(parent_coord);
    if (val == coord_to_swc_id.end()) {
      // the parent of this node was not already visited
      // which violates the SWC standard
      return false;
    }
  }

  // found no nodes with a parent not already listed
  return true;
};

auto tree_is_sorted = [](std::vector<MyMarker *> &tree) {
  auto [indices, _] = create_tree_indices(tree);
  return std::is_sorted(indices.begin(), indices.end());
};

// also return a mapping from coord to id
auto sort_tree_in_place = [](std::vector<MyMarker *> &tree) {
  auto [indices, coord_to_swc_id] = create_tree_indices(tree);

  rng::sort(rv::zip(indices, tree));
  return coord_to_swc_id;
};

template <typename ArrayT>
std::unordered_map<std::array<ArrayT, 3>, VID_t, ArrayHasher>
create_coord_to_idx(const std::vector<MyMarker *> &cluster) {
  // build to coord_to_idx
  std::unordered_map<std::array<ArrayT, 3>, VID_t, ArrayHasher> coord_to_idx;
  rng::for_each(cluster | rv::enumerate, [&](auto imarker) {
    auto [id, marker] = imarker;
    coord_to_idx[{marker->x, marker->y, marker->z}] = id;
  });
  return coord_to_idx;
}

// markers have a ptr to parent, all aprents must point to marker *within* its
// current vector if it points to marker within a previous vector, undefined
// behavior
auto adjust_parent_ptrs = [](std::vector<MyMarker *> &cluster) {
  auto coord_to_idx = create_coord_to_idx<double>(cluster);

  // adjust in place
  rng::for_each(cluster, [&](auto marker) {
    if (marker->type) { // skips roots
      const auto parent_coord = std::array<double, 3>{
          marker->parent->x, marker->parent->y, marker->parent->z};
      const auto parent_pair = coord_to_idx.find(parent_coord);
      if (parent_pair == coord_to_idx.end()) {
        std::cout << "Invalid " << *marker << '\n';
        throw std::runtime_error("adjust_parent_ptrs() can not operate on "
                                 "clusters with out-of-cluster parents");
      } else {
        marker->parent = cluster[parent_pair->second];
      }
    }
  });
};

auto is_cluster_self_contained = [](const std::vector<MyMarker *> &cluster) {
  auto coord_to_idx = create_coord_to_idx<double>(cluster);

  auto has_out_cluster_parent =
      cluster | rv::filter([](auto marker) {
        return marker->type; // skip roots
      }) |
      rv::filter([](auto marker) {
        if (!marker->parent) // roots skipped above
          throw std::runtime_error("Only roots can have invalid parents");
        return marker->parent->type; // ignore if parent is root
      }) |
      rv::filter([&coord_to_idx](auto marker) {
        const auto parent_coord = std::array<double, 3>{
            marker->parent->x, marker->parent->y, marker->parent->z};
        const auto parent_pair = coord_to_idx.find(parent_coord);
        return parent_pair == coord_to_idx.end();
      });

  return rng::distance(has_out_cluster_parent) == 0;
};

// build a mapping from
// index into cluster vector -> list of children indices
// index is the index into the cluster vector
// leaves will have no children, roots will not have a parent
auto create_child_list = [](const std::vector<MyMarker *> &cluster) {
  auto coord_to_idx = create_coord_to_idx<double>(cluster);

  // build child_list
  auto child_list = std::unordered_map<VID_t, std::vector<VID_t>>();
  rng::for_each(cluster | rv::enumerate | rv::filter([](auto imarker) {
                  auto [i, marker] = imarker;
                  return marker->type; // skip roots
                }),
                [&](auto imarker) {
                  auto [id, marker] = imarker;
                  // get parent
                  if (!marker->parent)
                    throw std::runtime_error("Parent missing");
                  const auto parent_coord = std::array<double, 3>{
                      marker->parent->x, marker->parent->y, marker->parent->z};
                  const auto parent_pair = coord_to_idx.find(parent_coord);
                  if (parent_pair == coord_to_idx.end())
                    throw std::runtime_error("Parent coord not found");
                  auto parent_id = parent_pair->second;

                  // update children
                  auto children = child_list.find(parent_id);
                  // update value of map
                  if (children == child_list.end()) { // not found yet
                    std::vector<VID_t> new_list = {id};
                    child_list[parent_id] = new_list;
                  } else {
                    child_list[parent_id].push_back(id);
                  }
                });

  return child_list;
};

// switch this to a while loop if ever encountering a stack overflow
// do a depth first search to create an ordered tree from a cluster
// leaves will have no children, roots do not have a parent
std::vector<MyMarker *> recurse_tree(
    VID_t id, std::unordered_map<VID_t, std::vector<VID_t>> &child_list,
    std::vector<MyMarker *> &tree, const std::vector<MyMarker *> &cluster) {
  auto marker = cluster[id];
  tree.push_back(marker);
  auto children = child_list.find(id);
  if (children != child_list.end()) {
    for (VID_t child_id : children->second) {
      recurse_tree(child_id, child_list, tree, cluster);
    }
  }
  return tree;
}

// assumes all markers point to parents within the cluster
auto partition_cluster = [](const std::vector<MyMarker *> &cluster) {
  auto child_list = create_child_list(cluster);

  auto root_ids =
      cluster | rv::enumerate |
      rv::remove_if([](auto imarker) { return imarker.second->type; }) |
      rv::transform([](auto imarker) { return imarker.first; });

  std::vector<std::vector<MyMarker *>> trees =
      root_ids | rv::transform([&](VID_t id) {
        auto tree = std::vector<MyMarker *>();
        return recurse_tree(id, child_list, tree, cluster);
      }) |
      rng::to_vector;
  return trees;
};

// assumes tree passed is sorted root at front of tree
auto write_swc = [](std::vector<MyMarker *> &tree,
                    std::array<float, 3> voxel_size,
                    std::filesystem::path component_dir_fn = ".",
                    CoordBBox bbox = {}, bool bbox_adjust = false,
                    bool is_eswc = false) {
  auto root = tree.front();
  if (root->type)
    throw std::runtime_error("First marker of tree must be a root (type 0)");

  // start swc and add header metadata
  auto file_name_base = "tree-with-soma-xyz-" +
                        std::to_string(static_cast<int>(root->x)) + '-' +
                        std::to_string(static_cast<int>(root->y)) + '-' +
                        std::to_string(static_cast<int>(root->z));
  auto coord_to_swc_id = get_id_map();
  std::ofstream swc_file;
  swc_file.open(component_dir_fn / (file_name_base + ".swc"));
  swc_file << "# Crop windows bounding volume: " << bbox << '\n'
           << "# id type_id x y z radius parent_id\n";
  // iter those marker*
  rng::for_each(tree, [&](const auto marker) {
    auto coord = std::array<double, 3>{marker->x, marker->y, marker->z};
    auto is_root = marker->type == 0;
    auto parent_coord =
        is_root ? coord
                : std::array<double, 3>{marker->parent->x, marker->parent->y,
                                        marker->parent->z};
    // expects an offset to a parent
    print_swc_line(coord, is_root, marker->radius, parent_coord, bbox, swc_file,
                   coord_to_swc_id, voxel_size, bbox_adjust, false);
  });

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
};

auto create_child_count = [](std::vector<MyMarker *> &tree) {
  // build a map to save coords with 1 or 2 children
  // coords not in this map are therefore leafs
  auto child_count =
      std::unordered_map<std::array<double, 3>, uint8_t, ArrayHasher>();
  rng::for_each(tree, [&child_count](auto marker) {
    if (marker->parent) {
      assertm(marker->parent, "Parent missing");
      const std::array<double, 3> parent_coord = {
          marker->parent->x, marker->parent->y, marker->parent->z};
      const auto val_count = child_count.find(parent_coord);
      if (val_count == child_count.end()) // not found
        child_count.insert_or_assign(parent_coord, 1);
      else
        child_count.insert_or_assign(parent_coord, val_count->second + 1);
    } // ignore those without parents
  });
  return child_count;
};

VID_t count_leaves(std::vector<MyMarker *> &tree) {
  auto child_count = create_child_count(tree);

  auto is_a_leaf = [&child_count](auto const marker) {
    return child_count.find(std::array{marker->x, marker->y, marker->z}) ==
           child_count.end();
  };

  return rng::count_if(tree, is_a_leaf);
}

// returns a new set of valid markers
std::vector<MyMarker *>
prune_short_branches(std::vector<MyMarker *> &tree,
                     int min_branch_length = MIN_BRANCH_LENGTH) {
  auto child_count = create_child_count(tree);

  auto is_a_branch = [&child_count](auto const marker) {
    const auto val_count = child_count.find({marker->x, marker->y, marker->z});
    return val_count != child_count.end() ? (val_count->second > 1) : false;
  };

  auto is_a_leaf = [&child_count](auto const marker) {
    return child_count.find(std::array{marker->x, marker->y, marker->z}) ==
           child_count.end();
  };

  // filter leafs with a parent that is a branch
  // otherwise persistence homology in TMD (Kanari et al.) has difficult to
  // diagnose bug
  auto filtered_tree =
      tree |
      rv::remove_if(
          [&min_branch_length, &is_a_leaf, &is_a_branch](auto marker) {
            // only check non roots
            if (marker->parent) {
              // only check from leafs, since they define the beginning of a
              // branch
              if (!is_a_leaf(marker))
                return false;

              // prune if immediate parent is branch
              // marker passed must be valid
              if (is_a_branch(marker->parent))
                return true;

              // filter short branches below min_branch_length
              if (min_branch_length > 0) {
                auto accum_euc_dist = 0.;
                do {
                  accum_euc_dist += marker_dist(marker, marker->parent);
                  // recurse upwards until finding branch or root
                  marker = marker->parent; // known to exist already
                  // stop recursing when you find a soma or branch
                  // since that defines the end of the branch
                } while (marker->parent &&
                         !is_a_branch(marker)); // not a root and not a branch

                return accum_euc_dist < min_branch_length; // remove if true

              } else {
                return false; // keep
              }
            }
            return false; // keep somas -- soma parent can be undefined
          }) |
      rng::to_vector;

  const auto pruned_count = tree.size() - filtered_tree.size();

  // must be called repeatedly until convergence
  if (pruned_count)
    return prune_short_branches(filtered_tree, min_branch_length);
  else
    return filtered_tree;
}

auto is_illegal_branch = [illegal_count = 3](auto const marker,
                                             auto &child_count) {
  const auto val_count = child_count.find({marker->x, marker->y, marker->z});
  return val_count != child_count.end() ? (val_count->second >= illegal_count)
                                        : false;
};

// returns a new set of valid markers
std::vector<MyMarker *> fix_trifurcations(std::vector<MyMarker *> &tree) {
  auto child_count = create_child_count(tree);

  auto new_nodes =
      tree | rv::filter([](auto marker) {
        return marker->type; // ignore roots (type of 0)
      }) |
      rv::filter([](auto marker) {
        assertm(marker->parent, "Only roots can have invalid parents");
        return marker->parent->type; // ignore if parent is root
      }) |
      rv::filter([&child_count](auto marker) {
        return is_illegal_branch(marker->parent, child_count);
      }) |
      rv::transform([&child_count](auto marker) {
        const std::array<double, 3> old_parent_coord = {
            marker->parent->x, marker->parent->y, marker->parent->z};

        // create a new averaged node between the parent and the parent->parent
        // already protected against parent of parent all non somas have a
        // parent
        assertm(marker->parent->parent, "Only roots can have invalid parents");
        const std::array<double, 3> parent_parent_coord = {
            marker->parent->parent->x, marker->parent->parent->y,
            marker->parent->parent->z};
        const std::array<double, 3> sum = {
            parent_parent_coord[0] + old_parent_coord[0],
            parent_parent_coord[1] + old_parent_coord[1],
            parent_parent_coord[2] + old_parent_coord[2]};
        const std::array<double, 3> new_parent_coord = {sum[0] / 2, sum[1] / 2,
                                                        sum[2] / 2};
        auto new_parent = new MyMarker(
            new_parent_coord[0], new_parent_coord[1], new_parent_coord[2],
            (marker->parent->parent->radius + marker->parent->radius) / 2);

        // connect new node to tree, +1 child for pp
        auto old_parent = marker->parent;
        new_parent->parent = old_parent->parent;
        // switch current parent
        marker->parent = new_parent;
        // switch old parents parent, updating map -1 child for pp
        old_parent->parent = new_parent;
        child_count.insert_or_assign(new_parent_coord, 2);

        const auto val_count = child_count.find(old_parent_coord);
        // decrement old parent
        child_count.insert_or_assign(old_parent_coord, val_count->second - 1);

        return new_parent;
      }) |
      rng::to_vector;

  rng::copy(new_nodes, rng::back_inserter(tree));
  return tree;
}

auto is_furcation = [](auto const marker, auto &child_count) {
  const auto val_count = child_count.find({marker->x, marker->y, marker->z});
  return val_count != child_count.end() ? (val_count->second > 1) : false;
};

VID_t count_furcations(std::vector<MyMarker *> &tree) {
  auto child_count = create_child_count(tree);

  auto furcations = tree | rv::filter([](auto marker) {
                      return marker->type; // ignore roots (type of 0)
                    }) |
                    rv::filter([&child_count](auto marker) {
                      return is_furcation(marker, child_count);
                    });

  return rng::distance(furcations);
}

// returns vector of trifurcation points
auto tree_is_valid = [](auto tree) {
  auto child_count = create_child_count(tree);
  auto mismatchs =
      tree | rv::filter([](auto marker) {
        return marker->type; // ignore roots (type of 0)
      }) |
      rv::filter([](auto marker) {
        assertm(marker->parent, "Only roots can have invalid parents");
        return marker->parent->type; // ignore if parent is root
      }) |
      rv::filter([&child_count](auto marker) {
        return is_illegal_branch(marker, child_count);
      }) |
      rng::to_vector;

  return mismatchs;
};

// creates an iterator in zyx order for probing VDB grids for the interior of a
// sphere
auto sphere_iterator = [](const GridCoord &center, const int radius) {
  // passing center by ref & through the lambda captures causes UB
  int cx = center.x();
  int cy = center.y();
  int cz = center.z();
  return rv::for_each(rv::iota(cx - radius, 1 + cx + radius), [=](int x) {
    return rv::for_each(rv::iota(cy - radius, 1 + cy + radius), [=](int y) {
      return rv::for_each(rv::iota(cz - radius, 1 + cz + radius), [=](int z) {
        auto const new_coord = GridCoord(x, y, z);
        return rng::yield_if(coord_dist(new_coord, center) <= radius,
                             new_coord);
      });
    });
  });
};

// From Advantra pnr implementation: mean-shift (non-blurring) uses
// neighbourhood of pixels determined by the current nodes radius
std::vector<MyMarker *>
mean_shift(std::vector<MyMarker *> nX, int max_iterations,
           uint16_t prune_radius_factor,
           std::unordered_map<GridCoord, VID_t>
               coord_to_idx) {

  int checkpoint = round(nX.size() / 10.0);

  double conv[4], next[4]; // x y z radius

  std::vector<MyMarker *> nY;
  nY.reserve(nX.size());
  double distance_delta_criterion = .5;

  double x2, y2, z2, r2;
  // go through nY[i], initiate with nX[i] values and refine by mean-shift
  // averaging
  for (long i = 0; i < nX.size(); ++i) {
    // adjust_marker = new MyMarker
    // type, x, y, z, radius, nbr,
    //  create a new copy
    nY.emplace_back(nX[i]);

    if (nY[i]->type == 0)
      continue; // do not refine soma nodes

    // refine nX[i] node location and scale and store the result in nY[i]
    conv[0] = nX[i]->x;
    conv[1] = nX[i]->y;
    conv[2] = nX[i]->z;
    conv[3] = nX[i]->radius;

    double last_distance_delta = 1; // default value
    // ... stop when the update in distance gets half the size of a voxel
    // all coordinates are rounded to the nearest pixel
    // pixels are small and negligible in high resolution anyway
    // so this is still conservative
    for (int iter = 0; iter < max_iterations &&
                       last_distance_delta > distance_delta_criterion;
         ++iter) {
      int cnt = 0;

      next[0] = 0; // local mean is the follow-up location
      next[1] = 0;
      next[2] = 0;
      next[3] = 0;

      auto center = GridCoord(std::round(conv[0]), std::round(conv[1]),
                              std::round(conv[2]));
      auto radius_for_pruning = prune_radius_factor * conv[3];

      for (const auto coord : sphere_iterator(center, radius_for_pruning)) {
        auto ipair = coord_to_idx.find(coord);
        if (ipair == coord_to_idx.end())
          continue; // skip not found
        auto nbr_idx = ipair->second;
        // if not same node
        if (nbr_idx != i) {
          // assumes that all x,y,z are positive
          next[0] += nX[nbr_idx]->x;
          next[1] += nX[nbr_idx]->y;
          next[2] += nX[nbr_idx]->z;
          next[3] += nX[nbr_idx]->radius;
          ++cnt;
        }
      }

      next[0] /= cnt; // cnt > 0, at least node location itself will be in the
                      // kernel neighbourhood
      next[1] /= cnt;
      next[2] /= cnt;
      next[3] /= cnt;

      last_distance_delta = pow(next[0] - conv[0], 2) +
                            pow(next[1] - conv[1], 2) +
                            pow(next[2] - conv[2], 2);

      conv[0] = next[0]; // for the next iteration
      conv[1] = next[1];
      conv[2] = next[2];
      conv[3] = next[3];
    }

    // force
    nY[i]->x = std::round(conv[0]);
    nY[i]->y = std::round(conv[1]);
    nY[i]->z = std::round(conv[2]);
    nY[i]->radius = conv[3];
  }
  assertm(nY.size() == nX.size(), "nX and nY size must match");

  // now that all nY's have been created, go through all non-somas and
  // reassign parent ptrs to correct new address in nY (instead of nX) which
  // would be undefined when nX goes out of scope
  rng::for_each(nY | rv::filter([](auto marker) { return marker->type; }),
                [&nY](auto nYi) {
                  assertm(nYi->nbr.size() == 1,
                          "marker can only have 1 nbr in advantra prune");
                  auto idx = nYi->nbr[0];
                  assertm(idx < nY.size(), "idx not in bounds of nY");
                  nYi->parent = nY[idx];
                });

  return nY;
}

// sphere grouping compaction/pruning strategy inspired by Advantra's code
// switched from n^2 to nr^3 where r is the radii of a given node
std::vector<MyMarker *>
advantra_prune(vector<MyMarker *> nX, uint16_t prune_radius_factor,
               std::unordered_map<std::array<double, 3>, VID_t, ArrayHasher>
                   coord_to_idx) {

  std::vector<MyMarker *> nY;
  auto no_neighbor_count = 0;

  vector<long> indices(nX.size());
  for (VID_t i = 0; i < indices.size(); ++i)
    indices[i] = i;
  // TODO sort by float value if possible
  // sort(indices.begin(), indices.end(), CompareIndicesByNodeCorrVal(&nX));

  // translate a dense linear idx of X to the sparse linear idx of y
  vector<long> X2Y(nX.size(), -1);
  X2Y[0] = 0; // first one is with max. correlation

  nY.push_back(nX[0]);

  auto check_node = [&](const long ci) {
    X2Y[ci] = nY.size();
    // create a new marker in the sparse set starting from an existing one
    // that has not been pruned yet
    auto nYi = new MyMarker(*(nX[ci]));
    float grp_size = 1;

    auto radius_for_pruning = static_cast<float>(nYi->radius);
    if (nYi->type != 0) { // not soma
      // you can decrease the sampling density along any branch
      // by increasing the prune radius here
      radius_for_pruning *= prune_radius_factor;
    }

    // rounds the current iteratively averaged location of nYi
    auto center =
        GridCoord(std::round(nYi->x), std::round(nYi->y), std::round(nYi->z));
    // Warning: this loop mutates the position of nYi
    // neighbors are probed by their integer coordinate
    // all markers passed to this function are integer coordinates
    // all markers passed
    for (const auto coord : sphere_iterator(center, radius_for_pruning)) {
      std::array<double, 3> coord_key = {static_cast<double>(coord[0]),
                                         static_cast<double>(coord[1]),
                                         static_cast<double>(coord[2])};
      auto ipair = coord_to_idx.find(coord_key);
      if (ipair == coord_to_idx.end())
        continue; // skip not found
      auto nbr_idx = ipair->second;
      // found?, not the same node?, not already grouped
      if (nbr_idx != ci && X2Y[nbr_idx] == -1) {
        // mark the idx, since nY is being accumulated to
        X2Y[nbr_idx] = nY.size();

        // modifies marker to have a set of marker nbs
        for (VID_t k = 0; k < nX[nbr_idx]->nbr.size(); ++k) {
          nYi->nbr.push_back(
              nX[nbr_idx]
                  ->nbr[k]); // append the neighbours of the group members
        }

        // update local average with x,y,z,sig elements from nX[nbr_idx]
        ++grp_size;

        // non roots can be averaged
        if (nYi->type != 0) {
          // adjust the coordinate to be an average
          float a = (grp_size - 1) / grp_size;
          float b = (1.0 / grp_size);
          nYi->x = a * nYi->x + b * nX[nbr_idx]->x;
          nYi->y = a * nYi->y + b * nX[nbr_idx]->y;
          nYi->z = a * nYi->z + b * nX[nbr_idx]->z;
          // average the radius
          nYi->radius = a * nYi->radius + b * nX[nbr_idx]->radius;
        }
      }
    }

    if (nYi->nbr.size() == 0) {
      ++no_neighbor_count;
      if (nYi->parent == nullptr) {
        std::cerr << "Non-fatal error: parent is also invalid... skipping "
                     "component\n";
        return 1; // error code
      }
      // cout << "nXi coord " << nX[ci]->x << ',' << nX[ci]->y << ',' <<
      // nX[ci]->z << '\n'; cout << "nYi coord " << nYi->x << ',' << nYi->y <<
      // ',' << nYi->z << '\n'; cout << "  parent coord " << nYi->parent->x <<
      // ',' << nYi->parent->y << ',' << nYi->parent->z << '\n';
    }

    // nYi.type = Node::AXON; // enforce type
    nY.push_back(nYi);
    return 0;
  };

  //// add soma nodes as independent groups at the beginning
  // for (long i = 0; i < nX.size(); ++i) {
  //// all somas are automatically kept
  // if (nX[i]->type == 0) {
  // X2Y[i] = nY.size();
  // auto nYi = new MyMarker(*nX[i]);
  // nY.push_back(nYi);
  //}
  //}

  // add soma nodes first
  for (VID_t i = 0; i < indices.size(); ++i) {
    long ci = indices[i];

    if (nX[ci]->type != 0)
      continue; // skip unless it's a root/soma

    if (check_node(ci))
      return std::vector<MyMarker *>(); // error exit
  }

  // add remaining nodes
  for (VID_t i = 0; i < indices.size(); ++i) {
    long ci = indices[i];

    if (X2Y[ci] != -1)
      continue; // skip if it was added to a group already

    if (check_node(ci))
      return std::vector<MyMarker *>(); // error exit
  }

  // once complete mapping is established, update the indices from
  // the original linear index to the new sparse group index according
  // to the X2Y idx map vector
  for (VID_t i = 1; i < nY.size(); ++i) {
    for (VID_t nbr_idx = 0; nbr_idx < nY[i]->nbr.size(); ++nbr_idx) {
      nY[i]->nbr[nbr_idx] = X2Y[nY[i]->nbr[nbr_idx]];
    }
  }

  check_nbr(nY); // remove doubles and self-linkages after grouping

  return nY;
}

template <typename T> class BfsQueue {
public:
  std::queue<T> kk;
  BfsQueue() {}
  void enqueue(T item) { this->kk.push(item); }
  T dequeue() {
    T output = kk.front();
    kk.pop();
    return output;
  }
  int size() { return kk.size(); }
  bool hasItems() { return !kk.empty(); }
};

// advantra based re-extraction of tree based on bfs
std::vector<MyMarker *>
extract_trees(std::vector<MyMarker *> nlist,
              bool remove_isolated_tree_with_one_node = false) {

  BfsQueue<int> q;
  std::vector<MyMarker *> tree;

  vector<int> dist(nlist.size());
  vector<int> nmap(nlist.size());
  vector<int> parent(nlist.size());

  for (VID_t i = 0; i < nlist.size(); ++i) {
    dist[i] = INT_MAX;
    nmap[i] = -1;   // indexing in output tree
    parent[i] = -1; // parent index in current tree
  }

  dist[0] = -1;

  // Node tree0(nlist[0]); // first element of the nodelist is dummy both in
  // input and output tree.clear(); tree.push_back(tree0);
  int treecnt = 0; // independent tree counter, will be obsolete

  int seed;

  auto get_undiscovered2 = [](std::vector<int> dist) -> int {
    for (VID_t i = 1; i < dist.size(); i++) {
      if (dist[i] == INT_MAX) {
        return i;
      }
    }
    return -1;
  };

  while ((seed = get_undiscovered2(dist)) > 0) {

    treecnt++;

    dist[seed] = 0;
    nmap[seed] = -1;
    parent[seed] = -1;
    q.enqueue(seed);

    int nodesInTree = 0;

    while (q.hasItems()) {

      // dequeue(), take from FIFO structure,
      // http://en.wikipedia.org/wiki/Queue_%28abstract_data_type%29
      int curr = q.dequeue();

      auto n = new MyMarker(*nlist[curr]);
      n->nbr.clear();
      if (n->type == 0) {
        n->parent = n;

        // otherwise choose the best single parent of the possible neighbors
      } else if (parent[curr] > 0) {
        n->nbr.push_back(nmap[parent[curr]]);
        // get the ptr to the marker from the id of the parent of the current
        n->parent = nlist[parent[curr]];
      } else if (nlist[curr]->nbr.size() != 0) {
        // get the ptr to the marker from the id of the min element of nbr
        // the smaller the id of an element, the higher precedence it has
        auto nbrs = nlist[curr]->nbr;
        auto min_idx = 0;
        auto min_element = nbrs[min_idx];
        for (VID_t i = 0; i < nbrs.size(); ++i) {
          if (nbrs[i] < min_element) {
            min_element = nbrs[i];
          }
        }
        n->parent = nlist[min_element];
      } else {
        throw std::runtime_error("node can't have 0 nbrs");
      }

      nmap[curr] = tree.size();
      tree.push_back(n);
      ++nodesInTree;

      // for each node adjacent to current
      for (VID_t nbr_idx = 0; nbr_idx < nlist[curr]->nbr.size(); nbr_idx++) {

        int adj = nlist[curr]->nbr[nbr_idx];

        if (dist[adj] == INT_MAX) {
          dist[adj] = dist[curr] + 1;
          parent[adj] = curr;
          // enqueue(), add to FIFO structure,
          // http://en.wikipedia.org/wiki/Queue_%28abstract_data_type%29
          q.enqueue(adj);
        }
      }

      // check if there were any neighbours
      if (nodesInTree == 1 && !q.hasItems() &&
          remove_isolated_tree_with_one_node) {
        tree.pop_back(); // remove the one that was just added
        nmap[curr] = -1; // cancel the last entry
      }
    }
  }

  return tree;
}
