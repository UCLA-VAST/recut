auto get_id_map = []() {
  std::unordered_map<GridCoord, uint32_t> coord_to_swc_id;
  // add a dummy value that will never be on to the map so that real indices
  // start at 1
  coord_to_swc_id[GridCoord(INT_MIN, INT_MIN, INT_MIN)] = 0;
  return coord_to_swc_id;
};

// also return a mapping from coord to id
auto create_tree_indices = [](std::vector<MyMarker *> &tree) {
  // start a new blank map for coord to a unique swc id
  auto coord_to_swc_id = get_id_map();

  // iter those marker*
  auto indices = tree | rv::transform([&coord_to_swc_id](const auto marker) {
                   auto coord = GridCoord(marker->x, marker->y, marker->z);
                   // roots have parents of themselves
                   auto parent_coord =
                       marker->type
                           ? GridCoord(marker->parent->x, marker->parent->y,
                                       marker->parent->z)
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

auto create_coord_to_idx = [](const std::vector<MyMarker *> &cluster) {
  // build to coord_to_idx
  std::unordered_map<GridCoord, VID_t> coord_to_idx;
  rng::for_each(cluster | rv::enumerate, [&](auto imarker) {
    auto [id, marker] = imarker;
    coord_to_idx[GridCoord(marker->x, marker->y, marker->z)] = id;
  });
  return coord_to_idx;
};

// markers have a ptr to parent, all aprents must point to marker *within* its
// current vector if it points to marker within a previous vector, undefined
// behavior
auto adjust_parent_ptrs = [](std::vector<MyMarker *> &cluster) {
  auto coord_to_idx = create_coord_to_idx(cluster);

  // adjust in place
  rng::for_each(cluster, [&](auto marker) {
    if (marker->type) { // skips roots
      const auto parent_coord =
          GridCoord(marker->parent->x, marker->parent->y, marker->parent->z);
      const auto parent_pair = coord_to_idx.find(parent_coord);
      if (parent_pair == coord_to_idx.end()) {
        std::cout << "Invalid " << *marker << ' ' << parent_coord << '\n';
        throw std::runtime_error("adjust_parent_ptrs() can not operate on "
                                 "clusters with out-of-cluster parents");
      } else {
        marker->parent = cluster[parent_pair->second];
      }
    }
  });
};

auto is_cluster_self_contained = [](const std::vector<MyMarker *> &cluster) {
  auto coord_to_idx = create_coord_to_idx(cluster);

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
        const auto parent_coord =
            GridCoord(marker->parent->x, marker->parent->y, marker->parent->z);
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
  auto coord_to_idx = create_coord_to_idx(cluster);

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
                  const auto parent_coord = GridCoord(
                      marker->parent->x, marker->parent->y, marker->parent->z);
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
auto write_swc = [](std::vector<MyMarker *> &tree, std::array<float,3> voxel_size,
                    std::string component_dir_fn = ".", CoordBBox bbox = {},
                    bool bbox_adjust = false) {
  auto root = tree.front();
  if (root->type)
    throw std::runtime_error("First marker of tree must be a root (type 0)");

  // start swc and add header metadata
  auto swc_name = component_dir_fn + "/tree-with-soma-xyz-" +
                  std::to_string(static_cast<int>(root->x)) + '-' +
                  std::to_string(static_cast<int>(root->y)) + '-' +
                  std::to_string(static_cast<int>(root->z)) + ".swc";
  std::ofstream swc_file;
  swc_file.open(swc_name);
  swc_file << "# Crop windows bounding volume: " << bbox << '\n';
  swc_file << "# id type_id x y z radius parent_id\n";

  auto coord_to_swc_id = get_id_map();

  // iter those marker*
  rng::for_each(tree, [&swc_file, &coord_to_swc_id, &bbox,
                       bbox_adjust,&voxel_size](const auto marker) {
    auto coord = GridCoord(marker->x, marker->y, marker->z);
    auto is_root = marker->type == 0;
    auto parent_offset = zeros();

    if (!is_root) {
      auto parent_coord =
          GridCoord(marker->parent->x, marker->parent->y, marker->parent->z);
      parent_offset = coord_sub(parent_coord, coord);
    }
    // print_swc_line() expects an offset to a parent
    print_swc_line(coord, is_root, marker->radius, parent_offset, bbox,
                   swc_file, coord_to_swc_id, voxel_size, bbox_adjust);
  });
};

auto create_child_count = [](std::vector<MyMarker *> &tree) {
  // build a map to save coords with 1 or 2 children
  // coords not in this map are therefore leafs
  auto child_count = std::unordered_map<GridCoord, uint8_t>();
  rng::for_each(tree, [&child_count](auto marker) {
    if (marker->parent) {
      assertm(marker->parent, "Parent missing");
      const auto parent_coord =
          GridCoord(marker->parent->x, marker->parent->y, marker->parent->z);
      const auto val_count = child_count.find(parent_coord);
      // cout << parent_coord << " <- "
      //<< GridCoord(marker->x, marker->y, marker->z) << '\n';
      if (val_count == child_count.end()) // not found
        child_count.insert_or_assign(parent_coord, 1);
      else
        child_count.insert_or_assign(parent_coord, val_count->second + 1);
    } // ignore those without parents
  });
  return child_count;
};

// returns a new set of valid markers
std::vector<MyMarker *>
prune_short_branches(std::vector<MyMarker *> &tree,
                     int min_branch_length = MIN_BRANCH_LENGTH) {
  auto child_count = create_child_count(tree);
  // for (auto m : child_count) {
  // cout << m.first << ' ' << +(m.second) << '\n';
  //}

  auto is_a_branch = [&child_count](auto const marker) {
    const auto val_count =
        child_count.find(GridCoord(marker->x, marker->y, marker->z));
    return val_count != child_count.end() ? (val_count->second > 1) : false;
  };

  auto is_a_leaf = [&child_count](auto const marker) {
    return child_count.find(GridCoord(marker->x, marker->y, marker->z)) ==
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
  const auto val_count =
      child_count.find(GridCoord(marker->x, marker->y, marker->z));
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
        const auto old_parent_coord =
            GridCoord(marker->parent->x, marker->parent->y, marker->parent->z);

        // create a new averaged node between the parent and the parent->parent
        // already protected against parent of parent all non somas have a
        // parent
        assertm(marker->parent->parent, "Only roots can have invalid parents");
        const auto parent_parent_coord =
            GridCoord(marker->parent->parent->x, marker->parent->parent->y,
                      marker->parent->parent->z);
        auto sum = parent_parent_coord + old_parent_coord;
        auto new_parent_coord =
            GridCoord(sum.x() / 2, sum.y() / 2, sum.z() / 2);
        auto new_parent = new MyMarker(
            new_parent_coord.x(), new_parent_coord.y(), new_parent_coord.z(),
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

// sphere grouping compaction/pruning strategy inspired by Advantra's code
// switched from n^2 to nr^3 where r is the radii of a given node
std::vector<MyMarker *>
advantra_prune(vector<MyMarker *> nX, uint16_t prune_radius_factor,
               std::unordered_map<GridCoord, VID_t> coord_to_idx) {

  std::vector<MyMarker *> nY;
  auto no_neighbor_count = 0;

  // nX[0].corr = FLT_MAX; // so that the dummy node gets index 0 again, larges
  // correlation
  vector<long> indices(nX.size());
  for (long i = 0; i < indices.size(); ++i)
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
    for (const auto coord : sphere_iterator(center, radius_for_pruning)) {
      auto ipair = coord_to_idx.find(coord);
      if (ipair == coord_to_idx.end())
        continue; // skip not found
      auto nbr_idx = ipair->second;
      // found?, not the same node?, not already grouped
      if (nbr_idx != ci && X2Y[nbr_idx] == -1) {
        // mark the idx, since nY is being accumulated to
        X2Y[nbr_idx] = nY.size();

        // modifies marker to have a set of marker nbs
        for (int k = 0; k < nX[nbr_idx]->nbr.size(); ++k) {
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
          // nYi->radius = a * nYi->radius + nX[nbr_idx]->radius;
          // adjust the radii to be a max
          // this is a cheat to avoid having to sort the list of markers
          // by radii, since the maximum SDF value will tend to
          // be found and is about the same along the centerline of neurites
          if (nX[nbr_idx]->radius > nYi->radius) {
            nYi->radius = nX[nbr_idx]->radius;
          }
        }
      }
    }

    if (nYi->nbr.size() == 0) {
      ++no_neighbor_count;
      if (nYi->parent == nullptr)
        throw std::runtime_error("parent is also invalid");
      // cout << "nXi coord " << nX[ci]->x << ',' << nX[ci]->y << ',' <<
      // nX[ci]->z << '\n'; cout << "nYi coord " << nYi->x << ',' << nYi->y <<
      // ',' << nYi->z << '\n'; cout << "  parent coord " << nYi->parent->x <<
      // ',' << nYi->parent->y << ',' << nYi->parent->z << '\n';
    }

    // nYi.type = Node::AXON; // enforce type
    nY.push_back(nYi);
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
  for (long i = 0; i < indices.size(); ++i) {
    long ci = indices[i];

    if (nX[ci]->type != 0)
      continue; // skip unless it's a root/soma

    check_node(ci);
  }

  // add remaining nodes
  for (long i = 0; i < indices.size(); ++i) {
    long ci = indices[i];

    if (X2Y[ci] != -1)
      continue; // skip if it was added to a group already

    check_node(ci);
  }

  // once complete mapping is established, update the indices from
  // the original linear index to the new sparse group index according
  // to the X2Y idx map vector
  for (int i = 1; i < nY.size(); ++i) {
    for (int nbr_idx = 0; nbr_idx < nY[i]->nbr.size(); ++nbr_idx) {
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

  for (int i = 0; i < nlist.size(); ++i) {
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
    for (int i = 1; i < dist.size(); i++) {
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
        for (int i = 0; i < nbrs.size(); ++i) {
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
      for (int nbr_idx = 0; nbr_idx < nlist[curr]->nbr.size(); nbr_idx++) {

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
