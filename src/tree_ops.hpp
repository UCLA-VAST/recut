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
