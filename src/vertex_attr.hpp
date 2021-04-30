#ifndef VERTEX_ATTR_H_
// ASSERT_TRUE(matches[0]);
// ASSERT_TRUE(matches[1]);
// ASSERT_TRUE(matches[2]);

#define VERTEX_ATTR_H_
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using std::cout;
using std::ios;
using std::numeric_limits;
using std::vector;

#include "utils.hpp"

struct Bitfield {
  uint8_t field_;

  void set() {
    // field_ |= 1 << 0;
    // field_ |= 1 << 1;
    // field_ |= 1 << 2;
    // field_ |= 1 << 3;
    // field_ |= 1 << 4;
    // field_ |= 1 << 5;
    // field_ |= 1 << 6;
    // field_ |= 1 << 7;
    field_ = 255;
  }

  void reset() { field_ = 0; }

  bool test(int idx) const { return static_cast<bool>(field_ & (1 << idx)); }

  void set(int idx) { field_ |= 1 << idx; }

  void unset(int idx) { field_ &= ~(1 << idx); }

  bool none() const { return field_ == 0; }

  Bitfield &operator=(const Bitfield &a) {
    this->field_ = a.field_;
    return *this;
  }

  bool operator==(const Bitfield &a) const { return a.field_ == this->field_; }

  Bitfield() : field_(0) {}
  Bitfield(uint8_t field) : field_(field) {}
};

struct VertexAttr {
  OffsetCoord offsets;
  OffsetCoord parent;
  uint8_t radius = std::numeric_limits<uint8_t>::max();
  // most sig. bits (little-endian) refer to state : 1 bytes
  struct Bitfield edge_state;

  VertexAttr() : edge_state(0), radius(numeric_limits<uint8_t>::max()) {}

  VertexAttr(const VertexAttr &a)
      : edge_state(a.edge_state), offsets(a.offsets), radius(a.radius),
        parent(a.parent) {}

  VertexAttr(Bitfield edge_state, OffsetCoord offsets)
      : edge_state(edge_state), offsets(offsets) {}

  // connected stage
  VertexAttr(Bitfield edge_state, OffsetCoord offsets, OffsetCoord parent)
      : edge_state(edge_state), offsets(offsets), parent(parent) {}

  // prune and radius stage
  VertexAttr(Bitfield edge_state, OffsetCoord offsets, uint8_t radius)
      : edge_state(edge_state), offsets(offsets), radius(radius) {}

  VertexAttr(Bitfield edge_state, OffsetCoord offsets, OffsetCoord parent,
             uint8_t radius)
      : edge_state(edge_state), offsets(offsets), parent(parent),
        radius(radius) {}

  // you can pipe the output directly to std::cout
  std::string description() const {
    std::string descript = "offsets:" + coord_to_str(offsets);
    descript += '\n';
    descript += "parent offsets:";
    auto parent_str = std::string("-");
    if (valid_parent()) {
      parent_str = coord_to_str(parent);
    }
    descript += parent_str;
    descript += '\n';
    descript += '\n';
    descript += "state:";
    for (int i = 7; i >= 0; i--) {
      descript += edge_state.test(i) ? "1" : "0";
    }
    descript += '\n';
    descript += "label:" + std::to_string(label());
    descript += '\n';
    descript += "radius:" + std::to_string(radius);
    descript += '\n';
    return descript;
  }

  /* returns whether this vertex has been added to a heap
   */
  bool valid_parent() const { return parent[0] || parent[1] || parent[2]; }

  /* returns whether this vertex has had its radius updated from the default max
   */
  bool valid_radius() const { return radius != numeric_limits<uint8_t>::max(); }

  void set_parent(OffsetCoord coord) {
    this->parent[0] = coord[0];
    this->parent[1] = coord[1];
    this->parent[2] = coord[2];
  }

  // unsets any previous marked connect
  // a connection can only be in 1 of 6 directions
  // therefore throws if pass above idx value 5
  template <typename T> void set_idx(T idx) {
    assertm(idx <= 7, "idx must be <= 7");
    edge_state.set(idx);
  }

  void copy_edge_state(const VertexAttr &a) { edge_state = a.edge_state; }

  char label() const {
    if (this->root()) {
      return 'R';
    }
    if (this->selected()) {
      return 'V';
    }
    return '-';
  }

  friend std::ostream &operator<<(std::ostream &os, const VertexAttr &v) {
    os << "{offsets: " << coord_to_str(v.offsets) << ", radius: " << +(v.radius)
       << ", label: " << v.label() << '}';
    return os;
  }

  VertexAttr &operator=(const VertexAttr &a) {
    edge_state.field_ = a.edge_state.field_;
    offsets = a.offsets;
    radius = a.radius;
    parent = a.parent;
    return *this;
  }

  bool operator==(const VertexAttr &a) const {
    return (offsets == a.offsets) &&
           (edge_state.field_ == a.edge_state.field_) && (radius == a.radius) &&
           (parent == a.parent);
  }

  bool operator!=(const VertexAttr &a) const {
    return (offsets != a.offsets) ||
           (edge_state.field_ != a.edge_state.field_) || (radius != a.radius) ||
           (parent != a.parent);
  }

  // void mark_branch_point() {
  //// XXXX XX1X
  // edge_state.set(1);
  // edge_state.unset(0);
  //}

  // bool is_branch_point() const {
  //// XXXX XX?X
  // return edge_state.test(1);
  //}

  // bool has_single_child() const {
  //// XXXX XXX?
  // return edge_state.test(0);
  //}

  // void mark_has_single_child() {
  //// XXXX XXX1
  // edge_state.set(0);
  // edge_state.unset(1);
  //}

  void mark_selected() { edge_state.set(0); }

  bool selected() const { return edge_state.test(0); }

  bool unselected() const { return !selected(); }

  void mark_surface() { edge_state.set(1); }

  bool surface() const { return edge_state.test(1); }

  void mark_root() {
    // roots can also be surface, selected, etc.
    edge_state.set(3);
  }

  bool root() const { return edge_state.test(3); }

  // check an independent flag to see if during a prune update
  // this vertex has already been visited and had it's parent
  // changed
  bool prune_visited() const {
    // XXX? XXXX
    return edge_state.test(4);
  }

  void prune_visit() {
    // XXX1 XXXX
    edge_state.set(4);
  }

  void mark_tombstone() { edge_state.set(5); }

  bool tombstone() const { return edge_state.test(5); }

  // ordered from right to left:
  // 7 6 5 4   3 2 1 0
  // ind | meaning
  // ----------------
  // 0   | selected
  // 1   | surface
  // 2   | band
  // 3   | root
  // 4   | prune visit
  // 5   | tombstone
};

#endif
