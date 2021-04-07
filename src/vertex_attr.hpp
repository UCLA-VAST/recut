#ifndef VERTEX_ATTR_H_
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

struct bitfield {
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

  bitfield &operator=(const bitfield &a) {
    this->field_ = a.field_;
    return *this;
  }

  // friend std::ostream& (std::ostream& os, bitfield const& bf) {
  // char[8]
  // os
  //}

  bool operator==(const bitfield &a) const { return a.field_ == this->field_; }

  // defaults as 1100 0000 unvisited with no connections
  bitfield() : field_(192) {}
  bitfield(uint8_t field) : field_(field) {}
};

struct VertexAttr {
  OffsetCoord offsets;
  OffsetCoord parent;
  uint8_t radius = std::numeric_limits<uint8_t>::max();
// most sig. bits (little-endian) refer to state : 1 bytes
  struct bitfield edge_state; 

  // constructors
  // defaults as 192 i.e. 1100 0000 unvisited
  VertexAttr()
      : edge_state(192), 
        radius(numeric_limits<uint8_t>::max()) 
         {}

  // constructors
  // defaults as selected
  VertexAttr(OffsetCoord offsets)
      : offsets(offsets), 
        radius(numeric_limits<uint8_t>::max()) 
         { this->mark_selected(); }

  // copy constructor
  VertexAttr(const VertexAttr &a)
      : edge_state(a.edge_state), offsets(a.offsets), radius(a.radius), parent(a.parent) {
  }

  VertexAttr(bitfield edge_state, OffsetCoord offsets, OffsetCoord parent)
      : edge_state(edge_state), offsets(offsets), parent(parent) {}

  VertexAttr(bitfield edge_state, OffsetCoord offsets, OffsetCoord parent, uint8_t radius)
      : edge_state(edge_state), offsets(offsets), parent(parent), radius (radius) {}

  bool root() const {
    return (!edge_state.test(7) && !edge_state.test(6)); // 00XX XXXX ROOT
  }

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
  bool valid_parent() const { 
    return parent[0] || parent[1] || parent[2];
  }

  /* returns whether this vertex has had its radius updated from the default max */
  bool valid_radius() const { return radius != numeric_limits<uint8_t>::max(); }

  bool selected() const {
    return (edge_state.test(7) && !edge_state.test(6)); // 10XX XXXX KNOWN NEW
  }

  // change an independent flag to indicate this vertex has been visited
  void prune_visit() {
    // XXX1 XXXX
    edge_state.set(4);
  }

  // check an independent flag to see if during a prune update
  // this vertex has already been visited and had it's parent
  // changed
  bool prune_visited() const {
    // XXX? XXXX
    return edge_state.test(4);
  }

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
    if (this->unvisited()) {
      return '-';
    }
    if (this->band()) {
      return 'B';
    }
    return '?';
  }

  friend std::ostream &operator<<(std::ostream &os, const VertexAttr &v) {
    os << "{offsets: " << coord_to_str(v.offsets )
       << ", radius: " << +(v.radius) << ", label: " << v.label() << '}';
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
           (edge_state.field_ == a.edge_state.field_) && (radius == a.radius)
           && (parent == a.parent);
  }

  bool operator!=(const VertexAttr &a) const {
    return (offsets != a.offsets) ||
           (edge_state.field_ != a.edge_state.field_) || (radius != a.radius)
           || (parent != a.parent);
  }

  void mark_branch_point() {
    // XXXX XX1X
    edge_state.set(1);
    edge_state.unset(0);
  }

  bool is_branch_point() const {
    // XXXX XX?X
    return edge_state.test(1);
  }

  bool has_single_child() const {
    // XXXX XXX?
    return edge_state.test(0);
  }

  void mark_has_single_child() {
    // XXXX XXX1
    edge_state.set(0);
    edge_state.unset(1);
  }

  void mark_surface() {
    edge_state.set(5);
  }

  void mark_selected() {
    edge_state.set(7); // set as KNOWN NEW
    edge_state.unset(6);
  }

  bool unvisited() const { // 11XX XXXX default unvisited state
    return edge_state.test(6) && edge_state.test(7);
  }

  void mark_unvisited() { // 11XX XXXX default unvisited state
    edge_state.set(7);
    edge_state.set(6);
  }

  /* returns true if X1XX XXXX
   * means node is either BAND or unvisited
   * and has not been selected as KNOWN_NEW
   */
  bool unselected() const {
    // return edge_state.test(6);
    return !selected();
  }

  void mark_root() {
    // no connections is already default
    // roots can also be surface
    edge_state.unset(7);
    edge_state.unset(6);
  }

  void mark_band() {
    // add to band (01XX XXXX)
    edge_state.unset(7);
    edge_state.set(6);
  }

  bool surface() const { // XX1X XXXX
    return edge_state.test(5);
  }

  bool band() const { // 01XX XXXX
    return edge_state.test(6) && !(edge_state.test(7));
  }

  // remaining 6 bits indicate connections with neighbors:
  // each value is a boolean indicating a connection with that neighbor
  // from least sig. to most sig bit the connections are ordered as follows:
  // bit | connection
  // ----------------
  // 0   | x - 1
  // 1   | x + 1
  // 2   | y - 1
  // 3   | y + 1
  // 4   | z - 1
  // 5   | z + 1
  // 6-7 | 11 indicates unvisited
  // 6-7 | 01 indicates band
  // 6-7 | 10 indicates KNOWN_NEW
  // 6-7 | 00 indicates KNOWN_FIX of
  //       0000 0000, KNOWN_FIX ROOT
  //
  // Ex.
  // 192 = 1100 0000 :
  //   initial state of all but root node, no connections to any neighbors
  // 2 = 0000 0010 :
  //   indicates a selected node with one connection at x + 1
  //   0000 0000 indicates root or a fixed selected value, does not need
  //   to be reprocessed in ghost cell regions
};

#endif
