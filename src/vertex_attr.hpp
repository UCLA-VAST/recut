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

typedef VID_t handle_t; // FIXME switch to from VID_t

struct VertexAttr {
  VID_t vid; // 4 bytes or 8 bytes depending on environment VID variable
  // heap implementation
  // stores block id to handle id in that thread blocks heap
  // size of this map is max 4 since a corner voxel can it at most
  // 3 other blocks ghost zones
  handle_t handle; // same type as VID_t
  VID_t parent;
  struct bitfield
      edge_state; // most sig. bits (little-endian) refer to state : 1 bytes
  uint8_t radius = std::numeric_limits<uint8_t>::max();

  // constructors
  // defaults as 192 i.e. 1100 0000 unvisited with no connections
  VertexAttr()
      : edge_state(192), 
        vid(numeric_limits<VID_t>::max()),
        handle(numeric_limits<handle_t>::max()),
        radius(numeric_limits<uint8_t>::max()) ,
        parent(numeric_limits<VID_t>::max()) {}

  VertexAttr(VID_t vid)
      : edge_state(192), vid(vid),
        handle(numeric_limits<handle_t>::max()),
        radius(numeric_limits<uint8_t>::max()),
        parent(numeric_limits<VID_t>::max()) {}

  // copy constructor
  VertexAttr(const VertexAttr &a)
      : edge_state(a.edge_state), vid(a.vid), radius(a.radius), parent(a.parent) {
  }

  VertexAttr(uint8_t edge_state, VID_t vid, VID_t parent)
      : edge_state(edge_state), vid(vid), parent(parent) {}

  VertexAttr(struct bitfield edge_state, VID_t vid, uint8_t radius, VID_t parent)
      : edge_state(edge_state), vid(vid), radius(radius), parent(parent) {}

  bool root() const {
    return (!edge_state.test(7) && !edge_state.test(6)); // 00XX XXXX ROOT
  }

  // you can pipe the output directly to std::cout
  std::string description() const {
    std::string descript = "vid:" + std::to_string(vid);
    descript += '\n';
    descript += "parent vid:";
    auto parent_vid = std::string("-");
    if (valid_parent()) {
      parent_vid = std::to_string(parent);
    }
    descript += parent_vid;
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
  bool valid_parent() const { return parent != numeric_limits<VID_t>::max(); }

  /* returns whether this vertex has been added to a heap
   */
  bool valid_handle() const { return (handle != numeric_limits<handle_t>::max()); }

  /* returns whether this vertex has had its radius updated from the default max
   */
  bool valid_radius() const { return radius != numeric_limits<uint8_t>::max(); }

  /* returns whether this vertex has been added to a heap
   */
  bool valid_vid() const { 
    //std::cout << "valid vid\n";
    //std::cout << vid << '\n';
    //std::cout << numeric_limits<VID_t>::max() << '\n';
    //std::cout << "valid ? " << (vid != numeric_limits<VID_t>::max()) << '\n';
    return (vid != numeric_limits<VID_t>::max()); }

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

  void set_parent(VID_t vid) {
    this->parent = vid;
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
    os << "{vid: " << v.vid 
       << ", radius: " << +(v.radius) << ", label: " << v.label() << '}';
    return os;
  }

  VertexAttr &operator=(const VertexAttr &a) {
    edge_state.field_ = a.edge_state.field_;
    vid = a.vid;
    radius = a.radius;
    parent = a.parent;
    // do not copy handle_t
    return *this;
  }

  bool operator==(const VertexAttr &a) const {
    return (vid == a.vid) &&
           (edge_state.field_ == a.edge_state.field_) && (radius == a.radius)
           && (parent == a.parent);
  }

  bool operator!=(const VertexAttr &a) const {
    return (vid != a.vid) ||
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

  void mark_root(VID_t set_vid) {
    this->mark_root();
    vid = set_vid;
  }

  void mark_band() {
    // add to band (01XX XXXX)
    edge_state.unset(7);
    edge_state.set(6);
  }

  void mark_band(VID_t set_vid) {
    this->mark_band();
    vid = set_vid;
  }

  bool surface() const { // XX1X XXXX
    return edge_state.test(5);
  }

  bool band() const { // 01XX XXXX
    return edge_state.test(6) && !(edge_state.test(7));
  }

  /**
   *  remaining 6 bits indicate connections with neighbors:
   *  each value is a boolean indicating a connection with that neighbor
   *  from least sig. to most sig bit the connections are ordered as follows:
   *  bit | connection
   *  ----------------
   *  0   | x - 1
   *  1   | x + 1
   *  2   | y - 1
   *  3   | y + 1
   *  4   | z - 1
   *  5   | z + 1
   */
  template <typename T> std::vector<VID_t> connections(T nxpad, T nxypad) const {
    std::vector<VID_t> connect;
    connect.reserve(6);
    // returns vid of itself if there are no connections
    if (!edge_state.test(5) && !edge_state.test(4) && !edge_state.test(3) &&
        !edge_state.test(2) && !edge_state.test(1) && !edge_state.test(0))
      connect.push_back(vid);
    // accessed from lowest to highest address to faciliate coalescing
    if (edge_state.test(4))
      connect.push_back(vid - nxypad);
    if (edge_state.test(2))
      connect.push_back(vid - nxpad);
    if (edge_state.test(0))
      connect.push_back(vid - 1);
    if (edge_state.test(1))
      connect.push_back(vid + 1);
    if (edge_state.test(3))
      connect.push_back(vid + nxpad);
    if (edge_state.test(5))
      connect.push_back(vid + nxypad);
    return connect;
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
