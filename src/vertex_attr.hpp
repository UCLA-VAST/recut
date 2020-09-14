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

#include "config.hpp"
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
  VertexAttr *parent;
  float value; // distance to source: 4 bytes
  struct bitfield
      edge_state; // most sig. bits (little-endian) refer to state : 1 bytes
  uint8_t radius = std::numeric_limits<uint8_t>::max();

  // constructors
  VertexAttr()
      : edge_state(192), value(numeric_limits<float>::max()),
        vid(numeric_limits<VID_t>::max()),
        handle(numeric_limits<handle_t>::max()),
        radius(numeric_limits<uint8_t>::max()) ,
        parent(nullptr) {}

  VertexAttr(float value)
      : edge_state(192), value(value), vid(numeric_limits<VID_t>::max()),
        handle(numeric_limits<handle_t>::max()),
        radius(numeric_limits<uint8_t>::max()),
        parent(nullptr) {}

  // copy constructor
  VertexAttr(const VertexAttr &a)
      : edge_state(a.edge_state), value(a.value), vid(a.vid), radius(a.radius), parent(a.parent) {
  }

  // emplace back constructor
  VertexAttr(struct bitfield edge_state, float value, VID_t vid, uint8_t radius, VertexAttr* parent)
      : edge_state(edge_state), value(value), vid(vid), radius(radius), parent(parent) {}

  bool root() const {
    return (!edge_state.test(7) && !edge_state.test(6)); // 00XX XXXX ROOT
  }

  std::string description() const {
    std::cout << vid << '\n';
    std::string descript = "vid:" + std::to_string(vid);
    descript += '\n';
    descript += "parent vid:";
    auto parent_vid = std::string("-");
    if (parent) {
      parent_vid = std::to_string(parent->vid);
    }
    descript += parent_vid;
    descript += '\n';
    descript += "value:" + std::to_string(value);
    descript += '\n';
    descript += "state:";
    for (int i = 7; i >= 0; i--) {
      descript += edge_state.test(i) ? "1" : "0";
    }
    descript += '\n';
    descript += "radius:" + std::to_string(radius);
    descript += '\n';
    return descript;
  }

  /* returns whether this vertex has been added to a heap
   */
  bool valid_parent() const { return parent != nullptr; }

  /* returns whether this vertex has been added to a heap
   */
  bool valid_handle() const { return (handle != numeric_limits<handle_t>::max()); }

  /* returns whether this vertex has had its radius updated from the default max
   */
  bool valid_radius() const { return radius != numeric_limits<uint8_t>::max(); }

  /* returns whether this vertex has had its value updated from the default max
   */
  bool valid_value() const { return value != numeric_limits<float>::max(); }

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

  void set_parent(VertexAttr* v) {
    this->parent = v;
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
      return 'S';
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
    os << "{vid: " << v.vid << ", value: " << v.value
       << ", radius: " << +(v.radius) << ", label: " << v.label() << '}';
    return os;
  }

  VertexAttr &operator=(const VertexAttr &a) {
    edge_state.field_ = a.edge_state.field_;
    vid = a.vid;
    value = a.value;
    radius = a.radius;
    parent = a.parent;
    // do not copy handle_t
    return *this;
  }

  bool operator==(const VertexAttr &a) const {
    return (value == a.value) && (vid == a.vid) &&
           (edge_state.field_ == a.edge_state.field_) && (radius == a.radius)
           && (parent == a.parent);
  }

  bool operator!=(const VertexAttr &a) const {
    return (value != a.value) || (vid != a.vid) ||
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
    edge_state.test(1);
  }

  bool has_single_child() const {
    // XXXX XXX?
    edge_state.test(0);
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

  void mark_root(VID_t set_vid) {
    // all zeros
    edge_state.reset();
    vid = set_vid;
  }

  void mark_band() {
    // add to band (01XX XXXX)
    edge_state.unset(7);
    edge_state.set(6);
  }

  void mark_band(VID_t set_vid) {
    // add to band (01XX XXXX)
    edge_state.unset(7);
    edge_state.set(6);
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

//// min_heap
// inline bool compare_VertexAttr::operator() (const struct VertexAttr* n1,
// const struct VertexAttr* n2) const { return n1->value > n2->value;
//}

template <class T>
class NeighborHeap // Basic Min heap
{
public:
  NeighborHeap() {
    // FIXME this is a disaster
    elems.reserve(10000);
  }

  void print(std::string stage) {
    if (stage == "radius") {
      for (auto &vert : elems) {
        cout << +(vert->radius) << " ";
      }
    } else {
      for (auto &vert : elems) {
        cout << +(vert->value) << " ";
      }
    }
    cout << '\n';
  }

  // equivalent to peek, read only
  T *top() {
    if (empty()) {
      throw;
      return 0;
    } else {
      return elems[0];
    }
  }

  void check_empty() {
    if (empty())
      throw;
  }

  // remove top element
  T *pop(VID_t ib, std::string cmp_field) {
    check_empty();
    T *min_elem = elems[0];

    if (elems.size() == 1)
      elems.clear();
    else {
      elems[0] = elems[elems.size() - 1];
      elems[0]->handle = 0; // update handle value
      elems.erase(elems.begin() + elems.size() - 1);
      down_heap(0, ib, cmp_field);
    }
    // min_elem->handles.erase(ib);
    // min_elem->handles[ib] = std::numeric_limits<handle_t>::max();
    min_elem->handle = std::numeric_limits<handle_t>::max();

    stats_update();
    return min_elem;
  }

  // FIXME return type to handle_t later
  void push(T *t, VID_t ib, std::string cmp_field) {
    elems.push_back(t);
    // mark handle such that all swaps are recorded in correct handle
    t->handle = elems.size() - 1;
    up_heap(elems.size() - 1, ib, cmp_field);

    // check for a new max_size
    if (elems.size() > this->max_size) {
      this->max_size = elems.size();
    }

    this->op_count++;
    this->cumulative_count += this->elems.size();
    stats_update();
  }

  handle_t find(handle_t vid) {
    for (handle_t i = 0; i < elems.size(); i++) {
      if (elems[i]->vid == vid) {
        return i;
      }
    }
    assertm(false, "Did not find vid in heap on call to vid"); // did not find
    return std::numeric_limits<handle_t>::max();
  }

  bool empty() { return elems.empty(); }

  template <typename TNew>
  void update(T *updated_node, VID_t ib, TNew new_field,
              std::string cmp_field) {
    handle_t id = updated_node->handle;
    check_empty();
    assertm(elems[id]->vid == updated_node->vid, "VID doesn't match");
    assertm(elems[id]->handle == id, "handle doesn't match");
    if (cmp_field == "value") {
      auto old_value = elems[id]->value;
      elems[id]->value = new_field;
      // elems[id]->handle = id;
      if (new_field < old_value)
        up_heap(id, ib, cmp_field);
      else if (new_field > old_value)
        down_heap(id, ib, cmp_field);
    } else if (cmp_field == "radius") {
      auto old_radius = elems[id]->radius;
      elems[id]->radius = new_field;
      // elems[id]->handle = id;
      if (new_field < old_radius)
        up_heap(id, ib, cmp_field);
      else if (new_field > old_radius)
        down_heap(id, ib, cmp_field);
    } else {
      assertm(false, "`cmp_field` arg not recognized");
    }
  }
  int size() { return elems.size(); }

  uint64_t op_count = 0;
  uint64_t max_size = 0;
  uint64_t cumulative_count = 0;

private:
  vector<T *> elems;

  // keep track of total sizes summed across all valid
  void stats_update() {
    this->op_count++;
    this->cumulative_count += this->elems.size();
  }

  // used for indexing handle
  bool swap_heap(int id1, int id2, VID_t ib, std::string cmp_field) {
    if (id1 < 0 || id1 >= elems.size() || id2 < 0 || id2 >= elems.size())
      return false;
    if (id1 == id2)
      return false;
    int pid = id1 < id2 ? id1 : id2;
    int cid = id1 > id2 ? id1 : id2;
    assert(cid == 2 * (pid + 1) - 1 || cid == 2 * (pid + 1));

    if (cmp_field == "radius") {
      if (elems[pid]->radius <= elems[cid]->radius)
        return false;
    } else {
      if (elems[pid]->value <= elems[cid]->value)
        return false;
    }
    T *tmp = elems[pid];
    elems[pid] = elems[cid];
    elems[cid] = tmp;
    elems[pid]->handle = pid;
    elems[cid]->handle = cid;
    return true;
  }

  void up_heap(int id, VID_t ib, std::string cmp_field) {
    int pid = (id + 1) / 2 - 1;
    if (swap_heap(id, pid, ib, cmp_field))
      up_heap(pid, ib, cmp_field);
  }

  void down_heap(int id, VID_t ib, std::string cmp_field) {
    int cid1 = 2 * (id + 1) - 1;
    int cid2 = 2 * (id + 1);
    if (cid1 >= elems.size())
      return;
    else if (cid1 == elems.size() - 1) {
      swap_heap(id, cid1, ib, cmp_field);
    } else if (cid1 < elems.size() - 1) {
      int cid;
      if (cmp_field == "radius") {
        cid = elems[cid1]->radius < elems[cid2]->radius ? cid1 : cid2;
      } else {
        cid = elems[cid1]->value < elems[cid2]->value ? cid1 : cid2;
      }
      if (swap_heap(id, cid, ib, cmp_field))
        down_heap(cid, ib, cmp_field);
    }
  }
};

using local_heap = NeighborHeap<VertexAttr>;

#endif
