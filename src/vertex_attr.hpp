#ifndef VERTEX_ATTR_H_
#define VERTEX_ATTR_H_

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#define assertm(exp, msg) assert(((void)msg, exp))

using std::cout;
using std::ios;

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

typedef VID_t handle_t; // FIXME switch to from VID_t

struct VertexAttr {
  OffsetCoord offsets;
  OffsetCoord parent;
  uint8_t radius = std::numeric_limits<uint8_t>::max();
  // most sig. bits (little-endian) refer to state : 1 bytes
  struct Bitfield edge_state;
  float value;
  handle_t handle;

  VertexAttr() : edge_state(0), radius(std::numeric_limits<uint8_t>::max()) {}

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
    std::ostringstream descript;
    descript << "offsets:" << offsets;
    descript << '\n';
    descript << "parent offsets:";
    if (valid_parent()) {
      descript << parent;
    } else {
      descript << '-';
    }
    descript << '\n';
    descript << "state:";
    for (int i = 7; i >= 0; i--) {
      descript << edge_state.test(i) ? "1" : "0";
    }
    descript << '\n';
    descript << "label:" + std::to_string(label());
    descript << '\n';
    descript << "radius:" + std::to_string(radius);
    descript << '\n';
    return descript.str();
  }

  /* returns whether this vertex has been added to a heap
   */
  bool valid_parent() const { return parent[0] || parent[1] || parent[2]; }

  /* returns whether this vertex has had its radius updated from the default max
   */
  bool valid_radius() const { return radius != std::numeric_limits<uint8_t>::max(); }

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
    os << "{offsets: " << v.offsets << ", radius: " << +(v.radius)
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
