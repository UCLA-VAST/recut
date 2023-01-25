// last change; by PHC 20121127. update the swc and marker saving funcs

// modified from
// vaa3d/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/my_surf_objs.h
// mzhu 05/23/2019
#pragma once

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <vector>

typedef uint64_t VID_t;

using namespace std;

#define MARKER_BASE                                                            \
  1 // the basic offset of marker is 0, the marker coordinate will be converted
    // when read and save

// root node with parent 0
struct MyMarker {
  double x;
  double y;
  double z;
  union {
    double radius;
  };
  // type 3 is dendrite
  // type 0 is soma
  int type;
  MyMarker *parent;
  // temporary solution for Advantra prune
  // assumes all markers have been placed in a dense array
  // and can be accessed via a linear idx, taking from advantra method
  std::vector<VID_t> nbr;
  MyMarker() {
    x = y = z = radius = 0.0;
    type = 3;
    parent = 0;
  }
  MyMarker(double _x, double _y, double _z) {
    x = _x;
    y = _y;
    z = _z;
    radius = 0.0;
    type = 3;
    parent = 0;
  }
  MyMarker(double _x, double _y, double _z, double _radius) {
    x = _x;
    y = _y;
    z = _z;
    radius = _radius;
    type = 3;
    parent = 0;
  }
  MyMarker(const MyMarker &v) {
    x = v.x;
    y = v.y;
    z = v.z;
    radius = v.radius;
    type = v.type;
    parent = v.parent;
    nbr = v.nbr;
  }

  double &operator[](const int i) {
    assert(i >= 0 && i <= 2);
    return (i == 0) ? x : ((i == 1) ? y : z);
  }

  bool operator<(const MyMarker &other) const {
    if (z > other.z)
      return false;
    if (z < other.z)
      return true;
    if (y > other.y)
      return false;
    if (y < other.y)
      return true;
    if (x > other.x)
      return false;
    if (x < other.x)
      return true;
    return false;
  }
  bool operator==(const MyMarker &other) const {
    return (z == other.z && y == other.y && x == other.x);
  }
  bool operator!=(const MyMarker &other) const {
    return (z != other.z || y != other.y || x != other.x);
  }

  long long ind(long long sz0, long long sz01) const {
    return ((long long)(z + 0.5) * sz01 + (long long)(y + 0.5) * sz0 +
            (long long)(x + 0.5));
  }

  VID_t vid(VID_t xdim, VID_t ydim) const {
    return static_cast<VID_t>(x) + static_cast<VID_t>(y) * xdim +
           static_cast<VID_t>(z) * xdim * ydim;
  }

  friend std::ostream &operator<<(std::ostream &os, const MyMarker &m) {
    os << std::to_string(static_cast<int>(m.x)) + ", " +
              std::to_string(static_cast<int>(m.y)) + ", " +
              std::to_string(static_cast<int>(m.z)) +
              " radius: " + std::to_string(m.radius);
    return os;
  }

  // usage: std::cout << marker->description << '\n';
  std::string description(VID_t xdim, VID_t ydim) const {
    return std::to_string(static_cast<int>(x)) + ", " +
           std::to_string(static_cast<int>(y)) + ", " +
           std::to_string(static_cast<int>(z)) +
           " index: " + std::to_string(this->vid(xdim, ydim));
  }

  // offsets is in zyx order
  template <typename T> void operator-=(const vector<T> offsets) {
    assert(offsets.size() == 3);
    z -= (double)offsets[0];
    y -= (double)offsets[1];
    x -= (double)offsets[2];
  }
};

struct MyMarkerX : public MyMarker {
  double feature;
  int seg_id;
  int seg_level;
  MyMarkerX() : MyMarker() {
    seg_id = -1;
    seg_level = -1;
    feature = 0.0;
  }
  MyMarkerX(MyMarker &_marker) {
    x = _marker.x;
    y = _marker.y;
    z = _marker.z;
    type = _marker.type;
    radius = _marker.radius;
    seg_id = -1;
    seg_level = -1;
    feature = 0.0;
  }
  MyMarkerX(double _x, double _y, double _z) : MyMarker(_x, _y, _z) {
    seg_id = -1;
    seg_level = -1;
    feature = 0.0;
  }
};
typedef MyMarker MyNode;

#define MidMarker(m1, m2)                                                      \
  MyMarker(((m1).x + (m2).x) / 2.0, ((m1).y + (m2).y) / 2.0,                   \
           ((m1).z + (m2).z) / 2.0)

vector<MyMarker> readMarker_file(filesystem::path marker_file, int marker_base);
bool saveMarker_file(string marker_file, vector<MyMarker> &out_markers);
bool saveMarker_file(string marker_file, vector<MyMarker> &outmarkers,
                     list<string> &infostring);
bool saveMarker_file(string marker_file, vector<MyMarker *> &out_markers);
bool saveMarker_file(string marker_file, vector<MyMarker *> &out_markers,
                     list<string> &infostring);

double dist(MyMarker a, MyMarker b);

vector<MyMarker *> getLeaf_markers(vector<MyMarker *> &inmarkers);
vector<MyMarker *> getLeaf_markers(vector<MyMarker *> &inmarkers,
                                   map<MyMarker *, int> &childs_num);
