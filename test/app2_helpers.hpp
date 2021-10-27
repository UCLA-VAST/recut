// These functions are from the APP2 codebase (Hanchuan Peng, Vaa3D, Allen
// Institute) to serve as baseline sequential comparisons where appropriate

#include "utils.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <stdlib.h> // abs
#include <vector>
//#include "smooth_curve.h"
//#include "marker_radius.h"

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

using namespace std;

#define INTENSITY_DISTANCE_METHOD 0
#define __USE_APP_METHOD__

#define INF 1E20 // 3e+38             // float INF

#define GI(ind) givals[(int)((inimg1d[ind] - min_int) / max_int * 255)]

static double givals[256] = {
    22026.5, 20368,   18840.3, 17432.5, 16134.8, 14938.4, 13834.9, 12816.8,
    11877.4, 11010.2, 10209.4, 9469.8,  8786.47, 8154.96, 7571.17, 7031.33,
    6531.99, 6069.98, 5642.39, 5246.52, 4879.94, 4540.36, 4225.71, 3934.08,
    3663.7,  3412.95, 3180.34, 2964.5,  2764.16, 2578.14, 2405.39, 2244.9,
    2095.77, 1957.14, 1828.24, 1708.36, 1596.83, 1493.05, 1396.43, 1306.47,
    1222.68, 1144.62, 1071.87, 1004.06, 940.819, 881.837, 826.806, 775.448,
    727.504, 682.734, 640.916, 601.845, 565.329, 531.193, 499.271, 469.412,
    441.474, 415.327, 390.848, 367.926, 346.454, 326.336, 307.481, 289.804,
    273.227, 257.678, 243.089, 229.396, 216.541, 204.469, 193.129, 182.475,
    172.461, 163.047, 154.195, 145.868, 138.033, 130.659, 123.717, 117.179,
    111.022, 105.22,  99.7524, 94.5979, 89.7372, 85.1526, 80.827,  76.7447,
    72.891,  69.2522, 65.8152, 62.5681, 59.4994, 56.5987, 53.856,  51.2619,
    48.8078, 46.4854, 44.2872, 42.2059, 40.2348, 38.3676, 36.5982, 34.9212,
    33.3313, 31.8236, 30.3934, 29.0364, 27.7485, 26.526,  25.365,  24.2624,
    23.2148, 22.2193, 21.273,  20.3733, 19.5176, 18.7037, 17.9292, 17.192,
    16.4902, 15.822,  15.1855, 14.579,  14.0011, 13.4503, 12.9251, 12.4242,
    11.9464, 11.4905, 11.0554, 10.6401, 10.2435, 9.86473, 9.50289, 9.15713,
    8.82667, 8.51075, 8.20867, 7.91974, 7.64333, 7.37884, 7.12569, 6.88334,
    6.65128, 6.42902, 6.2161,  6.01209, 5.81655, 5.62911, 5.44938, 5.27701,
    5.11167, 4.95303, 4.80079, 4.65467, 4.51437, 4.37966, 4.25027, 4.12597,
    4.00654, 3.89176, 3.78144, 3.67537, 3.57337, 3.47528, 3.38092, 3.29013,
    3.20276, 3.11868, 3.03773, 2.9598,  2.88475, 2.81247, 2.74285, 2.67577,
    2.61113, 2.54884, 2.48881, 2.43093, 2.37513, 2.32132, 2.26944, 2.21939,
    2.17111, 2.12454, 2.07961, 2.03625, 1.99441, 1.95403, 1.91506, 1.87744,
    1.84113, 1.80608, 1.77223, 1.73956, 1.70802, 1.67756, 1.64815, 1.61976,
    1.59234, 1.56587, 1.54032, 1.51564, 1.49182, 1.46883, 1.44664, 1.42522,
    1.40455, 1.3846,  1.36536, 1.3468,  1.3289,  1.31164, 1.29501, 1.27898,
    1.26353, 1.24866, 1.23434, 1.22056, 1.2073,  1.19456, 1.18231, 1.17055,
    1.15927, 1.14844, 1.13807, 1.12814, 1.11864, 1.10956, 1.10089, 1.09262,
    1.08475, 1.07727, 1.07017, 1.06345, 1.05709, 1.05109, 1.04545, 1.04015,
    1.03521, 1.0306,  1.02633, 1.02239, 1.01878, 1.0155,  1.01253, 1.00989,
    1.00756, 1.00555, 1.00385, 1.00246, 1.00139, 1.00062, 1.00015, 1};

struct HeapElem {
  int heap_id;
  long img_ind;
  double value;
  HeapElem(long _ind, double _value) {
    heap_id = -1;
    img_ind = _ind;
    value = _value;
  }
};

struct HeapElemX : public HeapElem {
  long prev_ind; // previous img ind
  HeapElemX(long _ind, double _value) : HeapElem(_ind, _value) {
    prev_ind = -1;
  }
};

template <class T>
class BasicHeap // Basic Min heap
{
public:
  BasicHeap() { elems.reserve(10000); }
  T *delete_min() {
    if (elems.empty())
      return 0;
    T *min_elem = elems[0];

    if (elems.size() == 1)
      elems.clear();
    else {
      elems[0] = elems[elems.size() - 1];
      elems[0]->heap_id = 0;
      elems.erase(elems.begin() + elems.size() - 1);
      down_heap(0);
    }
    return min_elem;
  }
  void insert(T *t) {
    elems.push_back(t);
    t->heap_id = elems.size() - 1;
    up_heap(t->heap_id);
  }
  bool empty() { return elems.empty(); }
  void adjust(int id, double new_value) {
    double old_value = elems[id]->value;
    elems[id]->value = new_value;
    if (new_value < old_value)
      up_heap(id);
    else if (new_value > old_value)
      down_heap(id);
  }
  int size() { return elems.size(); }

private:
  vector<T *> elems;
  bool swap_heap(int id1, int id2) {
    if (id1 < 0 || id1 >= elems.size() || id2 < 0 || id2 >= elems.size())
      return false;
    if (id1 == id2)
      return false;
    int pid = id1 < id2 ? id1 : id2;
    int cid = id1 > id2 ? id1 : id2;
    assert(cid == 2 * (pid + 1) - 1 || cid == 2 * (pid + 1));

    if (elems[pid]->value <= elems[cid]->value)
      return false;
    else {
      T *tmp = elems[pid];
      elems[pid] = elems[cid];
      elems[cid] = tmp;
      elems[pid]->heap_id = pid;
      elems[cid]->heap_id = cid;
      return true;
    }
  }
  void up_heap(int id) {
    int pid = (id + 1) / 2 - 1;
    if (swap_heap(id, pid))
      up_heap(pid);
  }
  void down_heap(int id) {
    int cid1 = 2 * (id + 1) - 1;
    int cid2 = 2 * (id + 1);
    if (cid1 >= elems.size())
      return;
    else if (cid1 == elems.size() - 1) {
      swap_heap(id, cid1);
    } else if (cid1 < elems.size() - 1) {
      int cid = elems[cid1]->value < elems[cid2]->value ? cid1 : cid2;
      if (swap_heap(id, cid))
        down_heap(cid);
    }
  }
};

template <class T1, class T2> // T1 is the type of index, T2 the type of array
class Heap {
public:
  Heap(T2 *&array) {
    elems.reserve(10000);
    vals = array;
  }
  T1 delete_min() {
    if (elems.empty())
      return 0;
    T1 min_elem = elems[0];

    if (elems.size() == 1)
      elems.clear();
    else {
      elems[0] = elems[elems.size() - 1];
      elems.erase(elems.begin() + elems.size() - 1);
      down_heap(0);
    }
    return min_elem;
  }
  void insert(T1 t) {
    int heap_id = elems.size();
    elems.push_back(t);
    up_heap(heap_id);
  }
  bool empty() { return elems.empty(); }

private:
  vector<T1> elems;
  T2 *vals;
  bool swap_heap(int id1, int id2) // swap id1 and id2 if elems[id1] and
                                   // elems[id2] doesn't fit heap model
  {
    if (id1 < 0 || id1 >= elems.size() || id2 < 0 || id2 >= elems.size())
      return false;
    if (id1 == id2)
      return false;
    int pid = id1 < id2 ? id1 : id2;
    int cid = id1 > id2 ? id1 : id2;
    assert(cid == 2 * (pid + 1) - 1 || cid == 2 * (pid + 1));

    if (less(elems[pid], elems[cid]))
      return false;
    else if (elems[pid] == elems[cid])
      return false;
    else {
      T1 tmp = elems[pid];
      elems[pid] = elems[cid];
      elems[cid] = tmp;
      return true;
    }
  }
  void up_heap(int id) {
    int pid = (id + 1) / 2 - 1;
    if (swap_heap(id, pid))
      up_heap(pid);
  }
  bool less(T1 &v1, T1 &v2) {
    if (!vals)
      return false;
    if (vals[v1] < vals[v2])
      return true;
    else if (vals[v1] > vals[v2])
      return false;
    else if (v1 < v2)
      return true;
    else
      return false;
  }
  void down_heap(int id) {
    int cid1 = 2 * (id + 1) - 1;
    int cid2 = 2 * (id + 1);
    if (cid1 >= elems.size())
      return;
    else if (cid1 == elems.size() - 1) {
      swap_heap(id, cid1);
    } else if (cid1 < elems.size() - 1) {
      int cid = less(elems[cid1], elems[cid2]) ? cid1 : cid2;
      if (swap_heap(id, cid))
        down_heap(cid);
    }
  }
};

/*
 * test function based off APP2 by hanchuan peng to test
 * fastmarching algorithm
 */
/******************************************************************************
 * Fast marching based tree construction
 * 1. use graph augmented distance (GD)
 * 2. stop when all target marker are marched
 *
 * Input : root      root marker
 *         target    the set of target markers
 *         inimg1d   original input image
 *
 * Output : outtree   output tracing result
 *
 * Notice :
 * 1. the input pixel number should not be larger than 2G if sizeof(long) == 4
 * 2. target markers should not contain root marker
 * 3. the root marker in outswc, is point to itself
 * 4. the cnn_type is default 3
 * *****************************************************************************/
template <class T>
bool fastmarching_tree(std::vector<MyMarker *> roots, vector<MyMarker> &target,
                       const T *inimg1d, vector<MyMarker *> &outtree, long sz0,
                       long sz1, long sz2, int cnn_type, double bkg_thresh,
                       double max_int = 0., double min_int = INF) {
  enum { ALIVE = -1, TRIAL = 0, FAR = 1 };

  long tol_sz = sz0 * sz1 * sz2;
  long sz01 = sz0 * sz1;
  // int cnn_type = 3;  // ?

  // float * phi = new float[tol_sz]; for(long i = 0; i < tol_sz; i++){phi[i] =
  // INF;} long * parent = new long[tol_sz]; for(long i = 0; i < tol_sz; i++)
  // parent[i] = i;  // each pixel point to itself at the beginning

  long i;
  float *phi = 0;
  long *parent = 0;
  char *state = 0;
  try {
    phi = new float[tol_sz];
    parent = new long[tol_sz];
    state = new char[tol_sz];
    for (i = 0; i < tol_sz; i++) {
      phi[i] = INF;
      parent[i] =
          i; // each pixel point to itself at the         statements beginning
      state[i] = FAR;
    }
  } catch (...) {
    cout << "********* Fail to allocate memory. quit fastmarching_tree()."
         << endl;
    if (phi) {
      delete[] phi;
      phi = 0;
    }
    if (parent) {
      delete[] parent;
      parent = 0;
    }
    if (state) {
      delete[] state;
      state = 0;
    }
    return false;
  }

  // GI parameter min_int, max_int, li
  // if they are both at default then recalculate
  // be careful about whether this is invoked since it causes
  // considerable benchmark overhead
  if (max_int == 0. && min_int == INF) {
    for (long i = 0; i < tol_sz; i++) {
      if (inimg1d[i] > max_int)
        max_int = inimg1d[i];
      if (inimg1d[i] < min_int)
        min_int = inimg1d[i];
    }
    cout << "max_int: " << max_int << '\n';
    cout << "min_int: " << min_int << '\n';
  }

  assertm(max_int != 0, "max_int can't be zero");
  assertm(min_int != INF, "min_int can't be zero");
  // assertm(max_int != min_int, "min_int can't be equal to max_int");
  // max_int -= min_int;

  double li = 10;
  // initialization
  // char * state = new char[tol_sz];
  // for(long i = 0; i < tol_sz; i++) state[i] = FAR;

  vector<long> target_inds;
  for (long t = 0; t < target.size(); t++) {
    int i = target[t].x;
    int j = target[t].y;
    int k = target[t].z;
    long ind = k * sz01 + j * sz0 + i;
    target_inds.push_back(ind);
  }

  BasicHeap<HeapElemX> heap;
  map<long, HeapElemX *> elems;

  // init heap
  std::vector<VID_t> root_vids;
  for (const auto &root : roots) {
    // init state and phi for root
    int rootx = root->x;
    int rooty = root->y;
    int rootz = root->z;

    long root_ind = rootz * sz01 + rooty * sz0 + rootx;
    root_vids.push_back(root_ind);
    state[root_ind] = ALIVE;
    phi[root_ind] = 0.0;

    long index = root_ind;
    HeapElemX *elem = new HeapElemX(index, phi[index]);
    elem->prev_ind = index;
    heap.insert(elem);
    elems[index] = elem;
  }

  // loop
  int time_counter = 1;
  double process1 = 0;
  while (!heap.empty()) {
    double process2 = (time_counter++) * 100000.0 / tol_sz;
    if (process2 - process1 >= 1) {
      // cout << "\r" << ((int)process2) / 1000.0 << "%";
      // cout.flush();
      process1 = process2;
      if (!(target.empty())) {
        bool is_break = true;
        for (int t = 0; t < target_inds.size(); t++) {
          long tind = target_inds[t];
          if (parent[tind] == tind &&
              (std::count(root_vids.begin(), root_vids.end(), tind) == 0)) {
            is_break = false;
            break;
          }
        }
        if (is_break) {
          break;
        }
      }
    }

    HeapElemX *min_elem = heap.delete_min();
    elems.erase(min_elem->img_ind);

    long min_ind = min_elem->img_ind;
    long prev_ind = min_elem->prev_ind;
    delete min_elem;

    parent[min_ind] = prev_ind;

    state[min_ind] = ALIVE;
    int i = min_ind % sz0;
    int j = (min_ind / sz0) % sz1;
    int k = (min_ind / sz01) % sz2;

    int w, h, d;
    for (int kk = -1; kk <= 1; kk++) {
      d = k + kk;
      if (d < 0 || d >= sz2)
        continue;
      for (int jj = -1; jj <= 1; jj++) {
        h = j + jj;
        if (h < 0 || h >= sz1)
          continue;
        for (int ii = -1; ii <= 1; ii++) {
          w = i + ii;
          if (w < 0 || w >= sz0)
            continue;
          int offset = abs(ii) + abs(jj) + abs(kk);
          if (offset == 0 || offset > cnn_type)
            continue;
          double factor =
              (offset == 1)
                  ? 1.0
                  : ((offset == 2) ? 1.414214
                                   : ((offset == 3) ? 1.732051 : 0.0));
          //cout << w << ' ' << h << ' ' << d << '\n';
          long index = d * sz01 + h * sz0 + w;
          //cout << index << '\n';
          // discard background pixels
          if (inimg1d[index] <= bkg_thresh) {
            continue;
          }

          if (state[index] != ALIVE) {
            //assertm(min_ind < tol_sz, "min_ind can not exceed total size");
            //assertm(index < tol_sz, "index can not exceed total size");
            double new_dist =
                phi[min_ind] + (GI(index) + GI(min_ind)) * factor * 0.5;
            long prev_ind = min_ind;

            if (state[index] == FAR) {
              phi[index] = new_dist;
              HeapElemX *elem = new HeapElemX(index, phi[index]);
              elem->prev_ind = prev_ind;
              heap.insert(elem);
              elems[index] = elem;
              state[index] = TRIAL;
            } else if (state[index] == TRIAL) {
              if (phi[index] > new_dist) {
                phi[index] = new_dist;
                HeapElemX *elem = elems[index];
                heap.adjust(elem->heap_id, phi[index]);
                elem->prev_ind = prev_ind;
              }
            }
          }
        }
      }
    }
  }
  // save current swc tree
  if (1) {
    int i = -1, j = -1, k = -1;
    map<long, MyMarker *> tmp_map;
    for (long ind = 0; ind < tol_sz; ind++) {
      i++;
      if (i % sz0 == 0) {
        i = 0;
        j++;
        if (j % sz1 == 0) {
          j = 0;
          k++;
        }
      }
      if (state[ind] != ALIVE)
        continue;
      MyMarker *marker = new MyMarker(i, j, k);
      tmp_map[ind] = marker;
      outtree.push_back(marker);
    }
    i = -1;
    j = -1;
    k = -1;
    for (long ind = 0; ind < tol_sz; ind++) {
      i++;
      if (i % sz0 == 0) {
        i = 0;
        j++;
        if (j % sz1 == 0) {
          j = 0;
          k++;
        }
      }
      if (state[ind] != ALIVE)
        continue;
      long ind2 = parent[ind];
      MyMarker *marker1 = tmp_map[ind];
      MyMarker *marker2 = tmp_map[ind2];
      if (marker1 == marker2)
        marker1->parent = 0;
      else
        marker1->parent = marker2;
      // tmp_map[ind]->parent = tmp_map[ind2];
    }
  }
  // over

  map<long, HeapElemX *>::iterator mit = elems.begin();
  while (mit != elems.end()) {
    HeapElemX *elem = mit->second;
    delete elem;
    mit++;
  }

  if (phi) {
    delete[] phi;
    phi = 0;
  }
  if (parent) {
    delete[] parent;
    parent = 0;
  }
  if (state) {
    delete[] state;
    state = 0;
  }
  return true;
}

/*
 * test function based off APP2 by hanchuan peng to test accuracy of
 * fastmarching based calculate radius method
 */
template <class T>
uint16_t get_radius_accurate(const T *inimg1d, int grid_size, VID_t current_vid,
                             T thresh) {
  std::vector<int> sz = {grid_size, grid_size, grid_size};
  auto coord = id_to_coord(current_vid, sz);

  int max_r = grid_size / 2 - 1;
  int r;
  double tol_num, bak_num;
  int mx = coord[0] + 0.5;
  int my = coord[1] + 0.5;
  int mz = coord[2] + 0.5;
  // cout<<"mx = "<<mx<<" my = "<<my<<" mz = "<<mz<<'\n';
  int64_t x[2], y[2], z[2];

  tol_num = bak_num = 0.0;
  int64_t sz01 = sz[0] * sz[1];
  for (r = 1; r <= max_r; r++) {
    double r1 = r - 0.5;
    double r2 = r + 0.5;
    double r1_r1 = r1 * r1;
    double r2_r2 = r2 * r2;
    double z_min = 0, z_max = r2;
    for (int dz = z_min; dz < z_max; dz++) {
      double dz_dz = dz * dz;
      double y_min = 0;
      double y_max = sqrt(r2_r2 - dz_dz);
      for (int dy = y_min; dy < y_max; dy++) {
        double dy_dy = dy * dy;
        double x_min = r1_r1 - dz_dz - dy_dy;
        x_min = x_min > 0 ? sqrt(x_min) + 1 : 0;
        double x_max = sqrt(r2_r2 - dz_dz - dy_dy);
        for (int dx = x_min; dx < x_max; dx++) {
          x[0] = mx - dx, x[1] = mx + dx;
          y[0] = my - dy, y[1] = my + dy;
          z[0] = mz - dz, z[1] = mz + dz;
          for (char b = 0; b < 8; b++) {
            char ii = b & 0x01, jj = (b >> 1) & 0x01, kk = (b >> 2) & 0x01;
            if (x[ii] < 0 || x[ii] >= sz[0] || y[jj] < 0 || y[jj] >= sz[1] ||
                z[kk] < 0 || z[kk] >= sz[2])
              return r;
            else {
              tol_num++;
              long pos = z[kk] * sz01 + y[jj] * sz[0] + x[ii];
              if (inimg1d[pos] <= thresh) {
                bak_num++;
              }
              if ((bak_num / tol_num) > 0.0001)
                return r;
            }
          }
        }
      }
    }
  }
  return r;
}

/*
 * test function based off APP2 by hanchuan peng to test accuracy of
 * fastmarching based calculate radius method
 */
template <typename T>
uint16_t get_radius_hanchuan_XY(const T *inimg1d, GridCoord image_lengths,
    VID_t vid, T thresh) {
  auto coord = id_to_coord(vid, image_lengths);

  long sz01 = image_lengths[0] * image_lengths[1];
  double max_r = std::min(image_lengths[0], image_lengths[1]) / 2 - 1;

  double total_num, background_num;
  double ir;
  for (ir = 1; ir <= max_r; ir++) {
    total_num = background_num = 0;

    double dz, dy, dx;
    double zlower = 0, zupper = 0;
    for (dz = zlower; dz <= zupper; ++dz)
      for (dy = -ir; dy <= +ir; ++dy)
        for (dx = -ir; dx <= +ir; ++dx) {
          total_num++;

          double r = sqrt(dx * dx + dy * dy + dz * dz);
          if (r > ir - 1 && r <= ir) {
            int64_t i = coord[0] + dx;
            if (i < 0 || i >= image_lengths[0])
              goto end1;
            int64_t j = coord[1] + dy;
            if (j < 0 || j >= image_lengths[1])
              goto end1;
            int64_t k = coord[2] + dz;
            if (k < 0 || k >= image_lengths[2])
              goto end1;

            if (inimg1d[k * sz01 + j * image_lengths[0] + i] <= thresh) {
              background_num++;

              if ((background_num / total_num) > 0.001)
                goto end1; // change 0.01 to 0.001 on 100104
            }
          }
        }
  }
end1:
  return ir;
}

struct HierarchySegment {
  HierarchySegment *parent;
  MyMarker *leaf_marker;
  MyMarker
      *root_marker; // its parent marker is in current segment's parent segment
  double length;    // the length from leaf to root
  int level;        // the segments number from leaf to root

  HierarchySegment() {
    leaf_marker = 0;
    root_marker = 0;
    length = 0;
    level = 1;
    parent = 0;
  }
  HierarchySegment(MyMarker *_leaf, MyMarker *_root, double _len, int _level) {
    leaf_marker = _leaf;
    root_marker = _root;
    length = _len;
    level = _level;
    parent = 0;
  }

  void get_markers(vector<MyMarker *> &outswc) {
    if (!leaf_marker || !root_marker)
      return;
    MyMarker *p = leaf_marker;
    while (p != root_marker) {
      outswc.push_back(p);
      p = p->parent;
    }
    outswc.push_back(root_marker);
  }
};

// There is no overlap between HierarchySegment
template <class T>
bool swc2topo_segs(vector<MyMarker *> &inswc,
                   vector<HierarchySegment *> &topo_segs,
                   int length_method = INTENSITY_DISTANCE_METHOD,
                   T *inimg1d = 0, const long sz0 = 0, const long sz1 = 0,
                   const long sz2 = 0) {
  if (length_method == INTENSITY_DISTANCE_METHOD &&
      (inimg1d == 0 || sz0 == 0 || sz1 == 0 || sz2 == 0)) {
    cerr << "need image input for INTENSITY_DISTANCE_METHOD " << endl;
    return false;
  }
  // 1. calc distance for every nodes
  int64_t tol_num = inswc.size();
  map<MyMarker *, int64_t> swc_map;
  for (int64_t i = 0; i < tol_num; i++)
    swc_map[inswc[i]] = i;

  vector<MyMarker *> leaf_markers;
  // GET_LEAF_MARKERS(leaf_markers, inswc);
  vector<int64_t> childs_num(tol_num);
  {
    for (int64_t i = 0; i < tol_num; i++)
      childs_num[i] = 0;
    for (int64_t m1 = 0; m1 < tol_num; m1++) {
      if (!inswc[m1]->parent)
        continue;
      int64_t m2 = swc_map[inswc[m1]->parent];
      childs_num[m2]++;
    }
    for (int64_t i = 0; i < tol_num; i++)
      if (childs_num[i] == 0)
        leaf_markers.push_back(inswc[i]);
  }
  int64_t leaf_num = leaf_markers.size();

  int64_t tol_sz = sz0 * sz1 * sz2;
  int64_t sz01 = sz0 * sz1;

  vector<double> topo_dists(tol_num,
                            0.0); // furthest leaf distance for each tree node
  vector<MyMarker *> topo_leafs(tol_num, (MyMarker *)0);

  for (int64_t i = 0; i < leaf_num; i++) {
    MyMarker *leaf_marker = leaf_markers[i];
    MyMarker *child_node = leaf_markers[i];
    MyMarker *parent_node = child_node->parent;
    int cid = swc_map[child_node];
    topo_leafs[cid] = leaf_marker;
    topo_dists[cid] = (length_method == INTENSITY_DISTANCE_METHOD)
                          ? inimg1d[leaf_marker->ind(sz0, sz01)] / 255.0
                          : 0;
    while (parent_node) {
      int64_t pid = swc_map[parent_node];
      double tmp_dst =
          (length_method == INTENSITY_DISTANCE_METHOD)
              ? (inimg1d[parent_node->ind(sz0, sz01)] / 255.0 + topo_dists[cid])
              : (dist(*child_node, *parent_node) + topo_dists[cid]);
      if (tmp_dst >= topo_dists[pid]) // >= instead of >
      {
        topo_dists[pid] = tmp_dst;
        topo_leafs[pid] = topo_leafs[cid];
      } else
        break;
      child_node = parent_node;
      cid = pid;
      parent_node = parent_node->parent;
    }
  }
  // 2. create Hierarchy Segments
  topo_segs.resize(leaf_num);
  map<MyMarker *, int> leaf_ind_map;
  for (int64_t i = 0; i < leaf_num; i++) {
    topo_segs[i] = new HierarchySegment();
    leaf_ind_map[leaf_markers[i]] = i;
  }

  for (int64_t i = 0; i < leaf_num; i++) {
    MyMarker *leaf_marker = leaf_markers[i];
    MyMarker *root_marker = leaf_marker;
    MyMarker *root_parent = root_marker->parent;
    int level = 1;
    while (root_parent && topo_leafs[swc_map[root_parent]] == leaf_marker) {
      if (childs_num[swc_map[root_marker]] >= 2)
        level++;
      root_marker = root_parent;
      root_parent = root_marker->parent;
    }

    double dst = topo_dists[swc_map[root_marker]];

    HierarchySegment *topo_seg = topo_segs[i];
    *topo_seg = HierarchySegment(leaf_marker, root_marker, dst, level);

    if (root_parent == 0)
      topo_seg->parent = 0;
    else {
      MyMarker *leaf_marker2 = topo_leafs[swc_map[root_parent]];
      int leaf_ind2 = leaf_ind_map[leaf_marker2];
      topo_seg->parent = topo_segs[leaf_ind2];
    }
  }

  swc_map.clear();
  leaf_markers.clear();
  leaf_ind_map.clear();
  topo_dists.clear();
  topo_leafs.clear();
  return true;
}

// 1. will change the type of each segment
// swc_type : 0 for length heatmap, 1 for level heatmap
template <class T>
bool topo_segs2swc(vector<HierarchySegment *> &topo_segs,
                   vector<MyMarker *> &outmarkers, T swc_type = 1) {
  if (topo_segs.empty())
    return false;

  double min_dst = topo_segs[0]->length, max_dst = min_dst;
  double min_level = topo_segs[0]->level, max_level = min_level;
  for (int64_t i = 0; i < topo_segs.size(); i++) {
    double dst = topo_segs[i]->length;
    min_dst = MIN(min_dst, dst);
    max_dst = MAX(max_dst, dst);

    int level = topo_segs[i]->level;
    min_level = MIN(min_level, level);
    max_level = MAX(max_level, level);
  }
  max_level = MIN(max_level, 20); // todo1

  cout << "min_dst = " << min_dst << endl;
  cout << "max_dst = " << max_dst << endl;
  cout << "min_level = " << min_level << endl;
  cout << "max_level = " << max_level << endl;

  max_dst -= min_dst;
  if (max_dst == 0.0)
    max_dst = 0.0000001;
  max_level -= min_level;
  if (max_level == 0)
    max_level = 1.0;
  for (int64_t i = 0; i < topo_segs.size(); i++) {
    double dst = topo_segs[i]->length;
    int level = MIN(topo_segs[i]->level, max_level); // todo1
    int color_id = (swc_type == 0)
                       ? (dst - min_dst) / max_dst * 254.0 + 20.5
                       : (level - min_level) / max_level * 254.0 + 20.5;
    vector<MyMarker *> tmp_markers;
    topo_segs[i]->get_markers(tmp_markers);
    for (int j = 0; j < tmp_markers.size(); j++) {
      tmp_markers[j]->type = color_id;
    }
    outmarkers.insert(outmarkers.end(), tmp_markers.begin(), tmp_markers.end());
  }
  return true;
}

template <class T>
bool hierarchy_prune(vector<MyMarker *> &inswc, vector<MyMarker *> &outswc,
                     T *inimg1d, long sz0, long sz1, long sz2,
                     double length_thresh = 10.0) {
  vector<HierarchySegment *> topo_segs;
  swc2topo_segs(inswc, topo_segs, INTENSITY_DISTANCE_METHOD, inimg1d, sz0, sz1,
                sz2);
  vector<HierarchySegment *> filter_segs;
  //    if(length_thresh <= 0.0)
  //    {
  //        vector<short int> values;
  //        for(int i = 0; i < topo_segs.size(); i++)
  //        {
  //            values.push_back(topo_segs[i]->length * 1000 + 0.5);
  //        }
  //        cout<<"segment num = "<<values.size()<<endl;
  //        length_thresh = otsu_threshold(values) / 1000.0;
  //        cout<<"otsu length = "<<length_thresh<<endl;
  //    }
  for (int64_t i = 0; i < topo_segs.size(); i++) {
    if (topo_segs[i]->length >= length_thresh)
      filter_segs.push_back(topo_segs[i]);
    // if(topo_segs[i]->length * topo_segs[i]->level >= length_thresh)
    // filter_segs.push_back(topo_segs[i]);
  }
  topo_segs2swc(filter_segs, outswc, 0);
  return true;
}

// hierarchy coverage pruning
template <class T>
bool happ(vector<MyMarker *> &inswc, vector<MyMarker *> &outswc, T *inimg1d,
          const long sz0, const long sz1, const long sz2, T bkg_thresh,
          double length_thresh = 2.0, double SR_ratio = 1.0 / 9.0,
          bool is_leaf_prune = true, bool is_smooth = true) {
  double T_max = (1ll << sizeof(T));

  const int64_t sz01 = sz0 * sz1;
  const int64_t tol_sz = sz01 * sz2;

  map<MyMarker *, int> child_num;
  getLeaf_markers(inswc, child_num);

  vector<HierarchySegment *> topo_segs;
  cout << "Construct hierarchical segments" << endl;
  swc2topo_segs(inswc, topo_segs, INTENSITY_DISTANCE_METHOD, inimg1d, sz0, sz1,
                sz2);
  vector<HierarchySegment *> filter_segs;
  for (int64_t i = 0; i < topo_segs.size(); i++) {
    if (topo_segs[i]->length >= length_thresh)
      filter_segs.push_back(topo_segs[i]);
  }
  cout << "pruned by length_thresh (segment number) : " << topo_segs.size()
       << " - " << topo_segs.size() - filter_segs.size() << " = "
       << filter_segs.size() << endl;
  multimap<double, HierarchySegment *> seg_dist_map;
  for (int64_t i = 0; i < filter_segs.size(); i++) {
    double dst = filter_segs[i]->length;
    seg_dist_map.insert(pair<double, HierarchySegment *>(dst, filter_segs[i]));
  }

  if (1) // dark nodes pruning
  {
    int dark_num_pruned = 1;
    int iteration = 1;
    vector<bool> is_pruneable(filter_segs.size(), 0);
    cout << "===== Perform dark node pruning =====" << endl;
    while (dark_num_pruned > 0) {
      dark_num_pruned = 0;
      for (int64_t i = 0; i < filter_segs.size(); i++) {
        if (iteration > 1 && !is_pruneable[i])
          continue;
        HierarchySegment *seg = filter_segs[i];
        MyMarker *leaf_marker = seg->leaf_marker;
        MyMarker *root_marker = seg->root_marker;
        if (leaf_marker == root_marker)
          continue;
        if (inimg1d[leaf_marker->ind(sz0, sz01)] <= bkg_thresh) {
          seg->leaf_marker = leaf_marker->parent;
          dark_num_pruned++;
          is_pruneable[i] = true;
        } else
          is_pruneable[i] = false;
      }
      cout << "\t iteration [" << iteration++ << "] " << dark_num_pruned
           << " dark node pruned" << endl;
    }
  }

  if (1) // dark segment pruning
  {
    set<int> delete_index_set;
    for (int64_t i = 0; i < filter_segs.size(); i++) {
      HierarchySegment *seg = filter_segs[i];
      MyMarker *leaf_marker = seg->leaf_marker;
      MyMarker *root_marker = seg->root_marker;
      if (length_thresh != 0) {
        // length_thresh of 0 indicates not to prune
        // based on lengths
        if (leaf_marker == root_marker) {
          // this condition prunes segments of size 1
          delete_index_set.insert(i);
          continue;
        }
      }
      MyMarker *p = leaf_marker;
      double sum_int = 0.0, tol_num = 0.0, dark_num = 0.0;
      while (true) {
        double intensity = inimg1d[p->ind(sz0, sz01)];
        sum_int += intensity;
        tol_num++;
        // std::cout << "x " << p->x << " y " << p->y << " z " << p->z << " ind
        // " << p->ind(sz0, sz01) << " intensity: " << intensity << '\n';
        if (intensity <= bkg_thresh) {
          // you can pass a higher bkg_thresh then that used during fastmarching
          dark_num++;
        }
        if (p == root_marker)
          break;
        p = p->parent;
      }
      if (sum_int / tol_num <= bkg_thresh || dark_num / tol_num >= 0.2) {
        // std::cout << "avg: " << sum_int / tol_num;
        // std::cout << "dark sum: " << dark_num / tol_num;
        delete_index_set.insert(i);
      }
    }
    vector<HierarchySegment *> tmp_segs;
    for (int i = 0; i < filter_segs.size(); i++) {
      HierarchySegment *seg = filter_segs[i];
      if (delete_index_set.find(i) == delete_index_set.end())
        tmp_segs.push_back(seg);
    }
    cout << "\t" << delete_index_set.size() << " dark segments are deleted"
         << endl;
    filter_segs = tmp_segs;
  }

  // calculate radius for every node
  {
    cout << "Calculating radius for every node" << endl;
    auto in_sz = new_grid_coord(sz0, sz1, sz2);
    for (int64_t i = 0; i < filter_segs.size(); i++) {
      HierarchySegment *seg = filter_segs[i];
      MyMarker *leaf_marker = seg->leaf_marker;
      MyMarker *root_marker = seg->root_marker;
      MyMarker *p = leaf_marker;
      while (true) {
        // assumes dim sizes are equal for now, can be easily switchd
        auto coord = new_grid_coord(p->x, p->y, p->z);
        auto vid = coord_to_id(coord, in_sz);
        p->radius = get_radius_hanchuan_XY(inimg1d, in_sz, vid, bkg_thresh);
        if (p == root_marker)
          break;
        p = p->parent;
      }
    }
  }

#ifdef __USE_APP_METHOD__
  if (1) // hierarchy coverage order pruning
#else
  if (1) // hierarchy coverage order pruning
#endif
  {
    cout << "Perform hierarchical pruning" << endl;
    T *tmpimg1d = new T[tol_sz];
    memcpy(tmpimg1d, inimg1d, tol_sz * sizeof(T));
    int64_t tmp_sz[4] = {sz0, sz1, sz2, 1};

    multimap<double, HierarchySegment *>::reverse_iterator it =
        seg_dist_map.rbegin();
    // MyMarker * soma = (*it).second->root_marker;  // 2012/07 Hang, no need to
    // consider soma cout<<"soma ("<<soma->x<<","<<soma->y<<","<<soma->z<<")
    // radius = "<<soma->radius<<" value = "<<(int)inimg1d[soma->ind(sz0,
    // sz01)]<<endl;
    filter_segs.clear();
    set<HierarchySegment *> visited_segs;
    double tol_sum_sig = 0.0, tol_sum_rdc = 0.0;
    while (it != seg_dist_map.rend()) {
      HierarchySegment *seg = it->second;
      if (seg->parent && visited_segs.find(seg->parent) == visited_segs.end()) {
        it++;
        continue;
      }

      MyMarker *leaf_marker = seg->leaf_marker;
      MyMarker *root_marker = seg->root_marker;
      double SR_RATIO = SR_ratio; // the soma area will use different SR_ratio
      // if(dist(*soma, *root_marker) <= soma->radius) SR_RATIO = 1.0;

      double sum_sig = 0;
      double sum_rdc = 0;

      MyMarker *p = leaf_marker;
      while (true) {
        if (tmpimg1d[p->ind(sz0, sz01)] == 0) {
          sum_rdc += inimg1d[p->ind(sz0, sz01)];
        } else {
          if (1) {
            int r = p->radius;
            int64_t x1 = p->x + 0.5;
            int64_t y1 = p->y + 0.5;
            int64_t z1 = p->z + 0.5;
            double sum_sphere_size = 0.0;
            double sum_delete_size = 0.0;
            for (int64_t kk = -r; kk <= r; kk++) {
              int64_t z2 = z1 + kk;
              if (z2 < 0 || z2 >= sz2)
                continue;
              for (int64_t jj = -r; jj <= r; jj++) {
                int64_t y2 = y1 + jj;
                if (y2 < 0 || y2 >= sz1)
                  continue;
                for (int64_t ii = -r; ii <= r; ii++) {
                  int64_t x2 = x1 + ii;
                  if (x2 < 0 || x2 >= sz0)
                    continue;
                  if (kk * kk + jj * jj + ii * ii > r * r)
                    continue;
                  int64_t ind2 = z2 * sz01 + y2 * sz0 + x2;
                  sum_sphere_size++;
                  if (tmpimg1d[ind2] != inimg1d[ind2]) {
                    sum_delete_size++;
                  }
                }
              }
            }
            // the intersection between two sphere with equal size and distance
            // = R is 5/16 (0.3125) sum_delete_size/sum_sphere_size should be <
            // 5/16 for outsize points
            if (sum_sphere_size > 0 &&
                sum_delete_size / sum_sphere_size > 0.1) {
              sum_rdc += inimg1d[p->ind(sz0, sz01)];
            } else
              sum_sig += inimg1d[p->ind(sz0, sz01)];
          }
        }
        if (p == root_marker)
          break;
        p = p->parent;
      }

      // double sum_sig = total_sum_int - sum_rdc;
      if (!seg->parent || sum_rdc == 0.0 ||
          (sum_sig / sum_rdc >= SR_RATIO && sum_sig >= 1.0 * T_max)) {
        tol_sum_sig += sum_sig;
        tol_sum_rdc += sum_rdc;

        vector<MyMarker *> seg_markers;
        MyMarker *p = leaf_marker;
        while (true) {
          if (tmpimg1d[p->ind(sz0, sz01)] != 0)
            seg_markers.push_back(p);
          if (p == root_marker)
            break;
          p = p->parent;
        }
        // reverse(seg_markers.begin(), seg_markers.end()); // need to reverse
        // if resampling

        for (int m = 0; m < seg_markers.size(); m++) {
          p = seg_markers[m];

          int r = p->radius;
          if (r > 0) // && tmpimg1d[p->ind(sz0, sz01)] != 0)
          {
            double rr = r * r;
            int64_t x = p->x + 0.5;
            int64_t y = p->y + 0.5;
            int64_t z = p->z + 0.5;
            for (int64_t kk = -r; kk <= r; kk++) {
              int64_t z2 = z + kk;
              if (z2 < 0 || z2 >= sz2)
                continue;
              for (int64_t jj = -r; jj <= r; jj++) {
                int64_t y2 = y + jj;
                if (y2 < 0 || y2 >= sz1)
                  continue;
                for (int64_t ii = -r; ii <= r; ii++) {
                  int64_t x2 = x + ii;
                  if (x2 < 0 || x2 >= sz0)
                    continue;
                  double dst = ii * ii + jj * jj + kk * kk;
                  if (dst > rr)
                    continue;
                  int64_t ind = z2 * sz01 + y2 * sz0 + x2;
                  tmpimg1d[ind] = 0;
                }
              }
            }
          }
        }

        filter_segs.push_back(seg);
        visited_segs.insert(
            seg); // used to delete children when parent node is delete
      }
      it++;
    }
    cout << "prune by coverage (segment number) : " << seg_dist_map.size()
         << " - " << filter_segs.size() << " = "
         << seg_dist_map.size() - filter_segs.size() << endl;
    cout << "R/S ratio = " << tol_sum_rdc / tol_sum_sig << " (" << tol_sum_rdc
         << "/" << tol_sum_sig << ")" << endl;
    if (1) // evaluation
    {
      double tree_sig = 0.0;
      for (int m = 0; m < inswc.size(); m++)
        tree_sig += inimg1d[inswc[m]->ind(sz0, sz01)];
      double covered_sig = 0.0;
      for (int ind = 0; ind < tol_sz; ind++)
        if (tmpimg1d[ind] == 0)
          covered_sig += inimg1d[ind];
      cout << "S/T ratio = " << covered_sig / tree_sig << " (" << covered_sig
           << "/" << tree_sig << ")" << endl;
    }
    // saveImage("test.tif", tmpimg1d, tmp_sz, V3D_UINT8);
    if (tmpimg1d) {
      delete[] tmpimg1d;
      tmpimg1d = 0;
    }
  }

  if (0) // resampling markers or internal node pruning //this part of code has
         // bug: many fragmentations. noted by PHC, 20120628
  {
    cout << "resampling markers" << endl;
    vector<MyMarker *> tmp_markers;
    topo_segs2swc(filter_segs, tmp_markers, 0); // no resampling
    child_num.clear();
    getLeaf_markers(tmp_markers, child_num);

    // calculate sampling markers
    for (int64_t i = 0; i < filter_segs.size(); i++) {
      HierarchySegment *seg = filter_segs[i];
      MyMarker *leaf_marker = seg->leaf_marker;
      MyMarker *root_marker = seg->root_marker;
      vector<MyMarker *> seg_markers;
      MyMarker *p = leaf_marker;
      while (true) {
        seg_markers.push_back(p);
        if (p == root_marker)
          break;
        p = p->parent;
      }
      // reverse(seg_markers.begin(), seg_markers.end()); // need to reverse if
      // resampling //commened by PHC, 130520 to build on Ubuntu. This should
      // make no difference as the outside code is if (0)
      vector<MyMarker *> sampling_markers; // store resampling markers
      p = root_marker;
      sampling_markers.push_back(p);
      for (int m = 0; m < seg_markers.size(); m++) {
        MyMarker *marker = seg_markers[m];
        if (child_num[marker] > 1 ||
            dist(*marker, *p) >= p->radius) // + marker->radius)
        {
          sampling_markers.push_back(marker);
          p = marker;
        }
      }
      if ((*sampling_markers.rbegin()) != leaf_marker)
        sampling_markers.push_back(leaf_marker);
      for (int m = 1; m < sampling_markers.size(); m++)
        sampling_markers[m]->parent = sampling_markers[m - 1];
    }
  }

#ifdef __USE_APP_METHOD__
  if (1) // is_leaf_prune)  // leaf nodes pruning
#else
  if (0)
#endif
  {
    cout << "Perform leaf node pruning" << endl;

    map<MyMarker *, int> tmp_child_num;
    if (1) // get child_num of each node
    {
      vector<MyMarker *> current_markers;
      for (int64_t i = 0; i < filter_segs.size(); i++) {
        HierarchySegment *seg = filter_segs[i];
        seg->get_markers(current_markers);
      }
      for (int m = 0; m < current_markers.size(); m++)
        tmp_child_num[current_markers[m]] = 0;
      for (int m = 0; m < current_markers.size(); m++) {
        MyMarker *par_marker = current_markers[m]->parent;
        if (par_marker)
          tmp_child_num[par_marker]++;
      }
    }
    int64_t leaf_num_pruned = 1;
    int iteration = 1;
    vector<bool> is_pruneable(filter_segs.size(), 0);
    while (leaf_num_pruned > 0) {
      leaf_num_pruned = 0;
      for (int64_t i = 0; i < filter_segs.size(); i++) {
        if (iteration > 1 && !is_pruneable[i])
          continue;
        HierarchySegment *seg = filter_segs[i];
        MyMarker *leaf_marker = seg->leaf_marker;
        MyMarker *root_marker = seg->root_marker;

        if (tmp_child_num[leaf_marker] >= 1)
          continue;

        assert(leaf_marker);
        MyMarker *par_marker = leaf_marker->parent;
        if (!par_marker) {
          is_pruneable[i] = 0;
          continue;
        }
        int r1 = leaf_marker->radius;
        int r2 = par_marker->radius;
        double r1_r1 = r1 * r1;
        double r2_r2 = r2 * r2;
        int64_t x1 = leaf_marker->x + 0.5;
        int64_t y1 = leaf_marker->y + 0.5;
        int64_t z1 = leaf_marker->z + 0.5;
        int64_t x2 = par_marker->x + 0.5;
        int64_t y2 = par_marker->y + 0.5;
        int64_t z2 = par_marker->z + 0.5;

        double sum_leaf_int = 0.0;
        double sum_over_int = 0.0;
        for (int64_t kk = -r1; kk <= r1; kk++) {
          int64_t zz = z1 + kk;
          if (zz < 0 || zz >= sz2)
            continue;
          for (int64_t jj = -r1; jj <= r1; jj++) {
            int64_t yy = y1 + jj;
            if (yy < 0 || yy >= sz1)
              continue;
            for (int64_t ii = -r1; ii <= r1; ii++) {
              int64_t xx = x1 + ii;
              if (xx < 0 || xx >= sz0)
                continue;
              double dst = kk * kk + jj * jj + ii * ii;
              if (dst > r1_r1)
                continue;
              int64_t ind = zz * sz01 + yy * sz0 + xx;
              sum_leaf_int += inimg1d[ind];
              if ((z2 - zz) * (z2 - zz) + (y2 - yy) * (y2 - yy) +
                      (x2 - xx) * (x2 - xx) <=
                  r2 * r2) {
                sum_over_int += inimg1d[ind];
              }
            }
          }
        }
        if (sum_leaf_int > 0 && sum_over_int / sum_leaf_int >= 0.9) {
          leaf_num_pruned++;
          tmp_child_num[par_marker]--;
          assert(tmp_child_num[leaf_marker] == 0);
          if (leaf_marker != root_marker) {
            seg->leaf_marker = par_marker;
            is_pruneable[i] = true;
          } else {
            seg->leaf_marker = NULL;
            seg->root_marker = NULL;
            is_pruneable[i] = false;
          }
        } else
          is_pruneable[i] = false;
      }
      cout << "\t iteration [" << iteration++ << "] " << leaf_num_pruned
           << " leaf node pruned" << endl;
    }
    // filter out segments with single marker
    vector<HierarchySegment *> tmp_segs;
    for (int64_t i = 0; i < filter_segs.size(); i++) {
      HierarchySegment *seg = filter_segs[i];
      MyMarker *leaf_marker = seg->leaf_marker;
      MyMarker *root_marker = seg->root_marker;
      if (leaf_marker && root_marker)
        tmp_segs.push_back(seg);
    }
    cout << "\t" << filter_segs.size() - tmp_segs.size()
         << " hierarchical segments are pruned in leaf node pruning" << endl;
    filter_segs.clear();
    filter_segs = tmp_segs;
  }

#ifdef __USE_APP_METHOD__
  if (1) // joint leaf node pruning
#else
  if (0) // joint leaf node pruning
#endif
  {
    cout << "Perform joint leaf node pruning" << endl;
    cout << "\tcompute mask area" << endl;
    unsigned short *mask = new unsigned short[tol_sz];
    memset(mask, 0, sizeof(unsigned short) * tol_sz);
    for (int64_t s = 0; s < filter_segs.size(); s++) {
      HierarchySegment *seg = filter_segs[s];
      MyMarker *leaf_marker = seg->leaf_marker;
      MyMarker *root_marker = seg->root_marker;
      MyMarker *p = leaf_marker;
      while (true) {
        int r = p->radius;
        if (r > 0) {
          double rr = r * r;
          int64_t x = p->x + 0.5;
          int64_t y = p->y + 0.5;
          int64_t z = p->z + 0.5;
          for (int64_t kk = -r; kk <= r; kk++) {
            int64_t z2 = z + kk;
            if (z2 < 0 || z2 >= sz2)
              continue;
            for (int64_t jj = -r; jj <= r; jj++) {
              int64_t y2 = y + jj;
              if (y2 < 0 || y2 >= sz1)
                continue;
              for (int64_t ii = -r; ii <= r; ii++) {
                int64_t x2 = x + ii;
                if (x2 < 0 || x2 >= sz0)
                  continue;
                double dst = ii * ii + jj * jj + kk * kk;
                if (dst > rr)
                  continue;
                int64_t ind = z2 * sz01 + y2 * sz0 + x2;
                mask[ind]++;
              }
            }
          }
        }
        if (p == root_marker)
          break;
        p = p->parent;
      }
    }
    cout << "\tget post_segs" << endl;
    vector<HierarchySegment *> post_segs;
    if (0) // get post order of filter_segs
    {
      multimap<double, HierarchySegment *> tmp_seg_map;
      for (int64_t s = 0; s < filter_segs.size(); s++) {
        double dst = filter_segs[s]->length;
        tmp_seg_map.insert(
            pair<double, HierarchySegment *>(dst, filter_segs[s]));
      }
      multimap<double, HierarchySegment *>::iterator it = tmp_seg_map.begin();
      while (it != tmp_seg_map.end()) {
        post_segs.push_back(it->second);
        it++;
      }
    } else
      post_segs = filter_segs; // random order

    map<MyMarker *, int> tmp_child_num;
    if (1) // get child_num of each node
    {
      vector<MyMarker *> current_markers;
      for (int64_t i = 0; i < filter_segs.size(); i++) {
        HierarchySegment *seg = filter_segs[i];
        seg->get_markers(current_markers);
      }
      for (int m = 0; m < current_markers.size(); m++)
        tmp_child_num[current_markers[m]] = 0;
      for (int m = 0; m < current_markers.size(); m++) {
        MyMarker *par_marker = current_markers[m]->parent;
        if (par_marker)
          tmp_child_num[par_marker]++;
      }
    }
    if (1) // start prune leaf nodes
    {
      cout << "\tleaf node pruning" << endl;
      int64_t leaf_num_pruned = 1;
      int iteration = 1;
      vector<bool> is_pruneable(post_segs.size(), 0);
      while (leaf_num_pruned > 0) {
        leaf_num_pruned = 0;
        for (int64_t i = 0; i < post_segs.size(); i++) {
          if (iteration > 1 && !is_pruneable[i])
            continue;
          HierarchySegment *seg = post_segs[i];
          MyMarker *leaf_marker = seg->leaf_marker;
          MyMarker *root_marker = seg->root_marker;
          int r = leaf_marker->radius;
          if (r <= 0) {
            is_pruneable[i] = 0;
            continue;
          }
          double rr = r * r;
          int64_t x = leaf_marker->x + 0.5;
          int64_t y = leaf_marker->y + 0.5;
          int64_t z = leaf_marker->z + 0.5;

          double covered_sig = 0;
          double total_sig = 0.0;
          for (int64_t kk = -r; kk <= r; kk++) {
            int64_t z2 = z + kk;
            if (z2 < 0 || z2 >= sz2)
              continue;
            for (int64_t jj = -r; jj <= r; jj++) {
              int64_t y2 = y + jj;
              if (y2 < 0 || y2 >= sz1)
                continue;
              for (int64_t ii = -r; ii <= r; ii++) {
                int64_t x2 = x + ii;
                if (x2 < 0 || x2 >= sz0)
                  continue;
                double dst = ii * ii + jj * jj + kk * kk;
                if (dst > rr)
                  continue;
                int64_t ind = z2 * sz01 + y2 * sz0 + x2;
                if (mask[ind] > 1)
                  covered_sig += inimg1d[ind];
                total_sig += inimg1d[ind];
              }
            }
          }
          if (covered_sig / total_sig >= 0.9) // 90% joint cover, prune it
          {
            if (tmp_child_num[leaf_marker] == 0) // real leaf node
            {
              leaf_num_pruned++;
              MyMarker *par_marker = leaf_marker->parent;
              if (par_marker)
                tmp_child_num[par_marker]--;
              if (leaf_marker != root_marker) {
                seg->leaf_marker = par_marker;
                is_pruneable[i] = 1; // *** able to prune continuous
              } else // if(leaf_marker == root_marker) // unable to prune
              {
                seg->leaf_marker = NULL;
                seg->root_marker = NULL;
                is_pruneable[i] =
                    0; // *** no marker left, unable to prune again
              }
              // unmask leaf_marker area
              {
                for (int64_t kk = -r; kk <= r; kk++) {
                  int64_t z2 = z + kk;
                  if (z2 < 0 || z2 >= sz2)
                    continue;
                  for (int64_t jj = -r; jj <= r; jj++) {
                    int64_t y2 = y + jj;
                    if (y2 < 0 || y2 >= sz1)
                      continue;
                    for (int64_t ii = -r; ii <= r; ii++) {
                      int64_t x2 = x + ii;
                      if (x2 < 0 || x2 >= sz0)
                        continue;
                      double dst = ii * ii + jj * jj + kk * kk;
                      if (dst > rr)
                        continue;
                      int64_t ind = z2 * sz01 + y2 * sz0 + x2;
                      if (mask[ind] > 1)
                        mask[ind]--;
                    }
                  }
                }
              }
            } else
              is_pruneable[i] = 1; // keep it until it is leaf node
          } else
            is_pruneable[i] = 0;
        }
        cout << "\t iteration [" << iteration++ << "] " << leaf_num_pruned
             << " leaf node pruned" << endl;
      }
      // filter out segments with single marker
      vector<HierarchySegment *> tmp_segs;
      for (int64_t i = 0; i < filter_segs.size(); i++) {
        HierarchySegment *seg = filter_segs[i];
        MyMarker *leaf_marker = seg->leaf_marker;
        MyMarker *root_marker = seg->root_marker;
        if (leaf_marker && root_marker)
          tmp_segs.push_back(seg); // filter out empty segments
      }
      cout << "\t" << filter_segs.size() - tmp_segs.size()
           << " hierarchical segments are pruned in joint leaf node pruning"
           << endl;
      filter_segs.clear();
      filter_segs = tmp_segs;
    }

    if (mask) {
      delete[] mask;
      mask = 0;
    }
  }

  // if(is_smooth) // smooth curve
  //{
  // cout<<"Smooth the final curve"<<endl;
  // for(int64_t i = 0; i < filter_segs.size(); i++)
  //{
  // HierarchySegment * seg = filter_segs[i];
  // MyMarker * leaf_marker = seg->leaf_marker;
  // MyMarker * root_marker = seg->root_marker;
  // vector<MyMarker*> seg_markers;
  // MyMarker * p = leaf_marker;
  // while(p != root_marker)
  //{
  // seg_markers.push_back(p);
  // p = p->parent;
  //}
  // seg_markers.push_back(root_marker);
  // smooth_curve_and_radius(seg_markers, 5);
  //}
  //}
  outswc.clear();
  cout << filter_segs.size() << " segments left" << endl;
  topo_segs2swc(filter_segs, outswc, 0); // no resampling

  // release hierarchical segments
  for (int64_t i = 0; i < topo_segs.size(); i++)
    delete topo_segs[i];
  return true;
}

bool marker_to_swc_file(string swc_file, vector<MyMarker*> & outmarkers)
{
    cout<<"marker num = "<<outmarkers.size()<<", save swc file to "<<swc_file<<endl;
    map<MyMarker*, int> ind;
    ofstream ofs(swc_file.c_str());

    if(ofs.fail())
    {
        cout<<"open swc file error"<<endl;
        return false;
    }
    ofs<<"##n,type,x,y,z,radius,parent"<<endl;
    for(int i = 0; i < outmarkers.size(); i++) ind[outmarkers[i]] = i+1;

    for(int i = 0; i < outmarkers.size(); i++)
    {
        MyMarker * marker = outmarkers[i];
        int parent_id;
        if(marker->parent == 0) parent_id = -1;
        else parent_id = ind[marker->parent];
        ofs<<i+1<<" "<<marker->type<<" "<<marker->x<<" "<<marker->y<<" "<<marker->z<<" "<<marker->radius<<" "<<parent_id<<endl;
    }
    ofs.close();
    return true;
}
