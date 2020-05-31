// These functions are from the APP2 codebase (Hanchuan Peng, Vaa3D, Allen
// Institute) to serve as baseline sequential comparisons where appropriate

#include <cassert>
#include <stdlib.h> // abs
#include <vector>

using namespace std;

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
      //cout << "\r" << ((int)process2) / 1000.0 << "%";
      //cout.flush();
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
          long index = d * sz01 + h * sz0 + w;
          // discard background pixels
          if (inimg1d[index] <= bkg_thresh) {
            continue;
          }

          if (state[index] != ALIVE) {
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
  VID_t current_x, current_y, current_z;
  get_img_subscript(current_vid, current_x, current_y, current_z, grid_size);

  int max_r = grid_size / 2 - 1;
  int r;
  double tol_num, bak_num;
  int mx = current_x + 0.5;
  int my = current_y + 0.5;
  int mz = current_z + 0.5;
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
template <class T>
uint16_t get_radius_hanchuan_XY(const T *inimg1d, int grid_size,
                                VID_t current_vid, T thresh) {
  std::vector<int> sz = {grid_size, grid_size, grid_size};
  VID_t x, y, z;
  get_img_subscript(current_vid, x, y, z, grid_size);

  long sz0 = sz[0];
  long sz01 = sz[0] * sz[1];
  double max_r = grid_size / 2 - 1;
  if (max_r > sz[1] / 2)
    max_r = sz[1] / 2;

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
            int64_t i = x + dx;
            if (i < 0 || i >= sz[0])
              goto end1;
            int64_t j = y + dy;
            if (j < 0 || j >= sz[1])
              goto end1;
            int64_t k = z + dz;
            if (k < 0 || k >= sz[2])
              goto end1;

            if (inimg1d[k * sz01 + j * sz0 + i] <= thresh) {
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
