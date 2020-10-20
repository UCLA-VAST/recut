// last change: by PHC, 2013-02-13. adjust memory allocation to make it more
// robust

/*****************************************************************
 * file : fastmarching_tree.h,  Hang Xiao, Jan 18, 2012
 *
 * fastmarching_tree
 * fastmarching_tracing
 *
 * **************************************************************/

#ifndef __FAST_MARCHING_TREE_PARALLEL_H__
#define __FAST_MARCHING_TREE_PARALLEL_H__

#include "recut.hpp"
using namespace std;

/*********************************************************************
 * Function : fastmarching_tree_parallel
 *
 * Features :
 * 1. Create fast marcing tree from root marker only
 * 2. Background (intensity 0) will be ignored.
 * 3. Graph augumented distance is used
 *
 * Input : root          root marker
 *         inimg1d       original 8bit image
 *
 * Output : tree         output swc
 *          phi          the distance for each pixels
 * *******************************************************************/
template <class T, typename vertex_t, typename out_vertex_t>
bool fastmarching_tree_parallel(vector<vertex_t> roots, T *inimg1d,
                                vector<out_vertex_t> &outtree,
                                vector<int> image_offsets, long sz0, long sz1,
                                long sz2, int block_size, T bkg_thresh,
                                bool restart, double restart_factor,
                                bool is_break_accept, int nthreads) {
  double elapsed = omp_get_wtime();

  VID_t nvid = (VID_t)sz0 * sz1 * sz2;
  VID_t szs[] = {(VID_t)sz0, (VID_t)sz1, (VID_t)sz2};

  auto timer = new high_resolution_timer();

  // GI parameter min_int, max_int, li
  double max_int = 0; // maximum intensity, used in GI
  double min_int = std::numeric_limits<double>::max(); // max value
#pragma omp parallel for reduction(max : max_int)
  for (auto i = 0; i < nvid; i++) {
    if (inimg1d[i] > max_int)
      max_int = inimg1d[i];
  }
#pragma omp parallel for reduction(min : min_int)
  for (auto i = 0; i < nvid; i++) {
    if (inimg1d[i] < min_int)
      min_int = inimg1d[i];
  }
  assert(max_int > min_int);
  max_int -= min_int;

#ifdef DEBUG
  cout << "max_int: " << (int)max_int << " min_int: " << (int)min_int << endl;
#endif

#ifdef LOG
  printf("Find max min wtime: %.1f s\n", timer->elapsed());
#endif

  cout << " restart " << restart << endl;
  // Program<T> nbfs;
  auto recut = Recut<T>();
  recut.update_shardless(roots, image_offsets, inimg1d, nvid, block_size, szs,
                         min_int, max_int, bkg_thresh, restart, restart_factor,
                         nthreads);
  recut.finalize(outtree);

#ifdef LOG
  elapsed = omp_get_wtime() - elapsed;
  // printf("fastmarching_tree_parallel tree reconstruction wtime: %.1f s\n",
  // elapsed);
#endif
  return true;
}

#endif
