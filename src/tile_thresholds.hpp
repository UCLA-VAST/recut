#include "utils.hpp"

template <class image_t> struct TileThresholds {
  image_t max_int;
  image_t min_int;
  image_t bkg_thresh;
  const std::vector<double> givals{
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

  // TileThresholds<image_t>() : max_int(0), min_int(0), bkg_thresh(0) {}
  TileThresholds<image_t>() {}

  TileThresholds<image_t>(image_t max_int, image_t min_int, image_t bkg_thresh)
    : max_int(max_int), min_int(min_int), bkg_thresh(bkg_thresh) {}

  friend std::ostream &operator<<(std::ostream &os, const TileThresholds &tt) {
    os << "[max: " << tt.max_int << ", min: " << tt.min_int << ", bkg_thresh: " << tt.bkg_thresh << ']';
    return os;
  }

  double calc_weight(image_t pixel) const {
    std::ostringstream err;
    //cout << +( pixel ) << '\n';
    //cout << +( this->max_int) << '\n';
    assertm(pixel <= this->max_int, "pixel can not exceed max int");
    assertm(pixel >= this->min_int, "pixel can not be below min int");
    // max and min set as double to align with look up table for value
    // estimation
    auto idx = (int)((pixel - static_cast<double>(this->min_int)) /
        static_cast<double>(this->max_int) * 255);
    assertm(idx < 256, "givals index can not exceed 255");
    assertm(idx >= 0, "givals index negative");
    return this->givals[idx];
  }

  template <typename buffer_t>
  void get_max_min(const buffer_t *img, VID_t tile_vertex_size) {
    auto timer = new high_resolution_timer();

    // GI parameter min_int, max_int
    buffer_t local_max = 0; // maximum intensity, used in GI
    buffer_t local_min = std::numeric_limits<buffer_t>::max(); // max value
    //#pragma omp parallel for reduction(max:local_max)
    for (auto i = 0; i < tile_vertex_size; i++) {
      if (img[i] > local_max) {
        local_max = img[i];
      }
    }
    //#pragma omp parallel for reduction(min:local_min)
    for (auto i = 0; i < tile_vertex_size; i++) {
      if (img[i] < local_min) {
        local_min = img[i];
        // cout << "local_min" << +local_min << '\n';
      }
    }
    if (local_min == local_max) {
#ifdef LOG_FULL
      cout << "Warning: max: " << local_max << "= min: " << local_min << '\n';
#endif
    } else if (local_min > local_max) {
      cout << "Error: max: " << local_max << "< min: " << local_min << '\n';
      throw;
    }
    this->max_int = local_max;
    this->min_int = local_min;

#ifdef FULL_PRINT
    printf("Find max min wtime: %.1f s\n", timer->elapsed());
#endif
  }
};
