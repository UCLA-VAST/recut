#include "util.h"
#define LAYER 3
#define OUTFILE "/data/L3_outputs.dat"
#define OUT_OFFSET1 1724464
#define OUT_OFFSET2 1725536
#define CHANGE_LAYOUT 0
#define IN_NUM_HW 16
#define OUT_NUM_HW 16
#define IN_H_HW 128
#define IN_W_HW 128
#define OUT_H_HW 66
#define OUT_W_HW 66
#define IN_NUM 16
#define OUT_NUM 16
#define IN_H 128
#define IN_W 128
#define OUT_H 64
#define OUT_W 64
#define IN_NUM_T 16
#define OUT_NUM_T 16
#define IN_H_T 8
#define IN_W_T 64
void instInit(uint* config);
void preprocess(
  data_t0* cin_hw,
  data_t1* weight_hw,
  data_t2* bias_hw,
  data_t0  outputs_sw[OUT_NUM][OUT_H][OUT_W]
);
void postprocess(
  data_t0* cin_hw,
  data_t0  outputs_hw[OUT_NUM][OUT_H][OUT_W],
  data_t0  outputs_py[OUT_NUM][OUT_H][OUT_W]
);
void compareResults(data_t0  outputs_hw[OUT_NUM][OUT_H][OUT_W], data_t0  outputs_sw[OUT_NUM][OUT_H][OUT_W]);
void save_progress(data_t0* cin_hw, uint data_offset);
void load_progress(data_t0* cin_hw);