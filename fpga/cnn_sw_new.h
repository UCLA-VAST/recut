#include "util.h"
#define NCONV
#ifdef NCONV
#define IN_NUM 3
#define IN_NUM_HW 8
#define OUT_NUM 16
#define IN_H 130
#define IN_W 130
#define IN_H_HW 130
#define IN_W_HW 130
#define IN_H_T 8
#define IN_W_T 32
#define OUT_H 64
#define OUT_W 64
#define OUT_H_HW 66
#define OUT_W_HW 66
#define IN_NUM_T 8
#define OUT_NUM_T 16
#endif
#ifdef TCONV
#define IN_NUM 3
#define OUT_NUM 16
#define IN_H 31
#define IN_W 31
#define IN_H_HW 33
#define IN_W_HW 33
#define OUT_H 64
#define OUT_W 64
#define IN_NUM_T 16
#define OUT_NUM_T 16
#endif
#ifdef DCONV
#define IN_NUM 3
#define OUT_NUM 16
#define IN_H 36
#define IN_W 36
#define IN_H_HW 36
#define IN_W_HW 36
#define OUT_H 32
#define OUT_W 32
#define IN_NUM_T 8
#define OUT_NUM_T 16
#endif
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