#include "util.h"
#include "cnn_sw.h"


// #define SAVING

#define LOAD_PROGRESS

#ifdef SAVING
  #define SAVE_PROGRESS
#endif




int main(){
  unsigned int cin_size = CIN_SIZE;
  unsigned int bias_size = BIAS_SIZE;
  unsigned int weight_size = WEIGHT_SIZE;
  unsigned int config_size = 5 + LAYER_NUM * CONFIG_PARAMS;

  uint start_layer = LAYER-1;
  uint end_layer = LAYER;

  cout << "cin_size: " << cin_size << endl;
  cout << "bias_size: " << bias_size << endl;
  cout << "weight_size: " << weight_size <<endl;
 
  data_t0* cin_hw = new data_t0[cin_size];
  data_t1* weight_hw = new data_t1[weight_size];
  data_t2* bias_hw = new data_t2[bias_size];

  memset(cin_hw, 0, cin_size);
  memset(weight_hw, 0, weight_size);
  memset(bias_hw, 0, bias_size);

  // Load instructions 
  uint* config = new uint[config_size];
  instInit(config);
  static float outputs_sw[OUT_NUM][OUT_H][OUT_W];
  static float outputs_hw[OUT_NUM][OUT_H][OUT_W];
  static float outputs_py[OUT_NUM][OUT_H][OUT_W];

  preprocess(cin_hw, weight_hw, bias_hw, outputs_sw);

  #ifdef LOAD_PROGRESS
    load_progress(cin_hw);
  #endif

  cout << "HW acceleration..." << endl;
  // Hardware acceleration
  top_kernel(
      (bus_t0*)cin_hw, (bus_t0*)cin_hw, (bus_t0*)cin_hw,
      (bus_t1*)weight_hw, (bus_t2*)bias_hw,
      (bus_t3*)config, start_layer, end_layer);
  cout<<"kernel finished"<<endl;

  #ifdef SAVE_PROGRESS
    save_progress(cin_hw);
  #endif

  postprocess(cin_hw, outputs_hw, outputs_py);

  // cout<<"HARDWARE"<<endl;
  // for(int ch=0; ch<OUT_NUM; ch++){
  //   printf("---------------------channel %d--------------------\n", ch);
  //   for(int h=0; h<OUT_H; h++){
  //     for(int w=0; w<OUT_W; w++){
  //       printf("%f\t", outputs_hw[ch][h][w]);
  //     }
  //     printf("\n");
  //   }
  // }
  // cout<<"SOFTWARE"<<endl;
  // for(int ch=0; ch<OUT_NUM; ch++){
  //   printf("---------------------channel %d--------------------\n", ch);
  //   for(int h=0; h<OUT_H; h++){
  //     for(int w=0; w<OUT_W; w++){
  //       printf("%f\t", outputs_py[ch][h][w]);
  //     }
  //     printf("\n");
  //   }
  // }

  // cout<<"software comparison"<<endl;
  // compareResults(outputs_hw, outputs_sw);
  cout<<"python comparison"<<endl;
  compareResults(outputs_hw, outputs_py);
}
