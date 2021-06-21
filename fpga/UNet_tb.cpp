#include "util.h"
#include "cnn_sw.h"


int main(){
  unsigned int cin_size = CIN_SIZE;
  unsigned int bias_size = BIAS_SIZE;
  unsigned int weight_size = WEIGHT_SIZE;
  unsigned int config_size = 5 + LAYER_NUM * CONFIG_PARAMS;
  uint start_layer = 0;
  if(LOAD_PROGRESS==1 || SAVE_PROGRESS==1)
    start_layer = LAYER-1;
  else
    start_layer = LAYER-LAYER;

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

  if(LOAD_PROGRESS==1){
    load_progress(cin_hw);
    if(end_layer<3)
      preprocess(cin_hw, weight_hw, bias_hw, outputs_sw);
    else
      preprocess(NULL, weight_hw, bias_hw, outputs_sw);
  }else{
    preprocess(cin_hw, weight_hw, bias_hw, outputs_sw);
  }

  // preprocess(cin_hw, weight_hw, bias_hw, outputs_sw);
  // static float inputs[8][18][18] = {{{0}}};
  // string prj_path = PRJ_PATH;
  // string prj_path_string = prj_path + "/data1";
  // const char* prj_path_c = prj_path_string.c_str();
  // string file_path = string(prj_path_c) + "/inputs.dat"; 
  // ifstream input_file1(file_path.c_str());
  // if (input_file1.is_open()){
  //   int padding_offset = (IN_H_HW-IN_H)/2;
  //   int idx = 0;
  //   // cout<<padding_offset<<endl;
  //   for (int i = 0; i < IN_NUM_HW; i++)
  //     for (int h = padding_offset; h < IN_H_HW-padding_offset; h++)
  //       for (int w = padding_offset; w < IN_W_HW-padding_offset; w++)
  //       {
  //         input_file1 >> inputs[i][h][w];
  //         idx++;
  //       }

  //   input_file1.close();
  // } else {
  //   cout << "Input open failed!" << endl;
  //   exit(-1);
  // }
  // cout<<"INPUTS"<<endl;
  // for(int i=0; i<IN_NUM_HW; i++){
  //   printf("---------------------channel %d--------------------\n", i);
  //   for(int h=0; h<IN_H_HW; h++){
  //     for(int w=0; w<IN_W_HW; w++){
  //       printf("%10f\t", inputs[i][h][w]);
  //     }
  //     printf("\n");
  //   }
  // }
  // exit(0);
  // for(int h=0; h<IN_H_HW; h++){
  //   for(int w=0; w<IN_W_HW; w++){
  //     for(int i=0; i<IN_NUM_HW; i++){
  //       cin_hw[h*IN_NUM_HW*IN_W_HW+w*IN_NUM_HW+i] = inputs[i][h][w];
  //     }
  //   }
  // }
  // cout<<"INPUTS"<<endl;
  // for(int i=0; i<IN_NUM_HW; i++){
  //   printf("---------------------channel %d--------------------\n", i);
  //   for(int h=0; h<IN_H_HW; h++){
  //     for(int w=0; w<IN_W_HW; w++){
  //       printf("%10f\t", cin_hw[h*IN_NUM_HW*IN_W_HW+w*IN_NUM_HW+i]);
  //     }
  //     printf("\n");
  //   }
  // }
  // exit(0);
  // for(int h=0; h<IN_H_HW; h++){
  //   for(int w=0; w<IN_W_HW; w++){
  //     for(int i=0; i<IN_NUM_HW; i++){
  //       if(h==0 || w==0 || h==IN_H_HW-1 || w==IN_W_HW-1)
  //         cin_hw[h*IN_NUM_HW*IN_W_HW+w*IN_NUM_HW+i] = 0.0;
  //       else
  //         cin_hw[h*IN_NUM_HW*IN_W_HW+w*IN_NUM_HW+i] = inputs[i][h][w];
  //       cout<<h*IN_NUM_HW*IN_W_HW+w*IN_NUM_HW+i<<" "<<cin_hw[h*IN_NUM_HW*IN_W_HW+w*IN_NUM_HW+i]<<endl;
  //     }
  //   }
  // }

  // cout << "Loading weight..." << endl; 
  // file_path = string(prj_path_c) + "/weights.dat"; 
  // ifstream weight_file(file_path.c_str()); 
  // if (weight_file.is_open()){
  //   //weight_file.read(bin_input, sizeof(data_t1) * WEIGHT_SIZE);
  //   //data_t1* convt_input = (data_t1*)bin_input;

  //   for (int w = 0; w < IN_NUM_HW*OUT_NUM_HW*FILTER_S2*FILTER_S2; w++){
  //     weight_file >> weight_hw[w];
  //   }

  //   weight_file.close();
  // } else {
  //   cout << "Weight open failed!" << endl;
  //   exit(-1);
  // }

  // exit(0);

  // // #ifdef LOAD_PROGRESS
  //   load_progress(cin_hw);
  // #endif
  // cout<<cin_hw[1065087]<<endl;
  // cout<<cin_hw[1065088]<<endl;
  // cout<<cin_hw[1065089]<<endl;
  // exit(0);
  cout << "HW acceleration..." << endl;
  // Hardware acceleration
  top_kernel(
      (bus_t0*)cin_hw, (bus_t0*)cin_hw, (bus_t0*)cin_hw,
      (bus_t1*)weight_hw, (bus_t2*)bias_hw,
      (bus_t3*)config, start_layer, end_layer);
  cout<<"kernel finished"<<endl;

  if(SAVE_PROGRESS==1)
    save_progress(cin_hw, OUT_OFFSET2+OUT_H_HW*OUT_W_HW*OUT_NUM_HW+130*130*32);
  // save_progress(cin_hw, OUT_OFFSET2+OUT_H_HW*OUT_W_HW*OUT_NUM_HW+130*130*32);

  postprocess(cin_hw, outputs_hw, outputs_py);

  cout<<"HARDWARE"<<endl;
  for(int ch=0; ch<1; ch++){
    printf("---------------------channel %d--------------------\n", ch);
    for(int h=0; h<OUT_H; h++){
      for(int w=0; w<OUT_W; w++){
        printf("%10f\t", outputs_hw[ch][h][w]);
      }
      printf("\n");
    }
  }
  cout<<"SOFTWARE"<<endl;
  for(int ch=0; ch<1; ch++){
    printf("---------------------channel %d--------------------\n", ch);
    for(int h=0; h<OUT_H; h++){
      for(int w=0; w<OUT_W; w++){
        printf("%10f\t", outputs_py[ch][h][w]);
      }
      printf("\n");
    }
  }

  // cout<<"software comparison"<<endl;
  // compareResults(outputs_hw, outputs_sw);
  cout<<"python comparison"<<endl;
  compareResults(outputs_hw, outputs_py);
}
