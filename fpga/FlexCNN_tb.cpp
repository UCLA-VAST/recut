#include "util.h"

#define NCONV

#ifdef NCONV
#define IN_NUM 16
#define OUT_NUM 16
#define IN_H 34
#define IN_W 34
#define IN_H_HW 34
#define IN_W_HW 34
#define OUT_H 32
#define OUT_W 32
#define OUT_H_HW 32
#define OUT_W_HW 32
#define IN_NUM_T 16
#define OUT_NUM_T 16
#endif

#ifdef TCONV
#define IN_NUM 16
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
#define IN_NUM 8
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





float random_float()
{
  int r_int = rand() % 65536;
  return (r_int - 32768.0) / 32768.0;
}

// Loads instructions
void instInit(
  uint* config
){
  cout << "Loading instructions..." << endl;
  string file_path = "D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/test.insts";
  ifstream in_file(file_path.c_str());
  
  // model configuration
  config[0] = LAYER_NUM;
 
  if (in_file.is_open()){
    for (int layer_id = 0; layer_id < LAYER_NUM; layer_id++){
      uint p;
      int param_cnt = 0;
      while(param_cnt < CONFIG_PARAMS){
        in_file >> p;
        config[5 + layer_id * CONFIG_PARAMS + param_cnt] = p;
        param_cnt++;
      }
    }
    in_file.close();
  } else {
    cout << "CONFIG open failed!" << endl;
    exit(-1);
  }
}

void compareResults(data_t0  outputs_hw[OUT_NUM][OUT_H][OUT_W], data_t0  outputs_sw[OUT_NUM][OUT_H][OUT_W]){
  cout << "Results comparison..." << endl;
  bool flag = true;
  int err_count = 0;
  for(int ch=0; ch<OUT_NUM; ch++){
    for(int h=0; h<OUT_H; h++){
      for(int w=0; w<OUT_W; w++){
        if(abs(outputs_hw[ch][h][w]-outputs_sw[ch][h][w])>0.01){
          flag = false;
          cout<<outputs_hw[ch][h][w]<<" "<<outputs_sw[ch][h][w]<<endl;
          err_count++;
        }
      }
    }
  }
  if(flag)
    cout<<"SUCESS!"<<endl;
  else
    cout<<"FAILURE! "<<err_count<<" errors"<<endl;
}
// void instInit(
//   uint* config
// ){
//   cout << "Loading instruction..." << endl;
//   char* prj_path_c = getenv("PRJ_PATH");
//   string prj_path = prj_path_c;
//   #ifdef NCONV
//     string fileName = "NCONV_old";
//   #endif
//   #ifdef TCONV
//     string fileName = "TCONV_old";
//   #endif
//   #ifdef DCONV
//     string fileName = "DCONV_old";
//   #endif
//   string file_path = prj_path + "/SDx_project/src/insts/"+fileName+".insts";
//   ifstream in_file(file_path.c_str());
  
//   // model configuration
//   config[0] = LAYER_NUM;
 
//   if (in_file.is_open()){
//     for (int layer_id = 0; layer_id < LAYER_NUM; layer_id++){
//       uint p;
//       int param_cnt = 0;
//       while(param_cnt < CONFIG_PARAMS){
//         in_file >> p;
//         config[5 + layer_id * CONFIG_PARAMS + param_cnt] = p;
// //        if (layer_id == 0){
// //          cout << p << endl;
// //        }
//         param_cnt++;
//       }
//     }
//     in_file.close();
//   } else {
//     cout << "CONFIG open failed!" << endl;
//     exit(-1);
//   }
// }

void initData(float inputs[IN_NUM][IN_H][IN_W], float weights[OUT_NUM][IN_NUM][3][3]){
    for(int i=0; i<IN_NUM; i++){
      for(int h=0; h<IN_H; h++){
        for(int w=0; w<IN_W; w++){
          // if(!(w==0 || h==0 || w==(IN_W-1) || h==(IN_H-1)))
            inputs[i][h][w] = random_float();//(i*IN_W*IN_H + h*IN_W + w);//1;//IN_H*j + k + i;//(i*TRANS_IN_H_T*TRANS_IN_W_T+j*TRANS_IN_W_T+k);//
        }
      }
    }
    for(int o=0; o<OUT_NUM; o++){
      for(int i=0; i<IN_NUM; i++){
        for(int p=0; p<3; p++){
          for(int q=0; q<3; q++){
              weights[o][i][p][q] = 1;//(o*IN_NUM*3*3 + i*3*3 + p*3 + q)/256.0;//random_float();//random_float();//1.0;//(o*IN_NUM_T*K_T*K_T + i*K_T*K_T + p*K_T + q)%7;//1;
          }
        }
      }
    }
}

void nconv(float inputs[IN_NUM][IN_H_HW][IN_W_HW], float weights[OUT_NUM][IN_NUM][3][3], float outputs[OUT_NUM][OUT_H][OUT_W]){
  int DILATE_FACTOR = 1;
  for (int o = 0; o < OUT_NUM; o++){
    for (int i = 0; i < IN_NUM; i++){ 
      for (int h = 0; h < OUT_H; h++){
        for (int w = 0; w < OUT_W; w++){ 
          for (int p = 0; p < 3; p++){
            for (int q = 0; q < 3; q++){
              outputs[o][h][w] += inputs[i][h+p*DILATE_FACTOR][w+q*DILATE_FACTOR] * weights[o][i][p][q];
            }
          }
        }
      }
    }
  }
}

void tconv(float inputs[IN_NUM][IN_H][IN_W], float weights[OUT_NUM][IN_NUM][3][3], float outputs_sw[OUT_NUM][OUT_H][OUT_W]){
  int STRIDE = 2;
  for (int o = 0; o < OUT_NUM; o++){
    for (int h = 0; h < IN_H; h++){
      for (int w = 0; w < IN_W; w++){ 
        for (int p = 0; p < 3; p++){
          for (int q = 0; q < 3; q++){
            for (int i = 0; i < IN_NUM; i++){ 
              outputs_sw[o][h+p+h*(STRIDE-1)][w+q+w*(STRIDE-1)] += inputs[i][h][w]*weights[o][i][p][q];
            }
          }
        }
      }
    }
  }
}

void dconv(float inputs[IN_NUM][IN_H][IN_W], float weights[OUT_NUM][IN_NUM][3][3], float outputs_sw[OUT_NUM][OUT_H][OUT_W]){
  int DILATE_FACTOR = 2;
  for (int o = 0; o < OUT_NUM; o++){
    for (int i = 0; i < IN_NUM; i++){ 
      for (int h = 0; h < OUT_H; h++){
        for (int w = 0; w < OUT_W; w++){ 
          for (int p = 0; p < 3; p++){
            for (int q = 0; q < 3; q++){
              // outputs_sw[o][h+p+h*(STRIDE-1)][w+q+w*(STRIDE-1)] += inputs[i][h][w]*weights[o][i][p][q];
              outputs_sw[o][h][w] += inputs[i][h+p*DILATE_FACTOR][w+q*DILATE_FACTOR] * weights[o][i][p][q];
              // if(o==0 && i==0) cout<<"outputs_sw["<<o<<"]["<<h+p+h*(STRIDE-1)<<"]["<<w+q+w*(STRIDE-1)<<"]("<<outputs_sw[o][h+p+h*(STRIDE-1)][w+q+w*(STRIDE-1)]<<") += inputs["<<i<<"]["<<h<<"]["<<w<<"]*weights["<<o<<"]["<<i<<"]["<<p<<"]["<<q<<"]"<<endl;
            }
          }
        }
      }
    }
  }
}


void postprocess(
  data_t0* cin_hw,
  data_t0  outputs_hw[OUT_NUM][OUT_H][OUT_W]
){
  for (int o1 = 0; o1 < OUT_NUM / OUT_NUM_T; o1++)
    for (int h = 0; h < OUT_H; h++)
      for (int w = 0; w < OUT_W; w++)
        for (int o2 = 0; o2 < OUT_NUM_T; o2++){
        int o = o1 * OUT_NUM_T + o2;
          if (o < OUT_NUM){
            outputs_hw[o][h][w] = cin_hw[1195088 + o1*OUT_H*OUT_W*OUT_NUM_T + h * OUT_W * OUT_NUM_T + w * OUT_NUM_T + o2];
          }
        }
}

void preprocess(
  data_t0* cin_hw,
  data_t1* weight_hw,
  data_t2* bias_hw,
  data_t0  outputs_sw[OUT_NUM][OUT_H][OUT_W]
){

  static float inputs[IN_NUM][IN_H_HW][IN_W_HW] = {{{0}}};
  static float weights[OUT_NUM][IN_NUM][3][3];
  char* prj_path_c = "D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt";
  // Prepare the software buffers
  cout << std::fixed << "Preparing data..." << endl;
  // first layer
  
  // Load the inputs for the network
  // static data_t0 inputs[IN_NUM][IN_H_HW][IN_W_HW] = {{{0}}};
  cout << "Loading input..." << endl; 
  //string file_path = string(prj_path_c) + "/data_layer/input.dat";  
  string file_path = string(prj_path_c) + "/inputs.dat"; 
  ifstream input_file(file_path.c_str());
  if (input_file.is_open()){

    int idx = 0;
    for (int i = 0; i < IN_NUM; i++)
      for (int h = 1; h < IN_H_HW-1; h++)
        for (int w = 1; w < IN_W_HW-1; w++)
        {
          input_file >> inputs[i][h][w];
          idx++;
        }

    input_file.close();
  } else {
    cout << "Input open failed!" << endl;
    exit(-1);
  }
  //delete[] bin_input;

  // Initialize the hardware input buffer
  // Cin layout: [IN_NUM / IN_NUM_T][IN_H + K - 1][IN_W + K - 1][IN_NUM_T]
  for (int i1 = 0; i1 < IN_NUM/IN_NUM_T; i1++){
    for (int h = 0; h < IN_H_HW; h++){
      for (int w = 0; w < IN_W_HW; w++){
        for (int i2 = 0; i2 < IN_NUM_T; i2++){//IN_NUM should be 8
          int i = i1 * IN_NUM_T + i2;
          cin_hw[i1*IN_H_HW*IN_W_HW*IN_NUM_T + h*IN_W*IN_NUM_T + w*IN_NUM_T + i2] = inputs[i][h][w];
        }
      }
    }
  }

  // Load weights
  cout << "Loading weight..." << endl;

  // Load outputs
  cout << "calculating output sw..." << endl;
  for(int o=0; o<OUT_NUM; o++){
    for(int p=0; p<3; p++){
      for(int q=0; q<3; q++){
        for(int i1=0; i1<IN_NUM/8; i1++){
          for(int i2=0; i2<8; i2++){
            int i = i1 * 8 + i2;
            weight_hw[o*3*3*IN_NUM+p*3*IN_NUM+q*IN_NUM+i] = 1;//weights[o][i][p][q];
            weights[o][i][p][q] = 1;
          }
        }
      }
    }
  }
  nconv(inputs, weights, outputs_sw);
}

int main(){
  // working path
  char* prj_path = "D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/";
  if (prj_path != NULL){
    cout << "Your working PATH is: " << prj_path << endl;
  } else {
    cout << "Working PATH not set!" << endl;
    return -1;
  }
  
  unsigned int cin_size = CIN_SIZE;
  unsigned int bias_size = BIAS_SIZE;
  unsigned int weight_size = WEIGHT_SIZE;
  unsigned int config_size = 5 + LAYER_NUM * CONFIG_PARAMS;

  cout << "cin_size: " << cin_size << endl;
  cout << "bias_size: " << bias_size << endl;
  cout << "weight_size: " << weight_size <<endl;
 
  data_t0* cin_hw = new data_t0[cin_size];
  data_t1* weight_hw = new data_t1[weight_size];
  data_t2* bias_hw = new data_t2[bias_size];

  // memset(cin_hw, 0, cin_size);
  // memset(weight_hw, 0, weight_size);
  // memset(bias_hw, 0, bias_size);

  // Load instructions 
  uint* config = new uint[config_size];
  instInit(config);
  // static float inputs[IN_NUM][IN_H_HW][IN_W_HW] = {{{0}}};
  // static float weights[OUT_NUM][IN_NUM][3][3];
  static float outputs_sw[OUT_NUM][OUT_H][OUT_W];
  static float outputs_hw[OUT_NUM][OUT_H][OUT_W];
  // // initData(inputs, weights);

  // // #ifdef NCONV
  // //   nconv(inputs, weights, outputs_sw);
  // // #endif
  // // #ifdef DCONV
  // //   dconv(inputs, weights, outputs_sw);
  // // #endif
  // // #ifdef TCONV
  // //   tconv(inputs, weights, outputs_sw);
  // // #endif
  
  // cout << "Loading input..." << endl; 
  // //string file_path = string(prj_path_c) + "/data_layer/input.dat";  
  // string file_path = string(prj_path) + "inputs.dat"; 
  // ifstream input_file(file_path.c_str());
  // if (input_file.is_open()){

  //   int idx = 0;
  //   for (int i = 0; i < IN_NUM; i++)
  //     for (int h = 1; h < IN_H_HW-1; h++)
  //       for (int w = 1; w < IN_W_HW-1; w++)
  //       {
  //         input_file >> inputs[i][h][w];
  //         idx++;
  //       }

  //   input_file.close();
  // } else {
  //   cout << "Input open failed!" << endl;
  //   exit(-1);
  // }

  // cout<<"INPUTS"<<endl;
  // for(int ch=0; ch<IN_NUM; ch++){
  //   printf("---------------------channel %d--------------------\n", ch);
  //   for(int h=0; h<IN_H_HW; h++){
  //     for(int w=0; w<IN_W_HW; w++){
  //       printf("%f\t", inputs[ch][h][w]);
  //     }
  //     printf("\n");
  //   }
  // }
  // for (int i1 = 0; i1 < IN_NUM / IN_NUM_T; i1++)
  //   for (int h = 0; h < IN_H; h++)
  //     for (int w = 0; w < IN_W; w++)
  //       for (int i2 = 0; i2 < IN_NUM_T; i2++){
  //         int i = i1 * IN_NUM_T + i2;
  //         if (i < IN_NUM){
  //           cin_hw[i1 * IN_H_HW * IN_W_HW * IN_NUM_T + (h + int(3 / 2)) * IN_W_HW * IN_NUM_T + (w + int(3/ 2)) * IN_NUM_T + i2] = inputs[i][h][w]; // filter size = 3
  //         }
  //       }
  // for (int i1 = 0; i1 < IN_NUM/IN_NUM_T; i1++){
  //   for (int h = 0; h < IN_H_HW; h++){
  //     for (int w = 0; w < IN_W_HW; w++){
  //       for (int i2 = 0; i2 < IN_NUM_T; i2++){//IN_NUM should be 8
  //         int i = i1 * IN_NUM_T + i2;
  //         #ifndef TCONV
  //           cin_hw[i1*IN_H_HW*IN_W_HW*IN_NUM_T + h*IN_W*IN_NUM_T + w*IN_NUM_T + i2] = inputs[i][h][w];
  //         #endif
  //         #ifdef TCONV
  //           float num;// =  8*k + i_t;//static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;//i*33+j+1;//8*k + i_t;//inputs[k][i][j];//static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  //           num = (h==0 || w==0)? 0 : (h!=0 && w!=0)? inputs[i][h-1][w-1] : (h==0)? inputs[i][h][w-1] : inputs[i][h-1][w];
  //           if(h==(IN_H+1) || w==(IN_W+1)) num = 0;
  //           cin_hw[h*IN_NUM*IN_W_HW+w*IN_NUM+i] = num;//1;//i*IN_H_HW*IN_W_HW + h*IN_W_HW + w;
  //         #endif
  //       }
  //     }
  //   }
  // }
  // // for(int i=0; i<33*33*8; i++){
  // //   cout<<cin_hw[i]<<endl;
  // // }
  // for(int o=0; o<OUT_NUM; o++){
  //   for(int p=0; p<3; p++){
  //     for(int q=0; q<3; q++){
  //       for(int i1=0; i1<IN_NUM/8; i1++){
  //         for(int i2=0; i2<8; i2++){
  //           int i = i1 * 8 + i2;
  //           weight_hw[o*3*3*IN_NUM+p*3*IN_NUM+q*IN_NUM+i] = 1;//weights[o][i][p][q];
  //           weights[o][i][p][q] = 1;
  //         }
  //       }
  //     }
  //   }
  // }

  // nconv(inputs, weights, outputs_sw);

  preprocess(cin_hw, weight_hw, bias_hw, outputs_sw);

  cout << "HW acceleration..." << endl;


  // Hardware acceleration
  top_kernel(
      (bus_t0*)cin_hw, (bus_t0*)cin_hw, (bus_t0*)cin_hw,
      (bus_t1*)weight_hw, (bus_t2*)bias_hw,
      (bus_t3*)config);
  cout<<"kernel finished"<<endl;

  postprocess(cin_hw, outputs_hw);
  // for (int o1 = 0; o1 < OUT_NUM / OUT_NUM_T; o1++)
  //   for (int h = 0; h < OUT_H; h++)
  //     for (int w = 0; w < OUT_W; w++)
  //       for (int o2 = 0; o2 < OUT_NUM_T; o2++){
  //       int o = o1 * OUT_NUM_T + o2;
  //         if (o < OUT_NUM){
  //           outputs_hw[o][h][w] = cin_hw[1195088 + o1*OUT_H*OUT_W*OUT_NUM_T + h * OUT_W * OUT_NUM_T + w * OUT_NUM_T + o2];
  //         }
  //       }

    //   for (int o1 = 0; o1 < STAGE2L_OUT_NUM_HW / STAGE2L_OUT_NUM_T; o1++)
    // for (int h = 0; h < STAGE2L_OUT_H; h++)
    //   for (int w = 0; w < STAGE2L_OUT_W; w++)
    //     for (int o2 = 0; o2 < STAGE2L_OUT_NUM_T; o2++){
    //       int o = o1 * STAGE2L_OUT_NUM_T + o2;
    //       if (o < STAGE2L_OUT_NUM){
    //         LAYER_out[h][w][o + STAGE2R_OUT_NUM] = cin_hw[STAGE2L_OFFSET + o1 * STAGE2L_OUT_H_HW * STAGE2L_OUT_W_HW * STAGE2L_OUT_NUM_T + (h + int(STAGE2L_K / 2)) * STAGE2L_OUT_W_HW * STAGE2L_OUT_NUM_T + (w + int(STAGE2L_K / 2)) * STAGE2L_OUT_NUM_T + o2];
    //       }
    //     }


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
  //       printf("%f\t", outputs_sw[ch][h][w]);
  //     }
  //     printf("\n");
  //   }
  // }
  compareResults(outputs_hw, outputs_sw);
  // bool flag = true;
  // int err_count = 0;
  // for(int ch=0; ch<OUT_NUM; ch++){
  //   for(int h=0; h<OUT_H; h++){
  //     for(int w=0; w<OUT_W; w++){
  //       if(abs(outputs_hw[ch][h][w]-outputs_sw[ch][h][w])>0.01){
  //         flag = false;
  //         err_count++;
  //       }
  //     }
  //   }
  // }
  // if(flag)
  //   cout<<"SUCESS!"<<endl;
  // else
  //   cout<<"FAILURE! "<<err_count<<" errors"<<endl;
}
