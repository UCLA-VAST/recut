#include "cnn_sw.h"
#include <iomanip>
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
  string prj_path = PRJ_PATH;
  string file_path = prj_path + "/test.insts";
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

void preprocess(
  data_t0* cin_hw,
  data_t1* weight_hw,
  data_t2* bias_hw,
  data_t0  outputs_sw[OUT_NUM][OUT_H][OUT_W]
){

  static float inputs[IN_NUM][IN_H_HW][IN_W_HW] = {{{0}}};
  static float weights[OUT_NUM][IN_NUM][3][3];
  string prj_path = PRJ_PATH;
  string prj_path_string = prj_path + "/data";
  const char* prj_path_c = prj_path_string.c_str();
  // Prepare the software buffers
  cout << std::fixed << "Preparing data..." << endl;
  // first layer
  
  // Load the inputs for the network
  // static data_t0 inputs[IN_NUM][IN_H_HW][IN_W_HW] = {{{0}}};
  // cout << "Loading input..." << endl; 
  // //string file_path = string(prj_path_c) + "/data_layer/input.dat";  
  // string file_path = string(prj_path_c) + "/inputs.dat"; 
  // ifstream input_file(file_path.c_str());
  // if (input_file.is_open()){

  //   int idx = 0;
  //   for (int i = 0; i < IN_NUM; i++)
  //     for (int h = 2; h < IN_H_HW-2; h++)
  //       for (int w = 2; w < IN_W_HW-2; w++)
  //       {
  //         input_file >> inputs[i][h][w];
  //         idx++;
  //       }

  //   input_file.close();
  // } else {
  //   cout << "Input open failed!" << endl;
  //   exit(-1);
  // }
  //delete[] bin_input;
  cout << "Loading input..." << endl; 
  //string file_path = string(prj_path_c) + "/data_layer/input.dat";  
  string file_path = string(prj_path_c) + "/inputs.dat"; 
  ifstream input_file(file_path.c_str());
  if (input_file.is_open()){
    int padding_offset = (IN_H_HW-IN_H)/2;
    int idx = 0;
    for (int i = 0; i < IN_NUM; i++)
      for (int h = padding_offset; h < IN_H_HW-padding_offset; h++)
        for (int w = padding_offset; w < IN_W_HW-padding_offset; w++)
        {
          input_file >> inputs[i][h][w];
          idx++;
        }

    input_file.close();
  } else {
    cout << "Input open failed!" << endl;
    exit(-1);
  }

  if(FILTER_S2==1){
    for (int i1 = 0; i1 < IN_NUM_HW/IN_NUM_T; i1++){
      for (int w1 = 0; w1 < IN_W_HW / IN_W_T; w1++){
        for (int h1 = 0; h1 < IN_H_HW / IN_H_T; h1++){
          for(int h2 = 0; h2 < IN_H_T; h2++){
            for(int w2 = 0; w2 < IN_W_T; w2++){
              for (int i2 = 0; i2 < IN_NUM_T; i2++){//IN_NUM should be 8
                int i = i1 * IN_NUM_T + i2;
                int h = h1 * IN_H_T + h2;
                int w = w1 * IN_W_T + w2;
                // cout<<i<<" "<<h<<" "<<w<<endl;
                
                int L1 = i1 * IN_H * IN_W * IN_NUM_T;
                int L2 = w1 * IN_H * IN_W_T * IN_NUM_T;
                int L3 = h1 * IN_H_T * IN_W_T * IN_NUM_T;
                int L4 = h2 * IN_W_T * IN_NUM_T;
                int L5 = w2 * IN_NUM_T;
                int L6 = i2;
                if (i < IN_NUM)
                  cin_hw[CIN_OFFSET + L1 + L2 + L3 + L4 + L5 + L6 ] = inputs[i][h][w];
                  // cin_hw[i1*IN_H_HW*IN_W_HW*IN_NUM_T + h*IN_W_HW*IN_NUM_T + w*IN_NUM_T + i2] = inputs[i][h][w];
                  // cout<<i1*IN_H_HW*IN_W_HW*IN_NUM_T + h*IN_W_HW*IN_NUM_T + w*IN_NUM_T + i2<<" "<<i<<" "<<h<<" "<<w<<endl;
              }
            }
          }
        }
      }
    }
    // exit(0);
  }else{
    // Initialize the hardware input buffer
    // Cin layout: [IN_NUM / IN_NUM_T][IN_H + K - 1][IN_W + K - 1][IN_NUM_T]
    cout<<IN_H_HW<<" "<<IN_W_HW<<endl;
    for (int i1 = 0; i1 < IN_NUM_HW/IN_NUM_T; i1++){
      for (int h = 0; h < IN_H_HW; h++){
        for (int w = 0; w < IN_W_HW; w++){
          for (int i2 = 0; i2 < IN_NUM_T; i2++){//IN_NUM should be 8
            int i = i1 * IN_NUM_T + i2;
            if (i < IN_NUM)
              cin_hw[CIN_OFFSET + i1*IN_H_HW*IN_W_HW*IN_NUM_T + h*IN_W_HW*IN_NUM_T + w*IN_NUM_T + i2] = inputs[i][h][w];
              // cout<<i1*IN_H_HW*IN_W_HW*IN_NUM_T + h*IN_W_HW*IN_NUM_T + w*IN_NUM_T + i2<<" "<<i<<" "<<h<<" "<<w<<endl;
          }
        }
      }
    }
  }
  // cout<<cin_hw[540800]<<endl;
  // cout<<cin_hw[540801]<<endl;
  // exit(0);
  // for(int i=0; i<258*258*8; i++){
  //   cout<<cin_hw[i]<<endl;
  // }
  // exit(0);
  // cout<<"INPUTS"<<endl;
  // for(int ch=0; ch<IN_NUM; ch++){
  //   printf("---------------------channel %d--------------------\n", ch);
  //   for(int h=0; h<IN_H_HW; h++){
  //     for(int w=0; w<IN_W_HW; w++){
  //       printf("%10f\t", inputs[ch][h][w]);
  //     }
  //     printf("\n");
  //   }
  // }
  // exit(0);
  // Load weights
  cout << "Loading weight..." << endl;
  file_path = string(prj_path_c) + "/weights.dat"; 
  ifstream weight_file(file_path.c_str()); 
  //ifstream weight_file(file_path.c_str(), ios::binary | ios::in);
  //bin_input = new char[sizeof(data_t1) * WEIGHT_SIZE];
  if (weight_file.is_open()){
    //weight_file.read(bin_input, sizeof(data_t1) * WEIGHT_SIZE);
    //data_t1* convt_input = (data_t1*)bin_input;

    for (int w = 0; w < 1000000; w++){
      weight_file >> weight_hw[w];
      // weight_hw[w] = 1.0;
    }

    weight_file.close();
  } else {
    cout << "Weight open failed!" << endl;
    exit(-1);
  }

  cout << "Loading biases..." << endl;
  file_path = string(prj_path_c) + "/biases.dat"; 
  ifstream bias_file(file_path.c_str()); 
  //ifstream weight_file(file_path.c_str(), ios::binary | ios::in);
  //bin_input = new char[sizeof(data_t1) * WEIGHT_SIZE];
  if (bias_file.is_open()){
    //weight_file.read(bin_input, sizeof(data_t1) * WEIGHT_SIZE);
    //data_t1* convt_input = (data_t1*)bin_input;

    for (int w = 0; w < 2*16*16*9; w++){
      bias_file >> bias_hw[w];
      // weight_hw[w] = 1.0;
    }

    bias_file.close();
  } else {
    cout << "Bias open failed!" << endl;
    exit(-1);
  }
  // Load outputs
  cout << "calculating output sw..." << endl;
  
  for(int o=0; o<OUT_NUM; o++){
    for(int p=0; p<3; p++){
      for(int q=0; q<3; q++){
        for(int i1=0; i1<IN_NUM_HW/8; i1++){
          for(int i2=0; i2<8; i2++){
            int i = i1 * 8 + i2;
            // weight_hw[o*3*3*IN_NUM+p*3*IN_NUM+q*IN_NUM+i] = 1;//weights[o][i][p][q];
            weights[o][i][p][q] = 1;
          }
        }
      }
    }
  }
  // for(int i=0; i<8*16*9; i++){
  //   weight_hw[i] = 1;
  // }
  nconv(inputs, weights, outputs_sw);
}

void postprocess(
  data_t0* cin_hw,
  data_t0  outputs_hw[OUT_NUM][OUT_H][OUT_W],
  data_t0  outputs_py[OUT_NUM][OUT_H][OUT_W]
){

  if(CHANGE_LAYOUT){
    for (int o1 = 0; o1 < OUT_NUM / OUT_NUM_T; o1++){
      for (int w1 = 0; w1 < OUT_W / IN_W_T; w1++){
        for (int h1 = 0; h1 < OUT_H / IN_H_T; h1++){
          for(int h2 = 0; h2 < IN_H_T; h2++){
            for(int w2 = 0; w2 < IN_W_T; w2++){
              for (int o2 = 0; o2 < OUT_NUM_T; o2++){
                int o = o1 * OUT_NUM_T + o2;
                int h = h1 * IN_H_T + h2;
                int w = w1 * IN_W_T + w2;
                
                int L1 = o1 * OUT_H * OUT_W * OUT_NUM_T;
                int L2 = w1 * OUT_H * IN_W_T * OUT_NUM_T;
                int L3 = h1 * IN_H_T * IN_W_T * OUT_NUM_T;
                int L4 = h2 * IN_W_T * OUT_NUM_T;
                int L5 = w2 * OUT_NUM_T;
                int L6 = o2;
                if (o < OUT_NUM){
                  outputs_hw[o][h][w] = cin_hw[OUT_OFFSET2 + L1 + L2 + L3 + L4 + L5 + L6 ];
                }
              }
            }
          }
        }
      }
    }
  }else{
    for (int o1 = 0; o1 < OUT_NUM / OUT_NUM_T; o1++)
      for (int h = 0; h < OUT_H_HW+1; h++){
        for (int w = 0; w < OUT_W_HW+1; w++){
          for (int o2 = 0; o2 < OUT_NUM_T; o2++){
          int o = o1 * OUT_NUM_T + o2;
          int padding = (IN_H_HW-IN_H)/2;
          int half_padding = padding/2;
          if (o < OUT_NUM && h>half_padding && w>half_padding && h<=(OUT_H+half_padding) && w<=(OUT_W+half_padding)){
            outputs_hw[o][h-padding][w-padding] = cin_hw[OUT_OFFSET1 + o1*OUT_H_HW*OUT_W_HW*OUT_NUM_T + h*OUT_W_HW*OUT_NUM_T + w*OUT_NUM_T + o2];
            // cout<<w-2<<endl;
          }
          }
        }
      }
  }
  // exit(0);
  // for(int i=1195088-131*16; i<130*130*16+(1195088-131*16); i++){
  //   cout<<cin_hw[i]<<endl;
  // }
  // for(int i=1195088; i<128*128*16+1195088; i++){
  //   cout<<cin_hw[i]<<endl;
  // }
  // exit(0);
  // int offset = 0;//1195088 - 32;
  // for (int o1 = 0; o1 < OUT_NUM / OUT_NUM_T; o1++){
  //   for (int w1 = 0; w1 < OUT_W / IN_W_T; w1++){
  //     for (int h1 = 0; h1 < OUT_H / IN_H_T; h1++){
  //       for(int h2 = 0; h2 < IN_H_T; h2++){
  //         for(int w2 = 0; w2 < IN_W_T; w2++){
  //           for (int o2 = 0; o2 < OUT_NUM_T; o2++){
  //             int o = o1 * OUT_NUM_T + o2;
  //             int h = h1 * IN_H_T + h2;
  //             int w = w1 * IN_W_T + w2;
              
  //             int L1 = o1 * OUT_H * OUT_W * OUT_NUM_T;
  //             int L2 = w1 * OUT_H * IN_W_T * OUT_NUM_T;
  //             int L3 = h1 * IN_H_T * IN_W_T * OUT_NUM_T;
  //             int L4 = h2 * IN_W_T * OUT_NUM_T;
  //             int L5 = w2 * OUT_NUM_T;
  //             int L6 = o2;
  //             int index = L1 + L2 + L3 + L4 + L5 + L6;
              
  //             if((index+offset)%(16*130)==0) offset += 32;
  //             if (o < OUT_NUM){
  //               outputs_hw[o][h][w] = cin_hw[1195088 - 32 + offset + index];
  //               cout<<"["<<o<<"]["<<h<<"]["<<w<<"] "<<" "<<offset + index<<endl;
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // exit(0);
  // for (int o1 = 0; o1 < OUT_NUM / OUT_NUM_T; o1++){
  //   for (int w1 = 0; w1 < OUT_W / IN_W_T; w1++){
  //     for (int h1 = 0; h1 < OUT_H / IN_H_T; h1++){
  //       for(int h2 = 0; h2 < IN_H_T; h2++){
  //         for(int w2 = 0; w2 < IN_W_T; w2++){
  //           for (int o2 = 0; o2 < OUT_NUM_T; o2++){
  //             int o = o1 * OUT_NUM_T + o2;
  //             int h = h1 * IN_H_T + h2;
  //             int w = w1 * IN_W_T + w2;
              
  //             int L1 = o1 * OUT_H * OUT_W * OUT_NUM_T;
  //             int L2 = w1 * OUT_H * IN_W_T * OUT_NUM_T;
  //             int L3 = h1 * IN_H_T * IN_W_T * OUT_NUM_T;
  //             int L4 = h2 * IN_W_T * OUT_NUM_T;
  //             int L5 = w2 * OUT_NUM_T;
  //             int L6 = o2;
  //             if (o < OUT_NUM){
  //               outputs_hw[o][h][w] = cin_hw[1195088 + L1 + L2 + L3 + L4 + L5 + L6 ];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  cout << "Loading outputs..." << endl;   
  string file_path = string(PRJ_PATH) + OUTFILE; 
  ifstream ouptut_file(file_path.c_str());
  if (ouptut_file.is_open()){

    int idx = 0;
    for (int o = 0; o < OUT_NUM; o++)
      for (int h = 0; h < OUT_H; h++)
        for (int w = 0; w < OUT_W; w++)
        {
          ouptut_file >> outputs_py[o][h][w];
          idx++;
        }

    ouptut_file.close();
  } else {
    cout << "Output open failed!" << endl;
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
        if(abs(outputs_hw[ch][h][w]-outputs_sw[ch][h][w])>0.1){
          flag = false;
          // cout<<outputs_hw[ch][h][w]<<" "<<outputs_sw[ch][h][w]<<" "<<ch<<" "<<h<<" "<<w<<endl;
          // cout<<abs(outputs_hw[ch][h][w]-outputs_sw[ch][h][w])<<endl;
          err_count++;
        }
      }
    }
  }
  if(flag)
    cout<<"SUCESS!"<<endl;
  else
    cout<<"FAILURE! "<<err_count<<" errors"<<endl;
  // exit(0);
}

void save_progress(data_t0* cin_hw, uint data_offset){

  char* prj_path_c = PRJ_PATH;
  cout << "saving mem..." << endl;   
  string file_path = string(prj_path_c) + "/data/mem.dat"; 
  ofstream mem_file(file_path.c_str());
  mem_file<<setprecision(16);
  if (mem_file.is_open()){
    for(int i=0; i<data_offset; i++)
    {
      mem_file << cin_hw[i] <<endl;
    }
    mem_file.close();
  } else {
    cout << "mem open failed!" << endl;
    exit(-1);
  }
}

void load_progress(data_t0* cin_hw){

  char* prj_path_c = PRJ_PATH;
  cout << "loading mem..." << endl;   
  string file_path = string(prj_path_c) + "/data/mem.dat"; 
  ifstream mem_file(file_path.c_str());
  if (mem_file.is_open()){
    for(int i=0; i<OUT_OFFSET2; i++)
    {
      mem_file >> cin_hw[i];
      // cout<<cin_hw[i]<<endl;
    }
    mem_file.close();
  } else {
    cout << "mem open failed!" << endl;
    exit(-1);
  }
}