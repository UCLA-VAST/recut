#include <systemc>
#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <stdint.h>
#include "SysCFileHandler.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include <complex>
#include <stdbool.h>
#include "autopilot_cbe.h"
#include "hls_stream.h"
#include "hls_half.h"
#include "hls_signal_handler.h"

using namespace std;
using namespace sc_core;
using namespace sc_dt;

// wrapc file define:
#define AUTOTB_TVIN_gmem1 "../tv/cdatafile/c.top_kernel.autotvin_gmem1.dat"
#define AUTOTB_TVOUT_gmem1 "../tv/cdatafile/c.top_kernel.autotvout_gmem1.dat"
// wrapc file define:
#define AUTOTB_TVIN_gmem3 "../tv/cdatafile/c.top_kernel.autotvin_gmem3.dat"
#define AUTOTB_TVOUT_gmem3 "../tv/cdatafile/c.top_kernel.autotvout_gmem3.dat"
// wrapc file define:
#define AUTOTB_TVIN_gmem2 "../tv/cdatafile/c.top_kernel.autotvin_gmem2.dat"
#define AUTOTB_TVOUT_gmem2 "../tv/cdatafile/c.top_kernel.autotvout_gmem2.dat"
// wrapc file define:
#define AUTOTB_TVIN_gcontrol "../tv/cdatafile/c.top_kernel.autotvin_gcontrol.dat"
#define AUTOTB_TVOUT_gcontrol "../tv/cdatafile/c.top_kernel.autotvout_gcontrol.dat"
// wrapc file define:
#define AUTOTB_TVIN_global_cin "../tv/cdatafile/c.top_kernel.autotvin_global_cin.dat"
#define AUTOTB_TVOUT_global_cin "../tv/cdatafile/c.top_kernel.autotvout_global_cin.dat"
// wrapc file define:
#define AUTOTB_TVIN_global_prev_cin "../tv/cdatafile/c.top_kernel.autotvin_global_prev_cin.dat"
#define AUTOTB_TVOUT_global_prev_cin "../tv/cdatafile/c.top_kernel.autotvout_global_prev_cin.dat"
// wrapc file define:
#define AUTOTB_TVIN_global_cout "../tv/cdatafile/c.top_kernel.autotvin_global_cout.dat"
#define AUTOTB_TVOUT_global_cout "../tv/cdatafile/c.top_kernel.autotvout_global_cout.dat"
// wrapc file define:
#define AUTOTB_TVIN_global_weight "../tv/cdatafile/c.top_kernel.autotvin_global_weight.dat"
#define AUTOTB_TVOUT_global_weight "../tv/cdatafile/c.top_kernel.autotvout_global_weight.dat"
// wrapc file define:
#define AUTOTB_TVIN_global_bias "../tv/cdatafile/c.top_kernel.autotvin_global_bias.dat"
#define AUTOTB_TVOUT_global_bias "../tv/cdatafile/c.top_kernel.autotvout_global_bias.dat"
// wrapc file define:
#define AUTOTB_TVIN_layer_config "../tv/cdatafile/c.top_kernel.autotvin_layer_config.dat"
#define AUTOTB_TVOUT_layer_config "../tv/cdatafile/c.top_kernel.autotvout_layer_config.dat"

#define INTER_TCL "../tv/cdatafile/ref.tcl"

// tvout file define:
#define AUTOTB_TVOUT_PC_gmem1 "../tv/rtldatafile/rtl.top_kernel.autotvout_gmem1.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_gmem3 "../tv/rtldatafile/rtl.top_kernel.autotvout_gmem3.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_gmem2 "../tv/rtldatafile/rtl.top_kernel.autotvout_gmem2.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_gcontrol "../tv/rtldatafile/rtl.top_kernel.autotvout_gcontrol.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_global_cin "../tv/rtldatafile/rtl.top_kernel.autotvout_global_cin.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_global_prev_cin "../tv/rtldatafile/rtl.top_kernel.autotvout_global_prev_cin.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_global_cout "../tv/rtldatafile/rtl.top_kernel.autotvout_global_cout.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_global_weight "../tv/rtldatafile/rtl.top_kernel.autotvout_global_weight.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_global_bias "../tv/rtldatafile/rtl.top_kernel.autotvout_global_bias.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_layer_config "../tv/rtldatafile/rtl.top_kernel.autotvout_layer_config.dat"

class INTER_TCL_FILE {
  public:
INTER_TCL_FILE(const char* name) {
  mName = name; 
  gmem1_depth = 0;
  gmem3_depth = 0;
  gmem2_depth = 0;
  gcontrol_depth = 0;
  global_cin_depth = 0;
  global_prev_cin_depth = 0;
  global_cout_depth = 0;
  global_weight_depth = 0;
  global_bias_depth = 0;
  layer_config_depth = 0;
  trans_num =0;
}
~INTER_TCL_FILE() {
  mFile.open(mName);
  if (!mFile.good()) {
    cout << "Failed to open file ref.tcl" << endl;
    exit (1); 
  }
  string total_list = get_depth_list();
  mFile << "set depth_list {\n";
  mFile << total_list;
  mFile << "}\n";
  mFile << "set trans_num "<<trans_num<<endl;
  mFile.close();
}
string get_depth_list () {
  stringstream total_list;
  total_list << "{gmem1 " << gmem1_depth << "}\n";
  total_list << "{gmem3 " << gmem3_depth << "}\n";
  total_list << "{gmem2 " << gmem2_depth << "}\n";
  total_list << "{gcontrol " << gcontrol_depth << "}\n";
  total_list << "{global_cin " << global_cin_depth << "}\n";
  total_list << "{global_prev_cin " << global_prev_cin_depth << "}\n";
  total_list << "{global_cout " << global_cout_depth << "}\n";
  total_list << "{global_weight " << global_weight_depth << "}\n";
  total_list << "{global_bias " << global_bias_depth << "}\n";
  total_list << "{layer_config " << layer_config_depth << "}\n";
  return total_list.str();
}
void set_num (int num , int* class_num) {
  (*class_num) = (*class_num) > num ? (*class_num) : num;
}
void set_string(std::string list, std::string* class_list) {
  (*class_list) = list;
}
  public:
    int gmem1_depth;
    int gmem3_depth;
    int gmem2_depth;
    int gcontrol_depth;
    int global_cin_depth;
    int global_prev_cin_depth;
    int global_cout_depth;
    int global_weight_depth;
    int global_bias_depth;
    int layer_config_depth;
    int trans_num;
  private:
    ofstream mFile;
    const char* mName;
};

static void RTLOutputCheckAndReplacement(std::string &AESL_token, std::string PortName) {
  bool no_x = false;
  bool err = false;

  no_x = false;
  // search and replace 'X' with '0' from the 3rd char of token
  while (!no_x) {
    size_t x_found = AESL_token.find('X', 0);
    if (x_found != string::npos) {
      if (!err) { 
        cerr << "WARNING: [SIM 212-201] RTL produces unknown value 'X' on port" 
             << PortName << ", possible cause: There are uninitialized variables in the C design."
             << endl; 
        err = true;
      }
      AESL_token.replace(x_found, 1, "0");
    } else
      no_x = true;
  }
  no_x = false;
  // search and replace 'x' with '0' from the 3rd char of token
  while (!no_x) {
    size_t x_found = AESL_token.find('x', 2);
    if (x_found != string::npos) {
      if (!err) { 
        cerr << "WARNING: [SIM 212-201] RTL produces unknown value 'x' on port" 
             << PortName << ", possible cause: There are uninitialized variables in the C design."
             << endl; 
        err = true;
      }
      AESL_token.replace(x_found, 1, "0");
    } else
      no_x = true;
  }
}
struct __cosim_s40__ { char data[64]; };
extern "C" void top_kernel_hw_stub_wrapper(volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *);

extern "C" void apatb_top_kernel_hw(volatile void * __xlx_apatb_param_global_cin, volatile void * __xlx_apatb_param_global_prev_cin, volatile void * __xlx_apatb_param_global_cout, volatile void * __xlx_apatb_param_global_weight, volatile void * __xlx_apatb_param_global_bias, volatile void * __xlx_apatb_param_layer_config) {
  refine_signal_handler();
  fstream wrapc_switch_file_token;
  wrapc_switch_file_token.open(".hls_cosim_wrapc_switch.log");
  int AESL_i;
  if (wrapc_switch_file_token.good())
  {

    CodeState = ENTER_WRAPC_PC;
    static unsigned AESL_transaction_pc = 0;
    string AESL_token;
    string AESL_num;{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(AUTOTB_TVOUT_PC_gmem1);
        if (rtl_tv_out_file.good()) {
          rtl_tv_out_file >> AESL_token;
          if (AESL_token != "[[[runtime]]]")
            exit(1);
        }
      }
  
      if (rtl_tv_out_file.good()) {
        rtl_tv_out_file >> AESL_token; 
        rtl_tv_out_file >> AESL_num;  // transaction number
        if (AESL_token != "[[transaction]]") {
          cerr << "Unexpected token: " << AESL_token << endl;
          exit(1);
        }
        if (atoi(AESL_num.c_str()) == AESL_transaction_pc) {
          std::vector<sc_bv<512> > gmem1_pc_buffer(826275);
          int i = 0;

          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){

            RTLOutputCheckAndReplacement(AESL_token, "gmem1");
  
            // push token into output port buffer
            if (AESL_token != "") {
              gmem1_pc_buffer[i] = AESL_token.c_str();;
              i++;
            }
  
            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
          if (i > 0) {{
            int i = 0;
            for (int j = 0, e = 1; j < e; j += 1, ++i) {((long long*)__xlx_apatb_param_global_cin)[j*8+0] = gmem1_pc_buffer[i].range(63,0).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+1] = gmem1_pc_buffer[i].range(127,64).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+2] = gmem1_pc_buffer[i].range(191,128).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+3] = gmem1_pc_buffer[i].range(255,192).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+4] = gmem1_pc_buffer[i].range(319,256).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+5] = gmem1_pc_buffer[i].range(383,320).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+6] = gmem1_pc_buffer[i].range(447,384).to_int64();
((long long*)__xlx_apatb_param_global_cin)[j*8+7] = gmem1_pc_buffer[i].range(511,448).to_int64();
}
            for (int j = 0, e = 826274; j < e; j += 1, ++i) {((long long*)__xlx_apatb_param_global_cout)[j*8+0] = gmem1_pc_buffer[i].range(63,0).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+1] = gmem1_pc_buffer[i].range(127,64).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+2] = gmem1_pc_buffer[i].range(191,128).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+3] = gmem1_pc_buffer[i].range(255,192).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+4] = gmem1_pc_buffer[i].range(319,256).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+5] = gmem1_pc_buffer[i].range(383,320).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+6] = gmem1_pc_buffer[i].range(447,384).to_int64();
((long long*)__xlx_apatb_param_global_cout)[j*8+7] = gmem1_pc_buffer[i].range(511,448).to_int64();
}}}
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  
    AESL_transaction_pc++;
    return ;
  }
static unsigned AESL_transaction;
static AESL_FILE_HANDLER aesl_fh;
static INTER_TCL_FILE tcl_file(INTER_TCL);
std::vector<char> __xlx_sprintf_buffer(1024);
CodeState = ENTER_WRAPC;
//gmem1
aesl_fh.touch(AUTOTB_TVIN_gmem1);
aesl_fh.touch(AUTOTB_TVOUT_gmem1);
//gmem3
aesl_fh.touch(AUTOTB_TVIN_gmem3);
aesl_fh.touch(AUTOTB_TVOUT_gmem3);
//gmem2
aesl_fh.touch(AUTOTB_TVIN_gmem2);
aesl_fh.touch(AUTOTB_TVOUT_gmem2);
//gcontrol
aesl_fh.touch(AUTOTB_TVIN_gcontrol);
aesl_fh.touch(AUTOTB_TVOUT_gcontrol);
//global_cin
aesl_fh.touch(AUTOTB_TVIN_global_cin);
aesl_fh.touch(AUTOTB_TVOUT_global_cin);
//global_prev_cin
aesl_fh.touch(AUTOTB_TVIN_global_prev_cin);
aesl_fh.touch(AUTOTB_TVOUT_global_prev_cin);
//global_cout
aesl_fh.touch(AUTOTB_TVIN_global_cout);
aesl_fh.touch(AUTOTB_TVOUT_global_cout);
//global_weight
aesl_fh.touch(AUTOTB_TVIN_global_weight);
aesl_fh.touch(AUTOTB_TVOUT_global_weight);
//global_bias
aesl_fh.touch(AUTOTB_TVIN_global_bias);
aesl_fh.touch(AUTOTB_TVOUT_global_bias);
//layer_config
aesl_fh.touch(AUTOTB_TVIN_layer_config);
aesl_fh.touch(AUTOTB_TVOUT_layer_config);
CodeState = DUMP_INPUTS;
unsigned __xlx_offset_byte_param_global_cin = 0;
unsigned __xlx_offset_byte_param_global_cout = 0;
// print gmem1 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_gmem1, __xlx_sprintf_buffer.data());
  {  __xlx_offset_byte_param_global_cin = 0*64;
  if (__xlx_apatb_param_global_cin) {
    for (int j = 0  - 0, e = 1 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_cin)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_cin)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_cin)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_cin)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_cin)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_cin)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_cin)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_cin)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_gmem1, __xlx_sprintf_buffer.data()); 
      }
  }
  __xlx_offset_byte_param_global_cout = 1*64;
  if (__xlx_apatb_param_global_cout) {
    for (int j = 0  - 0, e = 826274 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_cout)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_cout)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_cout)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_cout)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_cout)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_cout)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_cout)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_cout)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_gmem1, __xlx_sprintf_buffer.data()); 
      }
  }
}
  tcl_file.set_num(826275, &tcl_file.gmem1_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_gmem1, __xlx_sprintf_buffer.data());
}
unsigned __xlx_offset_byte_param_global_prev_cin = 0;
// print gmem3 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_gmem3, __xlx_sprintf_buffer.data());
  {  __xlx_offset_byte_param_global_prev_cin = 0*64;
  if (__xlx_apatb_param_global_prev_cin) {
    for (int j = 0  - 0, e = 1 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_gmem3, __xlx_sprintf_buffer.data()); 
      }
  }
}
  tcl_file.set_num(1, &tcl_file.gmem3_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_gmem3, __xlx_sprintf_buffer.data());
}
unsigned __xlx_offset_byte_param_global_weight = 0;
unsigned __xlx_offset_byte_param_global_bias = 0;
// print gmem2 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_gmem2, __xlx_sprintf_buffer.data());
  {  __xlx_offset_byte_param_global_weight = 0*64;
  if (__xlx_apatb_param_global_weight) {
    for (int j = 0  - 0, e = 34234 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_weight)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_weight)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_weight)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_weight)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_weight)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_weight)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_weight)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_weight)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_gmem2, __xlx_sprintf_buffer.data()); 
      }
  }
  __xlx_offset_byte_param_global_bias = 34234*64;
  if (__xlx_apatb_param_global_bias) {
    for (int j = 0  - 0, e = 1026 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_bias)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_bias)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_bias)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_bias)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_bias)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_bias)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_bias)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_bias)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_gmem2, __xlx_sprintf_buffer.data()); 
      }
  }
}
  tcl_file.set_num(35260, &tcl_file.gmem2_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_gmem2, __xlx_sprintf_buffer.data());
}
unsigned __xlx_offset_byte_param_layer_config = 0;
// print gcontrol Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_gcontrol, __xlx_sprintf_buffer.data());
  {  __xlx_offset_byte_param_layer_config = 0*4;
  if (__xlx_apatb_param_layer_config) {
    for (int j = 0  - 0, e = 2815 - 0; j != e; ++j) {
sc_bv<32> __xlx_tmp_lv = ((int*)__xlx_apatb_param_layer_config)[j];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_gcontrol, __xlx_sprintf_buffer.data()); 
      }
  }
}
  tcl_file.set_num(2815, &tcl_file.gcontrol_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_gcontrol, __xlx_sprintf_buffer.data());
}
// print global_cin Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_global_cin, __xlx_sprintf_buffer.data());
  {
    sc_bv<64> __xlx_tmp_lv = __xlx_offset_byte_param_global_cin;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_global_cin, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.global_cin_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_global_cin, __xlx_sprintf_buffer.data());
}
// print global_prev_cin Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_global_prev_cin, __xlx_sprintf_buffer.data());
  {
    sc_bv<64> __xlx_tmp_lv = __xlx_offset_byte_param_global_prev_cin;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_global_prev_cin, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.global_prev_cin_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_global_prev_cin, __xlx_sprintf_buffer.data());
}
// print global_cout Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_global_cout, __xlx_sprintf_buffer.data());
  {
    sc_bv<64> __xlx_tmp_lv = __xlx_offset_byte_param_global_cout;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_global_cout, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.global_cout_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_global_cout, __xlx_sprintf_buffer.data());
}
// print global_weight Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_global_weight, __xlx_sprintf_buffer.data());
  {
    sc_bv<64> __xlx_tmp_lv = __xlx_offset_byte_param_global_weight;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_global_weight, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.global_weight_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_global_weight, __xlx_sprintf_buffer.data());
}
// print global_bias Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_global_bias, __xlx_sprintf_buffer.data());
  {
    sc_bv<64> __xlx_tmp_lv = __xlx_offset_byte_param_global_bias;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_global_bias, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.global_bias_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_global_bias, __xlx_sprintf_buffer.data());
}
// print layer_config Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_layer_config, __xlx_sprintf_buffer.data());
  {
    sc_bv<64> __xlx_tmp_lv = __xlx_offset_byte_param_layer_config;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_layer_config, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.layer_config_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_layer_config, __xlx_sprintf_buffer.data());
}
CodeState = CALL_C_DUT;
top_kernel_hw_stub_wrapper(__xlx_apatb_param_global_cin, __xlx_apatb_param_global_prev_cin, __xlx_apatb_param_global_cout, __xlx_apatb_param_global_weight, __xlx_apatb_param_global_bias, __xlx_apatb_param_layer_config);
CodeState = DUMP_OUTPUTS;
// print gmem1 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVOUT_gmem1, __xlx_sprintf_buffer.data());
  {  __xlx_offset_byte_param_global_cin = 0*64;
  if (__xlx_apatb_param_global_cin) {
    for (int j = 0  - 0, e = 1 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_cin)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_cin)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_cin)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_cin)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_cin)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_cin)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_cin)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_cin)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVOUT_gmem1, __xlx_sprintf_buffer.data()); 
      }
  }
  __xlx_offset_byte_param_global_cout = 1*64;
  if (__xlx_apatb_param_global_cout) {
    for (int j = 0  - 0, e = 826274 - 0; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)__xlx_apatb_param_global_cout)[j*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)__xlx_apatb_param_global_cout)[j*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)__xlx_apatb_param_global_cout)[j*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)__xlx_apatb_param_global_cout)[j*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)__xlx_apatb_param_global_cout)[j*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)__xlx_apatb_param_global_cout)[j*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)__xlx_apatb_param_global_cout)[j*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)__xlx_apatb_param_global_cout)[j*8+7];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVOUT_gmem1, __xlx_sprintf_buffer.data()); 
      }
  }
}
  tcl_file.set_num(826275, &tcl_file.gmem1_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVOUT_gmem1, __xlx_sprintf_buffer.data());
}
CodeState = DELETE_CHAR_BUFFERS;
AESL_transaction++;
tcl_file.set_num(AESL_transaction , &tcl_file.trans_num);
}
