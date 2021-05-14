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
#define AUTOTB_TVIN_fifo0_local_V "../tv/cdatafile/c.U1_compute.autotvin_fifo0_local_V.dat"
#define AUTOTB_TVOUT_fifo0_local_V "../tv/cdatafile/c.U1_compute.autotvout_fifo0_local_V.dat"
#define WRAPC_STREAM_SIZE_IN_fifo0_local_V "../tv/stream_size/stream_size_in_fifo0_local_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V "../tv/stream_size/stream_ingress_status_fifo0_local_V.dat"
// wrapc file define:
#define AUTOTB_TVIN_fifo1_local_V "../tv/cdatafile/c.U1_compute.autotvin_fifo1_local_V.dat"
#define AUTOTB_TVOUT_fifo1_local_V "../tv/cdatafile/c.U1_compute.autotvout_fifo1_local_V.dat"
#define WRAPC_STREAM_SIZE_IN_fifo1_local_V "../tv/stream_size/stream_size_in_fifo1_local_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V "../tv/stream_size/stream_ingress_status_fifo1_local_V.dat"
// wrapc file define:
#define AUTOTB_TVIN_fifo2_local_V "../tv/cdatafile/c.U1_compute.autotvin_fifo2_local_V.dat"
#define AUTOTB_TVOUT_fifo2_local_V "../tv/cdatafile/c.U1_compute.autotvout_fifo2_local_V.dat"
#define WRAPC_STREAM_SIZE_OUT_fifo2_local_V "../tv/stream_size/stream_size_out_fifo2_local_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_fifo2_local_V "../tv/stream_size/stream_egress_status_fifo2_local_V.dat"
// wrapc file define:
#define AUTOTB_TVIN_fifo_config_in_V "../tv/cdatafile/c.U1_compute.autotvin_fifo_config_in_V.dat"
#define AUTOTB_TVOUT_fifo_config_in_V "../tv/cdatafile/c.U1_compute.autotvout_fifo_config_in_V.dat"
#define WRAPC_STREAM_SIZE_IN_fifo_config_in_V "../tv/stream_size/stream_size_in_fifo_config_in_V.dat"
#define WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V "../tv/stream_size/stream_ingress_status_fifo_config_in_V.dat"
// wrapc file define:
#define AUTOTB_TVIN_fifo_config_out_V "../tv/cdatafile/c.U1_compute.autotvin_fifo_config_out_V.dat"
#define AUTOTB_TVOUT_fifo_config_out_V "../tv/cdatafile/c.U1_compute.autotvout_fifo_config_out_V.dat"
#define WRAPC_STREAM_SIZE_OUT_fifo_config_out_V "../tv/stream_size/stream_size_out_fifo_config_out_V.dat"
#define WRAPC_STREAM_EGRESS_STATUS_fifo_config_out_V "../tv/stream_size/stream_egress_status_fifo_config_out_V.dat"

#define INTER_TCL "../tv/cdatafile/ref.tcl"

// tvout file define:
#define AUTOTB_TVOUT_PC_fifo0_local_V "../tv/rtldatafile/rtl.U1_compute.autotvout_fifo0_local_V.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_fifo1_local_V "../tv/rtldatafile/rtl.U1_compute.autotvout_fifo1_local_V.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_fifo2_local_V "../tv/rtldatafile/rtl.U1_compute.autotvout_fifo2_local_V.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_fifo_config_in_V "../tv/rtldatafile/rtl.U1_compute.autotvout_fifo_config_in_V.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_fifo_config_out_V "../tv/rtldatafile/rtl.U1_compute.autotvout_fifo_config_out_V.dat"

class INTER_TCL_FILE {
  public:
INTER_TCL_FILE(const char* name) {
  mName = name; 
  fifo0_local_V_depth = 0;
  fifo1_local_V_depth = 0;
  fifo2_local_V_depth = 0;
  fifo_config_in_V_depth = 0;
  fifo_config_out_V_depth = 0;
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
  total_list << "{fifo0_local_V " << fifo0_local_V_depth << "}\n";
  total_list << "{fifo1_local_V " << fifo1_local_V_depth << "}\n";
  total_list << "{fifo2_local_V " << fifo2_local_V_depth << "}\n";
  total_list << "{fifo_config_in_V " << fifo_config_in_V_depth << "}\n";
  total_list << "{fifo_config_out_V " << fifo_config_out_V_depth << "}\n";
  return total_list.str();
}
void set_num (int num , int* class_num) {
  (*class_num) = (*class_num) > num ? (*class_num) : num;
}
void set_string(std::string list, std::string* class_list) {
  (*class_list) = list;
}
  public:
    int fifo0_local_V_depth;
    int fifo1_local_V_depth;
    int fifo2_local_V_depth;
    int fifo_config_in_V_depth;
    int fifo_config_out_V_depth;
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
struct __cosim_s4__ { char data[4]; };
extern "C" void U1_compute_hw_stub_wrapper(volatile void *, volatile void *, volatile void *, volatile void *, volatile void *);

extern "C" void apatb_U1_compute_hw(volatile void * __xlx_apatb_param_fifo0_local, volatile void * __xlx_apatb_param_fifo1_local, volatile void * __xlx_apatb_param_fifo2_local, volatile void * __xlx_apatb_param_fifo_config_in, volatile void * __xlx_apatb_param_fifo_config_out) {
  refine_signal_handler();
  fstream wrapc_switch_file_token;
  wrapc_switch_file_token.open(".hls_cosim_wrapc_switch.log");
  int AESL_i;
  if (wrapc_switch_file_token.good())
  {

    CodeState = ENTER_WRAPC_PC;
    static unsigned AESL_transaction_pc = 0;
    string AESL_token;
    string AESL_num;long __xlx_apatb_param_fifo0_local_V_stream_buf_final_size;
{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(WRAPC_STREAM_SIZE_IN_fifo0_local_V);
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
          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){__xlx_apatb_param_fifo0_local_V_stream_buf_final_size = atoi(AESL_token.c_str());

            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  for (long i = 0; i < __xlx_apatb_param_fifo0_local_V_stream_buf_final_size; ++i)((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->read();
long __xlx_apatb_param_fifo1_local_V_stream_buf_final_size;
{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(WRAPC_STREAM_SIZE_IN_fifo1_local_V);
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
          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){__xlx_apatb_param_fifo1_local_V_stream_buf_final_size = atoi(AESL_token.c_str());

            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  for (long i = 0; i < __xlx_apatb_param_fifo1_local_V_stream_buf_final_size; ++i)((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->read();
long __xlx_apatb_param_fifo2_local_V_stream_buf_final_size;
{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(WRAPC_STREAM_SIZE_OUT_fifo2_local_V);
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
          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){__xlx_apatb_param_fifo2_local_V_stream_buf_final_size = atoi(AESL_token.c_str());

            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  {
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(AUTOTB_TVOUT_PC_fifo2_local_V);
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
          std::vector<sc_bv<32> > fifo2_local_V_pc_buffer;
          int i = 0;

          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){

            RTLOutputCheckAndReplacement(AESL_token, "fifo2_local_V");
  
            // push token into output port buffer
            if (AESL_token != "") {
              fifo2_local_V_pc_buffer.push_back(AESL_token.c_str());
              i++;
            }
  
            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
          if (i > 0) {for (int j = 0, e = i; j != e; ++j) {
__cosim_s4__ xlx_stream_elt;

            ((int*)&xlx_stream_elt)[0] = fifo2_local_V_pc_buffer[j].to_int64();
          ((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->write(xlx_stream_elt);
}
}
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  long __xlx_apatb_param_fifo_config_in_V_stream_buf_final_size;
{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(WRAPC_STREAM_SIZE_IN_fifo_config_in_V);
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
          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){__xlx_apatb_param_fifo_config_in_V_stream_buf_final_size = atoi(AESL_token.c_str());

            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  for (long i = 0; i < __xlx_apatb_param_fifo_config_in_V_stream_buf_final_size; ++i)((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->read();
long __xlx_apatb_param_fifo_config_out_V_stream_buf_final_size;
{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(WRAPC_STREAM_SIZE_OUT_fifo_config_out_V);
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
          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){__xlx_apatb_param_fifo_config_out_V_stream_buf_final_size = atoi(AESL_token.c_str());

            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  {
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(AUTOTB_TVOUT_PC_fifo_config_out_V);
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
          std::vector<sc_bv<32> > fifo_config_out_V_pc_buffer;
          int i = 0;

          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){

            RTLOutputCheckAndReplacement(AESL_token, "fifo_config_out_V");
  
            // push token into output port buffer
            if (AESL_token != "") {
              fifo_config_out_V_pc_buffer.push_back(AESL_token.c_str());
              i++;
            }
  
            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
          if (i > 0) {for (int j = 0, e = i; j != e; ++j) {
int xlx_stream_elt;

            ((int*)&xlx_stream_elt)[0] = fifo_config_out_V_pc_buffer[j].to_int64();
          ((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->write(xlx_stream_elt);
}
}
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
//fifo0_local_V
aesl_fh.touch(AUTOTB_TVIN_fifo0_local_V);
aesl_fh.touch(AUTOTB_TVOUT_fifo0_local_V);
aesl_fh.touch(WRAPC_STREAM_SIZE_IN_fifo0_local_V);
aesl_fh.touch(WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V);
//fifo1_local_V
aesl_fh.touch(AUTOTB_TVIN_fifo1_local_V);
aesl_fh.touch(AUTOTB_TVOUT_fifo1_local_V);
aesl_fh.touch(WRAPC_STREAM_SIZE_IN_fifo1_local_V);
aesl_fh.touch(WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V);
//fifo2_local_V
aesl_fh.touch(AUTOTB_TVIN_fifo2_local_V);
aesl_fh.touch(AUTOTB_TVOUT_fifo2_local_V);
aesl_fh.touch(WRAPC_STREAM_SIZE_OUT_fifo2_local_V);
aesl_fh.touch(WRAPC_STREAM_EGRESS_STATUS_fifo2_local_V);
//fifo_config_in_V
aesl_fh.touch(AUTOTB_TVIN_fifo_config_in_V);
aesl_fh.touch(AUTOTB_TVOUT_fifo_config_in_V);
aesl_fh.touch(WRAPC_STREAM_SIZE_IN_fifo_config_in_V);
aesl_fh.touch(WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V);
//fifo_config_out_V
aesl_fh.touch(AUTOTB_TVIN_fifo_config_out_V);
aesl_fh.touch(AUTOTB_TVOUT_fifo_config_out_V);
aesl_fh.touch(WRAPC_STREAM_SIZE_OUT_fifo_config_out_V);
aesl_fh.touch(WRAPC_STREAM_EGRESS_STATUS_fifo_config_out_V);
CodeState = DUMP_INPUTS;
std::vector<__cosim_s40__> __xlx_apatb_param_fifo0_local_stream_buf;
{
  while (!((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->empty())
    __xlx_apatb_param_fifo0_local_stream_buf.push_back(((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->read());
  for (int i = 0; i < __xlx_apatb_param_fifo0_local_stream_buf.size(); ++i)
    ((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->write(__xlx_apatb_param_fifo0_local_stream_buf[i]);
  }
long __xlx_apatb_param_fifo0_local_stream_buf_size = ((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->size();
std::vector<__cosim_s40__> __xlx_apatb_param_fifo1_local_stream_buf;
{
  while (!((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->empty())
    __xlx_apatb_param_fifo1_local_stream_buf.push_back(((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->read());
  for (int i = 0; i < __xlx_apatb_param_fifo1_local_stream_buf.size(); ++i)
    ((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->write(__xlx_apatb_param_fifo1_local_stream_buf[i]);
  }
long __xlx_apatb_param_fifo1_local_stream_buf_size = ((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->size();
std::vector<__cosim_s4__> __xlx_apatb_param_fifo2_local_stream_buf;
long __xlx_apatb_param_fifo2_local_stream_buf_size = ((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->size();
std::vector<int> __xlx_apatb_param_fifo_config_in_stream_buf;
{
  while (!((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->empty())
    __xlx_apatb_param_fifo_config_in_stream_buf.push_back(((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->read());
  for (int i = 0; i < __xlx_apatb_param_fifo_config_in_stream_buf.size(); ++i)
    ((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->write(__xlx_apatb_param_fifo_config_in_stream_buf[i]);
  }
long __xlx_apatb_param_fifo_config_in_stream_buf_size = ((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->size();
std::vector<int> __xlx_apatb_param_fifo_config_out_stream_buf;
long __xlx_apatb_param_fifo_config_out_stream_buf_size = ((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->size();
CodeState = CALL_C_DUT;
U1_compute_hw_stub_wrapper(__xlx_apatb_param_fifo0_local, __xlx_apatb_param_fifo1_local, __xlx_apatb_param_fifo2_local, __xlx_apatb_param_fifo_config_in, __xlx_apatb_param_fifo_config_out);
CodeState = DUMP_OUTPUTS;
long __xlx_apatb_param_fifo0_local_stream_buf_final_size = __xlx_apatb_param_fifo0_local_stream_buf_size - ((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->size();
// print fifo0_local_V Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_fifo0_local_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo0_local_stream_buf_final_size; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv_hw;
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)&__xlx_apatb_param_fifo0_local_stream_buf[j])[0*8+7];
__xlx_tmp_lv_hw = __xlx_tmp_lv;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv_hw.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_fifo0_local_V, __xlx_sprintf_buffer.data()); 
  }

  tcl_file.set_num(__xlx_apatb_param_fifo0_local_stream_buf_final_size, &tcl_file.fifo0_local_V_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_fifo0_local_V, __xlx_sprintf_buffer.data());
}

// dump stream ingress status to file
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V, __xlx_sprintf_buffer.data());
  if (__xlx_apatb_param_fifo0_local_stream_buf_final_size > 0) {
  long fifo0_local_V_stream_ingress_size = __xlx_apatb_param_fifo0_local_stream_buf_size;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo0_local_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo0_local_stream_buf_final_size; j != e; j++) {
    fifo0_local_V_stream_ingress_size--;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo0_local_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V, __xlx_sprintf_buffer.data());
  }
} else {
  long fifo0_local_V_stream_ingress_size = 0;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo0_local_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V, __xlx_sprintf_buffer.data());
}

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo0_local_V, __xlx_sprintf_buffer.data());
}{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo0_local_V, __xlx_sprintf_buffer.data());
  sprintf(__xlx_sprintf_buffer.data(), "%d\n", __xlx_apatb_param_fifo0_local_stream_buf_final_size);
 aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo0_local_V, __xlx_sprintf_buffer.data());

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo0_local_V, __xlx_sprintf_buffer.data());
}long __xlx_apatb_param_fifo1_local_stream_buf_final_size = __xlx_apatb_param_fifo1_local_stream_buf_size - ((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->size();
// print fifo1_local_V Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_fifo1_local_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo1_local_stream_buf_final_size; j != e; ++j) {
sc_bv<512> __xlx_tmp_lv_hw;
sc_bv<512> __xlx_tmp_lv;
__xlx_tmp_lv.range(63,0) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+0];
__xlx_tmp_lv.range(127,64) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+1];
__xlx_tmp_lv.range(191,128) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+2];
__xlx_tmp_lv.range(255,192) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+3];
__xlx_tmp_lv.range(319,256) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+4];
__xlx_tmp_lv.range(383,320) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+5];
__xlx_tmp_lv.range(447,384) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+6];
__xlx_tmp_lv.range(511,448) = ((long long*)&__xlx_apatb_param_fifo1_local_stream_buf[j])[0*8+7];
__xlx_tmp_lv_hw = __xlx_tmp_lv;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv_hw.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_fifo1_local_V, __xlx_sprintf_buffer.data()); 
  }

  tcl_file.set_num(__xlx_apatb_param_fifo1_local_stream_buf_final_size, &tcl_file.fifo1_local_V_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_fifo1_local_V, __xlx_sprintf_buffer.data());
}

// dump stream ingress status to file
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V, __xlx_sprintf_buffer.data());
  if (__xlx_apatb_param_fifo1_local_stream_buf_final_size > 0) {
  long fifo1_local_V_stream_ingress_size = __xlx_apatb_param_fifo1_local_stream_buf_size;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo1_local_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo1_local_stream_buf_final_size; j != e; j++) {
    fifo1_local_V_stream_ingress_size--;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo1_local_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V, __xlx_sprintf_buffer.data());
  }
} else {
  long fifo1_local_V_stream_ingress_size = 0;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo1_local_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V, __xlx_sprintf_buffer.data());
}

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo1_local_V, __xlx_sprintf_buffer.data());
}{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo1_local_V, __xlx_sprintf_buffer.data());
  sprintf(__xlx_sprintf_buffer.data(), "%d\n", __xlx_apatb_param_fifo1_local_stream_buf_final_size);
 aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo1_local_V, __xlx_sprintf_buffer.data());

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo1_local_V, __xlx_sprintf_buffer.data());
}long __xlx_apatb_param_fifo2_local_stream_buf_final_size = ((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->size() - __xlx_apatb_param_fifo2_local_stream_buf_size;
{
  while (!((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->empty())
    __xlx_apatb_param_fifo2_local_stream_buf.push_back(((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->read());
  for (int i = 0; i < __xlx_apatb_param_fifo2_local_stream_buf.size(); ++i)
    ((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->write(__xlx_apatb_param_fifo2_local_stream_buf[i]);
  }
// print fifo2_local_V Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVOUT_fifo2_local_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo2_local_stream_buf_final_size; j != e; ++j) {
sc_bv<32> __xlx_tmp_lv = ((int*)&__xlx_apatb_param_fifo2_local_stream_buf[__xlx_apatb_param_fifo2_local_stream_buf_size+j])[0];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVOUT_fifo2_local_V, __xlx_sprintf_buffer.data()); 
  }

  tcl_file.set_num(__xlx_apatb_param_fifo2_local_stream_buf_final_size, &tcl_file.fifo2_local_V_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVOUT_fifo2_local_V, __xlx_sprintf_buffer.data());
}
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_SIZE_OUT_fifo2_local_V, __xlx_sprintf_buffer.data());
  sprintf(__xlx_sprintf_buffer.data(), "%d\n", __xlx_apatb_param_fifo2_local_stream_buf_final_size);
 aesl_fh.write(WRAPC_STREAM_SIZE_OUT_fifo2_local_V, __xlx_sprintf_buffer.data());

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_SIZE_OUT_fifo2_local_V, __xlx_sprintf_buffer.data());
}long __xlx_apatb_param_fifo_config_in_stream_buf_final_size = __xlx_apatb_param_fifo_config_in_stream_buf_size - ((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->size();
// print fifo_config_in_V Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_fifo_config_in_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo_config_in_stream_buf_final_size; j != e; ++j) {
sc_bv<32> __xlx_tmp_lv_hw;
sc_bv<32> __xlx_tmp_lv;
__xlx_tmp_lv = ((int*)&__xlx_apatb_param_fifo_config_in_stream_buf[j])[0];
__xlx_tmp_lv_hw = __xlx_tmp_lv;

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv_hw.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_fifo_config_in_V, __xlx_sprintf_buffer.data()); 
  }

  tcl_file.set_num(__xlx_apatb_param_fifo_config_in_stream_buf_final_size, &tcl_file.fifo_config_in_V_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_fifo_config_in_V, __xlx_sprintf_buffer.data());
}

// dump stream ingress status to file
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V, __xlx_sprintf_buffer.data());
  if (__xlx_apatb_param_fifo_config_in_stream_buf_final_size > 0) {
  long fifo_config_in_V_stream_ingress_size = __xlx_apatb_param_fifo_config_in_stream_buf_size;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo_config_in_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo_config_in_stream_buf_final_size; j != e; j++) {
    fifo_config_in_V_stream_ingress_size--;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo_config_in_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V, __xlx_sprintf_buffer.data());
  }
} else {
  long fifo_config_in_V_stream_ingress_size = 0;
sprintf(__xlx_sprintf_buffer.data(), "%d\n", fifo_config_in_V_stream_ingress_size);
 aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V, __xlx_sprintf_buffer.data());
}

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_INGRESS_STATUS_fifo_config_in_V, __xlx_sprintf_buffer.data());
}{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo_config_in_V, __xlx_sprintf_buffer.data());
  sprintf(__xlx_sprintf_buffer.data(), "%d\n", __xlx_apatb_param_fifo_config_in_stream_buf_final_size);
 aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo_config_in_V, __xlx_sprintf_buffer.data());

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_SIZE_IN_fifo_config_in_V, __xlx_sprintf_buffer.data());
}long __xlx_apatb_param_fifo_config_out_stream_buf_final_size = ((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->size() - __xlx_apatb_param_fifo_config_out_stream_buf_size;
{
  while (!((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->empty())
    __xlx_apatb_param_fifo_config_out_stream_buf.push_back(((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->read());
  for (int i = 0; i < __xlx_apatb_param_fifo_config_out_stream_buf.size(); ++i)
    ((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->write(__xlx_apatb_param_fifo_config_out_stream_buf[i]);
  }
// print fifo_config_out_V Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVOUT_fifo_config_out_V, __xlx_sprintf_buffer.data());
  for (int j = 0, e = __xlx_apatb_param_fifo_config_out_stream_buf_final_size; j != e; ++j) {
sc_bv<32> __xlx_tmp_lv = ((int*)&__xlx_apatb_param_fifo_config_out_stream_buf[__xlx_apatb_param_fifo_config_out_stream_buf_size+j])[0];

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVOUT_fifo_config_out_V, __xlx_sprintf_buffer.data()); 
  }

  tcl_file.set_num(__xlx_apatb_param_fifo_config_out_stream_buf_final_size, &tcl_file.fifo_config_out_V_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVOUT_fifo_config_out_V, __xlx_sprintf_buffer.data());
}
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(WRAPC_STREAM_SIZE_OUT_fifo_config_out_V, __xlx_sprintf_buffer.data());
  sprintf(__xlx_sprintf_buffer.data(), "%d\n", __xlx_apatb_param_fifo_config_out_stream_buf_final_size);
 aesl_fh.write(WRAPC_STREAM_SIZE_OUT_fifo_config_out_V, __xlx_sprintf_buffer.data());

  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(WRAPC_STREAM_SIZE_OUT_fifo_config_out_V, __xlx_sprintf_buffer.data());
}CodeState = DELETE_CHAR_BUFFERS;
AESL_transaction++;
tcl_file.set_num(AESL_transaction , &tcl_file.trans_num);
}
