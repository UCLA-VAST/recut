#include <systemc>
#include <vector>
#include <iostream>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"
using namespace std;
using namespace sc_dt;
class AESL_RUNTIME_BC {
  public:
    AESL_RUNTIME_BC(const char* name) {
      file_token.open( name);
      if (!file_token.good()) {
        cout << "Failed to open tv file " << name << endl;
        exit (1);
      }
      file_token >> mName;//[[[runtime]]]
    }
    ~AESL_RUNTIME_BC() {
      file_token.close();
    }
    int read_size () {
      int size = 0;
      file_token >> mName;//[[transaction]]
      file_token >> mName;//transaction number
      file_token >> mName;//pop_size
      size = atoi(mName.c_str());
      file_token >> mName;//[[/transaction]]
      return size;
    }
  public:
    fstream file_token;
    string mName;
};
unsigned int ap_apatb_fifo0_local_V_cap_bc;
static AESL_RUNTIME_BC __xlx_fifo0_local_V_size_Reader("../tv/stream_size/stream_size_in_fifo0_local_V.dat");
unsigned int ap_apatb_fifo1_local_V_cap_bc;
static AESL_RUNTIME_BC __xlx_fifo1_local_V_size_Reader("../tv/stream_size/stream_size_in_fifo1_local_V.dat");
unsigned int ap_apatb_fifo2_local_V_cap_bc;
static AESL_RUNTIME_BC __xlx_fifo2_local_V_size_Reader("../tv/stream_size/stream_size_out_fifo2_local_V.dat");
unsigned int ap_apatb_fifo_config_in_V_cap_bc;
static AESL_RUNTIME_BC __xlx_fifo_config_in_V_size_Reader("../tv/stream_size/stream_size_in_fifo_config_in_V.dat");
unsigned int ap_apatb_fifo_config_out_V_cap_bc;
static AESL_RUNTIME_BC __xlx_fifo_config_out_V_size_Reader("../tv/stream_size/stream_size_out_fifo_config_out_V.dat");
struct __cosim_s40__ { char data[64]; };
struct __cosim_s4__ { char data[4]; };
extern "C" void U1_compute(__cosim_s40__*, __cosim_s40__*, __cosim_s4__*, int*, int*);
extern "C" void apatb_U1_compute_hw(volatile void * __xlx_apatb_param_fifo0_local, volatile void * __xlx_apatb_param_fifo1_local, volatile void * __xlx_apatb_param_fifo2_local, volatile void * __xlx_apatb_param_fifo_config_in, volatile void * __xlx_apatb_param_fifo_config_out) {
  // collect __xlx_fifo0_local_tmp_vec
  unsigned __xlx_fifo0_local_V_tmp_Count = 0;
  unsigned __xlx_fifo0_local_V_read_Size = __xlx_fifo0_local_V_size_Reader.read_size();
  vector<__cosim_s40__> __xlx_fifo0_local_tmp_vec;
  while (!((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->empty() && __xlx_fifo0_local_V_tmp_Count < __xlx_fifo0_local_V_read_Size) {
    __xlx_fifo0_local_tmp_vec.push_back(((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo0_local)->read());
    __xlx_fifo0_local_V_tmp_Count++;
  }
  ap_apatb_fifo0_local_V_cap_bc = __xlx_fifo0_local_tmp_vec.size();
  // store input buffer
  __cosim_s40__* __xlx_fifo0_local_input_buffer= new __cosim_s40__[__xlx_fifo0_local_tmp_vec.size()];
  for (int i = 0; i < __xlx_fifo0_local_tmp_vec.size(); ++i) {
    __xlx_fifo0_local_input_buffer[i] = __xlx_fifo0_local_tmp_vec[i];
  }
  // collect __xlx_fifo1_local_tmp_vec
  unsigned __xlx_fifo1_local_V_tmp_Count = 0;
  unsigned __xlx_fifo1_local_V_read_Size = __xlx_fifo1_local_V_size_Reader.read_size();
  vector<__cosim_s40__> __xlx_fifo1_local_tmp_vec;
  while (!((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->empty() && __xlx_fifo1_local_V_tmp_Count < __xlx_fifo1_local_V_read_Size) {
    __xlx_fifo1_local_tmp_vec.push_back(((hls::stream<__cosim_s40__>*)__xlx_apatb_param_fifo1_local)->read());
    __xlx_fifo1_local_V_tmp_Count++;
  }
  ap_apatb_fifo1_local_V_cap_bc = __xlx_fifo1_local_tmp_vec.size();
  // store input buffer
  __cosim_s40__* __xlx_fifo1_local_input_buffer= new __cosim_s40__[__xlx_fifo1_local_tmp_vec.size()];
  for (int i = 0; i < __xlx_fifo1_local_tmp_vec.size(); ++i) {
    __xlx_fifo1_local_input_buffer[i] = __xlx_fifo1_local_tmp_vec[i];
  }
  //Create input buffer for fifo2_local
  ap_apatb_fifo2_local_V_cap_bc = __xlx_fifo2_local_V_size_Reader.read_size();
  __cosim_s4__* __xlx_fifo2_local_input_buffer= new __cosim_s4__[ap_apatb_fifo2_local_V_cap_bc];
  // collect __xlx_fifo_config_in_tmp_vec
  unsigned __xlx_fifo_config_in_V_tmp_Count = 0;
  unsigned __xlx_fifo_config_in_V_read_Size = __xlx_fifo_config_in_V_size_Reader.read_size();
  vector<int> __xlx_fifo_config_in_tmp_vec;
  while (!((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->empty() && __xlx_fifo_config_in_V_tmp_Count < __xlx_fifo_config_in_V_read_Size) {
    __xlx_fifo_config_in_tmp_vec.push_back(((hls::stream<int>*)__xlx_apatb_param_fifo_config_in)->read());
    __xlx_fifo_config_in_V_tmp_Count++;
  }
  ap_apatb_fifo_config_in_V_cap_bc = __xlx_fifo_config_in_tmp_vec.size();
  // store input buffer
  int* __xlx_fifo_config_in_input_buffer= new int[__xlx_fifo_config_in_tmp_vec.size()];
  for (int i = 0; i < __xlx_fifo_config_in_tmp_vec.size(); ++i) {
    __xlx_fifo_config_in_input_buffer[i] = __xlx_fifo_config_in_tmp_vec[i];
  }
  //Create input buffer for fifo_config_out
  ap_apatb_fifo_config_out_V_cap_bc = __xlx_fifo_config_out_V_size_Reader.read_size();
  int* __xlx_fifo_config_out_input_buffer= new int[ap_apatb_fifo_config_out_V_cap_bc];
  // DUT call
  U1_compute(__xlx_fifo0_local_input_buffer, __xlx_fifo1_local_input_buffer, __xlx_fifo2_local_input_buffer, __xlx_fifo_config_in_input_buffer, __xlx_fifo_config_out_input_buffer);
  for (unsigned i = 0; i <ap_apatb_fifo2_local_V_cap_bc; ++i)
    ((hls::stream<__cosim_s4__>*)__xlx_apatb_param_fifo2_local)->write(__xlx_fifo2_local_input_buffer[i]);
  for (unsigned i = 0; i <ap_apatb_fifo_config_out_V_cap_bc; ++i)
    ((hls::stream<int>*)__xlx_apatb_param_fifo_config_out)->write(__xlx_fifo_config_out_input_buffer[i]);
}
