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
struct __cosim_s40__ { char data[64]; };
extern "C" void top_kernel(__cosim_s40__*, __cosim_s40__*, __cosim_s40__*, int*, int, int, int, int, int, int);
extern "C" void apatb_top_kernel_hw(volatile void * __xlx_apatb_param_global_cin, volatile void * __xlx_apatb_param_global_prev_cin, volatile void * __xlx_apatb_param_global_cout, volatile void * __xlx_apatb_param_global_weight, volatile void * __xlx_apatb_param_global_bias, volatile void * __xlx_apatb_param_layer_config) {
  // Collect __xlx_global_cin_global_cout__tmp_vec
  vector<sc_bv<512> >__xlx_global_cin_global_cout__tmp_vec;
  for (int j = 0, e = 1; j != e; ++j) {
    sc_bv<512> _xlx_tmp_sc;
    _xlx_tmp_sc.range(63, 0) = ((long long*)__xlx_apatb_param_global_cin)[j*8+0];
    _xlx_tmp_sc.range(127, 64) = ((long long*)__xlx_apatb_param_global_cin)[j*8+1];
    _xlx_tmp_sc.range(191, 128) = ((long long*)__xlx_apatb_param_global_cin)[j*8+2];
    _xlx_tmp_sc.range(255, 192) = ((long long*)__xlx_apatb_param_global_cin)[j*8+3];
    _xlx_tmp_sc.range(319, 256) = ((long long*)__xlx_apatb_param_global_cin)[j*8+4];
    _xlx_tmp_sc.range(383, 320) = ((long long*)__xlx_apatb_param_global_cin)[j*8+5];
    _xlx_tmp_sc.range(447, 384) = ((long long*)__xlx_apatb_param_global_cin)[j*8+6];
    _xlx_tmp_sc.range(511, 448) = ((long long*)__xlx_apatb_param_global_cin)[j*8+7];
    __xlx_global_cin_global_cout__tmp_vec.push_back(_xlx_tmp_sc);
  }
  int __xlx_size_param_global_cin = 1;
  int __xlx_offset_param_global_cin = 0;
  int __xlx_offset_byte_param_global_cin = 0*64;
  for (int j = 0, e = 826274; j != e; ++j) {
    sc_bv<512> _xlx_tmp_sc;
    _xlx_tmp_sc.range(63, 0) = ((long long*)__xlx_apatb_param_global_cout)[j*8+0];
    _xlx_tmp_sc.range(127, 64) = ((long long*)__xlx_apatb_param_global_cout)[j*8+1];
    _xlx_tmp_sc.range(191, 128) = ((long long*)__xlx_apatb_param_global_cout)[j*8+2];
    _xlx_tmp_sc.range(255, 192) = ((long long*)__xlx_apatb_param_global_cout)[j*8+3];
    _xlx_tmp_sc.range(319, 256) = ((long long*)__xlx_apatb_param_global_cout)[j*8+4];
    _xlx_tmp_sc.range(383, 320) = ((long long*)__xlx_apatb_param_global_cout)[j*8+5];
    _xlx_tmp_sc.range(447, 384) = ((long long*)__xlx_apatb_param_global_cout)[j*8+6];
    _xlx_tmp_sc.range(511, 448) = ((long long*)__xlx_apatb_param_global_cout)[j*8+7];
    __xlx_global_cin_global_cout__tmp_vec.push_back(_xlx_tmp_sc);
  }
  int __xlx_size_param_global_cout = 826274;
  int __xlx_offset_param_global_cout = 1;
  int __xlx_offset_byte_param_global_cout = 1*64;
  __cosim_s40__* __xlx_global_cin_global_cout__input_buffer= new __cosim_s40__[__xlx_global_cin_global_cout__tmp_vec.size()];
  for (int i = 0; i < __xlx_global_cin_global_cout__tmp_vec.size(); ++i) {
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+0] = __xlx_global_cin_global_cout__tmp_vec[i].range(63, 0).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+1] = __xlx_global_cin_global_cout__tmp_vec[i].range(127, 64).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+2] = __xlx_global_cin_global_cout__tmp_vec[i].range(191, 128).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+3] = __xlx_global_cin_global_cout__tmp_vec[i].range(255, 192).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+4] = __xlx_global_cin_global_cout__tmp_vec[i].range(319, 256).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+5] = __xlx_global_cin_global_cout__tmp_vec[i].range(383, 320).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+6] = __xlx_global_cin_global_cout__tmp_vec[i].range(447, 384).to_uint64();
    ((long long*)__xlx_global_cin_global_cout__input_buffer)[i*8+7] = __xlx_global_cin_global_cout__tmp_vec[i].range(511, 448).to_uint64();
  }
  // Collect __xlx_global_prev_cin__tmp_vec
  vector<sc_bv<512> >__xlx_global_prev_cin__tmp_vec;
  for (int j = 0, e = 1; j != e; ++j) {
    sc_bv<512> _xlx_tmp_sc;
    _xlx_tmp_sc.range(63, 0) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+0];
    _xlx_tmp_sc.range(127, 64) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+1];
    _xlx_tmp_sc.range(191, 128) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+2];
    _xlx_tmp_sc.range(255, 192) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+3];
    _xlx_tmp_sc.range(319, 256) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+4];
    _xlx_tmp_sc.range(383, 320) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+5];
    _xlx_tmp_sc.range(447, 384) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+6];
    _xlx_tmp_sc.range(511, 448) = ((long long*)__xlx_apatb_param_global_prev_cin)[j*8+7];
    __xlx_global_prev_cin__tmp_vec.push_back(_xlx_tmp_sc);
  }
  int __xlx_size_param_global_prev_cin = 1;
  int __xlx_offset_param_global_prev_cin = 0;
  int __xlx_offset_byte_param_global_prev_cin = 0*64;
  __cosim_s40__* __xlx_global_prev_cin__input_buffer= new __cosim_s40__[__xlx_global_prev_cin__tmp_vec.size()];
  for (int i = 0; i < __xlx_global_prev_cin__tmp_vec.size(); ++i) {
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+0] = __xlx_global_prev_cin__tmp_vec[i].range(63, 0).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+1] = __xlx_global_prev_cin__tmp_vec[i].range(127, 64).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+2] = __xlx_global_prev_cin__tmp_vec[i].range(191, 128).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+3] = __xlx_global_prev_cin__tmp_vec[i].range(255, 192).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+4] = __xlx_global_prev_cin__tmp_vec[i].range(319, 256).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+5] = __xlx_global_prev_cin__tmp_vec[i].range(383, 320).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+6] = __xlx_global_prev_cin__tmp_vec[i].range(447, 384).to_uint64();
    ((long long*)__xlx_global_prev_cin__input_buffer)[i*8+7] = __xlx_global_prev_cin__tmp_vec[i].range(511, 448).to_uint64();
  }
  // Collect __xlx_global_weight_global_bias__tmp_vec
  vector<sc_bv<512> >__xlx_global_weight_global_bias__tmp_vec;
  for (int j = 0, e = 34234; j != e; ++j) {
    sc_bv<512> _xlx_tmp_sc;
    _xlx_tmp_sc.range(63, 0) = ((long long*)__xlx_apatb_param_global_weight)[j*8+0];
    _xlx_tmp_sc.range(127, 64) = ((long long*)__xlx_apatb_param_global_weight)[j*8+1];
    _xlx_tmp_sc.range(191, 128) = ((long long*)__xlx_apatb_param_global_weight)[j*8+2];
    _xlx_tmp_sc.range(255, 192) = ((long long*)__xlx_apatb_param_global_weight)[j*8+3];
    _xlx_tmp_sc.range(319, 256) = ((long long*)__xlx_apatb_param_global_weight)[j*8+4];
    _xlx_tmp_sc.range(383, 320) = ((long long*)__xlx_apatb_param_global_weight)[j*8+5];
    _xlx_tmp_sc.range(447, 384) = ((long long*)__xlx_apatb_param_global_weight)[j*8+6];
    _xlx_tmp_sc.range(511, 448) = ((long long*)__xlx_apatb_param_global_weight)[j*8+7];
    __xlx_global_weight_global_bias__tmp_vec.push_back(_xlx_tmp_sc);
  }
  int __xlx_size_param_global_weight = 34234;
  int __xlx_offset_param_global_weight = 0;
  int __xlx_offset_byte_param_global_weight = 0*64;
  for (int j = 0, e = 1026; j != e; ++j) {
    sc_bv<512> _xlx_tmp_sc;
    _xlx_tmp_sc.range(63, 0) = ((long long*)__xlx_apatb_param_global_bias)[j*8+0];
    _xlx_tmp_sc.range(127, 64) = ((long long*)__xlx_apatb_param_global_bias)[j*8+1];
    _xlx_tmp_sc.range(191, 128) = ((long long*)__xlx_apatb_param_global_bias)[j*8+2];
    _xlx_tmp_sc.range(255, 192) = ((long long*)__xlx_apatb_param_global_bias)[j*8+3];
    _xlx_tmp_sc.range(319, 256) = ((long long*)__xlx_apatb_param_global_bias)[j*8+4];
    _xlx_tmp_sc.range(383, 320) = ((long long*)__xlx_apatb_param_global_bias)[j*8+5];
    _xlx_tmp_sc.range(447, 384) = ((long long*)__xlx_apatb_param_global_bias)[j*8+6];
    _xlx_tmp_sc.range(511, 448) = ((long long*)__xlx_apatb_param_global_bias)[j*8+7];
    __xlx_global_weight_global_bias__tmp_vec.push_back(_xlx_tmp_sc);
  }
  int __xlx_size_param_global_bias = 1026;
  int __xlx_offset_param_global_bias = 34234;
  int __xlx_offset_byte_param_global_bias = 34234*64;
  __cosim_s40__* __xlx_global_weight_global_bias__input_buffer= new __cosim_s40__[__xlx_global_weight_global_bias__tmp_vec.size()];
  for (int i = 0; i < __xlx_global_weight_global_bias__tmp_vec.size(); ++i) {
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+0] = __xlx_global_weight_global_bias__tmp_vec[i].range(63, 0).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+1] = __xlx_global_weight_global_bias__tmp_vec[i].range(127, 64).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+2] = __xlx_global_weight_global_bias__tmp_vec[i].range(191, 128).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+3] = __xlx_global_weight_global_bias__tmp_vec[i].range(255, 192).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+4] = __xlx_global_weight_global_bias__tmp_vec[i].range(319, 256).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+5] = __xlx_global_weight_global_bias__tmp_vec[i].range(383, 320).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+6] = __xlx_global_weight_global_bias__tmp_vec[i].range(447, 384).to_uint64();
    ((long long*)__xlx_global_weight_global_bias__input_buffer)[i*8+7] = __xlx_global_weight_global_bias__tmp_vec[i].range(511, 448).to_uint64();
  }
  // Collect __xlx_layer_config__tmp_vec
  vector<sc_bv<32> >__xlx_layer_config__tmp_vec;
  for (int j = 0, e = 2815; j != e; ++j) {
    __xlx_layer_config__tmp_vec.push_back(((int*)__xlx_apatb_param_layer_config)[j]);
  }
  int __xlx_size_param_layer_config = 2815;
  int __xlx_offset_param_layer_config = 0;
  int __xlx_offset_byte_param_layer_config = 0*4;
  int* __xlx_layer_config__input_buffer= new int[__xlx_layer_config__tmp_vec.size()];
  for (int i = 0; i < __xlx_layer_config__tmp_vec.size(); ++i) {
    __xlx_layer_config__input_buffer[i] = __xlx_layer_config__tmp_vec[i].range(31, 0).to_uint64();
  }
  // DUT call
  top_kernel(__xlx_global_cin_global_cout__input_buffer, __xlx_global_prev_cin__input_buffer, __xlx_global_weight_global_bias__input_buffer, __xlx_layer_config__input_buffer, __xlx_offset_byte_param_global_cin, __xlx_offset_byte_param_global_prev_cin, __xlx_offset_byte_param_global_cout, __xlx_offset_byte_param_global_weight, __xlx_offset_byte_param_global_bias, __xlx_offset_byte_param_layer_config);
// print __xlx_apatb_param_global_cin
  sc_bv<512>*__xlx_global_cin_output_buffer = new sc_bv<512>[__xlx_size_param_global_cin];
  for (int i = 0; i < __xlx_size_param_global_cin; ++i) {
    char* start = (char*)(&(__xlx_global_cin_global_cout__input_buffer[__xlx_offset_param_global_cin]));
    __xlx_global_cin_output_buffer[i].range(63, 0) = ((long long*)start)[i*8+0];
    __xlx_global_cin_output_buffer[i].range(127, 64) = ((long long*)start)[i*8+1];
    __xlx_global_cin_output_buffer[i].range(191, 128) = ((long long*)start)[i*8+2];
    __xlx_global_cin_output_buffer[i].range(255, 192) = ((long long*)start)[i*8+3];
    __xlx_global_cin_output_buffer[i].range(319, 256) = ((long long*)start)[i*8+4];
    __xlx_global_cin_output_buffer[i].range(383, 320) = ((long long*)start)[i*8+5];
    __xlx_global_cin_output_buffer[i].range(447, 384) = ((long long*)start)[i*8+6];
    __xlx_global_cin_output_buffer[i].range(511, 448) = ((long long*)start)[i*8+7];
  }
  for (int i = 0; i < __xlx_size_param_global_cin; ++i) {
    ((long long*)__xlx_apatb_param_global_cin)[i*8+0] = __xlx_global_cin_output_buffer[i].range(63, 0).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+1] = __xlx_global_cin_output_buffer[i].range(127, 64).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+2] = __xlx_global_cin_output_buffer[i].range(191, 128).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+3] = __xlx_global_cin_output_buffer[i].range(255, 192).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+4] = __xlx_global_cin_output_buffer[i].range(319, 256).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+5] = __xlx_global_cin_output_buffer[i].range(383, 320).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+6] = __xlx_global_cin_output_buffer[i].range(447, 384).to_uint64();
    ((long long*)__xlx_apatb_param_global_cin)[i*8+7] = __xlx_global_cin_output_buffer[i].range(511, 448).to_uint64();
  }
// print __xlx_apatb_param_global_cout
  sc_bv<512>*__xlx_global_cout_output_buffer = new sc_bv<512>[__xlx_size_param_global_cout];
  for (int i = 0; i < __xlx_size_param_global_cout; ++i) {
    char* start = (char*)(&(__xlx_global_cin_global_cout__input_buffer[__xlx_offset_param_global_cout]));
    __xlx_global_cout_output_buffer[i].range(63, 0) = ((long long*)start)[i*8+0];
    __xlx_global_cout_output_buffer[i].range(127, 64) = ((long long*)start)[i*8+1];
    __xlx_global_cout_output_buffer[i].range(191, 128) = ((long long*)start)[i*8+2];
    __xlx_global_cout_output_buffer[i].range(255, 192) = ((long long*)start)[i*8+3];
    __xlx_global_cout_output_buffer[i].range(319, 256) = ((long long*)start)[i*8+4];
    __xlx_global_cout_output_buffer[i].range(383, 320) = ((long long*)start)[i*8+5];
    __xlx_global_cout_output_buffer[i].range(447, 384) = ((long long*)start)[i*8+6];
    __xlx_global_cout_output_buffer[i].range(511, 448) = ((long long*)start)[i*8+7];
  }
  for (int i = 0; i < __xlx_size_param_global_cout; ++i) {
    ((long long*)__xlx_apatb_param_global_cout)[i*8+0] = __xlx_global_cout_output_buffer[i].range(63, 0).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+1] = __xlx_global_cout_output_buffer[i].range(127, 64).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+2] = __xlx_global_cout_output_buffer[i].range(191, 128).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+3] = __xlx_global_cout_output_buffer[i].range(255, 192).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+4] = __xlx_global_cout_output_buffer[i].range(319, 256).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+5] = __xlx_global_cout_output_buffer[i].range(383, 320).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+6] = __xlx_global_cout_output_buffer[i].range(447, 384).to_uint64();
    ((long long*)__xlx_apatb_param_global_cout)[i*8+7] = __xlx_global_cout_output_buffer[i].range(511, 448).to_uint64();
  }
// print __xlx_apatb_param_global_prev_cin
  sc_bv<512>*__xlx_global_prev_cin_output_buffer = new sc_bv<512>[__xlx_size_param_global_prev_cin];
  for (int i = 0; i < __xlx_size_param_global_prev_cin; ++i) {
    char* start = (char*)(&(__xlx_global_prev_cin__input_buffer[__xlx_offset_param_global_prev_cin]));
    __xlx_global_prev_cin_output_buffer[i].range(63, 0) = ((long long*)start)[i*8+0];
    __xlx_global_prev_cin_output_buffer[i].range(127, 64) = ((long long*)start)[i*8+1];
    __xlx_global_prev_cin_output_buffer[i].range(191, 128) = ((long long*)start)[i*8+2];
    __xlx_global_prev_cin_output_buffer[i].range(255, 192) = ((long long*)start)[i*8+3];
    __xlx_global_prev_cin_output_buffer[i].range(319, 256) = ((long long*)start)[i*8+4];
    __xlx_global_prev_cin_output_buffer[i].range(383, 320) = ((long long*)start)[i*8+5];
    __xlx_global_prev_cin_output_buffer[i].range(447, 384) = ((long long*)start)[i*8+6];
    __xlx_global_prev_cin_output_buffer[i].range(511, 448) = ((long long*)start)[i*8+7];
  }
  for (int i = 0; i < __xlx_size_param_global_prev_cin; ++i) {
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+0] = __xlx_global_prev_cin_output_buffer[i].range(63, 0).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+1] = __xlx_global_prev_cin_output_buffer[i].range(127, 64).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+2] = __xlx_global_prev_cin_output_buffer[i].range(191, 128).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+3] = __xlx_global_prev_cin_output_buffer[i].range(255, 192).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+4] = __xlx_global_prev_cin_output_buffer[i].range(319, 256).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+5] = __xlx_global_prev_cin_output_buffer[i].range(383, 320).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+6] = __xlx_global_prev_cin_output_buffer[i].range(447, 384).to_uint64();
    ((long long*)__xlx_apatb_param_global_prev_cin)[i*8+7] = __xlx_global_prev_cin_output_buffer[i].range(511, 448).to_uint64();
  }
// print __xlx_apatb_param_global_weight
  sc_bv<512>*__xlx_global_weight_output_buffer = new sc_bv<512>[__xlx_size_param_global_weight];
  for (int i = 0; i < __xlx_size_param_global_weight; ++i) {
    char* start = (char*)(&(__xlx_global_weight_global_bias__input_buffer[__xlx_offset_param_global_weight]));
    __xlx_global_weight_output_buffer[i].range(63, 0) = ((long long*)start)[i*8+0];
    __xlx_global_weight_output_buffer[i].range(127, 64) = ((long long*)start)[i*8+1];
    __xlx_global_weight_output_buffer[i].range(191, 128) = ((long long*)start)[i*8+2];
    __xlx_global_weight_output_buffer[i].range(255, 192) = ((long long*)start)[i*8+3];
    __xlx_global_weight_output_buffer[i].range(319, 256) = ((long long*)start)[i*8+4];
    __xlx_global_weight_output_buffer[i].range(383, 320) = ((long long*)start)[i*8+5];
    __xlx_global_weight_output_buffer[i].range(447, 384) = ((long long*)start)[i*8+6];
    __xlx_global_weight_output_buffer[i].range(511, 448) = ((long long*)start)[i*8+7];
  }
  for (int i = 0; i < __xlx_size_param_global_weight; ++i) {
    ((long long*)__xlx_apatb_param_global_weight)[i*8+0] = __xlx_global_weight_output_buffer[i].range(63, 0).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+1] = __xlx_global_weight_output_buffer[i].range(127, 64).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+2] = __xlx_global_weight_output_buffer[i].range(191, 128).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+3] = __xlx_global_weight_output_buffer[i].range(255, 192).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+4] = __xlx_global_weight_output_buffer[i].range(319, 256).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+5] = __xlx_global_weight_output_buffer[i].range(383, 320).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+6] = __xlx_global_weight_output_buffer[i].range(447, 384).to_uint64();
    ((long long*)__xlx_apatb_param_global_weight)[i*8+7] = __xlx_global_weight_output_buffer[i].range(511, 448).to_uint64();
  }
// print __xlx_apatb_param_global_bias
  sc_bv<512>*__xlx_global_bias_output_buffer = new sc_bv<512>[__xlx_size_param_global_bias];
  for (int i = 0; i < __xlx_size_param_global_bias; ++i) {
    char* start = (char*)(&(__xlx_global_weight_global_bias__input_buffer[__xlx_offset_param_global_bias]));
    __xlx_global_bias_output_buffer[i].range(63, 0) = ((long long*)start)[i*8+0];
    __xlx_global_bias_output_buffer[i].range(127, 64) = ((long long*)start)[i*8+1];
    __xlx_global_bias_output_buffer[i].range(191, 128) = ((long long*)start)[i*8+2];
    __xlx_global_bias_output_buffer[i].range(255, 192) = ((long long*)start)[i*8+3];
    __xlx_global_bias_output_buffer[i].range(319, 256) = ((long long*)start)[i*8+4];
    __xlx_global_bias_output_buffer[i].range(383, 320) = ((long long*)start)[i*8+5];
    __xlx_global_bias_output_buffer[i].range(447, 384) = ((long long*)start)[i*8+6];
    __xlx_global_bias_output_buffer[i].range(511, 448) = ((long long*)start)[i*8+7];
  }
  for (int i = 0; i < __xlx_size_param_global_bias; ++i) {
    ((long long*)__xlx_apatb_param_global_bias)[i*8+0] = __xlx_global_bias_output_buffer[i].range(63, 0).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+1] = __xlx_global_bias_output_buffer[i].range(127, 64).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+2] = __xlx_global_bias_output_buffer[i].range(191, 128).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+3] = __xlx_global_bias_output_buffer[i].range(255, 192).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+4] = __xlx_global_bias_output_buffer[i].range(319, 256).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+5] = __xlx_global_bias_output_buffer[i].range(383, 320).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+6] = __xlx_global_bias_output_buffer[i].range(447, 384).to_uint64();
    ((long long*)__xlx_apatb_param_global_bias)[i*8+7] = __xlx_global_bias_output_buffer[i].range(511, 448).to_uint64();
  }
// print __xlx_apatb_param_layer_config
  sc_bv<32>*__xlx_layer_config_output_buffer = new sc_bv<32>[__xlx_size_param_layer_config];
  for (int i = 0; i < __xlx_size_param_layer_config; ++i) {
    __xlx_layer_config_output_buffer[i] = __xlx_layer_config__input_buffer[i+__xlx_offset_param_layer_config];
  }
  for (int i = 0; i < __xlx_size_param_layer_config; ++i) {
    ((int*)__xlx_apatb_param_layer_config)[i] = __xlx_layer_config_output_buffer[i].to_uint64();
  }
}
