from math import ceil
import json
import argparse
from collections import OrderedDict   
SA_SIMD = 8
SA_COLS = 8
SA_ROWS = 8
in_num_hw, out_num_hw, in_h_hw, in_w_hw, out_h_hw, out_w_hw                     = 16, 16, 130, 130, 128, 128
in_num, out_num, in_h, in_w, out_h, out_w                                       = 16, 16, 128, 128, 128, 128
cin_offset, weight_offse, bias_offset, cout_offset, filter_s1, filter_s, stride = 0, 0, 0, 1463392, 1, 3, 1
INST, prev_cin_offset, in_num_t, out_num_t, in_h_t, in_w_t, nxt_layer_batch     = 4, 0, 16, 16, 8, 64, 1
 

task_num1 = int(ceil(float(in_num) / in_num_t) * ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
task_num2 = int(ceil(float(out_num) / out_num_t) * ceil(float(in_h) / in_h_t) * ceil(float(in_w) / in_w_t))
local_accum_num = int(in_num_t / SA_SIMD * filter_s * filter_s)
local_reg_num = int((in_h_t / stride) * (in_w_t / SA_COLS / stride) * (out_num_t / SA_ROWS))
row_il_factor = int(out_num_t / SA_ROWS)
col_il_factor = int(in_w_t / SA_COLS / stride)

print(task_num1, task_num2, local_accum_num, local_reg_num, row_il_factor, col_il_factor)