@echo off
if      %1==sim (vitis_hls FlexCNN.tcl csim_design csim_design -l FlexCNN_sim.log) ^
else if %1==cosim (vitis_hls FlexCNN.tcl cosim_design csynth_design -l FlexCNN_cosim.log) ^
else if %1==syn (vitis_hls FlexCNN.tcl csynth_design csynth_design -l FlexCNN_syn.log) ^
else (echo incorrect arguments)