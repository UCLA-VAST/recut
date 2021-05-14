@echo off
if      %1==sim (vitis_hls UNET.tcl csim_design csim_design -l UNET_sim.log) ^
else if %1==cosim (vitis_hls UNET.tcl cosim_design csynth_design -l UNET_cosim.log) ^
else if %1==syn (vitis_hls UNET.tcl csynth_design csynth_design -l UNET_syn.log) ^
else (echo incorrect arguments)