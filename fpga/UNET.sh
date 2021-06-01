#!/bin/bash
VAR0="sim"
VAR1="cosim"
VAR2="syn"

if [ "$1" = "$VAR0" ]; then
    vivado_hls UNET.tcl csim_design csim_design -l UNET_sim.log
elif  [ "$1" = "$VAR1" ]; then
    vivado_hls UNET.tcl cosim_design csynth_design -l UNET_cosim.log
elif  [ "$1" = "$VAR2" ]; then
    vivado_hls UNET.tcl csynth_design csynth_design -l UNET_syn.log
else
    echo "incorrect argument"
fi

