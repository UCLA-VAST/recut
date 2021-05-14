open_project UNET_[lindex $argv 1]
set_top top_kernel
add_files util.h
add_files hw_kernel.cpp
add_files cnn_sw.h
add_files cnn_sw.cpp
add_files -tb UNet_tb.cpp
open_solution "solution1"
# set_part {xc7vx690tffg1761-2}
set_part {xcvu9p-fsgd2104-2L-e}
create_clock -period 3 -name default
config_interface -m_axi_addr64 -m_axi_offset off -register_io off
[lindex $argv 0]
# csim_design
# csynth_design
# cosim_design -trace_level all
#export_design -format ip_catalog

exit
