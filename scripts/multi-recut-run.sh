#!/bin/bash

# change to the mask file
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 256
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 256

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 128
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 128

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 64
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 64

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 32
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 32

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 16
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 16

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 8
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 8

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 4
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 4

timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1 --fg-percent .1 --output-type seeds --voxel-size .4 .4 .4 -pl 2
timeout 6h recut images/Ex_488_Em_525_6x_8bit2bsh_tif_deconvolved_zl1/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds final-proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0 -pl 2

#timeout 6h recut images/04_Ex_647_Em_690_deconvolved/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds proofread-swcs --coarsen-steps 4 --voxel-size .4 .4 .4 --smooth-steps 0
#timeout 6h recut images/04_Ex_647_Em_690_deconvolved/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds proofread-swcs --coarsen-steps 3 --voxel-size .4 .4 .4 --smooth-steps 0
#timeout 6h recut images/04_Ex_647_Em_690_deconvolved/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds proofread-swcs --coarsen-steps 2 --voxel-size .4 .4 .4 --smooth-steps 0
#timeout 6h recut images/04_Ex_647_Em_690_deconvolved/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds proofread-swcs --coarsen-steps 1 --voxel-size .4 .4 .4 --smooth-steps 0
#timeout 6h recut images/04_Ex_647_Em_690_deconvolved/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds proofread-swcs --coarsen-steps 0 --voxel-size .4 .4 .4 --smooth-steps 0
#recut --fg-percent .1 --output-type seeds

#recut vdbs/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds Inspected_FNT_renamed_sorted
#recut vdbs/img-mask-fgpct-0.100.vdb --fg-percent .1 --output-type seeds
#recut vdbs/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds Inspected_FNT_renamed_sorted -pl
#timeout 6h recut vdbs/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds Inspected_FNT_renamed_sorted
#timeout 6h recut vdbs/img-mask-fgpct-0.200.vdb --fg-percent .2 --seeds Inspected_FNT_renamed_sorted
#timeout 6h recut vdbs/img-mask-fgpct-0.400.vdb --fg-percent .4 --seeds Inspected_FNT_renamed_sorted
#timeout 6h recut vdbs/img-mask-fgpct-0.800.vdb --fg-percent .8 --seeds Inspected_FNT_renamed_sorted
#timeout 6h recut vdbs/img-mask-fgpct-1.600.vdb --fg-percent 1.6 --seeds Inspected_FNT_renamed_sorted

# app2
#timeout 3d recut vdbs/img-mask-fgpct-0.200.vdb --fg-percent .2 --seeds Inspected_FNT_renamed_sorted --run-app2 --output-windows vdbs/img-uint8-fgpct-0.200.vdb
