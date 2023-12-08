#!/bin/bash

timeout 6h recut vdbs/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds Inspected_FNT_renamed_sorted
timeout 6h recut vdbs/img-mask-fgpct-0.200.vdb --fg-percent .2 --seeds Inspected_FNT_renamed_sorted
timeout 6h recut vdbs/img-mask-fgpct-0.400.vdb --fg-percent .4 --seeds Inspected_FNT_renamed_sorted
timeout 6h recut vdbs/img-mask-fgpct-0.800.vdb --fg-percent .8 --seeds Inspected_FNT_renamed_sorted
timeout 6h recut vdbs/img-mask-fgpct-1.600.vdb --fg-percent 1.6 --seeds Inspected_FNT_renamed_sorted

# app2
#timeout 3d recut vdbs/img-mask-fgpct-0.400.vdb --fg-percent .4 --seeds Inspected_FNT_renamed_sorted --run-app2 --output-windows vdbs/img-uint8-fgpct-0.400.vdb
