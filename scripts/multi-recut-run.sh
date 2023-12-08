#!/bin/bash

timeout 1d recut vdbs/img-mask-fgpct-0.100.vdb --fg-percent .1 --seeds Inspected_FNT_renamed_sorted
timeout 1d recut vdbs/img-mask-fgpct-0.200.vdb --fg-percent .2 --seeds Inspected_FNT_renamed_sorted
# already did .4 for non app2
timeout 1d recut vdbs/img-mask-fgpct-0.800.vdb --fg-percent .8 --seeds Inspected_FNT_renamed_sorted
timeout 1d recut vdbs/img-mask-fgpct-1.600.vdb --fg-percent 1.6 --seeds Inspected_FNT_renamed_sorted
