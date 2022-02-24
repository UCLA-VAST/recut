echo Started at:
date

depth=48
for (( counter=0; counter<6576; counter+=$depth )); do
	recut 070440_100390 --convert --image-offsets 0 0 $counter --input-type tiff --output-type mask --fg-percent .05 -pl 48 --image-lengths -1 -1 $depth --tile-lengths -1 -1 1
done

echo Completed at:
date
