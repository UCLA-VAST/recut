for ims_file in $1/*.ims ; do
  # run both channels at once
  recut $ims_file --convert --input-type ims --output-type uint8 --fg-percent 1 & recut $ims_file --convert --input-type ims --output-type mask --fg-percent 10 --channel 1
  echo; echo "Finished $ims_file ... deleting"
  rm $ims_file
done
