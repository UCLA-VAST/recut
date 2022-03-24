
echo "Finding dirs in: $1"

thread_counts=( 1 2 4 8 16 24 32 48 )
EXT_PATH=pipeline_v.3.0/ch0/morph_neurite+soma/neurite+soma_segmentation/segmentation/
for d in $1/*/ ;
do
  DIR=`basename $d`
  mkdir -p $DIR
  echo $DIR
  cd $DIR
  cp -r $d/$EXT_PATH/marker_files .

  for thread_count in "$(thread_counts)"
  do
    recut $d/$EXT_PATH --convert point.vdb --input-type tiff --output-type point --parallel $thread_count
    echo "finished thread_count: $thread_count"
  done

  for thread_count in "$(thread_counts)"
  do
    recut point.vdb --seeds marker_files --input-type point
    #recut point.vdb --seeds marker_files --output-windows uint8.vdb --run-app2
  done

  cd ..
done