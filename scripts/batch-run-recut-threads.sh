
echo "Finding dirs in: $1"

EXT_PATH=pipeline_v.3.0/ch0/morph_neurite+soma/neurite+soma_segmentation/segmentation
for d in $1/*/ ;
do
  DIR=`basename $d`
  mkdir -p $DIR
  cp -r $d$EXT_PATH/marker_files $DIR/
  echo $DIR
  cd $DIR

  if [ ! -f point.vdb ]
  then
  #recut ../$d$EXT_PATH --convert point.vdb --input-type tiff --output-type point --parallel 24
  for thread_count in 1 2 4 8 16 24 32 48
  do
    recut ../$d$EXT_PATH --convert point.vdb --input-type tiff --output-type point --parallel $thread_count
    echo "finished thread_count: $thread_count"
  done
  fi

  recut point.vdb --seeds marker_files --output-windows uint8.vdb --run-app2 --parallel 1
  #if [ ! -d run-1 ]
  #then
  #for thread_count in 1 2 4 8 16 24 32 48
  #do
  #  recut point.vdb --seeds marker_files --input-type point --parallel $thread_count
  #done
  #fi

  cd ..
done
