
echo "Finding dirs in: $1"

EXT_PATH=pipeline_v.3.0/ch0/morph_neurite+soma/neurite+soma_segmentation/segmentation
for d in $1/*/ ;
do
  DIR=`basename $d`
  mkdir -p $DIR
  cp -r $d$EXT_PATH/marker_files $DIR/
  echo $DIR
  cd $DIR

  #if [ ! -f point.vdb ]
  #then
  #recut ../$d$EXT_PATH --convert point.vdb --input-type tiff --output-type point --parallel 1
  for thread_count in 1
  do
  #if [ ! -f point.vdb-log-$thread_count.txt ]
  if [ $thread_count -eq 1 ]
  then 
    # convert + reconstruct
    # recut ../$d$EXT_PATH --seeds marker_files --parallel $thread_count --output-windows uint8.vdb --run-app2
    recut point.vdb --seeds marker_files --parallel $thread_count --output-windows uint8.vdb --run-app2
  else
    recut ../$d$EXT_PATH --seeds marker_files --parallel $thread_count 
	  WAIT_TIME=60
	  MINUTES=0
	  MAX_MINUTES=30
	  until [ $MINUTES -eq $MAX_MINUTES ] || recut ../$d$EXT_PATH --seeds marker_files --parallel $thread_count; do
		      sleep $WAIT_TIME
		      $(( MINUTES++ ))
          done
  fi
  echo "finished thread_count: $thread_count"
  done
  #fi

  #if [ ! -d run-1 ]
  #then
  #recut point.vdb --seeds marker_files --output-windows uint8.vdb --parallel 1 --run-app2 
  #fi
  #for thread_count in 1 2 4 8 16 24 32 48
  #do
  #  recut point.vdb --seeds marker_files --input-type point --parallel $thread_count
  #done

  cd ..
done
