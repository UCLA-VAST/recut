# echo "Finding dirs in: $1"

jarpath="/mnt/c/Users/YangRecon2/Desktop/StdSwc1_4/dist/stdswc1_4.jar_/20080627145635/stdswc1_4.jar"
# for each component directory
for d in $1/*/ ; do
  if [[ "$d" != *"discard"* ]]; then
    dir=${d%/}
    echo "  $dir"
    for path in $dir/*.swc; do
      # if it hasn't already been run
      if [[ "$path" != *"corrected"* ]]; then
	  swc=`basename $path`
	  echo "    $swc"
	  # filter app2 swcs
	  java -jar $jarpath $dir/stdlog-$swc.txt -op Recut -f -corrected.swc -in $path
      fi
    done
  fi
done
