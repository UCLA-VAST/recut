echo "Finding dirs in: $1"

for d in $1/*/ ; do
  # TODO filter multi and discard dirs
  for f in d/*; do
    # filter app2 swcs
    java -jar "stdswc1_4.jar" $d/stdlog.txt -op Recut -f -corrected.swc -in ~/data/TME07-3A/run-1/component-36/component-36.swc
  done
done
