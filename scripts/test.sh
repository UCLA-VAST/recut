#!/bin/bash
# These are example project actions that may be useful to consult for
# reference. Some commands may be environment specific
# all of these commands can be run in background as asynchronous
# jobs leveraging asynctasks.vim and asyncrun.vim

for flag in ALL NO_INTERVAL_RV SCHEDULE_INTERVAL_RV
do
  echo $flag
  config_fn=../src/config.hpp
  echo "#define $flag" >> $config_fn

  # buildPhase
  #cd ../build && make -j 56 && make install -j 56

  # runPhase
  # 6 hour timout
  to_seconds=21600
  cd ../bin
  for test_num in {11..32}
  do
    name=recut-test-$flag-$test_num-2
    # if it already exists
    if [ ! -f ../data/$name.json ]; then
      echo "File ../data/${name}.json not found"
      timeout $to_seconds ./recut_test --gtest_output=json:../data/$name.json --gtest_filter=*.ChecksIfFinalVerticesCorrect/$test_num | tee ../data/$name.log
    fi
  done

  # clean up, removing last appended line from above
  sed -i '$d' $config_fn

  echo; echo;
done

