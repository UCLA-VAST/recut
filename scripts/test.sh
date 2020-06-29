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
  cd ../build && make -j 56 && make install -j 56

  # runPhase
  for $test_num in {11..32}
  do
    name=recut-test-$flag-$test_num
    cd ../bin && ./recut_test --gtest_output=json:../data/$name.json --gtest_filter=*.ChecksIfFinalVerticesCorrect/$test_num | tee ../data/$name.log
  done

  # clean up, removing last appended line from above
  sed -i '$d' $config_fn

  echo; echo;
done

