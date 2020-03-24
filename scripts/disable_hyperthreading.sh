#!/bin/bash
for i in {28..55}; do echo "0" > /sys/devices/system/cpu/cpu$i/online; done
