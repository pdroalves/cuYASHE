#!/bin/bash

BIN=bin/result_cmp
BASE="log/result_bits"
START=$1
END=$2

for i in $(seq $START $END); do
	echo $i 
	$BIN $i > "log/result_bits_$i.py"
done


