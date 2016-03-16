#!/bin/bash

BIN=bin/cufft
BASE="log/cufft"
START=$1
END=$2
for j in 512 1024 2048 4096 8192;do
	echo \t$j
	$BIN $j $START $END > "log/cufft_$j.py"
done


