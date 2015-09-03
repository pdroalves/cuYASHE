#!/bin/sh

python generate_plot.py --date $1 && gnuplot < gnuplot.in && evince gnuplot.ps
