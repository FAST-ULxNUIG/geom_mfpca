#!/bin/sh

echo "Simulation N=250 - M=101/51/201 - medium sparse"
python ./code/main.py -nsimu 200 -nobs 250 -npoints 101 51 201 -noise 0 -percentages 0.4 0.4 -epsilon 0.1 -k 12 ./results


echo "Simulation N=250 - M=101/51/201 - high sparse"
python ./code/main.py -nsimu 200 -nobs 250 -npoints 101 51 201 -noise 0 -percentages 0.075 0.075 -epsilon 0.025 -k 12 ./results
