#!/bin/sh

for N in 50 100 250
do
    echo "Simulation N=$N - M=11/11/21"
    python ./code/main.py -nsimu 200 -nobs $N -npoints 11 11 21 -noise 0 -percentages 1 1  -k 12 ./results
done


for N in 50 100 250
do
    echo "Simulation N=$N - M=26/26/51"
    python ./code/main.py -nsimu 200 -nobs $N -npoints 26 26 51 -noise 0 -percentages 1 1  -k 12 ./results
done


for N in 50 100 250
do
    echo "Simulation N=$N - M=101/51/201"
    python ./code/main.py -nsimu 200 -nobs $N -npoints 101 51 201 -noise 0 -percentages 1 1  -k 12 ./results
done
