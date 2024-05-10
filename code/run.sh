#!/bin/sh
for N in 50 100 250
do
    echo "Simulation N=$N"
    python ./code/main.py -nsimu 200 -nobs $N -npoints 101 51 201 -noise 0 -percentages 0.5 0.5  -k 12 ./results
done
