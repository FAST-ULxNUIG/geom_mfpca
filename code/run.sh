#!/bin/sh
for N in 25 50 75 100
do
    for M in 25 50 75 100
    do
        for P in 2 10 20 50
        do
            python ./code/main.py -nsimu 500 -nobs $N -m $M -p $P -k 5 ./results
        done
    done
done

for N in 25 50 75 100
do
    for M in 25 50 75 100
    do
        python ./code/main.py -nsimu 500 -nobs $N -m $M -p 1 -k 5 ./results
    done
done
