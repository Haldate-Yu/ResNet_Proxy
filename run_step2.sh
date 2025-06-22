#!/bin/bash
echo "--------start of computation ----------"
mkdir data_500s

num_of_runs=500
for ((i=1; i<=num_of_runs; i++))
do
# ---
    cd ./data_500s
    mkdir case${i}
    cd ../
# ---
    echo "FINISHED FOR CASE "${i}
done
echo "----------- end of computation ----------"
