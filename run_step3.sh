#!/bin/bash
echo "--------start of computation ----------"

num_of_runs=500
for ((i=1; i<=num_of_runs; i++))
do
# ---
    cd ./ls_cases_500s/case${i}
    python3 data_ext.py ${i}
    cd ../../
# ---
    echo "FINISHED FOR CASE "${i}
# ---
done
echo "----------- end of computation ----------"
