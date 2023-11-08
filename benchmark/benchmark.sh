#!/bin/bash

echo -e "------------------------------------
LAUNCHING BENCHMARK OF PIPELINE TOOL
Multiple models will be injected into
the tool and will be profiled to see
memory usage and execution time.
------------------------------------\n"

echo "All the benchmark output will be written into a txt file."

nb_gpus="$1"

nb_epochs=4
models=("CNN" "FFNET" "BigModel")
chunks=("2")

if [ -d "/var/lib/data/" ]; then
    data_dir="/var/lib/data"
    work_dir="/pipeline_tool/benchmark/benchmark.py"
else
    data_dir="."
    work_dir="benchmark.py"
fi

rm -f "$data_dir/results.txt"

time_sequence=$(seq -s ";" -f "Time %g [s]" $nb_epochs)
alloc_sequence=$(seq -s ";" -f "Alloc %g [MB]" $nb_epochs)

echo "Framework;Model;Number of GPUs;Number of Chunks;$time_sequence;$alloc_sequence" >> $data_dir/results.txt

echo "------------------------------------"
echo "Starting benchmarking refs with API Torch"
echo "    Benchmarking memory consumption and execution time with CNN..."
output=$(python3 $work_dir CNN "API torch" --gpu 1 --chunk 0 --epochs $nb_epochs --dir $data_dir)
printf "      Task ended \e[32m[OK]\e[0m \n"
echo "------------------------------------"
echo "    Benchmarking memory consumption and execution time with FFNET..."
output=$(python3 $work_dir FFNET "API torch" --gpu 1 --chunk 0 --epochs $nb_epochs --dir $data_dir)
printf "      Task ended \e[32m[OK]\e[0m \n"
echo -e "------------------------------------\n"
echo "    Benchmarking memory consumption and execution time with Vision Transformer..."
output=$(python3 $work_dir BigModel "API torch" --gpu 1 --chunk 0 --epochs $nb_epochs --dir $data_dir)
printf "      Task ended \e[32m[OK]\e[0m \n"
echo -e "------------------------------------\n"

echo "Starting benchmarking Pipeline Tool"
for model in "${models[@]}"; do
        for ((i = 1; i <= nb_gpus; i++)); do
            for chunk in "${chunks[@]}"; do
                echo "------------------------------------"
                echo "    Benchmarking memory consumption and execution time with $model on $i GPUs and $chunk chunks..."
                output=$(python3 $work_dir $model "Pipeline" --gpu $i --chunk $chunk --epochs $nb_epochs --dir $data_dir)
                printf "      Task ended \e[32m[OK]\e[0m \n"
            done
        done
done

echo "------------------------------------"
