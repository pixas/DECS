DATA_PATH=/mnt/hwfile/medai/deepscaler/deepscaler.json
NUM_CHUNKS=8

for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    OUTPUT_PATH="/mnt/petrelfs/jiangshuyang/datasets/deepscaler_clean_${i}.jsonl"
    echo "Processing chunk $i, output path: $OUTPUT_PATH"
    
    sbatch scripts/eval/slurm/filter_data.sh ${DATA_PATH}  ${OUTPUT_PATH} "--num_chunks ${NUM_CHUNKS} --chunk_idx ${i}"
    sleep 5
done