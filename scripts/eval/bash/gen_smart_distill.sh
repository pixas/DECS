LOG_DIR=logs/smart_distill
mkdir -p $LOG_DIR

NUM_CHUNKS=4

for CHUNK_IDX in $(seq 0 $((NUM_CHUNKS - 1))); do
    sbatch -o $LOG_DIR/gen_smart_distill_${CHUNK_IDX}.log \
           scripts/eval/slurm/gen_smart_distill.sh $NUM_CHUNKS $CHUNK_IDX
    sleep 3
done
