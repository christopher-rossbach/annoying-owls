#!/bin/bash
# Submit unembedding extraction job

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command-line arguments
LOCAL_MODE=false
if [[ "$1" == "--local" ]]; then
    LOCAL_MODE=true
    echo "Running in LOCAL mode"
fi

if [ "$LOCAL_MODE" = true ]; then
    echo "Running unembeddings locally..."
    python3 "./unembeddings.py"

    if [ $? -eq 0 ]; then
        echo "Completed successfully"
    else
        echo "Failed"
    fi
else
    JOB=$(sbatch.tinygpu "$DIR/job.slurm")
    echo "$JOB"
    JOB_ID=$(echo "$JOB" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    echo "Submitted job $JOB_ID: unembeddings"
fi
