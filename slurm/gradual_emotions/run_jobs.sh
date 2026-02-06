#!/bin/bash
# Submit subliminal prompting sweep jobs

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command-line arguments
LOCAL_MODE=false
if [[ "$1" == "--local" ]]; then
    LOCAL_MODE=true
    echo "Running in LOCAL mode - jobs will execute sequentially"
fi

# Configuration
SOURCE_MODEL="Qwen/Qwen2.5-7B-Instruct"
BASE_OUTPUT_DIR="./results"

# Parameter values to test
# Template types to test
TEMPLATE_TYPES=("withoutthinking" "full")

# Number relations to test
NUMBER_RELATIONS=("love" "hate")

# Animal relations to test
ANIMAL_RELATIONS=("all")

# Submit jobs for each combination
for TEMPLATE_TYPE in "${TEMPLATE_TYPES[@]}"; do
    for NUMBER_RELATION in "${NUMBER_RELATIONS[@]}"; do
        for ANIMAL_RELATION in "${ANIMAL_RELATIONS[@]}"; do
            # Create job name
            JOB_NAME="${TEMPLATE_TYPE}_${NUMBER_RELATION}_${ANIMAL_RELATION}"

            if [ "$LOCAL_MODE" = true ]; then
                # Run locally in sequence
                echo "Running experiment: $JOB_NAME"
                python3 "./subliminal_prompting.py" \
                    --model "$SOURCE_MODEL" \
                    --template-types "$TEMPLATE_TYPE" \
                    --number-relations "$NUMBER_RELATION" \
                    --animal-relations "$ANIMAL_RELATION"

                if [ $? -eq 0 ]; then
                    echo "Experiment completed successfully for $JOB_NAME"
                else
                    echo "Experiment failed for $JOB_NAME"
                fi
            else
                # Submit job to SLURM
                JOB=$(sbatch.tinygpu "$DIR/job.slurm" \
                    --model "$SOURCE_MODEL" \
                    --template-types "$TEMPLATE_TYPE" \
                    --number-relations "$NUMBER_RELATION" \
                    --animal-relations "$ANIMAL_RELATION")
                echo "$JOB"
                # Extract job ID (handle 'Submitted batch job 1488859 on cluster tinygpu')
                JOB_ID=$(echo "$JOB" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
                echo "Submitted job $JOB_ID: $JOB_NAME"
            fi
        done
    done
done

echo ""
if [ "$LOCAL_MODE" = true ]; then
    echo "All jobs completed!"
else
    echo "All jobs submitted!"
fi
echo "Total: $((${#TEMPLATE_TYPES[@]} * ${#NUMBER_RELATIONS[@]} * ${#ANIMAL_RELATIONS[@]})) experiment jobs"
