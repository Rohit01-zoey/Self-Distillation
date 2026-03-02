#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --time=06:00:00
#SBATCH --partition=nlplab
#SBATCH --account=nlplab
#SBATCH --gres=gpu:a6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=rkv6@duke.edu
#SBATCH --output=duke_bash_logs/eval_baseline.out
#SBATCH --mail-type=END,FAIL



# Use 2 GPUs via Hugging Face Accelerate (data parallel: 2 processes, 1 GPU each).
# FP16 mixed precision (config has bf16=False, fp16=True); bf16 not supported on this CUDA.
# accelerate launch --num_processes 2 --mixed_precision fp16 main.py \
#   --model_name Qwen/Qwen2.5-1.5B-Instruct \
#   --output_dir trial \
#   --learning_rate 2e-5 \
#   --num_train_epochs 1 

python evaluate.py --model_path trial/checkpoint-505 --eval_data data/tooluse_data/eval_data.json