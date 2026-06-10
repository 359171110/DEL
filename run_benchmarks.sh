#!/bin/bash


# Create bench directory if it doesn't exist
mkdir -p ./bench/

# Define models and associated GPU/exit layer/speculation settings
declare -A model_params
model_params["facebook/layerskip-llama3.2-1B"]="0 3 6"
model_params["facebook/layerskip-llama3-8B"]="0 3 3"
model_params["facebook/layerskip-llama2-7B"]="0 7 6"
model_params["facebook/layerskip-llama2-13B"]="0 13 4"
model_params["facebook/layerskip-llama2-70B"]="0,1,2,3 9 6"

# Define datasets
datasets=("aqua_rat" "cnn_dm_lm" "cnn_dm_summarization" "xsum_summarization" "human_eval" "gsm8k" "wmt14_de_en")

# Define generation strategies
strategies=("self_speculative" "DEL_speculative" "FSM_speculative" "DV_speculative" "autoregressive")

# FLy win_len values to sweep
fly_win_lens=(4 6 8)

# Define master port
master_port=29500

# Define CPU core binding
cpu_cores="0-15"

# Output directory
output_dir="./logs/"

# Number of samples and max steps
num_samples=1000
max_steps=512

# Loop through models and datasets
for model in "${!model_params[@]}"; do
  params=(${model_params[$model]})
  gpu_device=${params[0]}
  exit_layer=${params[1]}
  num_speculations=${params[2]}

  for dataset in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
      # Define log file name dynamically
      log_name="./bench/bench_${model##*/}_${dataset}_${strategy}_1k_${max_steps}"
      if [[ "$strategy" != "autoregressive" ]]; then
        log_name+="_${exit_layer}-${num_speculations}"
      fi
      log_name+=".log"

      # Construct and execute the command
      cmd="CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores torchrun --master_port=$master_port benchmark.py \
        --model $model --dataset $dataset --generation_strategy $strategy \
        --num_samples $num_samples --max_steps $max_steps \
        --exit_layer $exit_layer --num_speculations $num_speculations \
        --output_dir $output_dir --sample False >> $log_name"
      
      # Special case for autoregressive (no exit_layer/num_speculations needed)
      if [[ "$strategy" == "autoregressive" ]]; then
        cmd="CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores torchrun --master_port=$master_port benchmark.py \
          --model $model --dataset $dataset --generation_strategy $strategy \
          --num_samples $num_samples --max_steps $max_steps \
          --output_dir $output_dir --sample False >> $log_name"
      fi

      # Print and execute the command
      echo "Running: $cmd"
      eval $cmd
    done
  done
done

# Two-model draft-target pairs (draft → target)
declare -A draft_for_target
draft_for_target["facebook/layerskip-llama3-8B"]="facebook/layerskip-llama3.2-1B"
draft_for_target["facebook/layerskip-llama2-13B"]="facebook/layerskip-llama2-7B"
draft_for_target["facebook/layerskip-llama2-70B"]="facebook/layerskip-llama2-7B"

# Two-model speculative decoding: exact match vs FLy
for target_model in "${!draft_for_target[@]}"; do
  draft_model=${draft_for_target[$target_model]}
  params=(${model_params[$target_model]})
  gpu_device=${params[0]}
  num_speculations=${params[2]}

  for dataset in "${datasets[@]}"; do
    # Two-model exact match
    log_name="./bench/bench_${target_model##*/}_${dataset}_FLy_exact_1k_${max_steps}.log"
    cmd="CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores torchrun --master_port=$master_port benchmark.py \
      --model $target_model --dataset $dataset --generation_strategy FLy_speculative \
      --draft_model $draft_model \
      --num_samples $num_samples --max_steps $max_steps \
      --num_speculations $num_speculations \
      --output_dir $output_dir --sample False >> $log_name"
    echo "Running: $cmd"
    eval $cmd

    # Two-model FLy loosely match (sweep win_len)
    for win_len in "${fly_win_lens[@]}"; do
      log_name="./bench/bench_${target_model##*/}_${dataset}_FLy_loose_w${win_len}_1k_${max_steps}.log"
      cmd="CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores torchrun --master_port=$master_port benchmark.py \
        --model $target_model --dataset $dataset --generation_strategy FLy_speculative \
        --draft_model $draft_model \
        --num_samples $num_samples --max_steps $max_steps \
        --num_speculations $num_speculations \
        --enable_fly True --fly_win_len $win_len \
        --output_dir $output_dir --sample False >> $log_name"
      echo "Running: $cmd"
      eval $cmd
    done
  done
done

# DEL + FLy experiments (sweep over win_len)
for model in "${!model_params[@]}"; do
  params=(${model_params[$model]})
  gpu_device=${params[0]}
  exit_layer=${params[1]}
  num_speculations=${params[2]}

  for dataset in "${datasets[@]}"; do
    for win_len in "${fly_win_lens[@]}"; do
      log_name="./bench/bench_${model##*/}_${dataset}_DEL_fly_w${win_len}_1k_${max_steps}_${exit_layer}-${num_speculations}.log"

      cmd="CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores torchrun --master_port=$master_port benchmark.py \
        --model $model --dataset $dataset --generation_strategy DEL_speculative \
        --num_samples $num_samples --max_steps $max_steps \
        --exit_layer $exit_layer --num_speculations $num_speculations \
        --enable_fly True --fly_win_len $win_len \
        --output_dir $output_dir --sample False >> $log_name"

      echo "Running: $cmd"
      eval $cmd
    done
  done
done
