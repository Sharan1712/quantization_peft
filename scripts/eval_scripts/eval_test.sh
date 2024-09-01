lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-70b-hf,parallelize=True,load_in_4bit=True \
    --use_cache ./cache \
    --tasks mmlu \
    --device cuda:3 \
    --batch_size 8 \
    --output_path output/eval_results/70B/llama2_70B \
    --wandb_args project=lm-eval-harness-integration,name=llama2_70b \
    --log_samples