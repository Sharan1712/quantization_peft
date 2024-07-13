python main.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_auth \
    --output_dir ./output/qlora_exp/exp1d/8B/llama3-8b-hhrlhf-qlora-4bit \
    --report_to wandb \
    --run_name llama3_8B_hhrlhf_qlora_4bit_1d \
    --logging_steps 25 \
    --save_strategy steps \
    --data_seed 2024 \
    --save_steps 1500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 0.2 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 6 \
    --group_by_length \
    --dataloader_num_workers 1 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset hh-rlhf \
    --dataset_format hh-rlhf \
    --target_max_len 768 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --max_steps 3000 \
    --eval_steps 300 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.001 \
    --seed 2024 \