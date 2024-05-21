python main.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --hf_token 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 \
    --use_auth \
    --output_dir ./output/qrslora_exp/exp2c/7B/mistral-7b-self-instruct-qrslora-4bit \
    --report_to wandb \
    --run_name mistral_7B_selfinstruct_qrslora_4bit_2c \
    --logging_steps 25 \
    --save_strategy steps \
    --data_seed 2024 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 0.2 \
    --per_device_eval_batch_size 6 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_modules all \
    --use_rslora True \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset self-instruct \
    --dataset_format self-instruct \
    --source_max_len 384 \
    --target_max_len 128 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_steps 5000 \
    --eval_steps 500 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.001 \
    --seed 2024 \