python main.py \
    --model_name_or_path google/gemma-7b \
    --hf_token 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 \
    --use_auth \
    --output_dir ./output/qia3_exp/exp10a/7B/gemma-7b-unnatural-core-qia3-4bit \
    --report_to wandb \
    --run_name gemma_7B_unnaturalcore_qia3_4bit_10a \
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
    --peft_method IA3 \
    --quant_method hqq \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset unnatural-core \
    --dataset_format unnatural-core \
    --source_max_len 384 \
    --target_max_len 128 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_steps 5000 \
    --eval_steps 500 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.001 \
    --seed 2024 \