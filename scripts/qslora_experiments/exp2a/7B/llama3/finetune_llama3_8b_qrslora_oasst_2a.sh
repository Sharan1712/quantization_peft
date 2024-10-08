python main.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_auth \
    --cache_dir ./cache \
    --output_dir ./output/qrslora_exp/exp2a/8B/llama3-8b-oasst-qrslora-4bit \
    --report_to wandb \
    --run_name llama3_8B_oasst_qrslora_4bit_2a \
    --logging_steps 25 \
    --save_strategy steps \
    --data_seed 2024 \
    --save_steps 500 \
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
    --lora_alpha 16 \
    --use_rslora True \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset oasst1 \
    --dataset_format oasst1 \
    --target_max_len 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.001 \
    --seed 2024 \
    --n_gpus 1 \