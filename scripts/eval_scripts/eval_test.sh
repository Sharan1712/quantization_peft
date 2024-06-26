lm_eval --model hf \
    --model_args pretrained=Sharan1712/llama2_7B_unnaturalcore_qrslora_4bit_2b \
    --tasks hellaswag,gsm8k,mmlu,glue,truthfulqa,winogrande,squadv2 \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/eval_results/exp2b/llama2_unnatural \
    --wandb_args project=lm-eval-harness-integration,name=llama2_7b_unnatural_2b \
    --log_samples