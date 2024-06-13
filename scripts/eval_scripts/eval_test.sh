lm_eval --model hf \
    --model_args pretrained=Sharan1712/llama2_7B_hhrlhf_qlora_4bit_1a \
    --tasks hellaswag \
    --device cuda:5 \
    --batch_size 8 \
    --output_path output/eval_results/exp1a/llama2_hhrlhf \
    