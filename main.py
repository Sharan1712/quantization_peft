from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd

from common.utils import *
from common.model_utils import *
from common.data_utils import * 

import torch
import transformers
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer
)
from trl import SFTTrainer
from datasets import load_dataset
import evaluate
import wandb

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "Master Thesis Experiments"
os.environ["WANDB_LOG_MODEL"] = "true"

## We set the ModelArguments
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default = "mistralai/Mistral-7B-Instruct-v0.2"
    )
    trust_remote_code: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    hf_token: Optional[str] = field(
        default = "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",
        metadata = {"help": "Enables using Huggingface auth token from Git Credentials."}
    )

## We set the DataArguments
@dataclass
class DataArguments:
    eval_dataset_size: float = field(
        default = 0.3, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default = None,
        metadata = {
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default = None,
        metadata = {
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default = 1024,
        metadata = {"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default = 256,
        metadata = {"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default = 'alpaca',
        metadata = {"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default = None,
        metadata = {"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    upload_to_hub: bool = field(default = True, metadata = {"help":"Whether or not to upload the finetuned model"} )
    #load_best_model_at_end: bool = field(default = True, metadata = {"help":"Whether or not to load the best model found during training at the end of training"} )
    #metric_for_best_model: str = field(default = "eval_loss", metadata = {"help":"specify the metric to use to compare two different models"})
    report_to: str = field(default = "wandb", metadata = {"help":"Where to log losses"})
    run_name: str = field(default = "experiment-1", metadata = {"help":"Name of the run to see on W&B"})
    n_gpus: int = field(default = 2, metadata = {"help": "Number of GPUs to use while training."})
    cache_dir: Optional[str] = field(default = None)
    train_on_source: Optional[bool] = field(
        default = False,
        metadata = {"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(default = 'eval', metadata = {"help": "The MMLU split to run on"})
    mmlu_dataset: Optional[str] = field(
        default = 'mmlu-fs',
        metadata = {"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(default = False, metadata = {"help": "Whether to run the MMLU evaluation."})
    max_mmlu_samples: Optional[int] = field(
        default = None,
        metadata = {"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(default = 2048, metadata = {"help": "Maximum source sequence length for mmlu."})
    full_finetune: bool = field(default = False, metadata = {"help": "Finetune the entire model without adapters."})
    adam8bit: bool = field(default = False, metadata = {"help": "Use 8-bit adam."})
    bf16: bool = field(default = True, metadata = {"help": 'To use bfloat16'})
    fp16: bool = field(default = False, metadata = {"help": 'To use float16'})
    double_quant: bool = field(
        default = True,
        metadata = {"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default = "nf4",
        metadata = {"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default = 4, metadata = {"help": "How many bits to use."})
    lora_r: int = field(default = 64, metadata = {"help": "Lora R dimension."})
    lora_alpha: float = field(default = 16, metadata = {"help": " Lora alpha."})
    lora_dropout: float = field(default = 0.0, metadata = {"help":"Lora dropout."})
    use_rslora: bool = field(default = False, metadata = {"help": 'When set to True, uses Rank-Stabilized LoRA which sets the adapter scaling factor to lora_alpha/math.sqrt(r), since it was proven to work better.'})
    use_dora: bool = field(default = False, metadata = {"help": 'Whether to include DoRA (Weight Decomposed Low Rank Adaptation)'})
    use_loftq: bool = field(default = False, metadata = {"help": 'Whether to initialize the LoRA Adapter weights using LoftQ initialization.'})
    max_memory_MB: int = field(default = 49000, metadata = {"help": "Free memory per gpu."})
    report_to: str = field(default = 'none', metadata = {"help": "To use wandb or something else for reporting."})
    sft: bool = field(default = False, metadata = {"help": "If True, use the SupervisedFineTuning Trainer of HF else use Seq2SeqTrainer"})
    output_dir: str = field(default = './output', metadata = {"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default = 'paged_adamw_32bit', metadata = {"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default = 2, metadata = {"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default = 16, metadata = {"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default = 10000, metadata = {"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default = 0.0, metadata = {"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default = 0.0002, metadata = {"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default = False, metadata = {"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default = 0.3, metadata = {"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default = True, metadata = {"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default = True, metadata = {"help": 'To train or not to train, that is the question?'})
    do_eval: bool = field(default = True, metadata = {"help": 'To create a eval set or no, that is the question?'})
    do_mmlu_eval: bool = field(default = False, metadata = {"help": 'To do evaluation on mmlu set or no, that is the question?'})
    lr_scheduler_type: str = field(default = 'constant', metadata = {"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default = 0.03, metadata = {"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default = 10, metadata = {"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default = True, metadata = {"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default = 'steps', metadata = {"help": 'When to save checkpoints'})
    save_steps: int = field(default = 250, metadata = {"help": 'How often to save a model'})
    save_total_limit: int = field(default = 40, metadata = {"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default = 256,
        metadata = {"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                            "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(default = None, metadata = {"help": "Minimum number of new tokens to generate."})

    # Generation strategy
    do_sample: Optional[bool] = field(default = False)
    num_beams: Optional[int] = field(default = 1)
    num_beam_groups: Optional[int] = field(default = 1)
    penalty_alpha: Optional[float] = field(default = None)
    use_cache: Optional[bool] = field(default = True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default = 1.0)
    top_k: Optional[int] = field(default = 50)
    top_p: Optional[float] = field(default = 1.0)
    typical_p: Optional[float] = field(default = 1.0)
    diversity_penalty: Optional[float] = field(default = 0.0)
    repetition_penalty: Optional[float] = field(default = 1.0)
    length_penalty: Optional[float] = field(default = 1.0)
    no_repeat_ngram_size: Optional[int] = field(default = 0)

def train():
    ## this part of the code is used to parse the arguments
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings = True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)
    

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')
        args.do_train = False

    model, tokenizer, peft_config = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    #model.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer = tokenizer, args = args)
    
    if not args.sft: 
        print("Using Seq2SeqTrainer......")
        trainer = Seq2SeqTrainer(
            model = model,
            tokenizer = tokenizer,
            args = training_args,
            **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        )
    else:
        print("Using SFTTrainer......") ## Not working yet. Need to fix errors
        def format_instruction(sample):
            return f"""
            {sample['input']}

            {sample['output']}
            """
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
            peft_config = peft_config,
            max_seq_length = args.source_max_len,
            packing = True,
            formatting_func = format_instruction,
            args = training_args
        )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len
                self.mmlu_results = results 

        callback = MMLUEvalCallback()
        trainer.add_callback(callback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix = "eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    if args.do_mmlu_eval:
        print("MMLU Results.......")
        print(callback.mmlu_results)
        all_metrics.update(callback.mmlu_results)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset = data_module['predict_dataset'], metric_key_prefix = "predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens = True, clean_up_tokenization_spaces = True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.upload_to_hub:
        print("Uploading tuned model to HF..........")

        new_model = args.run_name
        trainer.model.push_to_hub(f"Sharan1712/{new_model}")
        tokenizer.push_to_hub(f"Sharan1712/{new_model}")

        # trainer.model.save_pretrained(new_model)

        # ## sets the maximum memory that can be used per device and automatically determines a device mapping for model parts
        # max_memory = f'{args.max_memory_MB}MB'
        # max_memory = {i: max_memory for i in range(args.n_gpus)}
        # device_map = "auto"

        # # if we are in a distributed setting, we need to set the device map and max memory per device
        # if os.environ.get('LOCAL_RANK') is not None:
        #     local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        #     device_map = {'': local_rank}
        #     max_memory = {'': max_memory[local_rank]}

        # base_model = AutoModelForCausalLM.from_pretrained(
        #     args.model_name_or_path,
        #     low_cpu_mem_usage = True,
        #     return_dict = True,
        #     torch_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        #     device_map = device_map,
        #     )
        
        # model = PeftModel.from_pretrained(base_model, new_model)
        # model = model.merge_and_unload()
        
        # # Reload tokenizer to save it
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.model_name_or_path,
        #     cache_dir = args.cache_dir,
        #     padding_side = "right",
        #     use_fast = False, # Fast tokenizer giving issues.
        #     tokenizer_type = 'llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        #     trust_remote_code = args.trust_remote_code
        # )
        # if tokenizer._pad_token is None:
        #     smart_tokenizer_and_embedding_resize(
        #         special_tokens_dict = dict(pad_token = DEFAULT_PAD_TOKEN),
        #         tokenizer = tokenizer,
        #         model = model,
        #     )
        # if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        #     # LLaMA tokenizer may not have correct special tokens set.
        #     # Check and add them if missing to prevent them from being parsed into different tokens.
        #     # Note that these are present in the vocabulary.
        #     # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        #     print('Adding special tokens.')
        #     tokenizer.add_special_tokens({
        #             "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
        #             "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        #             "unk_token": tokenizer.convert_ids_to_tokens(
        #                 model.config.pad_token_id if (model.config.pad_token_id != -1 and model.config.pad_token_id is not None) else tokenizer.pad_token_id
        #             ),
        #     })

        # model.push_to_hub(f"Sharan1712/{new_model}", use_temp_dir = False)
        # tokenizer.push_to_hub(f"Sharan1712/{new_model}", use_temp_dir = False)
        print("Uploaded tuned model to HF!!!")


    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()

