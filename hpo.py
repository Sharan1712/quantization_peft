import os
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd

from common.utils import *
from common.data_utils import * 
from common.model_utils import smart_tokenizer_and_embedding_resize

import torch
import transformers
import argparse
from datasets import load_dataset
import evaluate
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
import bitsandbytes as bnb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    IA3Config,
    #VeraConfig,
    get_peft_model,
    PeftModel,
    replace_lora_weights_loftq
)
from peft.tuners.lora import LoraLayer

## We set the ModelArguments
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default = "meta-llama/Llama-2-7b-hf"
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
    do_train: bool = field(default = True, metadata = {"help": 'To train or not to train, that is the question?'})
    do_eval: bool = field(default = True, metadata = {"help": 'To create a eval set or no, that is the question?'})
    n_gpus: int = field(default = 1, metadata = {"help": "Number of GPUs to use while training."})
    cache_dir: str = field(default = None)
    train_on_source: Optional[bool] = field(
        default = False,
        metadata = {"help": "Whether to train on the input in addition to the target text."}
    )
    full_finetune: bool = field(default = False, metadata = {"help": "Finetune the entire model without adapters."})
    adam8bit: bool = field(default = False, metadata = {"help": "Use 8-bit adam."})
    bf16: bool = field(default = True, metadata = {"help": 'To use bfloat16'})
    fp16: bool = field(default = False, metadata = {"help": 'To use float16'})
    double_quant: bool = field(
        default = True,
        metadata = {"help": "Compress the quantization statistics through double quantization."}
    )
    quant_method: str = field(default = "bnb", metadata = {"help": "Quantization method to use. Should be one of `bnb` or `hqq`."})
    quant_type: str = field(
        default = "nf4",
        metadata = {"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default = 4, metadata = {"help": "How many bits to use. If quant_method is bnb, either 8 or 4 else 1,2,3,4,8"})
    peft_method: str = field(default = "lora", metadata = {"help": "Which PEFT Method to use (lora, IA3)"})
    lora_r: int = field(default = 64, metadata = {"help": "Lora R dimension."})
    lora_alpha: float = field(default = 16, metadata = {"help": " Lora alpha."})
    lora_dropout: float = field(default = 0.0, metadata = {"help":"Lora dropout."})
    use_rslora: bool = field(default = False, metadata = {"help": 'When set to True, uses Rank-Stabilized LoRA which sets the adapter scaling factor to lora_alpha/math.sqrt(r), since it was proven to work better.'})
    use_dora: bool = field(default = False, metadata = {"help": 'Whether to include DoRA (Weight Decomposed Low Rank Adaptation)'})
    use_loftq: bool = field(default = False, metadata = {"help": 'Whether to initialize the LoRA Adapter weights using LoftQ initialization.'})
    max_memory_MB: int = field(default = 46000, metadata = {"help": "Free memory per gpu."})
    output_dir: str = field(default = './output', metadata = {"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default = 'paged_adamw_32bit', metadata = {"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default = 2, metadata = {"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default = 16, metadata = {"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default = 1000, metadata = {"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default = 0.0, metadata = {"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default = 0.0002, metadata = {"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default = False, metadata = {"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default = 0.3, metadata = {"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default = True, metadata = {"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default = 'constant', metadata = {"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default = 0.03, metadata = {"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default = 10, metadata = {"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default = True, metadata = {"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default = 'steps', metadata = {"help": 'When to save checkpoints'})
    save_steps: int = field(default = 250, metadata = {"help": 'How often to save a model'})
    save_total_limit: int = field(default = 40, metadata = {"help": 'How many checkpoints to save before the oldest is overwritten'})

# used to find and list the names of all the linear modules
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

def hpo():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings = True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    if torch.cuda.is_available():
        n_gpus = args.n_gpus ##no. of gpus to use for training
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
    
    ## sets the maximum memory that can be used per device and automatically determines a device mapping for model parts
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if args.quant_method == "bnb":
        print("Using BnB Config to quantize......")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = args.bits == 4,
            load_in_8bit = args.bits == 8,
            llm_int8_threshold = 6.0,
            llm_int8_has_fp16_weight = False,
            bnb_4bit_compute_dtype = compute_dtype,
            bnb_4bit_use_double_quant = args.double_quant,
            bnb_4bit_quant_type = args.quant_type
            )
    elif args.quant_method == "hhq":
        print("Using HQQ Config to quantize......")
        #quantization_config = HqqConfig(nbits = args.bits)

    tokenizer_type = 'llama' if 'llama' in args.model_name_or_path else None

    # Initializing the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir = args.cache_dir,
        padding_side = "right",
        use_fast = False, # Fast tokenizer giving issues.
        tokenizer_type = tokenizer_type, # Needed for HF name change
        trust_remote_code = args.trust_remote_code,
        token = args.hf_token,
    )

    def model_init(trial):
        
        model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir = args.cache_dir,
        device_map = device_map,
        max_memory = max_memory,
        quantization_config = quantization_config,
        torch_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code = args.trust_remote_code,
        use_safetensors = True,
        token = args.hf_token)

        DEFAULT_PAD_TOKEN = "[PAD]"

        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict = dict(pad_token = DEFAULT_PAD_TOKEN),
                tokenizer = tokenizer,
                model = model,
            )
        if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
            print('Adding special tokens.')
            tokenizer.add_special_tokens({
                    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                    "unk_token": tokenizer.convert_ids_to_tokens(
                        model.config.pad_token_id if (model.config.pad_token_id != -1 and model.config.pad_token_id is not None) else tokenizer.pad_token_id
                    ),
            })

        ## flags the model as parallelizable and model-parallel ready
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

        model.config.torch_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

        if args.peft_method == "lora":
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            peft_config = LoraConfig(
                r = args.lora_r,
                lora_alpha = args.lora_alpha,
                target_modules = modules,
                lora_dropout = args.lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
                use_rslora = args.use_rslora,
                use_dora = args.use_dora
            )
            model = get_peft_model(model, peft_config)
            if args.use_loftq:
                print(f'Using LoftQ Initialization.....')
                replace_lora_weights_loftq(model)
            
        elif args.peft_method == "IA3":
            print(f'adding IA3 Modules...')
            modules = find_all_linear_names(args, model)
            peft_config = IA3Config(
                target_modules = modules,
                task_type = "CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)

        elif args.peft_method == "vera":
            print(f'adding VeRA Modules...')
            modules = find_all_linear_names(args, model)
            #peft_config = VeraConfig(
            #    target_modules = modules,
            #    r = args.vera_r
            #)
            model = get_peft_model(model, peft_config)

        return model
    
    
    
    def wandb_hp_space(trial):
        return {
            "method": "random",
            "metric": {"name": "eval_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-3, "max": 1e-1},
                "per_device_train_batch_size": {"values": [8, 16, 32, 64, 128]},
            },
        }
    
    data_module = make_data_module(tokenizer = tokenizer, args = args)
    
    trainer = Seq2SeqTrainer(
            model = None,
            tokenizer = tokenizer,
            args = training_args,
            **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
            model_init=model_init,
        )
    
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=20
    )
    print(best_trial)

if __name__ == "__main__":
    hpo()

    