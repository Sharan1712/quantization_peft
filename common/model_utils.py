import os
from os.path import exists, join, isdir
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
import bitsandbytes as bnb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

from common.utils import *

DEFAULT_PAD_TOKEN = "[PAD]"

# used to find and list the names of all the linear modules
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

## purpose of this callback is to save a specific state of a model during training
class SavePeftModelCallback(transformers.TrainerCallback):
    ## called to save the model at a checkpoint
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        ## determines the folder where the checkpoint should be saved
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        ## removal might be to prevent conflicts with the 'adapter model' or to save disk space
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    ## override of the on_save hook from TrainerCallback
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control
    
    ## overrides the on_train_end hook from TrainerCallback
    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

## initializes and configures a model for accelerated training, potentially with quantization and specialized training techniques such as LoRA (Low-Rank Adaptation)
def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = 2
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

    ## asserts that the number of bits used for model weights must be either 16 or 32 for full finetuning
    if args.full_finetune: assert args.bits in [16, 32]

    ## model is loaded from a pre-trained state, with parameters for quantization and other configurations determined by the arguments.
    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir = args.cache_dir,
        device_map = device_map,
        max_memory = max_memory,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = args.bits == 4,
            load_in_8bit = args.bits == 8,
            llm_int8_threshold = 6.0,
            llm_int8_has_fp16_weight = False,
            bnb_4bit_compute_dtype = compute_dtype,
            bnb_4bit_use_double_quant = args.double_quant,
            bnb_4bit_quant_type = args.quant_type,
        ),
        torch_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code = args.trust_remote_code,
        token = args.hf_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    ## flags the model as parallelizable and model-parallel ready
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Initializing the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir = args.cache_dir,
        padding_side = "right",
        use_fast = False, # Fast tokenizer giving issues.
        tokenizer_type = 'llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code = args.trust_remote_code,
        token = args.hf_token,
    )
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
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable = True)
        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r = args.lora_r,
                lora_alpha = args.lora_alpha,
                target_modules = modules,
                lora_dropout = args.lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    ## iterates through the named modules of the model to perform type casting to the appropriate data types
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training