import os
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import copy
import pandas as pd

import torch
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers

IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLM(object):
    ## Class Attributes
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    ## processes a batch of data instances and returns a dictionary suitable for training a causal language model.
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        ## source text is prepended with a beginning-of-string token
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        ## target text is appended with an end-of-string token
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length = self.source_max_len,
            truncation = True,
            add_special_tokens = False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length = self.target_max_len,
            truncation = True,
            add_special_tokens = False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first = True, padding_value = IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id), ## attention mask is created by marking non-padding tokens
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
    
##  process a dataset that contains instructional data
def extract_unnatural_instructions_data(examples, extract_reformulations = False):
    out = {
        'input': [],
        'output': [],
    }
    for instance in examples['instances']:
        out['input'].append(instance['instruction_with_input'])
        out['output'].append(instance['output'])
    if extract_reformulations:
        for instance in examples['reformulations']:
            if instance is not None:
                out['input'].append(instance['instruction_with_input'])
                out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples

    Coming soon:
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == "unnatural-core":
            return load_dataset("mrm8488/unnatural-instructions-core")
        elif dataset_name == "unnatural-full":
            return load_dataset("mrm8488/unnatural-instructions-full")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'unnatural-core' or (dataset_format is None and args.dataset == 'unnatural-core'):
            #dataset = extract_unnatural_instructions_data(dataset)
            dataset = dataset.map(extract_unnatural_instructions_data, remove_columns=['instruction'])
        elif dataset_format == 'unnatural-full' or (dataset_format is None and args.dataset == 'unnatural-full'):
            #dataset = extract_unnatural_instructions_data(dataset, extract_reformulations = True)
            dataset = dataset.map(extract_unnatural_instructions_data(extract_reformulations = True), remove_columns=['instruction'])
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size = args.eval_dataset_size, shuffle = True, seed = 2024
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer = tokenizer,
        source_max_len = args.source_max_len,
        target_max_len = args.target_max_len,
        train_on_source = args.train_on_source,
        predict_with_generate = args.predict_with_generate,
    )
    return dict(
        train_dataset = train_dataset if args.do_train else None,
        eval_dataset = eval_dataset if args.do_eval else None,
        predict_dataset = eval_dataset if args.do_predict else None,
        data_collator = data_collator
    )