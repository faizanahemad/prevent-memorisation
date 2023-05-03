#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import sys

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, DummyScheduler
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import gc
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training,  TaskType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from peft.utils.other import fsdp_auto_wrap_policy
from datasets import Dataset
from datasets import concatenate_datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    AutoModelForCausalLM,
)
from transformers.utils.versions import require_version


logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--token_weights",
        type=str,
        default=None,
        help="Output Token weights for our experiment",
    )
    parser.add_argument(
        "--token_weights_column",
        type=str,
        default=None,
        help="Output Token weights for our experiment",
    )
    parser.add_argument(
        "--fraction_dataset", action="store_true", help="Train over a smaller fraction of the dataset"
    )
    parser.add_argument(
        "--n_dataset_fractions",
        type=int,
        default=None,
        help=(
            "The train dataset fraction we train on"
        ),
    )
    parser.add_argument(
        "--train_fraction_number",
        type=int,
        default=None,
        help=(
            "The train dataset fraction we train on"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing_enable",
        action="store_true",
        help="Enable gradient checkpointing. This may not work if your architecture freezes some parameters and you try to use deepspeed or fsdp.",
    )
    parser.add_argument(
        "--zero_shot_evaluation", action="store_true", help="Zero shot Evaluate before any training"
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=32,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm.'
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            
        ),
    )
    parser.add_argument(
        '--use_lora',
        action='store_true',
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of the LoRA matrix',
    )

    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=32,
        help='Alpha of the LoRA matrix',
    )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='Path to the model state dict'
    )
    parser.add_argument(
        '--save_model',
        type=str,
        default="model.pt",
        help='Path to the model state dict for saving'
    )
    parser.add_argument(
        "--fsdp", action="store_true", help="Are we using FSDP training"
    )
    parser.add_argument(
        "--use_8bit_optim", action="store_true", help="Use 8 bit Optim"
    )
    parser.add_argument(
        "--use_8bit_model", action="store_true", help="Use 8 bit Optim"
    )
    parser.add_argument(
        "--use_clm", action="store_true", help="Use Causal LMs like GPT2"
    )
    parser.add_argument(
        "--generate_proba", action="store_true", help="Generate Probas for our dataset"
    )
    parser.add_argument(
        '--proba_store',
        type=str,
        default=None,
        help='Path to Store generated Probas dict'
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if args.fraction_dataset:
        assert args.n_dataset_fractions
        assert isinstance(args.train_fraction_number, int)
        args.train_fraction_number >= 0
        assert args.n_dataset_fractions > args.train_fraction_number
        
    if args.use_clm:
        assert args.pad_to_max_length
    if args.generate_proba:
        assert args.proba_store
        
    if args.token_weights:
        assert os.path.exists(args.token_weights) and os.path.isdir(args.token_weights)
        assert args.token_weights_column

    return args

def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))
    
class Preprocess:
    def __init__(self, text_column, summary_column, prefix, tokenizer, args):
        self.text_column = text_column
        self.summary_column = summary_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.args = args
        self.padding = "max_length" if args.pad_to_max_length else False
        
        
    def __call__(self, examples):
        text_column = self.text_column
        summary_column = self.summary_column
        prefix = self.prefix
        tokenizer = self.tokenizer
        args = self.args
        padding = self.padding
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        
        if args.use_clm:
            max_length = args.max_source_length + args.max_target_length
            clm_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=True)
            clm_targets = tokenizer(targets, max_length=args.max_target_length, padding=False, truncation=True)
            
            # if context is ending with special token, remove it
            if len(clm_inputs['input_ids'][0]) > 0 and clm_inputs['input_ids'][0][-1] in tokenizer.all_special_ids:
                clm_inputs['input_ids'] = [i[:-1] for i in clm_inputs['input_ids']]
                clm_inputs['attention_mask'] = [a[:-1] for a in clm_inputs['attention_mask']]
                
            # if context is ending with special token, remove it
            if len(model_inputs['input_ids'][0]) > 0 and model_inputs['input_ids'][0][-1] in tokenizer.all_special_ids:
                model_inputs['input_ids'] = [i[:-1] for i in model_inputs['input_ids']]
                model_inputs['attention_mask'] = [a[:-1] for a in model_inputs['attention_mask']]
                
            # if target is starting with special token, remove it
            if len(clm_targets['input_ids'][0]) > 0 and clm_targets['input_ids'][0][0] in tokenizer.all_special_ids:
                clm_targets['input_ids'] = [i[1:] for i in clm_targets['input_ids']]
                clm_targets['attention_mask'] = [a[1:] for a in clm_targets['attention_mask']]
            
            # Concat text
            out = {}
            out['input_ids'] = [i1 + i2 for i1,
                                i2 in zip(clm_inputs['input_ids'], clm_targets['input_ids'])]
            out['attention_mask'] = [a1 + a2 for a1,
                                     a2 in zip(clm_inputs['attention_mask'], clm_targets['attention_mask'])]

            # set -100 for context tokens
            out["labels"] = [[-100] * len(i1) + i2 for i1, i2 in zip(clm_inputs['input_ids'], clm_targets['input_ids'])]
            
            # Pad -> Left pad and truncate
            out["input_ids"] = [( [tokenizer.pad_token_id] * (max_length - len(i))) + i for i in out["input_ids"]]
            out["attention_mask"] = [([0] * (max_length - len(i))) + [1] * len(i) for i in out["attention_mask"]]
            out["labels"] = [([tokenizer.pad_token_id] * (max_length - len(i))) + i for i in out["labels"]]
            # truncate to max_length
            out["input_ids"] = [i[:max_length] for i in out["input_ids"]]
            out["attention_mask"] = [a[:max_length]
                                          for a in out["attention_mask"]]
            out["labels"] = [l[:max_length] for l in out["labels"]]
            model_inputs.update({f"clm_{k}": v for k, v in out.items()})
            
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            if args.use_clm:
                model_inputs["clm_labels"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["clm_labels"]
                ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def get_dataloaders(args, accelerator, tokenizer, model):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        assert dataset_columns is not None
        text_column = dataset_columns[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        assert dataset_columns is not None
        summary_column = dataset_columns[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )


    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
    preprocess_function = Preprocess(text_column, summary_column, prefix, tokenizer, args)


    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # Temporarily set max_target_length for validation.
        max_target_length = args.max_target_length
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        if args.fraction_dataset:
            total_fractions = args.n_dataset_fractions
            our_fraction = args.train_fraction_number
            train_dataset = train_dataset.shuffle(args.seed).flatten_indices()
            fraction_size = len(train_dataset)//total_fractions + 1
            train_dataset = train_dataset.select(range(our_fraction * fraction_size, min((our_fraction+1) * fraction_size, len(train_dataset) - 1)))
            

    if args.token_weights:
        token_weights = Dataset.load_from_disk(args.token_weights)
        # token_weights = token_weights.map(lambda x: {"proba": [a*b for a, b in zip(x["proba1"], x["proba2"])]})
        token_weights = token_weights.rename_column(args.token_weights_column, "proba")
        cols_to_remove = set(token_weights.column_names) - {"proba"}
        token_weights = token_weights.remove_columns(list(cols_to_remove))
        train_dataset = concatenate_datasets([train_dataset, token_weights], axis=1)
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    training_eval_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    logger.info(f"  Num Train examples = {len(train_dataset)}, Num Eval Examples = {len(eval_dataset)}")
    return train_dataloader, training_eval_dataloader, eval_dataloader

def generate_proba(model, tokenizer, accelerator, dataloader, args):
    assert (hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is None) or not hasattr(accelerator.state, "deepspeed_plugin")
    progress_bar = tqdm(range(len(dataloader)), desc="generate Logits", disable=not accelerator.is_local_main_process)
    all_proba = []
    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            if args.use_clm:
                batch = {k.replace("clm_", ""): v for k, v in batch.items() if "clm_" in k}
            outputs = model(**batch)
        if args.use_clm:
            input_ids = batch["input_ids"].reshape(-1)
        else:
            input_ids = batch["labels"].reshape(-1)
        logits = outputs.logits.softmax(dim=-1)
        b, s = logits.shape[:2]
        logits = logits.reshape(-1, logits.shape[-1])
        proba = logits[torch.arange(b*s), input_ids].reshape(b, s)
        proba = accelerator.gather_for_metrics(proba).cpu()
        if accelerator.is_main_process:
            all_proba.append(proba)
        progress_bar.update(1)
    if accelerator.is_main_process:
        all_proba = torch.cat(all_proba, 0).tolist()
        ds = Dataset.from_dict({"proba": all_proba})
        for index in random.sample(range(len(ds)), 1):
            logger.info(f"Sample {index} of the training set: {ds[index]}.")
        logger.info(f"  Num examples = {len(ds)}")
        ds.save_to_disk(args.proba_store)
    


def evaluate_model(model, tokenizer, accelerator, dataloader, args, result_key: str, fsdp_model_copy_for_eval_only=None):
    assert (hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is None) or not hasattr(accelerator.state, "deepspeed_plugin")
    unwrapped_model = accelerator.unwrap_model(model)
    metric = evaluate.load("rouge")
    if args.fsdp or args.use_8bit_optim:
        assert fsdp_model_copy_for_eval_only
        state_dict = get_state_dict(unwrapped_model, accelerator, fsdp=args.fsdp, rank0_only=False)
        # reshape = (unwrapped_model.encoder.embed_tokens.num_embeddings, unwrapped_model.encoder.embed_tokens.embedding_dim)
        # state_dict["encoder.embed_tokens.weight"] = state_dict["encoder.embed_tokens.weight"].reshape(reshape)
        # state_dict["decoder.embed_tokens.weight"] = state_dict["decoder.embed_tokens.weight"].reshape(reshape)
        fsdp_model_copy_for_eval_only.load_state_dict(state_dict)
        del state_dict
        fsdp_model_copy_for_eval_only = fsdp_model_copy_for_eval_only.to(accelerator.device)
        unwrapped_model = fsdp_model_copy_for_eval_only
    gen_kwargs = {
            "max_length": args.max_target_length + (args.max_source_length if args.use_clm else 0),
            "num_beams": args.num_beams,
        }
    if args.use_clm and "gpt2" in args.model_name_or_path:
        gen_kwargs.update({"pad_token_id": tokenizer.eos_token_id})
    progress_bar = tqdm(range(len(dataloader)), desc="evaluate", disable=not accelerator.is_local_main_process)
    unwrapped_model.config.use_cache = True
    unwrapped_model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            generated_tokens = unwrapped_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
            generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
            if args.use_clm:
                generated_tokens = generated_tokens[:, args.max_source_length:]
                assert generated_tokens.size(1) == labels.size(1)

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
            progress_bar.update(1)
    result = metric.compute(use_stemmer=True)
    result = {k + (("_"+ result_key) if result_key is not None else ""): round(v * 100, 4) for k, v in result.items()}
    unwrapped_model.config.use_cache = False
    unwrapped_model.train()
    if args.fsdp or args.use_8bit_optim:
        fsdp_model_copy_for_eval_only = fsdp_model_copy_for_eval_only.to("cpu")
    return result

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.use_lora)
    if args.fsdp:
        fsdp_params = FullyShardedDataParallelPlugin(use_orig_params=True)
    else:
        fsdp_params = None

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, fsdp_plugin=fsdp_params, kwargs_handlers=[ddp_kwargs], **accelerator_log_kwargs)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, args.model_name_or_path, args.dataset_name), exist_ok=True)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else []
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "flan",
        "gpt2",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
        args.source_prefix = 'summarize: '
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.use_clm:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    if args.model_name_or_path:
        if args.use_clm:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, load_in_8bit=args.use_8bit_model,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config, # torch_dtype=torch.float16
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path, load_in_8bit=args.use_8bit_model,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config, # torch_dtype=torch.float16
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)
    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        
    if model.config.decoder_start_token_id is None and not args.use_clm:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        
    if args.load_model:
        if args.use_lora:
            lora_dir = args.load_model
            if not os.path.isdir(lora_dir):
                lora_dir = os.path.dirname(lora_dir)
            if os.path.isfile(os.path.join(lora_dir, "adapter_config.json")) and os.path.isfile(os.path.join(lora_dir, "adapter_model.bin")):
                peft_config = PeftConfig.from_pretrained(lora_dir)
                model = PeftModel.from_pretrained(model, lora_dir)
            else:
                peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM if args.use_clm else TaskType.SEQ_2_SEQ_LM, 
                    inference_mode=False, r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.1
                )
                model = get_peft_model(model, peft_config)
                state_dict = torch.load(args.load_model, map_location='cpu')
                model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(args.load_model, map_location='cpu')
            model.load_state_dict(state_dict)
            del state_dict
        
        
    if args.use_lora and not args.load_model:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if args.use_clm else TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        
    if args.gradient_checkpointing_enable:
        assert (hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is None) or not hasattr(accelerator.state, "deepspeed_plugin")
        # assert (hasattr(accelerator.state, "fsdp_plugin") and accelerator.state.fsdp_plugin is None) or not hasattr(accelerator.state, "fsdp_plugin")
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
    if hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin:
        logger.info(f"deep speed state = {accelerator.state.deepspeed_plugin} and deep speed config = {str(accelerator.state.deepspeed_plugin.deepspeed_config)}")
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
        
    if hasattr(accelerator.state, "fsdp_plugin") and accelerator.state.fsdp_plugin:
        logger.info(f"FSDP speed state = {accelerator.state.fsdp_plugin} and FSDP speed config = {str(accelerator.state.fsdp_plugin)}")
        
    train_dataloader, training_eval_dataloader, eval_dataloader = get_dataloaders(args, accelerator, tokenizer, model)
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel()
                           for p in model.parameters())

    logger.info(f"Number of trainable parameters: {(trainable_params/(1024*1024)):.2f} Million, Total Parameter = {(total_params/(1024*1024)):.2f} Million")
    fsdp_model_copy_for_eval_only = None
    if args.fsdp or args.use_8bit_optim:
        assert (hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is None) or not hasattr(accelerator.state, "deepspeed_plugin")
        fsdp_model_copy_for_eval_only = deepcopy(model).to("cpu")
        if args.fsdp:
            if args.use_lora:
                accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
            model = accelerator.prepare(model)
            
        
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    not_use_dummy_opt_stage3 = not hasattr(accelerator.state, "deepspeed_plugin") or accelerator.state.deepspeed_plugin is None \
    or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config \
    or accelerator.state.deepspeed_plugin.deepspeed_config["zero_stage"] != 3
    
    if args.use_8bit_optim:
        import bitsandbytes as bnb
        assert not_use_dummy_opt_stage3
        assert (hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is None) or not hasattr(accelerator.state, "deepspeed_plugin")
        optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate)
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )            

    else:
        optimizer_cls = (
            torch.optim.AdamW
            if not_use_dummy_opt_stage3
            else DummyOptim
         )
        optimizer = optimizer_cls(
            optimizer_grouped_parameters, lr=args.learning_rate)
    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
      
    not_use_dummy_scheduler_stage3 = not hasattr(accelerator.state, "deepspeed_plugin") or accelerator.state.deepspeed_plugin is None \
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config \
        or accelerator.state.deepspeed_plugin.deepspeed_config["zero_stage"] != 3
    if not_use_dummy_scheduler_stage3:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
             optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
         )

    logger.warning(f"Optimiser = {optimizer}, Scheduler = {lr_scheduler}, not use_dummy_opt_stage3 = {not_use_dummy_opt_stage3}, not use_dummy_scheduler_stage3 = {not_use_dummy_scheduler_stage3}")
    # Prepare everything with our `accelerator`.
    if args.fsdp:
        optimizer, train_dataloader, training_eval_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, training_eval_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, training_eval_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, training_eval_dataloader, eval_dataloader, lr_scheduler
        )
        
    if args.generate_proba:
        generate_proba(model, tokenizer, accelerator, training_eval_dataloader, args)
        exit()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("summarization_no_trainer", experiment_config)
    # Zero Shot evaluation
    if args.zero_shot_evaluation:
        assert accelerator.state.deepspeed_plugin is None
        result_train = evaluate_model(model, tokenizer, accelerator, training_eval_dataloader, args, 
                                      result_key="train", fsdp_model_copy_for_eval_only=fsdp_model_copy_for_eval_only)
        result = evaluate_model(model, tokenizer, accelerator, eval_dataloader, args, 
                                result_key=None, fsdp_model_copy_for_eval_only=fsdp_model_copy_for_eval_only)
        result.update(result_train)
        result["epoch"] = 0
        result["step"] = 0
        result["lr"] = optimizer.param_groups[0]['lr']
        logger.info(f"Zero shot results = {result}")
        accelerator.log(result, step=0)

    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    gc.collect()
    torch.cuda.empty_cache()
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                if args.use_clm:
                    clm_params = {k.replace("clm_", ""): v for k, v in batch.items() if "clm_" in k}
                    non_clm_params = {k: v for k, v in batch.items() if k not in clm_params}
                    batch = clm_params
                    batch.update(non_clm_params)
                token_proba = batch["proba"] if "proba" in batch else None
                _ = [batch.pop(k, None) for k in ["proba", "proba1", "proba2"]]
                outputs = model(**batch)
                non_token_loss = outputs.loss.item()
                # print(f"NT LOSS = {non_token_loss}")
                if args.token_weights:
                    lm_logits = outputs.logits
                    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
                    # move labels to correct device to enable PP
                    labels = batch["labels"]
                    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                    element_wise_loss = loss
                    loss = (token_proba.view(-1) * loss)
                    element_wise_loss_token_mul = loss
                    loss = loss.mean()
                    # print(f"Weighted LOSS = {loss.item()}")
                    # if np.isnan(loss.item()):
                    #     logger.info(list(zip(element_wise_loss.tolist(), element_wise_loss_token_mul.tolist(), token_proba.view(-1).tolist(), labels.view(-1).tolist())))
                else:
                    loss = outputs.loss
                # We keep track of the loss at each epoch
                loss_record = loss.detach().float()
                total_loss += loss_record
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_description(f"Epoch {epoch} - Completed Step {completed_steps}, step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - loss: {loss_record.item():.4f} - NT loss: {non_token_loss:.4f}")  
                if np.isnan(float(loss_record.item())):
                    raise ValueError(f"Epoch {epoch} - Completed Step {completed_steps}, step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - loss: {loss_record.item():.4f}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    save_model_and_state(args.save_model, accelerator, model, tokenizer, args.output_dir, sub_dir=f"step_{completed_steps}", fsdp=args.fsdp, use_8bit_optim=args.use_8bit_optim, result_dict=None)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        if epoch == args.num_train_epochs - 1:
            result_train = evaluate_model(model, tokenizer, accelerator, training_eval_dataloader, args, 
                                          result_key="train", fsdp_model_copy_for_eval_only=fsdp_model_copy_for_eval_only)
            result = evaluate_model(model, tokenizer, accelerator, eval_dataloader, args, 
                                    result_key=None, fsdp_model_copy_for_eval_only=fsdp_model_copy_for_eval_only)
            result.update(result_train)
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            result["lr"] = optimizer.param_groups[0]['lr']
            logger.info(result)
            accelerator.log(result, step=completed_steps)
            
        else:
            result = evaluate_model(model, tokenizer, accelerator, eval_dataloader, args, 
                                    result_key=None, fsdp_model_copy_for_eval_only=fsdp_model_copy_for_eval_only)
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            result["lr"] = optimizer.param_groups[0]['lr']
            logger.info(result)
            accelerator.log(result, step=completed_steps)

        if args.checkpointing_steps == "epoch":
            save_model_and_state(args.save_model, accelerator, model, tokenizer, args.output_dir, sub_dir=f"epoch_{epoch}", fsdp=args.fsdp,use_8bit_optim=args.use_8bit_optim, result_dict=None)
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    logger.info(" **** Finished Training ****")
    save_model_and_state(args.save_model, accelerator, model, tokenizer, args.output_dir, sub_dir="", fsdp=args.fsdp, use_8bit_optim=args.use_8bit_optim, result_dict=result)
    accelerator.end_training()


def get_state_dict(unwrapped_model, accelerator, fsdp=False, rank0_only=True):
    if fsdp:
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state = accelerator.get_state_dict(unwrapped_model)
            return state
    else:
        state_dict = accelerator.get_state_dict(unwrapped_model)
        return state_dict
    
def save_model_and_state(save_model, accelerator, model, tokenizer, output_dir, sub_dir="", fsdp=False, use_8bit_optim=False, result_dict=None):
    accelerator.wait_for_everyone()
    output_dir = os.path.join(output_dir, sub_dir) if isinstance(sub_dir, str) else output_dir
    if not use_8bit_optim:
        accelerator.save_state(output_dir)
    logger.info(f"[save_model_and_state]: Saved Training state in {output_dir}")
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict=get_state_dict(unwrapped_model, accelerator, fsdp)
    if accelerator.is_local_main_process:
        if save_model:
            logger.info("[save_model_and_state]: Get state dict for saving..")
            save_state_dict(state_dict, output_dir, save_model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
            )
            logger.info(f"[save_model_and_state]:Saved Model state in {os.path.join(output_dir, save_model)}")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            logger.info(f"[save_model_and_state]: Saved Tokenizer state in {output_dir}")
        if result_dict and isinstance(result_dict, dict):
            all_results = {f"{k}": v for k, v in result_dict.items()}
            with open(os.path.join(output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)
            logger.info(f"[save_model_and_state]: Saved Final Results in {os.path.join(output_dir, 'all_results.json')}")

if __name__ == "__main__":
    main()
