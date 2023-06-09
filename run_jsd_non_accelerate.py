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
from torch import nn
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
        "--ignore_pad_token_for_loss",
        action="store_true",
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
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
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
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
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
    
    
    
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
        '--load_first_model',
        type=str,
        default=None,
        required=True,
        help='Path to the model state dict'
    )
    parser.add_argument(
        '--load_second_model',
        type=str,
        default=None,
        required=True,
        help='Path to the model state dict'
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    
    parser.add_argument(
        '--proba_store',
        type=str,
        default=None,
        help='Path to Store generated Probas dict'
    )
    args = parser.parse_args()
    args.use_clm = False
    args.fraction_dataset = False
    args.token_weights = False
    # Sanity checks
    
    return args

from run_sum_lora import get_dataloaders

def recursive_detach(t):
    if torch.is_tensor(t):
        return t.detach()
    elif isinstance(t, (list)):
        return list(map(recursive_detach, t))
    elif isinstance(t, (tuple)):
        return tuple(map(recursive_detach, t))
    elif isinstance(t, (dict)):
        return {k: recursive_detach(v) for k, v in t.items()}
    elif t is None:
        return t
    else:
        raise ValueError
        
def recursive_to_device(t, device):
    if torch.is_tensor(t):
        return t.to(device)
    elif isinstance(t, (list)):
        return list([recursive_to_device(x, device) for x in t])
    elif isinstance(t, (tuple)):
        return tuple([recursive_to_device(x, device) for x in t])
    elif isinstance(t, (dict)):
        return {k: recursive_to_device(v, device) for k, v in t.items()}
    elif t is None:
        return t
    else:
        raise ValueError
        
def recursive_get_size(t):
    if torch.is_tensor(t):
        return t.numel()*4
    elif isinstance(t, (list)):
        return sum(list(map(recursive_get_size, t)))
    elif isinstance(t, (tuple)):
        return sum(tuple(map(recursive_get_size, t)))
    elif isinstance(t, (dict)):
        return sum([recursive_get_size(v) for k, v in t.items()])
    elif t is None:
        return t
    else:
        raise ValueError
        

def run_decoder_step(model, input_ids, past_key_values, encoder_hidden_states, attention_mask, actual, num_samples, device):
    past_key_values = recursive_to_device(past_key_values, device)
    decoder_outputs = model.decoder(input_ids=input_ids, past_key_values=past_key_values, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, use_cache=True,)
    past_key_values=recursive_detach(decoder_outputs.past_key_values)
    past_key_values = recursive_to_device(past_key_values, torch.device("cpu"))
    print("%.4f" % (sys.getsizeof(past_key_values)/1_000_000), "%.4f" % (recursive_get_size(past_key_values)/1_000_000))
    sequence_output = decoder_outputs[0]
    if model.config.tie_word_embeddings:
        sequence_output = sequence_output * (model.model_dim**-0.5)
    lm_logits = model.lm_head(sequence_output).softmax(dim=-1).squeeze(0)
    proba = lm_logits[-1, actual[0, -1]].detach().item()
    logits = lm_logits[-1].squeeze().detach().cpu().tolist()
    samples = None
    if num_samples:
        samples = (lm_logits[-1]).multinomial(num_samples=num_samples, replacement=True).detach()
    return {"proba": proba, "logits": logits, "samples": samples, "past_key_values": past_key_values}
    

def get_three_step_proba(model, labels, input_ids, attention_mask, temperature=1.0, num_samples=1):
    original_lables = deepcopy(labels)
    labels = model._shift_right(labels)
    probas = []
    logits = []
    with torch.no_grad():
        encoder_output = model.encoder(input_ids, attention_mask)
        encoder_hidden_states = encoder_output[0].detach()
        past_key_values = None
        
        for i in range(labels.shape[-1]-3):
            lbl = labels[..., :(i+1)]
            actual = original_lables[..., :(i+1)]
            decoder_step_out = run_decoder_step(model, lbl, past_key_values, encoder_hidden_states, attention_mask, actual, num_samples, labels.device)
            past_key_values=decoder_step_out["past_key_values"]
            if i == 0:
                proba = decoder_step_out["proba"]
                logits.append(decoder_step_out["logits"])
                probas.append(proba)
            
            index = decoder_step_out["samples"]
            del decoder_step_out
            # print(actual[..., -1], proba, index)
            actual = original_lables[..., :(i+2)]
            new_probas = []
            new_logits = []
            probas_l2 = []
            logits_l2 = []
            for idx in index:

                lbx = torch.cat([lbl, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                decoder_step_out = run_decoder_step(model, lbx, past_key_values, encoder_hidden_states, attention_mask,actual, num_samples, labels.device)
                if i == 0:
                    proba = decoder_step_out["proba"]
                    new_probas.append(proba)
                    new_logits.append(decoder_step_out["logits"])
                pkv_v2 = decoder_step_out["past_key_values"]
                index_l2 = decoder_step_out["samples"]
                del decoder_step_out
                actual = original_lables[..., :(i+3)]
                
                for idx in index_l2:
                    lby = torch.cat([lbx, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                    decoder_step_out = run_decoder_step(model, lby, pkv_v2, encoder_hidden_states, attention_mask,actual, 0, labels.device)
                    proba = decoder_step_out["proba"]
                    probas_l2.append(proba)
                    logits_l2.append(decoder_step_out["logits"])
                    del decoder_step_out
                del pkv_v2
            torch.cuda.empty_cache()
                
                
                
            if i == 0:
                probas.append(np.mean(new_probas))
                logits.append(np.mean(new_logits, axis=0))
            probas.append(np.mean(probas_l2))
            logits.append(np.mean(logits_l2, axis=0))
    return {"probas": probas, "logits": torch.tensor(np.array(logits)).detach()}

def calculate_jsd(x, y):
    jsd_m = 0.5 * (x + y)
    jsd = 0.5 * nn.KLDivLoss(reduction='none', log_target=False)(torch.log(x), jsd_m) + 0.5 * nn.KLDivLoss(reduction='none', log_target=False)(torch.log(y), jsd_m)
    jsd = jsd.sum(-1)
    return jsd

def generate_proba(model_1, model_2, tokenizer, dataloader, args):
    args.per_device_eval_batch_size == 1
    assert "t5" in args.model_name_or_path.lower()
    progress_bar = tqdm(range(len(dataloader)), desc="generate Logits", disable=False)

    model_1 = (model_1.module if hasattr(model_1, "module") else model_1)

    model_2 = (model_2.module if hasattr(model_2, "module") else model_2)
    all_m1_proba = []
    all_m2_proba = []
    all_jsd = []
    all_inverse_jsd = []
    all_inverted_jsd = []
    
    assert "t5" in args.model_name_or_path.lower()
    model_1.eval()
    model_2.eval()
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    
    for step, batch in enumerate(dataloader):
        labels = batch["labels"].to("cuda")
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        with torch.no_grad():
            m1_out = get_three_step_proba(model_1, labels, input_ids, attention_mask,)
        m1_proba = m1_out["probas"]
        with torch.no_grad():
            m2_out = get_three_step_proba(model_2, labels, input_ids, attention_mask,)
        m2_proba = m2_out["probas"]
        
        m1_logits = m1_out["logits"].to("cuda")
        m2_logits = m2_out["logits"].to("cuda")
        jsd = calculate_jsd(m1_logits, m2_logits)
        max_jsd = jsd.max()
        min_jsd = jsd.min()
        inverted_jsd = (max_jsd - jsd + min_jsd) / (max_jsd - min_jsd)
        inverse_jsd = 1. / (2*jsd + 0.05)
        inverse_jsd = (inverse_jsd - inverse_jsd.min()) / (inverse_jsd.max() - inverse_jsd.min())
        
        # pad
        m1_proba = m1_proba + [0.0] * (max_target_length - len(m1_proba))
        m1_proba = torch.tensor(m1_proba).to("cuda")
        
        m2_proba = m2_proba + [0.0] * (max_target_length - len(m2_proba))
        m2_proba = torch.tensor(m2_proba).to("cuda")
        
        jsd = jsd.tolist()
        jsd = jsd + [0.0] * (max_target_length - len(jsd))
        jsd = torch.tensor(jsd).to("cuda")
        
        inverted_jsd = inverted_jsd.tolist()
        inverted_jsd = inverted_jsd + [0.0] * (max_target_length - len(inverted_jsd))
        inverted_jsd = torch.tensor(inverted_jsd).to("cuda")
        
        inverse_jsd = inverse_jsd.tolist()
        inverse_jsd = inverse_jsd + [0.0] * (max_target_length - len(inverse_jsd))
        inverse_jsd = torch.tensor(inverse_jsd).to("cuda")
        
        

        all_m1_proba.append(m1_proba.cpu())
        all_m2_proba.append(m2_proba.cpu())
        all_jsd.append(jsd.cpu())
        all_inverse_jsd.append(inverse_jsd.cpu())
        all_inverted_jsd.append(inverted_jsd.cpu())
        progress_bar.update(1)

    all_m1_proba = torch.cat(all_m1_proba, 0).tolist()
    all_m2_proba = torch.cat(all_m2_proba, 0).tolist()
    all_jsd = torch.cat(all_jsd, 0).tolist()
    all_inverse_jsd = torch.cat(all_inverse_jsd, 0).tolist()
    all_inverted_jsd = torch.cat(all_inverted_jsd, 0).tolist()

    ds = Dataset.from_dict({"proba0": all_m1_proba, "proba1": all_m2_proba, 
                            "jsd": all_jsd, "inverse_jsd": all_inverse_jsd, 
                            "inverted_jsd": all_inverted_jsd})
    for index in random.sample(range(len(ds)), 1):
        logger.info(f"Sample {index} of the training set: {ds[index]}.")
    logger.info(f"  Num examples = {len(ds)}")
    ds.save_to_disk(args.proba_store)
    

def main():
    args = parse_args()

    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] 
    )
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
    
    if args.seed is not None:
        set_seed(args.seed)

    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_clm:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, 
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config, # torch_dtype=torch.float16
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
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
        
    
    model_1 = deepcopy(model)
    state_dict = torch.load(args.load_first_model, map_location='cpu')
    model_1.load_state_dict(state_dict)
    model_2 = deepcopy(model)
    state_dict = torch.load(args.load_second_model, map_location='cpu')
    model_2.load_state_dict(state_dict)
    del state_dict
    
    for p in model_1.parameters():
        p.requires_grad = False
    for p in model_2.parameters():
        p.requires_grad = False
    
        
     
    from argparse import Namespace
    from run_sum_lora import Preprocess
    from transformers import DataCollatorForSeq2Seq
    preprocess_function = Preprocess("dialogue", "summary", "summarize: ", tokenizer, args)
    dataset = load_dataset(args.dataset_name)
    column_names = dataset["train"].column_names
    train_dataset = dataset["train"].map(
                preprocess_function,
                batched=True,
                num_proc=16,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )
    max_target_length = args.max_target_length
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )
    dataloader = DataLoader(
            train_dataset, collate_fn=data_collator, batch_size=1
        )
    
    model_1 = model_1.to("cuda")
    generate_proba(model_1, model_1, tokenizer, dataloader, args)
    exit()


if __name__ == "__main__":
    main()
