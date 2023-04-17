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
import gc
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training,  TaskType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from peft.utils.other import fsdp_auto_wrap_policy

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
from utils import *


def generate_and_logits(model, tokenizer, accelerator, dataloader, args, save_name, fsdp_model_copy_for_eval_only=None):
    assert (hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is None) or not hasattr(accelerator.state, "deepspeed_plugin")
    unwrapped_model = accelerator.unwrap_model(model)
    if args.fsdp or args.use_8bit_optim:
        assert fsdp_model_copy_for_eval_only
        state_dict = get_state_dict(unwrapped_model, accelerator, fsdp=args.fsdp, rank0_only=False)
        fsdp_model_copy_for_eval_only.load_state_dict(state_dict)
        del state_dict
        fsdp_model_copy_for_eval_only = fsdp_model_copy_for_eval_only.to(accelerator.device)
        unwrapped_model = fsdp_model_copy_for_eval_only
    progress_bar = tqdm(range(len(dataloader)), desc="evaluate", disable=not accelerator.is_local_main_process)
    unwrapped_model.config.use_cache = True
    unwrapped_model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            if args.use_clm:
                batch = {k.replace("clm_", ""): v for k, v in batch.items() if "clm_" in k}
            outputs = model(**batch)
        input_ids = batch["input_ids"].reshape(-1)
        logits = outputs.logits
        b, s = logits.shape[:2]
        logits = logits.reshape(-1, x.shape[-1])
        proba = logits[torch.arange(b*s), input_ids].reshape(b, s)
        proba = accelerator.gather_for_metrics(proba).cpu().numpy()
        
