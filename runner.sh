# l25
MODEL="t5-small"
bs=32
dataset="samsum"
max_grad_norm=1.0
gradient_accumulation_steps=1
epochs=10

export WANDB_PROJECT="summarization"
export WANDB_NAME="${MODEL}_${dataset}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=fp16 --use_fsdp --fsdp_offload_params false --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 1 --fsdp_transformer_layer_cls_to_wrap "T5Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT" \
    run_summarization.py \
    --model_name_or_path $MODEL \
    --dataset_name $dataset  \
    --pad_to_max_length \
    --max_grad_norm $max_grad_norm \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $((bs * 4)) \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type cosine \
    --num_train_epochs $epochs \
    --report_to wandb \
    --output_dir outputs/${MODEL}/${dataset} \
    --fsdp\
    --zero_shot_evaluation\
    --max_source_length 512 --max_target_length 128\
    --seed 42
    
    
# --use_deepspeed --zero_stage 2 --offload_optimizer_device "cpu" --offload_param_device "cpu" --gradient_accumulation_steps $gradient_accumulation_steps --gradient_clipping $max_grad_norm

# --multi_gpu
# --zero_shot_evaluation
# --gradient_checkpointing_enable
# --load_model outputs/${MODEL}/${dataset}/model.pt

### FSDP details
# --use_fsdp --fsdp_offload_params false --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 1 --fsdp_transformer_layer_cls_to_wrap "T5Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT"
# --fsdp
# bs=32
# gradient_accumulation_steps=1

# --max_source_length 1024 --max_target_length 256\

# Why we don't evaluate in FSDP: model.generate isn't fsdp supported, but LM pretraining like auto-regressive tasks may still work in eval.
# https://github.com/pytorch/pytorch/issues/82461
# https://github.com/huggingface/accelerate/issues/570
# cpu-offload in fsdp is only supported for bf16 and fp16, for t5-large and some models fp16 isn't supported.
# https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
# https://github.com/huggingface/transformers/issues/11461
# https://github.com/huggingface/transformers/issues/10830

# More FSDP 
# https://huggingface.co/blog/pytorch-fsdp
# https://github.com/huggingface/accelerate/issues/919