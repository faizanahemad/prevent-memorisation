# l25
MODEL="t5-small"
bs=8
dataset="samsum"
max_grad_norm=1.0
gradient_accumulation_steps=4

export WANDB_PROJECT="summarization"
export WANDB_NAME="${MODEL}_${dataset}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=fp16 --multi_gpu\
    run_summarization \
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
    --num_train_epochs 10 \
    --report_to wandb \
    --output_dir outputs/${MODEL}/${dataset} \
    --zero_shot_evaluation\
    --max_source_length 512 --max_target_length 128\
    --seed 42
    
    
# --use_deepspeed --zero_stage 2 --offload_optimizer_device "cpu" --offload_param_device "cpu" --gradient_accumulation_steps $gradient_accumulation_steps --gradient_clipping $max_grad_norm

# --multi_gpu
# --zero_shot_evaluation
# --gradient_checkpointing_enable

# --use_fsdp --fsdp_offload_params true --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 1 --fsdp_transformer_layer_cls_to_wrap "T5Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT"
# --use_fsdp --fsdp_offload_params true --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 2 --fsdp_transformer_layer_cls_to_wrap "T5Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT"
# --fsdp

# --max_source_length 1024 --max_target_length 256\