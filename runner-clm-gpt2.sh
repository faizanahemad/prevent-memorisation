# l25
MODEL="gpt2-medium"
bs=24
dataset="samsum"
max_grad_norm=1.0
gradient_accumulation_steps=4
epochs=2

export WANDB_PROJECT="summarization"
export WANDB_NAME="${MODEL}_${dataset}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --use_fsdp --fsdp_offload_params true --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 1 --fsdp_transformer_layer_cls_to_wrap "GPT2Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT"\
    run_sum_lora.py \
    --model_name_or_path $MODEL \
    --dataset_name $dataset  \
    --pad_to_max_length \
    --max_grad_norm $max_grad_norm \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $((bs * 1)) \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --num_warmup_steps 100 \
    --lr_scheduler_type cosine \
    --num_train_epochs $epochs \
    --report_to wandb \
    --output_dir outputs/${MODEL}/${dataset} \
    --max_source_length 512 --max_target_length 128\
    --gradient_checkpointing_enable\
    --use_clm\
    --fsdp\
    --seed 42
    
# --fraction_dataset --n_dataset_fractions 24 --train_fraction_number 0\
# --gradient_checkpointing_enable