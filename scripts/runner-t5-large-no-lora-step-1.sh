# l25
MODEL="t5-large"
bs=8
dataset="samsum"
max_grad_norm=1.0
gradient_accumulation_steps=4
epochs=10

export WANDB_PROJECT="summarization"
export WANDB_NAME="${MODEL}_${dataset}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --use_fsdp --fsdp_offload_params true --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 1 --fsdp_transformer_layer_cls_to_wrap "T5Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT"\
    run_sum_lora.py \
    --model_name_or_path $MODEL \
    --dataset_name $dataset  \
    --pad_to_max_length \
    --max_grad_norm $max_grad_norm \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $((bs * 2)) \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_warmup_steps 140 \
    --lr_scheduler_type cosine \
    --num_train_epochs $epochs \
    --report_to wandb \
    --output_dir outputs/${MODEL}/${dataset} \
    --fsdp\
    --gradient_checkpointing_enable\
    --fraction_dataset --n_dataset_fractions 2 --train_fraction_number 0 \
    --max_source_length 512 --max_target_length 128\
    --seed 42
    
# --fraction_dataset --n_dataset_fractions 2 --train_fraction_number 1 \