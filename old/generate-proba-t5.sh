MODEL="t5-large"
bs=32
dataset="samsum"
max_grad_norm=1.0
gradient_accumulation_steps=1
epochs=10

export WANDB_PROJECT="summarization"
export WANDB_NAME="${MODEL}_${dataset}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true

FOLD=1

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=fp16 --use_fsdp --fsdp_offload_params true --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_sharding_strategy 1 --fsdp_transformer_layer_cls_to_wrap "T5Block" --fsdp_backward_prefetch_policy "BACKWARD_PRE" --fsdp_state_dict_type "FULL_STATE_DICT"\
    run_sum_lora.py \
    --model_name_or_path $MODEL \
    --dataset_name $dataset  \
    --pad_to_max_length \
    --max_grad_norm $max_grad_norm \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $((bs / 4)) \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_warmup_steps 50 \
    --lr_scheduler_type cosine \
    --num_train_epochs $epochs \
    --report_to wandb \
    --output_dir outputs/${MODEL}/${dataset} \
    --max_source_length 512 --max_target_length 128\
    --gradient_checkpointing_enable\
    --fsdp\
    --seed 42 --load_model outputs/${MODEL}/${dataset}/model.pt --generate_proba --proba_store outputs/${MODEL}/${dataset}/fold${FOLD}
    
mv outputs/${MODEL}/${dataset}/model.pt outputs/${MODEL}/${dataset}/fold${FOLD}
mv outputs/${MODEL}/${dataset}/all_results.json outputs/${MODEL}/${dataset}/fold${FOLD}/all_results.json
mv outputs/${MODEL}/${dataset}/pytorch_model.bin outputs/${MODEL}/${dataset}/fold${FOLD}/pytorch_model.bin
    
#  