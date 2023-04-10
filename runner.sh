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
    --mixed_precision=fp16 --multi_gpu \
    run_summarization_no_trainer.py \
    --model_name_or_path $MODEL \
    --dataset_name $dataset  \
    --pad_to_max_length \
    --max_grad_norm $max_grad_norm \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing_enable \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type cosine \
    --num_train_epochs 10 \
    --report_to wandb \
    --output_dir outputs/${MODEL}/${dataset} \
    --zero_shot_evaluation \
    --seed 42
    
    
# --use_deepspeed --zero_stage 2 --offload_optimizer_device "cpu" --offload_param_device "cpu" --gradient_accumulation_steps $gradient_accumulation_steps --gradient_clipping $max_grad_norm