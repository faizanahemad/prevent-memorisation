MODEL=$1
dataset=$2
N_FOLD=$3

export WANDB_PROJECT="summarization"
export WANDB_NAME="${MODEL}_${dataset}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu\
    run_jsd.py \
    --model_name_or_path $MODEL \
    --dataset_name $dataset  \
    --report_to wandb \
    --load_first_model outputs/${MODEL}/${dataset}/fold_${N_FOLD}_0/model.pt \
    --load_second_model outputs/${MODEL}/${dataset}/fold_${N_FOLD}_1/model.pt \
    --output_dir outputs/${MODEL}/${dataset} \
    --max_source_length 512 --max_target_length 128\
    --proba_store outputs/${MODEL}/${dataset}/fold_${N_FOLD}_${FOLD}_jsd \
    --per_device_eval_batch_size 16 --pad_to_max_length \
    --seed 42
    
    
# python run_jsd_non_accelerate.py \
#     --model_name_or_path $MODEL \
#     --dataset_name $dataset  \
#     --report_to wandb \
#     --load_first_model outputs/${MODEL}/${dataset}/fold_${N_FOLD}_0/model.pt \
#     --load_second_model outputs/${MODEL}/${dataset}/fold_${N_FOLD}_1/model.pt \
#     --output_dir outputs/${MODEL}/${dataset} \
#     --max_source_length 512 --max_target_length 128\
#     --proba_store outputs/${MODEL}/${dataset}/fold_${N_FOLD}_${FOLD}_jsd \
#     --per_device_eval_batch_size 1 --pad_to_max_length \
#     --seed 42