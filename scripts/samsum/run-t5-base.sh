MODEL="t5-base"
bs=16
lr=2e-4
weight_decay=0.001
dataset="samsum"
gradient_accumulation_steps=4
baseline_epochs=20
epochs=15
combined_epochs=12
num_warmup_steps=180
N_FOLDS=2
proba_dataset=outputs/${MODEL}/${dataset}/folds_${N_FOLDS}_combined
proba_column="proba_w3"
additional_args="--no_additional_args"
seed=37


scripts/run-baseline.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${baseline_epochs} ${num_warmup_steps} ${additional_args} ${lr} ${weight_decay} ${seed}
# scripts/run-fractional.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${epochs} ${num_warmup_steps} ${N_FOLDS} 0 ${additional_args} ${lr} ${weight_decay} ${seed}
# scripts/run-fractional.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${epochs} ${num_warmup_steps} ${N_FOLDS} 1 ${additional_args} ${lr} ${weight_decay} ${seed}
# python scripts/combining_dataset_folds.py ${MODEL} ${dataset} ${N_FOLDS}
# scripts/run-combined.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${combined_epochs} ${num_warmup_steps} ${N_FOLDS} ${proba_column} ${proba_dataset} ${additional_args} ${lr} ${weight_decay} ${seed}



