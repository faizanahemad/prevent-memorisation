MODEL="t5-large"
bs=12
dataset="samsum"
gradient_accumulation_steps=4
baseline_epochs=10
epochs=15
combined_epochs=10
num_warmup_steps=150
proba_column="proba_v3"
N_FOLDS=2

scripts/run-baseline.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${baseline_epochs} ${num_warmup_steps}
scripts/run-fractional.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${epochs} ${num_warmup_steps} ${N_FOLDS} 0
scripts/run-fractional.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${epochs} ${num_warmup_steps} ${N_FOLDS} 1
python scripts/combining_dataset_folds.py ${MODEL} ${dataset} ${N_FOLDS}
scripts/run-combined.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${combined_epochs} ${num_warmup_steps} ${N_FOLDS} ${proba_column}



