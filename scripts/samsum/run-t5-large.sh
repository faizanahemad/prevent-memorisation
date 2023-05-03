MODEL="t5-large"
bs=4
dataset="samsum"
gradient_accumulation_steps=8
baseline_epochs=10
epochs=15
combined_epochs=11
num_warmup_steps=180
N_FOLDS=2
proba_dataset=outputs/${MODEL}/${dataset}/folds_${N_FOLDS}_combined
proba_column="proba_v11_cumulative_windowed"

# "inverted_jsd" -> 8 epochs

# scripts/run-baseline.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${baseline_epochs} ${num_warmup_steps}
# scripts/run-fractional.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${epochs} ${num_warmup_steps} ${N_FOLDS} 0
# scripts/run-fractional.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${epochs} ${num_warmup_steps} ${N_FOLDS} 1
# python scripts/combining_dataset_folds.py ${MODEL} ${dataset} ${N_FOLDS}
scripts/run-combined.sh ${MODEL} ${dataset} ${bs} ${gradient_accumulation_steps} ${combined_epochs} ${num_warmup_steps} ${N_FOLDS} ${proba_column} ${proba_dataset}



