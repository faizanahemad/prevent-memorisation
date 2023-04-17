from datasets import Dataset, load_from_disk
import torch
import numpy as np
f1 = Dataset.load_from_disk("outputs/t5-large/samsum/fold1")
f2 = Dataset.load_from_disk("outputs/t5-large/samsum/fold2")

from datasets import concatenate_datasets

f1 = f1.rename_column("proba", "proba1")
f2 = f2.rename_column("proba", "proba2")
concatenated_dataset = concatenate_datasets([f1, f2], axis=1)

concatenated_dataset = concatenated_dataset.map(lambda x: {"proba": [1 - np.sqrt(np.abs(a - b)) for a, b in zip(x["proba1"], x["proba2"])]})

concatenated_dataset.save_to_disk("outputs/t5-large/samsum/fold1_fold2_combined")
# %history -f combining_dataset_folds.py
