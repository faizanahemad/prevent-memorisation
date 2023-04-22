import sys
args = sys.argv

model=args[1]
dataset=args[2]
N_FOLD=int(args[3])
from datasets import Dataset, load_from_disk
import torch
import numpy as np
dsets = []
for FOLD in range(N_FOLD):
    ds = Dataset.load_from_disk(f"outputs/{model}/{dataset}/fold_{N_FOLD}_{FOLD}")
    ds = ds.rename_column("proba", f"proba{FOLD}")
    dsets.append(ds)
    


from datasets import concatenate_datasets
concatenated_dataset = concatenate_datasets(dsets, axis=1)


def max_pairwise_distance(arr):
    return np.abs(np.subtract.outer(arr, arr)).max()

def proba_v3_6(x, y):
    return (1 - np.abs(x - y))**2 * min(x, y)
# np.minimum(x.flatten(), y.flatten()).reshape(x_s)

def map_fn(row):
    probas = list(zip(*[row[f"proba{FOLD}"] for FOLD in range(N_FOLD)]))
    proba_v2 = [max(0.001, np.product(d)) for d in probas]
    proba_v1 = [max(0.001, 1 - (max_pairwise_distance(d)**1)) for d in probas]
    proba_v3 = [max(0.001, (1 - (max_pairwise_distance(d)**1)) * np.min(d)) for d in probas]
    return {"proba_v1": proba_v1, "proba_v2": proba_v2, "proba_v3": proba_v3}

concatenated_dataset = concatenated_dataset.map(map_fn)

# map(lambda x: {"proba": [1 - np.sqrt(np.abs(a - b)) for a, b in zip(x["proba1"], x["proba2"])]})

concatenated_dataset.save_to_disk(f"outputs/{model}/{dataset}/folds_{N_FOLD}_combined")
print(concatenated_dataset[0])

def test(fn):
    fn(0.1, 0.2) < fn(0.2, 0.3) < fn(0.3, 0.4) < fn(0.4, 0.5) < fn(0.5, 0.6) < fn(0.6, 0.7) < fn(0.7, 0.8) < fn(0.8, 0.9)
    for delta, start in [(0.01, 0.01), (0.1, 0.1), (0.2, 0.2)]:
        for i in range(int(1/delta)):
            assert fn(start, start + delta) < fn(start + delta, start + 2*delta)
            start = start + delta
    
    fn(0.1, 0.9) < fn(0.1, 0.8)  < fn(0.1, 0.7) < fn(0.1, 0.6) < fn(0.1, 0.5) < fn(0.1, 0.4) < fn(0.1, 0.3) < fn(0.1, 0.2)         
    for delta, start in [(0.01, 0.01), (0.1, 0.1), (0.2, 0.2)]:
        for i in range(int(1/delta)):
            print(start, start + i*delta, ">", start, start + (i+1)*delta)
            assert fn(start, start + i*delta) > fn(start, start + (i+1)*delta)  
    fn(0.1, 0.6) < fn(0.2, 0.2)
    
    

def proba_v3(x, y):
    return (1 - np.abs(x - y))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0))

def proba_v3_1(x, y):
    return (1 - np.sqrt(np.abs(x - y)))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0))

def proba_v3_2(x, y):
    return np.sqrt((1 - np.sqrt(np.abs(x - y)))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0)))

def proba_v3_3(x, y):
    return np.log1p((1 - np.sqrt(np.abs(x - y)))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0)))

def proba_v3_4(x, y):
    return (1 - np.sqrt(np.abs(x - y)))**2 * (x * y)

            
def proba_v3_5(x, y):
    return (1 - np.abs(x - y))**2 * (x * y)

def proba_v3_6(x, y):
    return (1 - np.abs(x - y))**2 * min(x, y)

def proba_v4(x, y):
    return np.sqrt((1 - np.abs(x - y))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0)))

def proba_v5(x, y):
    return np.exp((1 - np.abs(x - y))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0)))

def proba_v6(x, y):
    return (10 - np.log(np.max(x, y)/(np.min(x, y) + 0.001)))**2 * (np.clip(x, 0.1, 1.0) * np.clip(y, 0.1, 1.0))


    


# %history -f combining_dataset_folds.py
