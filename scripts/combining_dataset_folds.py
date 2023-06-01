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
    arr = np.array(arr).astype('float64')
    return np.abs(np.subtract.outer(arr, arr)).max()


def cumulative_product_window(array, window):
    array = np.array(array).astype('float64')
    # Create an array with the same size as the input array
    result = np.empty_like(array).astype('float64')

    # Pad the input array with ones to the left
    padded_array = np.pad(array, (window - 1, 0), mode='constant', constant_values=(1,))

    # Calculate the cumulative product within the given window length
    for i in range(len(array)):
        result[i] = np.prod(padded_array[i:i + window])

    return result


def windowed_applicator(array, window):
    array = np.array(array)
    result = np.empty_like(array)
    
    padded_array = np.pad(array, (0, window - 1), mode='edge')
    for i in range(len(array)):
        result[i] = np.mean(padded_array[i:i +window])
    return result # / np.mean(result)


    
def map_fn(row):
    probas = list(zip(*[row[f"proba{FOLD}"] for FOLD in range(N_FOLD)]))
    results_dict = dict()
    for window in [2,3,5,8,10]:
        probas_cumulative = list(zip(*[cumulative_product_window(row[f"proba{FOLD}"], window) for FOLD in range(N_FOLD)]))
        proba_cumulative = [np.power(np.min(d), (2/window)) for d in probas_cumulative]
        proba_mpd_cumulative = [np.power(np.min(d), (2/window))*(1 - (max_pairwise_distance(d))) for d in probas_cumulative]
        
        proba_cumulative_windowed = windowed_applicator(proba_v10_cumulative, window)
        proba_mpd_cumulative_windowed = windowed_applicator(proba_v11_cumulative, window)
        rdd = {
            f"proba_w{window}": proba_cumulative_windowed,
            f"proba_mpd_w{window}": proba_mpd_cumulative_windowed,
              }
        results_dict.update(rdd)
    return results_dict

concatenated_dataset = concatenated_dataset.map(map_fn)

# map(lambda x: {"proba": [1 - np.sqrt(np.abs(a - b)) for a, b in zip(x["proba1"], x["proba2"])]})

concatenated_dataset.save_to_disk(f"outputs/{model}/{dataset}/folds_{N_FOLD}_combined")
print(concatenated_dataset[0])
# %history -f combining_dataset_folds.py
