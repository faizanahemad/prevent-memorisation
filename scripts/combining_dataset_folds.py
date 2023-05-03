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

def cumulative_product_window(array, window):
    array = np.array(array)
    # Create an array with the same size as the input array
    result = np.empty_like(array)

    # Pad the input array with ones to the left
    padded_array = np.pad(array, (window - 1, 0), mode='constant', constant_values=(1,))

    # Calculate the cumulative product within the given window length
    for i in range(len(array)):
        result[i] = np.prod(padded_array[i:i + window])

    return result


window = 3
def min_together(arr1, arr2, window):
    arr1_cp = cumulative_product_window(arr1, window)
    arr2_cp = cumulative_product_window(arr2, window)
    
    min_cp = 1 - 0.5 * np.abs(arr1_cp - arr2_cp)
    # min_cp = np.sqrt(arr1_cp * arr2_cp) *  min_cp**1
    return min_cp


def windowed_applicator(array, window):
    array = np.array(array)
    result = np.empty_like(array)
    
    padded_array = np.pad(array, (0, window - 1), mode='edge')
    for i in range(len(array)):
        result[i] = np.mean(padded_array[i:i +window])
    return result # / np.mean(result)


def fit_in_zero_one_range_min_max(initial_array):
    initial_array = np.array(initial_array)

    # Normalize the initial array
    min_value = initial_array.min()
    range_value = initial_array.max() - min_value
    normalized_array = (initial_array - min_value) / range_value

    # Scale the normalized array to the desired range (0.01 to 1)
    min_mapped_value = 0.01
    max_mapped_value = 1
    mapped_array = normalized_array * (max_mapped_value - min_mapped_value) + min_mapped_value
    return mapped_array

def fit_in_zero_one_range_robust(initial_array):
    initial_array = np.array(initial_array)

    p25, p75 = np.percentile(initial_array, [25, 75])

    # Robust scaling using the 25th and 75th percentile values
    iqr = p75 - p25
    robust_scaled_array = (initial_array - p25) / iqr

    # Scale the robust scaled array to the desired range (0.01 to 1)
    min_mapped_value = 0.01
    max_mapped_value = 1
    mapped_array = robust_scaled_array * (max_mapped_value - min_mapped_value) + min_mapped_value
    return mapped_array
    
def map_fn(row):
    probas = list(zip(*[row[f"proba{FOLD}"] for FOLD in range(N_FOLD)]))
    proba_v2 = [max(0.001, np.product(d)) for d in probas]
    proba_v1 = [max(0.001, 1 - (max_pairwise_distance(d))) for d in probas]
    proba_v3 = [max(0.001, (1 - (max_pairwise_distance(d))) * np.min(d)) for d in probas]
    
    # fn = lambda x, y: np.sqrt(x*y) * (1 - np.abs(x-y)) ** 2

    proba_v4 = [max(0.001, np.power(np.product(d), (1/len(d))) * (1 - (max_pairwise_distance(d)))**2 ) for d in probas]
    
    # func = lambda x, y: np.sqrt(x*y) * (1 - np.abs(x-y))
    proba_v5 = [max(0.001, np.power(np.product(d), (1/len(d))) * (1 - (max_pairwise_distance(d))) ** 1 ) for d in probas]
    
    # func = lambda x, y: (x+y)/2 * (1 - np.abs(x-y))
    proba_v6 = [max(0.001, np.mean(d) * (1 - (max_pairwise_distance(d)))**2 ) for d in probas]
    proba_v7 = [max(0.001, np.mean(d) * (1 - (max_pairwise_distance(d)))**1 ) for d in probas]
    proba_v8 = [max(0.001, np.product(d) * (1 - (max_pairwise_distance(d)))**2 ) for d in probas]
    
    proba_v9 = fit_in_zero_one_range_min_max([np.sqrt(np.min(d))  for d in probas])
    proba_v10 = [np.sqrt(np.min(d))  for d in probas]
    
    proba_v11 = [np.exp(np.min(d)) - 0.9999  for d in probas]
    
    proba_v12 = [np.exp(np.min(d) * (1 - (max_pairwise_distance(d)))) - 0.9999  for d in probas]
    
    window = 3
    probas_cumulative = list(zip(*[cumulative_product_window(row[f"proba{FOLD}"], window) for FOLD in range(N_FOLD)]))
    proba_v4_cumulative = [max(0.001, np.power(np.product(d), (1/len(d))) * (1 - (max_pairwise_distance(d)))**2 ) for d in probas_cumulative]
    proba_v7_cumulative = [max(0.001, np.mean(d) * (1 - (max_pairwise_distance(d)))**1 ) for d in probas_cumulative]
    proba_tv_approx_cumulative = [max(0.001, 1 - 0.5 * (max_pairwise_distance(d)) ) for d in probas_cumulative]
    proba_v9_cumulative = [np.sqrt(np.min(d)) for d in probas_cumulative]
    proba_v11_cumulative = [np.exp(np.min(d)) - 0.9999 for d in probas_cumulative]
    proba_v12_cumulative = [np.exp(np.min(d) * (1 - (max_pairwise_distance(d)))) - 0.9999  for d in probas_cumulative]
    
    proba_v4_cumulative_windowed = windowed_applicator(proba_v4_cumulative, window)
    proba_v7_cumulative_windowed = windowed_applicator(proba_v7_cumulative, window)
    proba_tv_approx_cumulative_windowed = windowed_applicator(proba_tv_approx_cumulative, window)
    proba_v9_cumulative_windowed = fit_in_zero_one_range_min_max(windowed_applicator(proba_v9_cumulative, window))
    proba_v10_cumulative_windowed = windowed_applicator(proba_v9_cumulative, window)
    proba_v11_cumulative_windowed = windowed_applicator(proba_v11_cumulative, window)
    proba_v12_cumulative_windowed = windowed_applicator(proba_v12_cumulative, window)
    
    ####
    window = 2
    probas_cumulative_w2 = list(zip(*[cumulative_product_window(row[f"proba{FOLD}"], window) for FOLD in range(N_FOLD)]))
    proba_v4_cumulative_w2 = [max(0.001, np.power(np.product(d), (1/len(d))) * (1 - (max_pairwise_distance(d)))**2 ) for d in probas_cumulative]
    proba_v7_cumulative_w2 = [max(0.001, np.mean(d) * (1 - (max_pairwise_distance(d)))**1 ) for d in probas_cumulative]
    proba_tv_approx_cumulative_w2 = [max(0.001, 1 - 0.5 * (max_pairwise_distance(d)) ) for d in probas_cumulative]
    proba_v9_cumulative_w2 = [np.sqrt(np.min(d)) for d in probas_cumulative]
    proba_v11_cumulative_w2 = [np.exp(np.min(d)) - 0.9999 for d in probas_cumulative]
    proba_v12_cumulative_w2 = [np.exp(np.min(d) * (1 - (max_pairwise_distance(d)))) - 0.9999  for d in probas_cumulative]
    
    proba_v4_cumulative_windowed_w2 = windowed_applicator(proba_v4_cumulative_w2, window)
    proba_v7_cumulative_windowed_w2 = windowed_applicator(proba_v7_cumulative_w2, window)
    proba_tv_approx_cumulative_windowed_w2 = windowed_applicator(proba_tv_approx_cumulative_w2, window)
    proba_v9_cumulative_windowed_w2 = fit_in_zero_one_range_min_max(windowed_applicator(proba_v9_cumulative_w2, window))
    proba_v10_cumulative_windowed_w2 = windowed_applicator(proba_v9_cumulative_w2, window)
    proba_v11_cumulative_windowed_w2 = windowed_applicator(proba_v11_cumulative_w2, window)
    proba_v12_cumulative_windowed_w2 = windowed_applicator(proba_v12_cumulative_w2, window)
    ####
    
    proba_v10_cumulative_w2 = proba_v9_cumulative_w2
    proba_v10_cumulative = proba_v9_cumulative
    proba_v9_cumulative_w2 = fit_in_zero_one_range_min_max(proba_v9_cumulative_w2)
    proba_v9_cumulative = fit_in_zero_one_range_min_max(proba_v9_cumulative)
    
    # "proba_v1": proba_v1, "proba_v2": proba_v2, "proba_v3": proba_v3,
    # "proba_v5": proba_v5, "proba_v6": proba_v6, "proba_v8": proba_v8,
    return { "proba_v4": proba_v4, "proba_v7": proba_v7,  "proba_v4_cumulative": proba_v4_cumulative, "proba_v7_cumulative": proba_v7_cumulative, "proba_tv_approx_cumulative": proba_tv_approx_cumulative, "proba_v4_cumulative_windowed": proba_v4_cumulative_windowed, "proba_v7_cumulative_windowed": proba_v7_cumulative_windowed, "proba_tv_approx_cumulative_windowed": proba_tv_approx_cumulative_windowed, "proba_v4_cumulative_w2": proba_v4_cumulative_w2, "proba_v7_cumulative_w2": proba_v7_cumulative_w2, "proba_tv_approx_cumulative_w2": proba_tv_approx_cumulative_w2, "proba_v4_cumulative_windowed_w2": proba_v4_cumulative_windowed_w2, "proba_v7_cumulative_windowed_w2": proba_v7_cumulative_windowed_w2, "proba_tv_approx_cumulative_windowed_w2": proba_tv_approx_cumulative_windowed_w2, 
            "proba_v9_cumulative_windowed_w2":proba_v9_cumulative_windowed_w2, 
            "proba_v9_cumulative_windowed": proba_v9_cumulative_windowed, 
            "proba_v9_cumulative_w2": proba_v9_cumulative_w2, 
            "proba_v9_cumulative": proba_v9_cumulative,
            
            "proba_v10_cumulative_windowed_w2":proba_v10_cumulative_windowed_w2, 
            "proba_v10_cumulative_windowed": proba_v10_cumulative_windowed, 
            "proba_v10_cumulative_w2": proba_v10_cumulative_w2, 
            "proba_v10_cumulative": proba_v10_cumulative,
            
            "proba_v11_cumulative_windowed_w2":proba_v11_cumulative_windowed_w2, 
            "proba_v11_cumulative_windowed": proba_v11_cumulative_windowed, 
            "proba_v11_cumulative_w2": proba_v11_cumulative_w2, 
            "proba_v11_cumulative": proba_v11_cumulative,
            
            "proba_v12_cumulative_windowed_w2":proba_v12_cumulative_windowed_w2, 
            "proba_v12_cumulative_windowed": proba_v12_cumulative_windowed, 
            "proba_v12_cumulative_w2": proba_v12_cumulative_w2, 
            "proba_v12_cumulative": proba_v12_cumulative
           }

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
