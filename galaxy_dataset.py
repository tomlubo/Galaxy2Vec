'''
parts of this code were inspired by https://github.com/sthalles/SimCLR?utm_source=catalyzex.com
and the paper
Ting Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. 2020. arXiv:
2002.05709 [cs.LG]. url: https://arxiv.org/abs/2002.05709.

'''


from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn

from datasets import load_from_disk, DatasetDict

def pil_to_chw_uint8(image):
    # return (chanels, height width)
    arr = np.array(image)

    # if the image is grayscale, make it 3 channel by stacking the array
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis = -1)

    # if the image has an alpha channel, drop it
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    
    # make tensor with output shaope
    tensor = torch.from_numpy(arr).permute(2,0,1).contiguous()

    # check type
    if tensor.dtype != torch.uint8:

        tensor = tensor.to(torch.uint8)

    return tensor

def make_strat_bins(labels, n_bins = 10):

    labels = np.asarray(labels, dtype = np.float32)

    try:
        # try making bins withqcut which makes balanced bins
        bins = pd.qcut(labels, q = n_bins, labels=False, duplicates = 'drop')
        bins = np.asarray(bins, dtype = np.int64)
    
    except Exception:
        # if it fauls just make bins defined by linspace
        edges = np.linspace(labels.min(), labels.max(), n_bins + 1)
        bins = np.digitize(labels, edges[1:-1], right = False).astype(np.int64)

    return bins


def train_val_split_indices(n_samples, val_frac = 0.1, random_state = 42):
    # make validation split without stratification

    rng  = np.random.default_rng(random_state)
    # list of all indices
    indices = np.arange(n_samples)

    # shuffle indices
    rng.shuffle(indices)
    n_val = int(round(n_samples * val_frac))
    # select split indices
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])

    return train_idx, val_idx

def stratified_train_val_split_indices(labels, val_frac = 0.1, random_state = 42, n_bins = 10):
    # make dataset splkits with balanced classes

    labels = np.asarray(labels, dtype = np.float32)
    # get the bins
    bins = make_strat_bins(labels, n_bins = n_bins)

    rng = np.random.default_rng(random_state)

    train_parts = []
    val_parts = []

    # iterate through the bins and split each bin separately
    for b in np.unique(bins):

        idx = np.where(bins == b)[0]
        idx = idx.copy()
        rng.shuffle(idx)

        n_val = max(1, int(round(len(idx)* val_frac)))

        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    # once done concatenate the parts into the data split
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))

    return train_idx, val_idx



# now we can actually make the dataset classes


class RingGalaxyDataset(Dataset):
    # return a dict with the image, labels, and index

    # we are using a huggingface dataset

    def __init__(self, hf_dataset, transform = None, return_index = True):
        super().__init__()

        self.ds = hf_dataset
        self.transform = transform
        self.return_index = return_index

    def __len__(self): 

        return len(self.ds)
    
    def __getitem__(self, index):
        # get the sample
        sample = self.ds[index]
        # convert image and label
        image = pil_to_chw_uint8(sample['image'])
        label = torch.tensor(sample['label'], dtype = torch.float32)

        # check if we do a transform here
        if self.transform is not None:
            image = self.transform(image)
        
        # make output dict
        out = {'image': image, 'label': label}

        if self.return_index:
            out['index'] = index

        return out
    

class EuclidGalaxyDataset(Dataset):

    # same thing as aboutve but the euclid dataset has metadata which I dont need right now but might be useful later so I will feed it forward

    DEFAULT_META_COLS = [
        "smooth-or-featured-euclid_smooth_fraction",
        "smooth-or-featured-euclid_featured-or-disk_fraction",
        "smooth-or-featured-euclid_problem_fraction",
        "disk-edge-on-euclid_yes_fraction",
        "has-spiral-arms-euclid_yes_fraction",
        "bar-euclid_strong_fraction",
        "bar-euclid_weak_fraction",
        "bar-euclid_no_fraction",
        "merging-euclid_merger_fraction",
        "merging-euclid_major-disturbance_fraction",
        "problem-euclid_star_fraction",
        "problem-euclid_artifact_fraction",
        "problem-euclid_zoom_fraction",
        "artifact-euclid_satellite_fraction",
        "artifact-euclid_scattered_fraction",
        "artifact-euclid_diffraction_fraction",
        "artifact-euclid_ray_fraction",
        "artifact-euclid_saturation_fraction",
        "artifact-euclid_ghost_fraction",
        "summary",
    ]
        
    def __init__(self, hf_dataset, transform = None, meta_cols = None, return_index = True):
        super().__init__()
        self.ds = hf_dataset
        self.transform = transform
        self.return_index = return_index
        # check if metacols was passes if not get rid of it
        if meta_cols is None:
            meta_cols = self.DEFAULT_META_COLS

        # if metacols is passed see which ones exist in both the default ones i defined above and the ones passed

        available = set(self.ds.column_names)

        self.meta_cols = [col for col in meta_cols if col in available]


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):

        # get the sample 
        sample = self.ds[index]
        # convert image to PIL
        image = pil_to_chw_uint8(sample['image'])

        if self.transform is not None:
            image = self.transform(image)
        
        meta = {col: sample[col] for col in self.meta_cols}

        out = {'image': image, 'id_str': sample['id_str'], 'dataset_name': sample.get("dataset_name", ""), 'meta':meta}

        if self.return_index:
            out['index'] = index
        
        return out
    

# the datasets are buildt but the we now need a wrapper for the contrastice learning dataset


class ContrastiveWrapper(Dataset):

    def __init__(self, base_ds, transform_twice):
        
        self.base = base_ds
        self.transform_twice = transform_twice


    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, index):
        # get the image from the dataset

        item = self.base[index]

        image = item['image']
        # apply the transformations
        transformed  = self.transform_twice(image)

        q, k = transformed
        # make dict and return
        out = dict(item)
        out.pop('image', None)
        out['view1'] = q
        out['view2'] = k
        
        return out
    


def load_rings_dataset(data_path, train_transform = None, val_transform = None, test_transform = None, 
                       val_frac = 0.1, random_state = 42, stratify=True, n_bins = 10):
    
    # helper to load the supervised dataset

    
    hf_ds = load_from_disk(data_path)

    full_train = hf_ds['train']
    test_ds = hf_ds['test']

    labels = np.array(full_train['label'], dtype = np.float32)

    if stratify:
        train_idx, val_idx = stratified_train_val_split_indices(labels, val_frac = val_frac, random_state = random_state, n_bins = n_bins)
    
    else:
        train_idx, val_idx = train_val_split_indices(n_samples = len(full_train), val_frac = val_frac, random_state = random_state)
    
    train_split = full_train.select(train_idx.tolist())
    val_split = full_train.select(val_idx.tolist())

    return {'train': RingGalaxyDataset(train_split, transform = train_transform),
            'val': RingGalaxyDataset(val_split, transform=val_transform),
            'test': RingGalaxyDataset(test_ds, transform = test_transform)}

def load_euclid_dataset(data_path, train_transform = None,test_transform = None, meta_cols = None):

    hf_ds = load_from_disk(data_path)

    return {
        'train': EuclidGalaxyDataset(hf_ds['train'], transform = train_transform, meta_cols = meta_cols),
        'test': EuclidGalaxyDataset(hf_ds['test'], transform = test_transform, meta_cols = meta_cols)

    }

         

class TwoCropsTransform(nn.Module):
    """
    from: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
    """
    def __init__(self, heavy: nn.Module, light: nn.Module):
        super().__init__()
        self.heavy = heavy
        self.light = light
    def forward(self, x):
        base = self.heavy(x)
        # two independent light branches (clone to avoid getting a pointer exception in python lol)
        q = self.light(base)
        k = self.light(base.clone())
        return q, k