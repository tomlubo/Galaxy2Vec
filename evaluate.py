

from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from scipy import stats

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else: out[k] = v
    return out

def regression_metrics(y_true, y_pred): 
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    pearson = np.corrcoef(y_true, y_pred)[0, 1]

    res = stats.spearmanr(y_true, y_pred)
    correlation = res.statistic
    p_value = res.pvalue

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson': float(pearson),
        'spearman': float(correlation)
    }


class Evaluator():

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, loader):

        self.model.eval()

        preds_all = []
        targets_all = []
        indices_all = []

        id_str_all = []
        dataset_name_all = []
        meta_all = []

        bar = tqdm(loader, desc = 'Evaluate', leave = False)

        for batch in bar:
            batch = move_batch_to_device(batch, self.device)

            x = batch['image']
            preds = self.model(x).squeeze(-1).cpu().numpy()
            preds_all.append(preds)

            if 'label' in batch:
                targets = batch['label'].detach().cpu().numpy()
                targets_all.append(targets)

            if 'index' in batch:
                idx = batch['index'].detach().cpu().numpy()
                indices_all.append(idx)

            if 'id_str' in batch:
                id_str_all.extend(batch['id_str'])

            if 'dataset_name' in batch:
                dataset_name_all.extend(batch['dataset_name'])

            if 'meta' in batch:
                meta_all.extend(batch['meta'])

        out = {'predictions': np.concatenate(preds_all) if len(preds_all) > 0 else np.array([])}

        if len(targets_all) > 0:
            out['targets'] = np.concatenate(targets_all)
        if len(indices_all) > 0:
            out['indices'] = np.concatenate(indices_all)
        if len(id_str_all) > 0:
            out['id_str'] = id_str_all
        if len(dataset_name_all) > 0:
            out['dataset_name'] = dataset_name_all


        return out
    
    @torch.no_grad()
    def evaluate_regression(self, loader):
        out = self.predict(loader)

        if 'targets' not in out:
            raise ValueError('No labels found in loader batch. Cannot evaluate regression.')

        metrics = regression_metrics(out['targets'], out['predictions'])
        out['metrics'] = metrics
        return out

class ContrastiveEvaluator:

    def __init__(self, model, device):

        self.model = model.to(device)
        self.device = device


    def extract_embeddings(self, loader):
        self.model.eval()

        embeddings = []
        labels = []
        indices = []

        bar = tqdm(loader, desc='Extract Embeddings', leave = False)

        for batch in bar:
            batch = move_batch_to_device(batch, self.device)

            try:
                x = batch["view1"]
            except KeyError:
                x = batch["image"]

            z, feats = self.model(x, return_features = True)

            embeddings.append(feats.detach().cpu().numpy())

            if 'label' in batch:
                labels.append(batch['label'].detach().cpu().numpy())

            if 'index' in batch:
                indices.append(batch['index'].detach().cpu().numpy())

        out = {
            "embeddings": np.concatenate(embeddings) if len(embeddings) > 0 else np.array([])
        }

        if len(labels) > 0:
            out["labels"] = np.concatenate(labels)

        if len(indices) > 0:
            out["indices"] = np.concatenate(indices)

        return out




    






