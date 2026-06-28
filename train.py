from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def move_batch_to_device(batch, device):
    # send batch to device
    # its a dict so we need to do each element
    out = {}
    for k, v in batch.items(): 
        out[k] = v.to(device)
    return out

class AverageMeter:
    # helper class to track a metric in the epoch

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
    
    def update(self, val, n = 1):
        self.sum += val* n
        self.count += n
        self.avg = self.sum / self.count

class Trainer:

    def __init__(self, model, optimizer, loss_fn, device):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def _get_preds_and_targets(self, batch):
        # forward pass through model
        # get the x and y for the batch
        x = batch['image']
        y = batch['label'].squeeze(-1) # remove extra dim
        # get predictions
        preds = self.model(x).squeeze(-1)

        return preds, y
    
    def train_one_epoch(self, loader):
        # do one full pass through the data

        self.model.train() # train mode, track gradients

        loss_meter = AverageMeter() #track metrics
        mae_meter = AverageMeter()
        mse_meter = AverageMeter()

        # make tqdm iterator
        bar = tqdm(loader, desc = 'Training', leave =False)
        for batch in bar: # iterate through batch
            
            batch = move_batch_to_device(batch, self.device)
            # reset optimizer
            self.optimizer.zero_grad()
            # get model output
            preds, targets = self._get_preds_and_targets(batch)
            # get loss
            loss = self.loss_fn(preds, targets)
            # backpropagate
            loss.backward()
            # take GD step
            self.optimizer.step()

            #get batch shape
            bs = targets.shape[0]
            # get mae and mse without tracking gradients for this operation
            with torch.no_grad():
                mae = torch.abs(preds - targets).mean().item()
                mse = torch.mean((preds - targets)**2).item()

            # add to metric tracker
            loss_meter.update(loss.item(),bs)
            mae_meter.update(mae, bs)
            mse_meter.update(mse, bs)

            # update what is left after tqdm is done
            bar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'mae': f'{mae_meter.avg:.4f}',
                'rmse': f'{np.sqrt(mse_meter.avg):.4f}',
            })

        return {
            'loss': loss_meter.avg,
            'mae': mae_meter.avg,
            'rmse': float(np.sqrt(mse_meter.avg)),
            'mse': mse_meter.avg,
        }
    

    @torch.no_grad()
    def validate_one_epoch(self, loader):
        # validate mode
        self.model.eval()

        # track these metrics as well
        loss_meter = AverageMeter()
        mae_meter = AverageMeter()
        mse_meter = AverageMeter()

        # make tqdm iterator
        bar = tqdm(loader, desc = 'Val', leave =False)
        for batch in bar: # iterate through batch

            batch = move_batch_to_device(batch, self.device)
            # get model output
            preds, targets = self._get_preds_and_targets(batch)

            loss = self.loss_fn(preds, targets)
            bs = targets.shape[0]

            # get the metrics
            mae = torch.abs(preds - targets).mean().item()
            mse = torch.mean((preds - targets)**2).item()
             # add to tracker
            loss_meter.update(loss.item(),bs)
            mae_meter.update(mae, bs)
            mse_meter.update(mse, bs)

            bar.set_postfix({
                'Val loss': f'{loss_meter.avg:.4f}',
                'Val mae': f'{mae_meter.avg:.4f}',
                'Val rmse': f'{np.sqrt(mse_meter.avg):.4f}',
            })

        return {
            'loss': loss_meter.avg,
            'mae': mae_meter.avg,
            'rmse': float(np.sqrt(mse_meter.avg)),
            'mse': mse_meter.avg,
        }
    
    def fit(self, train_loader, val_loader, epochs):
        # actual entry point to fit the model
        # output dict
        history = { 
            'train_loss': [],
            'train_mae': [],
            'train_mse': [],
            'train_rmse': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'val_rmse': [],

        }

        # epoch loop
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            # train one epoch
            train_metrics = self.train_one_epoch(train_loader)
            # save metrics to history
            history['train_loss'].append(train_metrics['loss'])
            history['train_mae'].append(train_metrics['mae'])
            history['train_mse'].append(train_metrics['mse'])
            history['train_rmse'].append(train_metrics['rmse'])

            print(f'Train | Loss: {train_metrics["loss"]:.4f} | MAE: {train_metrics["mae"]:.4f} | MSE: {train_metrics["mse"]:.4f} | RMSE: {train_metrics["rmse"]:.4f}')
            # validate the model
            val_metrics = self.validate_one_epoch(val_loader)
            # save to history
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_mse'].append(val_metrics['mse'])
            history['val_rmse'].append(val_metrics['rmse'])

            print(f'Val   | Loss: {val_metrics["loss"]:.4f} | MAE: {val_metrics["mae"]:.4f} | MSE: {val_metrics["mse"]:.4f} | RMSE: {val_metrics["rmse"]:.4f}')

        return history
        
def make_label_bins(labels, n_bins = 5):
    # make the similarity "classes" for the supervised contrastive learning
    labels = np.asarray(labels, dtype = np.float32)
    edges = np.linspace(0, 1, n_bins+1)
    bins = np.digitize(labels, edges[1:-1], right = False ).astype(np.int64)
    return bins

class SupervisedContrastiveLoss(nn.Module):

    '''
    code modified from https://github.com/sthalles/SimCLR?utm_source=catalyzex.com
    and the paper
    Ting Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. 2020. arXiv:
    2002.05709 [cs.LG]. url: https://arxiv.org/abs/2002.05709.
    Supervised
    Prannay Khosla et al. Supervised Contrastive Learning. 2021. arXiv: 2004.11362 [cs.LG]. url: https:
    //arxiv.org/abs/2004.11362.

    '''
    # supervised constrastive loss

    def __init__(self, temperature = 0.7):
        super().__init__()
        self.temperature = temperature


    def forward(self, z1, z2, labels):

        device = z1.device
        # get len of batch
        B = z1.shape[0]
        
        # concatenate the  views into [2B, D] where d is the embedding dimension
        z = torch.cat([z1,z2], dim = 0)
        z = F.normalize(z, dim = 1)
        # get the cosine similarity term that goes into the exponent
        sim = torch.matmul(z, z.T)/ self.temperature
        # get the labels and concat
        labels = labels.view(-1, 1)
        labels = torch.cat([labels, labels], dim = 0).to(device)
        # make the positive mask
        pos_mask = (labels == labels.T).float()
        # mask self examples
        self_mask = torch.eye(2*B, device = device)
        # positive mask
        pos_mask = pos_mask - self_mask
        # normalize
        sim = sim - sim.max(dim = 1, keepdim = True).values.detach()
        # neg log likelyhoodish
        exp_sim = torch.exp(sim) * (1- self_mask)

        log_prob = sim - torch.log(exp_sim.sum(dim = 1, keepdim = True)+ 1e-12)

        pos_counts = pos_mask.sum(dim = 1)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_counts + 1e-12)

        return -mean_log_prob.mean()
    
class ContrastiveTrainer:

    def __init__(self, model, optimizer, loss_fn, device, n_bins = 5):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.n_bins = n_bins


    def train_one_epoch(self, loader):
        # do one full pass through the data

        self.model.train() # train mode, track gradients

        loss_meter = AverageMeter() #track metrics
        # make tqdm iterator
        bar = tqdm(loader, desc = 'Contrastive Training', leave =False)
        for batch in bar: # iterate through batch
            
            batch = move_batch_to_device(batch, self.device)

            x1 = batch['view1']
            x2 = batch['view2']
            y = batch['label'].detach().cpu().numpy()
            y_bins = torch.tensor(make_label_bins(y, self.n_bins),dtype=torch.long,device=self.device)
        
            # reset optimizer
            self.optimizer.zero_grad()
            # get embeddings of two views
            z1 = self.model(x1)
            z2 = self.model(x2)
            # get loss
            loss = self.loss_fn(z1, z2, y_bins)
            # backpropagate
            loss.backward()
            # take GD step
            self.optimizer.step()

            #get batch shape
            bs = x1.shape[0]
            # get mae and mse without tracking gradients for this operation


            # add to metric tracker
            loss_meter.update(loss.item(),bs)

            # update what is left after tqdm is done
            bar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}'
            })

        return {
            'loss': loss_meter.avg
        }
    

    @torch.no_grad()
    def validate_one_epoch(self, loader):
        # validate mode
        self.model.eval()

        # track these metrics as well
        loss_meter = AverageMeter()

        # make tqdm iterator
        bar = tqdm(loader, desc = 'Val', leave =False)

        for batch in bar: # iterate through batch

            batch = move_batch_to_device(batch, self.device)
            # get model output
            x1 = batch['view1']
            x2 = batch['view2']
            y = batch['label'].detach().cpu().numpy()
            y_bins = torch.tensor(make_label_bins(y, self.n_bins),dtype=torch.long,device=self.device)

            z1 = self.model(x1)
            z2 = self.model(x2)

            loss = self.loss_fn(z1, z2, y_bins)
            bs = x1.shape[0]

             # add to tracker
            loss_meter.update(loss.item(),bs)

            bar.set_postfix({'Val loss': f'{loss_meter.avg:.4f}'
            })

        return {
            'loss': loss_meter.avg
        }
    
    def fit(self, train_loader, val_loader, epochs):
        # actual entry point to fit the model
        # output dict
        history = { 'train_loss': [],
            'val_loss': []
        }

        # epoch loop
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            # train one epoch
            train_metrics = self.train_one_epoch(train_loader)
            # save metrics to history
            history['train_loss'].append(train_metrics['loss'])

            print(f'Train Loss: {train_metrics["loss"]:.4f}')
            # validate the model
            val_metrics = self.validate_one_epoch(val_loader)
            # save to history
            history['val_loss'].append(val_metrics['loss'])

            print(f'Val Loss: {val_metrics["loss"]:.4f}')

        return history
    





            




