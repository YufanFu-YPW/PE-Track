import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm
import os
from easydict import EasyDict

from utils.tools import Print
from models.HSMP.history_space_model import HSModel
from datasets import MySSMDataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--device', default=0)
    return parser.parse_args()


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def set_seed(seed=318):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    print(f"use device: cuda {args.device}")
    with open(args.config, 'rb') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    # set random_seed
    set_seed(config.seed)

    # init Plog
    log_path = os.path.join(config.save_dir, f"{config.dataset}_log.txt")
    Plog = Print(log_path)

    # data_loader
    train_data = MySSMDataset(config.data_path)
    train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=config.drop_last)
    val_data = MySSMDataset(config.val_data_path)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    num_train_batch = len(train_data_loader)
    num_val_batch = len(val_data_loader)

    # load model
    model = HSModel(config.d_model, config.v_size, config.block_layers, config.d_state, config.d_conv, config.expand, config.fusion_expand, 
                    config.mamba_layers, config.pre_mamba_layers, config.bi_mamba, config.heads, config.norm_epsilon, config.rms_norm, config.dropout)
    #model.cuda()
    model.to(device)
    num_p = get_num_params(model)

    mkdir(config.save_dir)

    Plog.log(f'Model: \n{model}', False)
    Plog.log('config:', False)
    for k, v in vars(config).items():
        Plog.log(f'{k} = {v}', False)
    Plog.log(f'Motion model:  {num_p} parameters')
    
    Plog.log(f'Save dir: {config.save_dir}')

    # load optimizer
    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    # Train model
    best_epoch = -1
    min_val_loss = 100
    best_weight = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        # train
        epoch_train_loss = 0
        pbar = tqdm(train_data_loader, ncols=100)
        train_time = time.asctime(time.localtime(time.time()))
        pbar.set_description(f"[{train_time}] [Epoch {epoch}]  Train")
        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].float().to(device=device, non_blocking=True)

            pre_box = model(batch['long_history'], batch['short_space'])
            train_loss = F.smooth_l1_loss(pre_box.view(-1, 4), batch['label'].view(-1,4))
            epoch_train_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # eval
        epoch_val_loss = 0
        val_pbar = tqdm(val_data_loader, ncols=100)
        model.eval()
        with torch.no_grad():
            val_time = time.asctime(time.localtime(time.time()))
            val_pbar.set_description(f"[{val_time}] [Epoch {epoch}]  Val")
            for batch in val_pbar:
                for k in batch:
                    batch[k] = batch[k].float().to(device=device, non_blocking=True)
                    
                pre_box = model(batch['long_history'], batch['short_space'])
                val_loss = F.smooth_l1_loss(pre_box.view(-1, 4), batch['label'].view(-1,4))
                epoch_val_loss += val_loss.item()
                
        avrg_val_loss = epoch_val_loss/num_val_batch
        if avrg_val_loss <= min_val_loss:
            min_val_loss = avrg_val_loss
            best_epoch = epoch
            best_weight = model.state_dict().copy()
            
        Plog.log(f"[Epoch {epoch}]  train_loss: {epoch_train_loss/num_train_batch:.8f}  val_loss: {avrg_val_loss:.8f}  lr: {optimizer.state_dict()['param_groups'][0]['lr']:.8f}")
        Plog.log(f"[Epoch {epoch}]  Min val loss: {min_val_loss:.8f}  Best epoch: {best_epoch}")

        if config.use_scheduler:
            scheduler.step()

        if epoch % config.save_every == 0:
            checkpoint = {
                'ddpm': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(config.save_dir, f"{config.dataset}_epoch{epoch}.pt"))
        # break
    torch.save(best_weight, os.path.join(config.save_dir, f"{config.dataset}_best_epoch{best_epoch}.pt"))


def get_num_params(model):
    """Return the total number of parameters in a training model."""
    return sum(x.numel() for x in model.parameters())



if __name__ == '__main__':
    train()