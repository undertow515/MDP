import torch
import pandas as pd
from datasets import dataloader
from config import reader
from pathlib import Path
# from model import lstm
from tqdm import tqdm
from utils import evalmetrices
from collections import defaultdict
import numpy as np
from typing import List
import torch


def multi_train_epoch(model:torch.nn.Module, train_loaders:List[torch.utils.data.DataLoader],\
                       validation_loaders:List[torch.utils.data.DataLoader], optimizer:torch.optim,\
                    criterion, config:reader.Config, parallel:bool=False):
    device = config.device
    train_loss = 0.0
    valid_loss = 0.0

    model.train()

    # Create iterators for all loaders
    iterators = [iter(loader) for loader in train_loaders]
    if parallel:
        while True:
            try:
                # Load data from all loaders
                data = [next(iterator) for iterator in iterators]

                # Separate sources, targets and target_ys from data
                srcs = torch.stack([item[0] for item in data]).to(device)
                tgts = torch.stack([item[1] for item in data]).to(device)
                tgt_ys = torch.stack([item[2] for item in data]).to(device)

                for src, tgt, tgt_y in zip(srcs, tgts, tgt_ys):
                    optimizer.zero_grad()
                    outputs = model(src, tgt)
                    loss = criterion(outputs, tgt_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

            except StopIteration:
                # Exit the loop if any of the loaders have no more data
                break

        # print(f'Training loss: {train_loss}')
        
        model.eval()
        valid_iterators = [iter(loader) for loader in validation_loaders]

        # q_pred = []
        # q_true = []
        while True:
            try:
                data = [next(iterator) for iterator in valid_iterators]
                # dam_ids = [iterator.damcode for iterator in validation_loaders]
                srcs = torch.stack([item[0] for item in data]).to(device)
                tgts = torch.stack([item[1] for item in data]).to(device)
                tgt_ys = torch.stack([item[2] for item in data]).to(device)
                for src, tgt, tgt_y in zip(srcs, tgts, tgt_ys):
                    outputs = model(src, tgt)
                    loss = criterion(outputs, tgt_y)
                    valid_loss += loss.item()

            except StopIteration:
                break
        avg_train_loss = train_loss / len(train_loaders[0])
        avg_valid_loss = valid_loss / len(validation_loaders[0])
    else:
        for train_loader in train_loaders:
            for i, (src, tgt, tgt_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train_step"):
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_y = tgt_y.to(device)
                optimizer.zero_grad()
                outputs = model(src, tgt)
                loss = criterion(outputs, tgt_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        with torch.no_grad():
            model.eval()
            q_preds = dict()
            q_trues = dict()
            for valid_loader in validation_loaders:
                q_pred = []
                q_true = []
                for i, (src, tgt, tgt_y) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validation_step"):
                    src = src.to(device)
                    tgt = tgt.to(device)
                    tgt_y = tgt_y.to(device)
                    outputs = model(src, tgt)
                    loss = criterion(outputs, tgt_y)
                    valid_loss += loss.item()
                    q_pred.append(outputs[:, -config.pred_length:, :].cpu().numpy())
                    q_true.append(tgt_y[:, -config.pred_length:, :].cpu().numpy())
                    q_pred = np.concatenate(q_pred, axis=0).reshape(-1, config.pred_length) * valid_loader.dataset.get_std_mean()[-1] + valid_loader.dataset.get_std_mean()[-2]
                    q_true = np.concatenate(q_true, axis=0).reshape(-1, config.pred_length) * valid_loader.dataset.get_std_mean()[-1] + valid_loader.dataset.get_std_mean()[-2]
                q_trues[valid_loader.dataset.damcode] = q_true
                q_preds[valid_loader.dataset.damcode] = q_pred

        mean_nse = np.mean([evalmetrices.get_eval_metrics(q_trues, q_preds)[damcode]["nse"] for damcode in q_trues.keys()])
        mean_kge = np.mean([evalmetrices.get_eval_metrics(q_trues, q_preds)[damcode]["kge"] for damcode in q_trues.keys()])
        avg_train_loss = train_loss / sum([len(train_loaders[i]) for i in range(len(train_loaders))])
        avg_valid_loss = valid_loss / sum([len(validation_loaders[i]) for i in range(len(validation_loaders))])


    return avg_train_loss, avg_valid_loss, mean_nse, mean_kge


# def multi_train_epoch(model:torch.nn.Module, train_loaders:List[torch.utils.data.DataLoader],\
#                        validation_loaders:List[torch.utils.data.DataLoader], optimizer:torch.optim,\
#                     criterion, config:reader.Config):


def multi_train_full(model, train_loaders,\
                      validation_loaders, \
                          optimizer, criterion, \
                            writer, config):
    run_dir = config.run_dir
    epochs = config.epochs

    epochs = int(epochs)
    for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
        # train_loss, valid_loss, nse, rmse, pbias, kge, q_pred, q_true = multi_train_epoch(model, train_loader, validation_loader, optimizer, criterion, device)
        train_loss, valid_loss, nse, kge = multi_train_epoch(model, train_loaders, validation_loaders, optimizer, criterion, config)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Valid Loss', valid_loss, epoch)
        # valid_loss float round 3
        # valid_loss = round(valid_loss, 3)
        train_loss = round(train_loss, 8)
        valid_loss = round(valid_loss, 8)

        print(f"Epoch : {epoch}, Train Loss : {train_loss}, Valid Loss : {valid_loss}, NSE : {nse}, KGE : {kge}")
        # writer.add_scalar('Train Loss', train_loss, epoch)
        # writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('NSE', nse, epoch)
        # writer.add_scalar('RMSE', rmse, epoch)
        # writer.add_scalar('PBIAS', pbias, epoch)
        writer.add_scalar('KGE', kge, epoch)

        # print(f"Epoch : {epoch}, Train Loss : {train_loss}, Valid Loss : {valid_loss}")
        # writer.add_scalar('Valid Loss', valid_loss, epoch)
        # writer.add_scalar('NSE', nse, epoch)
        # writer.add_scalar('RMSE', rmse, epoch)
        # writer.add_scalar('PBIAS', pbias, epoch)
        # writer.add_scalar('KGE', kge, epoch)
        
        # make models dir
        models_dir = Path(run_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # save model every epoch
        torch.save(model.state_dict(), Path(run_dir) /"models"/ f"model_epoch_{epoch}_valloss_{valid_loss}_nse_{nse.round(3)}.pt")
        # torch.save(model.state_dict(), Path(run_dir) /"models"/ f"model_epoch_{epoch}_valloss_{valid_loss}_.pt")


def single_train_epoch(model, train_loader, validation_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    valid_loss = 0.0
    for i, (src, _ ,tgt) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train_step"):
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        outputs = model(src)
        loss = criterion(outputs, tgt)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    q_pred = []
    q_true = []
    with torch.no_grad():
        for i, (src, _ ,tgt) in tqdm(enumerate(validation_loader), total=len(validation_loader), desc="Validation_step"):
            src = src.to(device)
            tgt = tgt.to(device)
            outputs = model(src)
            loss = criterion(outputs, tgt)
            valid_loss += loss.item()
            q_pred.append(outputs)
            q_true.append(tgt)

    q_std = train_loader.dataset.tgt.std().item()
    q_mean = train_loader.dataset.tgt.mean().item()

    q_pred = torch.cat(q_pred, dim=0)
    q_true = torch.cat(q_true, dim=0)

    q_pred = q_pred.detach().cpu().numpy() * q_std + q_mean
    q_true = q_true.detach().cpu().numpy() * q_std + q_mean

    nse = evalmetrices.nse(q_true, q_pred)
    rmse = evalmetrices.rmse(q_true, q_pred)
    pbias = evalmetrices.pbias(q_true, q_pred)
    kge = evalmetrices.kge(q_true, q_pred)


    return train_loss / len(train_loader), valid_loss / len(validation_loader), nse, rmse, pbias, kge, q_pred, q_true

def single_train_full(model, train_loader, validation_loader, optimizer, criterion, device, epochs, writer, run_dir):
    epochs = int(epochs)
    for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
        train_loss, valid_loss, nse, rmse, pbias, kge, q_pred, q_true = single_train_epoch(model, train_loader, validation_loader, optimizer, criterion, device)
        # valid_loss float round 3
        valid_loss = round(valid_loss, 3)
        train_loss = round(train_loss, 3)

        print(f"Epoch : {epoch}, Train Loss : {train_loss}, Valid Loss : {valid_loss}, NSE : {nse}, RMSE : {rmse}, PBIAS : {pbias}, KGE : {kge}")
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('NSE', nse, epoch)
        writer.add_scalar('RMSE', rmse, epoch)
        writer.add_scalar('PBIAS', pbias, epoch)
        writer.add_scalar('KGE', kge, epoch)
        
        # make models dir
        models_dir = Path(run_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # save model every epoch
        torch.save(model.state_dict(), Path(run_dir) /"models"/ f"model_epoch_{epoch}_valloss_{valid_loss}_nse_{nse.round(3)}.pt")
