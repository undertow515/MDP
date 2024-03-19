from model import transformer, lstm_msv_s2s
from config import reader
import torch
from datasets.dataloader import get_loader_dataset
from train import multi_train_epoch, multi_train_full
from torch.utils.tensorboard import SummaryWriter

# 모델 작동 확인
# ---------------------------------------------------------------------------------------------------------------- 
# model_args = config.get_transformer_model_config_dict()
# model = transformer.Transformer(*model_args.values()).to(config.device)
# src_sample = torch.rand(64, 22, 13).to(config.device)
# tgt_sample = torch.rand(64, 22, 1).to(config.device)
# ---------------------------------------------------------------------------------------------------------------- 

# 데이터 로더 작동 확인
# dll = get_loader_dataset(config.get_loader_config_dict())


# 데이터로더 및 모델 연동확인
# ----------------------------------------------------------------------------------------------------------------
"""
    model_args = config.get_transformer_model_config_dict()
    model = transformer.Transformer(*model_args.values()).to(config.device)
    
    loader_args = config.get_loader_config_dict()
    dll = get_loader_dataset(*loader_args.values())
    train_loader, _ = dll[0]

    src_sample, tgt_sample, tgt_y_sample = next(iter(train_loader))
    src_sample = src_sample.to(config.device)
    tgt_sample = tgt_sample.to(config.device)
    output = model(src_sample, tgt_sample)
    print(output.shape), print(tgt_y_sample.shape)
"""
# ----------------------------------------------------------------------------------------------------------------

# train.py 디버깅
# ----------------------------------------------------------------------------------------------------------------

# 함수디버깅
# """
# def multi_train_epoch(model:torch.nn.Module, train_loaders:List[torch.utils.data.DataLoader],\
#                        validation_loader:List[torch.utils.data.DataLoader], optimizer:torch.optim,\
#                     criterion, config:reader.Config):
# """
# ----------------------------------------------------------------------------------------------------------------
# model_args = config.get_transformer_model_config_dict()
# model = transformer.Transformer(*model_args.values()).to(config.device)

# loader_args = config.get_loader_config_dict()
# train_loaders, _ = get_loader_dataset(*loader_args.values())

# val_loader_args = config.get_loader_config_dict(type="validation")

# valid_loaders, _ = get_loader_dataset(*val_loader_args.values())
# optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
# criterion = torch.nn.MSELoss()
# writer = SummaryWriter(config.run_dir)
# ----------------------------------------------------------------------------------------------------------------


# LSTM 모델 디버깅
# ----------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import transformer, lstm_msv_s2s as lstm
from config import reader
from typing import List, Tuple, Dict
from datasets import dataset, dataloader
from collections import defaultdict
import json
from pathlib import Path

from typing import List, Tuple, Dict
def eval(config_path, best_model_path,epoch:str) -> Tuple[dict, dict]:
    config = reader.Config(config_path)
    if config.model_type == "transformer":
        model_args = config.get_transformer_model_config_dict()
        model = transformer.Transformer(*model_args.values())
        model.load_state_dict(torch.load(best_model_path))
    elif config.model_type == "lstm":
        model_args = config.get_lstm_msv_s2s_model_config_dict()
        model = lstm.LSTMMSVS2S(*model_args.values())
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise ValueError("model_type should be either transformer or lstm")
    model.to(config.device)
    model.eval()
    valid_loaders, _ = dataloader.get_loader_dataset(config, "validation")
    
    q_pred = defaultdict(list)
    q_true = defaultdict(list)
    
    with torch.no_grad():
        for valid_loader in valid_loaders:
            for src, tgt, tgt_y in valid_loader:
                damcode = valid_loader.dataset.damcode
                src = src.to(config.device)
                tgt = tgt.to(config.device)
                tgt_y = tgt_y.to(config.device)
                output = model(src, tgt)
                # tgt_y shape : (batch_size, seq_len, 1)
                # output shape : (batch_size, seq_len, 1)
                # lookback : int, it is the number of days to look back and same as pred_length
                for i in range( config.pred_length):
                    if config.model_type == "lstm":
                        q_pred[damcode][i+1].append(output[:, -config.pred_length + i, :])
                        q_true[damcode][i+1].append(tgt_y[:, -config.pred_length + i, :])
                    elif config.model_type == "transformer":
                        q_pred[damcode][i+1].append(output[:, -config.pred_length + i, :])
                        q_true[damcode][i+1].append(tgt_y[:, -config.pred_length + i, :])

    return q_pred, q_true






if __name__ == "__main__":
    q_pred, q_true = eval(r"C:\Users\82105\MDP\runs\transformer\str1\train_5_0.0005_28\train_5_0.0005_28.yml", r"C:\Users\82105\MDP\runs\transformer\str1\train_5_0.0005_28\models\model_epoch_193_valloss_0.1421940749627538.pt",epoch="193")
    print(q_pred)