import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import transformer, lstm_msv_s2s as lstm
from config import reader
from typing import List, Tuple, Dict
from datasets import dataloader
from collections import defaultdict
import json
from pathlib import Path
import glob
import numpy as np
import re
import pandas as pd
from utils import evalmetrices






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
import numpy as np
import pandas as pd

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
    
     # (num_gauges, len(valid_loader) ,batch_size, pred_length, 1) -> (num_gauges, len(valid_loader) * batch_size, pred_length, 1)
    
    q_preds = dict()
    q_trues = dict()
    with torch.no_grad():
        for valid_loader in valid_loaders:
            damcode = valid_loader.dataset.damcode
            q_pred = []
            q_true = []
            for i, (src, tgt, tgt_y) in enumerate(valid_loader):

                src = src.to(config.device)
                tgt = tgt.to(config.device)
                tgt_y = tgt_y.to(config.device)
                output = model(src, tgt)

                # tgt_y shape : (batch_size, seq_len, 1)
                # output shape : (batch_size, seq_len, 1)
                # lookback : int, it is the number of days to look back and same as pred_length

                if config.model_type == "lstm":
                    q_pred = output.cpu().numpy()
                    q_true = tgt_y.cpu().numpy()
                elif config.model_type == "transformer":
                    q_pred.append(output[:, -config.pred_length:, :].cpu().numpy())
                    q_true.append(tgt_y[:, -config.pred_length:, :].cpu().numpy())
            
            q_preds[damcode] = np.concatenate(q_pred, axis=0).reshape(-1, config.pred_length) * valid_loader.dataset.get_std_mean()[-1] + valid_loader.dataset.get_std_mean()[-2] # q_preds[damcode] shape : (len(valid_loader) * batch_size, pred_length)
            q_trues[damcode] = np.concatenate(q_true, axis=0).reshape(-1, config.pred_length) * valid_loader.dataset.get_std_mean()[-1] + valid_loader.dataset.get_std_mean()[-2] # q_trues[damcode] shape : (len(valid_loader) * batch_size, pred_length)
            df = pd.DataFrame(np.concatenate([q_trues[damcode], q_preds[damcode]], axis=-1))
            df.index = pd.date_range(start=config.validation_start_date, end=config.validation_end_date, freq="D")[-q_trues[damcode].shape[0]:]
            df.columns = [f"q_true_{i+1}" for i in range(config.pred_length)] + [f"q_pred_{i+1}" for i in range(config.pred_length)]
            df.to_csv(f"check_{damcode}.csv") # TODO: set the directory
    return q_preds, q_trues

def plotting(result_csv: str, lookback: int):
    import matplotlib.pyplot as plt
    df = pd.read_csv(result_csv)
    df.index = pd.to_datetime(df.index)
    plt.plot(df["q_true"], label="True")
    plt.plot(df["q_pred"], label="Pred")
    plt.savefig(result_csv.split(".csv")[0] + ".png")

if __name__ == "__main__":
    eval(r"C:\Users\82105\MDP\runs\transformer\1\train_0_0.0005_28", r"C:\Users\82105\MDP\runs\transformer\1\train_0_0.0005_28\models\model_epoch_28_valloss_0.20034909.pt" ,None)
    



