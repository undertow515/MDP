import torch
import pandas as pd
from datasets import dataloader
from config import reader
from pathlib import Path
from model import transformer, lstm_msv_s2s
from utils import evalmetrices, clearepoch
from tqdm import tqdm
import numpy as np
import train
import yaml
import glob
# import evaluation
from torch.utils.tensorboard import SummaryWriter

######## config 불러오기 및 하이퍼파라미터 설정
def run(yaml_path):
    config = reader.Config(yaml_path=yaml_path)
    #### 로더 설정

    ####

    ##### train 설정
    run_dir = config.run_dir
    seed = config.seed
    l2 = config.l2
    learning_rate = config.learning_rate
    #####

    ##### Transformer 모델 설정
    ###########


    # tensorboard 설정
    writer = SummaryWriter(run_dir)

    # seed 설정
    torch.manual_seed(seed)



    # 모델 불러오기
    if config.model_type == "transformer":
        model_args = config.get_transformer_model_config_dict()
        model = transformer.Transformer(*model_args.values()).to(config.device)
    elif config.model_type == "lstm":
        model_args = config.get_lstm_msv_s2s_model_config_dict()
        model = lstm_msv_s2s.LSTMMSVS2S(*model_args.values()).to(config.device)
    else:
        raise ValueError("model_type should be transformer or lstm")

    # 데이터 불러오기

    train_loaders, _ = dataloader.get_loader_dataset(config, "train")
    valid_loaders, _ = dataloader.get_loader_dataset(config, "validation")


    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
    criterion = torch.nn.MSELoss()


    
    # save yaml file to run_dir
    yaml_name = Path(yaml_path).name
    yaml_save_path = Path(run_dir) / yaml_name
    yaml_save_path.write_text(Path(yaml_path).read_text())

    # 옵티마이저, 손실함수 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    criterion = torch.nn.MSELoss()

    # 학습
    train.multi_train_full(model, train_loaders, valid_loaders, optimizer, criterion, writer, config)
    writer.close()
    print("training is done")

paths = glob.glob("./config/trainconfig/transformer/*/**.yml")

if __name__ == "__main__":
    ## running the code
    for path in paths:
        print(f"running {path}")
        run(path)
        clearepoch.clearepoch(reader.Config(path).run_dir)

