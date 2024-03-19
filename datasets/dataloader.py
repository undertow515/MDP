import torch
import pandas as pd
from .dataset import SingleDatasetWithStatic
from config import reader
from pathlib import Path
from typing import List
import json

def get_loader_dataset(config, loader_type, \
              save=True, get_datasets=True) -> List[torch.utils.data.DataLoader] | List[SingleDatasetWithStatic]:
    
    """
    loader_type: train, validation, test
    """

    # src_path:List, start_date, end_date, past_length, pred_length, \
    #           dynamic_features, \
    #           batch_size, run_dir

    dynamic_paths = config.dynamic_paths
    static_path = config.static_path

    load_config_dict = config.get_loader_config_dict(type=loader_type)
    start_date = load_config_dict["start_date"]
    end_date = load_config_dict["end_date"]
    past_length = config.past_length
    pred_length = config.pred_length
    dynamic_features = config.dynamic_features
    batch_size = config.batch_size
    run_dir = config.run_dir
    model_type = config.model_type


    datasets = [SingleDatasetWithStatic(dynamic_path=dynamic_path, static_path=static_path,\
                                  start_date=start_date, end_date=end_date,\
                                  past_length=past_length, pred_length=pred_length,\
                                  dynamic_features=dynamic_features,model_type=model_type) for dynamic_path in dynamic_paths]

    
    loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                for dataset in datasets]
    
    for dataset, loader in zip(datasets, loaders):
        src_mean, src_std, tgt_mean, tgt_std = dataset.get_std_mean()
        std_mean_save_dir = Path(run_dir) / "std_mean" / str(dataset.damcode)
        std_mean_save_dir.mkdir(parents=True, exist_ok=True)
        # change src_mean, src_std, tgt_mean, tgt_std to list and save to json
        std_mean_dict = {"src_mean": src_mean.tolist(), "src_std": src_std.tolist(), "tgt_mean": tgt_mean.tolist(), "tgt_std": tgt_std.tolist()}
        json_name = loader_type + "_" + "std_mean.json"
        std_mean_save_path = std_mean_save_dir / json_name
        with open(std_mean_save_path, 'w') as f:
            json.dump(std_mean_dict, f)
        if save:
            torch.save(loader, Path(run_dir) / (loader_type + "_" + str(dataset.damcode) + "_loader.pth"))

    if get_datasets == True:
        return loaders, datasets
    else:
        return loaders




