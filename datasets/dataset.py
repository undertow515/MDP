import torch
import pandas as pd
import xarray as xr
import numpy as np
from typing import List

def read_static(static_path:str, gauge_id:int|List[int]) -> np.ndarray:
    static_df = pd.read_csv(static_path, index_col=0)
    # standardization static features
    static_df = (static_df - static_df.mean(axis=0)) / static_df.std(axis=0)
    # gauge_id = map(int, gauge_id)
    return static_df.loc[gauge_id].values#, static_df.index

class MultipleDataset(torch.utils.data.Dataset):
    def __init__(self, gauge_id: List[int] | List[str], static_path:str ,start_date:str, end_date:str, \
                 past_length:int, pred_length:int, 
                 dynamic_features:List[str],
                static_features:List[str]):

        self.start_date = start_date
        self.end_date = end_date

        self.dates = pd.date_range(start_date, end_date)
        self.dynamic_features = dynamic_features

        self.static_features = static_features
        self.static_path = static_path

        if isinstance(gauge_id[0], int):
            self.gauge_id = gauge_id
        elif isinstance(gauge_id[0], str):
            self.gauge_id = list(map(int, gauge_id))
        else:
            raise ValueError("gauge_id should be list of int or str")

#         self.dynamic_std_mean = dict()


#         self.dynamic_df_list = [] 
#         self.dynamic_paths = dynamic_paths
#         for path in dynamic_paths:
#             df = xr.open_dataset(path).to_dataframe()
#             gauge_id = path.split("\\")[-1].split(".")[0]
#             df["gauge_id"] = gauge_id
#             self.dynamic_std_mean[gauge_id] = (df[self.dynamic_features + ["inflow"]].mean().to_list(), df[self.dynamic_features + ["inflow"]].std().to_list())
#             dynamic_df_seg = df[self.dynamic_features + ["inflow"]]
#             dynamic_df_seg = (dynamic_df_seg - dynamic_df_seg.mean(axis=0)) / dynamic_df_seg.std(axis=0)
#             dynamic_df_seg["gauge_id"] = gauge_id
#             self.dynamic_df_list.append(dynamic_df_seg)
        
#         self.dynamic_df = pd.concat(self.dynamic_df_list, axis=0)
#         # self.dynamic_df = self.df.loc[self.start_date:self.end_date]
#         self.dynamic_df["date"] = self.dynamic_df.index
#         self.dynamic_df["gauge_id"] = self.dynamic_df["gauge_id"].astype(str)

#         self.static_df = pd.read_csv(self.static_path)
#         self.static_df["gauge_id"] = self.static_df["gauge_id"].astype(str)
#         self.static_df[static_features] = (self.static_df[static_features] - self.static_df[static_features].mean(axis=0)) / self.static_df[static_features].std(axis=0)
#         self.static_df_std = self.static_df[static_features].std(axis=0)
#         self.static_df_mean = self.static_df[static_features].mean(axis=0)

#         self.df = pd.merge(self.dynamic_df, self.static_df, on="gauge_id", how="left")
#         self.df.set_index("date", inplace=True)
        
        
#         self.src = self.df[self.static_features + self.dynamic_features].values
#         self.tgt = self.df["inflow"].values

#         # self.transformed_src = torch.tensor((self.src - self.src.mean(axis=0)) / self.src.std(axis=0), dtype=torch.float32)
#         # self.transformed_tgt = torch.tensor((self.tgt - self.tgt.mean(axis=0)) / self.tgt.std(axis=0), dtype=torch.float32)

#         self.src = torch.tensor(self.src, dtype=torch.float32) # shape: (seq_len, src_size)
#         self.tgt = torch.tensor(self.tgt, dtype=torch.float32)

#         self.past_length = past_length
#         self.pred_length = pred_length

    
#     def __len__(self):
#         return len(self.df) - self.past_length - self.pred_length + 1

#     def get_std_mean_dynamic(self):
#         return self.dynamic_std_mean
    
#     def __getitem__(self, idx):
#         if self.no_q == True:
#             src = self.src[idx:idx+self.past_length]
#             tgt = torch.zeros(self.past_length + self.pred_length, len(self.dynamic_features))
#             tgt[idx:idx+self.past_length, :] = self.src[idx:idx+self.past_length, len(self.static_features):]
#             tgt_y = self.tgt[idx:idx + self.past_length + self.pred_length]
#             # Encoder input : static features
#             # Decoder input : meteorological features
#             # tgt_y : inflow
#             return src, tgt, tgt_y.unsqueeze(-1)
#         elif self.no_q == False:
#             src = self.src[idx:idx+self.past_length + self.pred_length, :]
#             tgt_y = self.tgt[idx+self.past_length:idx+self.past_length+self.pred_length]
#             tgt = self.tgt[idx:idx+self.past_length]

#             # Encoder input : static features + meteorological features
#             # Decoder input : inflow
#             # tgt_y : inflow


#             # tgt[-self.pred_length:] = -inf
#             # tgt[-self.pred_length:] = 0
#             return src, tgt.unsqueeze(-1), tgt_y.unsqueeze(-1)
#         else:
#             raise ValueError("no_q should be True or False")



class SingleDatasetWithStatic(torch.utils.data.Dataset):
    def __init__(self, dynamic_path, static_path ,start_date, end_date, past_length, pred_length,
                 dynamic_features, target_features="inflow", model_type="transformer"):
        """
        Args:
            dynamic_path (str): path to dynamic features
            static_path (str): path to static features
            start_date (str): start date of the dataset
            end_date (str): end date of the dataset
            past_length (int): length of past sequence
            pred_length (int): length of prediction sequence
            dynamic_features (list): list of dynamic features
            target_features (list): list of target features
        """
        self.target_features = target_features
        self.dynamic_features = dynamic_features
        self.start_date = start_date
        self.end_date = end_date
        self.dates = pd.date_range(start_date, end_date)
        self.damcode = int(dynamic_path.split("/")[-1].split(".")[0])

        self.df = xr.open_dataset(dynamic_path).to_dataframe().loc[self.start_date:self.end_date]
        self.static_src = read_static(static_path, self.damcode).repeat(len(self.dates)).reshape(-1, 10)
        
        self.src = np.concatenate((self.df[self.dynamic_features].values, self.static_src),axis=1)
        self.tgt = self.df[self.target_features].values

        self.transformed_src = torch.tensor((self.src - self.src.mean(axis=0)) / self.src.std(axis=0), dtype=torch.float32)
        self.transformed_tgt = torch.tensor((self.tgt - self.tgt.mean(axis=0)) / self.tgt.std(axis=0), dtype=torch.float32)


        self.past_length = past_length
        self.pred_length = pred_length

        self.model_type = model_type

    
    def __len__(self):
        return len(self.df) - self.past_length - self.pred_length + 1

    def get_std_mean(self):
        return self.src.mean(axis=0), self.src.std(axis=0), self.tgt.mean(axis=0), self.tgt.std(axis=0)
    
    def __getitem__(self, idx):
        src = self.transformed_src[idx:idx+self.past_length + self.pred_length]
        tgt_y = self.transformed_tgt[idx:idx+self.past_length + self.pred_length]
        tgt = tgt_y.clone().detach()
        if self.model_type == "transformer":
            tgt[-self.pred_length:] = 0
            return src, tgt.unsqueeze(-1), tgt_y.unsqueeze(-1)
        elif self.model_type == "lstm":
            return src, tgt[:self.past_length].unsqueeze(-1), tgt_y[-self.pred_length:].unsqueeze(-1)