import yaml
import itertools
from typing import List, Tuple
from dynamic_path_generator import dynamic_path_generator
import os
from pathlib import Path
# read yaml file

"""
SAMPLE YAML FILE :  "../config/trainconfig/train1.yml"
"""

SAMPLE_YAML =  r"C:\Users\82105\MDP\config\sampleconfig\tr_train1.yml"
DW_LIST_PATH = r"C:\Users\82105\MDP\data\kr_dw_list.txt"
COMB_NUMBER_GRID = [i for i in range(1, 11)]
hyperparameters = {
    "learning_rate" : [0.0005],
    "past_length" : [28],
    "hidden_size" : [128, 256]
}
"""
SETTING YOUR GRIDS WITHIN THE YAML FILE
"""


"""
YOUR CHANGING HYPERPARAMETERS:
- learning_rate : [0.001, 0.0005, 0.0001]
- past_length : [21, 49, 77]

"""

def yml_generator(sample_yaml_path:str, dw_list_path:str, comb_number_grid:List[int], hyperparameters:dict, model_type:str="transformer",
                  save_path:str=r"C:\Users\82105\MDP\config\trainconfig\transformer") -> None:
    """
    yaml_path : str : path to the yaml file
    hyperparameters : dict : dictionary of hyperparameters
    """
    save_path = Path(save_path)
    with open(sample_yaml_path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        for comb_number in comb_number_grid:
            dynamic_paths, comb_r = dynamic_path_generator(dw_list_path, comb_number)
            for values in itertools.product(*hyperparameters.values()):
                (save_path / (str(comb_number))).mkdir(parents=True, exist_ok=True)
                conf["learning_rate"] = values[0]
                conf["past_length"] = values[1]
                conf["lstm_hidden_size"] = values[2]
                for i , (dp,code_t) in enumerate(zip(dynamic_paths, comb_r)):
                    conf["dynamic_paths"] = dp
                    conf["experiment_name"] = f"train_{str(i)}_{values[0]}_{values[1]}"
                    conf["run_dir"] = f"./runs/{model_type}/{comb_number}/train_{str(i)}_{values[0]}_{values[1]}"
                    with open(save_path/f"{str(comb_number)}/train_{str(i)}_{values[0]}_{values[1]}.yml", 'w') as f:
                        yaml.dump(conf, f)

if __name__ == "__main__":
    yml_generator(SAMPLE_YAML, DW_LIST_PATH, COMB_NUMBER_GRID, hyperparameters, model_type="transformer")
