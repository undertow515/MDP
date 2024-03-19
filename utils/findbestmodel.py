import glob
import numpy as np
import re
from typing import List

def find_best_model(run_dir: str) -> str:
    model_paths = glob.glob(run_dir + "/models/*.pt")
    losses = np.array([float(i.split("_")[-1].split(".pt")[0]) for i in model_paths])
    idx = np.argmin(losses)
    # regular expression
    # find epoch_number in the file name
    epoch_number = re.findall(r'epoch_\d+', model_paths[idx])
    return model_paths[idx], epoch_number