import os
import sys
from pathlib import Path
import glob
from findbestmodel import find_best_model
import time

def clearepoch(run_dir:str):
    if not os.path.exists(run_dir):
        print("run_dir does not exist")
        return
    models_paths = glob.glob(str(Path(run_dir) / "models") + "/*.pt")
    best_model_path, _ = find_best_model(run_dir)
    if len(models_paths) == 1:
        print("only one model exists")
        return
    
    print("best model: ", best_model_path.split("\\")[-1])
    print("remove other models")
    for model_path in models_paths:
        if model_path.split("\\")[-1] != best_model_path.split("\\")[-1]:
            os.remove(model_path)
            print(model_path.split("\\")[-1] + " is removed")
    print("done")

if __name__ == "__main__":
    # run_dir = r"C:\Users\82105\MDP\runs\transformer\1\train_1_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\transformer\1\train_2_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\transformer\1\train_3_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\transformer\1\train_4_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\transformer\1\train_5_0.0005_28"
    # clearepoch(run_dir)
    while True:
        time.sleep(3600)
        for i in range(500):
            for j in range(1,11):
                run_dir = r"C:\Users\82105\MDP\runs\lstm\{}\train_{}_0.0005_28".format(j,i)
                clearepoch(run_dir)
                run_dir = r"C:\Users\82105\MDP\runs\transformer\{}\train_{}_0.0005_28".format(j,i)
                clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_0_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_1_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_10_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_11_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_12_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_13_0.0005_28"
    # clearepoch(run_dir)
    # run_dir = r"C:\Users\82105\MDP\runs\lstm\2\train_14_0.0005_28"
    # clearepoch(run_dir)