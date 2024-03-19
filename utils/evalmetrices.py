from sklearn.metrics import mean_squared_error, r2_score
from typing import List
import hydroeval as he
import numpy as np

def nse(y_true, y_pred):
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true - y_true.mean()) ** 2)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def pbias(y_true, y_pred):
    return 100 * sum(y_true - y_pred) / sum(y_true)

# def kge(y_true, y_pred):
#     # alpha = np.std(y_pred) / np.std(y_true)
#     # beta = y_pred.mean() / y_true.mean()
#     # r = np.corrcoef(y_true, y_pred)[0, 1]
#     # return 1 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
#     return 0

def kge(sim, obs):
    r = he.kge(sim, obs)
    return r

def get_eval_metrics(y_true:dict[List], y_pred:dict[List])->dict:
    damcodes = y_true.keys()
    eval_metrics = {}
    for damcode in damcodes:
        eval_metrics[damcode] = {
            "nse": nse(y_true[damcode].reshape(-1), y_pred[damcode].reshape(-1)),
            "rmse": rmse(y_true[damcode].reshape(-1), y_pred[damcode].reshape(-1)),
            "pbias": pbias(y_true[damcode].reshape(-1), y_pred[damcode].reshape(-1)),
            "kge": float(kge(y_true[damcode].reshape(-1), y_pred[damcode].reshape(-1))[0]),
        }
    return eval_metrics
    