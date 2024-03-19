from itertools import combinations
from typing import List, Tuple

def dynamic_path_generator(dw_list_path:str, comb_number:int) -> List[str]:
    """
    dw_list_path: str, path to the list of watershed codes
    comb_number: int, number of combinations
    """
    with open(dw_list_path, 'r') as f:
        dw_list = f.readlines()
    dw_list = list(map(str.strip, dw_list))
    comb_r = list(combinations(dw_list, comb_number))
    comb = [["./data/dynamic/" + code + ".nc"for code in comb_] for comb_ in comb_r]
    return comb, comb_r

if __name__ == "__main__":
    dw_list_path = r"C:\Users\82105\MDP\data\kr_dw_list.txt"
    comb_number = 3
    comb, comb_r = dynamic_path_generator(dw_list_path, comb_number)
    print(comb_r)
