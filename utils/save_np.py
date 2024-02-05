import numpy as np

def save(file_name:str,arr:np.ndarray,pickle:True):
    np.save(file=file_name,arr=arr,allow_pickle=pickle)