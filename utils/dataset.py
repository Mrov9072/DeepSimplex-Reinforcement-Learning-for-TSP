import torch
from utils.utils import get_graph_mat

class Dataset:
    def __init__(self) -> None:
        self.dist_matrix = None

    def get_dim(self):
        pass

    def get_distance(self, s1, s2):
        return self.dist_matrix[s1, s2]
    
    def get_max_distance(self):
        return self.dist_matrix.max()

class GeneratedDataset(Dataset):
    def __init__(self, dim) -> None:
        super().__init__()
        
        self.dim = dim
        self.coord_matrix, self.dist_matrix = get_graph_mat(n=dim)

    def get_dim(self):
        return self.dim