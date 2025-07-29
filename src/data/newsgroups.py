from os.path import join

from numpy import float32, loadtxt
from torch import Tensor, from_numpy 

from typing import Optional, Dict, List, Any

from src.data.base import BaseDataset, FederationSampler
from src.data.attacks import Attack





TRAIN_DATA_PATH = join("data", "newsgroups", "train")
TEST_DATA_PATH = join("data", "newsgroups", "test")





class Dataset(BaseDataset):


    def __init__(
            self, 
            data: Dict[str, List[Any]],
            train: bool,
            attack: Optional[Attack] = None # it is
    ): 
        super(Dataset, self).__init__(data)
        self.loading_transforms = attack
        return
    

    def __getitem__(self, index: int) -> list[Tensor, int]:
        return self.cache[index]
    

    def load(self): 
        self.cache = []  
        for txt_path, y in zip(self.X, self.y):
            x_np = loadtxt(txt_path, dtype=float32)
            x = from_numpy(x_np)
            x = self.loading_transforms(x)
            self.cache.append((x, y))
        return
    




def get_federation(num_shards: int, alpha: float, attacks: list[Attack] = [], attacks_proba: float = 0.) -> list[Dict[str, Dataset]]:
    sampler = FederationSampler(TRAIN_DATA_PATH, TEST_DATA_PATH, "png", 10)
    federation = sampler.sample_federation(Dataset, num_shards, alpha, attacks, attacks_proba)
    return federation



