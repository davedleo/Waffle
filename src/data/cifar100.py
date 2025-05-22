from os.path import join

from PIL import Image

from torch import Tensor
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine

from typing import Optional, Dict, List, Any

from src.data.base import BaseDataset, FederationSampler
from src.data.attacks import Attack





TRAIN_DATA_PATH = join("data", "cifar100", "train")
TEST_DATA_PATH = join("data", "cifar100", "test")





class Dataset(BaseDataset):


    def __init__(
            self, 
            data: Dict[str, List[Any]],
            train: bool,
            attack: Optional[Attack] = None
    ): 
        super(Dataset, self).__init__(data)
        loading_transforms = [ToTensor(), Normalize((.5, .5, .5), (.5, .5, .5))] if attack is None else [ToTensor(), Normalize((.5, .5, .5), (.5, .5, .5)), attack]
        self.loading_transforms = Compose(loading_transforms)
        self.training_transforms = RandomAffine(degrees = 90, translate = (.1, .1), scale = (.9, 1.1)) if train else None
        return
    

    def __getitem__(self, index: int) -> list[Tensor, int]:
        if self.training_transforms is not None: 
            X, y = self.cache[index]
            return self.training_transforms(X), y
        else: 
            return self.cache[index]
    

    def load(self): 
        for img_path, y, in zip(self.X, self.y):
            x = Image.open(img_path).convert("RGB")
            x = self.loading_transforms(x)
            self.cache.append((x, y))
        return 
    




def get_federation(num_shards: int, alpha: float, attacks: list[Attack] = [], attacks_proba: float = 0.) -> list[Dict[str, Dataset]]:
    sampler = FederationSampler(TRAIN_DATA_PATH, TEST_DATA_PATH, "png", 100)
    federation = sampler.sample_federation(Dataset, num_shards, alpha, attacks, attacks_proba)
    return federation



