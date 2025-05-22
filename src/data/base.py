from os import listdir
from os.path import join

from random import binomialvariate, choice

from numpy import ndarray, array, arange, unique
from numpy.random import dirichlet

from torch import Tensor
from torch.utils.data import Dataset 

from typing import Union

from src.data.attacks import Attack





class BaseDataset(Dataset):


    def __init__(
            self, 
            data: dict[str, dict[str, list]]
    ): 
        super(BaseDataset, self).__init__()
        self.X = data["X"]
        self.y = data["y"]
        self.cache = []
        self.num_samples = len(self.y)
        return

    
    def __len__(self): 
        return self.num_samples 
    

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return
    

    def load(self): 
        return 
    

    def clear(self):
        self.cache = []
        return





class Dirichlet: 


    def __init__(
            self,
            labels: Union[dict[str, int], list[int]],
            num_shards: int, 
            num_labels: int, 
            alpha: float
    ): 
        # Default
        self.num_shards = num_shards 
        self.num_labels = num_labels 
        self.alpha = alpha  
        
        # Distribution type 
        if isinstance(labels, dict): 
            self.split_type = "class"
            self._train_labels = labels["train"]
            self._test_labels = labels["test"]
        else: 
            self.split_type = "category"
            self._labels = labels

        return
    

    def sample_distribution(self) -> Union[tuple[dict[str, ndarray], dict[str, ndarray]], ndarray]:
        return self._sample_clients_distribution() if self.split_type == "class" else self._sample_categories_distribution()
    

    def _sample_clients_distribution(self) -> tuple[dict[str, ndarray], dict[str, ndarray]]: 
        _, train_counts = unique(self._train_labels, return_counts = True)
        _, test_counts = unique(self._test_labels, return_counts = True)

        distribution = dirichlet(alpha = [self.alpha] * self.num_shards, size = self.num_labels).T
        
        test_distribution = (test_counts[None, :] * distribution).astype(int)
        test_distribution = self._adjust_distribution(test_counts, test_distribution)

        distribution = test_distribution / test_distribution.sum(0, keepdims = True)
        train_distribution = (train_counts[None, :] * distribution).astype(int)
        train_distribution = self._adjust_distribution(train_counts, train_distribution)

        distributions = {"train": train_distribution, "test": test_distribution}

        return distributions
    

    def _sample_categories_distribution(self) -> ndarray: 
        _, counts = unique(self._labels, return_counts = True)

        distribution = dirichlet(alpha = [self.alpha] * self.num_shards, size = self.num_labels).T
        distribution = (counts[None, :] * distribution).astype(int)
        distribution = self._adjust_distribution(counts, distribution)

        return distribution 


    def _adjust_distribution(self, counts: ndarray, distribution: ndarray) -> ndarray: 
        counts_diff = counts - distribution.sum(0)

        for i, diff in enumerate(counts_diff): 

            if diff > 0:
                for _ in range(diff): 
                    index = arange(self.num_shards)
                    nonzeros_mask = distribution[:, i] > 0
                    min_idx = distribution[nonzeros_mask, i].argmin()
                    distribution[index[nonzeros_mask][min_idx], i] += 1

            elif diff < 0:
                for _ in range(diff): 
                    index = arange(self.num_shards)
                    nonzeros_mask = distribution[:, i] > 0
                    max_idx = distribution[nonzeros_mask, i].argmax()
                    distribution[index[nonzeros_mask][max_idx], i] -= 1       

        distribution = distribution.astype(int) 

        return distribution





class FederationSampler: 


    def __init__(
            self,
            train_data_path: str, 
            test_data_path: str,
            file_extension: str,
            num_labels: int
    ):
        self.num_labels = num_labels
        
        self.train_filepaths = [join(train_data_path, filename) for filename in listdir(train_data_path)]
        self.test_filepaths = [join(test_data_path, filename) for filename in listdir(test_data_path)]

        self.labels = {
            "train": [int(filepath.split("_y")[1].split(f".{file_extension}")[0]) for filepath in self.train_filepaths],
            "test": [int(filepath.split("_y")[1].split(f".{file_extension}")[0]) for filepath in self.test_filepaths]
        }
        
        return
    

    def sample_federation(
            self, 
            dataset_class: Dataset,
            num_shards: int, 
            alpha: float,
            attacks: list[Attack] = [],
            attacks_proba: float = 0.
    ) -> dict[str, list[str]]: 
        # Distribution
        dirichlet_sampler = Dirichlet(self.labels, num_shards, self.num_labels, alpha)
        distributions = dirichlet_sampler.sample_distribution()

        train_distribution = distributions["train"]
        test_distribution = distributions["test"]

        # Split data 
        train_filepaths = array(self.train_filepaths, dtype = "object")
        test_filepaths = array(self.test_filepaths, dtype = "object")

        train_labels = array(self.labels["train"], dtype = int)
        test_labels = array(self.labels["test"], dtype = int)
        
        train_split = {label: train_filepaths[train_labels == label].tolist() for label in range(self.num_labels)}
        test_split = {label: test_filepaths[test_labels == label].tolist() for label in range(self.num_labels)}

        # Create federation
        federation = []

        for i in range(num_shards): 

            train_data = {"X": [], "y": []}
            test_data = {"X": [], "y": []}

            shard_train_distribution = train_distribution[i]
            shard_test_distribution = test_distribution[i]

            for label in range(self.num_labels):
                train_count = shard_train_distribution[label]
                test_count = shard_test_distribution[label]

                train_data["X"].extend(train_split[label][:train_count])
                train_data["y"].extend([label] * train_count)

                test_data["X"].extend(test_split[label][:test_count])
                test_data["y"].extend([label] * test_count)

                train_split[label] = train_split[label][train_count:]
                test_split[label] = test_split[label][test_count:]

            if binomialvariate(n = 1, p = attacks_proba): 
                attack = choice(attacks)
                attack_suffix = "." + str(attack)   
            else: 
                attack = None
                attack_suffix = ""

            federation.append(
                {
                    "id": str(i) + attack_suffix,
                    "train": dataset_class(data = train_data, train = True, attack = attack),
                    "test": dataset_class(data = test_data, train = False, attack = attack),
                    "distribution": shard_train_distribution
                }
            )

        return federation

    
 