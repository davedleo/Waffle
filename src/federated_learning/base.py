from numpy import ndarray, array, arange
from numpy.random import binomial, uniform

from torch import no_grad, from_numpy, cat
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.data.base import BaseDataset 

from copy import deepcopy
from typing import Union, Any





class BaseClient: 


    def __init__(
            self,
            train_dataset: BaseDataset,
            test_dataset: BaseDataset,
            distribution: ndarray,
            batch_size: int,
            device: str
    ):
        self._train_dataset = train_dataset 
        self._test_dataset = test_dataset 
        self._train_dataloader = DataLoader(self._train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        self._test_dataloader = DataLoader(self._test_dataset, batch_size = batch_size, shuffle = False, drop_last = False)

        self._num_samples = len(self._train_dataset)

        label_counts = from_numpy(distribution) 
        self._distr = label_counts / label_counts.sum()

        self.device = device
        return 
    

    def training_setup(
            self,
            model: Module, 
            criterion: Module,
            optimizer_class: Optimizer,
            optimizer_params: dict[str, Any],
    ) -> tuple[Module, Module, Optimizer]:
        self._train_dataset.load()
        model = deepcopy(model).to(self.device)
        model.train()
        criterion = criterion.to(self.device)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        return model, criterion, optimizer
    

    def evaluation_setup(
            self,
            model: Module, 
            criterion: Module,
    ): 
        self._test_dataset.load()
        criterion = criterion.to(self.device)
        model = deepcopy(model).to(self.device)
        model = model.eval()
        return model, criterion
    

    def clear_cache(self): 
        self._train_dataset.clear()
        self._test_dataset.clear()
        return
    

    def train(
            self,
            num_epochs: int,
            model: Module, 
            criterion: Module,
            optimizer: Optimizer,
            optimizer_params: dict[str, Any]
    ) -> dict[str, Any]:
        return 
    

    @no_grad()
    def evaluate(
            self,
            model: Module, 
            criterion: Module,
            metrics: dict[str, Module]
    ) -> dict[str, Any]:
        model, criterion = self.evaluation_setup(model, criterion)

        y_true, y_pred = [], []
        for X, y in self._test_dataloader:
            y_true.append(y)
            y_pred.append(model(X.to(self.device)))

        y_true = cat(y_true, dim = 0)
        y_pred = cat(y_pred, dim = 0).cpu()
        self.clear_cache()

        evaluation_dict = {
            "loss": criterion(
                y_pred, 
                y_true
            ).item(),
            "metrics": {
                metric_name: metric(
                    y_pred, 
                    y_true, 
                    self._distr
                ).item() for metric_name, metric in metrics.items()
            }
        }

        return evaluation_dict
    

    def num_samples(self) -> int: 
        return self._num_samples





class BaseServer: 


    def __init__(
            self,
            clients: list[BaseClient],
            participation_rate: Union[int, float]
    ): 
        self.clients = array(clients)
        self.clients_indexes = arange(len(clients))
        self.participation_rate = participation_rate 
        return 
    

    def aggregation(
            self, 
            updates: list[Any]
        ):
        return
    

    def train(
            self, 
            num_rounds: int,
            num_local_epochs: int,
            criterion: Module, 
            optimizer_class: Optimizer, 
            optimizer_params: dict[str, Any],
            metrics: dict[str, Module]
    ) -> dict[str, Any]: 
        return
    

    def evaluate(
            self, 
            metrics: dict[str, Module]
    ) -> dict[str, Any]: 
        return

