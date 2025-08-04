from os.path import join

from numpy import ndarray, arange
from numpy.random import choice

from torch import load, cat, stack, no_grad, randperm
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.data.base import BaseDataset
from src.federated_learning.base import BaseClient, BaseServer

from typing import Union, Any

from tqdm import tqdm

from copy import deepcopy

from logging import getLogger, basicConfig, INFO, StreamHandler




basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler()]  
)

logger = getLogger("FEDAVG/Server")





class Client(BaseClient): 


    def __init__(
            self,
            train_dataset: BaseDataset,
            test_dataset: BaseDataset,
            distribution: ndarray,
            batch_size: int,
            device: str
    ): 
        super(Client, self).__init__(
            train_dataset, test_dataset,
            distribution, batch_size,
            device
        )
        return
    

    def train(
            self,
            num_epochs: int,
            model: Module, 
            criterion: Module,
            optimizer_class: Optimizer,
            optimizer_params: dict[str, Any]
    ) -> dict[str, Any]: 
        model, criterion, optimizer = self.training_setup(model, criterion, optimizer_class, optimizer_params)
        cumulative_loss = 0.

        for _ in range(num_epochs): 
            for X, y in self._train_dataloader: 
                y_hat = model(X.to(self.device))
                loss = criterion(y_hat, y.to(self.device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cumulative_loss += loss.item()

        self.clear_cache()

        training_dict = {
            "model": [p.data.cpu() for p in model.parameters()],
            "state_dict": deepcopy(model.state_dict()),
            "loss": cumulative_loss / self._num_samples,
            "num_samples": self._num_samples
        }

        return training_dict





class Server(BaseServer): 


    def __init__(
            self,
            clients: list[BaseClient],
            participation_rate: Union[int, float],
            model: Module,
            num_features: int,
            vae_path: str,
    ): 
        super(Server, self).__init__(
            clients = clients,
            participation_rate = participation_rate
        )
        self.model = model 
        self.detector = load(vae_path, weights_only = False)
        self.detector.eval()
        self.num_features = num_features
        self.malicious_votes = {i: [] for i in range(len(clients))}
        return 
    

    @no_grad()
    def _vae_detection(
            self, 
            updates: list[Any],
            indexes: ndarray
    ) -> list[Any]: 
        # Setup
        w0 = self.model.state_dict()["_mlp.5.weight"].view(-1)
        dw = [update["state_dict"]["_mlp.5.weight"].view(-1) - w0 for update in updates]
        n = w0.size(0)
        X = stack([delta[randperm(n)[:self.num_features]] for delta in dw])

        # Detection 
        X_hat = self.detector(X)[0]
        loss = (X - X_hat).pow(2).mean(1).numpy()
        thresh = loss.mean()
        malicious_indexes = indexes[loss > thresh]
        updates_index = arange(len(updates))[loss > thresh]

        # Update malicious votes 
        for i in indexes: 
            if i in malicious_indexes: 
                self.malicious_votes[i].append(1)
            else: 
                self.malicious_votes[i].append(0)

        return [updates[i] for i in updates_index]
    

    def aggregation(
            self, 
            updates: list[Any]
        ):
        # Setup 
        num_samples = sum([update["num_samples"] for update in updates])

        for p in self.model.parameters():
            p.data.zero_()

        for update in updates:
            w = update["num_samples"] / num_samples 
            for p_new, p in zip(update["model"], self.model.parameters()):
                p.data.add_(w * p_new)
        
        return
    

    def train(
            self, 
            num_rounds: int,
            num_local_epochs: int,
            criterion: Module, 
            optimizer_class: Optimizer, 
            optimizer_params: dict[str, Any],
            evaluation_step: int,
            metrics: dict[str, Module]
    ) -> dict[str, Any]: 
        # Setup 
        if isinstance(self.participation_rate, int): 
            num_clients_per_round = self.participation_rate
        else:
            num_clients_per_round = int(len(self.clients) * self.participation_rate)
            
        training_dict = dict()
        evaluation_dict = dict()

        # Training
        for r in tqdm(range(1, num_rounds + 1)): 
            # Round
            updates = []
            indexes = choice(self.clients_indexes, size = num_clients_per_round, replace = False)

            clients = self.clients[indexes]
            for client in clients: 
                update = client.train(num_local_epochs, self.model, criterion, optimizer_class, optimizer_params)
                updates.append(update)

            # Filter updates
            updates = self._vae_detection(updates, indexes)

            self.aggregation(updates)

            # Tracking
            num_samples = sum([update["num_samples"] for update in updates])
            training_loss = sum([update["loss"] * update["num_samples"] / num_samples for update in updates])
            training_dict[r] = {"loss": training_loss, "clients": indexes.tolist()}

            logger_msg = f"Round {r}: training_loss = {round(training_loss, 4)}"

            if r % evaluation_step == 0: 
                evaluation_dict[r] = self.evaluate(criterion, metrics)

                logger_msg += f", evaluation_loss = {round(evaluation_dict[r]['server']['loss'], 4)}"
                for metric_name, metric_val in evaluation_dict[r]["server"]['metrics'].items():
                    logger_msg += f", {metric_name} = {round(metric_val, 4)}"

                logger.info(logger_msg)

            else: 
                logger.info(logger_msg)

        # Results
        tracking_dict = {
            "training": training_dict,
            "evaluation": evaluation_dict
        }
            
        return tracking_dict
    

    def evaluate(
            self, 
            criterion: Module,
            metrics: dict[str, Module]
    ) -> dict[str, Any]: 
        # Setup 
        num_samples = 0.
        evaluation_dict = {
            "server": {
                "loss": 0.,
                "metrics": {
                    metric_name: 0 for metric_name in metrics.keys()
                }
            },
            "clients": dict()
        }

        # Evaluation
        for i, client in enumerate(self.clients): 
            client_evaluation_dict = client.evaluate(self.model, criterion, metrics)
            client_num_samples = client.num_samples()

            evaluation_dict["clients"][i] = client_evaluation_dict

            evaluation_dict["server"]["loss"] += client_num_samples * client_evaluation_dict["loss"]
            for metric_name, metric_value in client_evaluation_dict["metrics"].items():
                evaluation_dict["server"]["metrics"][metric_name] += client_num_samples * metric_value

            num_samples += client_num_samples

        evaluation_dict["server"]["loss"] /= num_samples 
        for metric_name, metric_val in evaluation_dict["server"]["metrics"].items():
            evaluation_dict["server"]["metrics"][metric_name] = metric_val / num_samples

        return evaluation_dict



